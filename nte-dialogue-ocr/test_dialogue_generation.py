import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wuwa-dubber/src"))
from wuwa_dubber.config import AppConfig
from wuwa_dubber.models import VoiceProfile
from dialogue_generation import GenerationProject, BatchGenerator, file_digest, select_voice


def write_wav(path, frames=2205):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        out.writeframes(b"\x10\x00" * frames)


class Client:
    def __init__(self):
        self.calls = []
        self.identity = "model-v1"
        self.fail = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.cancels = 0

    def cache_identity(self):
        return {"model": self.identity}

    def synthesize(self, utterance, profile, output):
        self.calls.append((utterance.text, profile.speaker))
        self.entered.set()
        if not self.release.wait(4):
            raise TimeoutError("test release timeout")
        if self.fail or utterance.text == "失败测试":
            raise RuntimeError("模拟服务失败")
        write_wav(output)
        return .01

    def cancel(self):
        self.cancels += 1


class Runtime:
    def __init__(self, directory, client):
        self.config = AppConfig(tts_backend="windows-sapi", use_fake_tts=False)
        self.cache_dir = Path(directory) / "cache"
        self.client = client
        self.system_mode = True
        self.profiles = {"伊洛伊": VoiceProfile("伊洛伊"), "娜娜莉": VoiceProfile("娜娜莉")}
        self.prepared = []

    @property
    def player(self):
        raise AssertionError("batch must never touch the audio player")

    def prepare(self, stop, progress, *, playback=True):
        self.prepared.append(playback)

    def select_profile(self, record):
        name = record.get("speaker") if record.get("speaker_status") == "verified" else ""
        return self.profiles.get(name, VoiceProfile("默认声线")), name not in self.profiles


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = GenerationProject.create(self.root / "projects")
        self.client = Client()
        self.runtime = Runtime(self.root, self.client)

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, callback=lambda event: None):
        return BatchGenerator(self.project, {}, callback, runtime_factory=lambda options: self.runtime)

    def record(self, **changes):
        return dict({"dialogue_id": "session-1", "body": "主唱大人准备好了吗？", "speaker": "伊洛伊",
                     "revision": 1, "status": "verified", "speaker_status": "verified",
                     "body_confidence": .99, "review_body_confidence": .98}, **changes)

    def import_records(self, records, filename="dialogue.jsonl"):
        path = self.root / "session" / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
        return self.project.import_records(path)

    def test_import_materializes_latest_and_deduplicates_revisions_file(self):
        result = self.import_records([self.record(revision=3, body="最新版"), self.record(), self.record(revision=2)])
        self.assertEqual(result["added"], 1)
        result = self.import_records([self.record(revision=2)], "revisions.jsonl")
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(self.project.rows[0].text, "最新版")

    def test_import_is_read_only_and_malformed_lines_do_not_drop_valid_rows(self):
        path = self.root / "dialogue.jsonl"
        path.write_text(json.dumps(self.record()) + "\n{bad\n[]\nnull", encoding="utf-8")
        before = path.read_bytes()
        result = self.project.import_records(path)
        self.assertEqual(len(result["warnings"]), 3)
        self.assertEqual(result["added"], 1)
        self.project.rows[0].edit(text="人工修正")
        self.project.save()
        self.assertEqual(before, path.read_bytes())

    def test_new_source_revision_preserves_local_edits_but_requires_review(self):
        self.import_records([self.record()])
        row = self.project.rows[0]
        row.edit(text="我的修订")
        row.confirmed = True
        self.import_records([self.record(body="OCR 新修订", revision=2)])
        self.assertEqual(row.text, "我的修订")
        self.assertFalse(row.confirmed)
        self.assertTrue(row.review_reason())

    def test_uncertain_clipped_low_confidence_and_nan_are_held(self):
        changes = [{"status": "candidate"}, {"source_clipped": True}, {"coverage_warning": True},
                   {"review_coverage_warning": True}, {"body_confidence": .4},
                   {"review_body_confidence": .4}, {"body_confidence": float("nan")}]
        for i, change in enumerate(changes):
            self.import_records([self.record(dialogue_id=str(i), **change)])
        summary = self.runner().run()
        self.assertEqual(summary["held"], len(changes))
        self.assertEqual(self.client.calls, [])
        for row in self.project.rows:
            row.confirmed = True
        self.assertFalse(any(r.review_reason() for r in self.project.rows))

    def test_blank_text_remains_held_after_confirmation(self):
        self.import_records([self.record(body=" ")])
        self.project.rows[0].confirmed = True
        self.assertTrue(self.project.rows[0].review_reason())

    def test_uncertain_speaker_routes_to_default_until_manual_confirmation(self):
        self.import_records([self.record(speaker_status="candidate")])
        row = self.project.rows[0]
        self.assertTrue(select_voice(self.runtime, row)[1])
        row.confirmed = True
        self.assertEqual(select_voice(self.runtime, row)[0].speaker, "伊洛伊")

    def test_generation_does_not_play_and_resume_uses_no_synthesis(self):
        self.project.add_text("伊洛伊：你好，准备好了吗？")
        result = self.runner().run()
        self.assertEqual(result["completed"], 1)
        self.assertEqual(self.runtime.prepared, [False])
        row = self.project.rows[0]
        self.assertTrue(self.project.audio_path(row).is_file())
        self.project = GenerationProject.load(self.project.directory)
        result = self.runner().run()
        self.assertEqual(result["resumed"], 1)
        self.assertEqual(len(self.client.calls), 1)

    def test_confirming_uncertain_speaker_invalidates_default_voice_output(self):
        self.import_records([self.record(speaker_status="candidate")])
        self.runner().run()
        row = self.project.rows[0]
        self.assertTrue(row.audio["default_voice"])
        previous = self.project.audio_path(row)
        row.set_confirmed(True)
        self.assertIsNone(self.project.audio_path(row))
        self.assertTrue(previous.exists())
        self.runner().run()
        self.assertEqual(row.audio["voice"], "伊洛伊")

    def test_generate_prepare_does_not_require_output_device(self):
        from dialogue_dubbing import VoiceRuntime
        runtime = VoiceRuntime.__new__(VoiceRuntime)
        runtime.client = type("Health", (), {"health": lambda _: {}})()
        runtime.system_mode = True
        runtime.system_router = type("Router", (), {"check_available": lambda _, health: None})()
        with patch("sounddevice.check_output_settings", side_effect=RuntimeError("no audio output")):
            runtime.prepare(threading.Event(), lambda message: None, playback=False)

    def test_duplicate_dialogue_reuses_cache_but_preserves_both_rows(self):
        self.project.add_text("伊洛伊：你好\n伊洛伊：你好")
        result = self.runner().run()
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["cached"], 1)
        self.assertEqual(len(self.client.calls), 1)
        self.assertNotEqual(self.project.rows[0].audio["file"], self.project.rows[1].audio["file"])

    def test_changed_text_and_voice_invalidate_previous_output(self):
        self.project.add_text("伊洛伊：你好")
        self.runner().run()
        row = self.project.rows[0]
        old = self.project.audio_path(row)
        row.edit(text="再见")
        self.assertIsNone(self.project.audio_path(row))
        self.assertTrue(row.review_reason())
        self.assertTrue(old.exists())
        row.confirmed = True
        row.edit(voice="system:young_female")
        self.runner().run()
        self.assertEqual(row.audio["voice"], "系统·青年女声")
        self.assertEqual(len(self.client.calls), 2)

    def test_changed_mode_resets_overrides_and_keeps_old_file(self):
        self.project.add_text("你好")
        self.runner().run()
        row = self.project.rows[0]
        path = self.project.audio_path(row)
        self.project.set_mode("character")
        self.assertEqual(row.voice, "auto")
        self.assertIsNone(self.project.audio_path(row))
        self.assertTrue(path.exists())

    def test_changed_model_identity_requires_new_audio(self):
        self.project.add_text("你好")
        self.runner().run()
        old_key = self.project.rows[0].audio["cache_key"]
        self.client.identity = "model-v2"
        self.runner().run()
        self.assertEqual(len(self.client.calls), 2)
        self.assertNotEqual(old_key, self.project.rows[0].audio["cache_key"])

    def test_changed_reference_contents_require_new_audio(self):
        ref = self.root / "reference.wav"
        write_wav(ref, 500)
        self.runtime.profiles["伊洛伊"].reference_audio = str(ref)
        self.project.add_text("伊洛伊：你好")
        self.runner().run()
        write_wav(ref, 600)
        self.runner().run()
        self.assertEqual(len(self.client.calls), 2)

    def test_failed_row_does_not_block_following_rows_and_can_retry(self):
        self.project.add_text("失败测试\n成功测试")
        result = self.runner().run()
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["completed"], 1)
        row = self.project.rows[0]
        row.edit(text="再次尝试")
        row.confirmed = True
        self.runner().run([row.id])
        self.assertEqual(row.status, "done")

    def test_failed_forced_refresh_preserves_complete_export_and_cache(self):
        self.project.add_text("你好")
        self.runner().run()
        row = self.project.rows[0]
        path = self.project.audio_path(row)
        checksum = file_digest(path)
        self.client.fail = True
        result = self.runner().run(force=True)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(file_digest(self.project.audio_path(row)), checksum)
        self.assertFalse(list(self.project.directory.glob("audio/.*.tmp.wav")))

    def test_late_result_after_cancel_is_never_published(self):
        self.project.add_text("伊洛伊：需要取消的台词\n下一条")
        self.client.release.clear()
        runner = self.runner()
        thread = threading.Thread(target=runner.run)
        thread.start()
        self.assertTrue(self.client.entered.wait(2))
        runner.cancel()
        self.client.release.set()
        thread.join(4)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.project.rows[0].status, "cancelled")
        self.assertEqual(self.project.rows[1].status, "pending")
        self.assertFalse(list((self.project.directory / "audio").glob("*")))
        self.assertFalse(list(self.runtime.cache_dir.glob("*.wav")))

    def test_cancel_relay_covers_request_registration_delay(self):
        self.project.add_text("测试")
        self.client.release.clear()
        runner = self.runner()
        thread = threading.Thread(target=runner.run)
        thread.start()
        self.assertTrue(self.client.entered.wait(2))
        runner.cancel()
        import time
        deadline = time.monotonic() + 2
        while self.client.cancels < 2 and time.monotonic() < deadline:
            time.sleep(.03)
        self.client.release.set()
        thread.join(4)
        self.assertGreaterEqual(self.client.cancels, 2)

    def test_cancel_before_start_makes_no_calls(self):
        self.project.add_text("测试")
        runner = self.runner()
        runner.cancel()
        runner.run()
        self.assertEqual(self.client.calls, [])

    def test_long_dialogue_is_lossless_and_exports_one_complete_wav(self):
        body = "".join(f"第{i}次出发，前往这座城市的另一边。" for i in range(32))
        self.project.add_text("伊洛伊：" + body)
        self.runner().run()
        self.assertEqual("".join(call[0] for call in self.client.calls), body)
        self.assertTrue(all(len(call[0]) <= 300 for call in self.client.calls))
        with wave.open(str(self.project.audio_path(self.project.rows[0])), "rb") as wav:
            self.assertEqual(wav.getnframes(), len(self.client.calls)*2205)
        calls = len(self.client.calls)
        self.project.rows[0].audio = {}
        result = self.runner().run()
        self.assertEqual(result["cached"], 1)
        self.assertEqual(len(self.client.calls), calls)

    def test_missing_or_corrupt_export_is_restored_from_cache(self):
        self.project.add_text("测试")
        self.runner().run()
        self.project.audio_path(self.project.rows[0]).write_bytes(b"truncated")
        self.assertIsNone(self.project.audio_path(self.project.rows[0]))
        result = self.runner().run()
        self.assertEqual(result["cached"], 1)
        self.assertEqual(len(self.client.calls), 1)

    def test_project_recovery_changes_inflight_to_cancelled(self):
        self.project.add_text("测试")
        self.project.rows[0].status = "generating"
        self.project.save()
        loaded = GenerationProject.load(self.project.directory)
        self.assertEqual(loaded.rows[0].status, "cancelled")

    def test_export_paths_cannot_escape_project(self):
        self.project.add_text("测试")
        row = self.project.rows[0]
        outside = self.root / "outside.wav"
        write_wav(outside)
        row.audio = dict(input_key=row.input_key("system"), file=str(outside), sha256=file_digest(outside))
        self.assertIsNone(self.project.audio_path(row))

    def test_unknown_or_wrong_mode_voice_is_not_silently_substituted(self):
        self.project.add_text("测试")
        self.project.rows[0].voice = "character:伊洛伊"
        result = self.runner().run()
        self.assertEqual(result["failed"], 1)
        self.assertEqual(self.client.calls, [])

    def test_csv_manifest_escapes_spreadsheet_formulas(self):
        self.project.add_text("=2+3")
        self.project.save()
        self.assertIn("'=2+3", (self.project.directory / "manifest.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
