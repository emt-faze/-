import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wuwa-dubber/src"))
from generator_voices import VoiceImportTask, ImportCancelled, load_profiles, check_names


def wav(path, seconds=5, sample=2000):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 32000, 0, "NONE", "not compressed"))
        audio.writeframes(int(sample).to_bytes(2, "little", signed=True) * int(32000*seconds))


class VoiceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text("{}", encoding="utf-8")
        self.source = self.root / "recording.mp4"
        self.source.write_bytes(b"source recording untouched")
        self.options = dict(speaker="新角色", aliases=["小名"], source=str(self.source),
                            prompt="这是参考片段里实际说出的台词。", start=1.25, duration=5, speed=1)
        self.paths_patch = patch("generator_voices.voice_paths", return_value=(self.root, self.root))
        self.paths_patch.start()

    def tearDown(self):
        self.paths_patch.stop()
        self.temp.cleanup()

    def task(self, **updates):
        task = VoiceImportTask({}, dict(self.options, **updates))
        task.convert = lambda source, destination, start, duration: wav(destination, duration)
        return task

    def test_import_publishes_complete_profile_and_relative_reference(self):
        profile = self.task().run()
        saved = load_profiles(self.root)
        self.assertEqual(len(saved), 1)
        self.assertEqual(profile["aliases"], ["小名"])
        reference = self.root / "profiles" / profile["reference_audio"]
        self.assertFalse(Path(profile["reference_audio"]).is_absolute())
        self.assertTrue(reference.is_file())
        self.assertEqual(VoiceImportTask.check_audio(reference), 5)
        self.assertEqual(self.source.read_bytes(), b"source recording untouched")
        info = json.loads((reference.parent / "import.json").read_text(encoding="utf-8"))
        self.assertEqual(info["start_seconds"], 1.25)

    def test_existing_speaker_or_alias_is_never_overwritten(self):
        first = self.task().run()
        reference = self.root / "profiles" / first["reference_audio"]
        before = reference.read_bytes()
        for speaker, aliases in [("新角色", []), ("另一个", ["小名"]), ("小 名", [])]:
            with self.assertRaisesRegex(ValueError, "已属于"):
                self.task(speaker=speaker, aliases=aliases).run()
        self.assertEqual(before, reference.read_bytes())
        self.assertEqual(len(load_profiles(self.root)), 1)

    def test_names_cannot_collide_during_conversion(self):
        task = self.task()
        def converter(source, target, start, duration):
            wav(target)
            self.task(aliases=[]).run()
        task.convert = converter
        with self.assertRaisesRegex(ValueError, "已属于"):
            task.run()
        self.assertEqual(len(load_profiles(self.root)), 1)
        self.assertFalse(list((self.root / ".voice-imports").iterdir()))

    def test_duplicate_or_empty_names_are_rejected(self):
        for speaker, aliases in [("新角色", ["新 角色"]), ("。", []), ("新角色", ["---"])]:
            with self.assertRaises(ValueError):
                check_names(self.root, speaker, aliases)

    def test_blank_transcript_and_invalid_speed_fail_before_conversion(self):
        for data in [{"prompt": ""}, {"speed": .1}, {"speed": float("nan")}]:
            task = self.task(**data)
            with patch.object(task, "convert") as converter, self.assertRaises(ValueError):
                task.run()
            converter.assert_not_called()

    def test_invalid_clip_range_rejected(self):
        for data in [{"start": -1}, {"duration": 2}, {"duration": 31}, {"start": float("nan")}]:
            with self.assertRaises(ValueError):
                self.task(**data).run()

    def test_conversion_failure_leaves_no_partial_voice(self):
        task = self.task()
        def broken(source, destination, start, duration):
            destination.write_bytes(b"partial")
            raise RuntimeError("decoder failed")
        task.convert = broken
        with self.assertRaises(RuntimeError):
            task.run()
        self.assertEqual(load_profiles(self.root), [])
        self.assertFalse(list((self.root / ".voice-imports").iterdir()))

    def test_too_short_silent_and_invalid_audio_rejected(self):
        for kind in ("short", "silent", "invalid"):
            task = self.task()
            def convert(source, destination, start, duration):
                if kind == "invalid":
                    destination.write_bytes(b"broken wave")
                else:
                    wav(destination, 1 if kind == "short" else 5, 0 if kind == "silent" else 2000)
            task.convert = convert
            with self.assertRaises(ValueError):
                task.run()
        self.assertEqual(load_profiles(self.root), [])

    def test_cancel_before_or_after_conversion_never_publishes(self):
        task = self.task()
        task.cancel()
        with self.assertRaises(ImportCancelled):
            task.run()
        task = self.task()
        def convert(source, destination, start, duration):
            wav(destination)
            task.cancel()
        task.convert = convert
        with self.assertRaises(ImportCancelled):
            task.run()
        self.assertEqual(load_profiles(self.root), [])

    def test_cancel_at_publish_boundary_preserves_other_profiles(self):
        self.task(speaker="原角色", aliases=[]).run()
        task = self.task()
        original = check_names
        calls = []
        def check(*args):
            calls.append(1)
            original(*args)
            if len(calls) == 2:
                task.cancel()
        with patch("generator_voices.check_names", side_effect=check), self.assertRaises(ImportCancelled):
            task.run()
        self.assertEqual([p.speaker for p in load_profiles(self.root)], ["原角色"])

    def test_preview_uses_selected_clip_without_registering_voice(self):
        task = self.task(speaker="", aliases=[], prompt="")
        with patch("wuwa_dubber.audio.player.AudioPlayer") as player:
            self.assertIsNone(task.run(preview=True))
            player.return_value.play.assert_called_once()
        self.assertEqual(load_profiles(self.root), [])
        self.assertFalse(list((self.root / ".voice-imports").iterdir()))

    def test_cancel_preview_cleans_temporary_audio(self):
        task = self.task()
        with patch("wuwa_dubber.audio.player.AudioPlayer") as player:
            player.return_value.play.side_effect = lambda *args, **kwargs: task.cancel()
            with self.assertRaises(ImportCancelled):
                task.run(preview=True)
        self.assertFalse(list((self.root / ".voice-imports").iterdir()))

    def test_role_name_is_never_used_as_a_filesystem_path(self):
        profile = self.task(speaker="../角色\\特殊:名字", aliases=[]).run()
        reference = (self.root / "profiles" / profile["reference_audio"]).resolve()
        self.assertTrue(reference.is_relative_to(self.root / "profiles"))
        self.assertTrue(reference.parent.name.startswith("added-"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
