"""Offscreen preview checks; uses tiny local WAVs and never starts a model."""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from wuwa_dubber.config import AppConfig, AppPaths
from wuwa_dubber.models import PipelineSnapshot, Utterance, VoiceProfile
from wuwa_dubber.ui.main_window import MainWindow
from wuwa_dubber.ui.speech_preview import SpeechPreview
from wuwa_dubber.voice_profiles import VoiceProfileStore


def write_wav(path: Path, sample: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * 160)


class PreviewDouble(QObject):
    result = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.cancelled = False

    def start(self) -> None:
        self.running = True

    def cancel(self) -> None:
        self.cancelled = True

    def isRunning(self) -> bool:
        return self.running

    def finish(self) -> None:
        self.running = False
        self.finished.emit()


class RecordingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.cancelled = False
        self.fail = False
        self.started = threading.Event()
        self.release: threading.Event | None = None

    def health(self) -> dict:
        return {"status": "ok", "backend": "preview-test"}

    def cache_identity(self) -> dict:
        return {"backend": "preview-test", "model_revision": "test-v1"}

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        self.texts.append(utterance.text)
        self.started.set()
        if self.release is not None and not self.release.wait(3):
            raise RuntimeError("Test synthesis was not released")
        if self.fail:
            raise RuntimeError("test synthesis failure")
        write_wav(output, sample=100 + len(self.texts))
        return 0.01

    def cancel(self) -> None:
        self.cancelled = True
        if self.release is not None:
            self.release.set()


class SilentPlayer:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.stopped = False

    def play(self, path: Path, *, cancel_event: threading.Event | None = None) -> None:
        if cancel_event is None or not cancel_event.is_set():
            self.paths.append(path)

    def stop(self) -> None:
        self.stopped = True


class PendingThread:
    alive = True

    def is_alive(self) -> bool:
        return self.alive


class StoppingPipeline:
    def __init__(self) -> None:
        self._threads = [PendingThread()]
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def snapshot(self) -> PipelineSnapshot:
        return PipelineSnapshot(running=False, message="已停止")


class PreviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="preview-ui-")
        self.paths = AppPaths(Path(self.directory.name))
        self.store = VoiceProfileStore(self.paths)
        self.config = AppConfig.from_dict({"tts_backend": "qwen3-tts", "narrator_profile": "通用旁白"})
        for speaker, folder in (("通用旁白", "narrator"), ("娜娜莉", "nanally")):
            write_wav(self.paths.profiles / folder / "reference.wav")
            self.store.save(
                VoiceProfile(speaker, reference_audio=f"{folder}/reference.wav", prompt_text="参考录音文字"),
                directory_name=folder,
            )
        with patch("wuwa_dubber.ui.main_window.collect_diagnostics", return_value=[]), patch(
            "wuwa_dubber.ui.main_window.AudioPlayer.list_devices", return_value=[]
        ):
            self.window = MainWindow(self.paths, self.config, self.store)
        for row in range(self.window.profile_table.rowCount()):
            if self.window.profile_table.item(row, 0).text() == "娜娜莉":
                self.window.profile_table.selectRow(row)
        self.window.preview_text.setPlainText("你好，欢迎来到异环。")

    def tearDown(self) -> None:
        if self.window.preview is not None:
            self.window.preview.finish()
        if isinstance(self.window.pipeline, StoppingPipeline):
            self.window.pipeline._threads[0].alive = False
            self.window._finish_pipeline_stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def test_normal_and_force_buttons_preserve_selected_voice_and_restore_state(self) -> None:
        for button_name, force in (("preview_button", False), ("preview_regenerate_button", True)):
            with self.subTest(force=force):
                worker = PreviewDouble()
                with patch("wuwa_dubber.ui.main_window.SpeechPreview", return_value=worker) as factory:
                    getattr(self.window, button_name).click()
                    self.assertEqual(factory.call_args.kwargs["force"], force)
                    self.assertEqual(factory.call_args.args[2].speaker, "娜娜莉")
                    self.assertEqual(factory.call_args.args[3].parent, self.paths.cache)
                    self.assertFalse(self.window.preview_button.isEnabled())
                    self.assertFalse(self.window.preview_regenerate_button.isEnabled())
                    self.assertFalse(self.window.start_button.isEnabled())
                    self.assertTrue(self.window.preview_stop_button.isEnabled())
                    self.window._preview_speech(force=True)
                    factory.assert_called_once()
                    worker.result.emit("试听完成，共 2 段，缓存命中 2/2 段。")
                    worker.finish()
                self.assertTrue(self.window.preview_button.isEnabled())
                self.assertTrue(self.window.preview_regenerate_button.isEnabled())
                self.assertTrue(self.window.start_button.isEnabled())
                self.assertFalse(self.window.preview_stop_button.isEnabled())
                self.assertIn("缓存命中 2/2", self.window.preview_status.text())

    def test_cancel_waits_for_worker_and_ignores_late_progress(self) -> None:
        worker = PreviewDouble()
        with patch("wuwa_dubber.ui.main_window.SpeechPreview", return_value=worker):
            self.window.preview_regenerate_button.click()
            self.window.preview_stop_button.click()
            self.assertTrue(worker.cancelled)
            self.assertFalse(self.window.preview_stop_button.isEnabled())
            self.assertFalse(self.window.preview_button.isEnabled())
            self.assertFalse(self.window.preview_regenerate_button.isEnabled())
            self.assertFalse(self.window.start_button.isEnabled())
            worker.result.emit("已排队的旧进度")
            self.assertEqual(self.window.preview_status.text(), "正在停止试听…")
            worker.finish()
        self.assertEqual(self.window.preview_status.text(), "试听已停止。")
        self.assertTrue(self.window.preview_button.isEnabled())
        self.assertTrue(self.window.preview_regenerate_button.isEnabled())
        self.assertTrue(self.window.start_button.isEnabled())

    def test_failure_restores_buttons_without_erasing_error(self) -> None:
        worker = PreviewDouble()
        with patch("wuwa_dubber.ui.main_window.SpeechPreview", return_value=worker):
            self.window.preview_button.click()
            worker.result.emit("试听失败：模型暂不可用")
            worker.finish()
        self.assertTrue(self.window.preview_button.isEnabled())
        self.assertTrue(self.window.preview_regenerate_button.isEnabled())
        self.assertIn("模型暂不可用", self.window.preview_status.text())

    def test_stop_keeps_pending_pipeline_and_blocks_new_model_requests(self) -> None:
        pipeline = StoppingPipeline()
        self.window.pipeline = pipeline
        self.window._stop()
        self.assertIs(self.window.pipeline, pipeline)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.preview_button.isEnabled())
        self.assertFalse(self.window.preview_regenerate_button.isEnabled())
        self.assertTrue(self.window._pipeline_stop_timer.isActive())
        with patch("wuwa_dubber.ui.main_window.DubPipeline") as realtime, patch(
            "wuwa_dubber.ui.main_window.SpeechPreview"
        ) as preview:
            self.window._start()
            self.window._preview_speech()
            realtime.assert_not_called()
            preview.assert_not_called()
        self.window._stop()
        self.assertEqual(pipeline.close_count, 1)
        pipeline._threads[0].alive = False
        self.window._finish_pipeline_stop()
        self.assertIsNone(self.window.pipeline)
        self.assertFalse(self.window._pipeline_stop_timer.isActive())
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertTrue(self.window.preview_button.isEnabled())
        self.assertTrue(self.window.preview_regenerate_button.isEnabled())

    def test_stopped_snapshot_retains_pipeline_until_threads_exit(self) -> None:
        pipeline = StoppingPipeline()
        self.window.pipeline = pipeline
        self.window._update_snapshot(pipeline.snapshot())
        self.assertIs(self.window.pipeline, pipeline)
        self.assertFalse(self.window.preview_button.isEnabled())
        self.assertEqual(self.window.status_label.text(), "正在停止")
        pipeline._threads[0].alive = False
        self.window._finish_pipeline_stop()
        self.assertIsNone(self.window.pipeline)
        self.assertTrue(self.window.preview_button.isEnabled())

    def test_close_waits_for_pending_pipeline(self) -> None:
        pipeline = StoppingPipeline()
        self.window.pipeline = pipeline
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertIs(self.window.pipeline, pipeline)
        pipeline._threads[0].alive = False
        self.window._finish_pipeline_stop()
        self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())


class PreviewSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="preview-session-")
        self.output = Path(self.directory.name) / "cache" / "preview.wav"
        self.config = AppConfig(use_fake_tts=True)
        self.profile = VoiceProfile("测试旁白")
        self.client = RecordingClient()

    def tearDown(self) -> None:
        self.app.processEvents()
        self.directory.cleanup()

    def make_worker(self, text: str, *, force: bool = False) -> tuple[SpeechPreview, SilentPlayer, list[str]]:
        player = SilentPlayer()
        with patch("wuwa_dubber.ui.speech_preview.AudioPlayer", return_value=player):
            worker = SpeechPreview(self.config, text, self.profile, self.output, force=force)
        messages: list[str] = []
        worker.result.connect(messages.append)
        return worker, player, messages

    def run_worker(self, worker: SpeechPreview) -> None:
        with patch("wuwa_dubber.ui.speech_preview.create_tts_client", return_value=self.client):
            worker.start()
            self.assertTrue(worker.wait(5000), "Preview worker did not finish")
        self.app.processEvents()

    def test_repeat_preview_uses_shared_disk_cache_and_force_regenerates(self) -> None:
        text = "你好，欢迎来到异环。"
        first, first_player, first_messages = self.make_worker(text)
        self.run_worker(first)
        self.assertEqual(self.client.texts, [text])
        self.assertEqual(len(first_player.paths), 1)
        cached_path = first_player.paths[0]
        old_audio = cached_path.read_bytes()
        self.assertNotEqual(cached_path.name, "preview.wav")
        self.assertIn("缓存命中 0/1", first_messages[-1])
        second, second_player, messages = self.make_worker(text)
        self.run_worker(second)
        self.assertEqual(self.client.texts, [text])
        self.assertEqual(second_player.paths, [cached_path])
        self.assertIn("缓存命中 1/1", messages[-1])
        self.assertIn("首段就绪", messages[-1])
        self.assertIn(str(self.output.parent.resolve()), messages[-1])
        forced, forced_player, forced_messages = self.make_worker(text, force=True)
        self.run_worker(forced)
        self.assertEqual(self.client.texts, [text, text])
        self.assertEqual(len(forced_player.paths), 1)
        self.assertNotEqual(cached_path.read_bytes(), old_audio)
        self.assertIn("缓存命中 0/1", forced_messages[-1])

    def test_failed_regeneration_retains_previous_audio(self) -> None:
        text = "这句话已有保存的音频。"
        first, player, _ = self.make_worker(text)
        self.run_worker(first)
        old_audio = player.paths[0].read_bytes()
        self.client.fail = True
        forced, forced_player, messages = self.make_worker(text, force=True)
        self.run_worker(forced)
        self.assertEqual(player.paths[0].read_bytes(), old_audio)
        self.assertEqual(forced_player.paths, [])
        self.assertIn("试听失败", messages[-1])
        self.client.fail = False
        repeated, _, repeated_messages = self.make_worker(text)
        self.run_worker(repeated)
        self.assertEqual(len(self.client.texts), 2)
        self.assertIn("缓存命中 1/1", repeated_messages[-1])

    def test_multisentence_preview_reuses_and_refreshes_every_segment(self) -> None:
        text = "今天我们一起去河边散步。等到傍晚再回家吃晚饭。明天继续探索城市里的新故事。"
        first, player, messages = self.make_worker(text)
        self.run_worker(first)
        parts = list(self.client.texts)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), text)
        self.assertEqual(len(player.paths), len(parts))
        self.assertIn(f"共 {len(parts)} 段", messages[-1])
        repeated, repeated_player, repeated_messages = self.make_worker(text)
        self.run_worker(repeated)
        self.assertEqual(self.client.texts, parts)
        self.assertEqual(repeated_player.paths, player.paths)
        self.assertIn(f"缓存命中 {len(parts)}/{len(parts)}", repeated_messages[-1])
        forced, _, forced_messages = self.make_worker(text, force=True)
        self.run_worker(forced)
        self.assertEqual(self.client.texts, parts + parts)
        self.assertIn(f"缓存命中 0/{len(parts)}", forced_messages[-1])

    def test_cancel_before_start_never_contacts_service(self) -> None:
        worker, player, messages = self.make_worker("取消这次试听。")
        worker.cancel()
        with patch("wuwa_dubber.ui.speech_preview.create_tts_client") as factory:
            worker.start()
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()
            factory.assert_not_called()
        self.assertTrue(player.stopped)
        self.assertEqual(messages, [])

    def test_cancel_during_generation_does_not_play_late_audio(self) -> None:
        self.client.release = threading.Event()
        worker, player, messages = self.make_worker("取消还没有生成完的这一句。")
        with patch("wuwa_dubber.ui.speech_preview.create_tts_client", return_value=self.client):
            worker.start()
            try:
                self.assertTrue(self.client.started.wait(2))
                worker.cancel()
                self.assertTrue(worker.wait(5000))
            finally:
                self.client.release.set()
                worker.wait(5000)
        self.app.processEvents()
        self.assertTrue(self.client.cancelled)
        self.assertTrue(player.stopped)
        self.assertEqual(player.paths, [])
        self.assertFalse(any("试听完成" in message or "试听失败" in message for message in messages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
