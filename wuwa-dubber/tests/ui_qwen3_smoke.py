"""Run with the existing UI environment; does not import the model environment."""
from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from wuwa_dubber.app import configure_chinese_font
from wuwa_dubber.config import AppConfig, AppPaths, load_config
from wuwa_dubber.models import VoiceProfile
from wuwa_dubber.ui.main_window import MainWindow
from wuwa_dubber.voice_profiles import VoiceProfileStore


def main() -> None:
    app = QApplication.instance() or QApplication([])
    configure_chinese_font(app)
    with tempfile.TemporaryDirectory(prefix="qwen3-ui-") as directory:
        paths = AppPaths(Path(directory))
        store = VoiceProfileStore(paths)
        config = AppConfig.from_dict({"tts_backend": "qwen3-tts"})
        with patch("wuwa_dubber.ui.main_window.collect_diagnostics", return_value=[]), patch("wuwa_dubber.ui.main_window.AudioPlayer.list_devices", return_value=[]):
            window = MainWindow(paths, config, store)
        try:
            window.resize(1100, 900)
            window.show()
            app.processEvents()
            assert window.backend_combo.currentData() == "qwen3-tts"
            assert window.tts_endpoint.text() == "http://127.0.0.1:9882"
            assert window.tts_endpoint.isEnabled()
            assert window.first_segment_limit.value() == 10
            assert window.segment_limit.value() == 36 and window.prefetch_window.value() == 2
            assert window.stability_frames.value() == 2
            window.first_segment_limit.setValue(0)
            window.segment_limit.setValue(48)
            window.prefetch_window.setValue(3)
            window.stability_frames.setValue(3)
            window._save_controls()
            restored = load_config(paths)
            assert restored.first_segment_max_chars == 0 and restored.segment_max_chars == 48
            assert restored.prefetch_window == 3 and restored.stability_frames == 3
            window.first_segment_limit.setValue(10)
            window.segment_limit.setValue(36)
            window.prefetch_window.setValue(2)
            window.stability_frames.setValue(2)
            window._save_controls()
            window.backend_combo.setCurrentIndex(window.backend_combo.findData("gpt-sovits"))
            assert window.tts_endpoint.text() == "http://127.0.0.1:9881"
            window.backend_combo.setCurrentIndex(window.backend_combo.findData("qwen3-tts"))
            assert window.tts_endpoint.text() == "http://127.0.0.1:9882"
            window.tts_endpoint.setText("http://localhost:12000/custom")
            window.backend_combo.setCurrentIndex(window.backend_combo.findData("windows-sapi"))
            assert not window.tts_endpoint.isEnabled()
            window.backend_combo.setCurrentIndex(window.backend_combo.findData("qwen3-tts"))
            assert window.tts_endpoint.text() == "http://localhost:12000/custom"
            window.fake_tts.setChecked(True)
            assert not window.tts_endpoint.isEnabled()
            window.fake_tts.setChecked(False)
            assert window.tts_endpoint.isEnabled()
            with patch("wuwa_dubber.ui.main_window.SpeechPreview") as preview_class:
                window._preview_speech()
                preview_class.assert_not_called()
                assert "参考声线" in window.preview_status.text()
                reference_dir = paths.profiles / "narrator"
                reference_dir.mkdir()
                audio = reference_dir / "reference.wav"
                with wave.open(str(audio), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(24000)
                    output.writeframes(b"\x00\x00" * 240)
                store.save(VoiceProfile("旁白", reference_audio="narrator/reference.wav", prompt_text="实际参考文字"), directory_name="narrator")
                window._refresh_profiles()
                preview_class.return_value = MagicMock()
                window._preview_speech()
                preview_class.assert_called_once()
                selected_profile = preview_class.call_args.args[2]
                assert Path(selected_profile.reference_audio) == audio
                assert selected_profile.prompt_text == "实际参考文字"
                assert preview_class.call_args.args[0].tts_backend == "qwen3-tts"
                window._preview_finished()
            window.tts_endpoint.setText("http://127.0.0.1:9882")
            window.centralWidget().setCurrentIndex(2)
            app.processEvents()
            target = Path(os.environ.get("WUWA_SMOKE_OUTPUT", str(Path(__file__).resolve().parents[2] / "outputs" / "qwen3-ui.png")))
            target.parent.mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(target))
            print("Qwen3 UI controls and preview validation passed: " + str(target.resolve()))
        finally:
            window.close()


if __name__ == "__main__":
    main()
