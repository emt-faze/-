from __future__ import annotations

from pathlib import Path
import threading
from PySide6.QtCore import QThread, Signal
from ..audio import AudioPlayer
from ..config import AppConfig
from ..models import Utterance, VoiceProfile
from ..speech_cache import SpeechCancelled
from ..speech_session import SpeechSession
from ..tts import create_tts_client


class SpeechPreview(QThread):
    result = Signal(str)

    def __init__(self, config: AppConfig, text: str, profile: VoiceProfile, output: Path, parent=None, *, force: bool = False):
        super().__init__(parent)
        self.config = AppConfig.from_dict(config.to_dict())
        self.text, self.profile, self.output = text, profile, output
        self.force = force
        self.client = None
        self.session: SpeechSession | None = None
        self.cancel_event = threading.Event()
        self.player = AudioPlayer(config.output_device, volume=config.output_volume)

    def run(self) -> None:
        try:
            if self.cancel_event.is_set() or self.isInterruptionRequested():
                return
            self.client = create_tts_client(self.config)
            health = self.client.health()
            if health.get("warming"):
                self._emit_progress("模型预热中…正在优先处理你的试听请求，可随时停止。")
            if self.cancel_event.is_set() or self.isInterruptionRequested():
                return
            self.session = SpeechSession(self.config, self.output.parent, self.client, self.player)
            report = self.session.run(
                Utterance(self.text, self.profile.speaker, source="preview"), self.profile,
                cancel_event=self.cancel_event, force=self.force, on_progress=self._emit_progress,
            )
            if self.cancel_event.is_set() or self.isInterruptionRequested():
                return
            self.result.emit(
                f"试听完成，共 {report.total_segments} 段，缓存命中 {report.cache_hits}/{report.total_segments} 段，"
                f"首段就绪 {report.first_audio_seconds:.2f} 秒。\n"
                f"音频保存在：{self.output.parent.resolve()}"
            )
        except SpeechCancelled:
            pass
        except Exception as exc:
            # Session cleanup also sets cancel_event when synthesis fails.
            # Only a user interruption should hide the underlying failure.
            if not self.isInterruptionRequested():
                self.result.emit(f"试听失败：{exc}")

    def _emit_progress(self, message: str) -> None:
        if not self.cancel_event.is_set() and not self.isInterruptionRequested():
            self.result.emit(message)

    def cancel(self) -> None:
        self.cancel_event.set()
        self.requestInterruption()
        if self.session is not None:
            self.session.cancel()
            return
        if self.client is not None and hasattr(self.client, "cancel"):
            self.client.cancel()
        self.player.stop()
