from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Rect:
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def clamped(self, frame_width: int, frame_height: int) -> "Rect":
        left = max(0, min(self.left, frame_width - 1))
        top = max(0, min(self.top, frame_height - 1))
        right = max(left + 1, min(self.left + self.width, frame_width))
        bottom = max(top + 1, min(self.top + self.height, frame_height))
        return Rect(left, top, right - left, bottom - top)

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "Rect":
        value = value or {}
        return cls(
            left=int(value.get("left", 0)),
            top=int(value.get("top", 0)),
            width=int(value.get("width", 0)),
            height=int(value.get("height", 0)),
        )


@dataclass(slots=True)
class CaptureProfile:
    source: str = "direct"
    window_title: str = ""
    virtual_camera_index: int = 0
    resolution_width: int = 1920
    resolution_height: int = 1080
    dialogue_rect: Rect = field(default_factory=Rect)
    speaker_rect: Rect = field(default_factory=Rect)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "window_title": self.window_title,
            "virtual_camera_index": self.virtual_camera_index,
            "resolution_width": self.resolution_width,
            "resolution_height": self.resolution_height,
            "dialogue_rect": self.dialogue_rect.to_dict(),
            "speaker_rect": self.speaker_rect.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CaptureProfile":
        value = value or {}
        return cls(
            source=str(value.get("source", "direct")),
            window_title=str(value.get("window_title", "")),
            virtual_camera_index=int(value.get("virtual_camera_index", 0)),
            resolution_width=int(value.get("resolution_width", 1920)),
            resolution_height=int(value.get("resolution_height", 1080)),
            dialogue_rect=Rect.from_dict(value.get("dialogue_rect")),
            speaker_rect=Rect.from_dict(value.get("speaker_rect")),
        )


@dataclass(slots=True)
class VoiceProfile:
    speaker: str
    aliases: list[str] = field(default_factory=list)
    reference_audio: str = ""
    prompt_text: str = ""
    reference_language: str = "zh"
    speed: float = 1.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "aliases": self.aliases,
            "reference_audio": self.reference_audio,
            "prompt_text": self.prompt_text,
            "reference_language": self.reference_language,
            "speed": self.speed,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VoiceProfile":
        return cls(
            speaker=str(value.get("speaker", "")).strip(),
            aliases=[str(item).strip() for item in value.get("aliases", []) if str(item).strip()],
            reference_audio=str(value.get("reference_audio", "")),
            prompt_text=str(value.get("prompt_text", "")),
            reference_language=str(value.get("reference_language", "zh")),
            speed=float(value.get("speed", 1.0)),
            enabled=bool(value.get("enabled", True)),
        )

    def resolve_reference(self, base_dir: Path) -> Path:
        path = Path(self.reference_audio)
        return path if path.is_absolute() else base_dir / path

    def validate(self, base_dir: Path, *, require_audio: bool = True) -> list[str]:
        errors: list[str] = []
        if not self.speaker:
            errors.append("角色名不能为空")
        if not self.prompt_text.strip():
            errors.append("参考台词不能为空")
        if not 0.5 <= self.speed <= 2.0:
            errors.append("语速必须位于 0.5 到 2.0 之间")
        if require_audio:
            reference = self.resolve_reference(base_dir)
            if not reference.is_file():
                errors.append(f"参考音频不存在: {reference}")
        return errors


@dataclass(slots=True)
class Utterance:
    text: str
    speaker: str = "旁白"
    source: str = "ocr"
    captured_at: float = 0.0
    fingerprint: str = ""
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(slots=True)
class SynthesisResult:
    request_id: str
    audio_path: Path
    elapsed_seconds: float


@dataclass(slots=True)
class PipelineSnapshot:
    running: bool = False
    paused: bool = False
    capture_ready: bool = False
    tts_ready: bool = False
    last_text: str = ""
    last_speaker: str = ""
    last_latency_seconds: float | None = None
    queue_depth: int = 0
    skipped_count: int = 0
    message: str = "尚未启动"
