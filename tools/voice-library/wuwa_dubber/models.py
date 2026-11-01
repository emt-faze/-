from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

# 框选坐标系版本：1 = 屏幕绝对坐标（旧），2 = 捕获帧图像像素（当前）。
LAYOUT_VERSION = 2


@dataclass(slots=True)
class Rect:
    """画面内的像素矩形。

    坐标一律以「捕获帧图像的左上角」为原点，单位是帧图像像素——
    不是屏幕坐标，也不是 Qt 逻辑坐标。框选时在冻结帧上取点，
    天然满足这个约定。

    ``ref_width`` / ``ref_height`` 记录框选时的帧尺寸。游戏分辨率、
    窗口大小或系统缩放变化后，实际帧尺寸会和它不同，裁剪前按比例
    映射即可继续对齐；为 0 表示旧配置，按原样使用。
    """

    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    ref_width: int = 0
    ref_height: int = 0

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def has_reference(self) -> bool:
        return self.ref_width > 0 and self.ref_height > 0

    def clamped(self, frame_width: int, frame_height: int) -> "Rect":
        """与画面求交集；完全落在画面外时返回空矩形。

        旧实现会把越界区域硬夹成 1 像素的细条，裁出来几乎空白，
        OCR 返回空却没有任何报错——表现为「静默失效」。现在返回空，
        由调用方显式告警。
        """
        left = max(0, self.left)
        top = max(0, self.top)
        right = min(frame_width, self.left + self.width)
        bottom = min(frame_height, self.top + self.height)
        if right <= left or bottom <= top:
            return Rect(0, 0, 0, 0)
        return Rect(left, top, right - left, bottom - top)

    def scaled_to(self, frame_width: int, frame_height: int) -> "Rect":
        """把「参考尺寸下的坐标」映射到「实际帧尺寸下的坐标」。"""
        if not self.has_reference:
            return self
        if self.ref_width == frame_width and self.ref_height == frame_height:
            return self
        sx = frame_width / self.ref_width
        sy = frame_height / self.ref_height
        return Rect(
            left=int(round(self.left * sx)),
            top=int(round(self.top * sy)),
            width=max(1, int(round(self.width * sx))),
            height=max(1, int(round(self.height * sy))),
            ref_width=frame_width,
            ref_height=frame_height,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "ref_width": self.ref_width,
            "ref_height": self.ref_height,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "Rect":
        value = value or {}
        return cls(
            left=int(value.get("left", 0)),
            top=int(value.get("top", 0)),
            width=int(value.get("width", 0)),
            height=int(value.get("height", 0)),
            ref_width=int(value.get("ref_width", 0)),
            ref_height=int(value.get("ref_height", 0)),
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
    # 框选坐标系版本。1 = 屏幕坐标（旧版，已废弃），2 = 帧图像像素（当前）。
    layout_version: int = LAYOUT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "window_title": self.window_title,
            "virtual_camera_index": self.virtual_camera_index,
            "resolution_width": self.resolution_width,
            "resolution_height": self.resolution_height,
            "dialogue_rect": self.dialogue_rect.to_dict(),
            "speaker_rect": self.speaker_rect.to_dict(),
            "layout_version": self.layout_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CaptureProfile":
        value = value or {}
        version = int(value.get("layout_version", 1))
        dialogue_rect = Rect.from_dict(value.get("dialogue_rect"))
        speaker_rect = Rect.from_dict(value.get("speaker_rect"))
        if version < LAYOUT_VERSION:
            # 旧配置存的是屏幕绝对坐标，而裁剪现在按帧图像像素解释，
            # 两者口径不同，沿用只会继续错位——清空并提示重新框选。
            dialogue_rect = Rect()
            speaker_rect = Rect()
        return cls(
            source=str(value.get("source", "direct")),
            window_title=str(value.get("window_title", "")),
            virtual_camera_index=int(value.get("virtual_camera_index", 0)),
            resolution_width=int(value.get("resolution_width", 1920)),
            resolution_height=int(value.get("resolution_height", 1080)),
            dialogue_rect=dialogue_rect,
            speaker_rect=speaker_rect,
            layout_version=LAYOUT_VERSION,
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
