from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import CaptureProfile
from .tool_discovery import discover_ffmpeg, discover_obs


@dataclass(slots=True)
class AppConfig:
    capture: CaptureProfile = field(default_factory=CaptureProfile)
    sample_fps: float = 5.0
    stability_frames: int = 3
    dedupe_history: int = 5
    speaker_match_threshold: float = 0.82
    pending_queue_limit: int = 2
    output_device: int | None = None
    output_volume: float = 1.0
    tts_backend: str = "gpt-sovits"
    tts_endpoint: str = "http://127.0.0.1:9881"
    tts_token: str = "local-only-change-me"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    obs_executable: str = ""
    narrator_profile: str = "旁白"
    use_fake_tts: bool = False
    log_ocr_text: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": self.capture.to_dict(),
            "sample_fps": self.sample_fps,
            "stability_frames": self.stability_frames,
            "dedupe_history": self.dedupe_history,
            "speaker_match_threshold": self.speaker_match_threshold,
            "pending_queue_limit": self.pending_queue_limit,
            "output_device": self.output_device,
            "output_volume": self.output_volume,
            "tts_backend": self.tts_backend,
            "tts_endpoint": self.tts_endpoint,
            "tts_token": self.tts_token,
            "ffmpeg_path": self.ffmpeg_path,
            "ffprobe_path": self.ffprobe_path,
            "obs_executable": self.obs_executable,
            "narrator_profile": self.narrator_profile,
            "use_fake_tts": self.use_fake_tts,
            "log_ocr_text": self.log_ocr_text,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AppConfig":
        defaults = cls()
        return cls(
            capture=CaptureProfile.from_dict(value.get("capture")),
            sample_fps=max(0.5, float(value.get("sample_fps", defaults.sample_fps))),
            stability_frames=max(1, int(value.get("stability_frames", defaults.stability_frames))),
            dedupe_history=max(1, int(value.get("dedupe_history", defaults.dedupe_history))),
            speaker_match_threshold=min(
                1.0,
                max(0.0, float(value.get("speaker_match_threshold", defaults.speaker_match_threshold))),
            ),
            pending_queue_limit=max(0, int(value.get("pending_queue_limit", defaults.pending_queue_limit))),
            output_device=(
                None if value.get("output_device") is None else int(value.get("output_device"))
            ),
            output_volume=min(1.5, max(0.0, float(value.get("output_volume", defaults.output_volume)))),
            tts_backend=str(value.get("tts_backend", defaults.tts_backend)),
            tts_endpoint=str(value.get("tts_endpoint", defaults.tts_endpoint)).rstrip("/"),
            tts_token=str(value.get("tts_token", defaults.tts_token)),
            ffmpeg_path=str(value.get("ffmpeg_path", defaults.ffmpeg_path)),
            ffprobe_path=str(value.get("ffprobe_path", defaults.ffprobe_path)),
            obs_executable=str(value.get("obs_executable", defaults.obs_executable)),
            narrator_profile=str(value.get("narrator_profile", defaults.narrator_profile)),
            use_fake_tts=bool(value.get("use_fake_tts", defaults.use_fake_tts)),
            log_ocr_text=bool(value.get("log_ocr_text", defaults.log_ocr_text)),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.5 <= self.sample_fps <= 30.0:
            errors.append("采样帧率必须位于 0.5 到 30 之间")
        if self.capture.source not in {"direct", "obs"}:
            errors.append("捕获来源必须是 direct 或 obs")
        if self.capture.dialogue_rect.is_empty:
            errors.append("尚未配置对白文字区域")
        if self.capture.speaker_rect.is_empty:
            errors.append("尚未配置角色名区域")
        if not self.tts_endpoint.startswith("http://127.0.0.1") and not self.tts_endpoint.startswith("http://localhost"):
            errors.append("TTS 接口必须仅监听本机")
        return errors


@dataclass(slots=True)
class AppPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for path in (self.root, self.profiles, self.cache, self.logs):
            path.mkdir(parents=True, exist_ok=True)



def default_data_root() -> Path:
    override = os.environ.get("WUWA_DUBBER_DATA")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


def load_config(paths: AppPaths | None = None) -> AppConfig:
    paths = paths or AppPaths(default_data_root())
    paths.ensure()
    if not paths.config.is_file():
        config = AppConfig()
        config.ffmpeg_path, config.ffprobe_path = discover_ffmpeg()
        config.obs_executable = discover_obs()
        save_config(config, paths)
        return config
    try:
        value = json.loads(paths.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppConfig()
    config = AppConfig.from_dict(value)
    changed = False
    ffmpeg_path = Path(config.ffmpeg_path)
    ffmpeg_is_usable = ffmpeg_path.is_file() and ffmpeg_path.is_absolute()
    if not ffmpeg_is_usable and (config.ffmpeg_path in {"", "ffmpeg"} or not ffmpeg_path.is_absolute()):
        ffmpeg, ffprobe = discover_ffmpeg()
        if ffmpeg != config.ffmpeg_path:
            config.ffmpeg_path = ffmpeg
            config.ffprobe_path = ffprobe
            changed = True
    if not config.obs_executable or not Path(config.obs_executable).is_file():
        obs = discover_obs()
        if obs and obs != config.obs_executable:
            config.obs_executable = obs
            changed = True
    if changed:
        save_config(config, paths)
    return config


def save_config(config: AppConfig, paths: AppPaths | None = None) -> None:
    paths = paths or AppPaths(default_data_root())
    paths.ensure()
    temporary = paths.config.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(paths.config)
