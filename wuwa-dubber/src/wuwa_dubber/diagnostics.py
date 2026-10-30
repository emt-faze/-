from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, CLONING_BACKENDS
from .voice_profiles import VoiceProfileStore


@dataclass(slots=True)
class DiagnosticItem:
    name: str
    status: str
    detail: str


def collect_diagnostics(config: AppConfig, voice_store: VoiceProfileStore) -> list[DiagnosticItem]:
    items = [
        _module("PySide6", "PySide6"),
        _module("windows-capture", "windows_capture"),
        _module("OpenCV", "cv2"),
        _module("RapidOCR", "rapidocr_onnxruntime"),
        _module("sounddevice", "sounddevice"),
        _module("NumPy", "numpy"),
        _module("pynput", "pynput"),
        _command("FFmpeg（导入参考音频时需要）", config.ffmpeg_path, optional=True),
        _command("FFprobe（可选，WAV 时长可直接读取）", config.ffprobe_path, optional=True),
        _command("OBS（使用虚拟摄像头时需要）", config.obs_executable, optional=config.capture.source != "obs"),
    ]
    profiles = voice_store.list()
    cloning = config.tts_backend in CLONING_BACKENDS and not config.use_fake_tts
    items.append(DiagnosticItem("声线档案", "ok" if profiles or not cloning else "warning", f"{len(profiles)} 个" + ("；系统朗读无需参考声线" if not cloning else "")))
    try:
        from .tts import create_tts_client
        client = create_tts_client(config)
        if hasattr(client, "timeout"):
            client.timeout = 5.0
        health = client.health()
        items.append(DiagnosticItem("当前朗读引擎", "ok", str(health)))
    except Exception as exc:
        items.append(DiagnosticItem("当前朗读引擎", "missing", str(exc)))
    errors = config.validate()
    items.append(DiagnosticItem("画面与区域配置", "warning" if errors else "ok", "；".join(errors) or "已配置，需在实际游戏中确认"))
    items.append(_nvidia())
    return items


def _module(name: str, module: str) -> DiagnosticItem:
    found = importlib.util.find_spec(module) is not None
    return DiagnosticItem(name, "ok" if found else "missing", "已安装" if found else "未安装")


def _command(name: str, configured: str, *, optional: bool = False) -> DiagnosticItem:
    if not configured:
        return DiagnosticItem(name, "optional" if optional else "missing", "未配置")
    path = Path(configured)
    found = str(path) if path.is_absolute() and path.is_file() else shutil.which(configured)
    return DiagnosticItem(name, "ok" if found else ("optional" if optional else "missing"), str(found or configured))


def _nvidia() -> DiagnosticItem:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return DiagnosticItem("NVIDIA GPU", "ok", result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return DiagnosticItem("NVIDIA GPU", "missing", "未检测到 nvidia-smi")
