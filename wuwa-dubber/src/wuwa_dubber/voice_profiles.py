from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

from .config import AppPaths
from .models import VoiceProfile


class VoiceProfileStore:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.paths.ensure()

    def list(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        for path in sorted(self.paths.profiles.glob("*/profile.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                profile = VoiceProfile.from_dict(value)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            profiles.append(profile)
        return profiles

    def get(self, speaker: str) -> VoiceProfile | None:
        for profile in self.list():
            if profile.speaker == speaker:
                return profile
        return None

    def delete(self, speaker: str) -> bool:
        profile = self.get(speaker)
        if profile is None:
            return False
        directory = profile.resolve_reference(self.paths.profiles).parent
        if directory.parent != self.paths.profiles:
            return False
        shutil.rmtree(directory)
        return True

    def save(self, profile: VoiceProfile, *, directory_name: str | None = None) -> VoiceProfile:
        errors = profile.validate(self.paths.profiles, require_audio=True)
        if errors:
            raise ValueError("; ".join(errors))
        safe_name = directory_name or _safe_directory_name(profile.speaker)
        directory = self.paths.profiles / safe_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "profile.json").write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return profile

    def import_reference(
        self,
        *,
        speaker: str,
        source_audio: Path,
        prompt_text: str,
        aliases: list[str],
        speed: float,
        ffmpeg_path: str,
        ffprobe_path: str,
    ) -> VoiceProfile:
        if not source_audio.is_file():
            raise ValueError(f"参考音频不存在: {source_audio}")
        directory = self.paths.profiles / _safe_directory_name(speaker)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "reference.wav"
        filter_graph = (
            "highpass=f=80,"
            "loudnorm=I=-20:LRA=11:TP=-1.5,"
            "areverse,"
            "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB,"
            "areverse"
        )
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "32000",
            "-c:a",
            "pcm_s16le",
            "-af",
            filter_graph,
            str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        except FileNotFoundError as exc:
            raise ValueError(f"找不到 FFmpeg: {ffmpeg_path}") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"FFmpeg 处理失败: {exc.stderr.strip()}") from exc
        duration = _probe_duration(output, ffprobe_path)
        if duration is not None and not 3.0 <= duration <= 30.0:
            output.unlink(missing_ok=True)
            raise ValueError(f"处理后参考音频为 {duration:.1f} 秒，应位于 3 到 30 秒之间")
        profile = VoiceProfile(
            speaker=speaker,
            aliases=aliases,
            reference_audio=str(output.relative_to(self.paths.profiles)),
            prompt_text=prompt_text,
            speed=speed,
        )
        return self.save(profile, directory_name=directory.name)


def _safe_directory_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    return cleaned or "voice"


def _probe_duration(path: Path, ffprobe_path: str) -> float | None:
    # Imported references are PCM WAV, so duration verification needs no
    # additional FFprobe binary when a standalone FFmpeg build is available.
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        pass
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, TimeoutError):
        return None
