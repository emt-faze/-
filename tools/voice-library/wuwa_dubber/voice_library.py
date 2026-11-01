from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import VoiceProfile
from .text_processing import normalize_text
from .voice_profiles import VoiceProfileStore

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus"})
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".idea",
        ".venv",
        ".vscode",
        "__pycache__",
        "artifacts",
        "assets",
        "data",
        "logs",
        "scripts",
        "src",
        "tests",
    }
)
_PREFERRED_AUDIO_DIRECTORIES = ("角色包_人声", "资料", "剧情")
_IDEAL_DURATION_SECONDS = 9.0
ProgressCallback = Callable[[int, int, str], None]
ReferenceImporter = Callable[..., VoiceProfile]


@dataclass(frozen=True, slots=True)
class CharacterAudioGroup:
    speaker: str
    directory: Path
    files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class AudioCandidate:
    path: Path
    duration_seconds: float
    source_priority: int


@dataclass(frozen=True, slots=True)
class VoiceBuildResult:
    speaker: str
    status: str
    message: str
    source_audio: Path | None = None
    duration_seconds: float | None = None
    prompt_text: str = ""
    profile: VoiceProfile | None = None


@dataclass(slots=True)
class BatchVoiceBuildSummary:
    model_name: str
    results: list[VoiceBuildResult] = field(default_factory=list)
    cancelled: bool = False

    @property
    def imported(self) -> int:
        return sum(result.status == "imported" for result in self.results)

    @property
    def skipped(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    @property
    def cancelled_count(self) -> int:
        return sum(result.status == "cancelled" for result in self.results)


class SpeechTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str = "medium",
        device: str = "auto",
        download_root: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.download_root = download_root
        self._model: object | None = None

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise ValueError(f"参考音频不存在: {audio_path}")
        model = self._get_model()
        segments, _ = model.transcribe(  # type: ignore[attr-defined]
            str(audio_path),
            language="zh",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            initial_prompt="以下是简体中文剧情对白。",
            without_timestamps=True,
        )
        text = "".join(segment.text.strip() for segment in segments)
        return normalize_transcript(text)

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "缺少 faster-whisper，请重新安装依赖或执行 "
                "python -m pip install faster-whisper"
            ) from exc

        attempts: list[tuple[str, str]]
        if self.device == "cpu":
            attempts = [("cpu", "int8")]
        elif self.device == "cuda":
            attempts = [("cuda", "float16")]
        else:
            attempts = [("cpu", "int8"), ("cuda", "float16")]

        errors: list[str] = []
        for device, compute_type in attempts:
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(self.download_root) if self.download_root else None,
                )
                logger.info(
                    "Loaded Whisper model %s on %s (%s)",
                    self.model_name,
                    device,
                    compute_type,
                )
                return self._model
            except Exception as exc:  # CTranslate2 raises runtime-specific exceptions.
                errors.append(f"{device}/{compute_type}: {exc}")
                logger.warning("Unable to load Whisper on %s: %s", device, exc)
        raise RuntimeError("Whisper 模型加载失败: " + "; ".join(errors))


class BatchVoiceProfileBuilder:
    def __init__(
        self,
        *,
        store: VoiceProfileStore,
        transcriber: SpeechTranscriber,
        ffmpeg_path: str,
        ffprobe_path: str,
        model_name: str = "small",
        max_candidates_per_speaker: int = 4,
        reference_importer: ReferenceImporter | None = None,
        duration_reader: Callable[[Path, str], float | None] | None = None,
    ) -> None:
        self.store = store
        self.transcriber = transcriber
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.model_name = model_name
        self.max_candidates_per_speaker = max(1, max_candidates_per_speaker)
        self.reference_importer = reference_importer or store.import_reference
        self.duration_reader = duration_reader

    def build(
        self,
        root: Path,
        *,
        skip_existing: bool = True,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> BatchVoiceBuildSummary:
        groups = scan_character_audio_groups(root)
        summary = BatchVoiceBuildSummary(model_name=self.model_name)
        total = len(groups)
        if progress is not None:
            progress(0, total, f"已发现 {total} 个角色目录")

        for index, group in enumerate(groups, start=1):
            if _is_cancelled(cancel_event):
                summary.cancelled = True
                break
            if progress is not None:
                progress(index - 1, total, f"正在处理 {group.speaker}")

            if skip_existing and self.store.get(group.speaker) is not None:
                summary.results.append(
                    VoiceBuildResult(
                        speaker=group.speaker,
                        status="skipped",
                        message="已有声线档案",
                    )
                )
                if progress is not None:
                    progress(index, total, f"{group.speaker} 已跳过")
                continue

            result, cancelled = self._build_one(group, cancel_event)
            summary.results.append(result)
            if cancelled:
                summary.cancelled = True
                break
            if progress is not None:
                progress(index, total, f"{group.speaker}: {result.message}")

        if summary.cancelled:
            completed = {result.speaker for result in summary.results}
            for group in groups:
                if group.speaker not in completed:
                    summary.results.append(
                        VoiceBuildResult(
                            speaker=group.speaker,
                            status="cancelled",
                            message="已取消",
                        )
                    )
        if progress is not None:
            progress(total, total, "批量建立完成" if not summary.cancelled else "批量建立已取消")
        return summary

    def _build_one(
        self,
        group: CharacterAudioGroup,
        cancel_event: threading.Event | None,
    ) -> tuple[VoiceBuildResult, bool]:
        candidates = rank_audio_candidates(
            group,
            ffprobe_path=self.ffprobe_path,
            max_candidates=self.max_candidates_per_speaker,
            duration_reader=self.duration_reader,
        )
        if not candidates:
            return (
                VoiceBuildResult(
                    speaker=group.speaker,
                    status="failed",
                    message="没有找到 3 到 30 秒的可用音频",
                ),
                False,
            )

        errors: list[str] = []
        for candidate in candidates:
            if _is_cancelled(cancel_event):
                return (
                    VoiceBuildResult(
                        speaker=group.speaker,
                        status="cancelled",
                        message="已取消",
                        source_audio=candidate.path,
                        duration_seconds=candidate.duration_seconds,
                    ),
                    True,
                )
            try:
                prompt_text = self.transcriber.transcribe(candidate.path)
            except Exception as exc:
                errors.append(f"{candidate.path.name} 转写失败: {exc}")
                continue
            if len(normalize_transcript(prompt_text)) < 2:
                errors.append(f"{candidate.path.name} 未识别到有效台词")
                continue
            try:
                profile = self.reference_importer(
                    speaker=group.speaker,
                    source_audio=candidate.path,
                    prompt_text=prompt_text,
                    aliases=[],
                    speed=1.0,
                    ffmpeg_path=self.ffmpeg_path,
                    ffprobe_path=self.ffprobe_path,
                )
            except Exception as exc:
                errors.append(f"{candidate.path.name} 导入失败: {exc}")
                continue
            return (
                VoiceBuildResult(
                    speaker=group.speaker,
                    status="imported",
                    message=f"已建立声线（{candidate.duration_seconds:.1f} 秒）",
                    source_audio=candidate.path,
                    duration_seconds=candidate.duration_seconds,
                    prompt_text=prompt_text,
                    profile=profile,
                ),
                False,
            )

        message = errors[-1] if errors else "没有可用的参考片段"
        if len(errors) > 1:
            message = f"{len(errors)} 个片段失败；{message}"
        return (
            VoiceBuildResult(
                speaker=group.speaker,
                status="failed",
                message=message,
            ),
            False,
        )


def scan_character_audio_groups(
    root: Path,
    *,
    excluded_directories: frozenset[str] = EXCLUDED_DIRECTORIES,
) -> list[CharacterAudioGroup]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"角色音频目录不存在: {root}")

    if _looks_like_character_directory(root):
        return [
            CharacterAudioGroup(
                speaker=root.name,
                directory=root,
                files=tuple(_collect_audio_files(root, excluded_directories)),
            )
        ]

    groups: list[CharacterAudioGroup] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or _is_excluded_directory(directory.name, excluded_directories):
            continue
        files = tuple(_collect_audio_files(directory, excluded_directories))
        if files:
            groups.append(
                CharacterAudioGroup(
                    speaker=directory.name,
                    directory=directory,
                    files=files,
                )
            )
    return groups


def rank_audio_candidates(
    group: CharacterAudioGroup,
    *,
    ffprobe_path: str,
    max_candidates: int = 4,
    duration_reader: Callable[[Path, str], float | None] | None = None,
) -> list[AudioCandidate]:
    reader = duration_reader or probe_audio_duration
    ranked: list[tuple[tuple[int, float, int, str], AudioCandidate]] = []
    for path in group.files:
        duration = reader(path, ffprobe_path)
        if duration is None or not 3.0 <= duration <= 30.0:
            continue
        duration_score = _duration_score(duration)
        candidate = AudioCandidate(
            path=path,
            duration_seconds=duration,
            source_priority=_source_priority(path),
        )
        key = (
            duration_score,
            candidate.source_priority,
            abs(duration - _IDEAL_DURATION_SECONDS),
            path.name,
        )
        ranked.append((key, candidate))
    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[: max(1, max_candidates)]]


def probe_audio_duration(path: Path, ffprobe_path: str) -> float | None:
    if path.suffix.lower() == ".flac":
        duration = _read_flac_duration(path)
        if duration is not None:
            return duration
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
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, TimeoutError):
        return None


def normalize_transcript(value: str) -> str:
    text = normalize_text(value or "")
    text = re.sub(r"\s+", "", text)
    return text.strip("。.!！?？,，;；:：-—_ ")


def _read_flac_duration(path: Path) -> float | None:
    try:
        with path.open("rb") as source:
            if source.read(4) != b"fLaC":
                return None
            while True:
                header = source.read(4)
                if len(header) != 4:
                    return None
                block_type = header[0] & 0x7F
                block_length = int.from_bytes(header[1:4], "big")
                if block_type != 0:
                    source.seek(block_length, os.SEEK_CUR)
                    continue
                if block_length < 34:
                    return None
                stream_info = source.read(block_length)
                sample_rate = (
                    (stream_info[10] << 12)
                    | (stream_info[11] << 4)
                    | (stream_info[12] >> 4)
                )
                total_samples = (
                    ((stream_info[13] & 0x0F) << 32)
                    | (stream_info[14] << 24)
                    | (stream_info[15] << 16)
                    | (stream_info[16] << 8)
                    | stream_info[17]
                )
                if sample_rate <= 0 or total_samples <= 0:
                    return None
                return total_samples / sample_rate
    except OSError:
        return None


def _collect_audio_files(root: Path, excluded_directories: frozenset[str]) -> list[Path]:
    files: list[Path] = []
    try:
        for current, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if not name.startswith(".") and not _is_excluded_directory(name, excluded_directories)
            ]
            current_path = Path(current)
            for filename in filenames:
                path = current_path / filename
                if path.suffix.lower() in AUDIO_EXTENSIONS:
                    try:
                        if path.stat().st_size > 0:
                            files.append(path)
                    except OSError:
                        continue
    except OSError:
        return []
    return sorted(files, key=lambda path: (_source_priority(path), path.name.casefold()))


def _looks_like_character_directory(root: Path) -> bool:
    if _contains_audio_files(root):
        return True
    return any(
        (root / directory).is_dir() for directory in _PREFERRED_AUDIO_DIRECTORIES
    )


def _contains_audio_files(root: Path) -> bool:
    try:
        return any(
            path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and path.stat().st_size > 0
            for path in root.iterdir()
        )
    except OSError:
        return False


def _is_excluded_directory(name: str, excluded_directories: frozenset[str]) -> bool:
    return name.startswith(".") or name.casefold() in excluded_directories


def _source_priority(path: Path) -> int:
    parts = {part.casefold() for part in path.parts}
    for index, directory in enumerate(_PREFERRED_AUDIO_DIRECTORIES):
        if directory.casefold() in parts:
            return index
    return len(_PREFERRED_AUDIO_DIRECTORIES)


def _duration_score(duration: float) -> int:
    if 6.0 <= duration <= 12.0:
        return 0
    if 5.0 <= duration <= 15.0:
        return 1
    return 2


def _is_cancelled(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()
