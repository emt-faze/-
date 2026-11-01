from __future__ import annotations

from pathlib import Path

from wuwa_dubber.config import AppPaths
from wuwa_dubber.models import VoiceProfile
from wuwa_dubber.voice_library import (
    BatchVoiceProfileBuilder,
    CharacterAudioGroup,
    normalize_transcript,
    probe_audio_duration,
    rank_audio_candidates,
    scan_character_audio_groups,
)
from wuwa_dubber.voice_profiles import VoiceProfileStore


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> str:
        return f"测试台词：{audio_path.stem}"


def test_scan_character_audio_groups_skips_project_directories(tmp_path: Path) -> None:
    _write(tmp_path / "今汐" / "角色包_人声" / "voice.flac")
    _write(tmp_path / "今汐" / "剧情" / "story.flac")
    _write(tmp_path / "安可" / "角色包_人声" / "voice.wav")
    _write(tmp_path / "data" / "ignored.flac")
    _write(tmp_path / ".venv" / "ignored.flac")

    groups = scan_character_audio_groups(tmp_path)

    assert [group.speaker for group in groups] == ["今汐", "安可"]
    assert groups[0].files[0].parent.name == "角色包_人声"

    single = scan_character_audio_groups(tmp_path / "今汐")
    assert len(single) == 1
    assert single[0].speaker == "今汐"


def test_rank_audio_candidates_prefers_voice_pack_and_ideal_duration(tmp_path: Path) -> None:
    files = (
        tmp_path / "角色" / "剧情" / "story.flac",
        tmp_path / "角色" / "角色包_人声" / "short.flac",
        tmp_path / "角色" / "角色包_人声" / "ideal.flac",
    )
    for path in files:
        _write(path)
    durations = {"story.flac": 9.1, "short.flac": 4.0, "ideal.flac": 8.5}
    group = CharacterAudioGroup("角色", tmp_path, files)

    candidates = rank_audio_candidates(
        group,
        ffprobe_path="unused",
        max_candidates=2,
        duration_reader=lambda path, _: durations[path.name],
    )

    assert [candidate.path.name for candidate in candidates] == ["ideal.flac", "story.flac"]


def test_probe_audio_duration_reads_flac_stream_info(tmp_path: Path) -> None:
    path = tmp_path / "voice.flac"
    path.write_bytes(_flac_header(sample_rate=32000, total_samples=9 * 32000))

    duration = probe_audio_duration(path, "missing-ffprobe")

    assert duration is not None
    assert abs(duration - 9.0) < 0.01


def test_batch_builder_skips_existing_and_imports_other_roles(tmp_path: Path) -> None:
    _write(tmp_path / "source" / "今汐" / "角色包_人声" / "a.flac")
    _write(tmp_path / "source" / "安可" / "角色包_人声" / "b.flac")

    paths = AppPaths(tmp_path / "data")
    store = VoiceProfileStore(paths)
    existing_dir = paths.profiles / "existing"
    existing_dir.mkdir()
    (existing_dir / "reference.wav").write_bytes(b"RIFF")
    store.save(
        VoiceProfile(
            speaker="今汐",
            reference_audio="existing/reference.wav",
            prompt_text="已有台词",
        )
    )

    calls: list[dict[str, object]] = []

    def importer(**kwargs: object) -> VoiceProfile:
        calls.append(kwargs)
        return VoiceProfile(
            speaker=str(kwargs["speaker"]),
            reference_audio="generated/reference.wav",
            prompt_text=str(kwargs["prompt_text"]),
        )

    builder = BatchVoiceProfileBuilder(
        store=store,
        transcriber=FakeTranscriber(),
        ffmpeg_path="unused-ffmpeg",
        ffprobe_path="unused-ffprobe",
        reference_importer=importer,
        duration_reader=lambda path, _: 8.0 if path.name == "b.flac" else None,
    )
    summary = builder.build(tmp_path / "source", skip_existing=True)

    assert summary.imported == 1
    assert summary.skipped == 1
    assert summary.failed == 0
    assert calls[0]["speaker"] == "安可"
    assert calls[0]["prompt_text"] == "测试台词：b"


def test_normalize_transcript_removes_layout_whitespace() -> None:
    assert normalize_transcript("  你好，  世界。 ") == "你好，世界"


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")


def _flac_header(*, sample_rate: int, total_samples: int) -> bytes:
    packed = (
        (sample_rate << 44)
        | (1 << 41)
        | (15 << 36)
        | total_samples
    )
    stream_info = b"\x00" * 10 + packed.to_bytes(8, "big") + b"\x00" * 16
    return b"fLaC" + bytes([0x80]) + len(stream_info).to_bytes(3, "big") + stream_info
