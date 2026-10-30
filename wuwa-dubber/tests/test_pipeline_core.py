from __future__ import annotations

from pathlib import Path

from wuwa_dubber.capture import CaptureFrame
from wuwa_dubber.capture.static import StaticCaptureSource
from wuwa_dubber.config import AppConfig, AppPaths
from wuwa_dubber.models import Rect, Utterance, VoiceProfile
from wuwa_dubber.ocr.base import OcrLine
from wuwa_dubber.pipeline import DubPipeline
from wuwa_dubber.voice_profiles import VoiceProfileStore


class RegionImage:
    def __init__(self, left: int) -> None:
        self.left = left
        self.shape = (100, 100, 3)


class RegionFrame:
    shape = (100, 100, 3)

    def __getitem__(self, key: object) -> RegionImage:
        horizontal = key[1]  # type: ignore[index]
        return RegionImage(int(horizontal.start or 0))


class RegionOcr:
    def recognize(self, image: RegionImage) -> list[OcrLine]:
        if image.left < 50:
            return [OcrLine("今汐", 1.0)]
        return [OcrLine("漂泊者，你来了。", 1.0)]


class NullPlayer:
    def play(self, path: Path, **kwargs: object) -> None:
        return None

    def stop(self) -> None:
        return None


class NullTts:
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        output.write_bytes(b"RIFF")
        return 0.0


def test_fake_ocr_reaches_playback_queue(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path)
    config = AppConfig()
    config.capture.dialogue_rect = Rect(60, 0, 40, 100)
    config.capture.speaker_rect = Rect(0, 0, 40, 100)
    config.stability_frames = 3
    store = VoiceProfileStore(paths)
    reference_dir = paths.profiles / "narrator"
    reference_dir.mkdir()
    (reference_dir / "reference.wav").write_bytes(b"RIFF")
    store.save(
        VoiceProfile(
            speaker="旁白",
            reference_audio="narrator/reference.wav",
            prompt_text="参考台词",
        )
    )
    pipeline = DubPipeline(
        config,
        paths,
        store,
        capture_source=StaticCaptureSource(RegionFrame()),
        ocr_engine=RegionOcr(),
        tts_client=NullTts(),
        audio_player=NullPlayer(),  # type: ignore[arg-type]
    )
    frame = CaptureFrame(RegionFrame(), 100, 100, captured_at=1.0)
    pipeline._process_frame(frame)
    pipeline._process_frame(frame)
    pipeline._process_frame(frame)
    assert [item.text for item in pipeline.queue.pending] == ["漂泊者，你来了。"]
    assert Path(pipeline._resolve_profile("旁白").reference_audio).is_absolute()
