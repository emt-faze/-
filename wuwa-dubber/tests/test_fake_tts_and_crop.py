import wave
from pathlib import Path

from wuwa_dubber.models import Rect, Utterance, VoiceProfile
from wuwa_dubber.pipeline import crop_image
from wuwa_dubber.tts.fake import FakeTtsClient


class Frame:
    def __init__(self, rows: int, columns: int) -> None:
        self.rows = rows
        self.columns = columns
        self.shape = (rows, columns, 3)

    def __getitem__(self, value: object) -> "Frame":
        return Frame(2, 3)


def test_crop_image_uses_rectangle() -> None:
    frame = Frame(1080, 1920)
    cropped = crop_image(frame, Rect(1, 2, 10, 20))
    assert cropped.shape == (2, 3, 3)


def test_fake_tts_creates_valid_wav(tmp_path: Path) -> None:
    client = FakeTtsClient()
    output = tmp_path / "voice.wav"
    elapsed = client.synthesize(Utterance(text="测试台词"), VoiceProfile(speaker="旁白"), output)
    assert elapsed >= 0
    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 32000
        assert handle.getnframes() > 0
