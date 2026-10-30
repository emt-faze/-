from pathlib import Path

from wuwa_dubber.config import AppConfig, AppPaths, load_config, save_config
from wuwa_dubber.models import Rect, VoiceProfile
from wuwa_dubber.voice_profiles import VoiceProfileStore


def test_config_round_trip(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path)
    config = AppConfig()
    config.capture.dialogue_rect = Rect(100, 700, 1200, 220)
    config.capture.speaker_rect = Rect(250, 650, 500, 70)
    save_config(config, paths)
    loaded = load_config(paths)
    assert loaded.capture.dialogue_rect == config.capture.dialogue_rect
    assert loaded.tts_endpoint == "http://127.0.0.1:9881"


def test_voice_profile_store_saves_valid_profile(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path)
    store = VoiceProfileStore(paths)
    profile_dir = paths.profiles / "voice"
    profile_dir.mkdir()
    audio = profile_dir / "reference.wav"
    audio.write_bytes(b"RIFF-test")
    profile = VoiceProfile(
        speaker="旁白",
        aliases=["叙述者"],
        reference_audio="voice/reference.wav",
        prompt_text="这是参考音频。",
    )
    store.save(profile)
    loaded = store.get("旁白")
    assert loaded is not None
    assert loaded.aliases == ["叙述者"]
