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


def test_old_config_gets_latency_defaults_without_losing_saved_options() -> None:
    config = AppConfig.from_dict({"output_volume": 0.4, "stability_frames": 4})
    assert config.first_segment_max_chars == 10 and config.segment_max_chars == 36
    assert config.output_volume == 0.4 and config.stability_frames == 4
    config.first_segment_max_chars, config.segment_max_chars = 0, 48
    restored = AppConfig.from_dict(config.to_dict())
    assert restored.first_segment_max_chars == 0 and restored.segment_max_chars == 48


def test_ocr_default_and_saved_latency_options_are_compatible() -> None:
    config = AppConfig.from_dict({})
    assert config.stability_frames == 2 and config.prefetch_window == 2
    saved = AppConfig.from_dict({"stability_frames": 3, "prefetch_window": 1})
    restored = AppConfig.from_dict(saved.to_dict())
    assert restored.stability_frames == 3 and restored.prefetch_window == 1
    assert AppConfig.from_dict({"prefetch_window": 9}).prefetch_window == 3


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
