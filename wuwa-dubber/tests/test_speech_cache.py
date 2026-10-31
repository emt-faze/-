from __future__ import annotations

import hashlib
import json
import threading
import wave
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

from wuwa_dubber.config import AppConfig
from wuwa_dubber.models import Utterance, VoiceProfile
from wuwa_dubber.speech_cache import SpeechCache, SpeechCancelled, valid_wav


def write_audio(path: Path, value: int = 1) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(bytes((value, 0)) * 480)


class CacheClient:
    def __init__(self):
        self.calls = []
        self.identity_calls = 0
        self.revision = "one"
        self.fail = False
        self.cancel_during = None

    def cache_identity(self):
        self.identity_calls += 1
        return {"backend": "test", "model_revision": self.revision, "inference": {"seed": 42}}

    def synthesize(self, utterance, profile, output):
        self.calls.append(utterance.text)
        write_audio(output, len(self.calls))
        if self.cancel_during is not None:
            self.cancel_during.set()
        if self.fail:
            raise RuntimeError("intentional backend failure")
        return 0.25


def cache_setup(tmp_path):
    config = AppConfig(tts_backend="qwen3-tts")
    reference = tmp_path / "reference.wav"
    write_audio(reference)
    profile = VoiceProfile("角色", reference_audio=str(reference), prompt_text="参考台词")
    client = CacheClient()
    return config, profile, client, SpeechCache(config, tmp_path / "cache", client)


def test_cache_shared_between_preview_and_ocr(tmp_path: Path) -> None:
    config, profile, client, cache = cache_setup(tmp_path)
    first = cache.get_or_create(Utterance("  你好! ", "角色", source="preview"), profile, cancel_event=threading.Event())
    config.output_volume = 0.1
    config.tts_token = "changed-token"
    profile.aliases.append("识别别名")
    another_client = CacheClient()
    another = SpeechCache(config, cache.cache_dir, another_client)
    second = another.get_or_create(Utterance("你好！", "识别别名", source="ocr"), profile, cancel_event=threading.Event())
    assert not first.cache_hit and second.cache_hit
    assert first.path == second.path
    assert second.synthesis_seconds == 0
    assert client.calls == ["你好！"] and another_client.calls == []


def test_cache_identity_is_obtained_once_per_instance(tmp_path: Path) -> None:
    _, profile, client, cache = cache_setup(tmp_path)
    utterance = Utterance("测试")
    original = cache.path_for(utterance, profile)
    client.revision = "two"
    assert cache.path_for(utterance, profile) == original
    assert cache.lookup(utterance, profile) is None
    assert client.identity_calls == 1
    another = SpeechCache(cache.config, cache.cache_dir, client)
    assert another.path_for(utterance, profile) != original
    assert client.identity_calls == 2


def test_boundary_fix_does_not_reuse_old_audio(tmp_path: Path) -> None:
    config, profile, client, cache = cache_setup(tmp_path)
    utterance = Utterance("失误了啊，鉴定师。", "角色")
    old = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    with patch.object(client, "cache_identity", return_value={
        "backend": "test", "model_revision": "one",
        "inference": {"seed": 42, "implementation_version": 4,
                      "audio_tokenizer_dtype": "float32", "repetition_penalty": 1.0},
    }):
        revised = SpeechCache(config, cache.cache_dir, client)
        new = revised.get_or_create(utterance, profile, cancel_event=threading.Event())
        repeated = revised.get_or_create(utterance, profile, cancel_event=threading.Event())
    assert old.path != new.path and old.path.is_file()
    assert not new.cache_hit and repeated.cache_hit
    assert len(client.calls) == 2


def test_cache_tracks_reference_contents_and_voice_settings(tmp_path: Path) -> None:
    config, profile, _, cache = cache_setup(tmp_path)
    utterance = Utterance("测试")
    original = cache.path_for(utterance, profile)
    write_audio(Path(profile.reference_audio), 2)
    assert cache.path_for(utterance, profile) != original
    write_audio(Path(profile.reference_audio), 1)
    assert cache.path_for(utterance, profile) == original
    for change in ({"speaker": "另一个角色"}, {"speed": 1.2}, {"prompt_text": "另一句参考"}, {"reference_language": "en"}):
        assert cache.path_for(utterance, replace(profile, **change)) != original
    config.tts_backend = "gpt-sovits"
    assert cache.path_for(utterance, profile) != original


def test_identical_reference_at_new_path_reuses_cache(tmp_path: Path) -> None:
    _, profile, _, cache = cache_setup(tmp_path)
    utterance = Utterance("相同内容")
    original = cache.path_for(utterance, profile)
    second = tmp_path / "moved.wav"
    second.write_bytes(Path(profile.reference_audio).read_bytes())
    assert cache.path_for(utterance, replace(profile, reference_audio=str(second))) == original


def test_system_voice_pitch_and_speed_never_reuse_wrong_audio(tmp_path: Path) -> None:
    cache = SpeechCache(AppConfig(tts_backend="windows-sapi"), tmp_path / "cache", CacheClient())
    profile = VoiceProfile("系统声音", system_voice="male", system_pitch=1)
    utterance = Utterance("相同的一句台词")
    variants = [profile, replace(profile, system_voice="female"), replace(profile, system_pitch=-3),
                replace(profile, speed=.9)]
    results = [cache.get_or_create(utterance, item, cancel_event=threading.Event()) for item in variants]
    assert len({item.path for item in results}) == 4
    assert not any(item.cache_hit for item in results)
    again = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    assert again.cache_hit and again.path == results[0].path


def test_force_failure_preserves_old_valid_audio(tmp_path: Path) -> None:
    _, profile, client, cache = cache_setup(tmp_path)
    utterance = Utterance("重做失败也不能损坏旧缓存")
    first = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    original = first.path.read_bytes()
    client.fail = True
    try:
        cache.get_or_create(utterance, profile, cancel_event=threading.Event(), force=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failed synthesis was accepted")
    assert first.path.read_bytes() == original and valid_wav(first.path)
    assert list(cache.cache_dir.iterdir()) == [first.path]


def test_force_success_replaces_old_audio(tmp_path: Path) -> None:
    _, profile, _, cache = cache_setup(tmp_path)
    utterance = Utterance("重新生成")
    first = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    previous = first.path.read_bytes()
    second = cache.get_or_create(utterance, profile, cancel_event=threading.Event(), force=True)
    assert not second.cache_hit and first.path == second.path
    assert second.path.read_bytes() != previous


def test_cancelled_generation_does_not_publish_audio(tmp_path: Path) -> None:
    _, profile, client, cache = cache_setup(tmp_path)
    cancellation = threading.Event()
    client.cancel_during = cancellation
    try:
        cache.get_or_create(Utterance("已取消"), profile, cancel_event=cancellation)
    except SpeechCancelled:
        pass
    else:
        raise AssertionError("Cancelled generation was accepted")
    assert list(cache.cache_dir.iterdir()) == []


def test_pre_cancelled_cache_hit_does_not_play_or_synthesize(tmp_path: Path) -> None:
    _, profile, client, cache = cache_setup(tmp_path)
    utterance = Utterance("已有缓存")
    cached = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    cancellation = threading.Event()
    cancellation.set()
    try:
        cache.get_or_create(utterance, profile, cancel_event=cancellation)
    except SpeechCancelled:
        pass
    else:
        raise AssertionError("Cancellation was ignored on a cache hit")
    assert len(client.calls) == 1 and valid_wav(cached.path)


def test_truncated_wav_is_rebuilt_even_when_first_frame_exists(tmp_path: Path) -> None:
    _, profile, client, cache = cache_setup(tmp_path)
    utterance = Utterance("重建不完整文件")
    first = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    first.path.write_bytes(first.path.read_bytes()[:-100])
    with wave.open(str(first.path), "rb") as audio:
        assert audio.readframes(1)  # The legacy first-frame-only check accepted this.
    assert not valid_wav(first.path)
    assert cache.lookup(utterance, profile) is None
    rebuilt = cache.get_or_create(utterance, profile, cancel_event=threading.Event())
    assert not rebuilt.cache_hit and valid_wav(rebuilt.path)
    assert len(client.calls) == 2


def test_legacy_v1_audio_is_migrated_without_synthesis(tmp_path: Path) -> None:
    config, profile, client, cache = cache_setup(tmp_path)
    utterance = Utterance("旧版整句缓存", "角色")
    settings = config.to_dict()
    old = {
        "version": 1, "text": utterance.text, "speaker": utterance.speaker,
        "profile": profile.to_dict(),
        "reference_sha256": hashlib.sha256(Path(profile.reference_audio).read_bytes()).hexdigest(),
        "tts": {key: value for key, value in settings.items() if key.startswith(("tts_", "sapi_")) or key == "use_fake_tts"},
        "client": f"{type(client).__module__}.{type(client).__qualname__}",
        "client_settings": client.cache_identity(),
    }
    cache.cache_dir.mkdir()
    old_path = cache.cache_dir / (hashlib.sha256(json.dumps(old, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() + ".wav")
    write_audio(old_path)
    found = cache.lookup(utterance, profile)
    assert found is not None and found.cache_hit and found.path != old_path
    assert old_path.exists() and found.path.read_bytes() == old_path.read_bytes()
    assert client.calls == []
