from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

from wuwa_dubber.config import AppConfig
from wuwa_dubber.models import Utterance, VoiceProfile
from wuwa_dubber.speech_cache import SpeechCache, SpeechCancelled
from wuwa_dubber.speech_session import SpeechSession


TEXT = "第一句话已经准备好了。第二句话随后就会播放。第三句话也要保持顺序。"


class Model:
    def __init__(self, hold_call=0, fail_call=0):
        self.calls = []
        self.hold_call, self.fail_call = hold_call, fail_call
        self.held = threading.Event()
        self.release = threading.Event()
        self.second_ready = threading.Event()

    def cache_identity(self):
        return {"model_revision": "test-v1"}

    def synthesize(self, utterance, profile, output):
        self.calls.append(utterance.text)
        number = len(self.calls)
        if number == self.hold_call:
            self.held.set()
            assert self.release.wait(3), "Test did not release the held model call"
        if number == self.fail_call:
            raise RuntimeError("deliberate generation failure")
        with wave.open(str(output), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(number.to_bytes(2, "little") * 160)
        if number == 2:
            self.second_ready.set()
        return .01

    def cancel(self):
        self.release.set()


class Player:
    def __init__(self, hold_first=False):
        self.paths = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.hold_first = hold_first

    def play(self, path, *, cancel_event=None):
        if cancel_event and cancel_event.is_set():
            return
        self.paths.append(path)
        self.started.set()
        if self.hold_first and len(self.paths) == 1:
            assert self.release.wait(3), "Test did not release playback"

    def stop(self):
        self.release.set()


def background(session, text=TEXT):
    results, errors = [], []
    def work():
        try:
            results.append(session.run(Utterance(text), VoiceProfile("旁白"), cancel_event=threading.Event()))
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=work)
    thread.start()
    return thread, results, errors


def test_first_segment_plays_while_next_is_generated_and_lookahead_is_bounded(tmp_path: Path):
    model, player = Model(), Player(hold_first=True)
    session = SpeechSession(AppConfig(), tmp_path, model, player)
    thread, results, errors = background(session)
    try:
        assert player.started.wait(2)
        assert model.second_ready.wait(2), "Next sentence was not prepared during playback"
        assert len(model.calls) == 2, "More than one sentence was generated ahead"
        assert len(player.paths) == 1, "First playback must precede the remaining segments"
        player.release.set()
        thread.join(3)
        assert not thread.is_alive() and not errors, errors
        assert len(model.calls) == 3 and len(player.paths) == 3
        assert results[0].total_segments == 3
    finally:
        session.cancel()
        thread.join(3)


def test_cancellation_during_prefetch_discards_late_audio_and_remaining_segments(tmp_path: Path):
    model, player = Model(hold_call=2), Player(hold_first=True)
    session = SpeechSession(AppConfig(), tmp_path, model, player)
    thread, results, errors = background(session)
    assert player.started.wait(2) and model.held.wait(2)
    session.cancel()
    thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SpeechCancelled), errors
    assert len(model.calls) == 2 and len(player.paths) == 1
    assert len(list(tmp_path.glob("*.wav"))) == 1, "Cancelled prefetch was published as cache"
    assert not results


def test_new_session_reuses_sentence_cache_and_force_refreshes(tmp_path: Path):
    config, profile = AppConfig(), VoiceProfile("旁白")
    first = Model()
    report = SpeechSession(config, tmp_path, first, Player()).run(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    assert report.total_segments == 3 and report.cache_hits == 0
    second = Model()
    replay = SpeechSession(config, tmp_path, second, Player()).run(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    assert second.calls == [] and replay.cache_hits == 3 and replay.paths == report.paths
    refresh = SpeechSession(config, tmp_path, second, Player()).run(
        Utterance(TEXT), profile, cancel_event=threading.Event(), force=True)
    assert len(second.calls) == 3 and refresh.cache_hits == 0


def test_existing_whole_utterance_cache_is_used_before_splitting(tmp_path: Path):
    config, model, profile = AppConfig(), Model(), VoiceProfile("旁白")
    whole = SpeechCache(config, tmp_path, model).get_or_create(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    report = SpeechSession(config, tmp_path, model, Player()).run(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    assert len(model.calls) == 1
    assert report.total_segments == 1 and report.cache_hits == 1 and report.paths == (whole.path,)


def test_latency_upgrade_reuses_all_existing_legacy_sentence_audio(tmp_path: Path):
    config = AppConfig(tts_backend="qwen3-tts")
    model, profile = Model(), VoiceProfile("旁白")
    cache = SpeechCache(config, tmp_path, model)
    parts = ["今天的天气很不错，我们一起出去走走吧。", "接着再去看看新开的书店。"]
    original = [cache.get_or_create(Utterance(text), profile, cancel_event=threading.Event()).path for text in parts]
    report = SpeechSession(config, tmp_path, model, Player()).run(
        Utterance("".join(parts)), profile, cancel_event=threading.Event())
    assert report.paths == tuple(original)
    assert report.cache_hits == 2 and len(model.calls) == 2


def test_configured_two_segment_prefetch_is_bounded_and_plays_in_order(tmp_path: Path):
    model, player = Model(), Player(hold_first=True)
    config = AppConfig(tts_backend="qwen3-tts", first_segment_max_chars=0, prefetch_window=2)
    session = SpeechSession(config, tmp_path, model, player)
    text = TEXT + "第四句话最后才会播放。"
    thread, results, errors = background(session, text)
    try:
        assert player.started.wait(2)
        deadline = time.monotonic() + 2
        while len(model.calls) < 3 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(model.calls) == 3 and len(player.paths) == 1
        player.release.set()
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert len(player.paths) == 4 and results[0].total_segments == 4
        assert "".join(model.calls) == text
    finally:
        session.cancel()
        thread.join(3)


def test_memory_failure_reduces_prefetch_and_retries_only_once(tmp_path: Path):
    class MemoryModel(Model):
        def __init__(self):
            super().__init__()
            self.failed = False
        def synthesize(self, utterance, profile, output):
            if len(self.calls) == 1 and not self.failed:
                self.failed = True
                raise MemoryError("simulated out of memory")
            return super().synthesize(utterance, profile, output)
    model, progress = MemoryModel(), []
    config = AppConfig(tts_backend="qwen3-tts", first_segment_max_chars=0, prefetch_window=2)
    report = SpeechSession(config, tmp_path, model, Player()).run(
        Utterance(TEXT), VoiceProfile("旁白"), cancel_event=threading.Event(), on_progress=progress.append)
    assert model.failed and report.total_segments == 3
    assert any("退回" in message for message in progress)
    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_cancel_in_second_lookahead_discards_inflight_audio(tmp_path: Path):
    model, player = Model(hold_call=3), Player(hold_first=True)
    config = AppConfig(tts_backend="qwen3-tts", first_segment_max_chars=0, prefetch_window=2)
    session = SpeechSession(config, tmp_path, model, player)
    thread, results, errors = background(session, TEXT + "第四句话不会播放。")
    assert player.started.wait(2) and model.held.wait(2)
    session.cancel()
    thread.join(3)
    assert not thread.is_alive() and not results
    assert len(errors) == 1 and isinstance(errors[0], SpeechCancelled)
    assert len(player.paths) == 1 and len(model.calls) == 3
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_persistent_memory_failure_is_reported_after_one_retry(tmp_path: Path):
    class ExhaustedModel(Model):
        def synthesize(self, utterance, profile, output):
            self.calls.append(utterance.text)
            raise MemoryError("simulated out of memory")
    model, player = ExhaustedModel(), Player()
    config = AppConfig(tts_backend="qwen3-tts", prefetch_window=2)
    session = SpeechSession(config, tmp_path, model, player)
    thread, results, errors = background(session)
    thread.join(3)
    assert not thread.is_alive() and not results
    assert len(errors) == 1 and isinstance(errors[0], MemoryError), errors
    assert len(model.calls) == 2 and not player.paths
    assert not list(tmp_path.glob("*.wav"))


def test_cancel_before_session_start_never_calls_model(tmp_path: Path):
    model, player = Model(), Player()
    session = SpeechSession(AppConfig(), tmp_path, model, player)
    session.cancel()
    try:
        session.run(Utterance(TEXT), VoiceProfile("旁白"), cancel_event=threading.Event())
    except SpeechCancelled:
        pass
    else:
        raise AssertionError("Pre-cancelled session unexpectedly ran")
    assert not model.calls and not player.paths


def test_failed_later_segment_reports_error_after_first_playback_without_hanging(tmp_path: Path):
    model, player = Model(fail_call=2), Player()
    thread, results, errors = background(SpeechSession(AppConfig(), tmp_path, model, player))
    thread.join(3)
    assert not thread.is_alive() and not results
    assert len(errors) == 1 and "deliberate generation failure" in str(errors[0])
    assert len(player.paths) == 1 and len(model.calls) == 2


def test_force_refresh_replaces_old_whole_line_with_complete_new_sentences(tmp_path: Path):
    config, model, profile = AppConfig(), Model(), VoiceProfile("旁白")
    whole = SpeechCache(config, tmp_path, model).get_or_create(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    original = whole.path.read_bytes()
    refreshed = SpeechSession(config, tmp_path, model, Player()).run(
        Utterance(TEXT), profile, cancel_event=threading.Event(), force=True)
    assert refreshed.total_segments == 3 and len(model.calls) == 4
    assert whole.path.read_bytes() != original
    with wave.open(str(whole.path), "rb") as wav:
        assert wav.getnframes() == 480, "Refreshed whole line did not include all sentences"
    replay = SpeechSession(config, tmp_path, model, Player()).run(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    assert len(model.calls) == 4 and replay.cache_hits == 1 and replay.paths == (whole.path,)


def test_force_failure_keeps_previous_complete_whole_line(tmp_path: Path):
    config, model, profile = AppConfig(), Model(fail_call=3), VoiceProfile("旁白")
    whole = SpeechCache(config, tmp_path, model).get_or_create(
        Utterance(TEXT), profile, cancel_event=threading.Event())
    original = whole.path.read_bytes()
    try:
        SpeechSession(config, tmp_path, model, Player()).run(
            Utterance(TEXT), profile, cancel_event=threading.Event(), force=True)
    except RuntimeError as exc:
        assert "deliberate" in str(exc)
    else:
        raise AssertionError("Expected second sentence to fail")
    assert whole.path.read_bytes() == original
