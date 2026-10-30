from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

from wuwa_dubber.config import AppConfig, AppPaths
from wuwa_dubber.models import Rect, Utterance, VoiceProfile
from wuwa_dubber.ocr.fake import FakeOcrEngine
from wuwa_dubber.pipeline import DubPipeline
from wuwa_dubber.voice_profiles import VoiceProfileStore


class CountingCapture:
    def __init__(self) -> None:
        self.reads = 0
        self.stopped = False

    def start(self) -> None:
        self.stopped = False

    def read(self, timeout: float = 1.0) -> None:
        self.reads += 1
        return None

    def stop(self) -> None:
        self.stopped = True


class RecordingPlayer:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.played = threading.Event()
        self.stop_count = 0

    def play(self, path: Path, *, cancel_event: threading.Event | None = None) -> None:
        if cancel_event is None or not cancel_event.is_set():
            self.paths.append(path)
            self.played.set()

    def stop(self) -> None:
        self.stop_count += 1


class ControlledTts:
    def __init__(self, *, block_first: bool = True, fail_first: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []
        self.block_first = block_first
        self.fail_first = fail_first

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        self.calls.append(utterance.text)
        if len(self.calls) == 1:
            self.started.set()
            if self.block_first and not self.release.wait(3.0):
                raise RuntimeError("Test synthesis was not released")
            if self.fail_first:
                raise RuntimeError("Broken backend")
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 160)
        return 0.01


def make_pipeline(tmp_path: Path, tts: ControlledTts) -> DubPipeline:
    paths = AppPaths(tmp_path)
    config = AppConfig(sample_fps=10, pending_queue_limit=3)
    config.capture.window_title = "Test capture"
    config.capture.dialogue_rect = Rect(0, 0, 10, 10)
    config.capture.speaker_rect = Rect(0, 0, 10, 10)
    return DubPipeline(
        config, paths, VoiceProfileStore(paths), capture_source=CountingCapture(),
        ocr_engine=FakeOcrEngine(), tts_client=tts, audio_player=RecordingPlayer(),
    )


def enqueue(pipeline: DubPipeline, text: str) -> None:
    with pipeline._queue_lock:
        pipeline.queue.enqueue(Utterance(text=text, captured_at=time.monotonic()))


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for pipeline")
        threading.Event().wait(0.01)


def test_jump_during_synthesis_does_not_play_old_result(tmp_path: Path) -> None:
    tts = ControlledTts()
    pipeline = make_pipeline(tmp_path, tts)
    try:
        pipeline.start()
        enqueue(pipeline, "旧台词")
        assert tts.started.wait(2.0)
        enqueue(pipeline, "中间台词")
        enqueue(pipeline, "最新台词")
        pipeline.jump_to_latest()
        tts.release.set()
        assert pipeline.player.played.wait(2.0)
        assert tts.calls == ["旧台词", "最新台词"]
        assert len(pipeline.player.paths) == 1
        assert pipeline.snapshot().last_text == "最新台词"
    finally:
        tts.release.set()
        pipeline.stop()


def test_pause_resume_still_discards_inflight_old_result(tmp_path: Path) -> None:
    tts = ControlledTts()
    pipeline = make_pipeline(tmp_path, tts)
    try:
        pipeline.start()
        enqueue(pipeline, "暂停前台词")
        assert tts.started.wait(2.0)
        pipeline.pause()
        assert pipeline.player.stop_count > 0
        pipeline.resume()
        enqueue(pipeline, "恢复后台词")
        tts.release.set()
        assert pipeline.player.played.wait(2.0)
        assert len(pipeline.player.paths) == 1
        assert pipeline.snapshot().last_text == "恢复后台词"
    finally:
        tts.release.set()
        pipeline.stop()


def test_stop_during_synthesis_prevents_late_playback(tmp_path: Path) -> None:
    tts = ControlledTts()
    pipeline = make_pipeline(tmp_path, tts)
    pipeline.start()
    enqueue(pipeline, "停止后台词不能继续播放")
    assert tts.started.wait(2.0)
    stopper = threading.Thread(target=pipeline.stop)
    stopper.start()
    wait_until(lambda: not pipeline._running.is_set())
    tts.release.set()
    stopper.join(2.0)
    assert not stopper.is_alive()
    assert pipeline.player.paths == []
    assert pipeline.queue.depth == 0
    assert not any(thread.is_alive() for thread in pipeline._threads)


def test_idle_capture_is_rate_limited(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, ControlledTts(block_first=False))
    try:
        pipeline.start()
        threading.Event().wait(0.27)
        assert 1 <= pipeline.capture.reads <= 5
    finally:
        pipeline.stop()


def test_backend_exception_does_not_kill_playback_worker(tmp_path: Path) -> None:
    tts = ControlledTts(block_first=False, fail_first=True)
    pipeline = make_pipeline(tmp_path, tts)
    try:
        pipeline.start()
        enqueue(pipeline, "会失败的台词")
        enqueue(pipeline, "下一句正常台词")
        assert pipeline.player.played.wait(2.0)
        assert len(tts.calls) == 2
        assert pipeline.snapshot().last_text == "下一句正常台词"
    finally:
        pipeline.stop()


def test_replay_reuses_valid_audio_cache(tmp_path: Path) -> None:
    tts = ControlledTts(block_first=False)
    pipeline = make_pipeline(tmp_path, tts)
    try:
        pipeline.start()
        enqueue(pipeline, "同一句话重播")
        assert pipeline.player.played.wait(2.0)
        wait_until(lambda: pipeline.queue.depth == 0)
        pipeline.replay_last()
        wait_until(lambda: len(pipeline.player.paths) == 2)
        assert len(tts.calls) == 1
        assert pipeline.player.paths[0] == pipeline.player.paths[1]
    finally:
        pipeline.stop()


def test_cache_changes_for_backend_voice_and_reference_content(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path, ControlledTts(block_first=False))
    utterance = Utterance(text="缓存内容")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"first reference")
    profile = VoiceProfile(speaker="角色", reference_audio=str(reference), prompt_text="参考")
    original = pipeline._cache_path(utterance, profile)
    assert original == pipeline._cache_path(Utterance(text="缓存内容"), profile)
    reference.write_bytes(b"other reference")
    assert original != pipeline._cache_path(utterance, profile)
    reference.write_bytes(b"first reference")
    profile.speed = 1.1
    assert original != pipeline._cache_path(utterance, profile)
    profile.speed = 1.0
    pipeline.config.tts_backend = "different-backend"
    assert original != pipeline._cache_path(utterance, profile)


def test_corrupt_wav_is_not_reused(tmp_path: Path) -> None:
    target = tmp_path / "broken.wav"
    target.write_bytes(b"RIFF")
    assert not DubPipeline._valid_wav(target)
