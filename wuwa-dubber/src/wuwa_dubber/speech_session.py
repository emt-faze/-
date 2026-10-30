"""Shared sentence playback with one sentence of bounded synthesis lookahead."""
from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .config import AppConfig, CLONING_BACKENDS
from .models import Utterance, VoiceProfile
from .segmentation import split_for_speech
from .speech_cache import CachedSpeech, SpeechCache, SpeechCancelled, valid_wav
from .text_processing import normalize_text


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    index: int
    total: int
    text: str
    path: Path
    cache_hit: bool
    synthesis_seconds: float
    ready_seconds: float


@dataclass(frozen=True, slots=True)
class SpeechReport:
    total_segments: int
    cache_hits: int
    synthesis_seconds: float
    first_audio_seconds: float
    elapsed_seconds: float
    paths: tuple[Path, ...]


class SpeechSession:
    """Run once; model calls are serial, while playback overlaps the next call.

    Cancellation waits for this session's producer to finish before run returns.
    A client is never reused for the next caption while its old request survives.
    """

    def __init__(self, config: AppConfig, cache_dir: Path, client: Any, player: Any):
        self.config = config
        self.cache_dir = cache_dir
        self.client = client
        self.player = player
        self._cancelled = threading.Event()
        self._external_cancel: threading.Event | None = None
        self._started = False

    def _is_cancelled(self) -> bool:
        return self._cancelled.is_set() or bool(self._external_cancel and self._external_cancel.is_set())

    def _cancel_generation(self) -> None:
        cancel = getattr(self.client, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                # Local events still suppress output even if the endpoint exits.
                pass

    def cancel(self) -> None:
        self._cancelled.set()
        if self._external_cancel is not None:
            self._external_cancel.set()
        self.player.stop()
        self._cancel_generation()

    def _replace_combined(self, paths: list[Path], target: Path) -> None:
        """Refresh an existing whole-line cache only after all parts succeed."""
        temporary = target.with_name(f".{target.stem}.{uuid4().hex}.joining.wav")
        try:
            format_key = None
            with wave.open(str(temporary), "wb") as output:
                for path in paths:
                    if self._is_cancelled():
                        raise SpeechCancelled("配音已取消")
                    with wave.open(str(path), "rb") as source:
                        current = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                        if format_key is None:
                            output.setparams(source.getparams())
                            format_key = current
                        elif current != format_key:
                            raise ValueError("各段音频格式不一致，未替换原缓存")
                        while data := source.readframes(16384):
                            output.writeframesraw(data)
            if self._is_cancelled():
                raise SpeechCancelled("配音已取消")
            if not valid_wav(temporary):
                raise ValueError("合并的完整台词无效，未替换原缓存")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def run(
        self, utterance: Utterance, profile: VoiceProfile, *, cancel_event: threading.Event,
        force: bool = False, on_segment: Callable[[SegmentEvent], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> SpeechReport:
        if self._started:
            raise RuntimeError("A speech session can only run once")
        self._started = True
        self._external_cancel = cancel_event
        if self._is_cancelled():
            raise SpeechCancelled("配音已取消")
        started = time.monotonic()
        ready: queue.Queue = queue.Queue(maxsize=2)
        # Includes the currently playing segment. Acquire before synthesis so
        # a full ready queue cannot hide another unbounded in-flight request.
        credits = threading.BoundedSemaphore(2)
        done = object()
        played: list[SegmentEvent] = []
        first_audio_seconds = 0.0

        def put(value) -> bool:
            while not self._is_cancelled():
                try:
                    ready.put(value, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def produce():
            try:
                cache = SpeechCache(self.config, self.cache_dir, self.client)
                normalized = replace(utterance, text=normalize_text(utterance.text))
                if not normalized.text:
                    raise ValueError("朗读文字不能为空")
                if self._is_cancelled():
                    raise SpeechCancelled("配音已取消")
                whole = cache.lookup(normalized, profile)
                max_chars = 36 if self.config.tts_backend in CLONING_BACKENDS else 80
                parts = [normalized.text] if whole and not force else split_for_speech(normalized.text, max_chars=max_chars)
                generated_paths = []
                for index, text in enumerate(parts, 1):
                    while not credits.acquire(timeout=0.05):
                        if self._is_cancelled():
                            return
                    if self._is_cancelled():
                        return
                    if on_progress:
                        on_progress(f"正在准备第 {index}/{len(parts)} 段，优先使用已有配音…")
                    part = replace(normalized, text=text, request_id=uuid4().hex)
                    audio: CachedSpeech = whole if whole and not force else cache.get_or_create(
                        part, profile, cancel_event=cancel_event, force=force,
                    )
                    if self._is_cancelled():
                        return
                    generated_paths.append(audio.path)
                    event = SegmentEvent(index, len(parts), text, audio.path, audio.cache_hit,
                                         audio.synthesis_seconds, time.monotonic() - started)
                    if not put(event):
                        return
                if force and whole and len(parts) > 1:
                    self._replace_combined(generated_paths, whole.path)
                put(done)
            except BaseException as exc:
                put(exc)

        producer = threading.Thread(target=produce, name="sentence-synthesis", daemon=True)
        producer.start()
        completed = False
        try:
            while True:
                if self._is_cancelled():
                    raise SpeechCancelled("配音已取消")
                try:
                    item = ready.get(timeout=0.05)
                except queue.Empty:
                    if self._is_cancelled():
                        raise SpeechCancelled("配音已取消")
                    if not producer.is_alive() and ready.empty():
                        raise RuntimeError("语音生成线程意外结束")
                    continue
                if item is done:
                    completed = True
                    break
                if isinstance(item, BaseException):
                    raise item
                if self._is_cancelled():
                    raise SpeechCancelled("配音已取消")
                if on_segment:
                    on_segment(item)
                if self._is_cancelled():
                    raise SpeechCancelled("配音已取消")
                if not played:
                    first_audio_seconds = time.monotonic() - started
                if on_progress:
                    origin = "缓存" if item.cache_hit else "新生成"
                    on_progress(f"正在播放第 {item.index}/{item.total} 段（{origin}）")
                self.player.play(item.path, cancel_event=cancel_event)
                if self._is_cancelled():
                    raise SpeechCancelled("配音已取消")
                played.append(item)
                credits.release()
        finally:
            if not completed:
                self.cancel()
            # Retry cancellation to cover the narrow race between the cache's
            # event check and the backend publishing its active request ID.
            retry_at = 0.0
            while producer.is_alive():
                if self._is_cancelled() and time.monotonic() >= retry_at:
                    self._cancel_generation()
                    retry_at = time.monotonic() + 0.2
                producer.join(0.05)
        return SpeechReport(len(played), sum(p.cache_hit for p in played),
            sum(p.synthesis_seconds for p in played), first_audio_seconds,
            time.monotonic() - started, tuple(p.path for p in played))
