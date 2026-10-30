from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from dataclasses import replace
from collections.abc import Callable
from typing import Any

from .audio import AudioPlayer
from .capture import CaptureError, CaptureSource, create_capture_source
from .config import AppConfig, AppPaths
from .event_log import JsonlEventLogger
from .models import PipelineSnapshot, Rect, Utterance, VoiceProfile
from .ocr import OcrEngine, RapidOcrEngine
from .playback_queue import PlaybackQueue
from .speaker_matcher import SpeakerMatcher
from .speech_cache import SpeechCache, SpeechCancelled, valid_wav
from .speech_session import SegmentEvent, SpeechSession
from .stability import StableLineDetector
from .text_processing import merge_ocr_lines
from .tts import SynthesisError, create_tts_client
from .voice_profiles import VoiceProfileStore

logger = logging.getLogger(__name__)


class DubPipeline:
    def __init__(
        self,
        config: AppConfig,
        paths: AppPaths,
        voice_store: VoiceProfileStore,
        *,
        capture_source: CaptureSource | None = None,
        ocr_engine: OcrEngine | None = None,
        tts_client: Any | None = None,
        audio_player: AudioPlayer | None = None,
        on_snapshot: Callable[[PipelineSnapshot], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.paths.ensure()
        self.voice_store = voice_store
        self.capture = capture_source or create_capture_source(config.capture, fps=config.sample_fps)
        self.ocr = ocr_engine or RapidOcrEngine()
        self.tts = tts_client or create_tts_client(config)
        self.player = audio_player or AudioPlayer(config.output_device, volume=config.output_volume)
        self.detector = StableLineDetector(
            required_frames=config.stability_frames,
            history_size=config.dedupe_history,
        )
        self.matcher = SpeakerMatcher(voice_store.list(), threshold=config.speaker_match_threshold)
        self.queue = PlaybackQueue(config.pending_queue_limit)
        self.logger = JsonlEventLogger(paths.logs, log_ocr_text=config.log_ocr_text)
        self._on_snapshot = on_snapshot
        self._on_error = on_error
        self._snapshot = PipelineSnapshot()
        self._snapshot_lock = threading.Lock()
        self._queue_lock = threading.RLock()
        self._running = threading.Event()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_utterance: Utterance | None = None
        self._active_cancel: threading.Event | None = None
        self._active_session: SpeechSession | None = None
        self._generation = 0

    def snapshot(self) -> PipelineSnapshot:
        with self._snapshot_lock:
            return replace(self._snapshot)

    def start(self) -> None:
        if self._running.is_set():
            return
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("上次语音生成仍在结束，请稍后重试启动")
        errors = self.config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        self.tts.health()
        self._stop_event.clear()
        self._paused.clear()
        self.detector.reset()
        try:
            self.capture.start()
        except Exception:
            self.capture.stop()
            raise
        self._running.set()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="wuwa-capture-loop", daemon=True),
            threading.Thread(target=self._playback_loop, name="wuwa-playback-loop", daemon=True),
        ]
        self._update_snapshot(running=True, paused=False, capture_ready=True, tts_ready=True, message="正在监听剧情文字")
        for thread in self._threads:
            thread.start()
        self.logger.write("pipeline_started")

    def stop(self) -> None:
        self._request_stop()
        try:
            self.capture.stop()
        except Exception:
            logger.exception("Unable to stop capture cleanly")
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=3.0)
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        self._update_snapshot(running=False, paused=False, capture_ready=False, queue_depth=0, message="已停止")
        self.logger.write("pipeline_stopped")

    def _cancel_current(self) -> None:
        # Caller holds _queue_lock so cancellation cannot miss a new playback.
        self._generation += 1
        if self._active_cancel is not None:
            self._active_cancel.set()
            cancel = self._active_session.cancel if self._active_session is not None else getattr(self.tts, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    logger.exception("Unable to cancel voice synthesis")
        self.queue.finish_current()

    def _request_stop(self) -> None:
        self._running.clear()
        self._stop_event.set()
        with self._queue_lock:
            self._cancel_current()
            self.queue.clear_pending()
        self.player.stop()

    def close(self) -> None:
        self.stop()

    def pause(self) -> None:
        with self._queue_lock:
            self._paused.set()
            self._cancel_current()
            depth = self.queue.depth
        self.player.stop()
        self._update_snapshot(paused=True, queue_depth=depth, message="已暂停并停止当前语音，识别仍在继续")

    def resume(self) -> None:
        self._paused.clear()
        self._update_snapshot(paused=False, message="正在监听剧情文字")

    def toggle_pause(self) -> None:
        self.resume() if self._paused.is_set() else self.pause()

    def replay_last(self) -> None:
        if self._last_utterance is None:
            return
        with self._queue_lock:
            self.queue.enqueue(replace(self._last_utterance, captured_at=time.monotonic()))
            depth = self.queue.depth
        self._update_snapshot(queue_depth=depth)

    def jump_to_latest(self) -> None:
        with self._queue_lock:
            self._cancel_current()
            latest = self.queue.jump_to_latest()
            depth = self.queue.depth
            skipped_count = len(self.queue.skipped)
        self.player.stop()
        message = "已跳到最新台词" if latest else "当前没有待播台词"
        self._update_snapshot(queue_depth=depth, skipped_count=skipped_count, message=message)

    def reload_profiles(self) -> None:
        profiles = self.voice_store.list()
        self.matcher.reload(profiles)
        self._update_snapshot(message=f"已载入 {len(profiles)} 个声线档案")

    def _capture_loop(self) -> None:
        interval = 1.0 / max(self.config.sample_fps, 0.5)
        while self._running.is_set():
            started = time.monotonic()
            try:
                frame = self.capture.read(timeout=1.0)
            except Exception as exc:
                self._handle_error(f"画面捕获失败: {exc}")
                self._request_stop()
                try:
                    self.capture.stop()
                except Exception:
                    logger.exception("Capture cleanup failed")
                self._update_snapshot(running=False, capture_ready=False, queue_depth=0)
                return
            if frame is not None:
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    logger.exception("Frame processing failed")
                    self._handle_error(f"文字识别失败: {exc}")
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _process_frame(self, frame: Any) -> None:
        dialogue_image = crop_image(frame.image, self.config.capture.dialogue_rect)
        dialogue_lines = self.ocr.recognize(dialogue_image)
        dialogue_text = merge_ocr_lines(line.text for line in dialogue_lines)
        speaker_text = "旁白"
        if not self.config.capture.speaker_rect.is_empty:
            speaker_image = crop_image(frame.image, self.config.capture.speaker_rect)
            speaker_lines = self.ocr.recognize(speaker_image)
            speaker_text = merge_ocr_lines(line.text for line in speaker_lines) or "旁白"
        utterance = self.detector.observe(dialogue_text, speaker_text, captured_at=frame.captured_at)
        if utterance is None:
            return
        match = self.matcher.match(utterance.speaker)
        utterance.speaker = match.speaker if match else self.config.narrator_profile
        with self._queue_lock:
            dropped = self.queue.enqueue(utterance)
            depth = self.queue.depth
            skipped_count = len(self.queue.skipped)
        self.logger.write(
            "utterance_queued",
            request_id=utterance.request_id,
            text=utterance.text,
            speaker=utterance.speaker,
            matched_score=match.score if match else 0.0,
            dropped=[item.request_id for item in dropped],
        )
        self._update_snapshot(
            last_text=utterance.text,
            last_speaker=utterance.speaker,
            queue_depth=depth,
            skipped_count=skipped_count,
            message="已加入配音队列" if not dropped else "播放积压，已跳过快进",
        )

    def _playback_loop(self) -> None:
        while self._running.is_set():
            if self._paused.is_set():
                self._stop_event.wait(0.1)
                continue
            with self._queue_lock:
                utterance = None if self._paused.is_set() else self.queue.pop_next()
                generation = self._generation
                cancel_event = threading.Event()
                if utterance is not None:
                    self._active_cancel = cancel_event
            if utterance is None:
                self._stop_event.wait(0.05)
                continue
            try:
                profile = self._resolve_profile(utterance.speaker)
                session = SpeechSession(self.config, self.paths.cache, self.tts, self.player)
                with self._queue_lock:
                    self._active_session = session
                    if cancel_event.is_set():
                        continue

                def before_play(segment: SegmentEvent) -> None:
                    with self._queue_lock:
                        if (generation != self._generation or cancel_event.is_set()
                                or self._paused.is_set() or not self._running.is_set()):
                            raise SpeechCancelled("本句已被跳过")
                        self._last_utterance = utterance
                    latency = max(0.0, time.monotonic() - utterance.captured_at)
                    self._update_snapshot(
                        last_text=utterance.text, last_speaker=profile.speaker,
                        last_latency_seconds=latency,
                        message=f"正在播放第 {segment.index}/{segment.total} 段" + ("（缓存）" if segment.cache_hit else ""),
                    )
                    self.logger.write(
                        "synthesis_finished", request_id=utterance.request_id, speaker=profile.speaker,
                        segment_index=segment.index, segment_count=segment.total,
                        synthesis_seconds=segment.synthesis_seconds, latency_seconds=latency,
                        audio_path=str(segment.path), cache_hit=segment.cache_hit,
                    )

                report = session.run(utterance, profile, cancel_event=cancel_event, on_segment=before_play)
                self.logger.write("utterance_finished", request_id=utterance.request_id,
                    segments=report.total_segments, cache_hits=report.cache_hits,
                    first_audio_seconds=report.first_audio_seconds, synthesis_seconds=report.synthesis_seconds)
            except SpeechCancelled:
                if utterance is not None:
                    self.logger.write("synthesis_discarded", request_id=utterance.request_id)
            except Exception as exc:
                logger.exception("Synthesis or playback failed")
                # Internal errors also cancel the session's producer. Report
                # them unless a user action has actually superseded this line.
                if generation == self._generation and self._running.is_set() and not self._paused.is_set():
                    self.logger.write("synthesis_failed", request_id=utterance.request_id, error=str(exc))
                    self._handle_error(f"配音失败: {exc}")
            finally:
                with self._queue_lock:
                    if self.queue.current is utterance:
                        self.queue.finish_current()
                    if self._active_cancel is cancel_event:
                        self._active_cancel = None
                        self._active_session = None
                    depth = self.queue.depth
                self._update_snapshot(queue_depth=depth)

    def _cache_path(self, utterance: Utterance, profile: VoiceProfile) -> Path:
        return SpeechCache(self.config, self.paths.cache, self.tts).path_for(utterance, profile)

    @staticmethod
    def _valid_wav(path: Path) -> bool:
        return valid_wav(path)

    def _resolve_profile(self, speaker: str) -> VoiceProfile:
        profile = (
            self.voice_store.get(speaker)
            or self.voice_store.get(self.config.narrator_profile)
            or VoiceProfile(
                speaker=speaker,
                prompt_text="未配置参考台词",
                reference_audio="",
            )
        )
        if profile.reference_audio:
            profile = replace(
                profile,
                reference_audio=str(profile.resolve_reference(self.paths.profiles)),
            )
        return profile

    def _handle_error(self, message: str) -> None:
        self.logger.write("error", message=message)
        self._update_snapshot(message=message)
        if self._on_error is not None:
            self._on_error(message)

    def _update_snapshot(self, **changes: Any) -> None:
        with self._snapshot_lock:
            snapshot = replace(self._snapshot, **changes)
            self._snapshot = snapshot
        if self._on_snapshot is not None:
            self._on_snapshot(snapshot)


def crop_image(image: Any, rect: Rect) -> Any:
    if rect.is_empty:
        return image
    try:
        height, width = image.shape[:2]
    except (AttributeError, ValueError):
        return image
    safe = rect.clamped(width, height)
    cropped = image[safe.top : safe.top + safe.height, safe.left : safe.left + safe.width]
    if getattr(cropped, "shape", (0,)) and len(cropped.shape) == 3 and cropped.shape[2] == 4:
        try:
            import cv2

            return cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)
        except ImportError:
            return cropped[:, :, :3]
    return cropped
