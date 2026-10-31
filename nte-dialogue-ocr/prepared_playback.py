"""Match fresh subtitles to a frozen project index and play existing WAVs only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import threading
import time
import unicodedata


def text_key(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(text or '')).casefold() if c.isalnum())


@dataclass(frozen=True)
class PreparedLine:
    row_id: str
    text: str
    speaker: str
    voice: str
    path: Path
    sha256: str
    character_voice: bool

    @property
    def identity(self):
        return text_key(self.text), text_key(self.voice), self.sha256


class PreparedLibrary:
    def __init__(self, project, aliases=None):
        self.aliases = dict(aliases or {})
        self.lines, self.by_text = [], {}
        self.excluded = 0
        for row in project.rows:
            path = project.audio_path(row) if row.status == 'done' and not row.review_reason() else None
            if path is None:
                self.excluded += 1
                continue
            line = PreparedLine(row.id, row.text, row.speaker, row.audio.get('voice', ''), path,
                                row.audio['sha256'], project.mode == 'character')
            self.lines.append(line)
            self.by_text.setdefault(text_key(row.text), []).append(line)

    def role_key(self, name):
        key = text_key(name)
        return self.aliases.get(key, key)

    def match(self, text, speaker=''):
        choices = self.by_text.get(text_key(text), [])
        if not choices:
            return None, '当前项目没有这句的完整音频'
        role = self.role_key(speaker)
        if role:
            # Explicit character voice assignments can correct an OCR role name.
            # They also distinguish duplicate text spoken by different characters.
            voices = [line for line in choices if line.character_voice and self.role_key(line.voice) == role]
            choices = voices or [line for line in choices if self.role_key(line.speaker) == role]
            if not choices:
                return None, '台词相同，但角色不对应，等待角色名识别稳定'
        distinct = {line.identity: line for line in choices}
        if len(distinct) != 1:
            return None, '这句有多个音频版本，请识别角色名或在项目中明确声线'
        return next(iter(distinct.values())), ''


class PreparedPlayback:
    """One audio owner, one replaceable pending line, no synthesis dependency."""
    def __init__(self, library, player, callback=lambda event: None, *, clock=time.monotonic,
                 max_age=2.5, confirmations=2):
        self.library, self.player, self.callback = library, player, callback
        self.clock, self.max_age, self.confirmations = clock, max_age, confirmations
        self._condition = threading.Condition(threading.RLock())
        self._closed = self._paused = False
        self._pending = self._active = None
        self._active_line = None
        self._serial = 0
        self._body = ''
        self._current = self._candidate = self._consumed = None
        self._count = 0
        self._last_capture = self._minimum_capture = -math.inf
        self._thread = threading.Thread(target=self._consume, name='prepared-dialogue-playback', daemon=True)
        self._thread.start()

    @property
    def alive(self):
        return self._thread.is_alive()

    @property
    def generation(self):
        with self._condition:
            return self._serial

    def _emit(self, state, message, line=None, **details):
        self.callback(dict(state=state, message=message, row_id=line.row_id if line else None,
                           text=line.text if line else '', voice=line.voice if line else '',
                           generation=details.pop('generation', self._serial), **details))

    def _cancel_locked(self):
        self._serial += 1
        self._pending = None
        if self._active:
            self._active.set()

    def invalidate(self, *, captured_at=None, unavailable=False):
        with self._condition:
            if self._closed:
                return
            if captured_at is not None and captured_at <= self._minimum_capture:
                return
            was_current = bool(self._current or self._candidate or self._pending or self._active)
            if unavailable and (self._active or self._pending):
                self._consumed = None  # Resume only after fresh, confirmed subtitles return.
            self._minimum_capture = max(self._minimum_capture,
                                        self.clock() if captured_at is None else captured_at)
            self._cancel_locked()
            self._current = self._candidate = None
            self._count = 0
            if was_current:
                self._emit('waiting', '画面暂不可用，已停止播放' if unavailable else '字幕画面已变化，等待新台词')

    def observe(self, observation):
        with self._condition:
            if self._closed:
                return
            try:
                captured = float(observation.get('captured_monotonic', observation.get('monotonic_seconds')))
                age = self.clock()-captured
                score = float(observation.get('body_confidence', 0))
            except (ValueError, TypeError):
                self.invalidate()
                return
            if not math.isfinite(captured) or captured < self._minimum_capture or captured <= self._last_capture:
                return
            self._last_capture = captured
            if not -1 <= age <= self.max_age:
                self.invalidate()
                self._emit('stale', '识别结果已过期，等待最新字幕')
                return
            body = text_key(observation.get('body'))
            if body != self._body:
                self._cancel_locked()
                self._consumed = None
                self._body = body
            if not body:
                self._current = self._candidate = None
                self._count = 0
                self._emit('blank', '当前没有需要播放的字幕')
                return
            if (not math.isfinite(score) or score < .9 or any(observation.get(key) for key in
                    ('source_clipped', 'coverage_warning', 'review_coverage_warning'))):
                self._cancel_locked()
                self._current = self._candidate = None
                self._count = 0
                self._emit('held', '字幕可能缺字或识别不稳定，等待下一次识别')
                return
            try:
                speaker_score = float(observation.get('speaker_confidence') or 0)
            except (ValueError, TypeError):
                speaker_score = 0
            speaker = observation.get('speaker', '') if math.isfinite(speaker_score) and speaker_score >= .9 else ''
            line, reason = self.library.match(observation.get('body'), speaker)
            if line is None:
                self._cancel_locked()
                self._current = self._candidate = None
                self._count = 0
                self._emit('unmatched', reason, recognized=observation.get('body', ''))
                return
            if self._current and self._current.identity != line.identity:
                self._cancel_locked()
            self._current = line
            if self._candidate != line.identity:
                self._candidate, self._count = line.identity, 0
            # A render-stable observation already represents several distinct
            # captured frames. Do not wait for another expensive OCR/review pass.
            self._count += self.confirmations if observation.get('render_stable') is True else 1
            if self._paused or line.identity == self._consumed:
                return
            if ((self._pending and self._pending[0].identity==line.identity)
                    or (self._active and not self._active.is_set() and self._active_line==line.identity)):
                return
            if self._count < self.confirmations:
                self._emit('matching', '已找到音频，等待字幕稳定', line)
                return
            self._queue_locked(line)

    def _queue_locked(self, line):
        self._cancel_locked()
        self._pending = (line, self._serial)
        self._emit('queued', '已匹配，准备播放本地音频', line)
        self._condition.notify_all()

    def pause(self, value=True):
        with self._condition:
            self._paused = bool(value)
            if value:
                self._cancel_locked()
            elif (self._current and self._count >= self.confirmations
                  and self.clock()-self._last_capture <= self.max_age):
                self._queue_locked(self._current)
            self._emit('paused' if value else 'ready', '已暂停自动播放，识别仍继续' if value else '已继续自动播放')

    def skip(self):
        with self._condition:
            self._cancel_locked()
            if self._current:
                self._consumed = self._current.identity
            self._emit('skipped', '已跳过当前句，等待下一句')

    def replay(self):
        with self._condition:
            if (not self._closed and not self._paused and self._current
                    and self._count >= self.confirmations and self.clock()-self._last_capture <= self.max_age):
                self._queue_locked(self._current)
            else:
                self._emit('held', '当前没有已确认且可播放的字幕')

    def stop(self):
        with self._condition:
            self._closed = True
            self._cancel_locked()
            self._condition.notify_all()

    def close(self, timeout=None):
        self.stop()
        self._thread.join(timeout)
        return not self.alive

    def _consume(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                (line, generation), self._pending = self._pending, None
                cancel = threading.Event()
                self._active = cancel
                self._active_line = line.identity
            started = False
            def on_started():
                nonlocal started
                with self._condition:
                    if not cancel.is_set() and generation==self._serial:
                        started=True
                        self._consumed=line.identity
                        self._emit('playing','正在播放：'+(line.speaker or line.voice),line,generation=generation)
            try:
                from wuwa_dubber.speech_cache import valid_wav
                if not valid_wav(line.path) or hashlib.sha256(line.path.read_bytes()).hexdigest() != line.sha256:
                    raise ValueError('音频已更改或损坏，请停止自动播放并重新生成这句')
                with self._condition:
                    if self.clock()-self._last_capture > self.max_age and not cancel.is_set():
                        self._consumed = self._current = self._candidate = None
                        self._count = 0
                        self._emit('stale', '等待最新字幕，已取消过期播放', line, generation=generation)
                        continue
                if not cancel.is_set():
                    self.player.play(line.path, cancel_event=cancel, on_started=on_started)
                if started and not cancel.is_set():
                    self._emit('completed', '本句播放完成，等待下一句', line, generation=generation)
            except Exception as exc:
                if not cancel.is_set():
                    with self._condition:
                        self._consumed=line.identity  # Do not loop on a device/file error; explicit replay can retry.
                    self._emit('error', '播放失败：'+str(exc), line, generation=generation)
            finally:
                with self._condition:
                    if self._active is cancel:
                        self._active = None
                        self._active_line = None
                    self._condition.notify_all()
