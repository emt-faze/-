from __future__ import annotations

from collections import deque
from collections.abc import Callable

from .models import Utterance


class PlaybackQueue:
    """Bounded FIFO queue that keeps the newest content when the reader falls behind."""

    def __init__(self, pending_limit: int = 2) -> None:
        if pending_limit < 0:
            raise ValueError("pending_limit cannot be negative")
        self.pending_limit = pending_limit
        self._pending: deque[Utterance] = deque()
        self._current: Utterance | None = None
        self._skipped: deque[Utterance] = deque(maxlen=20)
        self._on_skip: Callable[[Utterance], None] | None = None

    @property
    def pending(self) -> tuple[Utterance, ...]:
        return tuple(self._pending)

    @property
    def current(self) -> Utterance | None:
        return self._current

    @property
    def skipped(self) -> tuple[Utterance, ...]:
        return tuple(self._skipped)

    @property
    def depth(self) -> int:
        return len(self._pending) + (1 if self._current is not None else 0)

    def set_skip_callback(self, callback: Callable[[Utterance], None] | None) -> None:
        self._on_skip = callback

    def enqueue(self, utterance: Utterance) -> tuple[Utterance, ...]:
        dropped: list[Utterance] = []
        self._pending.append(utterance)
        # Even with no backlog, an idle worker needs one handoff slot.
        capacity = max(1, self.pending_limit) if self._current is None else self.pending_limit
        while len(self._pending) > capacity and self._pending:
            dropped.append(self._pending.popleft())
        for item in dropped:
            self._skipped.append(item)
            if self._on_skip is not None:
                self._on_skip(item)
        return tuple(dropped)

    def pop_next(self) -> Utterance | None:
        if not self._pending:
            self._current = None
            return None
        self._current = self._pending.popleft()
        return self._current

    def finish_current(self) -> None:
        self._current = None

    def clear_pending(self) -> tuple[Utterance, ...]:
        items = tuple(self._pending)
        self._pending.clear()
        return items

    def jump_to_latest(self) -> Utterance | None:
        if not self._pending:
            return None
        latest = self._pending[-1]
        for item in self._pending:
            if item is not latest:
                self._skipped.append(item)
                if self._on_skip is not None:
                    self._on_skip(item)
        self._pending.clear()
        self._pending.append(latest)
        return latest

    def replay_skipped(self) -> tuple[Utterance, ...]:
        items = tuple(self._skipped)
        self._skipped.clear()
        for item in items:
            self.enqueue(item)
        return items
