from __future__ import annotations

from collections import deque
import time

from .models import Utterance
from .text_processing import fingerprint_text, is_likely_ui_text, normalize_speaker, normalize_text


class StableLineDetector:
    """Only emits an utterance after OCR returns the same line repeatedly."""

    def __init__(self, *, required_frames: int = 3, history_size: int = 5) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.required_frames = required_frames
        self.history_size = history_size
        self._candidate_text = ""
        self._candidate_speaker = ""
        self._candidate_count = 0
        self._history: deque[str] = deque(maxlen=history_size)

    def reset(self) -> None:
        self._candidate_text = ""
        self._candidate_speaker = ""
        self._candidate_count = 0
        self._history.clear()

    def observe(self, text: str, speaker: str = "旁白", *, captured_at: float | None = None) -> Utterance | None:
        normalized = normalize_text(text)
        normalized_speaker = normalize_speaker(speaker) or "旁白"
        captured_at = captured_at if captured_at is not None else time.monotonic()

        if is_likely_ui_text(normalized):
            self._clear_candidate()
            return None

        if normalized == self._candidate_text and normalized_speaker == self._candidate_speaker:
            self._candidate_count += 1
        else:
            self._candidate_text = normalized
            self._candidate_speaker = normalized_speaker
            self._candidate_count = 1

        if self._candidate_count < self.required_frames:
            return None

        fingerprint = fingerprint_text(normalized_speaker + "\n" + normalized)
        if fingerprint in self._history:
            return None

        self._history.append(fingerprint)
        return Utterance(
            text=normalized,
            speaker=normalized_speaker,
            captured_at=captured_at,
            fingerprint=fingerprint,
        )

    def _clear_candidate(self) -> None:
        self._candidate_text = ""
        self._candidate_speaker = ""
        self._candidate_count = 0
