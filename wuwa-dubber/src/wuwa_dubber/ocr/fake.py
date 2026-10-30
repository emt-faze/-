from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import OcrLine


class FakeOcrEngine:
    """Deterministic OCR engine used by tests and the UI demo mode."""

    def __init__(self, lines: Iterable[str] = ()) -> None:
        self._lines = list(lines)
        self._index = 0

    def set_lines(self, lines: Iterable[str]) -> None:
        self._lines = list(lines)
        self._index = 0

    def recognize(self, image: Any) -> list[OcrLine]:
        if not self._lines:
            return []
        text = self._lines[min(self._index, len(self._lines) - 1)]
        self._index += 1
        return [OcrLine(text=text, confidence=1.0)]
