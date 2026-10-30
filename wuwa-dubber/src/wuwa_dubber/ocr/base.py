from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class OcrLine:
    text: str
    confidence: float = 0.0
    box: Any = None


class OcrEngine(Protocol):
    def recognize(self, image: Any) -> list[OcrLine]: ...
