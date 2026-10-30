from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class CaptureError(RuntimeError):
    pass


@dataclass(slots=True)
class CaptureFrame:
    image: Any
    width: int = 0
    height: int = 0
    captured_at: float = 0.0


class CaptureSource(Protocol):
    def start(self) -> None: ...

    def read(self, timeout: float = 1.0) -> CaptureFrame | None: ...

    def stop(self) -> None: ...
