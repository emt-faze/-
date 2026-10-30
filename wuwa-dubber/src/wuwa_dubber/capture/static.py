from __future__ import annotations

import threading
import time

from .base import CaptureFrame


class StaticCaptureSource:
    """Reusable frame source for tests and UI demo mode."""

    def __init__(self, image: object, width: int = 1920, height: int = 1080) -> None:
        self.image = image
        self.width = width
        self.height = height
        self._latest: CaptureFrame | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._latest = CaptureFrame(
            image=self.image,
            width=self.width,
            height=self.height,
            captured_at=time.monotonic(),
        )

    def read(self, timeout: float = 1.0) -> CaptureFrame | None:
        _ = timeout
        if self._stop.is_set():
            return None
        return self._latest

    def stop(self) -> None:
        self._stop.set()
