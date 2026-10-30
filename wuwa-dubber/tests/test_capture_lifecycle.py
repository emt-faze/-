from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

from wuwa_dubber.capture.base import CaptureError
from wuwa_dubber.capture.direct import DirectWindowCapture


class NativeControl:
    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.waiting = threading.Event()

    def wait(self) -> None:
        self.waiting.set()
        self.stopped.wait(3.0)

    def stop(self) -> None:
        self.stopped.set()


class NativeCapture:
    instances: list[NativeCapture] = []

    def __init__(self, **options: object) -> None:
        self.options = options
        self.control = NativeControl()
        self.instances.append(self)

    def event(self, handler):
        setattr(self, handler.__name__, handler)
        return handler

    def start_free_threaded(self) -> NativeControl:
        return self.control


def test_stop_capture_without_waiting_for_another_frame_and_restart() -> None:
    NativeCapture.instances.clear()
    module = SimpleNamespace(WindowsCapture=NativeCapture, InternalCaptureControl=object)
    with patch.dict(sys.modules, {"windows_capture": module}):
        capture = DirectWindowCapture("Game", fps=5)
        try:
            capture.start()
            first = NativeCapture.instances[-1]
            assert first.control.waiting.wait(1.0)
            assert first.options["minimum_update_interval"] == 200
            capture.stop()
            assert first.control.stopped.is_set()
            assert capture._thread is None
            capture.start()
            second = NativeCapture.instances[-1]
            assert second is not first
            assert second.control.waiting.wait(1.0)
            assert not capture._stop.is_set()
        finally:
            capture.stop()


def test_closed_game_window_is_reported_instead_of_using_stale_frame() -> None:
    NativeCapture.instances.clear()
    module = SimpleNamespace(WindowsCapture=NativeCapture, InternalCaptureControl=object)
    with patch.dict(sys.modules, {"windows_capture": module}):
        capture = DirectWindowCapture("Game")
        try:
            capture.start()
            native = NativeCapture.instances[-1]
            assert native.control.waiting.wait(1.0)
            native.on_closed()
            try:
                capture.read(timeout=0.01)
            except CaptureError as exc:
                assert "关闭" in str(exc)
            else:
                raise AssertionError("Closed capture should raise CaptureError")
        finally:
            capture.stop()
