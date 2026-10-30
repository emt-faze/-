from __future__ import annotations

import threading
import time

from .base import CaptureError, CaptureFrame


class DirectWindowCapture:
    """Windows Graphics Capture adapter with a pull-style latest-frame buffer."""

    def __init__(self, window_title: str, *, fps: float = 5.0) -> None:
        if not window_title.strip():
            raise ValueError("window_title is required for direct capture")
        self.window_title = window_title.strip()
        self.fps = fps
        self._latest: CaptureFrame | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._control = None
        self._closed = False
        self._error: str = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._closed = False
        self._error = ""
        self._latest = None
        try:
            from windows_capture import InternalCaptureControl, WindowsCapture
        except ImportError as exc:
            raise CaptureError(
                "直接窗口捕获需要 windows-capture，请安装项目 capture 依赖"
            ) from exc

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=self.window_title,
            minimum_update_interval=max(1, round(1000.0 / max(0.5, self.fps))),
        )
        self._capture = capture

        @capture.event
        def on_frame_arrived(frame, capture_control: InternalCaptureControl) -> None:
            if self._stop.is_set():
                capture_control.stop()
                return
            image = getattr(frame, "frame_buffer", None)
            if image is None:
                return
            try:
                image = image.copy()
            except AttributeError:
                pass
            shape = getattr(image, "shape", (0, 0))
            width = int(getattr(frame, "width", shape[1] if len(shape) > 1 else 0))
            height = int(getattr(frame, "height", shape[0] if shape else 0))
            snapshot = CaptureFrame(image=image, width=width, height=height, captured_at=time.monotonic())
            with self._lock:
                self._latest = snapshot
                self._ready.set()
            if self._stop.is_set():
                capture_control.stop()

        @capture.event
        def on_closed() -> None:
            self._closed = True
            self._ready.set()

        def run() -> None:
            try:
                self._control = capture.start_free_threaded()
                if self._stop.is_set():
                    self._control.stop()
                else:
                    self._control.wait()
            except Exception as exc:  # pragma: no cover - depends on Windows capture runtime
                self._error = str(exc)
                self._ready.set()
            finally:
                self._closed = True
                self._ready.set()

        self._thread = threading.Thread(target=run, name="wuwa-direct-capture", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0) -> CaptureFrame | None:
        if self._stop.is_set():
            return None
        if self._error:
            raise CaptureError(self._error)
        if not self._ready.wait(timeout=max(0.0, timeout)):
            return None
        if self._error:
            raise CaptureError(self._error)
        if self._closed and not self._stop.is_set():
            raise CaptureError("游戏窗口已关闭或捕获已结束")
        with self._lock:
            frame = self._latest
            self._ready.clear()
        return frame

    def stop(self) -> None:
        self._stop.set()
        self._ready.set()
        if self._control is not None:
            self._control.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread is None or not self._thread.is_alive():
            self._thread = None
            self._capture = None
            self._control = None
