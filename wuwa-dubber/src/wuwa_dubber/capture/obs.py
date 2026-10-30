from __future__ import annotations

import threading
import time

from .base import CaptureError, CaptureFrame


class ObsVirtualCameraCapture:
    """OpenCV/DirectShow adapter for the OBS Virtual Camera output."""

    def __init__(self, device_index: int = 0, *, fps: float = 5.0) -> None:
        self.device_index = device_index
        self.fps = fps
        self._capture = None
        self._latest: CaptureFrame | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            import cv2
        except ImportError as exc:
            raise CaptureError("OBS 虚拟摄像头需要 opencv-python，请安装项目 capture 依赖") from exc
        capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise CaptureError(f"无法打开 OBS 虚拟摄像头设备 {self.device_index}")
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._capture = capture

        def run() -> None:
            interval = 1.0 / max(self.fps, 0.5)
            while not self._stop.is_set():
                started = time.monotonic()
                ok, image = capture.read()
                if not ok or image is None:
                    self._error = "OBS 虚拟摄像头返回空帧"
                    time.sleep(interval)
                    continue
                height, width = image.shape[:2]
                frame = CaptureFrame(image=image, width=width, height=height, captured_at=time.monotonic())
                with self._lock:
                    self._latest = frame
                self._ready.set()
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
            capture.release()

        self._thread = threading.Thread(target=run, name="wuwa-obs-capture", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0) -> CaptureFrame | None:
        if self._error and self._latest is None:
            raise CaptureError(self._error)
        if not self._ready.wait(timeout=max(0.0, timeout)):
            return None
        with self._lock:
            frame = self._latest
        self._ready.clear()
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._capture = None
