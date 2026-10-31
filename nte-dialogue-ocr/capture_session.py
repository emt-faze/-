"""Latest-frame capture with physical client coordinates and visibility guards.

WGC is the default. Capture and cropped frame copies stay outside the GUI.
The optional MSS mode keeps its instance on its owning worker thread.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import threading
import time

import numpy as np
import win32gui
import win32con

from capture_windows import (GameCapture as LegacyCapture, CaptureTimeout, _assert_window_identity,
                             _usable_frame, enable_dpi_awareness)
from ocr_core import roi_to_pixels


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_at: float
    image: np.ndarray
    client_size: tuple[int, int]
    origin: tuple[int, int]
    screen_rect: tuple[int, int, int, int]
    grab_ms: float
    transparent_overlays: tuple[int, ...] = ()

    def __iter__(self):
        return iter((self.sequence, self.captured_at, self.image))

    def __getitem__(self, index):
        return (self.sequence, self.captured_at, self.image)[index]


def region_bounds(client_size, rois=None):
    width, height = client_size
    if not rois:
        return 0, 0, width, height
    boxes = [roi_to_pixels((height, width), roi) for roi in rois if roi is not None]
    if not boxes:
        return 0, 0, width, height
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _cloaked(hwnd):
    value = wintypes.DWORD()
    result = ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 14,
                                                       ctypes.byref(value), ctypes.sizeof(value))
    return result == 0 and bool(value.value)


def _point_occluders(window,rect):
    """Also inspect input surfaces (for compositor/notification child windows)."""
    found=set()
    for x in np.linspace(rect[0]+2,rect[2]-3,11,dtype=int):
        for y in np.linspace(rect[1]+2,rect[3]-3,4,dtype=int):
            handle=win32gui.WindowFromPoint((int(x),int(y)))
            if not handle:
                continue
            root=win32gui.GetAncestor(handle,win32con.GA_ROOT)
            if root and root!=window.hwnd and win32gui.IsWindowVisible(root):
                if intersects(rect,win32gui.GetWindowRect(handle)):
                    found.add(root)
    return sorted(found)


def visible_region(window, rois=None, *, require_foreground=True, check_occlusion=True):
    """Return pause reason and exact screen/client geometry; never activate here."""
    _assert_window_identity(window)
    if win32gui.IsIconic(window.hwnd) or not win32gui.IsWindowVisible(window.hwnd):
        return "minimized", None
    if require_foreground and win32gui.GetForegroundWindow() != window.hwnd:
        return "not_foreground", None
    _, _, width, height = win32gui.GetClientRect(window.hwnd)
    if min(width, height) < 100:
        return "invalid_geometry", None
    x, y = win32gui.ClientToScreen(window.hwnd, (0, 0))
    left, top, right, bottom = region_bounds((width, height), rois)
    rect = (x+left, y+top, x+right, y+bottom)
    if not check_occlusion:
        # WGC obtains the window surface, not the desktop pixels covering it.
        return "ready", {"client_size": (width, height), "origin": (left, top),
                         "screen_rect": rect, "occluders": [], "transparent_overlays": []}
    occluders = []
    transparent_overlays = []
    reached_game = False

    def visit(hwnd, _):
        nonlocal reached_game
        if hwnd == window.hwnd:
            reached_game = True
            return False
        try:
            if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd) or _cloaked(hwnd):
                return True
            bounds = win32gui.GetWindowRect(hwnd)
            if intersects(rect, bounds):
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                if style & win32con.WS_EX_LAYERED:
                    try:
                        color, alpha, flags = win32gui.GetLayeredWindowAttributes(hwnd)
                        if flags & 2 and alpha == 0:
                            return True
                    except win32gui.error:
                        pass
                if style & win32con.WS_EX_LAYERED and style & win32con.WS_EX_TRANSPARENT:
                    # Per-pixel, mouse-through overlays can own a transparent
                    # full-screen rectangle (observed NVIDIA CEF overlay).
                    # Win32 has no public per-pixel visibility query for them.
                    # Record that limitation; do not claim their full bounding
                    # rectangle is opaque or silently call them verified clear.
                    transparent_overlays.append(hwnd)
                else:
                    occluders.append(hwnd)
        except win32gui.error:
            pass
        return True

    win32gui.EnumWindows(visit, None)
    if not reached_game:
        return "window_unavailable", None
    occluders.extend(handle for handle in _point_occluders(window,rect) if handle not in occluders)
    geometry = {"client_size": (width, height), "origin": (left, top),
                "screen_rect": rect, "occluders": occluders,
                "transparent_overlays": transparent_overlays}
    return ("occluded" if occluders else "ready"), geometry


PAUSE_TEXT = {
    "starting": "正在连接画面…", "not_foreground": "已暂停：请切回游戏",
    "minimized": "已暂停：游戏已最小化或隐藏", "occluded": "已暂停：文字区域被其他窗口遮挡",
    "invalid_geometry": "等待游戏窗口尺寸稳定", "window_unavailable": "等待游戏窗口",
    "offscreen": "已暂停：采集区域超出屏幕", "changing": "窗口正在移动，等待新画面",
    "blank": "等待有效游戏画面", "stale": "等待新游戏画面", "stopped": "采集已停止",
    "ready": "正在获取游戏画面",
}


class CaptureSession:
    def __init__(self, window, interval_ms=200, backend="wgc", rois=None, max_age=.75,
                 require_foreground=True, first_frame_timeout=2.5):
        if backend not in ("mss", "wgc", "printwindow"):
            raise ValueError("请选择 mss、wgc 或 printwindow；不会自动切换截图方式")
        if interval_ms <= 0 or max_age <= 0 or first_frame_timeout <= 0:
            raise ValueError("截图间隔和有效时间必须为正数")
        self.window, self.backend_name = window, backend
        self.interval = interval_ms / 1000
        self.rois, self.max_age = rois, max_age
        self.require_foreground = require_foreground
        self.first_frame_timeout = first_frame_timeout
        self.fallback_reason = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._error = None
        self._state = "starting"
        self._geometry = None
        self._sequence = 0
        self._durations = deque(maxlen=1000)
        self._ages = deque(maxlen=1000)
        self._captured = self._discarded = 0

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def status_text(self):
        return PAUSE_TEXT.get(self.state, self.state)

    def _pause(self, state, geometry=None):
        with self._lock:
            self._state, self._geometry = state, geometry
            self._latest = None

    def _visibility(self):
        return visible_region(self.window, self.rois,
                              require_foreground=self.require_foreground,
                              check_occlusion=self.backend_name != "wgc")

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="game-capture", daemon=True)
            self._thread.start()
        return self

    def _run(self):
        enable_dpi_awareness()
        legacy = mss_instance = None
        try:
            if self.backend_name == "mss":
                import mss
                mss_instance = mss.mss()
            else:
                legacy = LegacyCapture(self.window, interval_ms=int(self.interval*1000),
                                       backend=self.backend_name, first_frame_timeout=self.first_frame_timeout).start()
            legacy_sequence = 0
            stable_geometry = None
            while not self._stop.is_set():
                began = time.monotonic()
                state, geometry = self._visibility()
                if state != "ready":
                    stable_geometry = None
                    self._pause(state, geometry)
                    self._stop.wait(self.interval)
                    continue
                geometry_key = (geometry["client_size"], geometry["origin"], geometry["screen_rect"])
                if stable_geometry != geometry_key:
                    stable_geometry = geometry_key
                    self._pause("changing", geometry)
                    self._stop.wait(self.interval)
                    continue
                rect = geometry["screen_rect"]
                tick = time.perf_counter()
                capture_started = time.monotonic()
                if mss_instance is not None:
                    monitor = mss_instance.monitors[0]
                    if not (monitor["left"] <= rect[0] < rect[2] <= monitor["left"]+monitor["width"]
                            and monitor["top"] <= rect[1] < rect[3] <= monitor["top"]+monitor["height"]):
                        self._pause("offscreen", geometry)
                        self._stop.wait(self.interval)
                        continue
                    image = np.array(mss_instance.grab({"left": rect[0], "top": rect[1],
                                      "width": rect[2]-rect[0], "height": rect[3]-rect[1]}))[:, :, :3].copy()
                    # Conservative age: pixels were sampled sometime during
                    # grab, not necessarily at the moment it returned.
                    captured_at = capture_started
                else:
                    packet = legacy.read(legacy_sequence)
                    if packet is None:
                        self._pause("stale", geometry)
                        self._stop.wait(.05)
                        continue
                    legacy_sequence, captured_at, full_image = packet
                    if tuple(full_image.shape[1::-1]) != tuple(geometry["client_size"]):
                        self._pause("changing", geometry)
                        self._stop.wait(self.interval)
                        continue
                    left, top, right, bottom = region_bounds(geometry["client_size"], self.rois)
                    image = full_image[top:bottom, left:right].copy()
                grab_ms = (time.perf_counter()-tick)*1000
                age_ms = (time.monotonic()-captured_at)*1000
                if grab_ms > self.max_age*1000 or age_ms > self.max_age*1000:
                    self._discarded += 1
                    self._pause("stale", geometry)
                    self._stop.wait(self.interval)
                    continue
                after_state, after_geometry = self._visibility()
                if after_state != "ready" or after_geometry != geometry:
                    self._discarded += 1
                    self._pause(after_state if after_state != "ready" else "changing", after_geometry)
                    continue
                if not _usable_frame(image):
                    self._pause("blank", geometry)
                    self._stop.wait(self.interval)
                    continue
                image.setflags(write=False)
                with self._lock:
                    self._sequence += 1
                    self._latest = FramePacket(self._sequence, captured_at, image,
                                               geometry["client_size"], geometry["origin"], rect, grab_ms)
                    if geometry.get("transparent_overlays"):
                        from dataclasses import replace
                        self._latest = replace(self._latest, transparent_overlays=tuple(geometry["transparent_overlays"]))
                    self._state, self._geometry = "ready", geometry
                    self._captured += 1
                    self._durations.append(grab_ms)
                    self._ages.append(age_ms)
                self._stop.wait(max(0, self.interval-(time.monotonic()-began)))
        except Exception as exc:
            with self._lock:
                self._error = exc
                self._latest = None
                self._state = "error"
        finally:
            if mss_instance is not None:
                mss_instance.close()
            if legacy is not None:
                legacy.stop()

    def peek(self):
        """Nonblocking GUI access; no Win32 call and no screenshot on the GUI thread."""
        with self._lock:
            packet = self._latest
            if self._state != "ready" or packet is None or time.monotonic()-packet.captured_at > self.max_age:
                return None
            return packet

    def read(self, after_sequence=0):
        if self._error is not None:
            if isinstance(self._error, CaptureTimeout):
                raise self._error
            raise RuntimeError(f"{self.backend_name} 截图失败：{self._error}；请手动选择其他方式") from self._error
        packet = self.peek()
        return packet if packet is not None and packet.sequence > after_sequence else None

    def ready_for_ocr(self):
        if self._error is not None:
            raise RuntimeError(f"{self.backend_name} 截图失败：{self._error}；请手动选择其他方式") from self._error
        state, geometry = self._visibility()
        if state != "ready":
            self._pause(state, geometry)
            return False
        with self._lock:
            previous = self._geometry
        if previous != geometry:
            self._pause("changing", geometry)
            return False
        return self.state == "ready"

    def metrics(self):
        with self._lock:
            values = list(self._durations)
            ages = list(self._ages)
            return {"backend": self.backend_name, "sample_fps": 1/self.interval,
                    "captured_frames": self._captured, "discarded_frames": self._discarded,
                    "grab_ms_mean": float(np.mean(values)) if values else None,
                    "grab_ms_p95": float(np.percentile(values, 95)) if values else None,
                    "frame_age_ms_mean": float(np.mean(ages)) if ages else None,
                    "frame_age_ms_p95": float(np.percentile(ages, 95)) if ages else None,
                    "timing_note": "WGC grab 为最新帧读取与裁剪耗时，帧年龄从回调收到画面算起" if self.backend_name == "wgc" else "采样开始到读取的时间",
                    "latest_only": True, "state": self._state}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3)
            if self._thread.is_alive():
                raise RuntimeError("截图线程尚未结束，暂不能启动第二个采集实例")
        self._pause("stopped")


def snapshot(window, timeout=8, backend="wgc", stop=None):
    stop = stop or threading.Event()
    capture = CaptureSession(window, backend=backend, require_foreground=backend != "wgc").start()
    try:
        deadline = time.monotonic()+timeout
        while not stop.is_set() and time.monotonic() < deadline:
            packet = capture.read()
            if packet is not None:
                return packet.image
            stop.wait(.05)
        raise RuntimeError("未取得画面："+capture.status_text)
    finally:
        capture.stop()
