"""Capture a verified NTE window; never fall back to desktop pixels.

PrintWindow runs in a separate process because an unresponsive window may
block that Windows API indefinitely. Shared memory carries pixels, and only
short control messages cross the pipe.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, replace
import multiprocessing
from multiprocessing import shared_memory
import ntpath
import re
import threading
import time

import numpy as np
import win32gui
import win32process

GAME_EXE = "htgame.exe"
LAUNCHER_EXES = {"ntegame.exe", "nteglobalgame.exe", "ntetwgame.exe"}
LAUNCHER_CLASS = re.compile(r"Qt\d+QWindowOwnDC\Z")


def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        ctypes.windll.user32.SetProcessDPIAware()


def _query_process_path(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = wintypes.DWORD(32768)
        path = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return path.value
    finally:
        kernel.CloseHandle(handle)


def _window_process(hwnd):
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    return pid, _query_process_path(pid)


def _window_kind(process_path, class_name):
    exe = ntpath.basename(process_path).lower()
    if exe == GAME_EXE and class_name == "UnrealWindow":
        return "game"
    if exe in LAUNCHER_EXES and LAUNCHER_CLASS.fullmatch(class_name):
        return "launcher"
    return None


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    class_name: str
    width: int
    height: int
    pid: int = 0
    process_path: str = ""

    @property
    def is_launcher(self):
        return _window_kind(self.process_path, self.class_name) == "launcher"


def list_game_windows(include_hidden_launchers=False):
    """Find validated game windows, optionally including tray-hidden launchers.

    Hidden launchers are returned only to the startup flow, which can restore
    their real main window. Hidden games and tiny/auxiliary Qt windows are never
    included, and ordinary UI/capture discovery retains its visible-only default.
    """
    found = []
    def visit(hwnd, _):
        try:
            visible = bool(win32gui.IsWindowVisible(hwnd))
            if not visible and not include_hidden_launchers:
                return True
            cls = win32gui.GetClassName(hwnd)
            if cls != "UnrealWindow" and not LAUNCHER_CLASS.fullmatch(cls):
                return True
            pid, path = _window_process(hwnd)
            kind = _window_kind(path, cls)
            if kind is None or (not visible and kind != "launcher"):
                return True
            _, _, width, height = win32gui.GetClientRect(hwnd)
            if win32gui.IsIconic(hwnd):
                left, top, right, bottom = win32gui.GetWindowPlacement(hwnd)[4]
                width, height = right - left, bottom - top
            if width <= 200 or height <= 200:
                return True
            found.append(GameWindow(hwnd, win32gui.GetWindowText(hwnd), cls, width, height, pid, path))
        except (OSError, win32gui.error):
            pass
        return True
    win32gui.EnumWindows(visit, None)
    return sorted(found, key=lambda item: (item.is_launcher, -item.width * item.height))


def _bind_window(window):
    if not win32gui.IsWindow(window.hwnd):
        raise RuntimeError("游戏窗口已关闭，请刷新并重新连接")
    pid, path = _window_process(window.hwnd)
    cls = win32gui.GetClassName(window.hwnd)
    if not _window_kind(path, cls):
        raise RuntimeError("目标不是已验证的异环游戏或启动器窗口")
    if window.pid and window.pid != pid:
        raise RuntimeError("窗口所属进程已变化，请重新连接")
    if window.process_path and ntpath.normcase(window.process_path) != ntpath.normcase(path):
        raise RuntimeError("窗口所属程序已变化，请重新连接")
    if window.class_name != cls:
        raise RuntimeError("窗口类型已变化，请重新连接")
    return replace(window, pid=pid, process_path=path, title=win32gui.GetWindowText(window.hwnd))


def _assert_window_identity(window):
    if not win32gui.IsWindow(window.hwnd):
        raise RuntimeError("游戏窗口已关闭，请刷新并重新连接")
    try:
        pid, path = _window_process(window.hwnd)
        cls = win32gui.GetClassName(window.hwnd)
    except (OSError, win32gui.error) as exc:
        raise RuntimeError("无法验证游戏窗口所属进程，请重新连接；权限不足时以管理员身份启动工具") from exc
    if (pid != window.pid or cls != window.class_name
            or ntpath.normcase(path) != ntpath.normcase(window.process_path)
            or not _window_kind(path, cls)):
        raise RuntimeError("窗口所属进程或类型已变化，已停止捕获，请重新连接")


def _wgc_title_is_unique(window):
    matches = []
    def visit(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and window.title.casefold() in win32gui.GetWindowText(hwnd).casefold():
            matches.append(hwnd)
        return True
    win32gui.EnumWindows(visit, None)
    return bool(window.title) and matches == [window.hwnd]


class GeometryMismatch(RuntimeError):
    pass


class CaptureTimeout(RuntimeError):
    """A capture worker timed out; identity is checked before any reconnect."""
    pass


def client_image(frame, hwnd):
    """Convert BGRA/BGR to the exact client area, rejecting unknown geometry."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise GeometryMismatch("捕获图像格式无效")
    _, _, width, height = win32gui.GetClientRect(hwnd)
    fh, fw = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise GeometryMismatch("游戏客户区尺寸暂时无效")
    if (fw, fh) == (width, height):
        return np.array(frame[:, :, :3], copy=True, order="C")
    cx, cy = win32gui.ClientToScreen(hwnd, (0, 0))
    rectangles = [win32gui.GetWindowRect(hwnd)]
    rect = wintypes.RECT()
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect)) == 0:
        rectangles.insert(0, (rect.left, rect.top, rect.right, rect.bottom))
    for left, top, right, bottom in rectangles:
        if abs(right - left - fw) <= 2 and abs(bottom - top - fh) <= 2:
            x, y = cx - left, cy - top
            if x >= 0 and y >= 0 and x + width <= fw and y + height <= fh:
                return np.array(frame[y:y + height, x:x + width, :3], copy=True, order="C")
    raise GeometryMismatch(f"捕获尺寸 {fw}×{fh} 与客户区 {width}×{height} 不一致，请重新连接窗口")


def _usable_frame(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
        return False
    sample = frame[::max(1, frame.shape[0] // 96), ::max(1, frame.shape[1] // 160)].reshape(-1, 3)
    if int(sample.max()) <= 3:
        return False
    return bool(np.any(sample.max(axis=0).astype(int) - sample.min(axis=0).astype(int) > 2))


def _printwindow_image(hwnd):
    """ONLY call in the worker process: PrintWindow itself has no timeout."""
    import win32ui
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0 or width * height > 50_000_000:
        raise GeometryMismatch("窗口尺寸无效或过大")
    window_dc = win32gui.GetWindowDC(hwnd)
    source_dc = memory_dc = bitmap = previous = None
    try:
        source_dc = win32ui.CreateDCFromHandle(window_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        previous = memory_dc.SelectObject(bitmap)
        print_window = ctypes.windll.user32.PrintWindow
        print_window.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        print_window.restype = wintypes.BOOL
        if not print_window(hwnd, memory_dc.GetSafeHdc(), 2):
            raise RuntimeError("PrintWindow 未返回有效画面，请尝试无边框窗口模式")
        raw = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8).reshape(height, width, 4)
        return client_image(raw, hwnd)
    finally:
        if memory_dc is not None and previous is not None:
            memory_dc.SelectObject(previous)
        if bitmap is not None:
            win32gui.DeleteObject(bitmap.GetHandle())
        if memory_dc is not None:
            memory_dc.DeleteDC()
        if source_dc is not None:
            source_dc.DeleteDC()
        if window_dc:
            win32gui.ReleaseDC(hwnd, window_dc)


def _printwindow_worker(window, memory_name, capacity, connection):
    enable_dpi_awareness()
    memory = shared_memory.SharedMemory(name=memory_name)
    try:
        while True:
            if connection.recv() != "capture":
                break
            try:
                _assert_window_identity(window)
                if win32gui.IsIconic(window.hwnd) or not win32gui.IsWindowVisible(window.hwnd):
                    connection.send(("paused",))
                    continue
                frame = _printwindow_image(window.hwnd)
                if frame.nbytes > capacity:
                    raise RuntimeError("窗口分辨率已超过当前缓冲区，请重新连接")
                target = np.ndarray(frame.shape, dtype=np.uint8, buffer=memory.buf)
                target[:] = frame
                del target
                connection.send(("frame", time.monotonic(), frame.shape))
            except GeometryMismatch as exc:
                connection.send(("geometry", str(exc)))
            except Exception as exc:
                connection.send(("error", str(exc)))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()
        memory.close()


class _PrintWindowBackend:
    def __init__(self, window, interval_seconds, request_timeout=2.5):
        self.window = window
        self.interval = interval_seconds
        self.timeout = request_timeout
        self.process = self.connection = self.memory = None
        self.pending_since = None
        self.last_request = -float("inf")

    def start(self):
        capacity = max(self.window.width * self.window.height, 3840 * 2160) * 3
        self.memory = shared_memory.SharedMemory(create=True, size=capacity)
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe(duplex=True)
        self.process = context.Process(target=_printwindow_worker,
                                       args=(self.window, self.memory.name, capacity, child), daemon=True)
        try:
            self.process.start()
        except Exception:
            self.stop()
            raise
        finally:
            child.close()
        return self

    def poll(self, now):
        result = None
        if self.connection.poll():
            message = self.connection.recv()
            self.pending_since = None
            if message[0] == "error":
                raise RuntimeError(message[1])
            if message[0] == "geometry":
                raise GeometryMismatch(message[1])
            if message[0] == "frame":
                shape = tuple(message[2])
                if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0 or np.prod(shape) > self.memory.size:
                    raise RuntimeError("截图子进程返回了无效尺寸")
                result = (float(message[1]), np.ndarray(shape, dtype=np.uint8, buffer=self.memory.buf).copy())
        if self.pending_since is not None and now - self.pending_since > self.timeout:
            self.stop()
            raise CaptureTimeout("窗口截图超过等待时限，已结束截图子进程；请恢复游戏或尝试无边框窗口模式后重连")
        if not self.process.is_alive():
            raise RuntimeError("窗口截图子进程已结束，请重新连接")
        if self.pending_since is None and now - self.last_request >= self.interval:
            self.connection.send("capture")
            self.pending_since = self.last_request = now
        return result

    def stop(self):
        process = self.process
        if process is not None and process.pid:
            if process.is_alive():
                try:
                    self.connection.send("stop")
                except (OSError, EOFError):
                    pass
                process.join(.2)
            if process.is_alive():
                process.terminate()
                process.join(.3)
            if process.is_alive():
                process.kill()
                process.join(.2)
            if not process.is_alive():
                process.close()
        self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None


class GameCapture:
    def __init__(self, window, interval_ms=200, backend="auto", first_frame_timeout=2.5):
        if backend not in ("auto", "wgc", "printwindow"):
            raise ValueError("backend 必须是 auto、wgc 或 printwindow")
        if interval_ms < 1 or first_frame_timeout <= 0:
            raise ValueError("截图间隔和等待时限必须为正数")
        self.window = window
        self.interval_ms = interval_ms
        self.backend = backend
        self.first_frame_timeout = first_frame_timeout
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.control = self.capture = None
        self.closed = False
        self._started = False
        self._backend_name = "none"
        self._backend_started = 0.0
        self._ever_received = False
        self._wgc_failure = None
        self._geometry_error = self._geometry_since = None
        self._paused_since = None
        self._print_backend = None
        self.fallback_reason = None
        self._last_wgc_timestamp = None

    @property
    def backend_name(self):
        return self._backend_name

    def start(self):
        if self._started:
            return self
        self.window = _bind_window(self.window)
        self.closed = False
        self._started = True
        try:
            if self.backend == "printwindow":
                self._start_printwindow()
            else:
                try:
                    self._start_wgc()
                except Exception as exc:
                    if self.backend != "auto":
                        raise RuntimeError("WGC 无法捕获所选窗口：" + str(exc)) from exc
                    self.fallback_reason = str(exc)
                    self._start_printwindow()
        except Exception:
            self.stop()
            raise
        return self

    def _start_wgc(self):
        if not _wgc_title_is_unique(self.window):
            raise RuntimeError("窗口标题为空或存在同名可见窗口，无法安全地通过标题选择 WGC 目标")
        from windows_capture import WindowsCapture
        self._backend_name = "wgc"
        self._backend_started = time.monotonic()
        self._ever_received = False
        self._wgc_failure = None
        self._last_wgc_timestamp = None
        self.capture = WindowsCapture(window_name=self.window.title, cursor_capture=False,
                                      draw_border=False, minimum_update_interval=self.interval_ms)
        @self.capture.event
        def on_frame_arrived(frame, capture_control):
            try:
                if self.closed or self._backend_name != "wgc":
                    capture_control.stop()
                    return
                _assert_window_identity(self.window)
                if win32gui.IsIconic(self.window.hwnd) or not win32gui.IsWindowVisible(self.window.hwnd):
                    self._clear_latest()
                    return
                stamp = getattr(frame, "timespan", None)
                # windows-capture 1.5 can invoke the callback twice for a
                # padded row pitch. Those are not two fresh observations.
                if stamp is not None and stamp == self._last_wgc_timestamp:
                    return
                self._last_wgc_timestamp = stamp
                self._accept_frame(client_image(frame.frame_buffer, self.window.hwnd), time.monotonic())
            except GeometryMismatch as exc:
                self._geometry_failed(str(exc))
            except Exception as exc:
                self._wgc_failure = str(exc)
        @self.capture.event
        def on_closed():
            if not self.closed and self._backend_name == "wgc":
                self._wgc_failure = "WGC 会话已结束"
        self.control = self.capture.start_free_threaded()

    def _stop_wgc(self):
        control, self.control = self.control, None
        self.capture = None
        if control is not None:
            control.stop()

    def _start_printwindow(self):
        self._backend_name = "printwindow"
        self._stop_wgc()
        self._clear_latest()
        self._geometry_since = self._geometry_error = None
        self._backend_started = time.monotonic()
        self._ever_received = False
        if not win32gui.IsIconic(self.window.hwnd) and win32gui.IsWindowVisible(self.window.hwnd):
            self._print_backend = _PrintWindowBackend(self.window, self.interval_ms / 1000).start()

    def _clear_latest(self):
        with self.lock:
            self.latest = None

    def _geometry_failed(self, message):
        with self.lock:
            self.latest = None
            self._geometry_error = message
            if self._geometry_since is None:
                self._geometry_since = time.monotonic()

    def _accept_frame(self, data, captured_at):
        if not _usable_frame(data):
            self._clear_latest()
            return False
        with self.lock:
            self._geometry_since = self._geometry_error = None
            self.sequence += 1
            self.latest = (self.sequence, captured_at, data)
            self._ever_received = True
        return True

    def read(self, after_sequence=0):
        if self.closed or not self._started:
            raise RuntimeError("窗口捕获已停止")
        _assert_window_identity(self.window)
        now = time.monotonic()
        if win32gui.IsIconic(self.window.hwnd) or not win32gui.IsWindowVisible(self.window.hwnd):
            self._clear_latest()
            self._geometry_since = self._geometry_error = None
            if self._paused_since is None:
                self._paused_since = now
            if self._print_backend is not None:
                self._print_backend.stop()
                self._print_backend = None
            return None
        if self._paused_since is not None:
            self._backend_started += now - self._paused_since
            self._paused_since = None
            self._clear_latest()
        if self._geometry_since is not None and now - self._geometry_since > 1.5:
            raise RuntimeError(self._geometry_error)
        if self._backend_name == "wgc":
            failure = self._wgc_failure
            if not failure and not self._ever_received and now - self._backend_started > self.first_frame_timeout:
                failure = "WGC 在等待时限内没有返回有效画面"
            if failure:
                if self.backend != "auto":
                    raise RuntimeError(failure)
                self.fallback_reason = failure
                self._start_printwindow()
        if self._backend_name == "printwindow":
            if self._print_backend is None:
                self._print_backend = _PrintWindowBackend(self.window, self.interval_ms / 1000).start()
            try:
                packet = self._print_backend.poll(now)
                if packet is not None:
                    self._accept_frame(packet[1], packet[0])
            except GeometryMismatch as exc:
                self._geometry_failed(str(exc))
        with self.lock:
            if self.latest is None or self.latest[0] <= after_sequence or now - self.latest[1] > 2.5:
                return None
            return self.latest

    def stop(self):
        self.closed = True
        self._started = False
        self._clear_latest()
        try:
            self._stop_wgc()
        finally:
            if self._print_backend is not None:
                self._print_backend.stop()
                self._print_backend = None


def snapshot(window, timeout=8, backend="auto"):
    capture = GameCapture(window, backend=backend).start()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = capture.read()
            if result is not None:
                return result[2]
            time.sleep(.05)
        raise RuntimeError("未收到有效游戏画面；请恢复游戏并尝试无边框窗口模式后重试")
    finally:
        capture.stop()
