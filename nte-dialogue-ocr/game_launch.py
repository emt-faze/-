"""Launcher-driven startup, independently implemented from documented Win32 APIs.

The sequence is informed by ok-nte's LauncherTask: locate the official launcher,
capture its UI, click a verified Start button, then attach to HTGame/UnrealWindow.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import cv2
import numpy as np
import win32con
import win32gui
import win32api
import win32process

from capture_windows import CaptureTimeout, list_game_windows, _assert_window_identity
from capture_session import CaptureSession as GameCapture

ROOT = Path(__file__).resolve().parent
LAUNCHERS = {"ntegame.exe", "nteglobalgame.exe", "ntetwgame.exe"}
START_LABELS = {"开始游戏", "启动游戏", "開始遊戲", "啟動遊戲", "startgame", "launchgame", "launch"}


class LaunchCancelled(RuntimeError):
    pass


def is_admin():
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def running_process_names():
    """Check process existence even before its main window has been created."""
    from ctypes import wintypes
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise RuntimeError("无法检查游戏进程，未执行启动操作")
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        found = set()
        okay = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while okay:
            name = entry.szExeFile.lower()
            if name in LAUNCHERS or name == "htgame.exe":
                found.add(name)
            okay = kernel.Process32NextW(handle, ctypes.byref(entry))
        return found
    finally:
        kernel.CloseHandle(handle)


def relaunch_admin(arguments):
    """Use the normal Windows consent flow; a declined prompt is an error."""
    python = Path(os.path.abspath(os.sys.executable))
    pythonw = python.with_name("pythonw.exe")
    if pythonw.exists():
        python = pythonw
    params = subprocess.list2cmdline([str(ROOT / "app.py"), *arguments])
    shell = ctypes.windll.shell32.ShellExecuteW
    shell.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                      ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
    shell.restype = ctypes.c_void_p
    result = shell(None, "runas", str(python), params, str(ROOT), 1)
    if not result or int(result) <= 32:
        raise RuntimeError("管理员授权未完成，自动启动尚未执行。可以稍后再次点击。")


def _launcher_candidate(value):
    value = str(value or "").strip()
    match = re.search(r'([A-Za-z]:\\[^"\r\n]*?\b(?:NTEGame|NTEGlobalGame|NTETWGame)\.exe)', value, re.I)
    candidates = [Path(match.group(1))] if match else []
    base = Path(value.strip('"')) if value else None
    if base:
        candidates.append(base)
        for name in ("NTEGame.exe", "NTEGlobalGame.exe", "NTETWGame.exe"):
            candidates.extend([base / name, base / "NTELauncher" / name])
    for candidate in candidates:
        try:
            if candidate.name.lower() in LAUNCHERS and candidate.is_file():
                return str(candidate.resolve())
        except (OSError, ValueError):
            continue
    return ""


def find_launcher_path(config):
    configured = _launcher_candidate(config.get("launcher_path", ""))
    if configured:
        return configured
    for window in list_game_windows():
        if window.is_launcher and getattr(window, "process_path", ""):
            candidate = _launcher_candidate(window.process_path)
            if candidate:
                return candidate
    import winreg
    keys = (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for key_path in keys:
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                try:
                    with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ | view) as key:
                        for index in range(winreg.QueryInfoKey(key)[0]):
                            try:
                                with winreg.OpenKey(key, winreg.EnumKey(key, index)) as subkey:
                                    values = []
                                    for name in ("DisplayName", "InstallLocation", "DisplayIcon", "UninstallString"):
                                        try:
                                            values.append(str(winreg.QueryValueEx(subkey, name)[0]))
                                        except OSError:
                                            pass
                                    if not any(token in " ".join(values).lower() for token in ("neverness", "ntegame", "ntelauncher", "异环")):
                                        continue
                                    for value in values:
                                        candidate = _launcher_candidate(value)
                                        if candidate:
                                            return candidate
                            except OSError:
                                continue
                except OSError:
                    continue
    return ""


def restore_window(window):
    if not win32gui.IsWindow(window.hwnd):
        raise RuntimeError("窗口已关闭")
    if win32gui.IsIconic(window.hwnd) or not win32gui.IsWindowVisible(window.hwnd):
        ctypes.windll.user32.ShowWindowAsync(ctypes.c_void_p(window.hwnd), win32con.SW_RESTORE)
        win32gui.PostMessage(window.hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)


def _check_stop(stop):
    if stop.is_set():
        raise LaunchCancelled("已取消自动连接，已打开的游戏保留运行")


def _wait_window(want_launcher, stop, timeout, progress, settle=.35):
    end = time.monotonic() + timeout
    last_signature = None
    stable_since = time.monotonic()
    while time.monotonic() < end:
        _check_stop(stop)
        candidates = [w for w in list_game_windows(include_hidden_launchers=want_launcher) if w.is_launcher == want_launcher]
        if candidates:
            window = candidates[0]
            restore_window(window)
            size = win32gui.GetClientRect(window.hwnd)[2:]
            signature = (window.hwnd, *size)
            usable = win32gui.IsWindowVisible(window.hwnd) and not win32gui.IsIconic(window.hwnd) and min(size) >= 300
            if settle <= 0 and usable:
                return replace(window, width=size[0], height=size[1])
            if signature != last_signature or not usable:
                last_signature = signature
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= settle:
                return replace(window, width=size[0], height=size[1])
        stop.wait(.25)
    raise RuntimeError("等待启动器窗口超时" if want_launcher else "等待游戏窗口超时；请查看启动器是否出现更新、登录或权限提示")


class EntryClickGate:
    """Retry only while the same title screen continuously remains visible."""
    def __init__(self, retry_seconds=3.0, max_clicks=3):
        self.retry_seconds = retry_seconds
        self.max_clicks = max_clicks
        self.clicks = 0
        self.last_click = None
        self.title_departed = False

    def observe(self, state, now):
        if state != "title":
            if self.clicks:
                self.title_departed = True
            return False
        if self.title_departed or self.clicks >= self.max_clicks:
            return False
        return self.last_click is None or now - self.last_click >= self.retry_seconds

    def clicked(self, now):
        self.clicks += 1
        self.last_click = now


def _focus_window_worker(window):
    """Keep input-queue activation isolated from UI and cancellation workers."""
    attached = []
    current_thread = win32api.GetCurrentThreadId()
    try:
        _assert_window_identity(window)
        win32gui.PeekMessage(0, 0, 0, win32con.PM_NOREMOVE)
        foreground = win32gui.GetForegroundWindow()
        handles = {window.hwnd, foreground} - {0}
        threads = {win32process.GetWindowThreadProcessId(h)[0] for h in handles}
        for other_thread in threads - {0, current_thread}:
            win32process.AttachThreadInput(current_thread, other_thread, True)
            attached.append(other_thread)
        win32gui.BringWindowToTop(window.hwnd)
        win32gui.SetForegroundWindow(window.hwnd)
    except (OSError, win32gui.error, RuntimeError):
        pass
    finally:
        for other_thread in reversed(attached):
            try:
                win32process.AttachThreadInput(current_thread, other_thread, False)
            except win32gui.error:
                pass


def activate_game(window, stop):
    _assert_window_identity(window)
    restore_window(window)
    _check_stop(stop)
    if win32gui.GetForegroundWindow() == window.hwnd:
        return
    try:
        win32gui.BringWindowToTop(window.hwnd)
        win32gui.SetForegroundWindow(window.hwnd)
    except win32gui.error:
        pass
    for _ in range(3):
        _check_stop(stop)
        if win32gui.GetForegroundWindow() == window.hwnd:
            return
        stop.wait(.1)
    # Input-queue attachment can block inside Win32. A short-lived process gives
    # Stop and the timeout a way to end our own helper without ending the game.
    import multiprocessing
    helper = multiprocessing.get_context("spawn").Process(target=_focus_window_worker, args=(window,), daemon=True)
    helper.start()
    try:
        deadline = time.monotonic() + 2
        while helper.is_alive() and time.monotonic() < deadline:
            _check_stop(stop)
            stop.wait(.05)
    finally:
        if helper.is_alive():
            helper.terminate()
        helper.join(.5)
    _check_stop(stop)
    _assert_window_identity(window)
    if win32gui.GetForegroundWindow() != window.hwnd:
        raise RuntimeError("Windows 未将游戏置于前台，尚未点击；请切回游戏后再试")


def click_enter_game(window, button, frame_size, stop, captured_at=None):
    """A normal foreground mouse click at a verified, freshly detected button."""
    from ctypes import wintypes
    _check_stop(stop)
    if window.is_launcher:
        raise RuntimeError("进入游戏点击只能作用于游戏主窗口")
    activate_game(window, stop)
    _assert_window_identity(window)
    _, _, width, height = win32gui.GetClientRect(window.hwnd)
    x, y = button["center"]
    if (width, height) != tuple(frame_size) or not (.25*width < x < .75*width and .60*height < y < .95*height):
        raise RuntimeError("游戏尺寸或进入按钮位置已变化，未发送点击")
    _check_stop(stop)
    if win32gui.GetForegroundWindow() != window.hwnd:
        raise RuntimeError("游戏已失去前台，未发送点击")
    win32api.SetCursorPos(win32gui.ClientToScreen(window.hwnd, (int(x), int(y))))
    _check_stop(stop)
    if win32gui.GetForegroundWindow() != window.hwnd:
        raise RuntimeError("移动鼠标后游戏已失去前台，未发送点击")
    if captured_at is not None and time.monotonic() - captured_at > 3:
        return False
    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]
    class InputUnion(ctypes.Union):
        _fields_ = [("mi", MouseInput)]
    class Input(ctypes.Structure):
        _anonymous_ = ("data",)
        _fields_ = [("type", wintypes.DWORD), ("data", InputUnion)]
    events = (Input * 2)()
    events[0].mi.dwFlags = win32con.MOUSEEVENTF_LEFTDOWN
    events[1].mi.dwFlags = win32con.MOUSEEVENTF_LEFTUP
    send = ctypes.windll.user32.SendInput
    send.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    send.restype = wintypes.UINT
    sent = send(2, events, ctypes.sizeof(Input))
    if sent != 2:
        if sent == 1:
            # A partial insertion may have left the physical button down.
            release = Input()
            release.mi.dwFlags = win32con.MOUSEEVENTF_LEFTUP
            send(1, ctypes.byref(release), ctypes.sizeof(Input))
        raise RuntimeError("Windows 没有接受完整鼠标点击，请检查管理员权限和游戏前台状态")
    return True


def locate_start_button(frame, engine):
    """Only accept explicit launch text in the launcher's bottom-right region."""
    h, w = frame.shape[:2]
    x0, y0 = int(w * .65), int(h * .72)
    roi = frame[y0:int(h * .97), x0:w]
    detections, _ = engine(roi, use_cls=False)
    matches = []
    update_needed = False
    for points, raw, confidence in detections or []:
        label = re.sub(r"\s+", "", str(raw)).lower()
        if label in {"更新", "更新游戏", "更新遊戲", "立即更新", "update", "updategame"} and confidence >= .85:
            update_needed = True
        if label not in START_LABELS or float(confidence) < .90:
            continue
        box = np.asarray(points, dtype=float)
        if box.shape != (4, 2) or not np.isfinite(box).all():
            continue
        left, top = box.min(axis=0) + [x0, y0]
        right, bottom = box.max(axis=0) + [x0, y0]
        if not (.025 * w < right-left < .32*w and .008*h < bottom-top < .1*h):
            continue
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            continue
        # The verified launcher uses a pale enabled button with dark text.
        patch = frame[max(0, int(top)-8):min(h, int(bottom)+8), max(0, int(left)-12):min(w, int(right)+12)]
        pale = np.all(patch >= 175, axis=2)
        if float(pale.mean()) < .40:
            continue
        matches.append({"text": str(raw), "confidence": float(confidence),
                        "box": [float(left), float(top), float(right), float(bottom)],
                        "center": [int((left+right)/2), int((top+bottom)/2)]})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("画面中有多个启动按钮，无法确定点击目标")
    if update_needed:
        raise RuntimeError("启动器显示需要更新游戏；请完成更新后再次自动连接")
    return None


def click_start(window, button, stop=None):
    stop = stop if stop is not None else threading.Event()
    _check_stop(stop)
    if "htgame.exe" in running_process_names():
        return False
    current = next((w for w in list_game_windows() if w.hwnd == window.hwnd and w.is_launcher), None)
    if current is None or current.pid != window.pid or win32gui.IsIconic(window.hwnd):
        raise RuntimeError("启动器窗口已变化，未发送点击")
    x, y = button["center"]
    _, _, width, height = win32gui.GetClientRect(window.hwnd)
    if (width, height) != (window.width, window.height) or not (0 <= x < width and 0 <= y < height):
        raise RuntimeError("启动器尺寸已变化，请重新尝试自动连接")
    position = (int(y) << 16) | int(x)
    _check_stop(stop)
    try:
        win32gui.PostMessage(window.hwnd, win32con.WM_MOUSEMOVE, 0, position)
        win32gui.PostMessage(window.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, position)
        win32gui.PostMessage(window.hwnd, win32con.WM_LBUTTONUP, 0, position)
    except Exception as exc:
        raise RuntimeError("Windows 拒绝向启动器发送点击，请使用工具的管理员启动入口") from exc
    return True


def _save_evidence(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(".png", frame)
    if not okay:
        raise RuntimeError("无法保存画面验证文件")
    encoded.tofile(str(path))


def ensure_game(config, stop=None, progress=None, timeout=180, preview=None):
    stop = stop if stop is not None else threading.Event()
    progress = progress if progress is not None else lambda message: None
    report_dir = ROOT / "validation"
    report_dir.mkdir(exist_ok=True)
    began = time.monotonic()
    run_dir = report_dir / datetime.now().strftime("startup-%Y%m%d-%H%M%S-%f")
    run_dir.mkdir()
    report = {"started_at": datetime.now().astimezone().isoformat(), "steps": [], "clicked": False,
              "success": None, "in_game_verified": False, "acceptance": config.get("startup_acceptance", "pending"), "milestones_seconds": {},
              "evidence_directory": str(run_dir), "live_preview": str(run_dir / "current.jpg")}
    engine = None
    latest_frame = None
    preview_at = -float("inf")
    saved_preview_at = -float("inf")

    def persist():
        value = json.dumps(report, ensure_ascii=False, indent=2)
        (report_dir / "last-launch.json").write_text(value, encoding="utf-8")
        (run_dir / "report.json").write_text(value, encoding="utf-8")

    def update(message, event=None):
        elapsed = round(time.monotonic() - began, 3)
        entry = {"time": datetime.now().astimezone().isoformat(), "message": message, "elapsed_seconds": elapsed}
        if event:
            entry["event"] = event
            report["milestones_seconds"].setdefault(event, elapsed)
            report["phase"] = event
        report["steps"].append(entry)
        persist()
        progress(message)

    def get_engine():
        nonlocal engine
        if engine is None:
            from rapidocr_onnxruntime import RapidOCR
            from ocr_core import DEFAULT_ENGINE_OPTIONS
            engine = RapidOCR(**DEFAULT_ENGINE_OPTIONS)
        return engine

    def show_frame(frame):
        nonlocal latest_frame, preview_at, saved_preview_at
        latest_frame = frame
        now = time.monotonic()
        if now - preview_at >= .5:
            preview_at = now
            if preview:
                scale = min(1.0, 960 / frame.shape[1])
                preview(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        if now - saved_preview_at >= 3:
            saved_preview_at = now
            ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                image.tofile(str(run_dir / "current.jpg"))

    try:
        _check_stop(stop)
        windows = list_game_windows(include_hidden_launchers=True)
        processes = running_process_names()
        games = [w for w in windows if not w.is_launcher]
        report["cold_start"] = not games and "htgame.exe" not in processes
        report["initial_target_processes"] = sorted(processes)
        update("开始检查完整启动流程", "started")
        if games or "htgame.exe" in processes:
            update("检测到游戏已运行，正在连接画面…")
        else:
            if not is_admin():
                raise RuntimeError("自动打开游戏需要管理员权限，请使用“一键打开游戏并连接”入口完成 Windows 授权")
            launchers = [w for w in windows if w.is_launcher]
            if not launchers and not processes.intersection(LAUNCHERS):
                path = find_launcher_path(config)
                if not path:
                    raise RuntimeError("未找到官方启动器，请先选择 NTEGame.exe 启动器路径")
                _check_stop(stop)
                update("正在打开官方启动器…", "launcher_requested")
                _check_stop(stop)
                subprocess.Popen([path], cwd=str(Path(path).parent), close_fds=True,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
                report["launcher_path"] = path
            launcher = _wait_window(True, stop, 45, update, settle=0 if launchers else .35)
            report["launcher"] = vars(launcher)
            update("正在从启动器画面确认“开始游戏”按钮…", "launcher_ready")
            activate_game(launcher, stop)
            capture = GameCapture(launcher, backend=config.get("capture_backend", "mss")).start()
            try:
                sequence = 0
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    _check_stop(stop)
                    if "htgame.exe" in running_process_names():
                        break
                    packet = capture.read(sequence)
                    if packet:
                        sequence, captured_at, frame = packet
                        show_frame(frame)
                        button = locate_start_button(frame, get_engine())
                        if button:
                            _check_stop(stop)
                            report["start_button"] = button
                            report["launcher_capture_backend"] = capture.backend_name
                            _save_evidence(run_dir / "01-launcher.png", frame)
                            clicked = report["clicked"] = click_start(launcher, button, stop)
                            update("已点击“开始游戏”，等待游戏窗口…" if clicked else "游戏进程已经启动，等待游戏窗口…", "launcher_clicked")
                            break
                    stop.wait(.2)
                else:
                    raise RuntimeError("没有识别到可用的“开始游戏”按钮，请检查启动器画面")
            finally:
                capture.stop()
        game = _wait_window(False, stop, timeout, update, settle=0 if games else .35)
        _check_stop(stop)
        report["game"] = vars(game)
        update("游戏窗口已出现，正在观察启动与进入游戏过程…", "game_window")
        activate_game(game, stop)
        capture = GameCapture(game, backend=config.get("capture_backend", "mss")).start()
        try:
            from game_scene import classify_frame
            end = time.monotonic() + config.get("entry_timeout_seconds", 120)
            gate = EntryClickGate()
            sequence = 0
            last_state = None
            in_game_count = 0
            last_classify = -float("inf")
            in_game_since = None
            reconnects = 0
            while time.monotonic() < end:
                _check_stop(stop)
                try:
                    packet = capture.read(sequence)
                except CaptureTimeout as exc:
                    _check_stop(stop)
                    _assert_window_identity(game)
                    if reconnects >= 3:
                        raise
                    reconnects += 1
                    report.setdefault("capture_reconnects", []).append({
                        "elapsed_seconds": round(time.monotonic() - began, 3),
                        "previous_backend": capture.backend_name, "reason": str(exc)})
                    update(f"加载期间截图暂时超时，正在重新连接画面（{reconnects}/3）…", "capture_reconnecting")
                    capture.stop()
                    _check_stop(stop)
                    capture = GameCapture(game, backend=config.get("capture_backend", "mss")).start()
                    sequence = 0
                    in_game_count = 0
                    in_game_since = None
                    gate.observe("unknown", time.monotonic())
                    continue
                if packet:
                    _check_stop(stop)
                    sequence, captured_at, frame = packet
                    show_frame(frame)
                    report["game_capture_backend"] = capture.backend_name
                    report["capture_fallback_reason"] = capture.fallback_reason
                    report["frame_size"] = [frame.shape[1], frame.shape[0]]
                    if "first_game_frame" not in report["milestones_seconds"]:
                        update("已取得游戏画面，继续检查“进入游戏”入口…", "first_game_frame")
                    now = time.monotonic()
                    if now - last_classify < .3:
                        stop.wait(.05)
                        continue
                    last_classify = now
                    observation = classify_frame(frame, get_engine())
                    _check_stop(stop)
                    state = observation["state"]
                    report["last_scene"] = observation
                    should_click = gate.observe(state, time.monotonic())
                    if state == "blocked":
                        _save_evidence(run_dir / "blocked.png", frame)
                        raise RuntimeError("游戏进入流程被提示阻止：" + "; ".join(str(x) for x in observation.get("evidence", [])))
                    if state == "title":
                        if "title_screen" not in report["milestones_seconds"]:
                            _save_evidence(run_dir / "02-title.png", frame)
                            update("检测到“进入游戏”，准备点击中下方入口…", "title_screen")
                        if should_click and time.monotonic() - captured_at <= 3:
                            button = observation["enter_button"]
                            if not click_enter_game(game, button, report["frame_size"], stop, captured_at=captured_at):
                                continue
                            gate.clicked(time.monotonic())
                            report.setdefault("entry_clicks", []).append({"elapsed_seconds": round(time.monotonic()-began, 3),
                                                                         "button": button, "attempt": gate.clicks})
                            update(f"已点击“进入游戏”（第 {gate.clicks} 次），等待加载…", "enter_clicked")
                        elif gate.clicks >= gate.max_clicks and time.monotonic() - gate.last_click >= 3:
                            raise RuntimeError("已尝试 3 次进入游戏，入口仍未消失；停止继续点击")
                    elif gate.clicks and last_state == "title":
                        update("进入入口已消失，等待游戏内界面…", "entry_disappeared")
                    if state == "in_game":
                        in_game_count += 1
                        if in_game_since is None:
                            in_game_since = time.monotonic()
                        if in_game_count >= 2 and time.monotonic() - in_game_since >= .2:
                            _check_stop(stop)
                            report.update(in_game_verified=True, success=True, verification=observation,
                                          in_game_image=str(run_dir / "03-in-game.png"))
                            update("已确认进入游戏内部，正在保存验收画面…", "in_game")
                            _save_evidence(run_dir / "03-in-game.png", frame)
                            _save_evidence(report_dir / "game-connected.png", frame)
                            _check_stop(stop)
                            update("完整启动已完成，可以校准文字区域或继续识别", "completed")
                            return {"window": game, "frame": frame, "report": report}
                    else:
                        in_game_count = 0
                        in_game_since = None
                    last_state = state
                stop.wait(.05)
            if latest_frame is not None:
                _save_evidence(run_dir / "timeout.png", latest_frame)
            raise RuntimeError("未在时限内确认进入游戏内部；请查看当前画面及启动记录，不能把标题或加载界面视为成功")
        finally:
            capture.stop()
    except Exception as exc:
        report.update(success=False, error=str(exc), phase="cancelled" if isinstance(exc, LaunchCancelled) else "failed")
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["total_seconds"] = round(time.monotonic() - began, 3)
        persist()
