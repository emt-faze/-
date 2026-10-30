"""Window identity, backend fallbacks and bounded capture; no game interaction."""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import sys

import numpy as np

import capture_windows as capture


WINDOW = capture.GameWindow(101, "异环", "UnrealWindow", 1280, 720, 42, r"D:\NTE\HTGame.exe")


def frame():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[10:20, 10:60] = (50, 150, 250)
    return image


class WindowIdentityTests(unittest.TestCase):
    def test_launcher_requires_real_executable_and_main_window_class(self):
        for name in ("NTEGame.exe", "NTEGlobalGame.exe", "NTETWGame.exe"):
            window = capture.GameWindow(1, "anything", "Qt51517QWindowOwnDC", 1000, 600, 1, "D:\\" + name)
            self.assertTrue(window.is_launcher)
            self.assertFalse(replace(window, class_name="Qt51517QWindowToolSaveBitsOwnDC").is_launcher)
        self.assertFalse(capture.GameWindow(1, "NTEGame", "Qt51517QWindowOwnDC", 1000, 600).is_launcher)

    def test_listing_uses_executable_and_class_not_title_and_excludes_small_hidden(self):
        rows = {
            1: (True, "UnrealWindow", (1280, 720), r"D:\NTE\HTGame.exe", "different localized title"),
            2: (True, "Qt51517QWindowOwnDC", (1000, 600), r"D:\NTEGame.exe", "NTEGame"),
            3: (True, "Qt51517QWindowOwnDC", (2, 2), r"D:\NTEGame.exe", "NTEGame"),
            4: (False, "Qt51517QWindowOwnDC", (1000, 600), r"D:\NTEGame.exe", "NTEGame"),
            5: (True, "UnrealWindow", (1280, 720), r"D:\another.exe", "异环"),
            6: (True, "Qt51517QWindowOwnDC", (1280, 720), r"D:\pythonw.exe", "异环 · 对话文字识别"),
            7: (True, "Qt51517QWindowToolSaveBitsOwnDC", (1000, 600), r"D:\NTEGame.exe", "NTEGame"),
            8: (False, "UnrealWindow", (1280, 720), r"D:\NTE\HTGame.exe", "异环"),
            9: (False, "Qt51517QWindowOwnDC", (1000, 600), r"D:\another.exe", "NTEGame"),
            10: (False, "Qt51517QWindowOwnDC", (2, 2), r"D:\NTEGame.exe", "NTEGame"),
            11: (False, "Qt51517QWindowOwnDCIcon", (1000, 600), r"D:\NTEGame.exe", "NTEGame"),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(capture.win32gui, "EnumWindows", side_effect=lambda visit, arg: [visit(key, arg) for key in rows]))
            stack.enter_context(patch.object(capture.win32gui, "IsWindowVisible", side_effect=lambda hwnd: rows[hwnd][0]))
            stack.enter_context(patch.object(capture.win32gui, "IsIconic", return_value=False))
            stack.enter_context(patch.object(capture.win32gui, "GetClassName", side_effect=lambda hwnd: rows[hwnd][1]))
            stack.enter_context(patch.object(capture.win32gui, "GetClientRect", side_effect=lambda hwnd: (0, 0, *rows[hwnd][2])))
            stack.enter_context(patch.object(capture, "_window_process", side_effect=lambda hwnd: (hwnd * 10, rows[hwnd][3])))
            stack.enter_context(patch.object(capture.win32gui, "GetWindowText", side_effect=lambda hwnd: rows[hwnd][4]))
            self.assertEqual([item.hwnd for item in capture.list_game_windows()], [1, 2])
            self.assertEqual([item.hwnd for item in capture.list_game_windows(include_hidden_launchers=True)], [1, 2, 4])

    def test_tray_hidden_main_launcher_is_opt_in_without_exposing_hidden_game(self):
        with patch.object(capture.win32gui, "EnumWindows", side_effect=lambda visit, arg: [visit(1, arg), visit(2, arg)]), \
             patch.object(capture.win32gui, "IsWindowVisible", return_value=False), \
             patch.object(capture.win32gui, "IsIconic", return_value=False), \
             patch.object(capture.win32gui, "GetClassName", side_effect=lambda hwnd: "Qt51517QWindowOwnDC" if hwnd == 1 else "UnrealWindow"), \
             patch.object(capture.win32gui, "GetClientRect", return_value=(0, 0, 1920, 1080)), \
             patch.object(capture.win32gui, "GetWindowText", side_effect=lambda hwnd: "NTEGame" if hwnd == 1 else "异环"), \
             patch.object(capture, "_window_process", side_effect=lambda hwnd: (38312 if hwnd == 1 else 40000, r"D:\NTEGame.exe" if hwnd == 1 else r"D:\HTGame.exe")):
            self.assertEqual(capture.list_game_windows(), [])
            windows = capture.list_game_windows(include_hidden_launchers=True)
        self.assertEqual([window.hwnd for window in windows], [1])
        self.assertTrue(windows[0].is_launcher)
        self.assertEqual(windows[0].pid, 38312)

    def test_minimized_launcher_uses_restore_size_and_remains_discoverable(self):
        with patch.object(capture.win32gui, "EnumWindows", side_effect=lambda visit, arg: visit(1, arg)), \
             patch.object(capture.win32gui, "IsWindowVisible", return_value=True), \
             patch.object(capture.win32gui, "IsIconic", return_value=True), \
             patch.object(capture.win32gui, "GetClassName", return_value="Qt51517QWindowOwnDC"), \
             patch.object(capture.win32gui, "GetClientRect", return_value=(0, 0, 237, 39)), \
             patch.object(capture.win32gui, "GetWindowPlacement", return_value=(0, 2, (0, 0), (0, 0), (100, 100, 1300, 900))), \
             patch.object(capture.win32gui, "GetWindowText", return_value="NTEGame"), \
             patch.object(capture, "_window_process", return_value=(10, r"D:\NTEGame.exe")):
            result = capture.list_game_windows()
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].is_launcher)
        self.assertEqual((result[0].width, result[0].height), (1200, 800))

    def test_handle_reuse_pid_path_or_class_change_is_rejected(self):
        for pid, path, cls in ((43, WINDOW.process_path, WINDOW.class_name),
                               (42, r"D:\other\HTGame.exe", WINDOW.class_name),
                               (42, WINDOW.process_path, "OtherClass")):
            with self.subTest(pid=pid, path=path, cls=cls), \
                 patch.object(capture.win32gui, "IsWindow", return_value=True), \
                 patch.object(capture, "_window_process", return_value=(pid, path)), \
                 patch.object(capture.win32gui, "GetClassName", return_value=cls):
                with self.assertRaisesRegex(RuntimeError, "进程或类型已变化"):
                    capture._assert_window_identity(WINDOW)

    def test_same_title_visible_windows_are_ambiguous_for_wgc(self):
        with patch.object(capture.win32gui, "EnumWindows", side_effect=lambda visit, arg: [visit(hwnd, arg) for hwnd in [101, 102]]), \
             patch.object(capture.win32gui, "IsWindowVisible", return_value=True), \
             patch.object(capture.win32gui, "GetWindowText", return_value="异环"):
            self.assertFalse(capture._wgc_title_is_unique(WINDOW))

    def test_substring_title_is_also_ambiguous_for_wgc(self):
        with patch.object(capture.win32gui, "EnumWindows", side_effect=lambda visit, arg: [visit(hwnd, arg) for hwnd in [101, 102]]), \
             patch.object(capture.win32gui, "IsWindowVisible", return_value=True), \
             patch.object(capture.win32gui, "GetWindowText", side_effect=lambda hwnd: "异环" if hwnd == 101 else "异环攻略 - 浏览器"):
            self.assertFalse(capture._wgc_title_is_unique(WINDOW))


class PixelTests(unittest.TestCase):
    def test_native_duplicate_timestamp_is_ignored_and_restart_resets_timestamp(self):
        class NativeCapture:
            def __init__(self,**kwargs):
                pass
            def event(self,fn):
                setattr(self,fn.__name__,fn)
                return fn
            def start_free_threaded(self):
                return Mock()
        instance=capture.GameCapture(WINDOW,backend='wgc')
        native_frame=SimpleNamespace(timespan=123,frame_buffer=frame())
        with patch.dict(sys.modules,{'windows_capture':SimpleNamespace(WindowsCapture=NativeCapture)}), \
             patch.object(capture,'_wgc_title_is_unique',return_value=True), \
             patch.object(capture,'_assert_window_identity'), \
             patch.object(capture.win32gui,'IsIconic',return_value=False), \
             patch.object(capture.win32gui,'IsWindowVisible',return_value=True), \
             patch.object(capture,'client_image',return_value=frame()):
            instance._start_wgc()
            instance.capture.on_frame_arrived(native_frame,Mock())
            instance.capture.on_frame_arrived(native_frame,Mock())
            self.assertEqual(instance.sequence,1)
            instance._stop_wgc()
            instance._start_wgc()
            instance.capture.on_frame_arrived(native_frame,Mock())
            self.assertEqual(instance.sequence,2)
            instance._stop_wgc()

    def test_bgra_conversion_keeps_channel_order_and_owns_memory(self):
        native = np.zeros((40, 80, 4), dtype=np.uint8)
        native[:] = (10, 20, 30, 255)
        with patch.object(capture.win32gui, "GetClientRect", return_value=(0, 0, 80, 40)):
            result = capture.client_image(native, 1)
        self.assertEqual(result[0, 0].tolist(), [10, 20, 30])
        native[:] = 0
        self.assertEqual(result[0, 0].tolist(), [10, 20, 30])
        self.assertTrue(result.flags.c_contiguous)

    def test_nonclient_borders_are_removed_by_client_origin(self):
        native = np.arange(60 * 100 * 4, dtype=np.uint8).reshape(60, 100, 4)
        with patch.object(capture.win32gui, "GetClientRect", return_value=(0, 0, 80, 30)), \
             patch.object(capture.win32gui, "ClientToScreen", return_value=(110, 220)), \
             patch.object(capture.win32gui, "GetWindowRect", return_value=(100, 200, 200, 260)), \
             patch.object(capture.ctypes.windll.dwmapi, "DwmGetWindowAttribute", return_value=1):
            result = capture.client_image(native, 1)
        np.testing.assert_array_equal(result, native[20:50, 10:90, :3])

    def test_empty_black_and_flat_frames_are_not_valid(self):
        self.assertFalse(capture._usable_frame(None))
        self.assertFalse(capture._usable_frame(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertFalse(capture._usable_frame(np.zeros((40, 80, 3), dtype=np.uint8)))
        self.assertFalse(capture._usable_frame(np.full((40, 80, 3), 255, dtype=np.uint8)))
        self.assertTrue(capture._usable_frame(frame()))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(capture, "_assert_window_identity"))
        self.stack.enter_context(patch.object(capture, "_bind_window", return_value=WINDOW))
        self.minimized = self.stack.enter_context(patch.object(capture.win32gui, "IsIconic", return_value=False))
        self.stack.enter_context(patch.object(capture.win32gui, "IsWindowVisible", return_value=True))
        self.clock = self.stack.enter_context(patch.object(capture.time, "monotonic", return_value=10.0))

    def running_wgc(self, backend="auto"):
        instance = capture.GameCapture(WINDOW, backend=backend)
        instance._started = True
        instance._backend_name = "wgc"
        instance._backend_started = 10.0
        return instance

    def test_wgc_start_failure_uses_window_only_fallback(self):
        instance = capture.GameCapture(WINDOW)
        fake = Mock()
        self.stack.enter_context(patch.object(capture, "_PrintWindowBackend", return_value=Mock(start=Mock(return_value=fake))))
        with patch.object(instance, "_start_wgc", side_effect=RuntimeError("ambiguous title")):
            instance.start()
        self.assertEqual(instance.backend_name, "printwindow")
        self.assertEqual(instance.fallback_reason, "ambiguous title")

    def test_first_frame_timeout_falls_back_and_forced_wgc_reports_error(self):
        instance = self.running_wgc()
        fake = Mock(poll=Mock(return_value=None))
        self.stack.enter_context(patch.object(capture, "_PrintWindowBackend", return_value=Mock(start=Mock(return_value=fake))))
        self.clock.return_value = 12.6
        self.assertIsNone(instance.read())
        self.assertEqual(instance.backend_name, "printwindow")
        forced = self.running_wgc("wgc")
        with self.assertRaisesRegex(RuntimeError, "等待时限"):
            forced.read()

    def test_minimize_clears_pending_pixels_and_stops_print_worker(self):
        instance = self.running_wgc()
        instance._backend_name = "printwindow"
        worker = instance._print_backend = Mock()
        instance._accept_frame(frame(), 10)
        self.minimized.return_value = True
        self.assertIsNone(instance.read())
        self.assertIsNone(instance.latest)
        self.assertIsNone(instance._print_backend)
        worker.stop.assert_called_once()

    def test_black_frame_clears_cached_previous_frame(self):
        instance = self.running_wgc()
        self.assertTrue(instance._accept_frame(frame(), 10))
        self.assertFalse(instance._accept_frame(np.zeros((40, 80, 3), dtype=np.uint8), 10.1))
        self.assertIsNone(instance.read())

    def test_sequence_and_age_prevent_reusing_frames(self):
        instance = self.running_wgc()
        instance._accept_frame(frame(), 10)
        self.assertEqual(instance.read()[0], 1)
        self.assertIsNone(instance.read(after_sequence=1))
        self.clock.return_value = 12.6
        self.assertIsNone(instance.read())

    def test_brief_geometry_transition_recovers_but_persistent_mismatch_errors(self):
        instance = self.running_wgc()
        instance._geometry_failed("尺寸不一致")
        self.clock.return_value = 10.3
        self.assertIsNone(instance.read())
        instance._accept_frame(frame(), 10.3)
        self.assertIsNotNone(instance.read())
        instance._geometry_failed("尺寸不一致")
        self.clock.return_value = 11.9
        with self.assertRaisesRegex(RuntimeError, "尺寸不一致"):
            instance.read()

    def test_printwindow_stalled_request_is_bounded_and_worker_is_stopped(self):
        backend = capture._PrintWindowBackend(WINDOW, .2, request_timeout=2.5)
        backend.connection = Mock(poll=Mock(return_value=False))
        backend.process = Mock()
        backend.pending_since = 1
        with patch.object(backend, "stop") as stop:
            with self.assertRaisesRegex(RuntimeError, "等待时限"):
                backend.poll(3.6)
        stop.assert_called_once()

    def test_printwindow_copies_shared_pixels_before_requesting_next_frame(self):
        backend = capture._PrintWindowBackend(WINDOW, .2)
        pixels = bytearray(frame().tobytes())
        backend.memory = Mock(buf=pixels, size=len(pixels))
        backend.connection = Mock(poll=Mock(return_value=True), recv=Mock(return_value=("frame", 10, (40, 80, 3))))
        backend.process = Mock(is_alive=Mock(return_value=True))
        backend.pending_since = 9.9
        result = backend.poll(10)
        pixels[:] = bytes(len(pixels))
        self.assertTrue(result[1].any())
        backend.connection.send.assert_called_once_with("capture")

    def test_printwindow_termination_waits_are_bounded(self):
        backend = capture._PrintWindowBackend(WINDOW, .2)
        worker = backend.process = Mock(pid=88, is_alive=Mock(side_effect=[True, True, False, False]))
        backend.connection = Mock()
        backend.memory = Mock()
        backend.stop()
        worker.terminate.assert_called_once()
        for call in worker.join.call_args_list:
            self.assertLessEqual(call.args[0], .3)
        self.assertIsNone(backend.process)

    def test_snapshot_passes_backend_and_always_closes(self):
        fake = Mock(read=Mock(return_value=(1, 10, frame())))
        factory = self.stack.enter_context(patch.object(capture, "GameCapture", return_value=Mock(start=Mock(return_value=fake))))
        result = capture.snapshot(WINDOW, backend="printwindow")
        self.assertEqual(result.shape, (40, 80, 3))
        factory.assert_called_once_with(WINDOW, backend="printwindow")
        fake.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
