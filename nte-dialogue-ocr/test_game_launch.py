from pathlib import Path
from contextlib import ExitStack
import threading
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

import game_launch as launch
from capture_windows import GameWindow


def detected(text="开始游戏", score=.99):
    return [[[80, 50], [150, 50], [150, 75], [80, 75]], text, score]


class Engine:
    def __init__(self, detections):
        self.detections = detections
    def __call__(self, frame, **kwargs):
        return self.detections, None


class FocusActivationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.window = GameWindow(900, "异环", "UnrealWindow", 2560, 1600,
                                 pid=700, process_path=r"D:\NTE\HTGame.exe")
        self.stop = threading.Event()
        self.identity = self.stack.enter_context(patch.object(launch, "_assert_window_identity"))
        self.stack.enter_context(patch.object(launch, "restore_window"))
        self.foreground = self.stack.enter_context(patch.object(launch.win32gui, "GetForegroundWindow", return_value=777))
        self.stack.enter_context(patch.object(launch.win32gui, "BringWindowToTop"))
        self.stack.enter_context(patch.object(launch.win32gui, "SetForegroundWindow"))
        self.wait = self.stack.enter_context(patch.object(self.stop, "wait", return_value=False))
        self.helper = Mock()
        self.context = Mock(Process=Mock(return_value=self.helper))
        self.get_context = self.stack.enter_context(patch("multiprocessing.get_context", return_value=self.context))

    def test_already_foreground_never_spawns_focus_helper(self):
        self.foreground.return_value = self.window.hwnd
        launch.activate_game(self.window, self.stop)
        self.get_context.assert_not_called()
        self.wait.assert_not_called()
        self.identity.assert_called_once_with(self.window)

    def test_cancel_during_helper_wait_terminates_and_joins_only_owned_helper(self):
        self.helper.is_alive.return_value = True
        def cancel_helper_wait(seconds):
            if seconds == .05:
                self.stop.set()
            return self.stop.is_set()
        self.wait.side_effect = cancel_helper_wait
        with self.assertRaises(launch.LaunchCancelled):
            launch.activate_game(self.window, self.stop)
        self.context.Process.assert_called_once_with(target=launch._focus_window_worker,
                                                    args=(self.window,), daemon=True)
        self.helper.start.assert_called_once_with()
        self.helper.terminate.assert_called_once_with()
        self.helper.join.assert_called_once_with(.5)
        self.identity.assert_called_once_with(self.window)

    def test_focus_helper_timeout_is_bounded_and_does_not_imply_foreground(self):
        self.helper.is_alive.return_value = True
        # No actual sleeps: the final clock observation crosses the 2 s bound.
        with patch.object(launch.time, "monotonic", side_effect=[10.0, 10.5, 11.0, 12.1]):
            with self.assertRaisesRegex(RuntimeError, "未将游戏置于前台"):
                launch.activate_game(self.window, self.stop)
        self.helper.terminate.assert_called_once_with()
        self.helper.join.assert_called_once_with(.5)
        self.assertEqual(self.identity.call_count, 2)
        self.assertEqual([call.args[0] for call in self.wait.call_args_list], [.1, .1, .1, .05, .05])

    def test_clean_helper_exit_still_requires_actual_foreground_verification(self):
        self.helper.is_alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, "未将游戏置于前台"):
            launch.activate_game(self.window, self.stop)
        self.helper.terminate.assert_not_called()
        self.helper.join.assert_called_once_with(.5)
        self.assertEqual(self.identity.call_count, 2)

    def test_completed_helper_can_return_only_after_verified_foreground(self):
        self.helper.is_alive.return_value = False
        self.foreground.side_effect = [777, 777, 777, 777, self.window.hwnd]
        launch.activate_game(self.window, self.stop)
        self.helper.terminate.assert_not_called()
        self.helper.join.assert_called_once_with(.5)
        self.assertEqual(self.identity.call_count, 2)


class CaptureReconnectTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.window = GameWindow(900, "异环", "UnrealWindow", 320, 200,
                                 pid=700, process_path=r"D:\NTE\HTGame.exe")
        self.stop = threading.Event()
        self.now = 0.0
        self.frame = np.zeros((200, 320, 3), np.uint8)
        self.stack.enter_context(patch.object(launch, "list_game_windows", return_value=[self.window]))
        self.stack.enter_context(patch.object(launch, "running_process_names", return_value={"htgame.exe"}))
        self.stack.enter_context(patch.object(launch, "_wait_window", return_value=self.window))
        self.identity = self.stack.enter_context(patch.object(launch, "_assert_window_identity"))
        self.activate = self.stack.enter_context(patch.object(launch, "activate_game"))
        self.capture_factory = self.stack.enter_context(patch.object(launch, "GameCapture"))
        self.click = self.stack.enter_context(patch.object(launch, "click_enter_game", return_value=True))
        self.classify = self.stack.enter_context(patch("game_scene.classify_frame"))
        self.stack.enter_context(patch("rapidocr_onnxruntime.RapidOCR", return_value=Mock()))
        self.stack.enter_context(patch.object(launch.time, "monotonic", side_effect=lambda: self.now))
        self.stack.enter_context(patch.object(self.stop, "wait", side_effect=self.wait))
        self.stack.enter_context(patch.object(Path, "mkdir"))
        self.stack.enter_context(patch.object(Path, "write_text"))
        self.stack.enter_context(patch.object(launch, "_save_evidence"))
        self.stack.enter_context(patch.object(launch.cv2, "imencode", return_value=(False, None)))

    def wait(self, seconds):
        self.now += seconds
        return self.stop.is_set()

    def capture(self, outcomes):
        capture = Mock(backend_name="PrintWindow", fallback_reason="")
        capture.start.return_value = capture
        outcomes = iter(outcomes)
        def read(sequence):
            # Deliberately pass the title retry delay between fresh frames.
            # Only the preserved click gate may suppress the second title click.
            self.now += 4
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome, self.now, self.frame
        capture.read.side_effect = read
        return capture

    def test_timeout_reconnects_same_backend_and_preserves_entry_click_gate(self):
        first = self.capture([1, launch.CaptureTimeout("loading timed out")])
        replacement = self.capture([1, 2, 3])
        self.capture_factory.side_effect = [first, replacement]
        title = {"state": "title", "enter_button": {"center": [160, 165]}, "evidence": []}
        world = {"state": "in_game", "evidence": ["verified HUD"]}
        self.classify.side_effect = [title, title, world, world]
        result = launch.ensure_game({"capture_backend": "printwindow"}, stop=self.stop)
        self.assertTrue(result["report"]["success"])
        self.assertTrue(result["report"]["in_game_verified"])
        self.assertEqual(result["report"]["acceptance"], "pending")
        self.assertEqual(len(result["report"]["capture_reconnects"]), 1)
        self.capture_factory.assert_has_calls([
            call(self.window, backend="printwindow"),
            call(self.window, backend="printwindow")])
        self.assertEqual(self.capture_factory.call_count, 2)
        self.identity.assert_called_once_with(self.window)
        first.stop.assert_called_once_with()
        replacement.stop.assert_called_once_with()
        self.assertEqual(replacement.read.call_args_list, [call(0), call(1), call(2)])
        self.click.assert_called_once()
        self.assertEqual(len(result["report"]["entry_clicks"]), 1)

    def test_cancel_arriving_with_timeout_prevents_identity_check_and_reconnect(self):
        first = self.capture([])
        self.capture_factory.return_value = first
        def cancel_read(sequence):
            self.stop.set()
            raise launch.CaptureTimeout("loading timed out")
        first.read.side_effect = cancel_read
        with self.assertRaises(launch.LaunchCancelled):
            launch.ensure_game({"capture_backend": "printwindow"}, stop=self.stop)
        self.capture_factory.assert_called_once_with(self.window, backend="printwindow")
        self.identity.assert_not_called()
        first.stop.assert_called_once_with()
        self.click.assert_not_called()

    def test_cancel_during_reconnect_progress_prevents_replacement_capture(self):
        first = self.capture([launch.CaptureTimeout("loading timed out")])
        self.capture_factory.return_value = first
        def cancel_progress(message):
            if "重新连接画面" in message:
                self.stop.set()
        with self.assertRaises(launch.LaunchCancelled):
            launch.ensure_game({"capture_backend": "printwindow"}, stop=self.stop, progress=cancel_progress)
        self.capture_factory.assert_called_once_with(self.window, backend="printwindow")
        self.identity.assert_called_once_with(self.window)
        self.assertTrue(first.stop.called)
        self.click.assert_not_called()

    def test_other_capture_errors_are_not_retried(self):
        first = self.capture([RuntimeError("window identity changed")])
        self.capture_factory.return_value = first
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            launch.ensure_game({"capture_backend": "printwindow"}, stop=self.stop)
        self.capture_factory.assert_called_once_with(self.window, backend="printwindow")
        first.stop.assert_called_once_with()

    def test_wgc_observes_existing_world_without_requesting_foreground(self):
        capture = self.capture([1, 2])
        self.capture_factory.return_value = capture
        self.classify.return_value = {"state": "in_game", "evidence": ["verified HUD"]}
        result = launch.ensure_game({"capture_backend": "wgc"}, stop=self.stop)
        self.assertTrue(result["report"]["in_game_verified"])
        self.activate.assert_not_called()
        self.click.assert_not_called()
        self.capture_factory.assert_called_once_with(self.window, backend="wgc", require_foreground=False, first_frame_timeout=10)

    def test_wgc_initial_frame_timeout_reconnects_then_confirms_world(self):
        first = self.capture([launch.CaptureTimeout('WGC first frame timed out')])
        replacement = self.capture([1, 2])
        self.capture_factory.side_effect = [first, replacement]
        self.classify.return_value = {"state": "in_game", "evidence": ["verified HUD"]}
        result = launch.ensure_game({"capture_backend": "wgc"}, stop=self.stop)
        self.assertTrue(result['report']['success'])
        self.assertEqual(len(result['report']['capture_reconnects']), 1)
        self.capture_factory.assert_has_calls([
            call(self.window, backend='wgc', require_foreground=False, first_frame_timeout=10),
            call(self.window, backend='wgc', require_foreground=False, first_frame_timeout=10),
        ])
        first.stop.assert_called_once_with()
        replacement.stop.assert_called_once_with()
        self.click.assert_not_called()

    def test_foreground_denial_waits_for_fresh_title_instead_of_aborting(self):
        self.capture_factory.return_value = self.capture([1, 2, 3, 4])
        title = {"state": "title", "enter_button": {"center": [160, 165]}, "evidence": []}
        world = {"state": "in_game", "evidence": ["verified HUD"]}
        self.classify.side_effect = [title, title, world, world]
        self.click.side_effect = [launch.GameFocusRequired('foreground unavailable'), True]
        result = launch.ensure_game({"capture_backend": "wgc"}, stop=self.stop)
        self.assertTrue(result["report"]["success"])
        self.assertEqual(len(result["report"]["entry_clicks"]), 1)
        self.assertIn('waiting_foreground', result['report']['milestones_seconds'])


class LauncherTests(unittest.TestCase):
    def test_restored_launcher_uses_current_client_size_for_start_click(self):
        # A minimized window is discovered using its normal outer rectangle;
        # once restored, clicks must use its smaller, current client rectangle.
        original = GameWindow(123, "NTEGame", "Qt51517QWindowOwnDC", 1200, 800,
                              pid=10, process_path=r"D:\NTEGame.exe")
        stop = threading.Event()
        with patch.object(launch, "list_game_windows", return_value=[original]), \
             patch.object(launch, "restore_window"), \
             patch.object(launch, "running_process_names", return_value=set()), \
             patch.object(launch.win32gui, "IsWindowVisible", return_value=True), \
             patch.object(launch.win32gui, "IsIconic", return_value=False), \
             patch.object(launch.win32gui, "GetClientRect", return_value=(0, 0, 1180, 760)), \
             patch.object(launch.win32gui, "PostMessage") as post:
            restored = launch._wait_window(True, stop, 1, lambda message: None, settle=0)
            self.assertEqual((restored.width, restored.height), (1180, 760))
            self.assertEqual((restored.hwnd, restored.pid, restored.process_path),
                             (original.hwnd, original.pid, original.process_path))
            self.assertTrue(launch.click_start(restored, {"center": [900, 670]}, stop))
            self.assertEqual([call.args[1] for call in post.call_args_list],
                             [launch.win32con.WM_MOUSEMOVE, launch.win32con.WM_LBUTTONDOWN,
                              launch.win32con.WM_LBUTTONUP])

    def test_cancel_during_final_launcher_validation_prevents_mouse_messages(self):
        window = GameWindow(123, "NTEGame", "Qt51517QWindowOwnDC", 1200, 800,
                            pid=10, process_path=r"D:\NTEGame.exe")
        stop = threading.Event()
        def read_size_and_cancel(hwnd):
            stop.set()  # User presses Stop after the initial cancellation check.
            return 0, 0, 1200, 800
        with patch.object(launch, "running_process_names", return_value=set()), \
             patch.object(launch, "list_game_windows", return_value=[window]), \
             patch.object(launch.win32gui, "IsIconic", return_value=False), \
             patch.object(launch.win32gui, "GetClientRect", side_effect=read_size_and_cancel), \
             patch.object(launch.win32gui, "PostMessage") as post:
            with self.assertRaises(launch.LaunchCancelled):
                launch.click_start(window, {"center": [900, 670]}, stop)
            post.assert_not_called()

    def test_cancel_during_launch_status_update_prevents_process_creation(self):
        stop = threading.Event()
        def cancel_on_launch_message(message):
            if "正在打开官方启动器" in message:
                stop.set()
        with patch.object(launch, "is_admin", return_value=True), \
             patch.object(launch, "running_process_names", return_value=set()), \
             patch.object(launch, "list_game_windows", return_value=[]), \
             patch.object(launch, "find_launcher_path", return_value=r"D:\NTEGame.exe"), \
             patch.object(launch.subprocess, "Popen") as popen, \
             patch.object(Path, "mkdir"), patch.object(Path, "write_text"):
            with self.assertRaises(launch.LaunchCancelled):
                launch.ensure_game({}, stop=stop, progress=cancel_on_launch_message)
            popen.assert_not_called()

    def test_entry_retries_require_three_seconds_and_stop_at_three(self):
        gate = launch.EntryClickGate()
        self.assertTrue(gate.observe("title", 0))
        gate.clicked(0)
        self.assertFalse(gate.observe("title", 2.99))
        self.assertTrue(gate.observe("title", 3))
        gate.clicked(3)
        self.assertTrue(gate.observe("title", 6))
        gate.clicked(6)
        self.assertFalse(gate.observe("title", 9))

    def test_entry_disappearing_stops_retries_even_if_title_returns(self):
        gate = launch.EntryClickGate()
        gate.clicked(0)
        self.assertFalse(gate.observe("unknown", .4))
        self.assertFalse(gate.observe("title", 4))

    def test_loading_before_title_does_not_consume_click_allowance(self):
        gate = launch.EntryClickGate()
        self.assertFalse(gate.observe("unknown", 0))
        self.assertTrue(gate.observe("title", 30))

    def test_cancelled_entry_does_not_activate_or_click_game(self):
        stop = threading.Event()
        stop.set()
        with patch.object(launch, "activate_game") as activate, self.assertRaises(launch.LaunchCancelled):
            launch.click_enter_game(None, {}, (2560, 1600), stop)
        activate.assert_not_called()

    def test_only_exact_enabled_start_label_is_clicked(self):
        frame = np.full((600, 1000, 3), 220, np.uint8)
        result = launch.locate_start_button(frame, Engine([detected()]))
        self.assertEqual(result["center"], [765, 494])
        for text in ("立即充值", "开始", "开始游戏攻略", "修复游戏", "下载游戏"):
            self.assertIsNone(launch.locate_start_button(frame, Engine([detected(text)])))
        self.assertIsNone(launch.locate_start_button(frame, Engine([detected(score=.75)])))
        self.assertIsNone(launch.locate_start_button(np.zeros_like(frame), Engine([detected()])))

    def test_updates_and_ambiguous_buttons_do_not_trigger_click(self):
        frame = np.full((600, 1000, 3), 220, np.uint8)
        with self.assertRaisesRegex(RuntimeError, "更新"):
            launch.locate_start_button(frame, Engine([detected("更新游戏")]))
        with self.assertRaisesRegex(RuntimeError, "多个"):
            launch.locate_start_button(frame, Engine([detected(), detected()]))

    def test_stale_window_identity_does_not_receive_click(self):
        original = GameWindow(123, "NTEGame", "Qt51517QWindowOwnDC", 1920, 1080,
                              pid=10, process_path=r"D:\NTEGame.exe")
        changed = GameWindow(123, "NTEGame", "Qt51517QWindowOwnDC", 1920, 1080,
                             pid=11, process_path=r"D:\NTEGame.exe")
        with patch.object(launch, "list_game_windows", return_value=[changed]), \
             patch.object(launch, "running_process_names", return_value=set()), \
             patch.object(launch.win32gui, "PostMessage") as post:
            with self.assertRaisesRegex(RuntimeError, "变化"):
                launch.click_start(original, {"center": [1600, 950]})
            post.assert_not_called()

    def test_cancelled_wait_does_not_open_game(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(launch.LaunchCancelled):
            launch._wait_window(False, event, 10, lambda text: None)

    def test_missing_admin_stops_before_launch_or_click(self):
        with patch.object(launch, "is_admin", return_value=False), \
             patch.object(launch, "running_process_names", return_value=set()), \
             patch.object(launch, "list_game_windows", return_value=[]), \
             patch.object(launch.subprocess, "Popen") as popen, \
             patch.object(Path, "write_text"):
            with self.assertRaisesRegex(RuntimeError, "管理员"):
                launch.ensure_game({})
            popen.assert_not_called()

    def test_loading_game_process_prevents_another_launcher_start(self):
        with patch.object(launch, "running_process_names", return_value={"htgame.exe"}), \
             patch.object(launch, "list_game_windows", return_value=[]), \
             patch.object(launch, "_wait_window", side_effect=RuntimeError("waiting for game window")) as wait, \
             patch.object(launch.subprocess, "Popen") as popen, \
             patch.object(Path, "write_text"):
            with self.assertRaisesRegex(RuntimeError, "waiting for game"):
                launch.ensure_game({})
            self.assertFalse(wait.call_args.args[0])
            popen.assert_not_called()

    def test_game_started_between_detection_and_click_prevents_click(self):
        with patch.object(launch, "running_process_names", return_value={"htgame.exe"}), \
             patch.object(launch.win32gui, "PostMessage") as post:
            self.assertFalse(launch.click_start(None, {}))
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
