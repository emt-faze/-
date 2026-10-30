import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import capture_session as capture
from capture_windows import GameWindow


class CaptureSessionTests(unittest.TestCase):
    window = GameWindow(123, "异环", "UnrealWindow", 200, 100, 55, r"D:\HTGame.exe")
    geometry = {"client_size": (200, 100), "origin": (20, 30), "screen_rect": (20, 30, 180, 90), "occluders": []}

    def test_region_union_and_negative_monitor_coordinates(self):
        self.assertEqual(capture.region_bounds((1000, 500), [[.1,.6,.9,.9],[.1,.5,.3,.6]]), (100,250,900,450))
        self.assertTrue(capture.intersects((-100, 0, 100, 100), (-20, 20, 20, 30)))
        self.assertFalse(capture.intersects((0, 0, 10, 10), (10, 0, 20, 10)))

    def test_mss_instance_created_and_closed_in_one_thread_and_latest_only(self):
        owner = []
        class FakeMSS:
            monitors = [{"left": -300, "top": 0, "width": 500, "height": 200}]
            def __init__(self):
                owner.append(("create", threading.get_ident()))
            def grab(self, rectangle):
                owner.append(("grab", threading.get_ident()))
                image = np.zeros((rectangle["height"], rectangle["width"], 4), np.uint8)
                image[:20] = len(owner)*3
                return image
            def close(self):
                owner.append(("close", threading.get_ident()))
        with patch("mss.mss", FakeMSS), patch.object(capture, "enable_dpi_awareness"), \
             patch.object(capture, "visible_region", return_value=("ready", self.geometry)):
            session = capture.CaptureSession(self.window, interval_ms=20, backend="mss").start()
            try:
                deadline = time.monotonic()+1
                while session.metrics()["captured_frames"] < 3 and time.monotonic() < deadline:
                    time.sleep(.01)
                packet = session.read()
                self.assertGreaterEqual(packet.sequence, 3)
                self.assertEqual(packet.origin, (20,30))
                self.assertEqual(packet.client_size, (200,100))
                self.assertEqual(packet.image.shape, (60,160,3))
                self.assertFalse(packet.image.flags.writeable)
                self.assertIsNone(session.read(packet.sequence))
            finally:
                session.stop()
        self.assertEqual(sum(x[0]=="create" for x in owner), 1)
        self.assertEqual(len({x[1] for x in owner}), 1)
        self.assertNotEqual(owner[0][1], threading.get_ident())
        self.assertEqual(owner[-1][0], "close")

    def test_focus_loss_invalidates_saved_frame_and_mss_failure_never_falls_back(self):
        session = capture.CaptureSession(self.window, backend="mss")
        frame = np.ones((10,10,3), np.uint8)
        session._latest = capture.FramePacket(1, time.monotonic(), frame, (200,100), (0,0), (0,0,10,10), 1)
        session._state = "ready"
        with patch.object(capture, "visible_region", return_value=("not_foreground", None)):
            self.assertFalse(session.ready_for_ocr())
            self.assertIsNone(session.peek())
        with patch("mss.mss", side_effect=RuntimeError("unavailable")), \
             patch.object(capture, "LegacyCapture") as legacy, patch.object(capture, "enable_dpi_awareness"):
            broken = capture.CaptureSession(self.window, backend="mss").start()
            broken._thread.join(1)
            with self.assertRaisesRegex(RuntimeError, "手动"):
                broken.read()
            legacy.assert_not_called()
            broken.stop()

    def test_minimized_foreground_and_occlusion_guards(self):
        def enum(callback, data):
            for handle in [789,123]:
                if callback(handle,data) is False:
                    break
        with patch.object(capture, "_assert_window_identity"), \
             patch.object(capture.win32gui, "IsWindowVisible", return_value=True), \
             patch.object(capture.win32gui, "IsIconic", return_value=False) as iconic, \
             patch.object(capture.win32gui, "GetForegroundWindow", return_value=123) as foreground, \
             patch.object(capture.win32gui, "GetClientRect", return_value=(0,0,200,100)), \
             patch.object(capture.win32gui, "ClientToScreen", return_value=(-200,0)), \
             patch.object(capture.win32gui, "GetWindowRect", return_value=(-160,40,-100,90)), \
             patch.object(capture.win32gui, "GetWindowLong", return_value=0), \
             patch.object(capture.win32gui, "EnumWindows", side_effect=enum), \
             patch.object(capture, "_point_occluders", return_value=[]), \
             patch.object(capture, "_cloaked", return_value=False):
            self.assertEqual(capture.visible_region(self.window)[0], "occluded")
            self.assertEqual(capture.visible_region(self.window, check_occlusion=False)[0], "ready")
            foreground.return_value = 789
            self.assertEqual(capture.visible_region(self.window)[0], "not_foreground")
            self.assertEqual(capture.visible_region(self.window, check_occlusion=False,
                                                    require_foreground=False)[0], "ready")
            iconic.return_value = True
            self.assertEqual(capture.visible_region(self.window)[0], "minimized")

    def test_moving_window_invalidates_pre_move_packet_before_next_capture_tick(self):
        session = capture.CaptureSession(self.window)
        session._geometry = self.geometry
        session._state = 'ready'
        session._latest = capture.FramePacket(1,time.monotonic(),np.ones((60,160,3),np.uint8),
                                              (200,100),(20,30),(20,30,180,90),1)
        moved = dict(self.geometry, screen_rect=(25,30,185,90))
        with patch.object(capture,'visible_region',return_value=('ready',moved)):
            self.assertFalse(session.ready_for_ocr())
            self.assertIsNone(session.peek())


if __name__ == "__main__":
    unittest.main(verbosity=2)
