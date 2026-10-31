import json
from pathlib import Path
import uuid
import unittest
from unittest.mock import patch

import numpy as np

import app
from capture_windows import list_game_windows


class IntegrationTests(unittest.TestCase):
    def test_ocr_tool_is_not_mistaken_for_the_game(self):
        titles = {1: "异环 · 对话文字识别", 2: "异环", 3: "NTEGame"}
        def enumerate_windows(callback, data):
            for hwnd in titles:
                callback(hwnd, data)
        with patch("capture_windows.win32gui.EnumWindows", side_effect=enumerate_windows), \
             patch("capture_windows.win32gui.GetWindowText", side_effect=lambda hwnd: titles[hwnd]), \
             patch("capture_windows.win32gui.IsWindowVisible", return_value=True), \
             patch("capture_windows.win32gui.IsIconic", return_value=False), \
             patch("capture_windows.win32gui.GetClientRect", return_value=(0, 0, 1920, 1080)), \
             patch("capture_windows._window_process", side_effect=lambda hwnd: (hwnd * 100, r"D:\HTGame.exe" if hwnd == 2 else r"D:\NTEGame.exe")), \
             patch("capture_windows.win32gui.GetClassName", side_effect=lambda hwnd: {1: "Qt6112QWindowIcon", 2: "UnrealWindow", 3: "Qt51517QWindowOwnDC"}[hwnd]):
            result = list_game_windows()
        self.assertEqual([window.hwnd for window in result], [2, 3])
        self.assertTrue(result[1].is_launcher)

    def test_changed_aspect_ratio_requires_recalibration(self):
        app.check_geometry(np.zeros((720, 1280, 3), np.uint8), {"reference_size": [1920, 1080]})
        with self.assertRaises(RuntimeError):
            app.check_geometry(np.zeros((800, 1280, 3), np.uint8), {"reference_size": [1920, 1080]})



if __name__ == "__main__":
    unittest.main(verbosity=2)
