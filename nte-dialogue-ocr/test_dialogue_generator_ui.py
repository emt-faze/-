import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PySide6 import QtWidgets, QtCore
from dialogue_generator_ui import GeneratorWindow
from dialogue_generation import GenerationProject
from test_dialogue_generation import Client, Runtime


class GeneratorUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = Client()
        self.runtime = Runtime(self.root, self.client)
        self.patches = [patch("dialogue_generator_ui.VoiceRuntime", return_value=self.runtime),
                        patch("dialogue_dubbing.VoiceRuntime", return_value=self.runtime)]
        for patched in self.patches:
            patched.start()
        self.window = GeneratorWindow({"mode": "system"}, root=self.root)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window.busy():
            self.window.cancel_generation()
            self.window.cancel_preview()
            self.client.release.set()
            self.wait(lambda: not self.window.busy())
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        self.app.processEvents()
        for patched in reversed(self.patches):
            patched.stop()
        self.temp.cleanup()

    def wait(self, condition, seconds=5):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(condition(), "UI operation timed out")

    def add(self, text):
        self.window.project.add_text(text)
        self.window.render()
        self.window.table.setCurrentCell(0, 2)

    def test_switching_rows_preserves_edit_and_requires_reconfirmation(self):
        self.add("伊洛伊：你好\n娜娜莉：出发吧")
        self.window.text.setPlainText("修改后的台词")
        self.window.table.setCurrentCell(1, 2)
        first = self.window.project.rows[0]
        self.assertEqual(first.text, "修改后的台词")
        self.assertFalse(first.confirmed)
        self.assertTrue(first.review_reason())
        self.assertEqual(self.window.text.toPlainText(), "出发吧")
        saved = GenerationProject.load(self.window.project.directory)
        self.assertEqual(saved.rows[0].text, "修改后的台词")

    def test_generation_controls_progress_and_stop_are_responsive(self):
        self.add("伊洛伊：你好\n娜娜莉：出发吧")
        self.client.release.clear()
        self.window.start_generation()
        self.wait(self.client.entered.is_set)
        self.assertFalse(self.window.table.isEnabled())
        self.assertFalse(self.window.mode.isEnabled())
        self.assertTrue(self.window.stop_generation.isEnabled())
        self.window.cancel_generation()
        self.client.release.set()
        self.wait(lambda: not self.window.busy())
        self.assertEqual(self.window.project.rows[0].status, "cancelled")
        self.assertEqual(self.window.project.rows[1].status, "pending")
        self.assertTrue(self.window.table.isEnabled())
        self.window.start_generation()
        self.wait(lambda: not self.window.busy())
        self.assertTrue(all(row.status == "done" for row in self.window.project.rows))
        self.assertEqual(self.window.progress.value(), 2)

    def test_current_row_preview_is_rejected_after_edit(self):
        self.add("伊洛伊：你好")
        self.window.start_generation()
        self.wait(lambda: not self.window.busy())
        self.window.text.setPlainText("新台词")
        self.window.save_editor()
        self.window.start_preview()
        self.assertIsNone(self.window.preview_worker)
        self.assertIn("请先生成", self.window.status.text())

    def test_per_row_voice_and_bulk_voice_survive_save(self):
        self.add("伊洛伊：你好\n娜娜莉：出发吧")
        self.window.voice.setCurrentIndex(self.window.voice.findData("system:teen_female"))
        self.window.save_editor()
        self.assertEqual(self.window.project.rows[0].voice, "system:teen_female")
        self.window.batch_voice.setCurrentIndex(self.window.batch_voice.findData("system:senior_male"))
        self.window.apply_voice()
        loaded = GenerationProject.load(self.window.project.directory)
        self.assertTrue(all(row.voice == "system:senior_male" for row in loaded.rows))

    def test_duplicate_generator_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "已打开"):
            GeneratorWindow({"mode": "system"}, root=self.root)

    def test_close_cancels_and_waits_for_worker_before_releasing_lock(self):
        self.add("伊洛伊：你好")
        self.client.release.clear()
        self.window.start_generation()
        self.wait(self.client.entered.is_set)
        self.window.close()
        self.assertTrue(self.window.closing)
        self.assertTrue(self.window.isVisible())
        self.client.release.set()
        self.wait(lambda: not self.window.isVisible())
        self.assertFalse(self.window.busy())
        self.assertFalse(list((self.window.project.directory / "audio").glob("*.wav")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
