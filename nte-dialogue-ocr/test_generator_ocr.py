import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock

from PySide6 import QtCore, QtWidgets
from generator_ocr import CollectionWorker, OcrCollectionDialog, save_ocr_settings
from dialogue_generation import GenerationProject
from test_dialogue_generation import Runtime, Client
from dialogue_generator_ui import GeneratorWindow
from capture_windows import GameWindow


class OcrIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(calibrated=True, body_roi=[.1,.8,.9,.9], capture_backend="wgc",
                           dubbing={"enabled":True,"mode":"system","system_default_voice":"teen_male"},
                           preserved_setting="keep me")
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.session = self.root / "records/2026-10-31/session"
        self.session.mkdir(parents=True)
        self.game = GameWindow(123, "异环", "UnrealWindow", 1920, 1080, 55, r"D:\HTGame.exe")

    def tearDown(self):
        self.app.processEvents()
        self.temp.cleanup()

    def record(self, **values):
        return dict({"dialogue_id":"session-1", "revision":1, "body":"主唱大人准备好了吗？",
                     "speaker":"伊洛伊", "status":"candidate", "speaker_status":"verified",
                     "body_confidence":.99}, **values)

    def wait(self, condition):
        end = time.monotonic()+5
        while not condition() and time.monotonic()<end:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(condition())

    def test_collection_calls_existing_ocr_and_disables_speech_only_in_copy(self):
        events = []
        def runner(window, config, stop, callback, **kwargs):
            self.assertIs(window, self.game)
            self.assertFalse(config["dubbing"]["enabled"])
            self.assertEqual(kwargs["records_root"], self.root)
            kwargs["on_session"](self.session)
            event = self.record()
            kwargs["on_record"](event)
            event["body"] = "mutated later"
            callback({"event":event})
            callback({"status_text":"已复核"})
            (self.session / "dialogue.jsonl").write_text(json.dumps(self.record(revision=2,status="verified")),encoding="utf-8")
        worker = CollectionWorker(self.game, self.config, self.root, runner=runner)
        worker.message.connect(events.append)
        worker.run()
        self.assertTrue(self.config["dubbing"]["enabled"])
        self.assertEqual([e["record"]["body"] for e in events if "record" in e], ["主唱大人准备好了吗？"])
        self.assertEqual(events[-1]["final_file"], str(self.session / "dialogue.jsonl"))

    def test_stop_flush_file_is_delivered_even_after_worker_error(self):
        events = []
        def runner(window, config, stop, callback, **kwargs):
            kwargs["on_session"](self.session)
            (self.session / "dialogue.jsonl").write_text(json.dumps(self.record()),encoding="utf-8")
            raise RuntimeError("capture lost")
        worker = CollectionWorker(self.game, self.config, self.root, runner=runner)
        worker.message.connect(events.append)
        worker.run()
        self.assertTrue(any("error" in e for e in events))
        self.assertIn("final_file", events[-1])

    def test_live_revision_and_later_file_import_share_one_project_row(self):
        project = GenerationProject.create(self.root / "projects")
        project.receive_record(self.record(), self.session)
        self.assertTrue(project.rows[0].review_reason())
        project.receive_record(self.record(revision=2,status="verified"), self.session)
        self.assertFalse(project.rows[0].review_reason())
        path = self.session / "dialogue.jsonl"
        path.write_text(json.dumps(self.record(revision=2,status="verified")),encoding="utf-8")
        result = project.import_records(path)
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(len(project.rows), 1)

    def test_manual_correction_survives_new_live_revision(self):
        project = GenerationProject.create(self.root / "projects")
        project.receive_record(self.record(), self.session)
        project.rows[0].edit(text="人工核对文字")
        project.rows[0].set_confirmed(True)
        project.receive_record(self.record(revision=2,status="verified"), self.session)
        self.assertEqual(project.rows[0].text, "人工核对文字")
        self.assertTrue(project.rows[0].review_reason())

    def test_malformed_live_record_rejected(self):
        project = GenerationProject.create(self.root / "projects")
        for changes in ({"body":None}, {"revision":-1}, {"speaker":[]}, {"dialogue_id":""}):
            with self.assertRaises(ValueError):
                project.receive_record(self.record(**changes), self.session)
        self.assertEqual(project.rows, [])

    def test_capture_lease_prevents_duplicate_ocr_workers(self):
        from app import run_loop
        lease = QtCore.QLockFile(str(self.root / "capture.lock"))
        self.assertTrue(lease.tryLock(0))
        with patch("app._run_loop") as loop, self.assertRaisesRegex(RuntimeError, "另一个窗口"):
            run_loop(self.game, self.config, threading.Event(), lambda value:None, records_root=self.root)
        loop.assert_not_called()
        lease.unlock()
        with patch("app._run_loop") as loop:
            run_loop(self.game, self.config, threading.Event(), lambda value:None, records_root=self.root)
            loop.assert_called_once()

    def test_calibration_preserves_voice_and_other_settings(self):
        saved = save_ocr_settings(self.root, {"speaker_roi":[.2,.3,.4,.5]})
        self.assertEqual(saved["dubbing"], self.config["dubbing"])
        self.assertEqual(saved["preserved_setting"], "keep me")
        with self.assertRaises(ValueError):
            save_ocr_settings(self.root, {"dubbing":{}})

    def test_generator_receives_and_persists_records_without_generating(self):
        runtime = Runtime(self.root, Client())
        with patch("dialogue_generator_ui.VoiceRuntime", return_value=runtime):
            window = GeneratorWindow({"mode":"system"}, root=self.root)
            try:
                window.receive_ocr({"source":str(self.session),"record":self.record()})
                window.receive_ocr({"source":str(self.session),"record":self.record(revision=2,status="verified")})
                saved = GenerationProject.load(window.project.directory)
                self.assertEqual(len(saved.rows), 1)
                self.assertEqual(saved.rows[0].revision, 2)
                self.assertEqual(saved.rows[0].status, "pending")
                self.assertEqual(runtime.client.calls, [])
                window.receive_ocr({"source":str(self.root.parent),"record":self.record(dialogue_id="outside")})
                self.assertEqual(len(window.project.rows), 1)
            finally:
                window.close()

    def test_cjk_spacing_keeps_review_label_but_changed_words_do_not(self):
        with patch("capture_windows.list_game_windows", return_value=[self.game]):
            dialog = OcrCollectionDialog(self.root)
        try:
            text = '以前……「平行线」排练结束后。'
            record = self.record(body=text,status='verified')
            dialog.receive({'record':record,'source':str(self.session)})
            dialog.receive({'observation':dict(body='以前…… 「平行线」排练结束后。',speaker=record['speaker'])})
            self.assertIn('已复核',dialog.current.toPlainText())
            dialog.receive({'observation':dict(body='以前……「平行线」排练没结束。',speaker=record['speaker'])})
            self.assertNotIn('已复核',dialog.current.toPlainText())
        finally:
            dialog.close()

    def test_closing_ocr_dialog_stops_worker_before_returning(self):
        entered = threading.Event()
        def runner(window, config, stop, callback, **kwargs):
            entered.set()
            stop.wait(4)
        with patch("capture_windows.list_game_windows", return_value=[self.game]):
            dialog = OcrCollectionDialog(self.root)
        dialog.worker = CollectionWorker(self.game, self.config, self.root, dialog, runner=runner)
        dialog.worker.finished.connect(dialog.finished_work)
        dialog.show()
        dialog.worker.start()
        self.assertTrue(entered.wait(2))
        dialog.close()
        self.wait(lambda:not dialog.worker.isRunning())
        self.wait(lambda:not dialog.isVisible())

    def test_closing_generator_waits_for_modal_capture_and_saves_final_record(self):
        runtime = Runtime(self.root, Client())
        def runner(window, config, stop, callback, **kwargs):
            kwargs["on_session"](self.session)
            stop.wait(3)
            (self.session / "dialogue.jsonl").write_text(json.dumps(self.record()),encoding="utf-8")
        def factory(game, config, root, parent):
            return CollectionWorker(game, config, root, parent, runner=runner)
        with patch("dialogue_generator_ui.VoiceRuntime", return_value=runtime), \
             patch("capture_windows.list_game_windows", return_value=[self.game]), \
             patch("generator_ocr.CollectionWorker", side_effect=factory):
            window = GeneratorWindow({"mode":"system"}, root=self.root)
            window.show()
            project_path=window.project.directory
            QtCore.QTimer.singleShot(0, lambda:window.ocr_dialog.start_collection())
            QtCore.QTimer.singleShot(100, window.close)
            window.open_ocr()
            self.wait(lambda:not window.isVisible())
            saved=GenerationProject.load(project_path)
            self.assertEqual(len(saved.rows), 1)

    def test_failed_system_settings_save_does_not_change_active_options(self):
        runtime = Runtime(self.root, Client())
        with patch("dialogue_generator_ui.VoiceRuntime", return_value=runtime):
            window = GeneratorWindow({"mode":"system"}, root=self.root)
        fake=Mock()
        fake.exec.return_value=QtWidgets.QDialog.Accepted
        fake.result_options=dict(system_default_voice="teen_female",system_character_voices={"角色":"teen_female"})
        try:
            with patch("system_voice_dialog.SystemVoiceDialog",return_value=fake), \
                 patch("dialogue_generation.atomic_text",side_effect=PermissionError("read only")):
                window.system_voice_settings()
            self.assertEqual(window.options, {"mode":"system"})
            self.assertIn("未完成", window.status.text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
