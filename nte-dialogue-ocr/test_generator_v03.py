"""Regression evidence for punctuation, reversible deletion and manual entry."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets

from dialogue_generation import GenerationProject, BatchGenerator, file_digest
from dialogue_generator_ui import GeneratorWindow, ManualDialogueDialog
from generator_ocr import OcrCollectionDialog, CollectionWorker
from ocr_core import DialogueOCR, standalone_ellipsis
from test_ocr_core import FakeEngine, detection
from test_dialogue_generation import Client, Runtime


class StandaloneEllipsisTests(unittest.TestCase):
    def picture(self, count=6, *, shift=0, background=30, color=(255,255,255), irregular=False):
        frame = np.full((120, 1500, 3), background, np.uint8)
        for index in range(count):
            x = int(750-(count-1)*5+index*10+shift)
            if irregular and index == 2:
                x += 4
            cv2.circle(frame, (x, 50), 2, color, -1)
        return frame

    def test_user_screenshot_and_resized_copies_recover_six_dots(self):
        path = Path(__file__).parent/'validation/fixtures/standalone-ellipsis-body.png'
        if not path.exists():
            self.skipTest('Private saved-game evidence is not included in the public package')
        image = cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        for scale in (1, 1.25, 1.5, 2):
            frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            with self.subTest(scale=scale):
                text, boxes = standalone_ellipsis(frame)
                self.assertEqual(text, '……')
                self.assertEqual(len(boxes), 6)

    def test_empty_model_results_produce_evidence_and_require_human_review(self):
        core = DialogueOCR([0,0,1,1], engine=FakeEngine([[],[]]), centered_body=True)
        event = core.process_frame(self.picture())['event']
        self.assertEqual(event['body'], '……')
        self.assertEqual(event['raw_model_text'], '')
        line = event['body_lines'][0]
        self.assertEqual(line['confidence_source'], 'pixel_geometry')
        self.assertEqual(len(line['punctuation_evidence'][0]['dot_boxes']),6)
        reviewed = core.verify(event['dialogue_id'])
        self.assertEqual(reviewed['review_text'], '……')
        self.assertEqual(reviewed['status'], 'needs_review')

    def test_fresh_speaker_is_read_even_when_body_models_detect_nothing(self):
        frame = self.picture()
        # Match the actual name style; a detector label on a blank/black patch
        # is now correctly rejected as background evidence.
        frame[5:20,715:785]=(105,180,250)
        name=detection('明音凛',x=710,y=3,width=80,height=20)
        outputs = [[],[name],[],[name]]
        core = DialogueOCR([0,0,1,1],[0,0,1,.2],engine=FakeEngine(outputs),centered_body=True)
        first = core.process_frame(frame)['event']
        self.assertEqual(first['speaker'], '明音凛')
        second = core.process_frame(frame)['event']
        self.assertTrue(second['speaker_identity_confirmed'])
        self.assertEqual(second['speaker_identity'], '明音凛')

    def test_unsupported_count_background_color_spacing_and_extra_ink_are_rejected(self):
        pictures = [self.picture(count) for count in (0,1,2,4,5,7,9)]
        pictures += [self.picture(shift=300), self.picture(background=180),
                     self.picture(color=(10,255,255)), self.picture(irregular=True)]
        with_word = self.picture()
        cv2.rectangle(with_word,(630,40),(643,64),(255,255,255),-1)
        pictures.append(with_word)
        two_rows = self.picture()
        for x in range(725, 776, 10):
            cv2.circle(two_rows,(x,95),2,(255,255,255),-1)
        pictures.append(two_rows)
        for i, frame in enumerate(pictures):
            self.assertEqual(standalone_ellipsis(frame), ('',[]), f'negative case {i}')

    def test_three_dots_offsets_and_model_dot_token(self):
        frame = self.picture(3)
        core = DialogueOCR([0,0,1,1], engine=FakeEngine([[]]),centered_body=True)
        lines, text = core._recognize_image(frame,(11,22),2,True,True)
        self.assertEqual(text,'…')
        self.assertEqual(lines[0]['bbox'][0],[749,70])
        core.engine = FakeEngine([[detection('...',x=738,y=48,width=25,height=5)]])
        self.assertEqual(core._recognize_image(frame,recover_punctuation=True,centered=True)[1],'…')

    def test_fallback_is_body_only_and_does_not_overwrite_ordinary_text(self):
        core = DialogueOCR([0,0,1,1],engine=FakeEngine([[],[]]))
        self.assertEqual(core._recognize_image(self.picture(),recover_punctuation=True)[1],'')
        self.assertEqual(core._recognize_image(self.picture(),centered=True)[1],'')
        frame = np.full((120,1500,3),30,np.uint8)
        cv2.rectangle(frame,(690,44),(810,48),(255,255,255),-1)
        core.engine = FakeEngine([[detection('正常台词……',x=680,y=40,width=140,height=25)]])
        with patch('ocr_core.standalone_ellipsis') as recover:
            self.assertEqual(core._recognize_image(frame,recover_punctuation=True,centered=True)[1],'正常台词……')
            recover.assert_not_called()


class ProjectEditingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = GenerationProject.create(self.root/'projects')
        self.client = Client()
        self.runtime = Runtime(self.root,self.client)

    def tearDown(self):
        self.temp.cleanup()

    def test_manual_multiline_entry_validates_and_routes_explicit_voice(self):
        with self.assertRaisesRegex(ValueError,'正文'):
            self.project.add_dialogue(' \n ')
        row = self.project.add_dialogue(' 第一行\n第二行 ', ' 伊洛伊 ', 'system:teen_female')
        self.assertEqual((row.text,row.speaker,row.voice),('第一行\n第二行','伊洛伊','system:teen_female'))
        self.assertTrue(row.confirmed)
        self.assertEqual(row.route_record()['speaker_identity'],'伊洛伊')
        result = BatchGenerator(self.project,{},runtime_factory=lambda _:self.runtime).run()
        self.assertEqual(result['completed'],1)
        self.assertEqual(len(self.client.calls),1)
        self.assertIsNotNone(self.project.audio_path(row))

    def test_deletion_preserves_audio_cache_and_undo_after_reopen(self):
        self.project.add_text('伊洛伊：你好\n娜娜莉：我们出发吧\n最后一句')
        original_ids = [r.id for r in self.project.rows]
        BatchGenerator(self.project,{},runtime_factory=lambda _:self.runtime).run()
        audio = {str(p): file_digest(p) for p in self.root.rglob('*.wav')}
        self.assertEqual(self.project.delete_rows(original_ids[::2]),2)
        self.project.save()
        manifest = json.loads((self.project.directory/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual([r['id'] for r in manifest], [original_ids[1]])
        saved = GenerationProject.load(self.project.directory)
        self.assertEqual(len(saved.deleted_batches),1)
        saved.undo_delete()
        self.assertEqual([r.id for r in saved.rows],original_ids)
        self.assertTrue(all(saved.audio_path(r) for r in saved.rows))
        self.assertEqual(audio,{str(p):file_digest(p) for p in self.root.rglob('*.wav')})

    def test_deleted_ocr_revision_stays_deleted_and_restores_latest_evidence(self):
        source = self.root/'records'
        record = dict(dialogue_id='session-1',revision=1,body='原句',speaker='甲',status='candidate')
        self.project.receive_record(record,source)
        identity = self.project.rows[0].id
        self.project.delete_rows([identity])
        self.project.save()
        self.project = GenerationProject.load(self.project.directory)
        result = self.project.receive_record(dict(record,revision=2,body='新修订'),source)
        self.assertEqual(result['deleted'],1)
        self.assertEqual(self.project.rows,[])
        source.mkdir()
        path = source/'dialogue.jsonl'
        path.write_text(json.dumps(record),encoding='utf-8')
        self.project.import_records(path)
        self.assertEqual(self.project.rows,[])
        self.project.undo_delete()
        self.assertEqual(self.project.rows[0].text,'新修订')
        self.assertEqual(self.project.rows[0].id,identity)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')),record)

    def test_multiple_deletions_restore_positions_and_mode_change_invalidates_archived_audio(self):
        self.project.add_text('一\n二\n三\n四')
        original = [row.id for row in self.project.rows]
        self.project.rows[1].audio = {'file':'old.wav'}
        self.project.delete_rows([original[1],original[3]])
        self.project.delete_rows([original[0]])
        self.project.add_dialogue('新添加')
        self.project.set_mode('character')
        self.project.set_mode('system')
        self.project.undo_delete()
        self.project.undo_delete()
        self.assertEqual([r.id for r in self.project.rows[:4]],original)
        self.assertEqual(self.project.rows[1].audio,{})
        self.assertEqual(self.project.undo_delete(),[])

    def test_silent_ocr_and_manual_entries_do_not_start_model_even_if_checked(self):
        self.project.receive_record(dict(dialogue_id='dots',revision=0,body='……',speaker='明音凛'),self.root/'records')
        self.project.add_dialogue('......')
        self.assertTrue(all(not r.selected for r in self.project.rows))
        for row in self.project.rows:
            row.selected = True
            row.confirmed = True
        with patch.object(self.runtime,'prepare') as prepare:
            result = BatchGenerator(self.project,{},runtime_factory=lambda _:self.runtime).run()
            prepare.assert_not_called()
        self.assertEqual(result['silent'],2)
        self.assertEqual(result['held'],0)
        self.assertEqual(self.client.calls,[])
        self.assertEqual([r['state'] for r in json.loads((self.project.directory/'manifest.json').read_text(encoding='utf-8'))],['silent','silent'])

    def test_old_project_load_and_duplicate_archived_ids_validation(self):
        self.project.add_dialogue('旧项目')
        self.project.save()
        path = self.project.directory/'project.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['version'] = 1
        data.pop('deleted_batches')
        path.write_text(json.dumps(data),encoding='utf-8')
        self.assertEqual(GenerationProject.load(path).rows[0].text,'旧项目')
        data['deleted_batches'] = [dict(deleted_at='today',entries=[dict(index=0,row=data['rows'][0])])]
        path.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaises(ValueError):
            GenerationProject.load(path)


class EditingUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = Runtime(self.root,Client())
        self.patcher = patch('dialogue_generator_ui.VoiceRuntime',return_value=self.runtime)
        self.patcher.start()
        self.window = GeneratorWindow({'mode':'system'},root=self.root)
        self.window.project.add_text('甲：第一句\n乙：第二句\n丙：第三句')
        self.window.refresh_selection(0)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.patcher.stop()
        self.temp.cleanup()

    def test_delete_commits_pending_editor_and_undo_preserves_it_without_touching_next_row(self):
        self.window.text.setPlainText('尚未自动保存的修改')
        self.assertTrue(self.window.edit_timer.isActive())
        self.window.delete_current()
        self.assertFalse(self.window.edit_timer.isActive())
        self.assertEqual(self.window.text.toPlainText(),'第二句')
        self.window.undo_delete()
        self.assertEqual(self.window.text.toPlainText(),'尚未自动保存的修改')
        self.assertEqual(self.window.project.rows[1].text,'第二句')
        saved = GenerationProject.load(self.window.project.directory)
        self.assertEqual(saved.rows[0].text,'尚未自动保存的修改')

    def test_delete_checked_all_clear_editor_and_recover(self):
        self.window.delete_checked()
        self.assertEqual(self.window.table.rowCount(),0)
        self.assertIsNone(self.window.editor_row)
        self.assertFalse(self.window.editor.isEnabled())
        self.assertTrue(self.window.undo_button.isEnabled())
        self.window.undo_delete()
        self.assertEqual(self.window.table.rowCount(),3)
        self.assertEqual(self.window.editor_row.speaker,'甲')
        self.assertFalse(self.window.undo_button.isEnabled())

    def test_manual_dialog_rejects_blank_then_appends_multiline_and_selects_it(self):
        def fill():
            dialog = self.window.manual_dialog
            dialog.accept()
            self.assertIn('正文',dialog.error.text())
            self.assertTrue(dialog.isVisible())
            dialog.speaker.setEditText('新增角色')
            dialog.voice.setCurrentIndex(dialog.voice.findData('system:teen_female'))
            dialog.text.setPlainText('手动台词\n仍属于同一句')
            dialog.accept()
        QtCore.QTimer.singleShot(0,fill)
        self.window.add_dialogue()
        self.assertEqual(len(self.window.project.rows),4)
        row = self.window.editor_row
        self.assertEqual(row.speaker,'新增角色')
        self.assertEqual(row.text,'手动台词\n仍属于同一句')
        self.assertEqual(row.voice,'system:teen_female')
        self.assertTrue(row.confirmed)
        self.assertEqual(GenerationProject.load(self.window.project.directory).rows[-1].id,row.id)

    def test_manual_cancel_and_busy_guards_do_not_modify_project(self):
        QtCore.QTimer.singleShot(0,lambda:self.window.manual_dialog.reject())
        self.window.add_dialogue()
        self.assertEqual(len(self.window.project.rows),3)
        with patch.object(self.window,'busy',return_value=True):
            self.window.set_busy(True)
            self.window.delete_checked()
            self.window.undo_delete()
            self.window.add_dialogue()
            self.assertFalse(self.window.manual_button.isEnabled())
            self.assertFalse(self.window.delete_checked_button.isEnabled())
        self.assertEqual(len(self.window.project.rows),3)

    def test_save_failure_blocks_deletion_and_preserves_unsaved_editor(self):
        self.window.text.setPlainText('有待保存')
        with patch.object(self.window.project,'save',side_effect=PermissionError('read only')):
            self.window.delete_current()
        self.assertEqual(len(self.window.project.rows),3)
        self.assertEqual(self.window.project.rows[0].text,'有待保存')
        self.assertIn('未能保存',self.window.status.text())

    def test_fresh_blank_observation_clears_old_text_and_late_review_does_not_restore_it(self):
        (self.root/'config.json').write_text('{}',encoding='utf-8')
        with patch('capture_windows.list_game_windows',return_value=[]):
            dialog = OcrCollectionDialog(self.root)
        collected = []
        dialog.collected.connect(collected.append)
        record = dict(dialogue_id='1',body='上一句',speaker='甲',status='candidate')
        try:
            dialog.receive(dict(record=record,source='source'))
            self.assertIn('上一句',dialog.current.toPlainText())
            dialog.receive(dict(observation=dict(body='',speaker='')))
            dialog.receive(dict(record=dict(record,status='verified'),source='source'))
            self.assertEqual(dialog.current.toPlainText(),'')
            dialog.receive(dict(observation=dict(body='……',speaker='明音凛')))
            self.assertIn('明音凛',dialog.current.toPlainText())
            self.assertNotIn('甲',dialog.current.toPlainText())
            self.assertEqual(len(collected),2)
        finally:
            dialog.close()

    def test_worker_forwards_observation_without_importing_it_as_a_record(self):
        events = []
        def run(window, config, stop, callback, **hooks):
            hooks['on_observation'](dict(body='……',speaker='明音凛'))
            hooks['on_observation'](dict(body='',speaker=''))
        worker = CollectionWorker(None,{},self.root,runner=run)
        worker.message.connect(events.append)
        worker.run()
        self.assertEqual([e['observation']['body'] for e in events],['……',''])
        self.assertTrue(all('record' not in e for e in events))


if __name__ == '__main__':
    unittest.main()
