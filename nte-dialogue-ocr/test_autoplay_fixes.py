"""Regression evidence for background names and the reported long subtitle."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from frame_gate import SubtitleChangeGuard
from ocr_core import DialogueOCR,roi_to_pixels

EVIDENCE=Path(__file__).resolve().parent/'validation/autoplay-fixes-20261031/evidence'


@unittest.skipUnless((EVIDENCE/'dialogue.jsonl').is_file(),'Saved game evidence is not installed')
class ReportedSubtitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[json.loads(s) for s in (EVIDENCE/'dialogue.jsonl').read_text(encoding='utf-8').splitlines()]

    def image(self,number,kind='body'):
        record=self.records[number-1]
        return cv2.imdecode(np.fromfile(EVIDENCE/record[kind+'_image'],np.uint8),cv2.IMREAD_COLOR)

    def offset(self,number,kind='body'):
        record=self.records[number-1]
        return roi_to_pixels(record['frame_size'][::-1],record[kind+'_roi'])[:2]

    def detector(self,number,kind='body',reader=None):
        record=self.records[number-1]
        offset=self.offset(number,kind)
        scale=2 if kind=='body' else 1
        raw=[[(np.asarray(line['model_bbox'])-offset)*scale,line['model_text'],line['confidence']]
             for line in record[kind+'_lines']]
        class Detector:
            def __call__(self,image,**kwargs):
                return copy.deepcopy(raw),None
            def recognize_line(self,image):
                return reader or ('',0.)
        return Detector()

    def test_vertical_background_brand_does_not_join_character_name(self):
        for number in (16,17,18,19):
            core=DialogueOCR([0,0,1,1],engine=self.detector(number,'speaker'))
            lines,name=core._recognize_image(self.image(number,'speaker'),self.offset(number,'speaker'),speaker_style=True)
            self.assertEqual(name,'明音凛')
            self.assertEqual(len(lines),1)

    def test_two_character_wrapped_last_row_keeps_tilted_detector_box(self):
        core=DialogueOCR([0,0,1,1],engine=self.detector(19))
        lines,text=core._recognize_image(self.image(19),self.offset(19),2,centered=True,recover_punctuation=True)
        self.assertTrue(text.endswith('而\n已。'),text)
        self.assertEqual(len({line['row'] for line in lines}),2)

    def test_existing_first_stroke_is_not_mistaken_for_an_extra_missing_word(self):
        core=DialogueOCR([0,0,1,1],engine=self.detector(25,reader=('一言为定。',.9999)))
        lines,text=core._recognize_image(self.image(25),self.offset(25),2,centered=True,recover_punctuation=True)
        self.assertEqual(text,'一言为定。')
        self.assertFalse(any(line['coverage_warning'] for line in lines))
        self.assertTrue(lines[0]['line_recognition_check']['accepted'])

    def test_moving_background_and_name_sign_do_not_interrupt_same_long_sentence(self):
        guard=SubtitleChangeGuard()
        observation=dict(self.records[15],captured_monotonic=10.)
        guard.arm(self.image(16),observation,self.offset(16))
        for i,number in enumerate((17,17,18,18,19,19),1):
            self.assertFalse(guard.changed(self.image(number),10+i*.2))
        self.assertTrue(guard.changed(self.image(20),12.))
        self.assertFalse(guard.changed(self.image(20),12.2))

    def test_late_old_ocr_cannot_rearm_guard_after_new_page(self):
        guard=SubtitleChangeGuard()
        observation=dict(self.records[15],captured_monotonic=10.)
        guard.arm(self.image(16),observation,self.offset(16))
        self.assertTrue(guard.changed(self.image(20),11.))
        guard.arm(self.image(16),observation,self.offset(16))
        self.assertIsNone(guard.reference)
        guard.arm(self.image(20),dict(self.records[19],captured_monotonic=11.2),self.offset(20))
        self.assertIsNotNone(guard.reference)


class GlyphChangeTests(unittest.TestCase):
    def test_small_change_needs_two_new_frames_and_background_outside_text_is_ignored(self):
        original=np.zeros((30,80),bool)
        original[5:20,10:40]=True
        observation=dict(body='旧字幕',body_confidence=.99,captured_monotonic=1.,
                         body_lines=[dict(bbox=[[10,5],[40,5],[40,20],[10,20]])])
        guard=SubtitleChangeGuard()
        with patch('frame_gate.glyph_mask',side_effect=lambda image:image):
            guard.arm(original,observation,(0,0))
            outside=original.copy()
            outside[22:28,55:70]=True
            self.assertFalse(guard.changed(outside,1.2))
            self.assertFalse(guard.changed(outside,1.4))
            different=outside.copy()
            different[6:10,11:20]=False
            self.assertFalse(guard.changed(different,1.6))
            self.assertFalse(guard.changed(different,1.6))
            self.assertTrue(guard.changed(different,1.8))


if __name__=='__main__':
    unittest.main(verbosity=2)
