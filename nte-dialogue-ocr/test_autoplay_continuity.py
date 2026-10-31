"""Saved camera-change regression: complete name and insignificant CJK spaces."""
import copy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from ocr_core import DialogueOCR, _text_key, roi_to_pixels
from test_ocr_core import FakeEngine, detection


EVIDENCE = Path(__file__).resolve().parent/'validation/autoplay-continuity-20261031/evidence'


class SavedDetector:
    def __init__(self, record, review=False, name=('明音凛', .91653)):
        self.record, self.review, self.name = record, review, name
        self.line_calls = []

    def __call__(self, image, **kwargs):
        kind = 'body' if image.shape[1] > 1000 else 'speaker'
        offset = roi_to_pixels(self.record['frame_size'][::-1], self.record[kind+'_roi'])[:2]
        scale = 2 if kind == 'body' else 1
        lines = self.record[('review_' if self.review else '')+kind+'_lines']
        return copy.deepcopy([[(np.asarray(line['model_bbox'])-offset)*scale,
                               line['model_text'], line['confidence']] for line in lines]), None

    def recognize_line(self, image):
        self.line_calls.append(image.shape)
        return self.name


@unittest.skipUnless((EVIDENCE/'dialogue.jsonl').is_file(), 'Saved scene transition is not installed')
class SavedSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(s) for s in (EVIDENCE/'dialogue.jsonl').read_text(encoding='utf-8').splitlines()]

    def image(self, record, kind):
        return cv2.imdecode(np.fromfile(EVIDENCE/record[kind+'_image'], np.uint8), 1)

    def frame(self, record):
        w, h = record['frame_size']
        image = np.zeros((h,w,3), np.uint8)
        for kind in ('body','speaker'):
            x0,y0,x1,y1 = roi_to_pixels((h,w), record[kind+'_roi'])
            image[y0:y1,x0:x1] = self.image(record,kind)
        return image

    def core(self, record):
        return DialogueOCR(record['body_roi'],record['speaker_roi'],centered_body=True,
                           candidate_scale=2,engine=SavedDetector(record),review_engine=SavedDetector(record,True))

    def test_all_three_saved_scenes_keep_complete_name_and_pass_review(self):
        for record in self.records[15:18]:
            with self.subTest(record=record['dialogue_id']):
                core = self.core(record)
                first = core.process_frame(self.frame(record),now=1)
                self.assertEqual(first['observation']['speaker'], '明音凛')
                reviewed = core.verify(first['event']['dialogue_id'])
                self.assertEqual(reviewed['status'], 'verified')
                self.assertEqual(reviewed['speaker_status'], 'verified')
                if record['speaker'] == '音':
                    line = first['observation']['speaker_lines'][0]
                    self.assertEqual(line['model_text'], '音')
                    self.assertTrue(line['speaker_recognition_check']['accepted'])
                    self.assertEqual(line['confidence'], .91653)
                    self.assertEqual(core.engine.line_calls, [(33,88,3)])

    def test_same_subtitle_keeps_one_identity_across_scene_changes(self):
        core = self.core(self.records[15])
        first = core.process_frame(self.frame(self.records[15]), now=1)
        core.verify(first['event']['dialogue_id'])
        for i, record in enumerate(self.records[16:18],2):
            core.engine.record = core.review_engine.record = record
            result = core.process_frame(self.frame(record),now=i)
            self.assertIn(result['status'], ('duplicate','revision'))
            self.assertEqual(result['observation']['speaker'], '明音凛')
            if result['event']:
                self.assertEqual(result['event']['dialogue_id'], first['event']['dialogue_id'])
        self.assertEqual(core._sequence, 1)

    def test_uncertain_or_unrelated_name_reread_cannot_replace_original(self):
        record = self.records[16]
        for name in (('明音凛',.89), ('另一角色',.99), ('',.99), ('明音凛',float('nan'))):
            with self.subTest(name=name):
                core = self.core(record)
                core.engine.name = name
                result = core.process_frame(self.frame(record),now=1)['observation']
                self.assertEqual(result['speaker'], '音')
                self.assertFalse(result['speaker_lines'][0]['speaker_recognition_check']['accepted'])

    def test_well_fitted_name_does_not_require_an_extra_recognition_pass(self):
        record = self.records[15]
        core = self.core(record)
        core.process_frame(self.frame(record),now=1)
        self.assertEqual(core.engine.line_calls, [])


class ReviewSpacingTests(unittest.TestCase):
    def test_cjk_layout_spaces_do_not_change_review_result_or_saved_text(self):
        text = '以前……「平行线」结束后，我们只是想玩游戏而\n已。'
        reviewed_text = '以前…… 「平行线」结束后，我们只是想玩游戏而已。'
        engine = FakeEngine([[detection(text,.99)],[detection(reviewed_text,.98)]])
        core = DialogueOCR([0,0,1,1],engine=engine)
        first = core.process_frame(np.zeros((100,200,3),np.uint8),now=1)['event']
        result = core.verify(first['dialogue_id'])
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['original_text'], text)
        self.assertEqual(result['review_text'], reviewed_text)

    def test_lexical_punctuation_and_latin_word_changes_remain_distinct(self):
        for before, after in (('不要开门。','要开门。'),('奖励10。','奖励11。'),
                              ('要去吗？','要去吗。'),('AB C','A BC'),('Play','play')):
            with self.subTest(before=before,after=after):
                self.assertNotEqual(_text_key(before),_text_key(after))

    def test_low_confidence_review_is_still_held_despite_equal_text(self):
        engine = FakeEngine([[detection('以前……「平行线」',.99)],[detection('以前…… 「平行线」',.89)]])
        core = DialogueOCR([0,0,1,1],engine=engine)
        first = core.process_frame(np.zeros((100,200,3),np.uint8),now=1)['event']
        result = core.verify(first['dialogue_id'])
        self.assertTrue(result['agreement'])
        self.assertEqual(result['status'], 'needs_review')


if __name__ == '__main__':
    unittest.main(verbosity=2)
