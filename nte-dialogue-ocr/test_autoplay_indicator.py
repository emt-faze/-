"""The saved continue icon must not poison an otherwise complete subtitle."""
import copy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from ocr_core import DialogueOCR, roi_to_pixels


EVIDENCE = Path(__file__).resolve().parent / 'validation/autoplay-indicator-20261031/evidence'


@unittest.skipUnless((EVIDENCE/'dialogue.jsonl').is_file(), 'Saved game evidence is not installed')
class SavedIndicatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(s) for s in (EVIDENCE/'dialogue.jsonl').read_text(encoding='utf-8').splitlines()]

    def image(self, record, kind):
        return cv2.imdecode(np.fromfile(EVIDENCE/record[kind+'_image'], np.uint8), cv2.IMREAD_COLOR)

    def detector(self, record, *, review=False):
        detections = {}
        for kind, scale in (('body', 2), ('speaker', 1)):
            offset = roi_to_pixels(record['frame_size'][::-1], record[kind+'_roi'])[:2]
            lines = record[('review_' if review else '')+kind+'_lines']
            detections[kind] = [[(np.asarray(line['model_bbox'])-offset)*scale,
                                 line['model_text'], line['confidence']] for line in lines]
        def detect(image, **kwargs):
            return copy.deepcopy(detections['body' if image.shape[1] > 1000 else 'speaker']), None
        return detect

    def core(self, record):
        return DialogueOCR(record['body_roi'], record['speaker_roi'], centered_body=True,
                           candidate_scale=2, engine=self.detector(record),
                           review_engine=self.detector(record, review=True))

    def frame(self, record):
        w, h = record['frame_size']
        frame = np.zeros((h, w, 3), np.uint8)
        for kind in ('body', 'speaker'):
            x0, y0, x1, y1 = roi_to_pixels((h, w), record[kind+'_roi'])
            frame[y0:y1, x0:x1] = self.image(record, kind)
        return frame

    def test_all_reported_frames_have_complete_body_and_matching_role(self):
        for record in self.records:
            with self.subTest(frame=record['dialogue_id']):
                core = self.core(record)
                result = core.process_frame(self.frame(record), now=1)
                observation = result['observation']
                self.assertEqual(observation['body'], '不，还没到最后。')
                self.assertEqual(observation['speaker'], '瓦亚波洛尼亚')
                self.assertGreater(observation['body_confidence'], .99)
                self.assertFalse(observation['source_clipped'])
                self.assertFalse(observation['coverage_warning'])
                review = core.verify(result['event']['dialogue_id'])
                self.assertEqual(review['status'], 'verified')
                self.assertEqual(review['speaker_status'], 'verified')

    def test_reused_body_keeps_clean_text_and_confidence(self):
        record = self.records[0]
        core = self.core(record)
        frame = self.frame(record)
        core.process_frame(frame, now=1)
        reused = core.process_frame(frame, now=1.2, reuse_body=True)['observation']
        self.assertTrue(reused['body_pixels_unchanged'])
        self.assertEqual(reused['body'], '不，还没到最后。')
        self.assertGreater(reused['body_confidence'], .99)


class PunctuationBoundaryTests(unittest.TestCase):
    def recognize(self, rows, *, centered=True):
        image = np.zeros((200, 600, 3), np.uint8)
        detections = []
        for text, (x, y, w, h), confidence in rows:
            image[y+h//3:y+2*h//3, x+2:x+w-2] = 255
            detections.append([[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],text,confidence])
        core = DialogueOCR([0,0,1,1], engine=lambda *a, **kw:(copy.deepcopy(detections), None))
        return core._recognize_image(image, centered=centered)

    def test_low_confidence_words_remain_uncertain(self):
        lines, body = self.recognize([('还没到最后。',(230,20,140,30),.99),
                                      ('不',(290,170,20,25),.55)])
        self.assertIn('不', body)
        self.assertEqual(min(line['confidence'] for line in lines), .55)

    def test_adjacent_wrapped_punctuation_is_retained(self):
        lines, body = self.recognize([('还没到最后',(230,110,140,30),.99),
                                      ('。',(290,150,20,25),.55)])
        self.assertEqual(body, '还没到最后\n。')
        self.assertEqual(len(lines), 2)

    def test_real_ellipsis_and_short_final_word_are_retained(self):
        for tail in ('……', '...', '已。'):
            with self.subTest(tail=tail):
                _, body = self.recognize([('前面的正文。',(230,20,140,30),.99),
                                          (tail,(280,170,40,25),.99)])
                self.assertEqual(body, '前面的正文。\n'+tail)

    def test_punctuation_without_a_text_anchor_is_not_discarded(self):
        for text in ('。', '……'):
            _, body = self.recognize([(text,(285,170,30,25),.99)])
            self.assertEqual(body, text)

    def test_only_detached_bottom_mark_in_centered_dialogue_is_filtered(self):
        rows = [('不，还没到最后。',(230,20,140,30),.99), ('·',(285,170,30,25),.55)]
        _, body = self.recognize(rows)
        self.assertEqual(body, '不，还没到最后。')
        _, generic = self.recognize(rows, centered=False)
        self.assertEqual(generic, '不，还没到最后。\n·')


if __name__ == '__main__':
    unittest.main(verbosity=2)
