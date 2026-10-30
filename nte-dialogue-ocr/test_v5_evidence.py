"""Regressions for the reported missing stroke and repeated real dialogue pages."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from frame_gate import FrameStability
from ocr_core import DialogueOCR
from ocr_engines import RapidModel

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / 'records/2026-10-30/3bc69b87ef3a'
LIVE = ROOT / 'validation/wgc-dual-20261030/live-03-gui'


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


class SavedDetector:
    def __init__(self, lines):
        self.rows = [[((np.asarray(line['model_bbox']) - [211, 869]) * 2).tolist(),
                      line['model_text'], line['confidence']] for line in lines]

    def __call__(self, image, **kwargs):
        return copy.deepcopy(self.rows), None


class SavedDetectorWithReader(SavedDetector):
    def __init__(self, lines, response):
        super().__init__(lines)
        self.response = response
        self.read_crops = []

    def recognize_line(self, image):
        self.read_crops.append(image.copy())
        return self.response


@unittest.skipUnless((SESSION / 'dialogue.jsonl').exists(), 'Real evidence is not installed')
class V5EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.reported = json.loads((SESSION / '3bc69b87ef3a-000027.json').read_text(encoding='utf-8'))

    def make_core(self, response=None):
        make = (lambda lines: SavedDetector(lines)) if response is None else (
            lambda lines: SavedDetectorWithReader(lines, response))
        return DialogueOCR([0, 0, 1, 1], candidate_scale=2,
                           engine=make(self.reported['body_lines']),
                           review_engine=make(self.reported['review_body_lines']))

    def test_real_uncovered_stroke_is_read_with_source_text_and_same_id_preserved(self):
        core = self.make_core(('一言为定。', .99991))
        source = read_image(SESSION / self.reported['body_image'])
        original = source.copy()
        first = core.process_frame(source)['event']
        review = core.verify(first['dialogue_id'])
        self.assertEqual(first['body'], '一言为定。')
        self.assertEqual(first['raw_model_text'], '言为定。')
        self.assertEqual(review['review_text'], '一言为定。')
        self.assertEqual(review['dialogue_id'], first['dialogue_id'])
        self.assertEqual(review['status'], 'verified')
        check = first['body_lines'][0]['line_recognition_check']
        self.assertTrue(check['accepted'])
        self.assertTrue(check['uncovered_ink_boxes'])
        # The crop supplied to recognition really contains the missing stroke.
        self.assertLessEqual(check['crop_box'][0], 691)
        self.assertGreater(core.engine.read_crops[0].shape[1], 194)
        np.testing.assert_array_equal(source, original)

    def test_equal_high_scores_cannot_verify_uncovered_ink_without_a_reader(self):
        core = self.make_core()
        first = core.process_frame(read_image(SESSION / self.reported['body_image']))['event']
        review = core.verify(first['dialogue_id'])
        self.assertGreater(first['body_confidence'], .999)
        self.assertEqual(review['body'], review['review_text'])
        self.assertTrue(review['coverage_warning'])
        self.assertEqual(review['status'], 'needs_review')

    def test_an_inconclusive_or_changed_core_line_stays_uncertain(self):
        for text, score in [('言为定。', .9999), ('一言为定。', .4), ('一语为定。', .9999)]:
            with self.subTest(text=text, score=score):
                core = self.make_core((text, score))
                first = core.process_frame(read_image(SESSION / self.reported['body_image']))['event']
                review = core.verify(first['dialogue_id'])
                self.assertEqual(first['body'], '言为定。')
                self.assertEqual(review['status'], 'needs_review')
                self.assertFalse(first['body_lines'][0]['line_recognition_check']['accepted'])

    def test_neighboring_detected_word_is_not_read_twice(self):
        # Detector outputs for preserved frame 549. The next word's 下 is already
        # covered by the second box; expanding the first would duplicate it.
        lines = [
            dict(model_bbox=[[645, 873], [850, 873], [850, 905], [645, 905]],
                 model_text='今天就先到这里吧，', confidence=.99924),
            dict(model_bbox=[[859.5, 874], [1264.5, 874], [1264.5, 902.5], [859.5, 902.5]],
                 model_text='下次再战，鼓手大人应该不会怯战吧？', confidence=.99939),
        ]
        engine = SavedDetectorWithReader(lines, ('今天就先到这里吧，下', .9999))
        core = DialogueOCR([0, 0, 1, 1], candidate_scale=2, engine=engine)
        event = core.process_frame(read_image(LIVE / 'frames/00549.png')[62:])['event']
        self.assertEqual(event['body'], '今天就先到这里吧，下次再战，鼓手大人应该不会怯战吧？')
        self.assertEqual(engine.read_crops, [])

    def test_same_real_pages_do_not_retrigger_from_antialias_brightness(self):
        for begin, end in [(486, 501), (544, 561)]:
            gate = FrameStability(.45)
            changed = []
            for index in range(begin, end + 1):
                if gate.ready(read_image(LIVE / f'frames/{index:05d}.png')[62:], index * .5):
                    if not gate.unchanged:
                        changed.append(index)
                    gate.commit()
            self.assertEqual(len(changed), 1, (begin, end, changed))


class RapidModelFlagTests(unittest.TestCase):
    def test_line_read_does_not_disable_detection_on_the_next_frame(self):
        calls = []

        def engine(image, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(txts=['一言为定。'], scores=[.99],
                                   boxes=np.array([[[0, 0], [100, 0], [100, 30], [0, 30]]]))

        wrapper = RapidModel.__new__(RapidModel)
        wrapper.engine = engine
        frame = np.zeros((40, 140, 3), np.uint8)
        self.assertEqual(wrapper.recognize_line(frame), ('一言为定。', .99))
        self.assertEqual(wrapper(frame)[0][0][1], '一言为定。')
        self.assertEqual([call['use_det'] for call in calls], [False, True])


class ChangedGlyphTests(unittest.TestCase):
    def test_digits_punctuation_and_negation_still_trigger_a_new_page(self):
        from PIL import Image, ImageDraw, ImageFont
        font_path = Path('C:/Windows/Fonts/msyh.ttc')
        if not font_path.exists():
            self.skipTest('Chinese test font is not installed')
        font = ImageFont.truetype(str(font_path), 30)

        def render(text):
            canvas = Image.new('RGB', (520, 65), 'black')
            ImageDraw.Draw(canvas).text((10, 10), text, font=font, fill='white')
            return np.asarray(canvas).copy()

        for before, after in [('还有10次。', '还有11次。'), ('可以。', '不可以。'), ('去吧。', '去吧？')]:
            with self.subTest(before=before, after=after):
                gate = FrameStability(.45)
                for i in range(4):
                    gate.ready(render(before), i * .2)
                gate.commit()
                for i in range(4, 8):
                    ready = gate.ready(render(after), i * .2)
                self.assertTrue(ready)
                self.assertFalse(gate.unchanged)


if __name__ == '__main__':
    unittest.main()
