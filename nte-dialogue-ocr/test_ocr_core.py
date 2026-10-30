"""Deterministic state/geometry tests; no game capture or model download."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from ocr_core import DialogueOCR, crop_roi, roi_to_pixels, validate_roi, leading_ellipsis


def detection(text, confidence=0.98, x=0, y=5, width=35, height=10):
    return [[[x, y], [x + width, y], [x + width, y + height], [x, y + height]], text, confidence]


class FakeEngine:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        value = next(self.results)
        if isinstance(value, Exception):
            raise value
        return value, None


class ROITests(unittest.TestCase):
    def test_invalid_rois_fail_instead_of_silently_clipping(self):
        for roi in (None, "0,0,1,1", [0, 0, 1], [False, 0, 1, 1], [0, 0, float("nan"), 1],
                    [-0.01, 0, 1, 1], [0, 0, 1.01, 1], [0.5, 0, 0.5, 1], [0, 0.8, 1, 0.2]):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                validate_roi(roi)

    def test_crop_keeps_fractional_edge_and_does_not_mutate_frame(self):
        frame = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        roi = [0.1, 0.2, 0.9, 0.9]
        self.assertEqual(roi_to_pixels(frame.shape, roi), (1, 1, 10, 7))
        result = crop_roi(frame, roi)
        np.testing.assert_array_equal(result, frame[1:7, 1:10])
        self.assertTrue(result.flags.c_contiguous)
        result[:] = 0
        self.assertNotEqual(int(frame[1, 1, 0]), 0)

    def test_invalid_frame_formats_rejected(self):
        for frame in (np.zeros((0, 10, 3), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8),
                      np.zeros((10, 10, 3), dtype=float)):
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                crop_roi(frame, [0, 0, 1, 1])


class PunctuationTests(unittest.TestCase):
    def test_real_regression_recovers_visible_dots_at_both_scales_and_keeps_model_text(self):
        from pathlib import Path
        import cv2
        source = Path(__file__).parent/'records/2026-10-29/b2943853ae9c/images/b2943853ae9c-000001_body.png'
        if not source.exists():
            self.skipTest('Private real-game evidence is not included in the public package')
        image = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
        box = [[950, 7], [1100, 7], [1100, 44], [950, 44]]
        for scale in (1, 2):
            with self.subTest(scale=scale):
                detection_box = (np.asarray(box)*scale).tolist()
                core = DialogueOCR([0, 0, 1, 1], engine=FakeEngine([[ [detection_box, '味道如何？', .98] ]]))
                lines, text = core._recognize_image(image, (281,1320), scale, recover_punctuation=True)
                self.assertEqual(text, '……味道如何？')
                self.assertEqual(lines[0]['model_text'], '味道如何？')
                self.assertEqual(len(lines[0]['punctuation_evidence'][0]['dot_boxes']), 6)
                self.assertLess(lines[0]['bbox'][0][0], 1200)
                self.assertEqual(lines[0]['model_bbox'][0], [1231.,1327.])

    def test_unsupported_dot_counts_and_unoutlined_background_do_not_insert_punctuation(self):
        import cv2
        box = [[150,10],[230,10],[230,50],[150,50]]
        for count in (0, 1, 2, 4, 5, 7):
            frame = np.full((70,250,3), 30, np.uint8)
            for index in range(count):
                cv2.circle(frame, (144-index*11,30), 2, (255,255,255), -1)
            self.assertEqual(leading_ellipsis(frame,box)[0], '', f'count={count}')
        frame = np.full((70,250,3), 180, np.uint8)
        for index in range(6):
            cv2.circle(frame, (144-index*11,30), 2, (255,255,255), -1)
        self.assertEqual(leading_ellipsis(frame,box)[0], '')

    def test_existing_punctuation_is_never_doubled(self):
        core=DialogueOCR([0,0,1,1],engine=FakeEngine([[detection('……原句')]]))
        with patch('ocr_core.leading_ellipsis') as recover:
            core.process_frame(np.zeros((100,200,3),np.uint8))
            recover.assert_not_called()

    def test_all_real_animation_regressions_keep_six_dots_including_partial_model_prefix(self):
        from pathlib import Path
        import cv2
        root = Path(__file__).parent/'records/2026-10-29/1f9996037ca3'
        if not root.exists():
            self.skipTest('Private real-game evidence is not included in the public package')
        for path in sorted(root.glob('1f9996037ca3-*.json')):
            record=json.loads(path.read_text(encoding='utf-8'))
            image=cv2.imdecode(np.fromfile(root/record['body_image'],np.uint8),cv2.IMREAD_COLOR)
            for scale,key in ((1,'body_lines'),(2,'review_body_lines')):
                line=record[key][0]
                box=((np.asarray(line['model_bbox'])-np.asarray([281,1320]))*scale).tolist()
                core=DialogueOCR([0,0,1,1],engine=FakeEngine([[[box,line['model_text'],.98]]]))
                _,text=core._recognize_image(image,scale=scale,recover_punctuation=True)
                self.assertEqual(text,'……味道如何？',f'{path.name} scale={scale}')


class RevisionTests(unittest.TestCase):
    frame = np.zeros((100, 200, 3), np.uint8)

    def core(self, outputs, speaker=False):
        return DialogueOCR([0, 0, 1, 1], [0, 0, .5, .3] if speaker else None,
                           engine=FakeEngine(outputs), min_confidence=.9)

    def test_first_result_is_immediate_even_when_low_quality(self):
        core = self.core([[detection("不要开门。", .55)], [detection("不要开门。", .85)]])
        first = core.process_frame(self.frame, now=0)
        self.assertEqual(first["status"], "new")
        self.assertEqual(first["event"]["status"], "candidate")
        reviewed = core.verify(first["event"]["dialogue_id"])
        self.assertEqual(reviewed["status"], "needs_review")
        self.assertEqual(reviewed["revision"], 1)
        self.assertFalse(reviewed["human_confirmed"])

    def test_verified_requires_agreement_and_both_scores(self):
        for raw_score, reviewed_text, review_score, expected in (
            (.95, "奖励 10。", .96, "verified"),
            (.89, "奖励 10。", .99, "needs_review"),
            (.99, "奖励 10。", .89, "needs_review"),
            (.99, "奖励 11。", .99, "needs_review")):
            with self.subTest(expected=expected, text=reviewed_text):
                core = self.core([[detection("奖励 10。", raw_score)], [detection(reviewed_text, review_score)]])
                first = core.process_frame(self.frame)["event"]
                revision = core.verify(first["dialogue_id"])
                self.assertEqual(revision["status"], expected)
                self.assertEqual(revision["body"], "奖励 10。")
                self.assertEqual(revision["original_text"], "奖励 10。")
                self.assertEqual(revision["review_text"], reviewed_text)
                self.assertEqual(first["dialogue_id"], revision["dialogue_id"])

    def test_review_uses_original_crop_after_new_dialogue_and_frame_mutation(self):
        seen = []
        results = iter([[detection("原句")], [detection("新句")], [detection("原句")]])
        def engine(image, **kwargs):
            seen.append(int(image[0, 0, 0]))
            return next(results), None
        core = DialogueOCR([0, 0, 1, 1], engine=engine)
        original = np.full_like(self.frame, 20)
        first = core.process_frame(original, now=0)["event"]
        original[:] = 99
        second = core.process_frame(np.full_like(self.frame, 70), now=1)["event"]
        revision = core.verify(first["dialogue_id"])
        self.assertEqual(seen, [20, 70, 20])
        self.assertEqual(revision["dialogue_id"], first["dialogue_id"])
        self.assertNotEqual(revision["dialogue_id"], second["dialogue_id"])
        self.assertIn(second["dialogue_id"], core.pending_ids)

    def test_one_character_number_or_negation_change_is_always_new(self):
        texts = ["可以。", "可以。", "不可以。", "奖励10。", "奖励11。"]
        core = self.core([[detection(text, .5)] for text in texts])
        statuses = [core.process_frame(self.frame, now=i)["status"] for i in range(len(texts))]
        self.assertEqual(statuses, ["new", "duplicate", "new", "new", "new"])

    def test_missing_or_low_quality_name_does_not_repeat_body(self):
        outputs = []
        for name, score in [("甲", .98), ("甲", .98), (None, None), ("乙", .5), ("乙", .98), ("乙", .98)]:
            outputs.extend([[detection("同一句正文。")], [detection(name, score)] if name else None])
        core = self.core(outputs, speaker=True)
        events = [core.process_frame(self.frame, now=i) for i in range(6)]
        self.assertEqual([e["status"] for e in events], ["new", "revision", "duplicate", "duplicate", "duplicate", "new"])
        self.assertEqual(events[0]["event"]["dialogue_id"], events[1]["event"]["dialogue_id"])
        self.assertNotEqual(events[0]["event"]["dialogue_id"], events[-1]["event"]["dialogue_id"])

    def test_name_review_quality_is_separate_from_body(self):
        core = self.core([[detection("正文。")], [detection("甲", .4)],
                          [detection("正文。")], [detection("乙", .7)]], speaker=True)
        event = core.process_frame(self.frame)["event"]
        revision = core.verify(event["dialogue_id"])
        self.assertEqual(revision["status"], "verified")
        self.assertEqual(revision["speaker_status"], "needs_review")

    def test_text_touching_crop_edge_is_not_verified_even_when_models_agree(self):
        core=self.core([[detection('不完整',y=90,height=10)],[detection('不完整',y=180,height=20)]])
        image=self.frame.copy()
        image[95:100,12:16]=255  # Actual outlined glyph ink touches the crop bottom.
        first=core.process_frame(image)['event']
        self.assertTrue(first['source_clipped'])
        self.assertEqual(core.verify(first['dialogue_id'])['status'],'needs_review')

    def test_cropped_capture_and_scaled_review_preserve_client_coordinates(self):
        core = DialogueOCR([.2, .6, .4, .7], engine=FakeEngine([
            [detection("正文", x=10, y=4)], [detection("正文", x=20, y=8, width=70, height=20)]]))
        crop = np.zeros((100, 240, 3), np.uint8)
        event = core.process_frame(crop, client_size=(1000, 500), origin=(180, 280))["event"]
        review = core.verify(event["dialogue_id"])
        self.assertEqual(event["body_lines"][0]["bbox"][0], [210., 304.])
        self.assertEqual(review["review_body_lines"][0]["bbox"][0], [210., 304.])
        self.assertEqual(event["frame_size"], [1000, 500])
        with self.assertRaises(ValueError):
            core.region_image(crop, core.body_roi, (1000, 500), (300, 400))

    def test_pause_does_not_turn_unavailable_time_into_blank(self):
        core = self.core([[detection("原句")], None, None, [detection("原句")]])
        core.process_frame(self.frame, now=0)
        core.process_frame(self.frame, now=.1)
        core.pause(now=1)
        core.process_frame(self.frame, now=100)
        self.assertEqual(core.process_frame(self.frame, now=100.1)["status"], "duplicate")

    def test_blank_reset_and_pause_preserve_pending_original_review(self):
        core = self.core([[detection("原句")], None, None, [detection("原句")]])
        first = core.process_frame(self.frame, now=0)["event"]
        core.process_frame(self.frame, now=.1)
        core.process_frame(self.frame, now=.9)
        second = core.process_frame(self.frame, now=1)["event"]
        self.assertNotEqual(first["dialogue_id"], second["dialogue_id"])
        core.pause(now=2)
        self.assertEqual(len(core.pending_ids), 2)

    def test_bad_frame_is_not_blank_and_backwards_time_is_rejected(self):
        core = self.core([[detection("正文")], None, RuntimeError("broken"), None, [detection("正文")]])
        core.process_frame(self.frame, now=0)
        core.process_frame(self.frame, now=.1)
        with self.assertRaises(RuntimeError):
            core.process_frame(self.frame, now=1)
        core.process_frame(self.frame, now=2)
        self.assertEqual(core.process_frame(self.frame, now=2.1)["status"], "duplicate")
        with self.assertRaises(ValueError):
            core.pause(now=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
