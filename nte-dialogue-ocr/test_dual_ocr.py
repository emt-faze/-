import unittest
import numpy as np
from ocr_core import DialogueOCR


class Engine:
    def __init__(self, name, texts):
        self.model_id = name
        self.texts = iter(texts)
        self.shapes = []
    def __call__(self, image, **kwargs):
        self.shapes.append(image.shape)
        text = next(self.texts)
        return [[[[10, 8], [80, 8], [80, 30], [10, 30]], text, .98]], None


class DualOCRTests(unittest.TestCase):
    def test_medium_review_is_bound_to_same_id_and_keeps_disagreement(self):
        fast = Engine("small", ["啊鸣。"])
        medium = Engine("medium", ["啊呜。"])
        core = DialogueOCR([0, 0, 1, 1], engine=fast, review_engine=medium,
                           candidate_scale=2, ocr_profile="dual-test")
        first = core.process_frame(np.zeros((80, 180, 3), np.uint8))["event"]
        self.assertEqual(medium.shapes, [])
        reviewed = core.verify(first["dialogue_id"])
        self.assertEqual(reviewed["dialogue_id"], first["dialogue_id"])
        self.assertEqual(reviewed["body"], "啊鸣。")
        self.assertEqual(reviewed["review_text"], "啊呜。")
        self.assertEqual(reviewed["status"], "needs_review")
        self.assertEqual(reviewed["candidate_model"], "small")
        self.assertEqual(reviewed["review_model"], "medium")
        self.assertEqual(reviewed["ocr_profile"], "dual-test")
        self.assertFalse(reviewed["human_confirmed"])

    def test_review_enlarges_body_but_keeps_name_original_size(self):
        fast = Engine("small", ["请问有人在吗？", "明音凛"])
        medium = Engine("medium", ["请问有人在吗？", "明音凛"])
        core = DialogueOCR([0, .4, 1, 1], [0, 0, 1, .4], engine=fast,
                           review_engine=medium, candidate_scale=2, speaker_review_scale=1)
        first = core.process_frame(np.zeros((100, 200, 3), np.uint8))["event"]
        core.verify(first["dialogue_id"])
        self.assertEqual(fast.shapes, [(120, 400, 3), (40, 200, 3)])
        self.assertEqual(medium.shapes, [(120, 400, 3), (40, 200, 3)])


if __name__ == "__main__":
    unittest.main()
