import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from dialogue_store import Recorder


class StoreTests(unittest.TestCase):
    def test_revision_replaces_same_dialogue_and_preserves_original_image(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/"validation") as directory:
            recorder = Recorder(directory, "test")
            first = dict(dialogue_id="test-1", event_id=1, revision=0, status="candidate", body="不要开门。",
                         original_text="不要开门。", review_text=None, speaker="甲", recognized_at="2026-10-29T00:00:00Z")
            saved = recorder.upsert(first, {"body":np.full((20,100,3),40,np.uint8)})
            revised = dict(first, revision=1, status="needs_review", review_text="不要关门。")
            final = recorder.upsert(revised)
            recorder.upsert(first, {"body":np.full((20,100,3),99,np.uint8)})
            rows = (recorder.directory/"dialogue.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows),1)
            self.assertEqual(json.loads(rows[0])["revision"],1)
            self.assertEqual(len((recorder.directory/"revisions.jsonl").read_text(encoding="utf-8").splitlines()),2)
            self.assertEqual(saved["body_image"],final["body_image"])
            image=cv2.imdecode(np.fromfile(recorder.directory/final["body_image"],np.uint8),cv2.IMREAD_COLOR)
            self.assertEqual(int(image[0,0,0]),40)
            self.assertEqual((recorder.directory/"dialogue.txt").read_text(encoding="utf-8").count("不要开门。"),1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
