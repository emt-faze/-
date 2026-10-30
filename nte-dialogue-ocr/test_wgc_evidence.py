"""Regressions for observed failures, using the preserved real game crops."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from frame_gate import FrameStability
from ocr_core import DialogueOCR, clipping_evidence, trailing_punctuation

ROOT=Path(__file__).resolve().parent
SESSION=ROOT/'records/2026-10-30/3261a68d0079'
LIVE=ROOT/'validation/wgc-dual-20261030/live-02'


def read(path):
    return cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)


@unittest.skipUnless((SESSION/'dialogue.jsonl').exists(),'Real evidence is not installed')
class WgcEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.records={r['event_id']:r for r in (json.loads(s) for s in (SESSION/'dialogue.jsonl').read_text(encoding='utf-8').splitlines())}

    def image(self,index):
        return read(SESSION/self.records[index]['body_image'])

    def raw(self,index,key='body_lines'):
        return [[((np.asarray(line['model_bbox'])-[211,869])*2).tolist(),line['model_text'],line['confidence']]
                for line in self.records[index][key]]

    def test_detector_padding_does_not_mean_cut_ink_but_cinematic_crop_does(self):
        flagged=[]
        for index,record in self.records.items():
            if clipping_evidence(self.image(index),record['body_lines'],(211,869)):
                flagged.append(index)
        self.assertEqual(flagged,[15])

    def test_visible_full_stop_keeps_same_real_page_from_becoming_a_second_record(self):
        outputs=iter((self.raw(7),self.raw(8)))
        helper=DialogueOCR([0,0,1,1],candidate_scale=2,engine=lambda *args,**kwargs:(next(outputs),None))
        first=helper.process_frame(self.image(7),now=0)
        second=helper.process_frame(self.image(8),now=1)
        self.assertTrue(first['event']['body'].endswith('。'))
        self.assertFalse(first['event']['raw_model_text'].endswith('。'))
        self.assertEqual(second['status'],'duplicate')
        self.assertIsNone(second['event'])

    def test_missing_and_partial_ellipsis_require_six_visible_dots(self):
        for index,truth in ((12,'倒是没有……'),(28,'哈哈……')):
            for key in ('body_lines','review_body_lines'):
                helper=DialogueOCR([0,0,1,1],engine=lambda *args,**kwargs:(self.raw(index,key),None))
                _,text=helper._recognize_image(self.image(index),scale=2,recover_punctuation=True)
                self.assertEqual(text,truth)
        image=self.image(3)
        poly=(np.asarray(self.records[3]['body_lines'][0]['model_bbox'])-[211,869])*2
        # A question mark is not a hollow full stop, even if OCR misses it.
        text,marks,_=trailing_punctuation(cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC),poly,'明音')
        self.assertEqual(text,'明音')
        self.assertEqual(marks,[])

    def test_missed_moving_shots_can_settle_without_waiting_for_background(self):
        for begin,end in ((546,556),(686,696)):
            gate=FrameStability(.45)
            ready=[]
            for index in range(begin,end+1):
                if gate.ready(read(LIVE/f'frames/{index:05d}.png')[62:],index*.5):
                    ready.append(index)
            self.assertTrue(ready,(begin,end))
            self.assertGreaterEqual(min(ready),begin+5)

    def test_new_local_character_is_not_hidden_by_a_long_sentence_variation_budget(self):
        gate=FrameStability(.45)
        baseline=np.zeros((80,500),bool); baseline[20:40,20:400]=True
        newer=baseline.copy(); newer[20:30,410:416]=True
        with patch('frame_gate.glyph_mask',side_effect=lambda image:image):
            for index in range(4):
                gate.ready(baseline,index*.2)
            gate.commit()
            self.assertFalse(gate.ready(newer,.8))
            self.assertFalse(gate.ready(newer,1.0))
            gate.ready(newer,1.2)
            self.assertTrue(gate.ready(newer,1.4))
            self.assertFalse(gate.unchanged)


if __name__=='__main__':
    unittest.main()
