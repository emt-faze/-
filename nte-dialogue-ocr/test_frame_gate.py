import unittest
from unittest.mock import patch
import numpy as np
from frame_gate import FrameStability

class FrameGateTests(unittest.TestCase):
    def test_five_fps_reaches_ready_and_a_persistent_single_pixel_is_not_merged(self):
        gate=FrameStability(.45)
        first=np.zeros((10,30),bool); first[2:7,3:12]=True
        with patch('frame_gate.glyph_mask',side_effect=lambda image:image):
            self.assertFalse(gate.ready(first,0))
            self.assertFalse(gate.ready(first,.2))
            self.assertFalse(gate.ready(first,.4))
            self.assertTrue(gate.ready(first,.6))
            self.assertFalse(gate.unchanged)
            gate.commit()
            self.assertTrue(gate.ready(first,.8))
            self.assertTrue(gate.unchanged)
            changed=first.copy(); changed[7,5]=True
            for at in (1.,1.2,1.4,1.6):
                ready=gate.ready(changed,at)
            self.assertTrue(ready)
            self.assertFalse(gate.unchanged)

    def test_progressive_rendering_is_held_and_focus_reset_discards_prior_consensus(self):
        gate=FrameStability(.45)
        with patch('frame_gate.glyph_mask',side_effect=lambda image:image):
            for index in range(5):
                mask=np.zeros((20,100),bool); mask[4:12,:10*(index+1)]=True
                self.assertFalse(gate.ready(mask,index*.2))
            self.assertFalse(gate.ready(mask,1.0))
            gate.ready(mask,1.2)
            self.assertTrue(gate.ready(mask,1.4))
            gate.commit()
            gate.reset()
            self.assertFalse(gate.ready(mask,2.0))

if __name__=='__main__':unittest.main()
