"""Wait for rendered glyphs to settle before sending a frame to OCR."""
import cv2
import numpy as np
from collections import deque

def glyph_mask(image):
    lower=image.min(axis=2)
    white=(lower>245)&(image.max(axis=2).astype(np.int16)-lower<12)
    dark=cv2.dilate((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<140).astype(np.uint8),np.ones((5,5),np.uint8))
    ink=(white & (dark>0)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    keep=np.zeros(count,dtype=bool)
    keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=3
    return keep[labels]


def soft_glyph_mask(image):
    """A photometric envelope, without spatial dilation of the glyph itself.

    Core pixels may flicker within their existing antialiased edge without
    implying a different word. New solid ink outside this envelope is new.
    Boolean masks are also accepted for deterministic gate diagnostics.
    """
    if image.ndim==2:
        return image.astype(bool)
    lower=image.min(axis=2)
    white=(lower>205)&(image.max(axis=2).astype(np.int16)-lower<40)
    dark=cv2.dilate((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<140).astype(np.uint8),np.ones((5,5),np.uint8))
    return white & (dark>0)

class FrameStability:
    def __init__(self,seconds=.45):
        self.seconds=seconds
        self.reset()

    def reset(self):
        self.history=deque(maxlen=4)
        self.template=None
        self.current=None
        self.unchanged=False

    def ready(self,image,now):
        mask=glyph_mask(image)
        if self.history and mask.shape!=self.history[-1][1].shape:
            self.reset()
        self.history.append((now,mask,soft_glyph_mask(image)))
        self.unchanged=False
        if len(self.history)<3 or now-self.history[0][0]<self.seconds:
            return False
        stack=np.stack([item[1] for item in self.history])
        sure_ink=np.all(stack,axis=0)
        sure_background=~np.any(stack,axis=0)
        uncertain=~(sure_ink|sure_background)
        # Antialiased strokes can flicker at scattered pixels while the text
        # is complete. A global 12-pixel limit blocked long moving-shot lines.
        # Bound both overall variation and each local glyph-sized patch; this
        # still waits for a newly rendered character, even in a long sentence.
        limit=max(12,round(np.count_nonzero(sure_ink)*.02))
        local=cv2.boxFilter(uncertain.astype(np.float32),-1,(32,32),normalize=False,
                           borderType=cv2.BORDER_CONSTANT)
        if np.count_nonzero(uncertain)>limit or local.max(initial=0)>12:
            return False
        # Reuse only when solid text stays within its actual photographed soft
        # edges. Text edit distance is never used, including punctuation changes.
        stable_background=~np.any(np.stack([item[2] for item in self.history]),axis=0)
        self.current=(sure_ink,stable_background)
        if self.template is not None:
            before_ink,before_background=self.template
            # A persistent changed pixel is a new glyph pattern. There is no
            # edit-distance/text-similarity threshold, even for one letter.
            self.unchanged=not (np.any(sure_ink & before_background) or np.any(stable_background & before_ink))
        return True

    def commit(self):
        if self.current is not None and not self.unchanged:
            self.template=tuple(value.copy() for value in self.current)
