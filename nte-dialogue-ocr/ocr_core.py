"""Local OCR and adjacent-dialogue tracking, independent of capture and UI.

Frames are uint8 BGR game-client images. All ROIs are normalized
``(left, top, right, bottom)`` coordinates relative to that image.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
import time
import unicodedata
import uuid
from typing import Any, Sequence

import numpy as np


DEFAULT_ENGINE_OPTIONS = {
    "intra_op_num_threads": 2,
    "inter_op_num_threads": 1,
    "det_limit_type": "max",
    "det_limit_side_len": 1280,
    "use_cls": False,
    "width_height_ratio": -1,
    # Keep the review image at its real 2x size. Detection still has its own
    # 1280px limit; the recognizer crops from this larger source image.
    "max_side_len": 8192,
}

OCR_PROFILE = "dialogue-v5-rapid6-dual"


def validate_roi(roi: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate normalized left/top/right/bottom; never silently clip settings."""
    if isinstance(roi, (str, bytes)):
        raise ValueError("ROI must contain four normalized numbers.")
    try:
        values = list(roi)
    except TypeError as exc:
        raise ValueError("ROI must contain four normalized numbers.") from exc
    if len(values) != 4 or any(isinstance(v, (bool, str, bytes)) for v in values):
        raise ValueError("ROI must contain four normalized numbers.")
    try:
        left, top, right, bottom = (float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROI must contain four normalized numbers.") from exc
    if not all(math.isfinite(v) for v in (left, top, right, bottom)):
        raise ValueError("ROI coordinates must be finite.")
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("ROI must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1.")
    return left, top, right, bottom


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image.")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("Expected a non-empty H x W x 3 BGR image.")


def roi_to_pixels(shape: Sequence[int], roi: Sequence[float]) -> tuple[int, int, int, int]:
    """Return a half-open pixel rectangle, rounding outwards to retain edge ink."""
    left, top, right, bottom = validate_roi(roi)
    if len(shape) < 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("Image dimensions must be positive.")
    height, width = int(shape[0]), int(shape[1])
    return (
        max(0, math.floor(left * width)),
        max(0, math.floor(top * height)),
        min(width, math.ceil(right * width)),
        min(height, math.ceil(bottom * height)),
    )


def crop_roi(frame: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    """Return an independent contiguous BGR crop; the input is never mutated."""
    _validate_frame(frame)
    left, top, right, bottom = roi_to_pixels(frame.shape, roi)
    return np.array(frame[top:bottom, left:right], copy=True, order="C")


def _build_engine(options: dict[str, Any]):
    import cv2
    cv2.setNumThreads(1)
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(**options)


def leading_ellipsis(image, polygon):
    """Recover only a visually evidenced run of 3/6 outlined white dots.

    Chinese UI ellipsis often falls outside the detector's word box. Require
    small neutral-white circular components, dark outlines, a common baseline
    and regular spacing immediately before the detected text. No word, number
    or name substitutions are performed.
    """
    import cv2
    points = np.asarray(polygon, dtype=float)
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    height = bottom-top
    if height < 8:
        return "", []
    x0=max(0,int(left-2.7*height))
    # The detector can include one, several or none of the dots. Inspect the
    # leading part inside its box too, so the last dot is not clipped.
    x1=min(image.shape[1],int(left+2.1*height))
    y0=max(0,int(top))
    y1=min(image.shape[0],int(bottom))
    if x1-x0 < height or y1-y0 < 5:
        return "", []
    patch=image[y0:y1,x0:x1]
    channel_min=patch.min(axis=2)
    white=((channel_min>205)&(patch.max(axis=2).astype(int)-channel_min<40)).astype(np.uint8)
    count, labels, stats, centers=cv2.connectedComponentsWithStats(white,8)
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    dots=[]
    for stat,center in zip(stats[1:],centers[1:]):
        x,y,w,h,area=stat
        if not (.045*height<=w<=.24*height and .045*height<=h<=.24*height and .6<=w/h<=1.7 and area>=.4*w*h):
            continue
        if abs(center[1]-(top+bottom)/2+y0) > .18*height:
            continue
        pad=max(1,round(.055*height))
        surrounding=gray[max(0,y-pad):min(gray.shape[0],y+h+pad),max(0,x-pad):min(gray.shape[1],x+w+pad)]
        # Cubic enlargement softens the thin dark outline. Check a dark-gray
        # ring rather than requiring near-black pixels to survive resizing.
        if (surrounding<140).mean()<.20:
            continue
        dots.append((float(center[0]+x0),float(center[1]+y0),[int(x+x0),int(y+y0),int(x+x0+w),int(y+y0+h)]))
    dots.sort()
    for n in (6,3):
        if len(dots) != n:
            continue
        group=dots[-n:]
        xs=np.array([dot[0] for dot in group])
        ys=np.array([dot[1] for dot in group])
        gaps=np.diff(xs)
        if (np.ptp(ys)<=.12*height and .12*height<=gaps.mean()<=.5*height
                and gaps.std()/gaps.mean()<=.22 and -2.1*height<=left-xs[-1]<=.6*height
                and xs[0]<=left+.25*height):
            return "…"*(n//3),[dot[2] for dot in group]
    return "", []


def standalone_ellipsis(image):
    """Find one isolated, centered run of 3/6 outlined white dots in a body ROI.

    Work from original pixels, independent of the OCR detector and its scale.
    Geometry is evidence, not a model confidence score or a second model vote.
    """
    import cv2
    height, width = image.shape[:2]
    if width < 100 or height < 8:
        return "", []
    lower = image.min(axis=2)
    white = ((lower > 200) & (image.max(axis=2).astype(int)-lower < 40)).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(white, 8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dots = []
    for (x, y, w, h, area), (cx, cy) in zip(stats[1:], centers[1:]):
        if not (1 <= w <= width*.006 and 1 <= h <= width*.006
                and 1/3 <= w/h <= 3 and area >= .5*w*h):
            continue
        pad = max(1, round(width*.0012))
        if x < pad or y < pad or x+w+pad > width or y+h+pad > height:
            continue
        ring = gray[y-pad:y+h+pad, x-pad:x+w+pad]
        if (ring < 140).mean() < .25:
            continue
        dots.append((float(cx), float(cy), [int(x), int(y), int(x+w), int(y+h)]))
    candidates = {}
    for _, baseline, _ in dots:
        aligned = sorted(d for d in dots if abs(d[1]-baseline) <= max(1, width*.001))
        groups = []
        for dot in aligned:
            if not groups or dot[0]-groups[-1][-1][0] > width*.012:
                groups.append([dot])
            else:
                groups[-1].append(dot)
        for group in groups:
            if len(group) not in (3, 6):
                continue
            xs = np.array([d[0] for d in group]); ys = np.array([d[1] for d in group])
            gaps = np.diff(xs)
            boxes = [d[2] for d in group]
            sizes = np.array([[b[2]-b[0], b[3]-b[1]] for b in boxes])
            if not (abs((xs[0]+xs[-1])/2-width/2) <= width*.08
                    and width*.0025 <= gaps.mean() <= width*.012
                    and gaps.std()/gaps.mean() <= .18
                    and np.ptp(ys) <= max(1, gaps.mean()*.15)
                    and np.all(sizes.max(axis=0) <= 2*sizes.min(axis=0)+1)
                    and sizes[:, 0].max() < gaps.min()*.65):
                continue
            # A detached punctuation run beside missed words must not replace
            # those words. Reject other white ink across the same text band.
            y0 = max(0, int(ys.min()-2*gaps.mean()))
            y1 = min(height, int(np.ceil(ys.max()+2*gaps.mean()))+1)
            band = white[y0:y1].copy()
            for x, y, right, bottom in boxes:
                band[max(0,y-y0):bottom-y0, x:right] = 0
            if int(band.sum()) > 2:
                continue
            candidates[tuple(tuple(b) for b in boxes)] = boxes
    if len(candidates) != 1:
        return "", []
    boxes = next(iter(candidates.values()))
    return "…"*(len(boxes)//3), boxes


def internal_ellipsis(image, polygon, text):
    """Correct an existing dot token only when its own image has 3/6 dots."""
    import cv2
    tokens=list(re.finditer(r'[.·…]+',text))
    if len(tokens)!=1:
        return text, []
    points=np.asarray(polygon,dtype=float)
    left,top=points.min(axis=0); right,bottom=points.max(axis=0)
    height=bottom-top
    if height<8:
        return text,[]
    x0=max(0,int(left-2.7*height)); x1=min(image.shape[1],int(right+2.7*height))
    y0=max(0,int(top)); y1=min(image.shape[0],int(bottom))
    patch=image[y0:y1,x0:x1]
    lower=patch.min(axis=2)
    white=((lower>205)&(patch.max(axis=2).astype(int)-lower<40)).astype(np.uint8)
    _,_,stats,centers=cv2.connectedComponentsWithStats(white,8)
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    dots=[]
    for stat,center in zip(stats[1:],centers[1:]):
        x,y,w,h,area=stat
        if not (.045*height<=w<=.24*height and .045*height<=h<=.24*height and .6<=w/h<=1.7 and area>=.4*w*h):
            continue
        if abs(center[1]+y0-(top+bottom)/2)>.18*height:
            continue
        pad=max(1,round(.055*height))
        ring=gray[max(0,y-pad):min(gray.shape[0],y+h+pad),max(0,x-pad):min(gray.shape[1],x+w+pad)]
        if (ring<140).mean()<.20:
            continue
        dots.append((center[0]+x0,center[1]+y0,[int(x+x0),int(y+y0),int(x+x0+w),int(y+y0+h)]))
    dots.sort()
    groups=[]
    for dot in dots:
        if not groups or dot[0]-groups[-1][-1][0]>.5*height:
            groups.append([dot])
        else:
            groups[-1].append(dot)
    runs=[]
    for group in groups:
        if len(group) not in (3,6):
            continue
        xs=np.array([d[0] for d in group]); ys=np.array([d[1] for d in group]); gaps=np.diff(xs)
        if np.ptp(ys)<=.12*height and .12*height<=gaps.mean()<=.5*height and gaps.std()/gaps.mean()<=.22:
            runs.append(group)
    if len(runs)!=1:
        return text,[]
    match=tokens[0]
    # Reject a nearby background run at a different horizontal location from
    # the model's punctuation. Punctuation occupies approximately two glyphs.
    expected=(match.start()+1)/(len(text)-(match.end()-match.start())+2)
    actual=(np.mean([d[0] for d in runs[0]])-left)/max(1,right-left)
    if abs(expected-actual)>.18:
        return text,[]
    return text[:match.start()]+('…'*(len(runs[0])//3))+text[match.end():],[d[2] for d in runs[0]]


def _text_key(text: str) -> str:
    # Preserve punctuation, digits, case and Chinese characters. A single-letter
    # or negation change is a different dialogue, regardless of similarity.
    value = " ".join(unicodedata.normalize("NFC", text).split())
    # CJK layout and detector fragments can insert spaces beside an ellipsis,
    # quote or wrapped character. These are not changes to the spoken text.
    # Keep spaces between Latin words so distinct word boundaries stay distinct.
    cjk = r"\u3400-\u9fff\uf900-\ufaff，。！？；：、…「」『』“”‘’（）【】《》〈〉"
    return re.sub(fr"(?<=[{cjk}]) | (?=[{cjk}])", "", value)


def trailing_punctuation(image, polygon, text):
    """Recover a visible outlined dot run or hollow full stop after the text.

    Only pixel geometry is used. No word/name/digit corrections or text-similarity
    deduplication are involved. A period requires a closed white ring with a hole.
    """
    import cv2
    points=np.asarray(polygon,dtype=float)
    left,top=points.min(axis=0); right,bottom=points.max(axis=0)
    height=bottom-top
    token=re.search(r'[.·…]+$',text)
    if height<8 or not text or (not token and not text[-1].isalnum()):
        return text,[],None
    x0=max(0,int(right-2.5*height)); x1=min(image.shape[1],int(right+2.2*height))
    y0=max(0,int(top)); y1=min(image.shape[0],int(np.ceil(bottom)))
    patch=image[y0:y1,x0:x1]
    if not patch.size:
        return text,[],None
    lower=patch.min(axis=2)
    white=((lower>205)&(patch.max(axis=2).astype(int)-lower<40)).astype(np.uint8)
    _,labels,stats,centers=cv2.connectedComponentsWithStats(white,8)
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    dots=[]; stops=[]
    for label,(stat,center) in enumerate(zip(stats[1:],centers[1:]),1):
        x,y,w,h,area=stat
        cx,cy=center+[x0,y0]
        pad=max(1,round(.055*height))
        ring=gray[max(0,y-pad):min(gray.shape[0],y+h+pad),max(0,x-pad):min(gray.shape[1],x+w+pad)]
        if (ring<140).mean()<.2:
            continue
        box=[int(x+x0),int(y+y0),int(x+x0+w),int(y+y0+h)]
        if (.045*height<=w<=.24*height and .045*height<=h<=.24*height and
            .6<=w/h<=1.7 and area>=.4*w*h and abs(cy-(top+bottom)/2)<=.18*height):
            dots.append((float(cx),float(cy),box))
        if (not token and .12*height<=w<=.36*height and .12*height<=h<=.36*height and
            .7<=w/h<=1.4 and .65<=(cy-top)/height<=.98 and right-.25*height<=cx<=right+.7*height):
            component=(labels[y:y+h,x:x+w]==label).astype(np.uint8)
            contours,hierarchy=cv2.findContours(component,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None and any(parent>=0 and cv2.contourArea(contours[i])>=1
                                            for i,(_,_,_,parent) in enumerate(hierarchy[0])):
                stops.append(box)
    groups=[]
    for dot in sorted(dots):
        if not groups or dot[0]-groups[-1][-1][0]>.5*height:
            groups.append([dot])
        else:
            groups[-1].append(dot)
    eligible=[]
    for group in groups:
        if len(group) not in (3,6):
            continue
        xs=np.array([d[0] for d in group]); ys=np.array([d[1] for d in group]); gaps=np.diff(xs)
        if (np.ptp(ys)<=.12*height and .12*height<=gaps.mean()<=.5*height and gaps.std()/gaps.mean()<=.22
            and xs[0]>=right-(1.6 if token else .25)*height and xs[-1]>=right-.35*height
            and xs[0]<=right+.6*height):
            eligible.append(group)
    if len(eligible)==1:
        group=eligible[0]
        return (text[:token.start()] if token else text)+('…'*(len(group)//3)),[d[2] for d in group],'outlined_ellipsis_dots'
    if len(stops)==1:
        return text+'。',stops,'outlined_full_stop'
    return text,[],None


def clipping_evidence(image, lines, offset=(0,0)):
    """Flag visible glyph ink at a crop edge, not detector padding alone."""
    from frame_gate import glyph_mask
    mask=glyph_mask(image)
    height,width=mask.shape
    evidence=[]
    for line in lines:
        points=np.asarray(line['bbox'])-np.asarray(offset)
        x0,y0=np.maximum(0,np.floor(points.min(axis=0)).astype(int))
        x1,y1=np.minimum([width,height],np.ceil(points.max(axis=0)).astype(int)+1)
        if x1<=x0 or y1<=y0:
            continue
        for edge,touches,ink in (
            ('top',y0<=1,mask[:2,x0:x1]),('bottom',y1>=height-1,mask[-2:,x0:x1]),
            ('left',x0<=1,mask[y0:y1,:2]),('right',x1>=width-1,mask[y0:y1,-2:])):
            if touches and int(ink.sum())>=3:
                evidence.append(dict(edge=edge,ink_pixels=int(ink.sum())))
    return evidence


def uncovered_edge_ink(image,polygon,other_polygons=()):
    """Find nearby outlined strokes outside a detector's horizontal envelope.

    A detached one-stroke character can fall outside both detectors. This only
    locates pixels; their linguistic identity is read by the recognizer.
    """
    import cv2
    from frame_gate import glyph_mask
    points=np.asarray(polygon,dtype=float)
    left,top=points.min(axis=0); right,bottom=points.max(axis=0)
    height=bottom-top
    if height<8:
        return []
    x0=max(0,int(left-height)); x1=min(image.shape[1],int(np.ceil(right+height)))
    y0=max(0,int(top)); y1=min(image.shape[0],int(np.ceil(bottom)))
    patch=image[y0:y1,x0:x1]
    if not patch.size:
        return []
    _,_,stats,_=cv2.connectedComponentsWithStats(glyph_mask(patch).astype(np.uint8),8)
    found=[]
    for x,y,w,h,area in stats[1:]:
        a,b=x+x0,y+y0
        if area<6 or w>1.5*height or h>height:
            continue
        # Require a nearby component, not a distant scene highlight. Components
        # crossing the box edge count too: the missing stroke may be half cut.
        before=a<left-2 and a+w>=left-.55*height and a+w<left+.5*height
        after=a+w>right+2 and a<=right+.55*height and a>right-.5*height
        if before or after:
            center=np.array([a+w/2,b+h/2])
            if any(np.all(center>=np.asarray(other).min(axis=0)) and np.all(center<=np.asarray(other).max(axis=0))
                   for other in other_polygons):
                # A detector-split neighboring word is already represented by
                # its own box. It must not be read twice by an expanded crop.
                continue
            found.append([int(a),int(b),int(a+w),int(b+h)])
    return found


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"


def _join_fragments(parts: list[str]) -> str:
    value = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not value:
            value = part
        elif (_is_cjk(value[-1]) and _is_cjk(part[0])) or (value[-1] in "，。！？；：、…" and _is_cjk(part[0])) or part[0] in "，。！？；：、,.!?;:)]}）】》」』”’" or value[-1] in "([{（【《「『“‘":
            value += part
        else:
            value += " " + part
    return value


def _arrange_lines(lines: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Group mildly uneven boxes on the same baseline, then order left to right."""
    rows: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda item: (item["_cy"], item["_x"])):
        best_row = None
        best_distance = math.inf
        for row in rows:
            row_center = float(np.median([item["_cy"] for item in row]))
            row_height = float(np.median([item["_height"] for item in row]))
            distance = abs(line["_cy"] - row_center)
            if distance <= 0.5 * min(line["_height"], row_height) and distance < best_distance:
                best_row, best_distance = row, distance
        if best_row is None:
            rows.append([line])
        else:
            best_row.append(line)
    rows.sort(key=lambda row: float(np.median([item["_cy"] for item in row])))
    ordered, text_rows = [], []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: item["_x"])
        text_rows.append(_join_fragments([item["text"] for item in row]))
        for item in row:
            clean = {key: value for key, value in item.items() if not key.startswith("_")}
            clean["row"] = row_index
            ordered.append(clean)
    return ordered, "\n".join(text_rows)


def model_text_from_lines(lines):
    rows = {}
    for line in lines:
        rows.setdefault(line["row"], []).append(line.get("model_text", line["text"]))
    return "\n".join(_join_fragments(parts) for parts in rows.values())


class DialogueOCR:
    """Immediate candidates followed by review of the same immutable source crop.

    Body identity uses exact text, never fuzzy similarity. Speaker identity is
    independently stabilized across fresh observations. Verification is an
    explicit operation so the caller can display/save a candidate first.
    """
    def __init__(self, body_roi, speaker_roi=None, min_confidence=.90,
                 blank_reset_seconds=.8, engine=None, engine_options=None, verify_scale=2.0,
                 centered_body=False, review_engine=None, speaker_review_scale=1.0,
                 ocr_profile="dialogue-v5-legacy4", candidate_scale=1.0):
        self.body_roi = validate_roi(body_roi)
        self.speaker_roi = validate_roi(speaker_roi) if speaker_roi is not None else None
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not math.isfinite(blank_reset_seconds) or blank_reset_seconds < .8:
            raise ValueError("blank_reset_seconds must be at least .8 seconds")
        if not 1 < verify_scale <= 4:
            raise ValueError("verify_scale must be in (1, 4]")
        options = dict(DEFAULT_ENGINE_OPTIONS)
        options.update(engine_options or {})
        self.engine = engine if engine is not None else _build_engine(options)
        self.review_engine = review_engine if review_engine is not None else self.engine
        if not 1 <= candidate_scale <= 4:
            raise ValueError("正文缩放必须在 1 到 4 之间")
        self.candidate_scale = float(candidate_scale)
        if not 1 <= speaker_review_scale <= 4:
            raise ValueError("人名复核缩放必须在 1 到 4 之间")
        self.speaker_review_scale = float(speaker_review_scale)
        self.ocr_profile = ocr_profile
        self.candidate_model = getattr(self.engine, "model_id", "RapidOCR-1.4.4/PP-OCRv4")
        self.review_model = getattr(self.review_engine, "model_id", self.candidate_model)
        self.min_confidence = float(min_confidence)
        self.blank_reset_seconds = float(blank_reset_seconds)
        self.verify_scale = float(verify_scale)
        self.centered_body = bool(centered_body)
        self.session_id = uuid.uuid4().hex[:12]
        self._sequence = 0
        self._last_body = None
        self._last_id = None
        self._last_now = None
        self._blank_since = None
        self._stable_speaker = ""
        self._speaker_pending = ""
        self._speaker_count = 0
        self._records = {}
        self._pending = {}
        self._body_cache = None

    def _time(self, now):
        value = time.monotonic() if now is None else float(now)
        if not math.isfinite(value) or (self._last_now is not None and value < self._last_now):
            raise ValueError("now must be finite and must not move backwards")
        self._last_now = value
        return value

    def pause(self, now=None, reason="capture_unavailable"):
        self._time(now)
        self._blank_since = None
        self._speaker_pending, self._speaker_count = "", 0
        return {"status": "paused", "reason": reason, "observation": None, "event": None}

    @property
    def pending_ids(self):
        return tuple(self._pending)

    def region_image(self, frame, roi, client_size=None, origin=(0, 0)):
        _validate_frame(frame)
        width, height = client_size or (frame.shape[1], frame.shape[0])
        left, top, right, bottom = roi_to_pixels((height, width), roi)
        ox, oy = origin
        if not (0 <= left-ox < right-ox <= frame.shape[1] and 0 <= top-oy < bottom-oy <= frame.shape[0]):
            raise ValueError("Captured crop does not contain the calibrated text region")
        return frame[top-oy:bottom-oy, left-ox:right-ox].copy(), (left, top)

    def _recognize_image(self, image, offset=(0, 0), scale=1.0, recover_punctuation=False, centered=False, engine=None,
                         speaker_style=False):
        import cv2
        scaled = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result, _ = (engine if engine is not None else self.engine)(scaled, use_cls=False)
        lines = []
        for detected in result or []:
            if not isinstance(detected, (tuple, list)) or len(detected) < 3:
                raise ValueError("OCR returned an unexpected detection record")
            polygon, text, score = detected[:3]
            if not isinstance(text, str) or not text.strip():
                continue
            points = np.asarray(polygon, dtype=float)
            confidence = float(score)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                raise ValueError("OCR returned an invalid text polygon")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("OCR returned an invalid confidence")
            if centered or speaker_style:
                tilt=abs(points[1,1]-points[0,1])
                horizontal=max(1,abs(points[1,0]-points[0,0]))
                x0,y0=np.maximum(0,points.min(axis=0).astype(int))
                x1,y1=np.minimum([scaled.shape[1],scaled.shape[0]],np.ceil(points.max(axis=0)).astype(int))
                patch=scaled[y0:y1,x0:x1]
                # Tiny two-character boxes can have a slanted detector envelope
                # even though the subtitle glyphs are horizontal (e.g. "已。").
                if not patch.size or tilt>max(.12*horizontal,.25*(y1-y0)):
                    continue
                if speaker_style:
                    blue,green,red=patch.astype(np.int16).transpose(2,0,1)
                    warm=(red>190)&(green>100)&(red-green>15)&(green-blue>15)
                    # A background sign's loose box can overlap the top of the
                    # name below it. Require name-colored ink in its own middle,
                    # not just a few orange pixels borrowed at the box edge.
                    middle=warm[len(warm)//4:max(len(warm)//4+1,3*len(warm)//4)]
                    if (warm.mean()<.025 or middle.mean()<.025 or x1-x0<.65*(y1-y0)
                            or abs((x0+x1)/2-scaled.shape[1]/2)>scaled.shape[1]*.18):
                        continue
                else:
                    white_ratio=float((patch.min(axis=2)>205).mean())
                    if not .025<=white_ratio<=.65:
                        continue
            model_text = text
            evidence = []
            model_points = points / scale + np.asarray(offset, dtype=float)
            name_check = None
            if speaker_style:
                # A loose name box may include dark background lettering. Read
                # the complete name-colored span when it is substantially
                # narrower than that box; never infer a name from the library.
                active_engine = engine if engine is not None else self.engine
                reader = getattr(active_engine, 'recognize_line', None)
                if reader is not None:
                    ys,xs = np.nonzero(warm)
                    pad = max(2, round((int(ys.max()-ys.min())+1)*.1))
                    left=max(0,int(xs.min())-pad); right=min(patch.shape[1],int(xs.max())+pad+1)
                    top=max(0,int(ys.min())-pad); bottom=min(patch.shape[0],int(ys.max())+pad+1)
                    if right-left < .85*patch.shape[1]:
                        name_check = dict(accepted=False, crop_box=[
                            (x0+left)/scale+offset[0], (y0+top)/scale+offset[1],
                            (x0+right)/scale+offset[0], (y0+bottom)/scale+offset[1]])
                        try:
                            name, name_score = reader(patch[top:bottom,left:right].copy())
                            name_check.update(text=name,confidence=name_score)
                            original, expanded = _text_key(text), _text_key(name)
                            if (math.isfinite(name_score) and self.min_confidence<=name_score<=1
                                    and len(expanded)>len(original) and original in expanded):
                                text=name
                                confidence=min(confidence,name_score)
                                points=np.array([[x0+left,y0+top],[x0+right,y0+top],
                                                 [x0+right,y0+bottom],[x0+left,y0+bottom]],dtype=float)
                                name_check['accepted']=True
                        except Exception as exc:
                            name_check['error']=str(exc)
            if recover_punctuation:
                corrected,dots=internal_ellipsis(scaled,points,text)
                if dots:
                    text=corrected
                    evidence=[{"kind":"outlined_ellipsis_dots","dot_boxes":[
                        [box[0]/scale+offset[0],box[1]/scale+offset[1],box[2]/scale+offset[0],box[3]/scale+offset[1]] for box in dots]}]
            if recover_punctuation and not text.lstrip().startswith(("…", "。")):
                prefix, dots = leading_ellipsis(scaled, points)
                if prefix:
                    text = prefix + text.lstrip(" .·")
                    evidence = [{"kind": "outlined_ellipsis_dots", "dot_boxes": [
                        [box[0]/scale+offset[0], box[1]/scale+offset[1],
                         box[2]/scale+offset[0], box[3]/scale+offset[1]] for box in dots]}]
                    left = min(box[0] for box in dots)
                    points = points.copy()
                    points[0, 0] = min(points[0, 0], left)
                    points[3, 0] = min(points[3, 0], left)
            if recover_punctuation:
                corrected,marks,kind=trailing_punctuation(scaled,points,text)
                if marks:
                    text=corrected
                    evidence.append({'kind':kind,'dot_boxes':[
                        [box[0]/scale+offset[0],box[1]/scale+offset[1],box[2]/scale+offset[0],box[3]/scale+offset[1]] for box in marks]})
                    points=points.copy()
                    points[1,0]=max(points[1,0],max(box[2] for box in marks))
                    points[2,0]=max(points[2,0],max(box[2] for box in marks))
            coverage_warning=False
            line_check=None
            if recover_punctuation:
                # Inspect original pixels: cubic enlargement changes thin-line
                # brightness and the apparent outline thickness.
                uncovered=[[int(np.floor(b[0]*scale)),int(np.floor(b[1]*scale)),
                            int(np.ceil(b[2]*scale)),int(np.ceil(b[3]*scale))]
                           for b in uncovered_edge_ink(image,points/scale,
                               [np.asarray(other[0],dtype=float)/scale for other in result if other is not detected])]
                if uncovered:
                    coverage_warning=True
                    active_engine=engine if engine is not None else self.engine
                    reader=getattr(active_engine,'recognize_line',None)
                    edge_boxes=[[b[0]/scale+offset[0],b[1]/scale+offset[1],b[2]/scale+offset[0],b[3]/scale+offset[1]] for b in uncovered]
                    line_check={'uncovered_ink_boxes':edge_boxes,'accepted':False}
                    if reader is not None:
                        pad=max(2,round(np.ptp(points[:,1])*.06))
                        lo=np.floor(points.min(axis=0)).astype(int); hi=np.ceil(points.max(axis=0)).astype(int)
                        x0=max(0,min(lo[0],min(b[0] for b in uncovered))-pad)
                        x1=min(scaled.shape[1],max(hi[0],max(b[2] for b in uncovered))+pad)
                        y0=max(0,lo[1]-pad); y1=min(scaled.shape[0],hi[1]+pad)
                        try:
                            line_text,line_score=reader(scaled[y0:y1,x0:x1].copy())
                            line_check.update(text=line_text,confidence=line_score,
                                crop_box=[x0/scale+offset[0],y0/scale+offset[1],x1/scale+offset[0],y1/scale+offset[1]])
                            # If the line reader still omits the extra ink, retain
                            # the candidate and uncertainty instead of guessing a word.
                            existing_key, expanded_key = _text_key(text), _text_key(line_text)
                            # The recognizer can read "一" even when the box
                            # starts halfway along its stroke. Identical text is
                            # evidence only for this overlapping, horizontal
                            # first stroke; detached ink may still be a lost word.
                            existing_edge = (expanded_key==existing_key and text.lstrip().startswith('一') and all(
                                b[0]<points[:,0].min()<b[2]
                                and b[2]-b[0]>=3*(b[3]-b[1])
                                and b[2]<=points[:,0].min()+.7*np.ptp(points[:,1]) for b in uncovered))
                            if (math.isfinite(line_score) and self.min_confidence <= line_score <= 1
                                    and ((len(expanded_key)>len(existing_key) and existing_key in expanded_key)
                                         or existing_edge)):
                                text=line_text
                                confidence=min(confidence,line_score)
                                points=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],dtype=float)
                                coverage_warning=False
                                line_check['accepted']=True
                                if existing_edge:
                                    line_check['reason']='detector_edge_overlaps_recognized_glyph'
                        except Exception as exc:
                            line_check['error']=str(exc)
            points = points / scale + np.asarray(offset, dtype=float)
            lines.append({"text": text, "confidence": confidence, "bbox": points.tolist(),
                          "model_text": model_text, "model_bbox": model_points.tolist(),
                          "punctuation_evidence": evidence,
                          "speaker_recognition_check":name_check,
                          "coverage_warning":coverage_warning,"line_recognition_check":line_check,
                          "_cy": float(points[:, 1].mean()), "_x": float(points[:, 0].min()),
                          "_height": max(1., float(np.ptp(points[:, 1])))})
        ordered, joined = _arrange_lines(lines)
        if centered:
            # Evaluate the whole row so a detector-split prefix is retained.
            # Horizontal centering applies to the game text row, not each box.
            rows={}
            for line in ordered:
                rows.setdefault(line['row'],[]).append(line)
            centered_rows=[]
            expected=offset[0]+image.shape[1]/2
            for row in rows.values():
                left=min(p[0] for line in row for p in line['bbox'])
                right=max(p[0] for line in row for p in line['bbox'])
                if abs((left+right)/2-expected)<=image.shape[1]*.08:
                    centered_rows.append(row)
            words=[line for row in centered_rows for line in row if any(c.isalnum() for c in line['text'])]
            text_bottom=max((p[1] for line in words for p in line['bbox']),default=math.inf)
            text_height=float(np.median([np.ptp(np.asarray(line['bbox'])[:,1]) for line in words])) if words else 0.
            kept=[]; texts=[]
            for row in centered_rows:
                text=_join_fragments([line['text'] for line in row])
                points=np.asarray([p for line in row for p in line['bbox']])
                top=float(points[:,1].min())
                # The continue diamond at the bottom of the dialogue area can
                # be read as one low-confidence dot. It is separated from the
                # subtitle by a large blank band; inline punctuation, adjacent
                # wrapped rows and real ellipses must retain their evidence.
                if (text in {'.','·','•','。','◇','◆','◈'}
                        and top>=offset[1]+image.shape[0]*.65
                        and top-text_bottom>1.2*text_height
                        and np.ptp(points[:,0])<=1.5*text_height):
                    continue
                kept.extend(row)
                texts.append(text)
            joined = '\n'.join(texts)
            if recover_punctuation and (not joined or re.fullmatch(r'[\s.·…。]+', joined)):
                punctuation, boxes = standalone_ellipsis(image)
                if punctuation:
                    absolute = [[b[0]+offset[0], b[1]+offset[1], b[2]+offset[0], b[3]+offset[1]] for b in boxes]
                    left = min(b[0] for b in absolute); right = max(b[2] for b in absolute)
                    top = min(b[1] for b in absolute); bottom = max(b[3] for b in absolute)
                    return [dict(text=punctuation, confidence=0., confidence_source='pixel_geometry',
                                 bbox=[[left,top],[right,top],[right,bottom],[left,bottom]], row=0,
                                 model_text=' '.join(str(item[1]) for item in result or []), model_bbox=[],
                                 punctuation_evidence=[dict(kind='standalone_outlined_ellipsis', dot_boxes=absolute)],
                                 coverage_warning=False, line_recognition_check=None)], punctuation
            return kept, joined
        return ordered, joined

    def _observe_speaker(self, text, confidence):
        key = _text_key(text)
        if not key or confidence is None or confidence < self.min_confidence:
            self._speaker_pending, self._speaker_count = "", 0
            return ""
        if key == self._stable_speaker:
            self._speaker_pending, self._speaker_count = "", 0
            return key
        self._speaker_count = self._speaker_count+1 if key == self._speaker_pending else 1
        self._speaker_pending = key
        if self._speaker_count >= 2:
            self._stable_speaker = key
            self._speaker_pending, self._speaker_count = "", 0
            return key
        return ""

    def process_frame(self, frame_bgr, now=None, *, client_size=None, origin=(0, 0), captured_at=None, reuse_body=False):
        if frame_bgr is None:
            return self.pause(now)
        current = self._time(now)
        began = time.perf_counter()
        try:
            body_image, body_offset = self.region_image(frame_bgr, self.body_roi, client_size, origin)
            cache_key=(body_image.shape,body_offset)
            if reuse_body and self._body_cache and self._body_cache[0]==cache_key:
                body_lines,body=self._body_cache[1:]
            else:
                body_lines, body = self._recognize_image(body_image, body_offset, self.candidate_scale, recover_punctuation=True,
                                                       centered=self.centered_body)
                self._body_cache=(cache_key,body_lines,body)
            speaker_image, speaker_offset = None, (0, 0)
            if self.speaker_roi is not None and body:
                speaker_image, speaker_offset = self.region_image(frame_bgr, self.speaker_roi, client_size, origin)
                speaker_lines, speaker = self._recognize_image(speaker_image, speaker_offset,
                                                               speaker_style=self.centered_body)
            else:
                speaker_lines, speaker = [], ""
        except Exception:
            self._blank_since = None
            self._speaker_pending, self._speaker_count = "", 0
            raise
        body_score = min((line["confidence"] for line in body_lines), default=0.)
        clip_evidence=clipping_evidence(body_image,body_lines,body_offset)
        source_clipped=bool(clip_evidence)
        speaker_score = min((line["confidence"] for line in speaker_lines), default=None)
        observed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        observation = {"body": body, "speaker": speaker, "body_lines": body_lines, "speaker_lines": speaker_lines,
                       "raw_model_text": model_text_from_lines(body_lines), "ocr_profile": self.ocr_profile,
                       "candidate_model": self.candidate_model, "review_model": self.review_model,
                       "candidate_scale": self.candidate_scale,
                       "body_pixels_unchanged": bool(reuse_body),
                       "source_clipped": source_clipped,
                       "clipping_evidence": clip_evidence,
                       "coverage_warning":any(line.get('coverage_warning',False) for line in body_lines),
                       "body_confidence": body_score, "speaker_confidence": speaker_score,
                       "confidence": body_score, "recognized_at": observed_at, "monotonic_seconds": current,
                       "captured_monotonic": captured_at,
                       "frame_size": list(client_size or (frame_bgr.shape[1], frame_bgr.shape[0])),
                       "crop_origin": list(origin), "body_roi": list(self.body_roi),
                       "speaker_roi": list(self.speaker_roi) if self.speaker_roi else None,
                       "processing_ms": round((time.perf_counter()-began)*1000, 3)}
        if not body:
            self._speaker_pending, self._speaker_count = "", 0
            if self._blank_since is None:
                self._blank_since = current
            if current-self._blank_since >= self.blank_reset_seconds-1e-9:
                self._last_body, self._last_id = None, None
            return {"status": "blank", "observation": observation, "event": None}
        self._blank_since = None
        identity = self._observe_speaker(speaker, speaker_score)
        body_key = _text_key(body)
        previous = self._records.get(self._last_id)
        changed_speaker = bool(previous and identity and previous.get("speaker_identity")
                               and identity != previous["speaker_identity"])
        if body_key == self._last_body and not changed_speaker:
            if previous and identity and not previous.get("speaker_identity"):
                revised = dict(previous, speaker=identity, speaker_identity=identity,
                               speaker_identity_confirmed=True, revision=previous["revision"]+1,
                               revision_reason="speaker_stabilized", revised_at=observed_at,
                               speaker_confirmation=observation["speaker_lines"])
                self._records[self._last_id] = revised
                return {"status": "revision", "observation": observation, "event": revised}
            return {"status": "duplicate", "observation": observation, "event": None}
        self._sequence += 1
        dialogue_id = f"{self.session_id}-{self._sequence:06d}"
        event = dict(observation, dialogue_id=dialogue_id, event_id=self._sequence,
                     original_text=body, review_text=None, original_speaker=speaker, review_speaker=None,
                     status="candidate", revision=0, revision_reason="first_pass",
                     speaker_status="candidate" if self.speaker_roi else "disabled",
                     speaker_identity=identity, speaker_identity_confirmed=bool(identity),
                     quality_threshold=self.min_confidence, human_confirmed=False,
                     candidate_latency_ms=round((time.monotonic()-captured_at)*1000, 3) if captured_at is not None else None)
        self._last_body, self._last_id = body_key, dialogue_id
        self._records[dialogue_id] = event
        self._pending[dialogue_id] = {"body": body_image, "body_offset": body_offset,
                                      "speaker": speaker_image, "speaker_offset": speaker_offset,
                                      "captured_at": captured_at}
        return {"status": "new", "observation": observation, "event": event}

    def evidence(self, dialogue_id):
        return self._pending.get(dialogue_id)

    def verify(self, dialogue_id):
        source = self._pending[dialogue_id]
        record = self._records[dialogue_id]
        started = time.perf_counter()
        body_lines, text = self._recognize_image(source["body"], source["body_offset"], self.verify_scale,
                                               recover_punctuation=True, centered=self.centered_body,
                                               engine=self.review_engine)
        if source["speaker"] is not None:
            name_lines, name = self._recognize_image(source["speaker"], source["speaker_offset"],
                                                   self.speaker_review_scale, engine=self.review_engine,
                                                   speaker_style=self.centered_body)
        else:
            name_lines, name = [], ""
        score = min((line["confidence"] for line in body_lines), default=0.)
        name_score = min((line["confidence"] for line in name_lines), default=0.)
        agreement = _text_key(text) == _text_key(record["original_text"])
        review_coverage_warning=any(line.get('coverage_warning',False) for line in body_lines)
        quality = (min(score, record["body_confidence"]) >= self.min_confidence
                   and not record.get('source_clipped',False)
                   and not record.get('coverage_warning',False) and not review_coverage_warning)
        status = "verified" if agreement and quality else "needs_review"
        name_agreement = bool(name) and _text_key(name) == _text_key(record["original_speaker"])
        name_quality = min(name_score, record["speaker_confidence"] or 0.) >= self.min_confidence
        revised = dict(record, review_text=text, review_speaker=name,
                       review_raw_model_text=model_text_from_lines(body_lines),
                       review_body_lines=body_lines, review_speaker_lines=name_lines,
                       review_body_confidence=score, review_speaker_confidence=name_score,
                       review_coverage_warning=review_coverage_warning,
                       status=status, revision=record["revision"]+1,
                       revision_reason="model_review" if self.review_engine is not self.engine else "scaled_review", agreement=agreement,
                       speaker_status=("verified" if name_agreement and name_quality else "needs_review") if self.speaker_roi else "disabled",
                       review_scale=self.verify_scale,
                       speaker_review_scale=self.speaker_review_scale,
                       review_processing_ms=round((time.perf_counter()-started)*1000, 3),
                       review_latency_ms=round((time.monotonic()-source["captured_at"])*1000, 3) if source["captured_at"] is not None else None,
                       revised_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        self._records[dialogue_id] = revised
        del self._pending[dialogue_id]
        return revised


