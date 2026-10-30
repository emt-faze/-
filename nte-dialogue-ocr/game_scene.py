"""Pure image classification for the NTE title-to-world transition.

No input, process, capture, or account operations live here. Templates are small
upstream screenshot crops; their pinned sources and license accompany the assets.
An absent entry button is never sufficient evidence that loading has completed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets" / "game-entry"
ENTER_ROI = (.30, .71, .70, .94)
ENTER_LABELS = {"进入游戏", "進入遊戲"}
SEARCH_REGIONS = {
    "login_setting": (.90, .005, 1.0, .13),
    "mini_map_arrow": (.045, .06, .13, .20),
    "health_bar_slash": (.40, .80, .64, .985),
    "char_1_text": (.92, .12, 1.0, .48),
    "char_2_text": (.92, .22, 1.0, .66),
    "auto_play": (.72, .005, .92, .13),
    "dialog_history": (.79, .005, .97, .13),
    "skip_dialog": (.91, .005, 1.0, .13),
    "dialog_click": (.36, .82, .64, .995),
}


def _check_frame(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR game-client frame")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 100:
        raise ValueError("Expected a nonempty game-client BGR frame")


def _crop(frame, roi):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h))
    return frame[y0:y1, x0:x1], x0, y0


def _ocr_region(frame, engine, roi):
    image, x, y = _crop(frame, roi)
    results, _ = engine(np.ascontiguousarray(image), use_cls=False)
    lines = []
    for points, text, score in results or []:
        points = np.asarray(points, dtype=float)
        score = float(score)
        if points.shape != (4, 2) or not np.isfinite(points).all() or not np.isfinite(score):
            continue
        left, top = points.min(axis=0) + [x, y]
        right, bottom = points.max(axis=0) + [x, y]
        if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
            continue
        lines.append({"text": str(text), "score": score,
                      "bbox": [float(left), float(top), float(right), float(bottom)]})
    return sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def locate_enter_button(frame, engine):
    """Locate an unambiguous, high-scoring entry label in the lower centre."""
    _check_frame(frame)
    matches = []
    for line in _ocr_region(frame, engine, ENTER_ROI):
        text = re.sub(r"\s+", "", line["text"])
        if text not in ENTER_LABELS or line["score"] < .90:
            continue
        left, top, right, bottom = line["bbox"]
        width, height = frame.shape[1], frame.shape[0]
        if not (.025 * width <= right-left <= .22 * width and .008 * height <= bottom-top <= .075 * height):
            continue
        matches.append(dict(line, center=[int((left+right)/2), int((top+bottom)/2)]))
    # Multiple candidates are deliberately not resolved by guessing a position.
    return matches[0] if len(matches) == 1 else None


@lru_cache(maxsize=16)
def _template(name):
    paths = sorted(ASSETS.glob(name + "_2560x1440_*.png"))
    if len(paths) != 1:
        return None
    return cv2.imdecode(np.fromfile(paths[0], np.uint8), cv2.IMREAD_COLOR)


def _scales(frame):
    h, w = frame.shape[:2]
    # UI scaling may follow width or height; search both instead of assuming 16:9.
    bases = (w / 2560, h / 1440)
    return sorted({round(base * offset, 3) for base in bases for offset in (.90, 1.0, 1.10)})


def _gray(frame, binary=False):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if binary:
        return cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)[1]
    return result


def _rotate(image, angle):
    if angle == 0:
        return image
    height, width = image.shape
    side = int(np.ceil(np.hypot(height, width))) + 2
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    matrix[0, 2] += side/2 - width/2
    matrix[1, 2] += side/2 - height/2
    rotated = cv2.warpAffine(image, matrix, (side, side), flags=cv2.INTER_NEAREST, borderValue=0)
    coordinates = cv2.findNonZero(rotated)
    if coordinates is None:
        return rotated
    x, y, w, h = cv2.boundingRect(coordinates)
    return cv2.copyMakeBorder(rotated[y:y+h, x:x+w], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)


def _match_feature(frame, name, threshold=.84):
    template = _template(name)
    if template is None:
        return None
    roi, offset_x, offset_y = _crop(frame, SEARCH_REGIONS[name])
    binary = name == "mini_map_arrow"
    search = _gray(roi, binary)
    original = _gray(template, binary)
    best = None
    angles = range(0, 360, 30) if binary else (0,)
    for scale in _scales(frame):
        width, height = max(3, round(original.shape[1]*scale)), max(3, round(original.shape[0]*scale))
        resized = cv2.resize(original, (width, height), interpolation=cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR)
        for angle in angles:
            candidate = _rotate(resized, angle)
            th, tw = candidate.shape
            if th >= search.shape[0] or tw >= search.shape[1] or float(candidate.std()) < 3:
                continue
            correlation = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(correlation)
            if best is None or score > best["score"]:
                x, y = location[0]+offset_x, location[1]+offset_y
                best = {"type": "template", "feature": name, "score": round(float(score), 4),
                        "bbox": [x, y, x+tw, y+th], "scale": scale, "angle": angle}
    if binary and best is not None and best["score"] >= .55:
        # The minimap marker rotates continuously. A 30-degree grid alone missed
        # the observed 202-degree marker (0.744 versus 0.932 after refinement).
        # Refine geometry, retaining the same confidence threshold and requiring
        # a separate HUD indicator in classify_frame.
        best_angle, best_scale = best["angle"], best["scale"]
        resized = cv2.resize(original,
                             (max(3, round(original.shape[1]*best_scale)), max(3, round(original.shape[0]*best_scale))),
                             interpolation=cv2.INTER_NEAREST)
        for angle in range(best_angle-15, best_angle+16):
            candidate = _rotate(resized, angle % 360)
            th, tw = candidate.shape
            if th >= search.shape[0] or tw >= search.shape[1]:
                continue
            correlation = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(correlation)
            if score > best["score"]:
                x, y = location[0]+offset_x, location[1]+offset_y
                best = {"type": "template", "feature": name, "score": round(float(score), 4),
                        "bbox": [x, y, x+tw, y+th], "scale": best_scale, "angle": angle % 360}
    return best if best is not None and best["score"] >= threshold else None


def _blocked_evidence(lines):
    findings = []
    hard_phrases = ("扫码登录", "掃碼登入", "账号登录", "帐号登录", "请输入验证码", "請輸入驗證碼",
                    "网络连接失败", "網絡連接失敗", "连接服务器失败", "服务器维护中", "伺服器維護中",
                    "资源更新中", "正在更新资源", "账号已在其他设备登录")
    for line in lines:
        if line["score"] >= .90 and any(phrase in re.sub(r"\s+", "", line["text"]) for phrase in hard_phrases):
            findings.append(dict(line, type="blocking_text"))
    return findings


def classify_frame(frame, engine):
    """Return conservative scene evidence; callers must confirm consecutive frames.

    ``in_game`` requires a minimap indicator plus an independent health/party
    indicator, or two story control icons plus actual dialogue text. A title
    frame needs both its entry label and the login settings symbol. No inference
    is made from a disappeared button, black frame, or arbitrary scene artwork.
    """
    _check_frame(frame)
    output = {"state": "unknown", "enter_button": None, "evidence": []}
    if float(frame[::8, ::8].std()) < 2:
        return output
    button = locate_enter_button(frame, engine)
    if button is not None:
        gear = _match_feature(frame, "login_setting", .78)
        output["enter_button"] = button
        output["evidence"] = [dict(button, type="entry_text")]
        if gear:
            output["state"] = "title"
            output["evidence"].append(gear)
        return output

    minimap = _match_feature(frame, "mini_map_arrow", .77)
    if minimap:
        health = _match_feature(frame, "health_bar_slash", .86)
        first = _match_feature(frame, "char_1_text", .89)
        second = _match_feature(frame, "char_2_text", .89)
        # Two numbered slots must occupy distinct rows, not the same glyph.
        party = first and second and second["bbox"][1] - first["bbox"][1] > .035 * frame.shape[0]
        if health or party:
            output.update(state="in_game", evidence=[minimap] + ([health] if health else []) + ([first, second] if party else []))
            return output

    history = _match_feature(frame, "dialog_history", .85)
    if history:
        play = _match_feature(frame, "auto_play", .87)
        skip = _match_feature(frame, "skip_dialog", .87)
        if play or skip:
            dialogue = _ocr_region(frame, engine, (.08, .65, .92, .96))
            body = [line for line in dialogue if line["score"] >= .90
                    and len(re.findall(r"[\u3400-\u9fff]", line["text"])) >= 6]
            if body:
                output.update(state="in_game", evidence=[history, play or skip, dict(body[0], type="dialogue_text")])
                return output

    # Only explicit account/network/update text is a blocking scene. Unknown
    # loading images remain unknown so the caller can continue bounded waiting.
    central_text = _ocr_region(frame, engine, (.18, .18, .82, .78))
    blockers = _blocked_evidence(central_text)
    if blockers:
        output.update(state="blocked", evidence=blockers)
    return output
