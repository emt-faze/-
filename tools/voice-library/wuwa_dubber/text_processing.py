from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

_UI_MARKERS = (
    "点击继续",
    "点击空白处继续",
    "按空格键继续",
    "自动播放",
    "跳过剧情",
    "任务提示",
)

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")
_BRACKET_NOISE_RE = re.compile(r"[<>《》【】\[\]（）()·•●○◇◆■□▶►]+")


def normalize_text(value: str) -> str:
    """Normalize OCR output while preserving story punctuation."""
    value = unicodedata.normalize("NFKC", value or "")
    for ascii_mark, chinese_mark in ((",", "，"), (":", "："), (";", "；"), ("!", "！"), ("?", "？")):
        value = value.replace(ascii_mark, chinese_mark)
    value = _CONTROL_RE.sub("", value)
    value = value.replace("…", "...").replace("——", "--")
    value = _SPACE_RE.sub(" ", value).strip()
    value = _BRACKET_NOISE_RE.sub("", value).strip()
    if value in _UI_MARKERS:
        return ""
    return value


def normalize_speaker(value: str) -> str:
    value = normalize_text(value)
    value = value.strip("：:;；,，。.!！?？")
    value = re.sub(r"[\s\-—_]", "", value)
    return value


def merge_ocr_lines(lines: Iterable[str]) -> str:
    cleaned = [normalize_text(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return ""
    merged = cleaned[0]
    for line in cleaned[1:]:
        if _needs_gap(merged, line):
            merged += " "
        merged += line
    return normalize_text(merged)


def _needs_gap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left[-1].isascii() and right[0].isascii():
        return left[-1].isalnum() and right[0].isalnum()
    return False


def fingerprint_text(value: str) -> str:
    normalized = normalize_text(value)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def is_likely_ui_text(value: str) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return True
    if normalized in _UI_MARKERS:
        return True
    if len(normalized) <= 1 and not normalized.isalnum():
        return True
    return False
