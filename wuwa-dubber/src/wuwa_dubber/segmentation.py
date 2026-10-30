"""Conservative, lossless speech segmentation with a 300-character hard limit."""
from __future__ import annotations

import re
import unicodedata

_PAIRS = {"（": "）", "(": ")", "[": "]", "【": "】", "{": "}",
          "《": "》", "〈": "〉", "“": "”", "‘": "’", "「": "」", "『": "』"}
_TERMINAL = set("。！？!?")
_SOFT = set("，,；;：:、")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "approx"}
_HARD_LIMIT = 300


def _period_ends(text: str, index: int) -> bool:
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    if before == "." or after == ".":
        return after != "."
    if before.isalnum() and after.isalnum():
        return False
    match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", text[:index])
    if match:
        word = match.group(1)
        if word.lower() in _ABBREVIATIONS or len(word) == 1 or "." in word:
            return False
    return True


def _boundaries(text: str):
    stack: list[str] = []
    hard, soft = [], []
    pending_terminal = False
    for index, char in enumerate(text):
        following = text[index + 1] if index + 1 < len(text) else ""
        if char in {"'", '"'}:
            # English apostrophes are part of words, not quotation delimiters.
            if char == "'" and index and following and text[index - 1].isalnum() and following.isalnum():
                pending_terminal = False
                continue
            if stack and stack[-1] == char:
                stack.pop()
                if not stack and pending_terminal:
                    hard.append(index + 1)
            else:
                stack.append(char)
                pending_terminal = False
            continue
        if char in _PAIRS:
            stack.append(_PAIRS[char])
            pending_terminal = False
            continue
        if stack and char == stack[-1]:
            stack.pop()
            if char not in {"”", "’", "」", "』"}:
                # A sentence inside a parenthetical is not the end of the
                # surrounding sentence, even when its closing bracket follows 。.
                pending_terminal = False
            elif not stack and pending_terminal:
                hard.append(index + 1)
            continue
        terminal = char in _TERMINAL or char == "…" or (char == "." and _period_ends(text, index))
        if terminal:
            pending_terminal = True
            # Keep punctuation runs and closing quotation marks with the sentence.
            if not stack and following not in _TERMINAL | {"…", ".", '"', "'"}:
                hard.append(index + 1)
        elif not char.isspace():
            pending_terminal = False
        if not stack and char in _SOFT:
            # Do not split decimal commas, times or numerical separators.
            if not (index and following and text[index - 1].isdigit() and following.isdigit()):
                soft.append(index + 1)
    return sorted(set(hard)), soft


def _safe_cut(text: str, limit: int) -> int:
    spaces = [match.end() for match in re.finditer(r"\s+", text[:limit])]
    if spaces and spaces[-1] > 0:
        return spaces[-1]
    cut = min(limit, len(text))
    # If a short English word or numeric token straddles the hard boundary,
    # move the cut before it. An overlong token still needs a last-resort cut.
    if cut < len(text) and text[cut - 1].isascii() and text[cut].isascii() and text[cut - 1].isalnum() and text[cut].isalnum():
        start = cut
        while start and text[start - 1].isascii() and (text[start - 1].isalnum() or text[start - 1] in ".-_/"):
            start -= 1
        if start:
            cut = start
    while 0 < cut < len(text) and (unicodedata.combining(text[cut]) or text[cut] in "\ufe0f\ufe0e\u200d" or text[cut - 1] == "\u200d"):
        cut -= 1
    return cut or min(limit, len(text))


def _split_long(text: str, max_chars: int) -> list[str]:
    result = []
    while len(text) > max_chars:
        _, soft = _boundaries(text)
        # Keep short introductions such as 她说： attached to their quotation.
        soft = [position for position in soft
                if not (text[position:position + 1] in {*_PAIRS, '"', "'"}
                        and sum(char.isalnum() for char in text[:position]) < 4)]
        before = [position for position in soft if position <= max_chars]
        if before:
            cut = before[-1]
        else:
            after = [position for position in soft if position <= _HARD_LIMIT]
            if after:
                cut = after[0]
            elif len(text) <= _HARD_LIMIT:
                break
            else:
                cut = _safe_cut(text, _HARD_LIMIT)
        result.append(text[:cut])
        text = text[cut:]
    if text:
        result.append(text)
    return result


def split_for_speech(text: str, *, max_chars: int = 80) -> list[str]:
    """Prefer sentence boundaries; never lose characters or exceed 300 chars.

    ``max_chars`` is a preferred chunk length, not a license to split short
    words or quoted passages. Unpunctuated passages may exceed it up to 300.
    Concatenating the returned chunks reproduces a nonblank input exactly.
    """
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars 必须是正整数")
    if not text or not text.strip():
        return []
    max_chars = min(max_chars, _HARD_LIMIT)
    boundaries, _ = _boundaries(text)
    sentences = []
    start = 0
    for end in boundaries:
        if end > start:
            sentences.append(text[start:end])
            start = end
    if start < len(text):
        sentences.append(text[start:])
    pieces = [piece for sentence in sentences for piece in _split_long(sentence, max_chars)]
    result: list[str] = []
    for piece in pieces:
        # Merge tiny interjections into an adjacent sentence, never a 1-char utterance.
        previous_short = result and sum(char.isalnum() for char in result[-1]) < 4
        current_short = sum(char.isalnum() for char in piece) < 4
        if result and (previous_short or current_short) and len(result[-1]) + len(piece) <= max_chars:
            result[-1] += piece
        else:
            result.append(piece)
    return result


def split_for_latency(text: str, *, first_max_chars: int = 10, max_chars: int = 36) -> list[str]:
    """缩短角色配音首段；设为 0 时保留原自然分段策略。"""
    if not isinstance(first_max_chars, int) or isinstance(first_max_chars, bool) or first_max_chars < 0:
        raise ValueError("首段字数必须是非负整数")
    pieces = split_for_speech(text, max_chars=max_chars)
    if not pieces or first_max_chars == 0:
        return pieces
    first_limit = min(first_max_chars, max_chars, _HARD_LIMIT)
    first = pieces[0]
    if len(first) > first_limit:
        hard, soft = _boundaries(first)
        # 优先已有停顿；没有停顿时保留字符顺序，尽量避开英文词和数字内部。
        boundaries = [p for p in (*hard, *soft) if p <= first_limit]
        cut = max(boundaries) if boundaries else _safe_cut(first, first_limit)
        pieces = [first[:cut], *split_for_speech(first[cut:] + "".join(pieces[1:]), max_chars=max_chars)]
    result = [pieces[0]]
    for piece in pieces[1:]:
        while len(piece) > max_chars:
            cut = _safe_cut(piece, max_chars)
            result.append(piece[:cut])
            piece = piece[cut:]
        if piece:
            result.append(piece)
    return result
