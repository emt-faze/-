from wuwa_dubber.text_processing import fingerprint_text, merge_ocr_lines, normalize_speaker, normalize_text


def test_normalize_text_handles_full_width_and_noise() -> None:
    assert normalize_text("  【测试】  你好。。。 ") == "测试 你好。。。"
    assert normalize_text("点击继续") == ""


def test_merge_ocr_lines_preserves_chinese_and_adds_ascii_gap() -> None:
    assert merge_ocr_lines(["这是", "一句台词"]) == "这是一句台词"
    assert merge_ocr_lines(["RTX", "4060"]) == "RTX 4060"


def test_speaker_normalization_and_fingerprint() -> None:
    assert normalize_speaker(" 今 汐： ") == "今汐"
    assert fingerprint_text("你好") == fingerprint_text("  你好  ")


def test_story_parentheses_survive_normalization_for_natural_segmentation() -> None:
    from wuwa_dubber.segmentation import split_for_speech
    text = "她想了一下（这里很安静。也很安全。），然后继续前进。"
    normalized = normalize_text(text)
    assert "(" in normalized and ")" in normalized
    pieces = split_for_speech(normalized, max_chars=36)
    assert len(pieces) == 1 and "".join(pieces) == normalized
