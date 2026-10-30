from wuwa_dubber.segmentation import split_for_speech
from wuwa_dubber.segmentation import split_for_latency


def test_natural_sentences_and_short_interjection_merge() -> None:
    text = "嗯！这次我们一起去看看。接下来还有新的故事！"
    assert split_for_speech(text) == ["嗯！这次我们一起去看看。", "接下来还有新的故事！"]


def test_chinese_and_ascii_ellipses_are_kept_intact() -> None:
    text = "这件事情……以后再告诉你。Please wait... Tomorrow we leave!"
    pieces = split_for_speech(text)
    assert "".join(pieces) == text
    assert pieces == ["这件事情……", "以后再告诉你。", "Please wait...", " Tomorrow we leave!"]


def test_decimal_abbreviation_and_english_period_boundaries() -> None:
    text = "Dr. Smith measured 3.14 units in the U.S. lab. This is complete."
    assert split_for_speech(text) == ["Dr. Smith measured 3.14 units in the U.S. lab.", " This is complete."]
    assert split_for_speech("例如 e.g. apples and i.e. fruit.") == ["例如 e.g. apples and i.e. fruit."]


def test_quotes_parentheses_and_apostrophes_preserve_meaning() -> None:
    text = "她说：“先别走！我们还得等一等。”然后我们（经过讨论，最后同意）留下了。"
    pieces = split_for_speech(text, max_chars=12)
    assert pieces == ["她说：“先别走！我们还得等一等。”", "然后我们（经过讨论，最后同意）留下了。"]
    assert "".join(pieces) == text
    assert split_for_speech("Don't stop. Let's go.") == ["Don't stop.", " Let's go."]


def test_long_sentence_prefers_clause_boundaries() -> None:
    text = "第一部分先介绍城市，第二部分介绍角色；第三部分介绍任务：最后再一起出发。"
    pieces = split_for_speech(text, max_chars=14)
    assert "".join(pieces) == text
    assert len(pieces) >= 3 and all(len(piece) <= 14 for piece in pieces)
    assert all(piece[-1] in "，；：。" for piece in pieces)


def test_parenthetical_sentences_do_not_end_the_outer_sentence() -> None:
    for text in (
        "她想了一下（这里很安静。也很安全。），然后继续前进。",
        "她想了一下(这里很安静。也很安全。)，然后继续前进。",
        "We decided (it is quiet. It is safe.) to continue.",
        "她说：“先想想（这里安全吗？）”然后继续前进。",
    ):
        assert split_for_speech(text) == [text]
    assert split_for_speech("她说：“已经准备好了！”然后继续前进。") == [
        "她说：“已经准备好了！”", "然后继续前进。"]


def test_unpunctuated_short_text_is_not_split_arbitrarily() -> None:
    text = "这是一段没有标点的台词" * 10
    assert len(text) > 80
    assert split_for_speech(text) == [text]


def test_hard_limit_is_enforced_without_dropping_characters() -> None:
    for text in ("字" * 901, "word " * 201, "a" * 700, "“" + "长句" * 400 + "。”", "数据3.14159" * 100):
        pieces = split_for_speech(text)
        assert "".join(pieces) == text
        assert all(0 < len(piece) <= 300 for piece in pieces)


def test_whitespace_punctuation_and_unicode_are_lossless() -> None:
    samples = ["  第一件事情！\n\n第二件事情？  ", "你好？！真的……好吧！", "(Wait! Really?) Continue.",
               "他说：\"It's 3.14, e.g. a sample.\" 然后离开。", "日期12:30，价格1,234.50。", "甲" * 299 + "e\u0301" + "乙" * 20]
    for text in samples:
        parts = split_for_speech(text, max_chars=8)
        assert "".join(parts) == text
        assert all(0 < len(piece) <= 300 for piece in parts)
        assert all(not piece.startswith("\u0301") for piece in parts)


def test_empty_and_invalid_limits() -> None:
    assert split_for_speech("") == []
    assert split_for_speech("  \n") == []
    for invalid in (0, -1, 1.5, True):
        try:
            split_for_speech("台词", max_chars=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid limit accepted")


def test_latency_split_is_bounded_and_lossless_for_five_text_shapes() -> None:
    samples = [
        "今天的天气很不错，我们一起出去走走吧。接着再去看看街角新开的那家书店。",
        "没有标点的台词也需要保持所有文字原本的顺序" * 8,
        "这是一段非常长的角色独白，" * 12 + "最后我们一起出发。",
        "好！走吧。现在出发！别忘了拿上钥匙。",
        "Dr. Smith measured 3.14 units, then checked the results. Everything was correct.",
    ]
    for text in samples:
        parts = split_for_latency(text)
        assert "".join(parts) == text
        assert 0 < len(parts[0]) <= 10
        assert all(0 < len(part) <= 36 for part in parts[1:])


def test_latency_split_prefers_pause_and_can_be_disabled() -> None:
    text = "今天的天气很不错，我们一起出去走走吧。"
    assert split_for_latency(text)[0] == "今天的天气很不错，"
    assert split_for_latency(text, first_max_chars=0) == split_for_speech(text, max_chars=36)
    assert split_for_latency("今天的天气很不错。") == ["今天的天气很不错。"]


def test_latency_split_keeps_quotation_content_order_and_numbers() -> None:
    text = "他说：\"It's 3.14, e.g. a sample.\" 然后我们（先商量，再出发）离开。"
    parts = split_for_latency(text, first_max_chars=10, max_chars=18)
    assert "".join(parts) == text
    assert any("3.14" in part for part in parts)
