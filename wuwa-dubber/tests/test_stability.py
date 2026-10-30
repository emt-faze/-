from wuwa_dubber.stability import StableLineDetector


def test_line_is_emitted_only_after_required_frames() -> None:
    detector = StableLineDetector(required_frames=3, history_size=5)
    assert detector.observe("漂泊者，你醒了。", "今汐") is None
    assert detector.observe("漂泊者，你醒了。", "今汐") is None
    utterance = detector.observe("漂泊者，你醒了。", "今汐")
    assert utterance is not None
    assert utterance.text == "漂泊者，你醒了。"
    assert utterance.speaker == "今汐"


def test_duplicate_line_is_ignored_until_history_rolls_over() -> None:
    detector = StableLineDetector(required_frames=1, history_size=2)
    assert detector.observe("第一句") is not None
    assert detector.observe("第一句") is None
    assert detector.observe("第二句") is not None
    assert detector.observe("第三句") is not None
    assert detector.observe("第一句") is not None


def test_ui_text_resets_candidate() -> None:
    detector = StableLineDetector(required_frames=2)
    assert detector.observe("正在说话") is None
    assert detector.observe("点击继续") is None
    assert detector.observe("正在说话") is None
    assert detector.observe("正在说话") is not None


def test_same_words_from_another_speaker_are_not_dropped() -> None:
    detector = StableLineDetector(required_frames=1)
    assert detector.observe("好的。", "今汐") is not None
    assert detector.observe("好的。", "长离") is not None
    assert detector.observe("好的。", "长离") is None
