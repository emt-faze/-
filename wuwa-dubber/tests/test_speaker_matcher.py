from wuwa_dubber.models import VoiceProfile
from wuwa_dubber.speaker_matcher import SpeakerMatcher


def test_speaker_matcher_prefers_exact_alias() -> None:
    matcher = SpeakerMatcher([
        VoiceProfile(speaker="今汐", aliases=["今 汐"]),
        VoiceProfile(speaker="漂泊者", aliases=["男漂泊者", "女漂泊者"]),
    ])
    exact = matcher.match("今汐：")
    fuzzy = matcher.match("女漂泊者")
    assert exact is not None and exact.score == 1.0
    assert fuzzy is not None and fuzzy.speaker == "漂泊者"


def test_speaker_matcher_returns_none_below_threshold() -> None:
    matcher = SpeakerMatcher([VoiceProfile(speaker="今汐")], threshold=0.9)
    assert matcher.match("完全无关的角色名") is None
