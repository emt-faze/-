from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import VoiceProfile
from .text_processing import normalize_speaker


def _similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio

        return float(ratio(left, right)) / 100.0
    except ImportError:
        return SequenceMatcher(None, left, right).ratio()


@dataclass(slots=True)
class SpeakerMatch:
    speaker: str
    score: float
    matched_alias: str = ""


class SpeakerMatcher:
    def __init__(self, profiles: list[VoiceProfile], *, threshold: float = 0.82) -> None:
        self.threshold = threshold
        self._entries: list[tuple[str, str, str]] = []
        self.reload(profiles)

    def reload(self, profiles: list[VoiceProfile]) -> None:
        self._entries.clear()
        for profile in profiles:
            if not profile.enabled:
                continue
            for alias in [profile.speaker, *profile.aliases]:
                normalized = normalize_speaker(alias)
                if normalized:
                    self._entries.append((normalized, profile.speaker, alias))

    def match(self, observed: str) -> SpeakerMatch | None:
        normalized = normalize_speaker(observed)
        if not normalized:
            return None

        for alias_normalized, speaker, alias in self._entries:
            if normalized == alias_normalized:
                return SpeakerMatch(speaker=speaker, score=1.0, matched_alias=alias)

        best: SpeakerMatch | None = None
        for alias_normalized, speaker, alias in self._entries:
            score = _similarity(normalized, alias_normalized)
            if score >= self.threshold and (best is None or score > best.score):
                best = SpeakerMatch(speaker=speaker, score=score, matched_alias=alias)
        return best
