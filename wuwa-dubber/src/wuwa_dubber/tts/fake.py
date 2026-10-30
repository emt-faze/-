from __future__ import annotations

import math
import struct
import time
import wave
from pathlib import Path
from typing import Any

from ..models import Utterance, VoiceProfile


class FakeTtsClient:
    """Zero-dependency WAV generator for end-to-end testing without a model."""

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "fake"}

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        started = time.monotonic()
        sample_rate = 32000
        duration = min(1.2, max(0.25, len(utterance.text) * 0.05))
        frames = int(sample_rate * duration)
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            for index in range(frames):
                envelope = min(1.0, index / 500.0, (frames - index) / 500.0)
                sample = int(6000 * envelope * math.sin(2 * math.pi * 220 * index / sample_rate))
                handle.writeframesraw(struct.pack("<h", sample))
        return time.monotonic() - started
