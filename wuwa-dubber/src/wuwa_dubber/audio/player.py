from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AudioDevice:
    index: int
    name: str
    host_api: str = ""
    channels: int = 0


class AudioPlayer:
    def __init__(self, device: int | None = None, *, volume: float = 1.0) -> None:
        self.device = device
        self.volume = min(1.5, max(0.0, volume))
        self._stop = threading.Event()
        self._play_lock = threading.RLock()

    @staticmethod
    def list_devices() -> list[AudioDevice]:
        try:
            import sounddevice as sd
        except ImportError:
            return []
        devices = []
        for index, item in enumerate(sd.query_devices()):
            if int(item.get("max_output_channels", 0)) <= 0:
                continue
            host = sd.query_hostapis(item["hostapi"])["name"]
            devices.append(
                AudioDevice(
                    index=index,
                    name=str(item["name"]),
                    host_api=str(host),
                    channels=int(item["max_output_channels"]),
                )
            )
        return devices

    def set_volume(self, volume: float) -> None:
        self.volume = min(1.5, max(0.0, volume))

    def play(self, path: Path, *, cancel_event: threading.Event | None = None) -> None:
        cancelled = cancel_event or threading.Event()
        with self._play_lock:
            if cancelled.is_set():
                return
            self.stop()
            self._stop.clear()
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            import winsound

            with wave.open(str(path), "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
            with self._play_lock:
                if cancelled.is_set() or self._stop.is_set():
                    return
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            import time

            deadline = time.monotonic() + duration
            while time.monotonic() < deadline and not cancelled.is_set():
                if self._stop.wait(min(0.05, max(0.0, deadline - time.monotonic()))):
                    break
            winsound.PlaySound(None, 0)
            return
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        if width not in {1, 2, 4}:
            raise ValueError(f"不支持 {width * 8} 位 WAV，请使用 8/16/32 位 PCM 音频")
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[width]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale
        if width == 1:
            audio -= 1.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        audio = np.clip(audio * self.volume, -1.0, 1.0)
        with self._play_lock:
            if cancelled.is_set() or self._stop.is_set():
                return
            sd.play(audio, sample_rate, device=self.device, blocking=False)
        while not self._stop.wait(0.05):
            if cancelled.is_set():
                break
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
        sd.stop()

    def stop(self) -> None:
        with self._play_lock:
            self._stop.set()
            try:
                import sounddevice as sd

                sd.stop()
            except ImportError:
                try:
                    import winsound

                    winsound.PlaySound(None, 0)
                except ImportError:
                    pass
