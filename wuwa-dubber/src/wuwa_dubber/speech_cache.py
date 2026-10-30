"""Content-addressed speech shared by previews and captured dialogue."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import threading
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from .models import Utterance, VoiceProfile
from .text_processing import normalize_text
from .tts.client import SynthesisError


class SpeechCancelled(Exception):
    """The caller no longer wants this speech."""


@dataclass(frozen=True, slots=True)
class CachedSpeech:
    path: Path
    cache_hit: bool
    synthesis_seconds: float


def valid_wav(path: Path) -> bool:
    """Check the RIFF envelope and read the entire declared PCM payload."""
    try:
        with Path(path).open("rb") as raw:
            header = raw.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                return False
            declared_size = struct.unpack_from("<I", header, 4)[0] + 8
            if declared_size < 44 or declared_size > Path(path).stat().st_size:
                return False
            raw.seek(0)
            with wave.open(raw, "rb") as audio:
                frames = audio.getnframes()
                frame_bytes = audio.getnchannels() * audio.getsampwidth()
                if frames <= 0 or frame_bytes <= 0 or audio.getframerate() <= 0 or audio.getcomptype() != "NONE":
                    return False
                while frames:
                    count = min(frames, 16384)
                    if len(audio.readframes(count)) != count * frame_bytes:
                        return False
                    frames -= count
        return True
    except (OSError, EOFError, wave.Error, ValueError, struct.error):
        return False


class SpeechCache:
    def __init__(self, config, cache_dir: Path, client):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.client = client
        self._identity_lock = threading.Lock()
        self._identity_ready = False
        self._identity = None

    def _client_identity(self):
        with self._identity_lock:
            if not self._identity_ready:
                identity = getattr(self.client, "cache_identity", None)
                if callable(identity):
                    # Detach the snapshot from health dictionaries a client may mutate.
                    self._identity = json.loads(json.dumps(identity(), ensure_ascii=False, sort_keys=True))
                self._identity_ready = True
        return self._identity

    def _parts(self, utterance: Utterance, profile: VoiceProfile):
        text = normalize_text(utterance.text)
        if not text:
            raise SynthesisError("朗读文字不能为空")
        reference_digest = ""
        if profile.reference_audio:
            reference = Path(profile.reference_audio)
            if not reference.is_absolute():
                raise ValueError("SpeechCache 需要绝对路径的参考音频")
            with reference.open("rb") as handle:
                reference_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        settings = self.config.to_dict()
        identity = self._client_identity()
        engine_settings = {
            key: value for key, value in settings.items()
            if key.startswith(("tts_", "sapi_"))
            and key not in {"tts_token", "tts_endpoint"}
        }
        engine_settings["use_fake_tts"] = settings.get("use_fake_tts", False)
        # Backends without a model identity cannot safely share across servers.
        if identity is None and settings.get("tts_backend") not in {"windows-sapi"} and not settings.get("use_fake_tts"):
            engine_settings["endpoint"] = str(settings.get("tts_endpoint", "")).rstrip("/")
        payload = {
            "version": 2, "normalization_version": 1, "text": text,
            "speaker": profile.speaker,
            "reference_sha256": reference_digest,
            "prompt_text": profile.prompt_text,
            "reference_language": profile.reference_language,
            "speed": profile.speed,
            "client": f"{type(self.client).__module__}.{type(self.client).__qualname__}",
            "engine": engine_settings, "client_settings": identity,
        }
        return payload, settings, reference_digest

    def _path(self, payload) -> Path:
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    def path_for(self, utterance: Utterance, profile: VoiceProfile) -> Path:
        return self._path(self._parts(utterance, profile)[0])

    def _legacy_paths(self, utterance, profile, parts):
        payload, settings, reference_digest = parts
        for text in dict.fromkeys((utterance.text, payload["text"])):
            old = {
                "version": 1, "text": text, "speaker": utterance.speaker,
                "profile": profile.to_dict(), "reference_sha256": reference_digest,
                "tts": {key: value for key, value in settings.items()
                        if key.startswith(("tts_", "sapi_")) or key == "use_fake_tts"},
                "client": payload["client"],
            }
            if callable(getattr(self.client, "cache_identity", None)):
                old["client_settings"] = payload["client_settings"]
            yield self._path(old)

    @staticmethod
    def _check_cancel(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise SpeechCancelled("本句生成已取消")

    def _lookup(self, utterance, profile, parts, cancel_event=None):
        target = self._path(parts[0])
        self._check_cancel(cancel_event)
        if valid_wav(target):
            self._check_cancel(cancel_event)
            return CachedSpeech(target, True, 0.0)
        for old_path in self._legacy_paths(utterance, profile, parts):
            if not valid_wav(old_path):
                continue
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.stem}.{uuid4().hex}.migrating.wav")
            try:
                shutil.copyfile(old_path, temporary)
                if not valid_wav(temporary):
                    continue
                self._check_cancel(cancel_event)
                temporary.replace(target)
                return CachedSpeech(target, True, 0.0)
            finally:
                temporary.unlink(missing_ok=True)
        return None

    def lookup(self, utterance: Utterance, profile: VoiceProfile) -> CachedSpeech | None:
        return self._lookup(utterance, profile, self._parts(utterance, profile))

    def get_or_create(self, utterance: Utterance, profile: VoiceProfile, *,
                      cancel_event: threading.Event, force: bool = False) -> CachedSpeech:
        self._check_cancel(cancel_event)
        parts = self._parts(utterance, profile)
        if not force:
            found = self._lookup(utterance, profile, parts, cancel_event)
            if found is not None:
                return found
        self._check_cancel(cancel_event)
        target = self._path(parts[0])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.stem}.{uuid4().hex}.generating.wav")
        try:
            request = replace(utterance, text=parts[0]["text"])
            elapsed = self.client.synthesize(request, profile, temporary)
            self._check_cancel(cancel_event)
            if not valid_wav(temporary):
                raise SynthesisError("语音后端未生成完整有效的 WAV 音频")
            self._check_cancel(cancel_event)
            temporary.replace(target)
            return CachedSpeech(target, False, elapsed)
        except Exception as exc:
            if cancel_event.is_set() and not isinstance(exc, SpeechCancelled):
                raise SpeechCancelled("本句生成已取消") from exc
            raise
        finally:
            temporary.unlink(missing_ok=True)
            # HTTP clients may fail while writing their own adjacent temporary file.
            temporary.with_suffix(temporary.suffix + ".tmp").unlink(missing_ok=True)
