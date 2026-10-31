from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from ..models import Utterance, VoiceProfile
from .url_policy import local_opener, validate_local_url

if TYPE_CHECKING:
    from ..config import AppConfig


class SynthesisError(RuntimeError):
    pass


class TtsClient(Protocol):
    def health(self) -> dict[str, Any]: ...

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float: ...


class GPTSoVITSClient:
    def __init__(self, endpoint: str, token: str, *, timeout: float = 45.0, health_timeout: float = 5.0) -> None:
        self.endpoint = validate_local_url(endpoint)
        self.token = token
        self.timeout = timeout
        self.health_timeout = health_timeout

    def _request(self, path: str, *, payload: dict[str, Any] | None = None, method: str = "POST", timeout: float | None = None) -> tuple[bytes, str]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "audio/wav, application/json",
            },
        )
        try:
            with local_opener().open(request, timeout=self.timeout if timeout is None else timeout) as response:
                return response.read(), response.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SynthesisError(f"TTS HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SynthesisError(f"无法连接 TTS 工作进程: {exc}") from exc

    def health(self) -> dict[str, Any]:
        body, _ = self._request("/v1/health", method="GET", timeout=self.health_timeout)
        try:
            result = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SynthesisError("TTS 健康检查返回了无效 JSON") from exc
        if not isinstance(result, dict) or result.get("status") != "ok":
            detail = result.get("detail", result) if isinstance(result, dict) else result
            raise SynthesisError(f"TTS 服务尚未就绪：{detail}")
        return result

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        started = time.monotonic()
        body, content_type = self._request(
            "/v1/synthesize",
            payload={
                "request_id": utterance.request_id,
                "text": utterance.text,
                "speaker": profile.speaker,
                "speaker_aliases": profile.aliases,
                "ref_audio_path": profile.reference_audio,
                "prompt_text": profile.prompt_text,
                "text_lang": "zh",
                "prompt_lang": profile.reference_language or "zh",
                "speed": profile.speed,
            },
        )
        if content_type == "application/json":
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise SynthesisError("TTS 返回了无法解析的 JSON") from exc
            raise SynthesisError(str(payload.get("detail") or payload.get("error") or payload))
        if not body.startswith(b"RIFF"):
            raise SynthesisError("TTS 未返回 WAV 音频")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(body)
        temporary.replace(output)
        return time.monotonic() - started


class Qwen3TtsClient(GPTSoVITSClient):
    """Qwen3-TTS runs in a separate local model environment behind the same API."""

    def __init__(self, endpoint: str, token: str, *, timeout: float = 180.0) -> None:
        super().__init__(endpoint, token, timeout=timeout, health_timeout=5.0)
        self._active_lock = threading.Lock()
        self._active_request_id: str | None = None

    def health(self) -> dict[str, Any]:
        result = super().health()
        if result.get("backend") != "qwen3-tts":
            raise SynthesisError("当前地址不是 Qwen3-TTS 服务，请检查已启动的服务和端口（默认 9882）")
        return result

    def cache_identity(self) -> dict[str, Any]:
        health = self.health()
        return {
            "backend": "qwen3-tts",
            "protocol_version": 1,
            "model_id": health.get("model_id", health.get("model", "")),
            "model_revision": health.get("model_revision", health.get("revision", "")),
            "inference": health.get("inference", {}),
            "inference_version": health.get("inference_version", ""),
            "sample_rate": health.get("sample_rate", ""),
            "dtype": health.get("dtype", ""),
        }

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        # A transport request gets a fresh ID, including a retry of a cancelled line.
        request = replace(utterance, request_id=uuid4().hex)
        with self._active_lock:
            if self._active_request_id is not None:
                raise SynthesisError("此客户端已有一句正在生成，请等待或先停止")
            self._active_request_id = request.request_id
        try:
            return super().synthesize(request, profile, output)
        except Exception:
            # A socket timeout does not stop the server's generation. Cancel
            # while the active ID is still available, then release this client.
            self.cancel()
            raise
        finally:
            with self._active_lock:
                if self._active_request_id == request.request_id:
                    self._active_request_id = None

    def cancel(self) -> None:
        with self._active_lock:
            request_id = self._active_request_id
        if request_id is None:
            return
        try:
            self._request("/v1/cancel", payload={"request_id": request_id}, timeout=2.0)
        except SynthesisError:
            # The caller's cancellation event still prevents late audio playback.
            pass


def create_tts_client(config: AppConfig) -> TtsClient:
    if config.use_fake_tts:
        from .fake import FakeTtsClient

        return FakeTtsClient()
    if config.tts_backend == "windows-sapi":
        from .windows import WindowsTtsClient

        return WindowsTtsClient()
    if config.tts_backend == "qwen3-tts":
        return Qwen3TtsClient(config.tts_endpoint, config.tts_token)
    if config.tts_backend != "gpt-sovits":
        raise ValueError(f"不支持的配音模式：{config.tts_backend}")
    return GPTSoVITSClient(config.tts_endpoint, config.tts_token)
