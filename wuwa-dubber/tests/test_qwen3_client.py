from __future__ import annotations

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from wuwa_dubber.config import AppConfig, AppPaths, endpoint_for_backend
from wuwa_dubber.diagnostics import DiagnosticItem, collect_diagnostics
from wuwa_dubber.models import Rect, Utterance, VoiceProfile
from wuwa_dubber.tts.client import GPTSoVITSClient, Qwen3TtsClient, SynthesisError, create_tts_client
from wuwa_dubber.voice_profiles import VoiceProfileStore


def ready_config() -> AppConfig:
    config = AppConfig.from_dict({"tts_backend": "qwen3-tts"})
    config.capture.window_title = "Test"
    config.capture.dialogue_rect = Rect(0, 0, 10, 10)
    return config


def test_qwen3_config_and_factory_preserve_other_modes() -> None:
    config = ready_config()
    assert config.tts_endpoint == "http://127.0.0.1:9882"
    assert config.validate() == []
    client = create_tts_client(config)
    assert isinstance(client, Qwen3TtsClient)
    assert client.timeout == 180.0 and client.health_timeout == 5.0
    assert isinstance(create_tts_client(AppConfig(tts_backend="gpt-sovits")), GPTSoVITSClient)
    assert type(create_tts_client(AppConfig())).__name__ == "WindowsTtsClient"
    config.use_fake_tts = True
    assert type(create_tts_client(config)).__name__ == "FakeTtsClient"


def test_backend_switch_changes_only_default_endpoints() -> None:
    assert endpoint_for_backend("qwen3-tts", "http://127.0.0.1:9881/") == "http://127.0.0.1:9882"
    assert endpoint_for_backend("gpt-sovits", "http://127.0.0.1:9882") == "http://127.0.0.1:9881"
    assert endpoint_for_backend("qwen3-tts", "") == "http://127.0.0.1:9882"
    custom = "http://localhost:12000/custom"
    for backend in ("qwen3-tts", "gpt-sovits", "windows-sapi"):
        assert endpoint_for_backend(backend, custom) == custom
    assert endpoint_for_backend("windows-sapi", "http://127.0.0.1:9882") == "http://127.0.0.1:9882"


def test_qwen3_requires_loopback_url() -> None:
    config = ready_config()
    config.tts_endpoint = "http://127.0.0.1.attacker.invalid:9882"
    assert any("本机" in error for error in config.validate())
    try:
        create_tts_client(config)
    except ValueError:
        pass
    else:
        raise AssertionError("Remote Qwen3 service was accepted")


def test_qwen3_health_has_short_timeout_and_checks_backend() -> None:
    client = Qwen3TtsClient("http://127.0.0.1:9882", "test")
    response = MagicMock()
    response.read.return_value = b'{"status":"ok","backend":"qwen3-tts"}'
    response.headers.get_content_type.return_value = "application/json"
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value = response
    with patch("wuwa_dubber.tts.client.local_opener", return_value=opener):
        client.health()
        assert opener.open.call_args.kwargs["timeout"] == 5.0
        client._request("/v1/synthesize", payload={"text": "test"})
        assert opener.open.call_args.kwargs["timeout"] == 180.0
    with patch.object(client, "_request", return_value=(b'{"status":"ok","backend":"gpt-sovits"}', "application/json")):
        try:
            client.health()
        except SynthesisError as exc:
            assert "9882" in str(exc)
        else:
            raise AssertionError("Wrong service accepted as Qwen3")


def test_qwen3_cache_identity_uses_model_revision_not_live_metrics() -> None:
    client = Qwen3TtsClient("http://127.0.0.1:9882", "test")
    health = {"status": "ok", "backend": "qwen3-tts", "model_id": "Qwen/Base", "model_revision": "rev-one", "uptime": 10}
    with patch.object(client, "health", side_effect=[health, {**health, "uptime": 20}, {**health, "model_revision": "rev-two"}]):
        first = client.cache_identity()
        assert client.cache_identity() == first
        assert client.cache_identity() != first
        assert first["model_revision"] == "rev-one"


def test_qwen3_cancel_targets_only_its_active_request(tmp_path: Path) -> None:
    client = Qwen3TtsClient("http://127.0.0.1:9882", "test")
    started = threading.Event()
    release = threading.Event()
    requests: list[tuple[str, dict, float | None]] = []

    def request(path, *, payload=None, method="POST", timeout=None):
        requests.append((path, payload, timeout))
        if path == "/v1/synthesize":
            started.set()
            assert release.wait(2)
            raise SynthesisError("cancelled")
        return b'{"status":"ok"}', "application/json"

    def synthesize() -> None:
        try:
            client.synthesize(Utterance("取消的台词", request_id="utterance-original"), VoiceProfile("旁白"), tmp_path / "cancelled.wav")
        except SynthesisError:
            pass

    with patch.object(client, "_request", side_effect=request):
        client.cancel()
        assert requests == []
        worker = threading.Thread(target=synthesize)
        worker.start()
        try:
            assert started.wait(2)
            client.cancel()
            assert requests[1][0] == "/v1/cancel"
            assert requests[1][1]["request_id"] == requests[0][1]["request_id"]
            assert requests[1][1]["request_id"] != "utterance-original"
            assert requests[1][2] == 2.0
        finally:
            release.set()
            worker.join(2)
        assert not worker.is_alive()
        client.cancel()
        assert len(requests) == 2
        assert client._active_request_id is None
        assert not (tmp_path / "cancelled.wav").exists()


def test_qwen3_http_sends_reference_and_authentication(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x00\x00" * 240)
    wav_bytes = buffer.getvalue()
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/synthesize"
            assert self.headers["Authorization"] == "Bearer local-test-token"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(wav_bytes)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Qwen3TtsClient(f"http://127.0.0.1:{server.server_port}", "local-test-token")
        profile = VoiceProfile("旁白", reference_audio="D:/audio/reference.wav", prompt_text="实际参考台词", reference_language="zh", speed=1.1)
        target = tmp_path / "试听.wav"
        client.synthesize(Utterance("新的中文台词"), profile, target)
        assert target.read_bytes() == wav_bytes
        assert seen[0]["text"] == "新的中文台词"
        assert seen[0]["ref_audio_path"] == profile.reference_audio
        assert seen[0]["prompt_text"] == profile.prompt_text
        assert seen[0]["speed"] == 1.1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_qwen3_diagnostics_require_reference_profile(tmp_path: Path) -> None:
    config = ready_config()
    client = MagicMock()
    client.health.return_value = {"status": "ok", "backend": "qwen3-tts"}
    with patch("wuwa_dubber.diagnostics._module", side_effect=lambda name, module: DiagnosticItem(name, "ok", "test")), patch("wuwa_dubber.diagnostics._command", side_effect=lambda name, configured, **kwargs: DiagnosticItem(name, "ok", "test")), patch("wuwa_dubber.diagnostics._nvidia", return_value=DiagnosticItem("GPU", "ok", "test")), patch("wuwa_dubber.tts.create_tts_client", return_value=client):
        items = collect_diagnostics(config, VoiceProfileStore(AppPaths(tmp_path)))
    item = next(item for item in items if item.name == "声线档案")
    assert item.status == "warning"
    assert "系统朗读无需" not in item.detail
