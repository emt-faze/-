from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from wuwa_dubber.tts.worker import TtsRequestHandler


def test_worker_rejects_nonlocal_upstream() -> None:
    from wuwa_dubber.tts.worker import UpstreamGptSoVITSBackend
    for url in ("https://example.com", "http://127.0.0.1.example.com:9880", "http://localhost@remote.invalid"):
        try:
            UpstreamGptSoVITSBackend(url)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonlocal upstream accepted")


class FakeBackend:
    def health(self) -> dict[str, str]:
        return {"status": "ok", "backend": "fake"}

    def synthesize(self, payload: dict[str, object]) -> bytes:
        assert payload["text"] == "测试"
        return b"RIFF" + b"\x00" * 40


def test_tts_worker_health_and_synthesis() -> None:
    handler = type(
        "TestTtsHandler",
        (TtsRequestHandler,),
        {"backend": FakeBackend(), "token": "secret"},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{base}/v1/health", headers=headers)) as response:
            health = json.loads(response.read())
        assert health["status"] == "ok"

        request = urllib.request.Request(
            f"{base}/v1/synthesize",
            data=json.dumps({"text": "测试"}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.headers.get_content_type() == "audio/wav"
            assert response.read().startswith(b"RIFF")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
