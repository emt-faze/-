from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from wuwa_dubber.tts.client import GPTSoVITSClient, SynthesisError
from wuwa_dubber.tts.url_policy import validate_local_url


def test_tts_endpoint_must_be_exact_loopback_host() -> None:
    for url in ("http://127.0.0.1:9881", "http://localhost:9881/", "http://[::1]:9881"):
        assert validate_local_url(url) == url.rstrip("/")
    for url in (
        "http://127.0.0.1.example.com:9881", "http://localhost.attacker.test", "https://example.com",
        "http://localhost@other.test", "http://other.test@localhost", "file:///test",
        "http://localhost:0", "http://localhost:99999", "http://localhost?redirect=1",
        "http://localhost/#fragment", "http://local\nhost", "https://localhost:9881",
    ):
        try:
            GPTSoVITSClient(url, "secret")
        except ValueError:
            pass
        else:
            raise AssertionError(f"nonlocal/invalid endpoint accepted: {url}")


def test_tts_degraded_and_invalid_health_fail() -> None:
    client = GPTSoVITSClient("http://127.0.0.1:9881", "secret")
    for body in (b'{"status":"degraded","detail":"model missing"}', b"[]", b"null", b"broken"):
        with patch.object(client, "_request", return_value=(body, "application/json")):
            try:
                client.health()
            except SynthesisError:
                pass
            else:
                raise AssertionError("unready health must fail")
    with patch.object(client, "_request", return_value=(b'{"status":"ok"}', "application/json")):
        assert client.health()["status"] == "ok"


def test_tts_http_redirect_is_rejected_without_following() -> None:
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirected")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = GPTSoVITSClient(f"http://127.0.0.1:{server.server_port}", "secret", timeout=2)
        try:
            client.health()
        except SynthesisError as exc:
            assert "重定向" in str(exc)
        else:
            raise AssertionError("redirect was followed")
        assert paths == ["/v1/health"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
