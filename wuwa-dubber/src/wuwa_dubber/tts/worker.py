from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from .url_policy import local_opener, validate_local_url


class TtsWorkerError(RuntimeError):
    pass


class UpstreamGptSoVITSBackend:
    """Proxy to a separately installed GPT-SoVITS API v2 service."""

    def __init__(self, upstream_url: str, *, timeout: float = 120.0) -> None:
        self.upstream_url = validate_local_url(upstream_url)
        self._opener = local_opener()
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.upstream_url}/docs", method="GET")
        try:
            with self._opener.open(request, timeout=5) as response:
                return {"status": "ok", "upstream": self.upstream_url, "http_status": response.status}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"status": "degraded", "upstream": self.upstream_url, "detail": str(exc)}

    def synthesize(self, payload: dict[str, Any]) -> bytes:
        text = str(payload.get("text", "")).strip()
        ref_audio = Path(str(payload.get("ref_audio_path", "")))
        if not text:
            raise TtsWorkerError("text is required")
        if not ref_audio.is_file():
            raise TtsWorkerError(f"reference audio not found: {ref_audio}")
        request_payload = {
            "text": text,
            "text_lang": str(payload.get("text_lang", "zh")),
            "ref_audio_path": str(ref_audio),
            "prompt_text": str(payload.get("prompt_text", "")),
            "prompt_lang": str(payload.get("prompt_lang", "zh")),
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
            "speed_factor": float(payload.get("speed", 1.0)),
        }
        request = urllib.request.Request(
            f"{self.upstream_url}/tts",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read()
                content_type = response.headers.get_content_type()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TtsWorkerError(f"GPT-SoVITS HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TtsWorkerError(f"Unable to reach GPT-SoVITS: {exc}") from exc
        if content_type == "application/json":
            raise TtsWorkerError(body.decode("utf-8", errors="replace")[:500])
        if not body.startswith(b"RIFF"):
            raise TtsWorkerError("GPT-SoVITS did not return WAV audio")
        return body


class TtsRequestHandler(BaseHTTPRequestHandler):
    backend: UpstreamGptSoVITSBackend
    token: str = ""
    server_version = "WuwaDubberTts/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/v1/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            return
        self._json(HTTPStatus.OK, self.backend.health())

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/reload":
            if not self._authorized():
                return
            self._json(HTTPStatus.OK, self.backend.health())
            return
        if self.path != "/v1/synthesize":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            wav = self.backend.synthesize(payload)
        except (ValueError, json.JSONDecodeError, TtsWorkerError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[tts-worker] {self.address_string()} {format % args}", flush=True)

    def _authorized(self) -> bool:
        if not self.token:
            return True
        expected = f"Bearer {self.token}"
        if self.headers.get("Authorization") != expected:
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False
        return True

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(host: str, port: int, *, upstream_url: str, token: str) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("The TTS worker may only bind to 127.0.0.1")
    handler = type("ConfiguredTtsRequestHandler", (TtsRequestHandler,), {
        "backend": UpstreamGptSoVITSBackend(upstream_url),
        "token": token,
    })
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GPT-SoVITS proxy worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WUWA_TTS_PORT", "9881")))
    parser.add_argument("--upstream", default=os.environ.get("WUWA_TTS_UPSTREAM", "http://127.0.0.1:9880"))
    parser.add_argument("--token", default=os.environ.get("WUWA_TTS_TOKEN", "local-only-change-me"))
    args = parser.parse_args()
    server = create_server(args.host, args.port, upstream_url=args.upstream, token=args.token)
    print(f"TTS worker listening on http://{args.host}:{args.port}", flush=True)
    print(f"GPT-SoVITS upstream: {args.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
