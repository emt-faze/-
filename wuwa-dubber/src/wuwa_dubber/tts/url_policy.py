"""The dubbing backend is deliberately restricted to this computer."""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request


def validate_local_url(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except (TypeError, ValueError) as exc:
        raise ValueError("TTS 地址格式无效") from exc
    if (
        parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or any(char.isspace() for char in url)
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("TTS 地址必须是本机 HTTP 地址（127.0.0.1、localhost 或 ::1）")
    return url.rstrip("/")


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("本地 TTS 接口不允许重定向")


def local_opener() -> urllib.request.OpenerDirector:
    # No inherited HTTP proxy: even loopback requests must remain local.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects())
