from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    def __init__(self, log_dir: Path, *, log_ocr_text: bool = True) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"session-{stamp}.jsonl"
        self.log_ocr_text = log_ocr_text
        self._logger = logging.getLogger("wuwa_dubber.events")

    def write(self, event: str, **fields: Any) -> None:
        payload = {"time": time.time(), "event": event, **fields}
        if not self.log_ocr_text and "text" in payload:
            payload["text"] = "[redacted]"
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._logger.warning("Unable to write event log: %s", exc)
