"""Create per-installation settings; preserve and back up existing user settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wuwa-dubber/src"))
from wuwa_dubber.config import AppConfig


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure(root: Path, *, with_model: bool = False, ffmpeg: str | None = None) -> dict:
    app_root = root / "wuwa-dubber"
    data = app_root / "data"
    config_file = data / "config.json"
    runtime_file = data / "model_runtime.json"
    # Invalid existing JSON must be repaired explicitly; never silently replace it.
    config = (json.loads(config_file.read_text(encoding="utf-8-sig"))
              if config_file.exists() else AppConfig().to_dict())
    if not isinstance(config, dict):
        raise ValueError("Existing application settings must be a JSON object")
    original = dict(config)
    if config.get("tts_token", "") in ("", "local-only-change-me"):
        config["tts_token"] = secrets.token_urlsafe(32)
    if with_model:
        config.update(tts_backend="qwen3-tts", tts_endpoint="http://127.0.0.1:9882", use_fake_tts=False)
        if ffmpeg:
            if not Path(ffmpeg).is_file():
                raise ValueError("The installed FFmpeg executable was not found")
            config["ffmpeg_path"] = str(Path(ffmpeg).resolve())
    changed = not config_file.exists() or original != config
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    backup = app_root / "backups" / ("before-install-" + stamp)
    for path, needed in ((config_file, changed), (runtime_file, with_model)):
        if needed and path.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / path.name)
    if changed:
        write_json(config_file, config)
    if with_model:
        write_json(runtime_file, {"model_root": "../local-voice-model"})
    for name in ("profiles", "cache", "logs"):
        (data / name).mkdir(parents=True, exist_ok=True)
    return {"backend": config.get("tts_backend"), "settings_updated": changed,
            "backup_created": backup.is_dir()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    print(json.dumps(configure(ROOT, with_model=args.with_model, ffmpeg=args.ffmpeg), ensure_ascii=False))


if __name__ == "__main__":
    main()
