"""Build a source/install ZIP from a strict file allowlist; never package local data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXED_FILES = (
    ".gitignore", ".gitattributes", "README.md", "THIRD_PARTY_NOTICES.md",
    "docs/latency-benchmarks.json",
    "安装基础环境.bat", "安装角色配音模型.bat", "启动AI配音.bat", "打开设置.bat", "停止本地模型.bat",
    "wuwa-dubber/README.md", "wuwa-dubber/使用说明.txt", "wuwa-dubber/pyproject.toml", "wuwa-dubber/constraints.txt",
    "wuwa-dubber/start-ai-dubbing.bat", "wuwa-dubber/start-wuwa-dubber.bat",
    "local-voice-model/README.md", "local-voice-model/qwen_server.py",
    "local-voice-model/download_models.py", "local-voice-model/requirements.txt",
    "local-voice-model/requirements-lock.txt", "local-voice-model/test_server_protocol.py",
    "local-voice-model/启动本地模型.bat", "local-voice-model/停止本地模型.bat",
)
SOURCE_DIRS = {
    "docs": {".md"}, "scripts": {".py", ".ps1"},
    "wuwa-dubber/src/wuwa_dubber": {".py"}, "wuwa-dubber/scripts": {".py", ".ps1"},
    "wuwa-dubber/tests": {".py"}, "wuwa-dubber/assets": {".ico"},
    "local-voice-model/scripts": {".ps1"},
}


def source_files(root: Path = ROOT) -> list[Path]:
    selected = {root / name for name in FIXED_FILES}
    for folder, extensions in SOURCE_DIRS.items():
        selected.update(p for p in (root / folder).rglob("*")
                        if p.is_file() and p.suffix in extensions and "__pycache__" not in p.parts)
    for path in selected:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or unsafe release input: {path.relative_to(root)}")
    return sorted(selected, key=lambda p: p.relative_to(root).as_posix())


def build(root: Path = ROOT, output: Path | None = None) -> dict:
    version = tomllib.loads((root / "wuwa-dubber/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    name = "game-dubbing-v" + version
    output = output or root / "releases"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (name + ".zip")
    records = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in source_files(root):
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            record = {"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            records.append(record)
            entry = zipfile.ZipInfo(name + "/" + relative, date_time=(2026, 10, 30, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, data)
        manifest = {"version": version, "files": records, "includes_model_weights": False,
                    "includes_personal_data": False}
        entry = zipfile.ZipInfo(name + "/FILE_MANIFEST.json", date_time=(2026, 10, 30, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(entry, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "SHA256SUMS.txt").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity validation failed")
        for item in records:
            data = bundle.read(name + "/" + item["path"])
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Archive manifest mismatch")
    return {"file": archive.name, "files": len(records), "bytes": archive.stat().st_size, "sha256": checksum}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(output=args.output), ensure_ascii=False, indent=2))
