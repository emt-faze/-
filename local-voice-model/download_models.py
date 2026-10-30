"""Download the public Qwen voice model into this deployment, without a global cache.

Uses a pinned Hugging Face commit, resumable .part files and upstream checksums.
No authentication token is needed or read. Re-running resumes this deployment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
LOGS = ROOT / "logs"
MANIFEST = MODEL / "download-manifest.json"
METADATA = LOGS / "model-source-metadata.json"
LOCK = threading.Lock()
USER_AGENT = "local-game-dubbing/0.4.0"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def log(message: str) -> None:
    line = f"{now()} {message}"
    with LOCK:
        print(line, flush=True)
        with (LOGS / "model-download.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def fetch_json(url: str) -> dict:
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=35) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log(f"Metadata attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Metadata fetch failed")


def wanted(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe upstream path: {name}")
    if name == "README.md":
        return True
    if len(path.parts) == 1:
        return path.suffix in {".json", ".safetensors"} or name in {"vocab.txt", "merges.txt"}
    return path.parts[0] == "speech_tokenizer" and path.suffix in {".json", ".safetensors"}


def hashes(path: Path) -> tuple[str, str, int]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest(), size


def validate(path: Path, entry: dict) -> dict:
    actual_sha, actual_git, actual_size = hashes(path)
    expected_size = entry.get("expected_size")
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(f"Size mismatch: {entry['path']}: {actual_size} != {expected_size}")
    expected_sha = entry.get("expected_sha256")
    expected_git = entry.get("git_blob_sha1")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch: {entry['path']}")
    if not expected_sha and expected_git and actual_git != expected_git:
        raise ValueError(f"Git blob SHA1 mismatch: {entry['path']}")
    return {"sha256": actual_sha, "git_blob_sha1_actual": actual_git,
            "size": actual_size, "verified": True, "verified_at": now()}


def download_urllib(url: str, partial: Path, expected_size: int | None) -> None:
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        mode = "wb"
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"Invalid resume range: {content_range}")
            mode = "ab"
        with partial.open(mode) as target:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                target.write(block)


def download_curl(url: str, partial: Path, expected_size: int | None) -> None:
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise RuntimeError("curl is unavailable")
    command = [executable, "--fail", "--location", "--silent", "--show-error",
               "--connect-timeout", "30", "--max-time", "7200", "--retry", "3",
               "--retry-delay", "3", "--continue-at", "-", "--output", str(partial), url]
    result = subprocess.run(command, capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr[-500:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="Previously saved official HF model metadata")
    parser.add_argument("--transport", choices=["urllib", "curl"], default="urllib")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--modelscope-fallback", action="store_true",
                        help="Allow the official ModelScope Qwen mirror, checked against HF checksums")
    parser.add_argument("--prefer-modelscope", action="store_true",
                        help="Use the official ModelScope mirror first, with the pinned HF checksums")
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    elif METADATA.is_file():
        metadata = json.loads(METADATA.read_text(encoding="utf-8-sig"))
    else:
        revision_path = f"/revision/{old_manifest['revision'] if old_manifest else REVISION}"
        metadata = fetch_json(f"https://huggingface.co/api/models/{REPO}{revision_path}?blobs=true")
        save_json(METADATA, metadata)
    revision = metadata.get("sha")
    if metadata.get("id") != REPO or revision != REVISION:
        raise ValueError("Metadata must identify the expected public model and an exact 40-character commit")
    if old_manifest and old_manifest.get("revision") != revision:
        raise ValueError("Existing deployment uses a different pinned revision; refusing to mix files")
    entries = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if not wanted(name):
            continue
        lfs = item.get("lfs") or {}
        size = lfs.get("size", item.get("size"))
        digest = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:") or None
        if name.endswith(".safetensors") and (not digest or size is None):
            raise ValueError(f"Missing official LFS size/SHA256 for {name}")
        entries.append({"path": name, "expected_size": size, "expected_sha256": digest,
                        "git_blob_sha1": item.get("blobId"),
                        "source_url": f"https://huggingface.co/{REPO}/resolve/{revision}/{urllib.parse.quote(name)}",
                        "status": "pending"})
    names = {item["path"] for item in entries}
    if not {"model.safetensors", "config.json", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json"}.issubset(names):
        raise ValueError("Official file list does not include the required main model and speech tokenizer")
    manifest = {"repository": REPO, "revision": revision,
                "metadata_url": f"https://huggingface.co/api/models/{REPO}/revision/{revision}?blobs=true",
                "created_at": (old_manifest or {}).get("created_at", now()), "updated_at": now(),
                "status": "downloading", "files": entries}
    save_json(MANIFEST, manifest)
    log(f"Pinned {REPO}@{revision}: {len(entries)} files, {sum(e['expected_size'] or 0 for e in entries):,} bytes")

    def persist() -> None:
        with LOCK:
            manifest["updated_at"] = now()
            save_json(MANIFEST, manifest)

    def received_bytes() -> int:
        total = 0
        for entry in entries:
            target = MODEL / entry["path"]
            # A worker may atomically rename .part between the two stat calls.
            for candidate in (target, target.with_name(target.name + ".part"), target):
                try:
                    total += candidate.stat().st_size
                    break
                except FileNotFoundError:
                    continue
        return total

    def transfer(entry: dict) -> None:
        target = MODEL.joinpath(*PurePosixPath(entry["path"]).parts)
        if not target.resolve().is_relative_to(MODEL.resolve()):
            raise ValueError("Download target escapes the model directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        if target.is_file():
            try:
                entry.update(validate(target, entry)); entry["status"] = "verified"
                persist(); log(f"Already verified {entry['path']}"); return
            except ValueError:
                target.replace(target.with_name(target.name + f".invalid-{int(time.time())}"))
        sources = [entry["source_url"]]
        if args.modelscope_fallback or args.prefer_modelscope:
            sources.append(f"https://modelscope.cn/models/{REPO}/resolve/master/{urllib.parse.quote(entry['path'])}")
        if args.prefer_modelscope:
            sources.reverse()
        failure = None
        for attempt in range(8):
            url = sources[min(attempt // 3, len(sources) - 1)]
            try:
                size = partial.stat().st_size if partial.is_file() else 0
                if size != entry["expected_size"]:
                    if entry["expected_size"] is not None and size > entry["expected_size"]:
                        partial.replace(partial.with_name(partial.name + f".oversize-{int(time.time())}"))
                    log(f"Downloading {entry['path']} (resume {size:,} bytes; source {urllib.parse.urlsplit(url).hostname})")
                    (download_curl if args.transport == "curl" else download_urllib)(url, partial, entry["expected_size"])
                checked = validate(partial, entry)
                partial.replace(target)
                entry.update(checked, status="verified", downloaded_from=url)
                persist(); log(f"Verified {entry['path']}: {entry['size']:,} bytes SHA256 {entry['sha256']}"); return
            except Exception as exc:
                failure = exc
                if isinstance(exc, ValueError) and partial.is_file():
                    partial.replace(partial.with_name(partial.name + f".invalid-{int(time.time())}"))
                log(f"Retry {attempt + 1} for {entry['path']}: {type(exc).__name__}: {exc}")
                time.sleep(min(2 ** attempt, 15))
        entry.update(status="failed", error=f"{type(failure).__name__}: {failure}")
        persist()
        raise RuntimeError(f"Could not download {entry['path']}") from failure

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
        futures = {pool.submit(transfer, entry): entry for entry in entries}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception as exc:
                    failures.append(str(exc))
            if pending:
                received = received_bytes()
                log(f"Progress: {received / 1024**2:.1f} MiB; {len(entries) - len(pending)}/{len(entries)} tasks finished")
    manifest["status"] = "failed" if failures else "complete"
    manifest["errors"] = failures
    manifest["completed_at"] = now() if not failures else None
    persist()
    save_json(LOGS / "model-download-report.json", manifest)
    log(f"Download {manifest['status']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
