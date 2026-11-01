from __future__ import annotations

import os
import shutil
from pathlib import Path


def discover_ffmpeg() -> tuple[str, str]:
    executable = shutil.which("ffmpeg")
    if executable:
        path = Path(executable)
        probe = path.with_name("ffprobe.exe")
        return str(path), str(probe if probe.is_file() else "ffprobe")

    roots = _candidate_roots()
    for root in roots:
        for match in _search(root, "ffmpeg.exe", max_depth=4):
            probe = match.with_name("ffprobe.exe")
            return str(match), str(probe if probe.is_file() else "ffprobe")
    return "ffmpeg", "ffprobe"


def discover_obs() -> str:
    executable = shutil.which("obs64")
    if executable:
        return executable
    registry_path = _registry_install_path()
    if registry_path:
        candidate = registry_path / "bin" / "64bit" / "obs64.exe"
        if candidate.is_file():
            return str(candidate)
        for match in _search(registry_path, "obs64.exe", max_depth=4):
            return str(match)
    return ""


def _candidate_roots() -> list[Path]:
    values = [
        os.environ.get("LOCALAPPDATA", ""),
        os.environ.get("USERPROFILE", ""),
        os.environ.get("ProgramFiles", ""),
        os.environ.get("ProgramFiles(x86)", ""),
        fr"{os.environ.get('LOCALAPPDATA', '')}\Programs",
        "E:\\ffmpeg",
        "F:\\ffmpeg",
        "E:\\tools",
        "F:\\tools",
        "E:\\",
        "F:\\",
        "C:\\",
    ]
    roots: list[Path] = []
    for value in values:
        path = Path(value) if value else None
        if path and path not in roots:
            roots.append(path)
    return roots


def _search(root: Path, filename: str, *, max_depth: int) -> list[Path]:
    if not root.is_dir():
        return []
    root_depth = len(root.parts)
    matches: list[Path] = []
    try:
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.parts) - root_depth
            if depth >= max_depth:
                directories.clear()
            directories[:] = [
                name
                for name in directories
                if name.lower()
                not in {
                    "$recycle.bin",
                    "windows",
                    "winsxs",
                    "system volume information",
                    ".git",
                    ".venv",
                    "node_modules",
                }
            ]
            if filename in files:
                matches.append(current_path / filename)
                break
    except OSError:
        return matches
    return matches


def _registry_install_path() -> Path | None:
    try:
        import winreg
    except ImportError:
        return None
    for hive, key_name in (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\OBS Studio"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\OBS Studio"),
        (winreg.HKEY_CURRENT_USER, r"Software\OBS Studio"),
    ):
        try:
            with winreg.OpenKey(hive, key_name) as key:
                value, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        path = Path(str(value))
        if path.is_dir():
            return path
    return None
