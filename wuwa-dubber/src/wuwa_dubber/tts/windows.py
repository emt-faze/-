"""Offline Chinese Windows speech. This is ordinary speech, not voice cloning."""
from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..models import Utterance, VoiceProfile
from .client import SynthesisError


# Text and file paths are JSON data on stdin, never interpolated into code.
# SAPI COM works with the desktop voices already installed on Windows.
_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$stream = $null
$voice = $null
try {
    $payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $voice = New-Object -ComObject SAPI.SpVoice
    $tokens = $voice.GetVoices()
    $chosen = $null
    $available = @()
    for ($i = 0; $i -lt $tokens.Count; $i++) {
        $token = $tokens.Item($i)
        $language = $token.GetAttribute('Language')
        $isChinese = $false
        foreach ($code in ($language -split ';')) {
            if (([Convert]::ToInt32($code, 16) -band 1023) -eq 4) { $isChinese = $true }
        }
        if ($isChinese) {
            $available += @{name=$token.GetDescription(); id=$token.Id; language=$language}
            if ($null -eq $chosen -or $language -eq '804') { $chosen = $token }
        }
        if ($payload.voice_id -and $token.Id -eq $payload.voice_id -and $isChinese) {
            $requested = $token
        }
    }
    if ($payload.voice_id) {
        if ($null -eq $requested) { throw 'The selected Chinese Windows voice is no longer available.' }
        $chosen = $requested
    }
    if ($null -eq $chosen) { throw 'No Chinese desktop speech voice is installed in Windows.' }
    $voice.Voice = $chosen
    if ($payload.action -eq 'synthesize') {
        $voice.Rate = [int]$payload.rate
        $voice.Volume = 100
        $stream = New-Object -ComObject SAPI.SpFileStream
        $stream.Format.Type = 22
        $stream.Open([string]$payload.output, 3, $false)
        $voice.AudioOutputStream = $stream
        # SVSFIsNotXML: text containing <...> is spoken as text, not interpreted.
        [void]$voice.Speak([string]$payload.text, 16)
        $stream.Close()
        $stream = $null
    }
    @{status='ok'; backend='windows-sapi'; voice=$chosen.GetDescription(); voice_id=$chosen.Id;
      language=$chosen.GetAttribute('Language'); voices=$available; sample_rate=22050;
      note='Windows Chinese speech; not character voice cloning'} | ConvertTo-Json -Depth 5 -Compress
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($null -ne $stream) { $stream.Close() }
    if ($null -ne $voice) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($voice) }
}
'''


class WindowsTtsClient:
    def __init__(self, *, timeout: float = 45.0) -> None:
        self.timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._health: dict[str, Any] | None = None

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        if os.name != "nt":
            raise SynthesisError("Windows 系统朗读只支持 Windows")
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        executable = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        command = [str(executable), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                   base64.b64encode(_SCRIPT.encode("utf-16-le")).decode("ascii")]
        with self._call_lock:
            try:
                with self._process_lock:
                    process = subprocess.Popen(
                        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self._process = process
                stdout, stderr = process.communicate(
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=self.timeout,
                )
                if process.returncode:
                    detail = stderr.decode("utf-8", errors="replace").strip()[:600]
                    raise SynthesisError(f"Windows 系统朗读失败或已取消：{detail or process.returncode}")
                result = json.loads(stdout.decode("utf-8-sig"))
                if not isinstance(result, dict) or result.get("status") != "ok":
                    raise SynthesisError("Windows 系统朗读返回了无效状态")
                return result
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise SynthesisError(f"Windows 系统朗读超过 {self.timeout:g} 秒，已停止") from exc
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise SynthesisError(f"无法运行 Windows 系统朗读：{exc}") from exc
            finally:
                with self._process_lock:
                    self._process = None

    def health(self) -> dict[str, Any]:
        self._health = self._invoke({"action": "health"})
        return dict(self._health)

    def cache_identity(self) -> dict[str, Any]:
        health = self._health or self.health()
        return {"backend": "windows-sapi", "voice": health["voice_id"],
                "version": 1, "format": "pcm16-mono-22050"}

    def cancel(self) -> None:
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.terminate()
                except OSError:
                    pass  # Process may have completed just before cancellation.

    def synthesize(self, utterance: Utterance, profile: VoiceProfile, output: Path) -> float:
        started = time.monotonic()
        if not utterance.text.strip():
            raise SynthesisError("朗读文字不能为空")
        if not math.isfinite(profile.speed) or not 0.5 <= profile.speed <= 2.0:
            raise SynthesisError("语速必须位于 0.5 到 2.0 之间")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}-{uuid4().hex}.tmp.wav")
        try:
            result = self._invoke({
                "action": "synthesize", "text": utterance.text, "output": str(temporary),
                "rate": max(-10, min(10, round(10 * math.log2(profile.speed)))),
                "voice_id": (self._health or {}).get("voice_id", ""),
            })
            with wave.open(str(temporary), "rb") as handle:
                if handle.getsampwidth() != 2 or handle.getnframes() <= 0 or handle.getcomptype() != "NONE":
                    raise SynthesisError("Windows 系统朗读未生成有效 PCM16 音频")
            temporary.replace(output)
            self._health = result
        except (wave.Error, EOFError, OSError) as exc:
            raise SynthesisError(f"无法保存朗读音频：{exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return time.monotonic() - started
