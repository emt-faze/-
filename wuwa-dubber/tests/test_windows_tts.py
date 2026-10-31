from __future__ import annotations

import base64
import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from wuwa_dubber.config import AppConfig
from wuwa_dubber.models import Utterance, VoiceProfile
from wuwa_dubber.tts import SynthesisError, WindowsTtsClient, create_tts_client


def test_windows_backend_selected_without_reference_audio() -> None:
    assert isinstance(create_tts_client(AppConfig(tts_backend="windows-sapi")), WindowsTtsClient)


def test_windows_script_never_interpolates_user_text() -> None:
    text = "你好。<test> `quotes` $() '中文' \"括号\"\n第二行"
    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = (b'{"status":"ok"}', b"")
    with patch("wuwa_dubber.tts.windows.subprocess.Popen", return_value=process) as launch:
        WindowsTtsClient()._invoke({"action": "synthesize", "text": text})
    command = launch.call_args.args[0]
    script = base64.b64decode(command[-1]).decode("utf-16-le")
    assert text not in script
    assert "Speak([string]$payload.text, 16)" in script
    assert json.loads(process.communicate.call_args.args[0])["text"] == text
    assert "-NoProfile" in command


def test_windows_synthesis_writes_valid_wav_and_keeps_old_file_on_failure(tmp_path: Path) -> None:
    client = WindowsTtsClient()
    seen: list[dict] = []

    def synthesize(payload):
        seen.append(payload)
        with wave.open(payload["output"], "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(22050)
            handle.writeframes(b"\0\0" * 2205)
        return {"status": "ok", "voice_id": "huihui"}

    output = tmp_path / "中文.wav"
    with patch.object(client, "_invoke", side_effect=synthesize):
        client.synthesize(Utterance("你好"), VoiceProfile("旁白", speed=1.0), output)
    original = output.read_bytes()
    assert seen[0]["rate"] == 0
    assert str(output) != seen[0]["output"]
    assert client.cache_identity()["voice"] == "huihui"
    with patch.object(client, "_invoke", side_effect=SynthesisError("cancelled")):
        try:
            client.synthesize(Utterance("第二句"), VoiceProfile("旁白"), output)
        except SynthesisError:
            pass
        else:
            raise AssertionError("cancellation must be reported")
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp.wav"))


def test_windows_timeout_kills_process_and_cancel_is_nonblocking() -> None:
    client = WindowsTtsClient(timeout=0.2)
    process = MagicMock()
    process.communicate.side_effect = [subprocess.TimeoutExpired("powershell", 0.2), (b"", b"")]
    with patch("wuwa_dubber.tts.windows.subprocess.Popen", return_value=process):
        try:
            client.health()
        except SynthesisError as exc:
            assert "已停止" in str(exc)
        else:
            raise AssertionError("timeout must fail")
    process.kill.assert_called_once()
    assert client._process is None
    active = MagicMock()
    active.poll.return_value = None
    client._process = active
    client.cancel()
    active.terminate.assert_called_once()
    active.wait.assert_not_called()


def test_windows_rejects_empty_text_and_invalid_speed(tmp_path: Path) -> None:
    client = WindowsTtsClient()
    with patch.object(client, "_invoke") as invoke:
        for text, speed in (("  ", 1.0), ("你好", float("nan")), ("你好", 9.0)):
            try:
                client.synthesize(Utterance(text), VoiceProfile("旁白", speed=speed), tmp_path / "test.wav")
            except SynthesisError:
                pass
            else:
                raise AssertionError("invalid synthesis request accepted")
        invoke.assert_not_called()


def test_character_preset_does_not_change_default_voice(tmp_path: Path) -> None:
    client = WindowsTtsClient()
    client._health = {"voice_id": "default", "default_voice_id": "default"}
    requests = []

    def synthesize(payload):
        requests.append(payload)
        with wave.open(payload["output"], "wb") as handle:
            handle.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            handle.writeframes(b"\1\0" * 220)
        return {"status": "ok", "voice_id": payload["voice_id"] or "default",
                "default_voice_id": "default"}

    with patch.object(client, "_invoke", side_effect=synthesize):
        client.synthesize(Utterance("男声"), VoiceProfile("男声", system_voice="male", system_pitch=-3), tmp_path / "male.wav")
        client.synthesize(Utterance("旁白"), VoiceProfile("旁白"), tmp_path / "default.wav")
    assert requests[0]["voice_id"] == "male" and requests[0]["pitch"] == -3
    assert requests[1]["voice_id"] == "" and requests[1]["pitch"] == 0
    assert client.cache_identity()["voice"] == "default"


def test_invalid_pitch_never_launches_synthesis(tmp_path: Path) -> None:
    client = WindowsTtsClient()
    with patch.object(client, "_invoke") as invoke:
        for pitch in (11, -11, 1.5, True):
            try:
                client.synthesize(Utterance("你好"), VoiceProfile("旁白", system_pitch=pitch), tmp_path / "test.wav")
            except SynthesisError:
                pass
            else:
                raise AssertionError("invalid pitch accepted")
        invoke.assert_not_called()
