"""Join reviewed OCR events to the shared character voice and WAV cache.

OCR/UI calls never wait for synthesis or a network cancellation request. One
consumer owns the speech client; at most the latest caption is pending.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parent


def resolved_path(value, base=ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def voice_paths(options):
    code = resolved_path(options.get("code_root", "../wuwa-dubber"))
    data = resolved_path(options.get("data_root", str(code / "data")))
    if not (code / "src/wuwa_dubber/speech_session.py").is_file():
        raise ValueError(f"找不到配音程序：{code}，请检查配音路径配置")
    if not (data / "config.json").is_file():
        raise ValueError(f"找不到声线设置：{data / 'config.json'}")
    source = str(code / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return code, data


class VoiceRuntime:
    def __init__(self, options):
        self.options = dict(options)
        self.code_root, self.data_root = voice_paths(options)
        from wuwa_dubber.config import AppConfig
        from wuwa_dubber.models import VoiceProfile
        from wuwa_dubber.speaker_matcher import SpeakerMatcher
        from wuwa_dubber.tts import create_tts_client
        from wuwa_dubber.audio.player import AudioPlayer
        # Read the existing settings without rewriting their capture settings.
        self.config = AppConfig.from_dict(json.loads((self.data_root / "config.json").read_text(encoding="utf-8-sig")))
        if options.get("mode") == "system":
            self.config = replace(self.config, tts_backend="windows-sapi", use_fake_tts=False)
        if self.config.use_fake_tts:
            raise ValueError("当前声线设置使用测试音，请先在配音设置中选择真实语音服务")
        self.system_mode = self.config.tts_backend == "windows-sapi"
        self.profiles = {}
        for path in sorted((self.data_root / "profiles").glob("*/profile.json")):
            try:
                profile = VoiceProfile.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
            except (ValueError, TypeError, OSError):
                if self.system_mode:
                    continue  # Optional clone references cannot block system speech.
                raise
            if not profile.enabled:
                continue
            profile = replace(profile, reference_audio=str(profile.resolve_reference(self.data_root / "profiles").resolve()))
            errors = profile.validate(self.data_root / "profiles", require_audio=self.config.tts_backend != "windows-sapi")
            if errors and not self.system_mode:
                raise ValueError(f"声线 {profile.speaker} 无法使用：" + "；".join(errors))
            self.profiles[profile.speaker] = profile
        if not self.system_mode and self.config.narrator_profile not in self.profiles:
            raise ValueError(f"缺少已启用的默认声线：{self.config.narrator_profile}，请在声线设置中选择")
        self.matcher = SpeakerMatcher(list(self.profiles.values()), threshold=self.config.speaker_match_threshold)
        self.client = create_tts_client(self.config)
        from system_voices import SystemVoiceRouter
        self.system_router = SystemVoiceRouter(options, self.profiles.values())
        self.fallback_enabled = bool(options.get("fallback_to_system", False)) and not self.system_mode
        if self.fallback_enabled and self.config.tts_backend == "qwen3-tts":
            self.client.timeout = max(10, min(150, float(options.get("wait_seconds", 40))))
        self.player = AudioPlayer(self.config.output_device, volume=self.config.output_volume)
        self.cache_dir = self.data_root / "cache"

    def model_root(self):
        runtime = self.data_root / "model_runtime.json"
        if not runtime.is_file():
            raise ValueError(f"缺少本地模型启动配置：{runtime}")
        # Published model paths are relative to the shared app, as in launch_app.py.
        return resolved_path(json.loads(runtime.read_text(encoding="utf-8-sig"))["model_root"], self.code_root)

    def prepare(self, stop, progress, *, playback=True):
        progress("正在检查语音服务和声音输出…" if playback else "正在检查语音生成服务…")
        if stop.is_set():
            return
        # Require the volume/device-aware player instead of silently ignoring
        # the saved device/volume through the legacy winsound fallback.
        if playback:
            import sounddevice as sd
            sd.check_output_settings(device=self.config.output_device, channels=1)
        try:
            health = self.client.health()
            if self.system_mode:
                self.system_router.check_available(health)
        except Exception:
            if stop.is_set():
                return
            endpoint = self.config.tts_endpoint.rstrip("/")
            from wuwa_dubber.tts.url_policy import managed_qwen_port
            port = managed_qwen_port(endpoint)
            if self.config.tts_backend != "qwen3-tts" or port is None:
                raise
            starter = self.model_root() / "scripts/start-server.ps1"
            if not starter.is_file():
                raise ValueError(f"找不到模型启动脚本：{starter}")
            logs = ROOT / "runtime-logs"
            logs.mkdir(exist_ok=True)
            log = logs / (datetime.now().strftime("model-start-%Y%m%d-%H%M%S-%f") + ".log")
            progress("正在启动本地角色语音模型，首次加载需要等待…")
            with log.open("wb") as output:
                process = subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(starter),
                     "-AppConfig", str(self.data_root / "config.json"), "-Port", str(port)],
                    stdout=output, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                deadline = time.monotonic() + 150
                while process.poll() is None:
                    if stop.wait(.1) or time.monotonic() > deadline:
                        process.terminate()
                        process.wait(timeout=5)
                        if stop.is_set():
                            return
                        raise RuntimeError(f"等待模型超时；可在“停止本地模型”释放资源。日志：{log}")
            if process.returncode:
                raise RuntimeError(f"模型未能启动，请检查日志：{log}")
            if stop.is_set():
                return
            self.client.health()
        progress("系统中文朗读已就绪 · 按角色设置选择系统声线" if self.system_mode else
                 "语音已就绪 · 已有声线：" + "、".join(self.profiles))

    def select_profile(self, record):
        if getattr(self, "system_mode", False):
            return self.system_router.select(record)
        # An uncertain/missing name must never inherit the previous character.
        name = ""
        if record.get("speaker_status") == "verified":
            name = record.get("speaker", "")
        elif record.get("speaker_identity_confirmed"):
            name = record.get("speaker_identity", "")
        match = self.matcher.match(name)
        if match:
            return self.profiles[match.speaker], False
        return self.profiles[self.config.narrator_profile], True

    def session(self):
        from wuwa_dubber.speech_session import SpeechSession
        return SpeechSession(self.config, self.cache_dir, self.client, self.player)

    def fallback_session(self):
        from wuwa_dubber.models import VoiceProfile
        from wuwa_dubber.speech_session import SpeechSession
        from wuwa_dubber.tts.windows import WindowsTtsClient
        config = replace(self.config, tts_backend="windows-sapi", use_fake_tts=False)
        return self.system_router.default, SpeechSession(config, self.cache_dir, WindowsTtsClient(), self.player)


def join_audio(paths, destination, stop):
    """Write a complete dialogue WAV only after every segment was played."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".wav.tmp")
    try:
        fmt = None
        with wave.open(str(temporary), "wb") as output:
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    current = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    if fmt is None:
                        fmt = current
                        output.setparams(source.getparams())
                    elif fmt != current:
                        raise ValueError("配音分段格式不一致，未导出整句")
                    while data := source.readframes(16384):
                        if stop.is_set():
                            return None
                        output.writeframesraw(data)
        if stop.is_set():
            return None
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


class DubbingController:
    def __init__(self, runtime, callback, *, policy="verified"):
        if policy not in {"verified", "candidate"}:
            raise ValueError("配音策略必须为 verified 或 candidate")
        self.runtime, self.callback, self.policy = runtime, callback, policy
        self._condition = threading.Condition(threading.RLock())
        self._journal_lock = threading.Lock()
        self._directory = None
        self._current = None
        self._seen = set()
        self._submitted = set()
        self._completed = set()
        self._pending = None
        self._active = None
        self._resume_id = None
        self._closed = False
        self._paused = False
        self._thread = threading.Thread(target=self._consume, name="dialogue-dubbing", daemon=True)
        self._thread.start()

    def set_directory(self, directory):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _emit(self, state, message, record=None, **details):
        event = {"state": state, "message": message, "time": datetime.now().astimezone().isoformat(), **details}
        if record:
            event.update(dialogue_id=record["dialogue_id"], text=record["body"],
                         observed_speaker=record.get("speaker", ""), ocr_status=record.get("status"),
                         source_kind=record.get("source_kind", "unknown"))
        with self._journal_lock:
            if self._directory:
                with (self._directory / "audio.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.callback({"speech": event})

    def _cancel_locked(self):
        self._pending = None
        if self._active is not None and not self._active[1].is_set():
            self._active[1].set()
            self._active[0].set()
            self.runtime.player.stop()
        self._condition.notify_all()

    def accept(self, record):
        identity = str(record.get("dialogue_id", ""))
        if not identity or not all(c.isalnum() or c in "-_" for c in identity):
            raise ValueError("无效的台词编号")
        with self._condition:
            if self._closed:
                return
            previous_id = self._current["dialogue_id"] if self._current else None
            if identity != previous_id:
                if identity in self._seen:
                    return  # A late review cannot resurrect an older page.
                self._cancel_locked()
                self._resume_id = None
                self._seen.add(identity)
            elif record.get("revision", 0) < self._current.get("revision", 0):
                return
            self._current = dict(record)
            if self._paused or identity in self._submitted:
                return
            if not str(record.get("body", "")).strip():
                return
            if record.get("source_clipped") or record.get("coverage_warning") or record.get("review_coverage_warning"):
                self._emit("held", "当前台词可能被裁切或漏字，已保留记录，等待下一句", record)
                return
            if self.policy == "verified" and record.get("status") != "verified":
                self._emit("held", "等待台词复核一致后配音" if record.get("status") == "candidate" else "台词复核不一致，已保留记录，本句暂不配音", record)
                return
            if float(record.get("body_confidence", 0)) < .9:
                self._emit("held", "台词识别分数不足，本句暂不配音", record)
                return
            self._submitted.add(identity)
            self._pending = dict(record)
            self._emit("queued", "台词已就绪，准备对应声音…", record)
            self._condition.notify_all()

    def suspend(self, reason="画面暂不可用，已停止当前配音", *, resume_when_visible=False):
        with self._condition:
            if self._closed:
                return
            had_work = (self._active is not None and not self._active[1].is_set()) or self._pending is not None
            if self._current:
                identity = self._current["dialogue_id"]
                if resume_when_visible and identity not in self._completed and (had_work or identity not in self._submitted):
                    self._resume_id = identity
                self._submitted.add(identity)
            if not resume_when_visible:
                self._resume_id = None
            self._cancel_locked()
            if had_work:
                self._emit("cancelled", reason, self._current)

    def confirm_visible(self, observation):
        """Resume an interrupted page only after fresh, stable OCR matches it.

        Revisions alone never resume audio. A changed/blank page, user skip or
        pause invalidates this recovery path; a completed page never enters it.
        """
        with self._condition:
            if self._closed or self._paused or self._active is not None or not self._current:
                return
            if self._resume_id != self._current["dialogue_id"]:
                return
            if (observation.get("body") != self._current["body"]
                    or observation.get("speaker", "") != self._current.get("speaker", "")):
                return
            self._submitted.discard(self._resume_id)
            self._resume_id = None
            self.accept(self._current)

    def pause(self, paused=True):
        with self._condition:
            self._paused = paused
            if paused:
                self.suspend("已暂停配音，仍继续记录台词")
            self._emit("paused" if paused else "ready", "已暂停配音" if paused else "已恢复配音，等待下一句")

    def skip(self):
        self.suspend("已跳过当前台词，等待下一句")

    def retry(self):
        with self._condition:
            if self._closed or self._paused or not self._current or self._active is not None:
                return
            self._submitted.discard(self._current["dialogue_id"])
            self.accept(self._current)

    def stop(self):
        with self._condition:
            self._closed = True
            self._cancel_locked()

    def close(self, timeout=None):
        self.stop()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _consume(self):
        from wuwa_dubber.models import Utterance
        from wuwa_dubber.speech_cache import SpeechCancelled
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                record, self._pending = self._pending, None
                cancel, externally_cancelled = threading.Event(), threading.Event()
                self._active = (cancel, externally_cancelled)
            try:
                profile, fallback = self.runtime.select_profile(record)
                self._emit("generating", f"{record.get('speaker') or '未识别角色'} → {profile.speaker}" + ("（默认声线）" if fallback else ""),
                           record, voice=profile.speaker, fallback=fallback)
                utterance = Utterance(record["body"], speaker=profile.speaker, source="nte-ocr",
                                      captured_at=record.get("captured_monotonic") or time.monotonic())
                segments_started = 0

                def segment(item):
                    nonlocal segments_started
                    if not cancel.is_set():
                        segments_started += 1
                        self._emit("segment", f"正在播放 {profile.speaker} · 第 {item.index}/{item.total} 段", record,
                                   voice=profile.speaker, path=str(item.path), cache_hit=item.cache_hit,
                                   segment_index=item.index, ready_seconds=item.ready_seconds)

                session = self.runtime.session()
                for attempt in range(2):
                    try:
                        report = session.run(utterance, profile, cancel_event=cancel, on_segment=segment)
                        break
                    except Exception as exc:
                        # An internal synthesis failure sets the session's event
                        # too. Only an external cancel forbids the fallback.
                        if (attempt or externally_cancelled.is_set() or isinstance(exc, SpeechCancelled)
                                or segments_started or not getattr(self.runtime, "fallback_enabled", False)):
                            raise
                        with self._condition:
                            if self._closed or self._paused or self._current["dialogue_id"] != record["dialogue_id"]:
                                raise SpeechCancelled("当前台词已取消") from exc
                            cancel = threading.Event()
                            self._active = (cancel, externally_cancelled)
                        profile, session = self.runtime.fallback_session()
                        utterance = replace(utterance, speaker=profile.speaker)
                        fallback = True
                        self._emit("fallback", "角色模型未能及时出声，改用系统中文朗读（非角色音色）", record,
                                   voice=profile.speaker, error=str(exc), backend="windows-sapi")
                if cancel.is_set():
                    raise SpeechCancelled("台词已切换或配音已停止")
                exported = None
                if self._directory:
                    exported = join_audio(report.paths, self._directory / "audio" / (record["dialogue_id"] + ".wav"), cancel)
                if not cancel.is_set():
                    with self._condition:
                        self._completed.add(record["dialogue_id"])
                    self._emit("completed", f"{profile.speaker} 配音完成 · 缓存 {report.cache_hits}/{report.total_segments} 段", record,
                               voice=profile.speaker, fallback=fallback, audio_path=str(exported) if exported else None,
                               segment_paths=[str(p) for p in report.paths], cache_hits=report.cache_hits,
                               total_segments=report.total_segments, first_audio_seconds=report.first_audio_seconds)
            except SpeechCancelled:
                self._emit("cancelled", "已取消旧台词配音", record)
            except Exception as exc:
                if externally_cancelled.is_set():
                    self._emit("cancelled", "已取消旧台词配音", record)
                else:
                    self._emit("error", "配音失败：" + str(exc), record)
            finally:
                with self._condition:
                    self._active = None
                    self._condition.notify_all()
