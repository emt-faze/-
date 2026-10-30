"""Local Qwen3-TTS server for the existing desktop dubbing application."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "downloads" / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "downloads" / "numba-cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
LOGGER = logging.getLogger("qwen_server")


@dataclass(slots=True)
class PhaseTimer:
    synchronize: Callable[[], None] | None = None
    phases: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ("model_load", "ref_encode", "text_prepare", "infer", "wav_write"), 0.0))
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def phase(self, name: str):
        # 等待异步 GPU 工作结束，避免把上一阶段耗时记到下一阶段。
        if self.synchronize:
            self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.synchronize:
                self.synchronize()
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - started) * 1000

    def result(self) -> dict[str, float]:
        elapsed = (time.perf_counter() - self.started) * 1000
        return {**{name: round(value, 3) for name, value in self.phases.items()},
                "other": round(max(0.0, elapsed - sum(self.phases.values())), 3),
                "total": round(elapsed, 3)}


def _guard_generation(generate, torch, cancellation, *, timeout_seconds=150.0):
    """Reject partial tokens when a generation deadline stops the talker."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Interrupted(StoppingCriteria):
        def __init__(self):
            self.deadline = time.monotonic() + timeout_seconds
            self.timed_out = False

        def __call__(self, input_ids, scores, **kwargs):
            if not cancellation.is_set() and time.monotonic() >= self.deadline:
                self.timed_out = True
            return torch.full((input_ids.shape[0],), cancellation.is_set() or self.timed_out,
                              device=input_ids.device, dtype=torch.bool)

    def guarded(*args, **kwargs):
        interrupted = Interrupted()
        criteria = list(kwargs.get("stopping_criteria") or [])
        criteria.append(interrupted)
        kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)
        result = generate(*args, **kwargs)
        if interrupted.timed_out:
            raise RuntimeError(f"生成超过 {timeout_seconds:g} 秒，已丢弃未完成音频，请缩短台词后重试。")
        return result

    return guarded


class VoiceEngine:
    def __init__(self, model_dir: Path, *, device: str = "cuda:0"):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.torch = torch
        self.cpu_threads = int(os.environ.get("QWEN_CPU_THREADS", "1"))
        torch.set_num_threads(self.cpu_threads)
        torch.set_num_interop_threads(1)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 NVIDIA GPU，请先运行本机环境检查。")
        self.device = device
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_id = None
        self.cancelled_requests = {}
        self.cancel_event = threading.Event()
        self.prompts = OrderedDict()
        self.prompt_cache_limit = max(1, min(64, int(os.environ.get("QWEN_PROMPT_CACHE_SIZE", "16"))))
        self.model_dir = model_dir.resolve()
        if not (self.model_dir / "model.safetensors").is_file():
            raise RuntimeError(f"模型未下载完整：{self.model_dir}")
        self.revision = self._revision()
        self.inference = {"dtype": "bfloat16" if device.startswith("cuda") else "float32",
                          "attention": "sdpa", "seed": 42, "max_new_tokens": 1024,
                          "non_streaming_mode": True, "implementation_version": 3}
        load_timer = PhaseTimer()
        with load_timer.phase("model_load"):
            self.model = Qwen3TTSModel.from_pretrained(
                str(self.model_dir), device_map=device,
                dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                attn_implementation="sdpa", local_files_only=True,
            )
        # qwen-tts 0.1.1 drops arbitrary generation kwargs at its outer wrapper.
        # Attach cancellation to the inner Hugging Face talker, where criteria
        # are evaluated between generated tokens; leave vendor files untouched.
        original_generate = self.model.model.talker.generate
        self.model.model.talker.generate = _guard_generation(original_generate, torch, self.cancel_event)
        self.load_seconds = load_timer.phases["model_load"] / 1000
        self.last_metrics = None
        self.warm = False
        self.warmup_running = False
        self.warmup_stop = threading.Event()
        self.warmup_thread: threading.Thread | None = None
        self.warmup_report: dict = {"status": "not_started"}

    def start_warmup(self, payload: dict | None) -> None:
        if getattr(self, "warmup_running", False):
            return
        self.warmup_stop = threading.Event()
        self.warm = False
        if payload is None:
            self.warmup_report = {"status": "no_reference"}
            LOGGER.warning("没有可用参考声线，跳过预热；导入声线后可正常生成。")
            return
        self.warmup_running = True
        self.warmup_report = {"status": "running", "speaker": payload.get("speaker", "")}

        def work():
            started = time.perf_counter()
            records = []
            status = "ready"
            try:
                for index, text in enumerate(("你好。", "测试。")):
                    if self.warmup_stop.is_set():
                        status = "interrupted"
                        break
                    _, metrics = self.synthesize({**payload, "text": text, "speed": 1.0,
                                                  "request_id": f"warmup-{index}"})
                    records.append(metrics)
                if len(records) == 2:
                    self.warm = True
            except Exception as exc:
                status = "interrupted" if self.warmup_stop.is_set() else "failed"
                LOGGER.warning("模型预热%s，后续请求仍可使用：%s", status, exc)
            finally:
                self.warmup_report = {"status": status, "speaker": payload.get("speaker", ""),
                                      "seconds": round(time.perf_counter() - started, 3), "runs": records}
                self.warmup_running = False
                LOGGER.info("warmup %s", json.dumps(self.warmup_report, ensure_ascii=False))

        self.warmup_thread = threading.Thread(target=work, name="qwen-warmup", daemon=True)
        self.warmup_thread.start()

    def _revision(self):
        manifest = self.model_dir / "download-manifest.json"
        if manifest.is_file():
            record = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if record.get("status") == "complete" and record.get("revision"):
                return record["revision"]
        # Downloader metadata may be adjacent to the model, so include a stable
        # size/mtime fingerprint even when a revision manifest is unavailable.
        values = []
        for name in ("model.safetensors", "speech_tokenizer/model.safetensors", "config.json"):
            p = self.model_dir / name
            stat = p.stat()
            values.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("|".join(values).encode()).hexdigest()

    def health(self):
        return {"status": "ok", "backend": "qwen3-tts",
                "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "model_revision": self.revision,
                "device": self.device, "inference": self.inference,
                "load_seconds": round(self.load_seconds, 3), "busy": self.lock.locked(),
                "warm": getattr(self, "warm", False), "warming": getattr(self, "warmup_running", False),
                "warmup": getattr(self, "warmup_report", {"status": "not_started"}),
                "prompt_cache": {"entries": len(self.prompts), "limit": getattr(self, "prompt_cache_limit", 16)},
                "last_metrics": self.last_metrics}

    def cancel(self, request_id):
        with self.state_lock:
            if not request_id:
                return False
            now = time.monotonic()
            self.cancelled_requests = {k: v for k, v in self.cancelled_requests.items() if now - v < 180}
            if len(self.cancelled_requests) >= 1024:
                self.cancelled_requests.pop(next(iter(self.cancelled_requests)))
            self.cancelled_requests[str(request_id)] = now
            matched = bool(request_id) and request_id == self.active_id
            if matched:
                self.cancel_event.set()
            return True

    def synthesize(self, payload):
        is_warmup = threading.current_thread() is getattr(self, "warmup_thread", None)
        timer = PhaseTimer(self.torch.cuda.synchronize if self.device.startswith("cuda") else None)
        with timer.phase("text_prepare"):
            import numpy as np
            import soundfile as sf

            text = str(payload.get("text", "")).strip()
            if not 1 <= len(text) <= 300:
                raise ValueError("每句文字须为 1 到 300 字，请将长段落分句。")
            ref = Path(str(payload.get("ref_audio_path", "")))
            if not ref.is_file() or ref.stat().st_size > 50 * 1024 * 1024:
                raise ValueError("请选择本地参考音频（不超过 50 MB）。")
            prompt = str(payload.get("prompt_text", "")).strip()
            if not prompt:
                raise ValueError("请填写参考音频中实际说出的文字。")
            speed = float(payload.get("speed", 1.0))
            if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
                raise ValueError("语速须为 0.5 到 2.0。")
            info = sf.info(str(ref))
            if not 3 <= info.duration <= 30:
                raise ValueError("参考录音请使用 3 到 30 秒的单人清晰语音。")
            language = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "auto": "Auto"}.get(str(payload.get("text_lang", "zh")))
            if language is None:
                raise ValueError("当前服务支持 zh、en、ja、ko 或 auto 语言代码。")
        # 用户请求优先于后台预热；取消预热后只等待当前 GPU 操作安全结束。
        waiting_for_warmup = not is_warmup and getattr(self, "warmup_running", False)
        if waiting_for_warmup:
            self.warmup_stop.set()
            with self.state_lock:
                if str(self.active_id or "").startswith("warmup-"):
                    self.cancel_event.set()
        acquired = self.lock.acquire(timeout=15.0) if waiting_for_warmup else self.lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("正在生成另一句语音，请稍后重试。")
        torch = self.torch
        torch.set_num_threads(self.cpu_threads)
        try:
            with self.state_lock:
                if is_warmup and self.warmup_stop.is_set():
                    raise RuntimeError("预热已让位给用户请求")
                self.cancel_event.clear()
                self.active_id = str(payload.get("request_id", ""))
                cancelled_at = self.cancelled_requests.pop(self.active_id, None)
                if cancelled_at is not None and time.monotonic() - cancelled_at < 180:
                    self.cancel_event.set()
            if self.cancel_event.is_set():
                raise RuntimeError("本句生成已取消")
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(42)
            with timer.phase("ref_encode"):
                key = hashlib.sha256(ref.read_bytes() + prompt.encode("utf-8")).hexdigest()
                prompt_cached = key in self.prompts
                if not prompt_cached:
                    with torch.inference_mode():
                        self.prompts[key] = self.model.create_voice_clone_prompt(
                            ref_audio=str(ref.resolve()), ref_text=prompt, x_vector_only_mode=False)
                    while len(self.prompts) > getattr(self, "prompt_cache_limit", 16):
                        self.prompts.popitem(last=False)
                self.prompts.move_to_end(key)
            cancellation = self.cancel_event

            if cancellation.is_set():
                raise RuntimeError("本句生成已取消")
            with timer.phase("infer"), torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=language, voice_clone_prompt=self.prompts[key],
                    non_streaming_mode=True, max_new_tokens=32 if is_warmup else 1024,
                )
            if cancellation.is_set():
                raise RuntimeError("本句生成已取消")
            with timer.phase("wav_write"):
                wav = np.asarray(wavs[0], dtype=np.float32)
                if speed != 1.0:
                    import librosa
                    wav = librosa.effects.time_stretch(wav, rate=speed)
                if len(wav) == 0 or not np.isfinite(wav).all():
                    raise RuntimeError("模型返回了无效音频，请检查参考录音。")
                output = io.BytesIO()
                sf.write(output, wav, sr, format="WAV", subtype="PCM_16")
            timing = timer.result()
            elapsed = timing["total"] / 1000
            metrics = {"request_id": self.active_id, "characters": len(text), "seconds": round(elapsed, 3),
                       "warmup": is_warmup,
                       "timing_ms": timing,
                       "audio_seconds": round(len(wav) / sr, 3), "sample_rate": sr,
                       "prompt_cache_hit": prompt_cached,
                       "cpu_threads": torch.get_num_threads(),
                       "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if self.device.startswith("cuda") else None,
                       "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1) if self.device.startswith("cuda") else None}
            self.last_metrics = metrics
            LOGGER.info("synthesis %s", json.dumps(metrics, ensure_ascii=False))
            with (ROOT / "logs" / "generation.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if cancellation.is_set():
                raise RuntimeError("本句生成已取消")
            if not is_warmup:
                self.warm = True
            return output.getvalue(), metrics
        finally:
            with self.state_lock:
                self.active_id = None
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    engine: VoiceEngine
    token: str

    def authorized(self):
        import hmac
        ok = hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.token)
        if not ok:
            self.send_json(401, {"error": "unauthorized"})
        return ok

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_body(status, data, "application/json; charset=utf-8")

    def send_body(self, status, data, content_type):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        if self.path != "/v1/health":
            self.send_json(404, {"error": "not found"})
        elif self.authorized():
            self.send_json(200, self.engine.health())

    def do_POST(self):
        if not self.authorized():
            return
        if self.path not in {"/v1/synthesize", "/v1/cancel"}:
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("请求长度不正确")
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.connection.settimeout(None)
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            if self.path == "/v1/cancel":
                self.send_json(200, {"cancelled": self.engine.cancel(payload.get("request_id"))})
                return
            response_format = payload.get("response_format", "wav")
            if response_format not in {"wav", "json"}:
                raise ValueError("response_format 必须是 wav 或 json")
            data, metrics = self.engine.synthesize(payload)
            # 原调用继续收到 WAV；诊断请求可显式选择带计时的 JSON。
            if response_format == "json":
                self.send_json(200, {"audio_wav_base64": base64.b64encode(data).decode("ascii"),
                                     "timing_ms": metrics["timing_ms"], "metrics": metrics})
            else:
                self.send_body(200, data, "audio/wav")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send_json(503, {"error": str(exc)})


def select_warmup_reference(config_file: Path) -> dict | None:
    # 优先上次试听声线，其次首个可用档案；不依赖随包分发录音。
    config = json.loads(config_file.read_text(encoding="utf-8-sig"))
    folder = config_file.parent / "profiles"
    candidates = []
    for profile_file in sorted(folder.glob("*/profile.json")):
        try:
            profile = json.loads(profile_file.read_text(encoding="utf-8-sig"))
            reference = Path(profile["reference_audio"])
            if not reference.is_absolute():
                reference = folder / reference
            if profile.get("enabled", True) and reference.is_file() and profile.get("prompt_text", "").strip():
                candidates.append({"speaker": profile.get("speaker", ""), "ref_audio_path": str(reference.resolve()),
                                   "prompt_text": profile["prompt_text"], "text_lang": "zh"})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    preferred = config.get("warmup_speaker", "")
    return next((p for p in candidates if p["speaker"] == preferred), candidates[0] if candidates else None)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-warmup", action="store_true", help="Disable background warmup for baseline measurements")
    parser.add_argument("--app-config", type=Path, default=ROOT.parent / "wuwa-dubber/data/config.json")
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    config = json.loads(args.app_config.read_text(encoding="utf-8-sig"))
    token = str(config["tts_token"])
    if not token:
        raise ValueError("请先在配音程序设置本地令牌。")
    print("Loading local Qwen3-TTS on", args.device, flush=True)
    engine = VoiceEngine(args.model_dir, device=args.device)
    Handler.engine, Handler.token = engine, token
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    if not args.no_warmup:
        try:
            engine.start_warmup(select_warmup_reference(args.app_config))
        except Exception:
            LOGGER.warning("无法准备预热声线，服务仍可使用", exc_info=True)
    print(json.dumps(engine.health(), ensure_ascii=False), flush=True)
    print(f"Ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
