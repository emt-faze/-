"""Local Qwen3-TTS server for the existing desktop dubbing application."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "downloads" / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "downloads" / "numba-cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
        self.model_dir = model_dir.resolve()
        if not (self.model_dir / "model.safetensors").is_file():
            raise RuntimeError(f"模型未下载完整：{self.model_dir}")
        self.revision = self._revision()
        self.inference = {"dtype": "bfloat16" if device.startswith("cuda") else "float32",
                          "attention": "sdpa", "seed": 42, "max_new_tokens": 1024,
                          "non_streaming_mode": True, "implementation_version": 3}
        started = time.monotonic()
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
        self.load_seconds = time.monotonic() - started
        self.last_metrics = None

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
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("正在生成另一句语音，请稍后重试。")
        started = time.monotonic()
        torch = self.torch
        torch.set_num_threads(self.cpu_threads)
        try:
            with self.state_lock:
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
            key = hashlib.sha256(ref.read_bytes() + prompt.encode("utf-8")).hexdigest()
            prompt_cached = key in self.prompts
            if not prompt_cached:
                with torch.inference_mode():
                    self.prompts[key] = self.model.create_voice_clone_prompt(
                        ref_audio=str(ref.resolve()), ref_text=prompt, x_vector_only_mode=False)
                while len(self.prompts) > 8:
                    self.prompts.popitem(last=False)
            self.prompts.move_to_end(key)
            cancellation = self.cancel_event

            if cancellation.is_set():
                raise RuntimeError("本句生成已取消")
            with torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=language, voice_clone_prompt=self.prompts[key],
                    non_streaming_mode=True, max_new_tokens=1024,
                )
            if cancellation.is_set():
                raise RuntimeError("本句生成已取消")
            wav = np.asarray(wavs[0], dtype=np.float32)
            if speed != 1.0:
                import librosa
                wav = librosa.effects.time_stretch(wav, rate=speed)
            if len(wav) == 0 or not np.isfinite(wav).all():
                raise RuntimeError("模型返回了无效音频，请检查参考录音。")
            output = io.BytesIO()
            sf.write(output, wav, sr, format="WAV", subtype="PCM_16")
            elapsed = time.monotonic() - started
            metrics = {"request_id": self.active_id, "characters": len(text), "seconds": round(elapsed, 3),
                       "audio_seconds": round(len(wav) / sr, 3), "sample_rate": sr,
                       "prompt_cache_hit": prompt_cached,
                       "cpu_threads": torch.get_num_threads(),
                       "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if self.device.startswith("cuda") else None,
                       "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1) if self.device.startswith("cuda") else None}
            self.last_metrics = metrics
            with (ROOT / "logs" / "generation.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if cancellation.is_set():
                raise RuntimeError("本句生成已取消")
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
            data, metrics = self.engine.synthesize(payload)
            self.send_body(200, data, "audio/wav")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.send_json(503, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--device", default="cuda:0")
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
