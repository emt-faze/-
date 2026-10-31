"""Model-free HTTP and cancellation regression tests for qwen_server.py.

Run with any existing Python 3.11+ interpreter:
    python test_server_protocol.py
No torch, NumPy, soundfile, model download, or GPU is required.
"""
from __future__ import annotations

import http.client
import base64
import io
import json
import os
import tempfile
import threading
import time
import unittest
import wave
from collections import OrderedDict
from contextlib import nullcontext
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import qwen_server


class FakeModel:
    def __init__(self):
        self.prompt_calls = 0
        self.generation_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls += 1
        return "fake prompt"

    def generate_voice_clone(self, **kwargs):
        self.generation_calls += 1
        self.started.set()
        if self.block and not self.release.wait(3):
            raise RuntimeError("Test did not release fake generation")
        return [[0.0] * 240], 24000


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="qwen-protocol-")
        self.root = Path(self.directory.name)
        (self.root / "logs").mkdir()
        self.reference = self.root / "reference.wav"
        with wave.open(str(self.reference), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\0\0" * 32000)
        self.duration = 4.0
        self.model = FakeModel()
        self.engine = qwen_server.VoiceEngine.__new__(qwen_server.VoiceEngine)
        self.engine.torch = SimpleNamespace(manual_seed=lambda seed: None, set_num_threads=lambda count: None,
                                           get_num_threads=lambda: 1, inference_mode=nullcontext)
        self.engine.cpu_threads = 1
        self.engine.device = "cpu"
        self.engine.lock = threading.Lock()
        self.engine.state_lock = threading.Lock()
        self.engine.active_id = None
        self.engine.cancelled_requests = {}
        self.engine.cancel_event = threading.Event()
        self.engine.prompts = OrderedDict()
        self.engine.model_dir = self.root
        self.engine.revision = "test-model-revision"
        self.engine.inference = {"seed": 42, "dtype": "float32"}
        self.engine.load_seconds = 0.01
        self.engine.last_metrics = None
        self.engine.model = self.model

        def write_wav(target, values, sample_rate, **kwargs):
            with wave.open(target, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(b"\0\0" * len(values))

        fake_numpy = SimpleNamespace(
            float32="float32", asarray=lambda values, **kwargs: values,
            isfinite=lambda values: SimpleNamespace(all=lambda: True),
        )
        fake_soundfile = SimpleNamespace(
            info=lambda path: SimpleNamespace(duration=self.duration), write=write_wav,
        )
        self.dependencies = patch.dict("sys.modules", {"numpy": fake_numpy, "soundfile": fake_soundfile})
        self.dependencies.start()
        self.root_patch = patch.object(qwen_server, "ROOT", self.root)
        self.root_patch.start()
        self.trace_patch = patch("traceback.print_exc")
        self.trace_patch.start()
        self.handler = type("TestHandler", (qwen_server.Handler,), {
            "engine": self.engine, "token": "protocol-test-token",
            "log_message": lambda *args: None,
        })
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.model.release.set()
        worker = getattr(self.engine, "warmup_thread", None)
        if worker is not None:
            self.engine.warmup_stop.set()
            worker.join(4)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.trace_patch.stop()
        self.root_patch.stop()
        self.dependencies.stop()
        self.directory.cleanup()

    def payload(self, **changes):
        return {
            "request_id": "request-one", "text": "测试中文配音。",
            "ref_audio_path": str(self.reference), "prompt_text": "实际参考文字。",
            "text_lang": "zh", "speed": 1.0, **changes,
        }

    def request(self, path, *, method="POST", payload=None, raw=None, token="protocol-test-token", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=4)
        request_headers = {"Content-Type": "application/json"}
        if token is not None:
            request_headers["Authorization"] = "Bearer " + token
        if headers:
            request_headers.update(headers)
        body = raw if raw is not None else (None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def test_health_and_synthesis_and_cancel_require_bearer_token(self):
        for method, path, payload in (
            ("GET", "/v1/health", None),
            ("POST", "/v1/synthesize", self.payload()),
            ("POST", "/v1/cancel", {"request_id": "request-one"}),
        ):
            for token in (None, "wrong"):
                with self.subTest(method=method, path=path, token=token):
                    status, _, body = self.request(path, method=method, payload=payload, token=token)
                    self.assertEqual(status, 401)
                    self.assertEqual(json.loads(body)["error"], "unauthorized")
        self.assertEqual(self.model.generation_calls, 0)
        self.assertEqual(self.engine.cancelled_requests, {})

    def test_authorized_health_has_stable_model_identity(self):
        status, content_type, body = self.request("/v1/health", method="GET")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        health = json.loads(body)
        self.assertEqual(health["backend"], "qwen3-tts")
        self.assertEqual(health["model_revision"], "test-model-revision")
        self.assertFalse(health["busy"])

    def test_unknown_route_is_not_synthesis(self):
        self.assertEqual(self.request("/v1/unknown", payload=self.payload())[0], 404)
        self.assertEqual(self.model.generation_calls, 0)

    def test_json_and_request_length_validation(self):
        cases = [
            (b"broken", {}), (b"[]", {}), (b"null", {}), (b'"text"', {}),
            (b"", {}), (b"{}", {"Content-Length": "-1"}),
            (b"{}", {"Content-Length": "65537"}),
            (b"{}", {"Content-Length": "not-a-number"}),
            (b"\xff", {}),
        ]
        for raw, headers in cases:
            with self.subTest(raw=raw, headers=headers):
                status, _, body = self.request("/v1/synthesize", raw=raw, headers=headers)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))
        self.assertEqual(self.model.generation_calls, 0)

    def test_synthesis_input_validation_does_not_touch_model(self):
        for changes in (
            {"text": ""}, {"text": "字" * 301}, {"prompt_text": ""},
            {"ref_audio_path": str(self.root / "missing.wav")},
            {"speed": 0.1}, {"speed": 2.1}, {"speed": "NaN"}, {"speed": "infinity"},
            {"text_lang": "unsupported"},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(self.request("/v1/synthesize", payload=self.payload(**changes))[0], 400)
        for duration in (2.9, 30.1):
            self.duration = duration
            self.assertEqual(self.request("/v1/synthesize", payload=self.payload())[0], 400)
        self.assertEqual(self.model.prompt_calls, 0)
        self.assertEqual(self.model.generation_calls, 0)

    def test_success_is_pcm_wav_and_prompt_is_reused(self):
        for request_id in ("first", "second"):
            status, content_type, body = self.request("/v1/synthesize", payload=self.payload(request_id=request_id))
            self.assertEqual(status, 200)
            self.assertEqual(content_type, "audio/wav")
            with wave.open(io.BytesIO(body), "rb") as audio:
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertEqual(audio.getframerate(), 24000)
                self.assertEqual(audio.getnframes(), 240)
        self.assertEqual(self.model.prompt_calls, 1)
        self.assertEqual(self.model.generation_calls, 2)
        self.assertTrue(self.engine.last_metrics["prompt_cache_hit"])
        self.assertFalse(self.engine.lock.locked())
        self.assertIsNone(self.engine.active_id)

    def test_quality_settings_rebuild_prompt_cache(self):
        self.engine.synthesize(self.payload())
        self.engine.synthesize(self.payload())
        self.assertEqual(self.model.prompt_calls, 1)
        self.engine.inference.update(audio_tokenizer_dtype="float32", implementation_version=4)
        self.engine.synthesize(self.payload())
        self.assertEqual(self.model.prompt_calls, 2)
        self.assertFalse(self.engine.last_metrics["prompt_cache_hit"])

    def test_punctuation_and_all_generated_frames_are_preserved(self):
        # Active final syllables must not be shortened by output processing.
        text = "失误了啊，鉴定师。"
        with patch.object(self.model, "generate_voice_clone", return_value=([[.25] * 55680], 24000)) as generate:
            data, _ = self.engine.synthesize(self.payload(text=text))
        self.assertEqual(generate.call_args.kwargs["text"], text)
        self.assertEqual(generate.call_args.kwargs["repetition_penalty"], 1.0)
        with wave.open(io.BytesIO(data), "rb") as audio:
            self.assertEqual(audio.getnframes(), 55680)

    def test_incomplete_attempt_is_retried_without_publishing_its_audio(self):
        with patch.object(self.model, "generate_voice_clone", side_effect=[
            qwen_server.IncompleteSpeech("partial"), ([[.25]*480], 24000),
        ]) as generate:
            data, metrics = self.engine.synthesize(self.payload())
        self.assertEqual([call.kwargs["repetition_penalty"] for call in generate.call_args_list], [1.0, 1.05])
        self.assertTrue(metrics["completion_retry"])
        with wave.open(io.BytesIO(data), "rb") as audio:
            self.assertEqual(audio.getnframes(), 480)

    def test_repeated_incomplete_output_is_an_error_and_not_a_wav(self):
        with patch.object(self.model, "generate_voice_clone", side_effect=qwen_server.IncompleteSpeech("partial")) as generate:
            with self.assertRaises(qwen_server.IncompleteSpeech):
                self.engine.synthesize(self.payload())
        self.assertEqual(generate.call_count, 2)
        self.assertIsNone(self.engine.last_metrics)
        self.assertFalse(self.engine.lock.locked())

    def test_cancelled_incomplete_output_never_starts_retry(self):
        def cancelled(**kwargs):
            self.engine.cancel_event.set()
            raise qwen_server.IncompleteSpeech("partial")
        with patch.object(self.model, "generate_voice_clone", side_effect=cancelled) as generate:
            with self.assertRaises(qwen_server.IncompleteSpeech):
                self.engine.synthesize(self.payload())
        self.assertEqual(generate.call_count, 1)

    def test_token_limit_without_end_token_is_not_a_complete_utterance(self):
        class Sequence:
            def __init__(self, ending):
                self.ending = ending
            def __getitem__(self, index):
                return SimpleNamespace(tolist=lambda: [self.ending])
        fake_transformers = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            partial = qwen_server._guard_generation(
                lambda **kwargs: SimpleNamespace(sequences=Sequence(7)), self.engine.torch, self.engine.cancel_event)
            complete = qwen_server._guard_generation(
                lambda **kwargs: SimpleNamespace(sequences=Sequence(2150)), self.engine.torch, self.engine.cancel_event)
        with self.assertRaises(qwen_server.IncompleteSpeech):
            partial(eos_token_id=2150)
        self.assertIsNotNone(complete(eos_token_id=2150))

    def test_diagnostic_json_preserves_wav_and_accounts_for_stage_timings(self):
        original = self.model.generate_voice_clone
        def delayed(**kwargs):
            time.sleep(0.04)
            return original(**kwargs)
        self.model.generate_voice_clone = delayed
        status, content_type, body = self.request("/v1/synthesize", payload=self.payload(response_format="json"))
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        result = json.loads(body)
        self.assertTrue(base64.b64decode(result["audio_wav_base64"]).startswith(b"RIFF"))
        timing = result["timing_ms"]
        names = ("model_load", "ref_encode", "text_prepare", "infer", "wav_write")
        self.assertTrue(all(timing[name] >= 0 for name in names))
        self.assertEqual(timing["model_load"], 0)
        self.assertGreaterEqual(timing["infer"], 35)
        self.assertLess(abs(timing["total"] - sum(timing[name] for name in names)), timing["total"] * 0.1)

    def test_invalid_response_format_does_not_start_generation(self):
        status, _, _ = self.request("/v1/synthesize", payload=self.payload(response_format="invalid"))
        self.assertEqual(status, 400)
        self.assertEqual(self.model.generation_calls, 0)

    def test_background_warmup_keeps_health_available_and_reuses_reference(self):
        self.model.block = True
        self.engine.start_warmup(self.payload())
        self.assertTrue(self.model.started.wait(2))
        status, _, body = self.request("/v1/health", method="GET")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["warm"])
        self.assertTrue(json.loads(body)["warming"])
        self.model.release.set()
        self.engine.warmup_thread.join(3)
        self.assertTrue(self.engine.warm)
        self.assertFalse(self.engine.warmup_running)
        self.assertEqual(self.model.prompt_calls, 1)
        self.assertEqual(self.model.generation_calls, 2)

    def test_user_request_preempts_warmup_and_is_not_rejected_as_busy(self):
        self.model.block = True
        self.engine.start_warmup(self.payload())
        self.assertTrue(self.model.started.wait(2))
        results = []
        user = threading.Thread(target=lambda: results.append(self.request("/v1/synthesize", payload=self.payload(request_id="user"))))
        user.start()
        self.assertTrue(self.engine.warmup_stop.wait(2))
        self.model.release.set()
        user.join(3)
        self.engine.warmup_thread.join(3)
        self.assertFalse(user.is_alive())
        self.assertEqual(results[0][0], 200)
        self.assertTrue(results[0][2].startswith(b"RIFF"))
        self.assertTrue(self.engine.warm)

    def test_warmup_failure_does_not_disable_later_requests(self):
        with patch.object(self.model, "generate_voice_clone", side_effect=RuntimeError("warmup failure")):
            self.engine.start_warmup(self.payload())
            self.engine.warmup_thread.join(3)
        self.assertEqual(self.engine.warmup_report["status"], "failed")
        self.assertFalse(self.engine.warmup_running)
        self.assertEqual(self.request("/v1/synthesize", payload=self.payload(request_id="after-failure"))[0], 200)

    def test_reference_content_change_invalidates_prompt_even_when_size_and_time_match(self):
        self.engine.synthesize(self.payload())
        stat = self.reference.stat()
        data = bytearray(self.reference.read_bytes())
        data[-1] = 1
        self.reference.write_bytes(data)
        os.utime(self.reference, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.engine.synthesize(self.payload(request_id="changed"))
        self.assertEqual(self.model.prompt_calls, 2)
        self.assertFalse(self.engine.last_metrics["prompt_cache_hit"])

    def test_reference_transcript_change_invalidates_prompt(self):
        self.engine.synthesize(self.payload())
        self.engine.synthesize(self.payload(request_id="changed", prompt_text="修改后的准确参考文字。"))
        self.assertEqual(self.model.prompt_calls, 2)

    def test_prompt_cache_evicts_least_recently_used_entry(self):
        self.engine.prompt_cache_limit = 2
        for text in ("甲", "乙", "甲", "丙"):
            self.engine.synthesize(self.payload(prompt_text=text))
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertEqual(self.model.prompt_calls, 3)
        self.engine.synthesize(self.payload(prompt_text="甲"))
        self.assertEqual(self.model.prompt_calls, 3)
        self.engine.synthesize(self.payload(prompt_text="乙"))
        self.assertEqual(self.model.prompt_calls, 4)

    def test_cancel_arriving_before_synthesis_is_remembered(self):
        status, _, body = self.request("/v1/cancel", payload={"request_id": "cancel-before-start"})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["cancelled"])
        status, _, body = self.request("/v1/synthesize", payload=self.payload(request_id="cancel-before-start"))
        self.assertEqual(status, 503)
        self.assertIn("取消", json.loads(body)["error"])
        self.assertEqual(self.model.prompt_calls, 0)
        self.assertEqual(self.model.generation_calls, 0)
        self.assertFalse(self.engine.lock.locked())
        self.assertIsNone(self.engine.active_id)
        self.assertEqual(self.request("/v1/synthesize", payload=self.payload(request_id="fresh-id"))[0], 200)

    def test_cancel_during_generation_discards_result_and_does_not_cancel_other_ids(self):
        self.model.block = True
        results = []
        worker = threading.Thread(target=lambda: results.append(self.request("/v1/synthesize", payload=self.payload(request_id="active-id"))))
        worker.start()
        try:
            self.assertTrue(self.model.started.wait(2))
            self.request("/v1/cancel", payload={"request_id": "unrelated-id"})
            self.assertFalse(self.engine.cancel_event.is_set())
            self.request("/v1/cancel", payload={"request_id": "active-id"})
            self.assertTrue(self.engine.cancel_event.is_set())
        finally:
            self.model.release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0][0], 503)
        self.assertIn("取消", json.loads(results[0][2])["error"])
        self.assertIsNone(self.engine.last_metrics)

    def test_second_generation_is_rejected_while_engine_is_busy(self):
        self.model.block = True
        results = []
        worker = threading.Thread(target=lambda: results.append(self.request("/v1/synthesize", payload=self.payload(request_id="first"))))
        worker.start()
        try:
            self.assertTrue(self.model.started.wait(2))
            status, _, body = self.request("/v1/synthesize", payload=self.payload(request_id="second"))
            self.assertEqual(status, 503)
            self.assertIn("另一句", json.loads(body)["error"])
        finally:
            self.model.release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0][0], 200)
        self.assertEqual(self.model.generation_calls, 1)

    def test_generation_deadline_rejects_partial_audio_and_releases_engine(self):
        fake_torch = SimpleNamespace(full=lambda shape, value, **kwargs: value, bool=bool)
        fake_transformers = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
        original_generate = self.model.generate_voice_clone

        def partial_generate(**kwargs):
            # Hugging Face returns partial tokens when a criterion stops it.
            criteria = kwargs["stopping_criteria"]
            self.assertTrue(criteria[-1](SimpleNamespace(shape=(1, 1), device="cpu"), None))
            return [[0.0] * 240], 24000

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            self.model.generate_voice_clone = qwen_server._guard_generation(
                partial_generate, fake_torch, self.engine.cancel_event, timeout_seconds=0.0)
        status, content_type, body = self.request("/v1/synthesize", payload=self.payload())
        self.assertEqual(status, 503)
        self.assertIn("application/json", content_type)
        self.assertIn("已丢弃未完成音频", json.loads(body)["error"])
        self.assertIsNone(self.engine.last_metrics)
        self.assertFalse((self.root / "logs" / "generation.jsonl").exists())
        self.assertFalse(self.engine.lock.locked())
        self.assertIsNone(self.engine.active_id)
        self.model.generate_voice_clone = original_generate
        self.assertEqual(self.request("/v1/synthesize", payload=self.payload(request_id="after-timeout"))[0], 200)

    def test_cancellation_records_are_bounded_and_expire(self):
        for index in range(1100):
            self.engine.cancel(f"cancel-{index}")
        self.assertLessEqual(len(self.engine.cancelled_requests), 1024)
        self.engine.cancelled_requests["expired"] = time.monotonic() - 181
        self.engine.cancel("next")
        self.assertNotIn("expired", self.engine.cancelled_requests)
        self.engine.cancelled_requests["old-request"] = time.monotonic() - 181
        self.assertEqual(self.request("/v1/synthesize", payload=self.payload(request_id="old-request"))[0], 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
