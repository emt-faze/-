"""Check portable setup, preservation of user data and release exclusions."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_release
import configure_install

ROOT = Path(__file__).resolve().parents[1]


class ReleaseSetupTests(unittest.TestCase):
    def test_fresh_configuration_and_relative_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configure_install.configure(root, with_model=True)
            data = root / "wuwa-dubber/data"
            config = json.loads((data / "config.json").read_text(encoding="utf-8"))
            runtime = json.loads((data / "model_runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(config["tts_backend"], "qwen3-tts")
            self.assertGreaterEqual(len(config["tts_token"]), 32)
            self.assertEqual(runtime["model_root"], "../local-voice-model")
            self.assertEqual(config["stability_frames"], 2)
            self.assertEqual(config["first_segment_max_chars"], 10)
            self.assertEqual(config["segment_max_chars"], 36)
            self.assertEqual(config["prefetch_window"], 2)

    def test_basic_reinstall_preserves_existing_settings_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configure_install.configure(root)
            data = root / "wuwa-dubber/data"
            config = json.loads((data / "config.json").read_text(encoding="utf-8"))
            config.update(output_device=8, narrator_profile="我的旁白", tts_backend="gpt-sovits",
                          stability_frames=3, prefetch_window=1)
            configure_install.write_json(data / "config.json", config)
            audio = data / "cache/keep.wav"
            audio.write_bytes(b"test-existing-audio")
            before = (data / "config.json").read_bytes()
            result = configure_install.configure(root)
            self.assertFalse(result["settings_updated"])
            self.assertEqual(before, (data / "config.json").read_bytes())
            self.assertEqual(audio.read_bytes(), b"test-existing-audio")

    def test_model_install_backs_up_settings_and_preserves_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configure_install.configure(root)
            config_file = root / "wuwa-dubber/data/config.json"
            before = config_file.read_bytes()
            result = configure_install.configure(root, with_model=True)
            after = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertTrue(result["backup_created"])
            self.assertEqual(json.loads(before)["tts_token"], after["tts_token"])
            backups = list((root / "wuwa-dubber/backups").glob("*/config.json"))
            self.assertEqual(backups[0].read_bytes(), before)

    def test_invalid_settings_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "wuwa-dubber/data/config.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"invalid-json")
            with self.assertRaises(ValueError):
                configure_install.configure(root, with_model=True)
            self.assertEqual(target.read_bytes(), b"invalid-json")

    def test_launcher_resolves_relative_model_path_after_moving(self):
        spec = importlib.util.spec_from_file_location("release_launcher", ROOT / "wuwa-dubber/scripts/launch_app.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory(prefix="dubbing moved ") as directory:
            root = Path(directory)
            configure_install.configure(root, with_model=True)
            starter = root / "local-voice-model/scripts/start-server.ps1"
            starter.parent.mkdir(parents=True)
            starter.write_text("exit 0", encoding="utf-8")
            with patch.object(launcher, "ROOT", root / "wuwa-dubber"), patch.dict(os.environ), patch.object(launcher.subprocess, "call", return_value=0) as call:
                os.environ.pop("WUWA_DUBBER_DATA", None)
                self.assertEqual(launcher.main(), 0)
                self.assertEqual(Path(call.call_args_list[0].args[0][5]), starter.resolve())

    def test_source_selection_excludes_runtime_and_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in build_release.FIXED_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source", encoding="utf-8")
            excluded = ["wuwa-dubber/data/config.json", "wuwa-dubber/data/profiles/voice.wav",
                        "wuwa-dubber/.venv/lib/private.py", "local-voice-model/models/weights.safetensors",
                        "local-voice-model/logs/session.json", "releases/old.zip"]
            for name in excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"local-only")
            selected = {p.relative_to(root).as_posix() for p in build_release.source_files(root)}
            self.assertFalse(selected.intersection(excluded))


if __name__ == "__main__":
    unittest.main()
