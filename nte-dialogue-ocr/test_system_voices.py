import json
from pathlib import Path
import tempfile
import unittest

from dialogue_dubbing import VoiceRuntime, voice_paths
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'wuwa-dubber/src'))
from wuwa_dubber.models import VoiceProfile
from system_voices import SystemVoiceRouter, preset_profile


class SystemVoiceTests(unittest.TestCase):
    def test_confirmed_character_and_alias_match_but_uncertain_names_use_default(self):
        router = SystemVoiceRouter({"system_default_voice": "middle_male", "system_character_voices": {"伊洛伊": "young_female"}},
                                   [VoiceProfile("伊洛伊", aliases=["奥涅伊洛伊"])])
        for record, expected, fallback in [
            ({"speaker": "伊洛伊", "speaker_status": "verified"}, "young_female", False),
            ({"speaker": "奥涅伊洛伊", "speaker_status": "verified"}, "young_female", False),
            ({"speaker": "伊洛伊", "speaker_status": "needs_review"}, "middle_male", True),
            ({"speaker": "伊洛依", "speaker_status": "verified"}, "middle_male", True),
            ({"speaker_identity": "伊洛伊", "speaker_identity_confirmed": True}, "young_female", False),
            ({}, "middle_male", True),
        ]:
            self.assertEqual(router.select(record), (preset_profile(expected), fallback))

    def test_unknown_preset_and_conflicting_aliases_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "未知"):
            SystemVoiceRouter({"system_default_voice": "removed"})
        with self.assertRaisesRegex(ValueError, "不同声音"):
            SystemVoiceRouter({"system_character_voices": {"甲": "young_male", "别名": "young_female"}},
                              [VoiceProfile("甲", aliases=["别名"])])

    def test_missing_installed_voice_is_not_silently_replaced_with_wrong_gender(self):
        router = SystemVoiceRouter({"system_default_voice": "young_male"})
        with self.assertRaisesRegex(ValueError, "未安装"):
            router.check_available({"voices": []})
        router.check_available({"voices": [{"id": router.default.system_voice}]})

    def test_system_mode_does_not_require_clone_profiles_or_start_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "config.json").write_text(json.dumps({"tts_backend": "qwen3-tts", "narrator_profile": "不存在"}), encoding="utf-8")
            broken = data / "profiles/broken"
            broken.mkdir(parents=True)
            (broken / "profile.json").write_text("invalid json", encoding="utf-8")
            runtime = VoiceRuntime({"mode": "system", "data_root": str(data)})
            self.assertTrue(runtime.system_mode)
            self.assertEqual(runtime.client.__class__.__name__, "WindowsTtsClient")
            self.assertEqual(runtime.select_profile({}), (preset_profile("default"), True))


if __name__ == "__main__":
    unittest.main()
