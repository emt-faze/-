"""Guard public-install isolation, portable model lookup and voice-setting preservation."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import configure_install
import configure_generator
import build_generator_release
import choose_default_voice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'nte-dialogue-ocr'))
from dialogue_dubbing import VoiceRuntime, voice_paths


class GeneratorReleaseTests(unittest.TestCase):
    def fresh(self, root):
        (root/'nte-dialogue-ocr').mkdir()
        shutil.copy2(ROOT/'nte-dialogue-ocr/config.example.json',root/'nte-dialogue-ocr/config.example.json')
        (root/'wuwa-dubber/src/wuwa_dubber').mkdir(parents=True)
        (root/'wuwa-dubber/src/wuwa_dubber/speech_session.py').touch()
        configure_install.configure(root, with_model=True)
        configure_generator.configure(root)

    def test_fresh_install_never_reads_desktop_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fresh(root)
            config=json.loads((root/'nte-dialogue-ocr/config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['dubbing']['data_root'],'../wuwa-dubber/data')
            from unittest.mock import patch
            import dialogue_dubbing
            original=dialogue_dubbing.resolved_path
            with patch.object(dialogue_dubbing,'resolved_path',side_effect=lambda v:original(v,root/'nte-dialogue-ocr')):
                code,data=voice_paths(config['dubbing'])
            self.assertEqual(data,root/'wuwa-dubber/data')
            runtime=VoiceRuntime.__new__(VoiceRuntime)
            runtime.code_root, runtime.data_root=code,data
            self.assertEqual(runtime.model_root(), root/'local-voice-model')

    def test_existing_calibration_paths_and_mode_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fresh(root)
            path=root/'nte-dialogue-ocr/config.json'
            value=json.loads(path.read_text(encoding='utf-8'))
            value.update(calibrated=True, body_roi=[.1,.2,.7,.8])
            value['dubbing'].update(mode='character',data_root='custom-data')
            path.write_text(json.dumps(value),encoding='utf-8')
            before=path.read_bytes()
            self.assertFalse(configure_generator.configure(root)['settings_updated'])
            self.assertEqual(path.read_bytes(),before)

    def test_default_voice_selection_preserves_token_and_backs_up(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fresh(root)
            data=root/'wuwa-dubber/data'
            from wuwa_dubber.models import VoiceProfile
            profile=data/'profiles/example'
            profile.mkdir()
            (profile/'reference.wav').write_bytes(b'placeholder')
            voice=VoiceProfile(speaker='测试角色',reference_audio='example/reference.wav',prompt_text='参考台词')
            (profile/'profile.json').write_text(json.dumps(voice.to_dict()),encoding='utf-8')
            config=data/'config.json'
            before=config.read_bytes()
            self.assertTrue(choose_default_voice.save_default(data,'测试角色'))
            after=json.loads(config.read_text(encoding='utf-8'))
            self.assertEqual(after['narrator_profile'],'测试角色')
            self.assertEqual(after['tts_token'],json.loads(before)['tts_token'])
            self.assertEqual(next((data/'backups').glob('*/config.json')).read_bytes(),before)
            self.assertFalse(choose_default_voice.save_default(data,'测试角色'))

    def test_unknown_voice_never_changes_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fresh(root)
            data=root/'wuwa-dubber/data'
            before=(data/'config.json').read_bytes()
            with self.assertRaises(ValueError):
                choose_default_voice.save_default(data,'missing')
            self.assertEqual((data/'config.json').read_bytes(),before)

    def test_release_contains_required_entries_and_excludes_session_state(self):
        names={p.relative_to(ROOT).as_posix() for p in build_generator_release.source_files(ROOT)}
        self.assertIn('打开台词生成工具.bat',names)
        self.assertIn('nte-dialogue-ocr/dialogue_generator_ui.py',names)
        self.assertIn('local-voice-model/qwen_server.py',names)
        self.assertFalse(any('/data/' in n or '/records/' in n or n.endswith('.onnx') for n in names))


if __name__=='__main__':
    unittest.main()
