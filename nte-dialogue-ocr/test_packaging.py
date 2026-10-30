"""Portable installation must preserve user settings and verified model files."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app
from scripts.download_models import prepare

TEMP_ROOT = Path(__file__).resolve().parent / 'validation'


class PackagingTests(unittest.TestCase):
    def test_existing_user_configuration_wins_over_example(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            example = root / 'config.example.json'
            config = root / 'config.json'
            example.write_text(json.dumps({'calibrated': False}), encoding='utf-8')
            with patch.object(app, 'ROOT', root), patch.object(app, 'CONFIG', config):
                self.assertFalse(app.load_config()['calibrated'])
                config.write_text(json.dumps({'calibrated': True, 'user_value': 'keep'}), encoding='utf-8')
                self.assertEqual(app.load_config(), {'calibrated': True, 'user_value': 'keep'})

    def test_bad_download_does_not_replace_existing_model(self):
        content = b'expected model'
        manifest = {'model.onnx': {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content), 'url': 'https://example.invalid/model'}}
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            model = root / 'model.onnx'
            model.write_bytes(b'existing file')
            with self.assertRaises(RuntimeError):
                prepare(manifest, root, opener=lambda *a, **k: io.BytesIO(b'corrupt'))
            self.assertEqual(model.read_bytes(), b'existing file')
            self.assertEqual(list(root.glob('*.part-*')), [])

    def test_valid_bundled_model_never_needs_network(self):
        content = b'expected model'
        manifest = {'model.onnx': {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content), 'url': 'https://example.invalid/model'}}
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            (root / 'model.onnx').write_bytes(content)
            def no_network(*args, **kwargs):
                raise AssertionError('Unexpected network request')
            prepare(manifest, root, opener=no_network)


if __name__ == '__main__':
    unittest.main()
