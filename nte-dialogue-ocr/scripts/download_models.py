"""Prepare the four pinned RapidOCR models, using atomic, hash-checked writes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']


def prepare(manifest, destination, verify_only=False, opener=urllib.request.urlopen):
    destination.mkdir(parents=True, exist_ok=True)
    for name, entry in manifest.items():
        if Path(name).name != name or not name.endswith('.onnx'):
            raise ValueError('Invalid model filename')
        path = destination / name
        if valid(path, entry):
            print('Verified: ' + name, flush=True)
            continue
        if verify_only:
            raise RuntimeError('Missing or invalid model: ' + name)
        temporary = destination / (name + '.part-' + uuid.uuid4().hex)
        try:
            print('Downloading: ' + name, flush=True)
            request = urllib.request.Request(entry['url'], headers={'User-Agent': 'nte-dialogue-ocr/5.0.0'})
            with opener(request, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if not valid(temporary, entry):
                raise RuntimeError('SHA-256 or size mismatch: ' + name)
            temporary.replace(path)
            print('Verified: ' + name, flush=True)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'assets/ocr-v6')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets/ocr-v6/models.json').read_text(encoding='utf-8'))
    prepare(manifest, args.models_dir, args.verify_only)
    print('All four local models are ready.')


if __name__ == '__main__':
    main()
