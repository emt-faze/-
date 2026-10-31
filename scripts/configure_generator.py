"""Initialize portable generator paths without changing existing user settings."""
from pathlib import Path
import json
from datetime import datetime, timezone
import shutil

ROOT = Path(__file__).resolve().parents[1]


def configure(root=ROOT):
    path = root / 'nte-dialogue-ocr/config.json'
    example = root / 'nte-dialogue-ocr/config.example.json'
    value = json.loads((path if path.exists() else example).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or not isinstance(value.get('dubbing', {}), dict):
        raise ValueError('Existing generator settings must be a JSON object')
    before = json.dumps(value, sort_keys=True)
    options = value.setdefault('dubbing', {})
    for key, setting in dict(code_root='../wuwa-dubber', data_root='../wuwa-dubber/data',
                             mode='system', enabled=False).items():
        options.setdefault(key, setting)
    changed = not path.exists() or before != json.dumps(value, sort_keys=True)
    if changed:
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
            backup = path.parent / 'backups' / ('before-generator-install-' + stamp) / path.name
            backup.parent.mkdir(parents=True)
            shutil.copy2(path, backup)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)
    return dict(settings_updated=changed)


if __name__ == '__main__':
    print(json.dumps(configure()))
