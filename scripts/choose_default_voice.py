"""Choose an imported fallback voice on a fresh public installation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nte-dialogue-ocr'))


def save_default(data_root, speaker):
    from wuwa_dubber.models import VoiceProfile
    names = []
    for path in sorted((data_root / 'profiles').glob('*/profile.json')):
        profile = VoiceProfile.from_dict(json.loads(path.read_text(encoding='utf-8-sig')))
        if profile.enabled and not profile.validate(data_root / 'profiles', require_audio=True):
            names.append(profile.speaker)
    if speaker not in names:
        raise ValueError('请选择已启用且参考音频有效的角色声线。')
    path = data_root / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if config.get('narrator_profile') == speaker:
        return False
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    backup = data_root / 'backups' / ('before-default-voice-' + stamp) / path.name
    backup.parent.mkdir(parents=True)
    shutil.copy2(path, backup)
    config['narrator_profile'] = speaker
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)
    return True


def main():
    from PySide6 import QtCore, QtWidgets
    from dialogue_dubbing import voice_paths
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    locks = []
    try:
        options = json.loads((ROOT/'nte-dialogue-ocr/config.json').read_text(encoding='utf-8-sig')).get('dubbing', {})
        _, data_root = voice_paths(options)
        for path in (ROOT/'nte-dialogue-ocr/generator.lock', data_root/'voice-import.lock'):
            lock = QtCore.QLockFile(str(path))
            lock.setStaleLockTime(0)
            if not lock.tryLock(0):
                raise ValueError('请先关闭台词生成工具和添加声线窗口，再设置默认角色声线。')
            locks.append(lock)
        from wuwa_dubber.models import VoiceProfile
        names = []
        for path in sorted((data_root/'profiles').glob('*/profile.json')):
            profile = VoiceProfile.from_dict(json.loads(path.read_text(encoding='utf-8-sig')))
            if profile.enabled and not profile.validate(data_root/'profiles', require_audio=True):
                names.append(profile.speaker)
        names = sorted(set(names))
        if not names:
            raise ValueError('请先在台词生成工具中添加至少一个有效的角色声线。')
        config = json.loads((data_root/'config.json').read_text(encoding='utf-8-sig'))
        current = names.index(config['narrator_profile']) if config.get('narrator_profile') in names else 0
        name, accepted = QtWidgets.QInputDialog.getItem(None, '设置默认角色声线',
            '无法按角色匹配时使用的声线：\n请先关闭其他配音设置窗口。', names, current, False)
        if accepted:
            save_default(data_root, name)
            QtWidgets.QMessageBox.information(None, '设置已保存', '默认角色声线：'+name+'\n重新打开台词生成工具后生效。')
        return 0
    except Exception as exc:
        QtWidgets.QMessageBox.warning(None, '默认声线未设置', str(exc))
        return 1
    finally:
        for lock in reversed(locks):
            lock.unlock()


if __name__ == '__main__':
    raise SystemExit(main())
