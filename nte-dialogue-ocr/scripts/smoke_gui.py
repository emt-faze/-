"""Check first-run GUI startup without attaching to or driving the game."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6 import QtCore, QtGui, QtWidgets
import app


def main():
    out = ROOT / 'validation/package-smoke'
    out.mkdir(parents=True, exist_ok=True)
    qt = QtWidgets.QApplication([])
    QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    result = dict(mode='offscreen_first_launch_no_game', success=False)

    def finish():
        try:
            window = next(w for w in qt.topLevelWidgets() if w.__class__.__name__ == 'Main')
            result.update(backend=window.backend_choice.currentData(), ocr_mode=window.ocr_choice.currentData(),
                          calibration_required=not window.config['calibrated'],
                          status=window.status.text(), not_topmost=not bool(window.windowFlags() & QtCore.Qt.WindowStaysOnTopHint))
            result['success'] = result['backend'] == 'wgc' and result['ocr_mode'] == 'rapid_dual' and result['calibration_required'] and result['not_topmost']
            window.grab().save(str(out / 'main.png'))
            window.close()
        except Exception as error:
            result['error'] = repr(error)
        finally:
            (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            qt.quit()

    QtCore.QTimer.singleShot(700, finish)
    with patch.object(app, 'list_game_windows', return_value=[]), \
         patch.object(app, 'CONFIG', out / 'missing-personal-config.json'), \
         patch('game_launch.relaunch_admin', side_effect=AssertionError('Unexpected admin request')), \
         patch.object(app, 'run_loop', side_effect=AssertionError('Unexpected game capture')):
        app.gui()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
