from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from wuwa_dubber.app import configure_chinese_font
from wuwa_dubber.config import AppConfig, AppPaths
from wuwa_dubber.ui.main_window import MainWindow
from wuwa_dubber.voice_profiles import VoiceProfileStore


def main() -> int:
    output = Path(os.environ.get("WUWA_SMOKE_OUTPUT", "artifacts/ui-smoke.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wuwa-ui-smoke-") as directory:
        paths = AppPaths(Path(directory))
        paths.ensure()
        app = QApplication([])
        configure_chinese_font(app)
        window = MainWindow(paths, AppConfig(), VoiceProfileStore(paths))
        window.resize(1100, 780)
        window.show()
        app.processEvents()
        tabs = window.centralWidget()
        saved: list[Path] = []
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            app.processEvents()
            image = window.grab().toImage()
            if image.isNull() or image.width() < 800 or image.height() < 500:
                raise RuntimeError(f"UI tab {index} is blank or incorrectly sized")
            target = output.with_name(f"{output.stem}-{index}{output.suffix}")
            if not image.save(str(target)):
                raise RuntimeError(f"Unable to save {target}")
            saved.append(target)
        print("UI smoke passed: " + ", ".join(str(path) for path in saved))
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
