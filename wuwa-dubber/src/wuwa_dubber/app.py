from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import AppPaths, default_data_root, load_config
from .diagnostics import collect_diagnostics
from .voice_profiles import VoiceProfileStore


def configure_chinese_font(app: object) -> None:
    from PySide6.QtGui import QFont, QFontDatabase

    families: list[str] = []
    for font_path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if not font_path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                break
    family = families[0] if families else "Microsoft YaHei UI"
    app.setFont(QFont(family, 10))  # type: ignore[attr-defined]


def _configure_logging(paths: AppPaths) -> None:
    paths.ensure()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(paths.logs / "application.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run_diagnostics(data_dir: Path | None = None) -> int:
    paths = AppPaths(data_dir.resolve() if data_dir else default_data_root())
    paths.ensure()
    config = load_config(paths)
    store = VoiceProfileStore(paths)
    for item in collect_diagnostics(config, store):
        print(f"[{item.status:7}] {item.name}: {item.detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="异环未配音剧情实时 AI 配音工具")
    parser.add_argument("--diagnostics", action="store_true", help="检查本地依赖后退出")
    parser.add_argument("--data-dir", type=Path, help="覆盖配置、声线和日志目录")
    args = parser.parse_args(argv)

    if args.diagnostics:
        return run_diagnostics(args.data_dir)

    paths = AppPaths(args.data_dir.resolve() if args.data_dir else default_data_root())
    paths.ensure()
    _configure_logging(paths)
    config = load_config(paths)
    store = VoiceProfileStore(paths)
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "缺少 PySide6。请先执行：python -m pip install -e '.[all]'",
            file=sys.stderr,
        )
        return 2
    from .ui.main_window import MainWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv[:1])
    app.setApplicationName("异环 AI 配音")
    app.setOrganizationName("LocalOnly")
    configure_chinese_font(app)
    window = MainWindow(paths, config, store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
