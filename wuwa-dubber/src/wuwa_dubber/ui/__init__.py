from .main_window import MainWindow


def create_main_window(paths, config, voice_store):  # type: ignore[no-untyped-def]
    return MainWindow(paths, config, voice_store)


__all__ = ["MainWindow", "create_main_window"]
