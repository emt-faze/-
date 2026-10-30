from __future__ import annotations

from collections.abc import Callable


class HotkeyManager:
    """Small optional wrapper around pynput global hotkeys."""

    def __init__(self, callbacks: dict[str, Callable[[], None]]) -> None:
        self.callbacks = callbacks
        self._listener = None

    def start(self) -> None:
        if self._listener is not None:
            return
        try:
            from pynput.keyboard import GlobalHotKeys
        except ImportError:
            return
        self._listener = GlobalHotKeys(self.callbacks)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
