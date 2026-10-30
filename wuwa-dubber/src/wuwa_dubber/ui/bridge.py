from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class PipelineBridge(QObject):
    pause_requested = Signal()
    replay_requested = Signal()
    jump_requested = Signal()
    snapshot = Signal(object)
    error = Signal(str)
