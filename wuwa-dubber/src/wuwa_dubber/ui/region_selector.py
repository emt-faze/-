from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QDialog
from ..models import Rect


def map_region(rectangle: QRect, displayed: QRect, width: int, height: int) -> Rect:
    """Map a preview selection to the captured image, independent of screen DPI."""
    clipped = rectangle.intersected(displayed)
    if clipped.isEmpty() or displayed.isEmpty():
        return Rect()
    left = round((clipped.left() - displayed.left()) * width / displayed.width())
    top = round((clipped.top() - displayed.top()) * height / displayed.height())
    right = round((clipped.right() + 1 - displayed.left()) * width / displayed.width())
    bottom = round((clipped.bottom() + 1 - displayed.top()) * height / displayed.height())
    return Rect(left, top, right - left, bottom - top).clamped(width, height)


class RegionSelector(QDialog):
    """Select on the actual capture, not desktop coordinates."""

    def __init__(self, image, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("框选文字区域")
        height, width = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        fmt = {1: QImage.Format.Format_Grayscale8, 3: QImage.Format.Format_BGR888,
               4: QImage.Format.Format_ARGB32}[channels]
        self.image = QImage(image.tobytes(), width, height, width * channels, fmt).copy()
        self._origin = QPoint()
        self._current = QPoint()
        self._selecting = False
        self.selected: Rect | None = None
        screen = QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen else QRect(0, 0, 1280, 900)
        self.resize(min(1120, area.width() - 80), min(800, area.height() - 80))
        self.setMinimumSize(400, 300)

    def display_rect(self) -> QRect:
        available = self.rect().adjusted(16, 56, -16, -16)
        size = self.image.size().scaled(available.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(available.x() + (available.width() - size.width()) // 2,
                     available.y() + (available.height() - size.height()) // 2,
                     size.width(), size.height())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.display_rect().contains(event.position().toPoint()):
            self._origin = self._current = event.position().toPoint()
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._selecting:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._selecting:
            return
        self._current = event.position().toPoint()
        self._selecting = False
        rectangle = QRect(self._origin, self._current).normalized()
        selected = map_region(rectangle, self.display_rect(), self.image.width(), self.image.height())
        if selected.width < 8 or selected.height < 8:
            self.update()
            return
        self.selected = selected
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#162131"))
        painter.setPen(QColor("#f0f4fa"))
        painter.drawText(QRect(16, 12, self.width() - 32, 36), Qt.AlignmentFlag.AlignVCenter,
                         "在捕获画面上拖动框选文字；松开确认，Esc 取消。")
        area = self.display_rect()
        painter.drawImage(area, self.image)
        if self._selecting:
            selected = QRect(self._origin, self._current).normalized().intersected(area)
            painter.fillRect(selected, QColor(70, 190, 255, 60))
            painter.setPen(QPen(QColor(70, 190, 255), 2))
            painter.drawRect(selected)
