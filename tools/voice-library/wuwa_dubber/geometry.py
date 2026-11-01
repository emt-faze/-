"""框选坐标换算（纯函数，不依赖 Qt，可单独测试）。

约定：``Rect`` 永远以「捕获帧图像左上角」为原点、以帧图像像素为单位。
框选器在冻结帧上取点，这里负责把窗口内的逻辑坐标换算成帧像素坐标。
"""

from __future__ import annotations

from .models import Rect


def widget_point_to_frame(
    x: float,
    y: float,
    widget_width: int,
    widget_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    """窗口内逻辑坐标 -> 帧图像像素坐标。

    缩放比例由「帧尺寸 / 窗口尺寸」直接得出，因此系统缩放 125%/150%、
    多屏、窗口大小变化都不需要额外处理。
    """
    safe_widget_w = max(1, int(widget_width))
    safe_widget_h = max(1, int(widget_height))
    safe_frame_w = max(1, int(frame_width))
    safe_frame_h = max(1, int(frame_height))
    return (
        x * safe_frame_w / safe_widget_w,
        y * safe_frame_h / safe_widget_h,
    )


def widget_rect_to_frame_rect(
    left: float,
    top: float,
    right: float,
    bottom: float,
    widget_width: int,
    widget_height: int,
    frame_width: int,
    frame_height: int,
) -> Rect:
    """把窗口内的一条对角线换算成帧像素矩形（已夹到画面内）。"""
    x1, y1 = widget_point_to_frame(
        left, top, widget_width, widget_height, frame_width, frame_height
    )
    x2, y2 = widget_point_to_frame(
        right, bottom, widget_width, widget_height, frame_width, frame_height
    )
    frame_w = max(1, int(frame_width))
    frame_h = max(1, int(frame_height))

    x_lo = int(round(min(x1, x2)))
    y_lo = int(round(min(y1, y2)))
    x_hi = int(round(max(x1, x2)))
    y_hi = int(round(max(y1, y2)))

    x_lo = max(0, min(x_lo, frame_w))
    y_lo = max(0, min(y_lo, frame_h))
    x_hi = max(0, min(x_hi, frame_w))
    y_hi = max(0, min(y_hi, frame_h))

    return Rect(
        left=x_lo,
        top=y_lo,
        width=max(0, x_hi - x_lo),
        height=max(0, y_hi - y_lo),
        ref_width=frame_w,
        ref_height=frame_h,
    )
