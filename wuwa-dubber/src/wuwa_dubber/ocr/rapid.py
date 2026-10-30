from __future__ import annotations

from typing import Any

from .base import OcrLine


class RapidOcrEngine:
    """Small CPU OCR adapter that works with both RapidOCR package generations."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("未安装 RapidOCR，请安装项目 ocr 依赖") from exc
        self._engine = RapidOCR()

    def recognize(self, image: Any) -> list[OcrLine]:
        result = self._engine(image)
        raw = result[0] if isinstance(result, tuple) and result else getattr(result, "boxes", result)
        if raw is None:
            return []
        lines: list[OcrLine] = []
        for item in raw:
            try:
                box, text, confidence = item[0], str(item[1]), float(item[2])
            except (TypeError, ValueError, IndexError):
                continue
            if text.strip():
                lines.append(OcrLine(text=text, confidence=confidence, box=box))
        return _sort_lines(lines)


def _sort_lines(lines: list[OcrLine]) -> list[OcrLine]:
    def key(line: OcrLine) -> tuple[float, float]:
        box = line.box
        try:
            points = list(box)
            top = min(float(point[1]) for point in points)
            left = min(float(point[0]) for point in points)
            return top, left
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0

    return sorted(lines, key=key)
