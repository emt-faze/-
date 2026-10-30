"""Offline OCR profiles; model provenance is kept with every dialogue."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "rapid_dual": ("快速识别＋较大模型复核（新方案）", "dialogue-v5-rapid6-dual"),
    "legacy": ("原有识别方案（对照基线）", "dialogue-v5-legacy4"),
}


class RapidModel:
    def __init__(self, size):
        from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion
        import cv2
        cv2.setNumThreads(1)
        model_root = ROOT / "assets/ocr-v6"
        manifest = json.loads((model_root / "models.json").read_text(encoding="utf-8"))
        params = {
            "Global.use_cls": False, "Global.text_score": .5,
            "Global.max_side_len": 8192, "Global.use_vertical_padding": False,
            "Global.model_root_dir": str(model_root),
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.limit_type": "max", "Det.limit_side_len": 1280,
            "Det.thresh": .3, "Det.box_thresh": .5, "Det.unclip_ratio": 1.6,
        }
        self.model_hashes = {}
        for part in ("Det", "Rec"):
            path = model_root / f"PP-OCRv6_{part.lower()}_{size}.onnx"
            if not path.is_file():
                raise RuntimeError("缺少本地中文模型：" + path.name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != manifest[path.name]["sha256"]:
                raise RuntimeError("本地中文模型校验失败：" + path.name)
            self.model_hashes[path.name] = actual
            params[part + ".model_path"] = str(path)
            params[part + ".engine_type"] = EngineType.ONNXRUNTIME
            params[part + ".model_type"] = getattr(ModelType, size.upper())
            params[part + ".ocr_version"] = OCRVersion.PPOCRV6
        self.model_id = "RapidOCR-3.10.0/PP-OCRv6-" + size
        self.engine = RapidOCR(params=params)

    def __call__(self, image, **kwargs):
        # RapidOCR remembers per-call flags: always restore detection after a
        # recognition-only line check.
        result = self.engine(image, use_det=True, use_cls=False, use_rec=True)
        if result.txts is None:
            return [], None
        rows = [[box.tolist(), str(text), float(score)] for box, text, score in
                zip(result.boxes, result.txts, result.scores)]
        return rows, None

    def recognize_line(self,image):
        result=self.engine(image,use_det=False,use_cls=False,use_rec=True)
        if not result.txts:
            return '',0.0
        return str(result.txts[0]),float(result.scores[0])


def make_ocr(config):
    from ocr_core import DialogueOCR, _build_engine, DEFAULT_ENGINE_OPTIONS
    mode = config.get("ocr_mode", "rapid_dual")
    if mode not in PROFILES:
        raise ValueError("未知识别方案：" + str(mode))
    if mode == "rapid_dual":
        first, review = RapidModel("small"), RapidModel("medium")
    else:
        first = review = _build_engine(DEFAULT_ENGINE_OPTIONS)
    return DialogueOCR(
        config["body_roi"], config.get("speaker_roi"),
        engine=first, review_engine=review, ocr_profile=PROFILES[mode][1],
        candidate_scale=2.0 if mode == "rapid_dual" else 1.0,
        min_confidence=config.get("verification_threshold", .9),
        blank_reset_seconds=config.get("blank_reset_seconds", .8),
        verify_scale=config.get("verify_scale", 2.0), speaker_review_scale=1.0,
        centered_body=config.get("centered_body", False),
    )
