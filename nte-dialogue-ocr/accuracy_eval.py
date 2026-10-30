"""Real-dialogue acceptance: immutable split, explicit transcription, CER.

No OCR text is ever copied into ground truth. Unlabelled or insufficient real
samples produce 'not_measured', never a claimed accuracy score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import unicodedata

from dialogue_store import atomic_text
from ocr_core import OCR_PROFILE

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "validation" / "accuracy" / "wgc-dual-v5" / "dataset.json"
REQUIRED_TAGS = {"long", "short", "multiline", "numbers", "negation", "names", "light_background", "dark_background"}


def split_digest(samples):
    payload = [(s['dialogue_id'], s['split'], s['body_image_sha256']) for s in samples]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest()


def normalize(text):
    # Discard visual wrapping only; retain spaces, punctuation, digits and case.
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\n", "").strip()


def distance(reference, prediction):
    previous = list(range(len(prediction)+1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(prediction, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(expected != actual)))
        previous = current
    return previous[-1]


def prepare(records_root, manifest_path=DEFAULT_MANIFEST):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split_frozen"):
            return manifest
    else:
        manifest = {"version": 2, "status": "collecting", "required_samples": 50,
                    "ocr_profile": OCR_PROFILE,
                    "known_tuning_texts": ["……味道如何？", "很少见到安魂曲来餐厅吃饭啊。", "请问有人在吗？"],
                    "split_seed": 20261029, "split_frozen": False, "samples": [],
                    "ground_truth_policy": "逐字查看原图填写，不能从识别结果自动生成；记录核对者来源"}
    ids = {item["dialogue_id"] for item in manifest["samples"]}
    raw_keys = {normalize(item["original_text"]) for item in manifest["samples"]}
    raw_keys.update(normalize(item['truth_body']) for item in manifest['samples'] if item.get('truth_body'))
    raw_keys.update(normalize(text) for text in manifest.get('development_exclusions',[]))
    records = []
    for path in Path(records_root).rglob("*.json"):
        if path.name in ("session.json", "dataset.json"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("source_kind") != "live_game_capture":
            continue
        if record.get("ocr_profile") != manifest.get("ocr_profile"):
            continue
        if record.get("review_text") is None or not record.get("body_image"):
            continue
        records.append((record.get("recognized_at", ""), path, record))
    for _, path, record in sorted(records):
        if len(manifest["samples"]) >= 50:
            break
        key = normalize(record["original_text"])
        if record["dialogue_id"] in ids or key in raw_keys:
            continue
        image = path.parent / record["body_image"]
        if not image.is_file():
            continue
        sample = {"dialogue_id": record["dialogue_id"], "record": str(path.resolve()),
                  "ocr_profile": record["ocr_profile"],
                  "body_image": str(image.resolve()), "body_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                  "original_text": record["original_text"], "review_text": record["review_text"],
                  "displayed_text": record["body"], "speaker": record.get("speaker", ""),
                  "review_speaker": record.get("review_speaker", ""), "status": record["status"],
                  "source_clipped": record.get("source_clipped",False),
                  "source_complete_confirmed": False,
                  "truth_body": None, "truth_speaker": None, "truth_checked": False,
                  "label_source": None, "tags": [], "split": None,
                  "candidate_latency_ms": record.get("candidate_latency_ms"),
                  "review_latency_ms": record.get("review_latency_ms")}
        if record.get("speaker_image"):
            name_image = path.parent/record["speaker_image"]
            if name_image.is_file():
                sample["speaker_image"] = str(name_image.resolve())
                sample["speaker_image_sha256"] = hashlib.sha256(name_image.read_bytes()).hexdigest()
        manifest["samples"].append(sample)
        ids.add(record["dialogue_id"])
        raw_keys.add(key)
    if len(manifest["samples"]) == 50:
        # A previously inspected development example cannot enter the holdout.
        known = {normalize(t) for t in manifest.get('known_tuning_texts', [])}
        tuning_ids=set(manifest.get('tuning_frozen_ids',[]))
        forced = {i for i,s in enumerate(manifest['samples']) if normalize(s['original_text']) in known or s['dialogue_id'] in tuning_ids}
        if len(forced) > 20:
            raise ValueError('Too many development samples for the 20-item tuning split')
        indices = [i for i in range(50) if i not in forced]
        random.Random(manifest["split_seed"]).shuffle(indices)
        tune = forced | set(indices[:20-len(forced)])
        for i, sample in enumerate(manifest["samples"]):
            sample["split"] = "tune" if i in tune else "acceptance"
        manifest.update(split_frozen=True, status="awaiting_transcription",
                        split_frozen_at=datetime.now(timezone.utc).isoformat(),
                        split_sha256=split_digest(manifest['samples']))
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def score_samples(samples, prediction_key, truth_key):
    characters = errors = exact = 0
    mistakes = []
    for sample in samples:
        expected = normalize(sample[truth_key])
        actual = normalize(sample.get(prediction_key) or "")
        edits = distance(expected, actual)
        characters += len(expected)
        errors += edits
        exact += expected == actual
        if edits:
            mistakes.append({"dialogue_id": sample["dialogue_id"], "expected": expected,
                             "predicted": actual, "edit_distance": edits, "image": sample["body_image"]})
    return {"samples": len(samples), "reference_characters": characters, "character_errors": errors,
            "cer": errors/characters if characters else None,
            "exact_sentences": exact, "exact_sentence_rate": exact/len(samples) if samples else None,
            "mistakes": mistakes}


def evaluate(manifest_path=DEFAULT_MANIFEST, subset="acceptance"):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    issues = []
    if len(samples) != 50 or not manifest.get("split_frozen"):
        issues.append(f"需要冻结 50 条真实台词，目前 {len(samples)} 条")
    if sum(s.get("split")=="tune" for s in samples) != 20 or sum(s.get("split")=="acceptance" for s in samples) != 30:
        issues.append("调参集/验收集必须分别为 20/30 条")
    if manifest.get('split_frozen') and manifest.get('split_sha256') != split_digest(samples):
        issues.append('冻结后的分组或原图标识发生变化，请保留原验收集')
    if any(s.get('ocr_profile') != manifest.get('ocr_profile') for s in samples):
        issues.append('样本混用了不同识别版本')
    chosen = [s for s in samples if s.get("split") == subset]
    for sample in samples:
        image = Path(sample["body_image"])
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != sample["body_image_sha256"]:
            issues.append(f"{sample['dialogue_id']} 原图缺失或已改变")
        if not sample.get("truth_checked") or not sample.get("label_source") or not isinstance(sample.get("truth_body"), str) or not sample["truth_body"].strip():
            issues.append(f"{sample['dialogue_id']} 未逐字核对正文")
        if not isinstance(sample.get("truth_speaker"), str):
            issues.append(f"{sample['dialogue_id']} 未核对角色名（没有人名时填写空字符串）")
        if sample.get("source_clipped") and not sample.get("source_complete_confirmed"):
            issues.append(f"{sample['dialogue_id']} 原图疑似截断，尚未人工确认完整")
        if sample.get('speaker_image'):
            name_image = Path(sample['speaker_image'])
            if not name_image.is_file() or hashlib.sha256(name_image.read_bytes()).hexdigest() != sample.get('speaker_image_sha256'):
                issues.append(f"{sample['dialogue_id']} 人名原图缺失或已改变")
    checked = [normalize(s["truth_body"]) for s in samples if isinstance(s.get("truth_body"),str)]
    if len(set(checked)) != len(checked):
        issues.append("存在重复的正确台词，需要补充不同样本")
    output = {"status": "not_measured", "subset": subset, "issues": issues,
              "targets": {"cer_max": .01, "exact_sentence_rate_min": .95},
              "normalization": "NFC；仅去掉换行与首尾空白；标点、数字、大小写和句内空格计入",
              "human_accuracy_guarantee": False}
    if not issues:
        output["raw"] = score_samples(chosen, "original_text", "truth_body")
        output["enlarged"] = score_samples(chosen, "review_text", "truth_body")
        output["displayed"] = score_samples(chosen, "displayed_text", "truth_body")
        output["speaker"] = score_samples(chosen, "speaker", "truth_speaker")
        output["speaker_enlarged"] = score_samples(chosen, "review_speaker", "truth_speaker")
        score = output["displayed"]
        passed = score["cer"] is not None and score["cer"] <= .01 and score["exact_sentence_rate"] >= .95
        coverage = {tag for s in samples for tag in s.get('tags', [])}
        missing = sorted(REQUIRED_TAGS-coverage)
        output['missing_coverage'] = missing
        output["status"] = ('incomplete_coverage' if missing else 'passed') if passed else "failed"
        output["needs_review_count"] = sum(s["status"]=="needs_review" for s in chosen)
        output["coverage_tags"] = sorted({tag for s in chosen for tag in s.get("tags",[])})
        import numpy as np
        output["latency_note"] = "从采样时间到结果生成；不包含文字出现到下一次采样的等待，也不等于显示器完成刷新"
        for key in ("candidate_latency_ms", "review_latency_ms"):
            values = [s[key] for s in chosen if isinstance(s.get(key),(int,float))]
            output[key] = {"median":float(np.median(values)),"p95":float(np.percentile(values,95))} if values else None
    output["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    destination = path.with_name(f"{subset}-report.json")
    atomic_text(destination, json.dumps(output, ensure_ascii=False, indent=2))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","evaluate"])
    parser.add_argument("--records",default=str(ROOT/"records"))
    parser.add_argument("--manifest",default=str(DEFAULT_MANIFEST))
    parser.add_argument("--subset",choices=["tune","acceptance"],default="acceptance")
    args=parser.parse_args()
    result=prepare(args.records,args.manifest) if args.action=="prepare" else evaluate(args.manifest,args.subset)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=="__main__":
    main()
