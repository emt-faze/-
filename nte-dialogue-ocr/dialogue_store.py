"""Materialized dialogue records plus an append-only revision trail."""
from datetime import datetime
import json
from pathlib import Path
import uuid

import cv2

STATUS_LABELS = {"candidate": "候选", "verified": "已复核", "needs_review": "待核对"}


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(".png", image)
    if not okay:
        raise RuntimeError("无法保存原始文字截图")
    encoded.tofile(str(path))


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class Recorder:
    def __init__(self, root=None, session_id=None):
        root = Path(root) if root else Path(__file__).resolve().parent
        session_id = session_id or uuid.uuid4().hex[:12]
        self.directory = root / "records" / datetime.now().strftime("%Y-%m-%d") / session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = {}

    def upsert(self, event, evidence=None):
        identity = event["dialogue_id"]
        if not all(c.isalnum() or c in "-_" for c in identity):
            raise ValueError("Invalid dialogue ID")
        previous = self.records.get(identity)
        if previous is not None and event["revision"] <= previous["revision"]:
            return previous
        record = dict(event)
        if previous:
            for key in ("body_image", "speaker_image"):
                if key in previous:
                    record[key] = previous[key]
        else:
            if evidence is None or evidence.get("body") is None:
                raise ValueError("A new dialogue requires its original body screenshot")
            for key in ("body", "speaker"):
                image = evidence.get(key)
                if image is not None:
                    relative = Path("images") / f"{identity}_{key}.png"
                    write_png(self.directory / relative, image)
                    record[key+"_image"] = relative.as_posix()
        if record.get("revision_reason") == "speaker_stabilized" and evidence and evidence.get("speaker") is not None:
            relative = Path("images") / f"{identity}_speaker_confirmation_r{record['revision']}.png"
            write_png(self.directory / relative, evidence["speaker"])
            record["speaker_confirmation_image"] = relative.as_posix()
        record["saved_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        atomic_text(self.directory/f"{identity}.json", json.dumps(record, ensure_ascii=False, indent=2))
        with (self.directory/"revisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+"\n")
        self.records[identity] = record
        ordered = sorted(self.records.values(), key=lambda item: item["event_id"])
        atomic_text(self.directory/"dialogue.jsonl", "".join(json.dumps(item, ensure_ascii=False)+"\n" for item in ordered))
        sections = []
        for item in ordered:
            label = STATUS_LABELS.get(item["status"], item["status"])
            text = f"[{item['recognized_at']}] {item.get('speaker', '')} · {label} · 修订 {item['revision']}\n{item['body']}"
            if item.get("review_text") is not None and item["review_text"] != item["original_text"]:
                text += "\n复核候选："+item["review_text"]
            sections.append(text)
        atomic_text(self.directory/"dialogue.txt", "\n\n".join(sections)+"\n")
        return record
