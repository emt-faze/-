"""Persistent offline dialogue jobs. Generation never starts an audio player.

OCR evidence is copied into a project; source records are always read-only.
The caller owns the project while run() is active (the Qt UI disables editing).
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import threading
import time
from uuid import uuid4
import wave


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


EVIDENCE_FIELDS = (
    "dialogue_id", "event_id", "revision", "body", "speaker", "status", "speaker_status",
    "speaker_identity", "speaker_identity_confirmed", "body_confidence", "review_body_confidence",
    "quality_threshold", "human_confirmed", "source_clipped", "coverage_warning",
    "review_coverage_warning", "original_text", "review_text", "recognized_at", "source_kind",
    "body_image", "speaker_image",
)


@dataclass
class DialogueRow:
    id: str
    source: str
    source_id: str
    revision: int
    evidence: dict
    speaker: str
    text: str
    voice: str = "auto"
    selected: bool = True
    confirmed: bool = False
    edited: bool = False
    status: str = "pending"
    error: str = ""
    audio: dict = field(default_factory=dict)

    @property
    def punctuation_only(self):
        return bool(self.text.strip()) and not any(char.isalnum() for char in self.text)

    def review_reason(self):
        if not self.text.strip():
            return "台词为空"
        if self.confirmed:
            return ""
        if self.edited:
            return "已编辑，请核对后确认"
        record = self.evidence
        if record.get("human_confirmed"):
            return ""
        if any(record.get(k) for k in ("source_clipped", "coverage_warning", "review_coverage_warning")):
            return "原图可能裁切或漏字，请核对"
        if record.get("status") != "verified":
            return "OCR 尚未复核一致"
        try:
            threshold = max(.9, float(record.get("quality_threshold", .9)))
            confidence = float(record.get("body_confidence", 0))
            reviewed = record.get("review_body_confidence")
            if not math.isfinite(confidence) or not math.isfinite(threshold) or (reviewed is not None and not math.isfinite(float(reviewed))):
                return "识别分数缺失或无效"
            if confidence < threshold or (reviewed is not None and float(reviewed) < threshold):
                return "识别分数不足，请核对"
        except (ValueError, TypeError):
            return "识别分数缺失或无效"
        return ""

    def input_key(self, mode):
        return digest(dict(text=self.text, speaker=self.speaker, voice=self.voice, mode=mode))

    def route_record(self):
        record = dict(self.evidence, body=self.text, speaker=self.speaker)
        if self.confirmed:
            record.update(speaker_status="verified", speaker_identity=self.speaker,
                          speaker_identity_confirmed=bool(self.speaker))
        return record

    def set_confirmed(self, value):
        def trusted_name(record):
            if record.get("speaker_status") == "verified":
                return record.get("speaker", "")
            return record.get("speaker_identity", "") if record.get("speaker_identity_confirmed") else ""
        previous = trusted_name(self.route_record())
        self.confirmed = bool(value)
        if self.voice == "auto" and previous != trusted_name(self.route_record()):
            self.status, self.error, self.audio = "pending", "", {}

    def edit(self, *, text=None, speaker=None, voice=None):
        changed = False
        for key, value in (("text", text), ("speaker", speaker), ("voice", voice)):
            if value is not None and getattr(self, key) != value:
                setattr(self, key, value)
                changed = True
                if key != "voice":
                    self.edited, self.confirmed = True, False
        if changed:
            self.status, self.error, self.audio = "pending", "", {}
        return changed


@dataclass
class DeletedEntry:
    index: int
    row: DialogueRow


@dataclass
class DeletedBatch:
    entries: list[DeletedEntry]
    deleted_at: str = field(default_factory=timestamp)


class GenerationProject:
    VERSION = 2

    def __init__(self, directory, *, mode="system", rows=None, created=None, deleted_batches=None):
        if mode not in {"system", "character"}:
            raise ValueError("不支持的生成模式")
        self.directory = Path(directory).resolve()
        self.mode = mode
        self.rows = list(rows or [])
        self.deleted_batches = list(deleted_batches or [])
        self.created = created or timestamp()

    @classmethod
    def create(cls, parent, mode="system"):
        folder = Path(parent) / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
        project = cls(folder, mode=mode)
        project.save()
        return project

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / "project.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("version") not in (1, cls.VERSION) or not isinstance(data.get("rows"), list):
            raise ValueError("不是受支持的台词生成项目")
        identities = set()
        def load_row(item):
            row = DialogueRow(**item)
            if (not re.fullmatch(r"[a-f0-9]{32}", row.id) or row.id in identities
                    or not all(isinstance(v, str) for v in (row.speaker, row.text, row.voice, row.source, row.source_id))
                    or not isinstance(row.evidence, dict) or not isinstance(row.audio, dict)):
                raise ValueError("项目中的台词数据不完整")
            identities.add(row.id)
            if row.status == "generating":
                row.status, row.error = "cancelled", "上次生成未结束，可继续生成"
            return row
        rows = [load_row(item) for item in data["rows"]]
        deleted_batches = []
        for batch in data.get("deleted_batches", []):
            entries = []
            for entry in batch["entries"]:
                if type(entry["index"]) is not int or entry["index"] < 0:
                    raise ValueError("删除记录的位置无效")
                entries.append(DeletedEntry(entry["index"], load_row(entry["row"])))
            deleted_batches.append(DeletedBatch(entries, batch["deleted_at"]))
        return cls(path.parent, mode=data["mode"], rows=rows, created=data.get("created"), deleted_batches=deleted_batches)

    def all_rows(self):
        return self.rows + [entry.row for batch in self.deleted_batches for entry in batch.entries]

    def delete_rows(self, ids):
        if any(row.status == "generating" for row in self.rows):
            raise ValueError("请先停止生成，再删除台词")
        identities = set(ids)
        entries = [DeletedEntry(index, row) for index, row in enumerate(self.rows) if row.id in identities]
        if entries:
            self.deleted_batches.append(DeletedBatch(entries))
            self.rows = [row for row in self.rows if row.id not in identities]
        return len(entries)

    def undo_delete(self):
        if any(row.status == "generating" for row in self.rows):
            raise ValueError("请先停止生成，再恢复台词")
        if not self.deleted_batches:
            return []
        batch = self.deleted_batches.pop()
        for entry in sorted(batch.entries, key=lambda item: item.index):
            self.rows.insert(min(entry.index, len(self.rows)), entry.row)
        return [entry.row.id for entry in batch.entries]

    def set_mode(self, mode):
        if mode not in {"system", "character"}:
            raise ValueError("不支持的生成模式")
        if mode != self.mode:
            self.mode = mode
            for row in self.all_rows():
                row.voice, row.status, row.audio, row.error = "auto", "pending", {}, ""

    def import_records(self, path):
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "dialogue.jsonl"
        source = str(path.parent)
        newest, warnings = {}, []
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identity = record["dialogue_id"]
                if not isinstance(identity, str) or not identity or not isinstance(record["body"], str):
                    raise ValueError("缺少台词编号或文字")
                if not isinstance(record.get("speaker", ""), str):
                    raise ValueError("角色名不是文字")
                revision = int(record.get("revision", 0))
                if revision < 0:
                    raise ValueError("修订编号无效")
                if identity not in newest or revision > int(newest[identity].get("revision", 0)):
                    newest[identity] = record
            except (ValueError, TypeError, KeyError) as exc:
                warnings.append(f"第 {number} 行已跳过：{exc}")
        return self._merge_records(newest, source, warnings)

    def receive_record(self, record, source_directory):
        """Accept a persisted OCR revision on the GUI thread, using the import rules."""
        if (not isinstance(record, dict) or not isinstance(record.get("dialogue_id"), str)
                or not record["dialogue_id"] or not isinstance(record.get("body"), str)
                or not isinstance(record.get("speaker", ""), str)
                or not isinstance(record.get("revision"), int) or record["revision"] < 0):
            raise ValueError("识别记录不完整，未加入项目")
        return self._merge_records({record["dialogue_id"]: record}, str(Path(source_directory).resolve()), [])

    def _merge_records(self, newest, source, warnings):
        added, updated, unchanged = 0, 0, 0
        # Archived rows remain tombstones. Re-import and late OCR revisions
        # update their evidence without putting them back into the active list.
        existing = {(r.source.casefold(), r.source_id): r for r in self.all_rows()}
        archived = {entry.row.id for batch in self.deleted_batches for entry in batch.entries}
        deleted = 0
        for identity, record in newest.items():
            evidence = {key: record[key] for key in EVIDENCE_FIELDS if key in record}
            revision = int(record.get("revision", 0))
            row = existing.get((source.casefold(), identity))
            if row:
                if row.id in archived:
                    deleted += 1
                if revision <= row.revision:
                    unchanged += 1
                    continue
                row.evidence, row.revision = evidence, revision
                if not row.edited:
                    row.text, row.speaker = record["body"].strip(), record.get("speaker", "").strip()
                row.confirmed, row.status, row.audio, row.error = False, "pending", {}, ""
                updated += 1
            else:
                row = DialogueRow(uuid4().hex, source, identity, revision, evidence,
                                  record.get("speaker", "").strip(), record["body"].strip())
                row.selected = not row.punctuation_only
                self.rows.append(row)
                added += 1
        return dict(added=added, updated=updated, unchanged=unchanged, warnings=warnings, deleted=deleted)

    def add_dialogue(self, text, speaker="", voice="auto"):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请填写台词正文")
        if not isinstance(speaker, str) or not isinstance(voice, str) or not voice.strip():
            raise ValueError("角色或声线无效")
        identity = uuid4().hex
        row = DialogueRow(identity, "manual", identity, 1, {"source_kind": "manual"},
                          speaker.strip(), text.strip(), voice=voice, confirmed=True, edited=True)
        row.selected = not row.punctuation_only
        self.rows.append(row)
        return row

    def add_text(self, text):
        count = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([^：:\n]{1,40})[：:]\s*(.+)$", line)
            speaker, body = (match.group(1).strip(), match.group(2).strip()) if match else ("", line)
            self.add_dialogue(body, speaker)
            count += 1
        return count

    def audio_path(self, row, *, verify=True):
        if row.punctuation_only:
            return None
        info = row.audio
        if not info or info.get("input_key") != row.input_key(self.mode):
            return None
        path = (self.directory / info.get("file", "")).resolve()
        if not path.is_relative_to(self.directory / "audio") or path.suffix.lower() != ".wav":
            return None
        if not path.is_file():
            return None
        if verify:
            from wuwa_dubber.speech_cache import valid_wav
            if not valid_wav(path) or file_digest(path) != info.get("sha256"):
                return None
        return path

    def save(self):
        payload = dict(version=self.VERSION, created=self.created, updated=timestamp(), mode=self.mode,
                       rows=[asdict(row) for row in self.rows], deleted_batches=[asdict(batch) for batch in self.deleted_batches])
        atomic_text(self.directory / "project.json", json.dumps(payload, ensure_ascii=False, indent=2))
        manifest = []
        for number, row in enumerate(self.rows, 1):
            info = row.audio if self.audio_path(row, verify=False) else {}
            manifest.append(dict(number=number, id=row.id, speaker=row.speaker, text=row.text,
                                 review=row.review_reason() or ("人工确认" if row.confirmed else "OCR 已复核"),
                                 voice=info.get("voice", ""), state="silent" if row.punctuation_only else row.status,
                                 file=info.get("file", ""), sha256=info.get("sha256", ""),
                                 duration_seconds=info.get("duration_seconds", ""), error=row.error,
                                 source=row.source, source_id=row.source_id, revision=row.revision))
        atomic_text(self.directory / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        stream = io.StringIO(newline="")
        columns = ["number", "id", "speaker", "text", "review", "voice", "state", "file", "sha256",
                   "duration_seconds", "error", "source", "source_id", "revision"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in manifest:
            # Imported dialogue is data, including when opened in a spreadsheet.
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                             for k, v in entry.items()})
        atomic_text(self.directory / "manifest.csv", "\ufeff" + stream.getvalue())


def latest_record(records_root):
    paths = list(Path(records_root).glob("*/*/dialogue.jsonl"))
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def select_voice(runtime, row):
    if row.voice == "auto":
        return runtime.select_profile(row.route_record())
    kind, _, identity = row.voice.partition(":")
    if runtime.system_mode and kind == "system":
        from system_voices import preset_profile
        return preset_profile(identity), False
    if not runtime.system_mode and kind == "character" and identity in runtime.profiles:
        return runtime.profiles[identity], False
    raise ValueError("所选声线与当前模式不一致或已被移除，请重新选择")


def join_wavs(paths, destination, stop):
    """Write a private staging file. The caller alone publishes it atomically."""
    from wuwa_dubber.speech_cache import SpeechCancelled, valid_wav
    fmt = None
    with wave.open(str(destination), "wb") as output:
        for path in paths:
            with wave.open(str(path), "rb") as source:
                current = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                if fmt is None:
                    fmt = current
                    output.setparams(source.getparams())
                elif fmt != current:
                    raise ValueError("分段音频格式不一致，未导出")
                while data := source.readframes(16384):
                    if stop.is_set():
                        raise SpeechCancelled("已停止生成")
                    output.writeframesraw(data)
    if stop.is_set():
        raise SpeechCancelled("已停止生成")
    if not valid_wav(destination):
        raise ValueError("合并音频不完整，未导出")


class BatchGenerator:
    def __init__(self, project, options, callback=lambda event: None, *, runtime_factory=None):
        from dialogue_dubbing import VoiceRuntime, voice_paths
        if runtime_factory is None:
            voice_paths(options)
        self.project = project
        self.options = dict(options, mode=project.mode, fallback_to_system=False)
        self.callback, self.runtime_factory = callback, runtime_factory or VoiceRuntime
        self.stop_event, self.finished = threading.Event(), threading.Event()
        self.lock = threading.RLock()
        self.runtime = None

    def cancel(self):
        # No process/network cancellation on the GUI thread.
        with self.lock:
            self.stop_event.set()

    def emit(self, message, **extra):
        self.callback(dict(message=message, **extra))

    def _relay_cancel(self):
        while not self.finished.wait(.08):
            if self.stop_event.is_set() and self.runtime:
                try:
                    self.runtime.client.cancel()
                except Exception:
                    pass

    def run(self, ids=None, *, force=False):
        from wuwa_dubber.models import Utterance
        from wuwa_dubber.speech_cache import SpeechCache, SpeechCancelled
        from wuwa_dubber.segmentation import split_for_speech
        requested = set(ids) if ids is not None else {r.id for r in self.project.rows if r.selected}
        rows = [r for r in self.project.rows if r.id in requested and not r.punctuation_only and not r.review_reason()]
        summary = dict(completed=0, cached=0, resumed=0, failed=0, cancelled=0,
                       silent=sum(r.id in requested and r.punctuation_only for r in self.project.rows),
                       held=sum(r.id in requested and not r.punctuation_only and bool(r.review_reason()) for r in self.project.rows))
        relay = threading.Thread(target=self._relay_cancel, daemon=True, name="batch-voice-cancel")
        relay.start()
        try:
            if not rows or self.stop_event.is_set():
                return summary
            self.runtime = self.runtime_factory(self.options)
            if self.project.mode == "character" and self.runtime.system_mode:
                raise ValueError("共享配音设置当前为系统朗读，请先在参考音频与音量窗口选择角色语音模型")
            self.runtime.prepare(self.stop_event, self.emit, playback=False)
            if self.stop_event.is_set():
                return summary
            cache = SpeechCache(self.runtime.config, self.runtime.cache_dir, self.runtime.client)
            output = self.project.directory / "audio"
            output.mkdir(parents=True, exist_ok=True)
            for number, row in enumerate(rows, 1):
                if self.stop_event.is_set():
                    break
                temp = output / f".{row.id}.{uuid4().hex}.tmp.wav"
                started = time.monotonic()
                try:
                    row.status, row.error = "generating", ""
                    self.project.save()
                    self.emit(f"正在生成 {number}/{len(rows)}：{row.speaker or '未指定角色'}", row_id=row.id,
                              current=number-1, total=len(rows))
                    profile, fallback = select_voice(self.runtime, row)
                    utterance = Utterance(row.text, speaker=profile.speaker, source="batch_generation")
                    full_key = cache.path_for(utterance, profile).stem
                    if not force and row.audio.get("cache_key") == full_key and self.project.audio_path(row):
                        with self.lock:
                            if self.stop_event.is_set():
                                raise SpeechCancelled()
                            row.status = "done"
                            summary["resumed"] += 1
                        continue
                    if row.audio.get("cache_key") != full_key:
                        row.audio = {}  # Model/reference changes invalidate the current output.
                    found = None if force else cache.lookup(utterance, profile)
                    cache_hit = found is not None
                    if found:
                        shutil.copyfile(found.path, temp)
                    elif len(row.text) <= 300:
                        speech = cache.get_or_create(utterance, profile, cancel_event=self.stop_event, force=force)
                        cache_hit = speech.cache_hit
                        shutil.copyfile(speech.path, temp)
                    else:
                        parts = split_for_speech(row.text, max_chars=300)
                        paths, cache_hit = [], True
                        for index, part in enumerate(parts, 1):
                            self.emit(f"第 {number}/{len(rows)} 条 · 分段 {index}/{len(parts)}", row_id=row.id)
                            speech = cache.get_or_create(Utterance(part, speaker=profile.speaker, source="batch_generation"),
                                                         profile, cancel_event=self.stop_event, force=force)
                            paths.append(speech.path)
                            cache_hit = cache_hit and speech.cache_hit
                        join_wavs(paths, temp, self.stop_event)
                        # Publish a full-sentence cache using a private sibling file.
                        target = cache.path_for(utterance, profile)
                        merged = target.with_name(f".{target.stem}.{uuid4().hex}.merged.wav")
                        try:
                            shutil.copyfile(temp, merged)
                            with self.lock:
                                if self.stop_event.is_set():
                                    raise SpeechCancelled()
                                merged.replace(target)
                        finally:
                            merged.unlink(missing_ok=True)
                    with wave.open(str(temp), "rb") as audio:
                        duration = audio.getnframes() / audio.getframerate()
                    checksum = file_digest(temp)
                    target = output / f"{row.id}_{full_key[:16]}.wav"
                    info = dict(file=target.relative_to(self.project.directory).as_posix(), cache_key=full_key,
                                input_key=row.input_key(self.project.mode), voice=profile.speaker,
                                default_voice=bool(fallback), sha256=checksum, duration_seconds=round(duration, 3),
                                elapsed_seconds=round(time.monotonic()-started, 3), cache_hit=cache_hit,
                                generated_at=timestamp())
                    with self.lock:
                        if self.stop_event.is_set():
                            raise SpeechCancelled()
                        temp.replace(target)
                        row.audio, row.status = info, "done"
                        summary["completed"] += 1
                        summary["cached"] += int(cache_hit)
                except Exception as exc:
                    if self.stop_event.is_set() or isinstance(exc, SpeechCancelled):
                        row.status, row.error = "cancelled", "已停止，可继续生成"
                        summary["cancelled"] += 1
                        break
                    row.status, row.error = "failed", str(exc)
                    summary["failed"] += 1
                finally:
                    temp.unlink(missing_ok=True)
                    self.project.save()
                    self.emit(f"已处理 {number}/{len(rows)}", row_id=row.id, current=number, total=len(rows))
        finally:
            self.finished.set()
            relay.join(timeout=2)
            self.project.save()
        return summary
