"""Add shared character profiles atomically from local audio or video clips."""
from __future__ import annotations

import array
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
from uuid import uuid4
import wave

from PySide6 import QtCore, QtWidgets

from dialogue_dubbing import voice_paths


class ImportCancelled(Exception):
    pass


def load_profiles(data_root):
    from wuwa_dubber.models import VoiceProfile
    profiles = []
    for path in sorted((Path(data_root) / "profiles").glob("*/profile.json")):
        try:
            profiles.append(VoiceProfile.from_dict(json.loads(path.read_text(encoding="utf-8-sig"))))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"已有声线资料无法读取，请先检查：{path.parent.name}") from exc
    return profiles


def check_names(data_root, speaker, aliases):
    from wuwa_dubber.text_processing import normalize_speaker
    requested = {}
    for name in [speaker, *aliases]:
        key = normalize_speaker(name).casefold()
        if not key:
            raise ValueError("角色名和别名必须包含有效文字")
        if key in requested:
            raise ValueError("新角色的名称或别名重复：" + name)
        requested[key] = name
    for profile in load_profiles(data_root):
        for name in [profile.speaker, *profile.aliases]:
            key = normalize_speaker(name).casefold()
            if key in requested:
                raise ValueError(f"“{requested[key]}”已属于声线“{profile.speaker}”，请使用不同的角色名或别名")


class VoiceImportTask:
    def __init__(self, options, values, progress=lambda message: None, *, data_root=None, ffmpeg=None):
        _, shared = voice_paths(options)
        self.data_root = Path(data_root or shared).resolve()
        self.values, self.progress = dict(values), progress
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        from wuwa_dubber.config import AppConfig
        self.config = AppConfig.from_dict(json.loads((shared / "config.json").read_text(encoding="utf-8-sig")))
        self.ffmpeg = ffmpeg or self.config.ffmpeg_path

    def cancel(self):
        with self.lock:
            self.stop_event.set()

    def check_cancel(self):
        if self.stop_event.is_set():
            raise ImportCancelled("已取消，未添加声线")

    def convert(self, source, destination, start, duration):
        from wuwa_dubber.tool_discovery import discover_ffmpeg
        executable = self.ffmpeg
        if not Path(executable).is_file() and not shutil.which(executable):
            executable, _ = discover_ffmpeg()
        command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-ss", str(start), "-i", str(source), "-t", str(duration), "-map", "0:a:0", "-vn",
                   "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", "-af",
                   "highpass=f=80,loudnorm=I=-20:LRA=11:TP=-1.5", str(destination)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        began = time.monotonic()
        try:
            while True:
                self.check_cancel()
                if time.monotonic()-began > 90:
                    raise ValueError("处理参考片段超时，请换用较短的本地素材")
                try:
                    _, error = process.communicate(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            self.check_cancel()
            if process.returncode:
                detail = error.decode("utf-8", errors="replace").strip()[-1200:]
                raise ValueError("无法读取声音，请确认文件含有音轨且起始时间有效：" + detail)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    @staticmethod
    def check_audio(path):
        from wuwa_dubber.speech_cache import valid_wav
        if not valid_wav(path):
            raise ValueError("没有得到完整音频，请检查起始时间和文件音轨")
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes()/audio.getframerate()
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or not 2.99 <= duration <= 30.01:
                raise ValueError(f"实际片段为 {duration:.2f} 秒，需要 3–30 秒；请调整截取范围")
            samples = array.array("h", audio.readframes(audio.getnframes()))
            if max(map(abs, samples), default=0) < 64:
                raise ValueError("片段几乎没有声音，请重新选择有人说话的范围")
        return duration

    def run(self, *, preview=False):
        from wuwa_dubber.models import VoiceProfile
        values = self.values
        source = Path(values["source"]).expanduser().resolve()
        if not source.is_file():
            raise ValueError("请选择本机的音频或录屏文件")
        start, duration = float(values["start"]), float(values["duration"])
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or not 3 <= duration <= 30:
            raise ValueError("起始时间需大于等于 0，片段时长需为 3–30 秒")
        speaker, aliases = values["speaker"].strip(), list(values["aliases"])
        identity = "added-" + uuid4().hex
        profile = VoiceProfile(speaker, aliases=aliases, prompt_text=values["prompt"].strip(),
                               speed=float(values["speed"]), reference_audio=f"{identity}/reference.wav")
        if not preview:
            errors = profile.validate(self.data_root / "profiles", require_audio=False)
            if errors:
                raise ValueError("；".join(errors))
            check_names(self.data_root, speaker, aliases)
        self.check_cancel()
        scratch = self.data_root / ".voice-imports"
        stage = scratch / identity
        stage.mkdir(parents=True, exist_ok=False)
        try:
            self.progress("正在截取并检查参考片段…")
            reference = stage / "reference.wav"
            self.convert(source, reference, start, duration)
            self.check_cancel()
            actual_duration = self.check_audio(reference)
            if preview:
                from wuwa_dubber.audio.player import AudioPlayer
                self.progress("正在试听所选参考片段；可点击停止试听。")
                AudioPlayer(self.config.output_device, volume=self.config.output_volume).play(reference, cancel_event=self.stop_event)
                self.check_cancel()
                return None
            (stage / "profile.json").write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            provenance = dict(source=str(source), start_seconds=start, requested_seconds=duration,
                              actual_seconds=actual_duration, operation="user_imported_reference_clip")
            (stage / "import.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
            lease = QtCore.QLockFile(str(self.data_root / "voice-import.lock"))
            lease.setStaleLockTime(0)
            if not lease.tryLock(0):
                raise ValueError("另一个窗口正在添加声线，请稍后重试")
            try:
                # Recheck after conversion in case another importer added an alias.
                check_names(self.data_root, speaker, aliases)
                target = self.data_root / "profiles" / identity
                target.parent.mkdir(parents=True, exist_ok=True)
                with self.lock:
                    self.check_cancel()
                    stage.rename(target)
            finally:
                lease.unlock()
            return profile.to_dict()
        finally:
            # Only this task's private directory is eligible for cleanup.
            if stage.exists():
                resolved, allowed = stage.resolve(), scratch.resolve()
                if resolved != allowed and resolved.is_relative_to(allowed):
                    shutil.rmtree(resolved)


class VoiceImportWorker(QtCore.QThread):
    message = QtCore.Signal(str)
    saved = QtCore.Signal(object)

    def __init__(self, options, values, preview=False, parent=None):
        super().__init__(parent)
        self.preview = preview
        self.task = VoiceImportTask(options, values, self.message.emit)

    def cancel(self):
        self.task.cancel()

    def run(self):
        try:
            result = self.task.run(preview=self.preview)
            if result is not None:
                self.saved.emit(result)
            else:
                self.message.emit("参考片段试听完成。请填写片段中实际说出的台词后保存。")
        except ImportCancelled:
            self.message.emit("试听已停止" if self.preview else "已取消，未添加声线")
        except Exception as exc:
            self.message.emit("未完成：" + str(exc))


class AddVoiceDialog(QtWidgets.QDialog):
    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.options = dict(options)
        _, data = voice_paths(options)
        self.worker, self.result_profile = None, None
        self.closing = False
        self.setWindowTitle("添加角色声线")
        self.resize(700, 600)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("选择只包含该角色说话的音频或录屏，取用 3–30 秒。建议选 5–15 秒、背景音乐较少的片段。\n"
                               "请在完整一句说完后截取，末尾保留约 0.3 秒停顿。先试听，再逐字填写实际听到的内容；"
                               "录音没说完的字不要写入，否则可能混到新配音开头。")
        note.setWordWrap(True)
        layout.addWidget(note)
        names = "、".join(profile.speaker for profile in load_profiles(data)) or "暂无"
        existing = QtWidgets.QLabel("已有角色：" + names)
        existing.setWordWrap(True)
        layout.addWidget(existing)
        self.form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.form_widget)
        self.speaker = QtWidgets.QLineEdit()
        self.aliases = QtWidgets.QLineEdit()
        self.aliases.setPlaceholderText("可选；多个别名用逗号分隔，供识别结果匹配角色")
        self.source = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("选择音频 / 录屏…")
        browse.clicked.connect(self.browse)
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(self.source, 1)
        file_row.addWidget(browse)
        self.start = QtWidgets.QDoubleSpinBox()
        self.start.setRange(0, 86400)
        self.start.setDecimals(2)
        self.start.setSuffix(" 秒")
        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(3, 30)
        self.duration.setDecimals(2)
        self.duration.setValue(10)
        self.duration.setSuffix(" 秒")
        clip = QtWidgets.QHBoxLayout()
        clip.addWidget(QtWidgets.QLabel("从"))
        clip.addWidget(self.start)
        clip.addWidget(QtWidgets.QLabel("开始，取用"))
        clip.addWidget(self.duration)
        clip.addStretch()
        self.prompt = QtWidgets.QPlainTextEdit()
        self.prompt.setPlaceholderText("逐字填写所选片段中实际说出的内容（包括开头笑声、语气词）。请勿补入截取范围外或尚未说完的字。")
        self.prompt.setMinimumHeight(115)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(.5, 2)
        self.speed.setSingleStep(.05)
        self.speed.setValue(1)
        form.addRow("角色名", self.speaker)
        form.addRow("角色别名", self.aliases)
        form.addRow("参考素材", file_row)
        form.addRow("截取范围", clip)
        form.addRow("片段中的台词", self.prompt)
        form.addRow("生成语速", self.speed)
        layout.addWidget(self.form_widget)
        self.status = QtWidgets.QLabel("同名角色及别名冲突会提示处理，已有声线保持不变。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.preview = QtWidgets.QPushButton("试听所选片段")
        self.stop = QtWidgets.QPushButton("停止试听 / 添加")
        self.save = QtWidgets.QPushButton("保存新角色声线")
        close = QtWidgets.QPushButton("关闭")
        self.preview.clicked.connect(lambda: self.start_task(True))
        self.save.clicked.connect(lambda: self.start_task(False))
        self.stop.clicked.connect(self.stop_task)
        close.clicked.connect(self.close)
        self.stop.setEnabled(False)
        for button in (self.preview, self.stop, self.save, close):
            row.addWidget(button)
        layout.addLayout(row)

    def browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择角色音频或游戏录屏", "",
                                                       "音频与视频 (*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.mp4 *.mkv *.mov *.webm)")
        if path:
            self.source.setText(path)

    def values(self):
        return dict(speaker=self.speaker.text().strip(),
                    aliases=[part.strip() for part in self.aliases.text().replace("，", ",").split(",") if part.strip()],
                    source=self.source.text().strip(), prompt=self.prompt.toPlainText().strip(),
                    start=self.start.value(), duration=self.duration.value(), speed=self.speed.value())

    def start_task(self, preview=False):
        if self.worker and self.worker.isRunning():
            return
        try:
            self.result_profile = None
            self.worker = VoiceImportWorker(self.options, self.values(), preview, self)
            self.worker.message.connect(self.status.setText)
            self.worker.saved.connect(lambda profile: setattr(self, "result_profile", profile))
            self.worker.finished.connect(self.finished_task)
            self.form_widget.setEnabled(False)
            self.preview.setEnabled(False)
            self.save.setEnabled(False)
            self.stop.setEnabled(True)
            self.worker.start()
        except Exception as exc:
            self.status.setText("无法开始：" + str(exc))

    def stop_task(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.stop.setEnabled(False)
            self.status.setText("正在停止，请稍候…")

    def finished_task(self):
        self.form_widget.setEnabled(True)
        self.preview.setEnabled(True)
        self.save.setEnabled(True)
        self.stop.setEnabled(False)
        if self.result_profile:
            self.accept()
        elif self.closing:
            self.close()

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.closing = True
            self.stop_task()
            event.ignore()
        else:
            event.accept()
