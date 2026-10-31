"""Local voice selection and cancellable playback preview."""
from __future__ import annotations

import json
from pathlib import Path
import threading

from PySide6 import QtCore, QtWidgets

from dialogue_dubbing import ROOT, VoiceRuntime
from system_voices import PRESETS, SystemVoiceRouter, preset_profile


class PreviewWorker(QtCore.QThread):
    message = QtCore.Signal(str)

    def __init__(self, options, preset, text, parent=None):
        super().__init__(parent)
        self.options, self.preset, self.text = options, preset, text
        self.stop_event = threading.Event()
        self.runtime = self.session = None

    def cancel(self):
        self.stop_event.set()
        if self.session:
            self.session.cancel()
        elif self.runtime:
            self.runtime.client.cancel()
            self.runtime.player.stop()

    def run(self):
        try:
            from wuwa_dubber.models import Utterance
            self.runtime = VoiceRuntime(dict(self.options, mode="system", system_default_voice=self.preset,
                                             system_character_voices={}))
            self.runtime.prepare(self.stop_event, self.message.emit)
            if self.stop_event.is_set():
                return
            self.session = self.runtime.session()
            profile = preset_profile(self.preset)
            self.session.run(Utterance(self.text, speaker=profile.speaker, source="voice_preview"),
                             profile, cancel_event=self.stop_event)
            self.message.emit("试听完成。实际剧情将使用相同的音色设置。")
        except Exception as exc:
            self.message.emit("试听已停止" if self.stop_event.is_set() else "试听失败：" + str(exc))


class SystemVoiceDialog(QtWidgets.QDialog):
    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统声线与试听")
        self.resize(680, 620)
        self.options = dict(options)
        self.runtime = VoiceRuntime(dict(options, mode="system"))
        self.worker = None
        self.closing = False
        self.result_options = None
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("使用本机中文声音，不占用角色模型显存。少年、青年、中年、老年是试制听感预设，"
                               "基于 3 个成年系统声音调整音调与语速。")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.default_voice = self.combo(options.get("system_default_voice", "default"))
        form.addRow("默认声音", self.default_voice)
        layout.addLayout(form)
        hint = QtWidgets.QLabel("角色名不确定或未指定声音时使用默认声音。已有角色的别名也会匹配；不按名字猜测年龄或性别。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["角色名", "系统声线"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)
        bindings = options.get("system_character_voices", {})
        names = set(bindings) | {p.speaker for p in self.runtime.profiles.values()}
        for log in sorted((ROOT / "records").glob("*/*/dialogue.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
            try:
                for line in log.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    if record.get("speaker_status") == "verified" and record.get("speaker"):
                        names.add(record["speaker"])
            except (OSError, ValueError):
                continue
        for name in sorted(names):
            self.add_row(name, bindings.get(name, ""))
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("添加角色")
        add.clicked.connect(lambda: self.add_row())
        remove = QtWidgets.QPushButton("删除所选角色设置")
        remove.clicked.connect(self.remove_rows)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        layout.addLayout(row)
        self.text = QtWidgets.QLineEdit("你好，准备好了吗？别担心，有我在。我们一起出发吧。")
        self.text.setMaxLength(500)
        layout.addWidget(self.text)
        row = QtWidgets.QHBoxLayout()
        self.preview_voice = self.combo(self.default_voice.currentData())
        self.preview = QtWidgets.QPushButton("试听所选声音")
        self.preview.clicked.connect(self.start_preview)
        row.addWidget(self.preview_voice, 1)
        row.addWidget(self.preview)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel("可先试听，再为角色选择声音。保存后开始识别即可应用。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setText("保存")
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    @staticmethod
    def combo(value, inherit=False):
        combo = QtWidgets.QComboBox()
        if inherit:
            combo.addItem("跟随默认声音", "")
        for identity, (label, *_rest) in PRESETS.items():
            combo.addItem(label, identity)
        index = combo.findData(value)
        if index < 0:
            raise ValueError(f"未知的系统声线设置：{value}")
        combo.setCurrentIndex(index)
        return combo

    def add_row(self, name="", preset=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        self.table.setCellWidget(row, 1, self.combo(preset, inherit=True))
        if not name:
            self.table.setCurrentCell(row, 0)
            self.table.editItem(self.table.item(row, 0))

    def remove_rows(self):
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def collect(self):
        from wuwa_dubber.text_processing import normalize_speaker
        bindings, seen = {}, set()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            preset = self.table.cellWidget(row, 1).currentData()
            if not name and not preset:
                continue
            normalized = normalize_speaker(name)
            if not normalized:
                raise ValueError("请填写角色名，或删除空白行")
            if normalized in seen:
                raise ValueError(f"角色名重复：{name}")
            seen.add(normalized)
            if preset:
                bindings[name] = preset
        options = dict(self.options, system_default_voice=self.default_voice.currentData(), system_character_voices=bindings)
        SystemVoiceRouter(options, self.runtime.profiles.values())
        return options

    def save(self):
        if self.worker and self.worker.isRunning():
            return
        try:
            self.result_options = self.collect()
            self.accept()
        except ValueError as exc:
            self.status.setText(str(exc))

    def start_preview(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("正在停止试听…")
            return
        if not self.text.text().strip():
            self.status.setText("请先填写试听台词")
            return
        self.worker = PreviewWorker(self.options, self.preview_voice.currentData(), self.text.text(), self)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.preview_finished)
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)
        self.preview.setText("停止试听")
        self.worker.start()

    def preview_finished(self):
        self.preview.setText("试听所选声音")
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(True)
        if self.closing:
            super().reject()

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.closing = True
            self.worker.cancel()
            self.status.setText("正在停止试听，随后关闭…")
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            event.ignore()
            self.reject()
        else:
            event.accept()
