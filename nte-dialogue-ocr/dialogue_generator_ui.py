"""Batch dialogue generation and preview, sharing the installed voices/cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading

from PySide6 import QtCore, QtGui, QtWidgets

from dialogue_dubbing import ROOT, VoiceRuntime, voice_paths
from dialogue_generation import BatchGenerator, GenerationProject, latest_record, select_voice
from system_voices import PRESETS


STATE_NAMES = {"pending": "待生成", "generating": "生成中", "done": "已完成",
               "failed": "失败，可重试", "cancelled": "已停止，可继续"}


class ManualDialogueDialog(QtWidgets.QDialog):
    def __init__(self, voices, speakers=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动添加台词")
        self.resize(600, 370)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.speaker = QtWidgets.QComboBox()
        self.speaker.setEditable(True)
        self.speaker.addItems([""] + sorted(set(name for name in speakers if name)))
        self.speaker.lineEdit().setPlaceholderText("可留空，也可填写新的角色名")
        form.addRow("角色", self.speaker)
        self.voice = QtWidgets.QComboBox()
        for label, identity in voices:
            self.voice.addItem(label, identity)
        form.addRow("声线", self.voice)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlaceholderText("输入一条完整台词；换行仍属于同一条台词。")
        self.text.setMinimumHeight(150)
        form.addRow("台词正文", self.text)
        layout.addLayout(form)
        note = QtWidgets.QLabel("添加后视为已人工核对；仅有省略号等标点的台词会保留，并标为无需配音。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error = QtWidgets.QLabel()
        self.error.setStyleSheet("color:#b33e35;")
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("添加台词", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton("取消", QtWidgets.QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.text.setFocus()

    def accept(self):
        if not self.text.toPlainText().strip():
            self.error.setText("请先填写台词正文。")
            self.text.setFocus()
            return
        super().accept()


class GenerationWorker(QtCore.QThread):
    update = QtCore.Signal(dict)
    result = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, project, options, ids, force=False, parent=None):
        super().__init__(parent)
        self.runner = BatchGenerator(project, options, self.update.emit)
        self.ids, self.force = ids, force

    def run(self):
        try:
            self.result.emit(self.runner.run(self.ids, force=self.force))
        except Exception as exc:
            self.error.emit("生成已停止" if self.runner.stop_event.is_set() else "生成失败：" + str(exc))

    def cancel(self):
        self.runner.cancel()


class AudioPreviewWorker(QtCore.QThread):
    message = QtCore.Signal(str)

    def __init__(self, path, config, parent=None):
        super().__init__(parent)
        self.path, self.config = path, config
        self.stop_event = threading.Event()

    def cancel(self):
        self.stop_event.set()

    def run(self):
        try:
            import sounddevice as sd
            from wuwa_dubber.audio.player import AudioPlayer
            sd.check_output_settings(device=self.config.output_device, channels=1)
            if not self.stop_event.is_set():
                player = AudioPlayer(self.config.output_device, volume=self.config.output_volume)
                player.play(self.path, cancel_event=self.stop_event)
            self.message.emit("试听已停止" if self.stop_event.is_set() else "试听完成")
        except Exception as exc:
            self.message.emit("试听失败：" + str(exc))


class GeneratorWindow(QtWidgets.QDialog):
    def __init__(self, options, parent=None, *, root=ROOT, project_path=None):
        super().__init__(parent)
        self.root = Path(root)
        self.options = dict(options)
        voice_paths(options)
        self.window_lock = QtCore.QLockFile(str(self.root / "generator.lock"))
        self.window_lock.setStaleLockTime(0)
        if not self.window_lock.tryLock(0):
            raise ValueError("台词生成工具已打开，请使用已有窗口")
        self.projects_root = self.root / "generation-projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.project_lock = None
        self.project = None
        self.runtime = None
        self.worker = None
        self.preview_worker = None
        self.ocr_dialog = None
        self.playback_dialog = None
        self.voice_dialog = None
        self.manual_dialog = None
        self.closing = False
        self.rendering = False
        self.editor_row = None
        self.idle_controls = []
        self.setWindowTitle("异环 · 台词生成工具")
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.resize(1180, 850)
        self.setMinimumSize(950, 740)
        self.setStyleSheet("""
            QDialog { background:#f5f7fa; }
            QPushButton { padding:6px 10px; }
            QLineEdit, QPlainTextEdit, QComboBox { padding:4px; background:white; }
            QTableWidget { background:white; gridline-color:#e6eaf0; }
            QHeaderView::section { background:#e9eef4; padding:7px; border:0; }
            QPushButton#primary { background:#167d77; color:white; font-weight:600; border-radius:4px; }
            QPushButton#primary:disabled { background:#adc4c2; }
        """)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        title = QtWidgets.QLabel("台词生成工具")
        title.setStyleSheet("font-size:24px; font-weight:600; color:#172c43;")
        outer.addWidget(title)
        outer.addWidget(QtWidgets.QLabel("识别 / 导入台词 → 核对文字和声线 → 批量生成 → 试听 / 游戏内自动播放"))
        connected_tools = QtWidgets.QHBoxLayout()
        self.ocr_button = self.button(connected_tools, "识别游戏台词…", self.open_ocr)
        self.auto_play_button = self.button(connected_tools, "自动识别并播放…", self.open_playback)
        self.add_voice_button = self.button(connected_tools, "添加角色声线…", self.add_voice)
        self.button(connected_tools, "系统角色声线设置…", self.system_voice_settings)
        tools_note = QtWidgets.QLabel("采集台词或播放本项目已生成的音频；声线与原配音工具共用。")
        tools_note.setWordWrap(True)
        connected_tools.addWidget(tools_note, 1)
        outer.addLayout(connected_tools)
        toolbar = QtWidgets.QHBoxLayout()
        self.button(toolbar, "导入最近记录", self.import_latest)
        self.button(toolbar, "选择识别记录…", self.choose_import)
        self.manual_button = self.button(toolbar, "手动添加台词…", self.add_dialogue)
        self.button(toolbar, "粘贴台词…", self.paste_text)
        toolbar.addStretch()
        self.button(toolbar, "新建项目", self.new_project)
        self.button(toolbar, "打开项目…", self.open_project)
        self.button(toolbar, "打开输出文件夹", self.open_output, idle=False)
        outer.addLayout(toolbar)
        modes = QtWidgets.QHBoxLayout()
        modes.addWidget(QtWidgets.QLabel("生成方式"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("系统中文声（较快）", "system")
        self.mode.addItem("已有角色声线（本地模型）", "character")
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.idle_controls.append(self.mode)
        modes.addWidget(self.mode)
        modes.addSpacing(15)
        modes.addWidget(QtWidgets.QLabel("将勾选项的声线设为"))
        self.batch_voice = QtWidgets.QComboBox()
        modes.addWidget(self.batch_voice, 1)
        self.idle_controls.append(self.batch_voice)
        self.button(modes, "应用声线", self.apply_voice)
        outer.addLayout(modes)
        self.mode_note = QtWidgets.QLabel()
        self.mode_note.setWordWrap(True)
        self.mode_note.setStyleSheet("color:#52647a;")
        outer.addWidget(self.mode_note)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["生成", "角色", "台词", "文字核对", "指定声线", "生成结果", "实际声线"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(43)
        self.table.setMinimumHeight(190)
        header = self.table.horizontalHeader()
        for index, width in ((0, 46), (1, 100), (3, 105), (4, 125), (5, 140), (6, 160)):
            header.resizeSection(index, width)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.itemChanged.connect(self.check_changed)
        self.table.currentCellChanged.connect(self.current_changed)
        outer.addWidget(self.table, 1)
        selection = QtWidgets.QHBoxLayout()
        self.button(selection, "全选", lambda: self.check_all(True))
        self.button(selection, "全不选", lambda: self.check_all(False))
        self.button(selection, "确认勾选项已核对", self.confirm_checked)
        self.delete_current_button = self.button(selection, "删除当前行", self.delete_current)
        self.delete_checked_button = self.button(selection, "删除勾选项", self.delete_checked)
        self.delete_checked_button.setToolTip("删除左侧打勾的所有台词；可用撤销删除恢复。")
        self.undo_button = self.button(selection, "撤销删除", self.undo_delete)
        selection.addStretch()
        self.count_label = QtWidgets.QLabel()
        selection.addWidget(self.count_label)
        outer.addLayout(selection)
        self.editor = QtWidgets.QGroupBox("当前行 · 修改后自动保存在项目中")
        box = QtWidgets.QVBoxLayout(self.editor)
        fields = QtWidgets.QHBoxLayout()
        fields.addWidget(QtWidgets.QLabel("角色"))
        self.speaker = QtWidgets.QLineEdit()
        self.speaker.setPlaceholderText("未指定角色将使用默认声线")
        fields.addWidget(self.speaker, 1)
        fields.addWidget(QtWidgets.QLabel("声线"))
        self.voice = QtWidgets.QComboBox()
        fields.addWidget(self.voice, 1)
        self.button(fields, "查看原图", self.show_source)
        box.addLayout(fields)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlaceholderText("选择一行，在这里修正台词。")
        self.text.setFixedHeight(76)
        box.addWidget(self.text)
        confirmation = QtWidgets.QHBoxLayout()
        self.confirmed = QtWidgets.QCheckBox("我已核对当前台词和角色")
        confirmation.addWidget(self.confirmed)
        confirmation.addStretch()
        self.button(confirmation, "保存当前修改", self.save_editor)
        box.addLayout(confirmation)
        self.evidence = QtWidgets.QLabel()
        self.evidence.setWordWrap(True)
        self.evidence.setStyleSheet("color:#5f6f82;")
        box.addWidget(self.evidence)
        self.speaker.textEdited.connect(self.unconfirm)
        self.text.textChanged.connect(self.unconfirm)
        outer.addWidget(self.editor)
        actions = QtWidgets.QHBoxLayout()
        self.generate = self.button(actions, "生成 / 继续勾选项", self.start_generation)
        self.generate.setObjectName("primary")
        self.button(actions, "重试失败项", lambda: self.start_generation(failed_only=True))
        self.button(actions, "重新生成当前行", lambda: self.start_generation(force=True))
        self.stop_generation = self.button(actions, "停止生成", self.cancel_generation, idle=False)
        self.stop_generation.setEnabled(False)
        actions.addStretch()
        self.preview = self.button(actions, "试听当前行", self.start_preview)
        self.stop_preview = self.button(actions, "停止试听", self.cancel_preview, idle=False)
        self.stop_preview.setEnabled(False)
        outer.addLayout(actions)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setMaximumHeight(17)
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel("生成时不会播放声音。生成完成后，选择一行即可试听。")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(28)
        outer.addWidget(self.status)
        self.project_label = QtWidgets.QLabel()
        self.project_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.project_label.setStyleSheet("color:#67798e; font-size:11px;")
        outer.addWidget(self.project_label)
        self.edit_timer = QtCore.QTimer(self)
        self.edit_timer.setSingleShot(True)
        self.edit_timer.timeout.connect(self.save_editor)
        for signal in (self.speaker.textEdited, self.text.textChanged, self.voice.currentIndexChanged, self.confirmed.toggled):
            signal.connect(self.queue_save)
        last = self.projects_root / "last-project.txt"
        if project_path is None and last.is_file():
            candidate = Path(last.read_text(encoding="utf-8").strip())
            if (candidate / "project.json").is_file():
                project_path = candidate
        if project_path:
            self.use_project(GenerationProject.load(project_path))
        else:
            self.use_project(GenerationProject.create(self.projects_root, self.options.get("mode", "system")))

    def button(self, layout, label, callback, *, idle=True):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(lambda _checked=False: callback())
        layout.addWidget(button)
        if idle:
            self.idle_controls.append(button)
        return button

    def busy(self):
        return bool((self.worker and self.worker.isRunning()) or
                    (self.preview_worker and self.preview_worker.isRunning()))

    def use_project(self, project):
        self.save_editor()
        if self.project and self.project.directory == project.directory:
            return
        lock = QtCore.QLockFile(str(project.directory / "project.lock"))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            raise ValueError("该项目正在另一个窗口中使用")
        if self.project_lock:
            self.project_lock.unlock()
        self.project_lock, self.project = lock, project
        self.editor_row = None
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(project.mode))
        self.mode.blockSignals(False)
        self.refresh_voices()
        self.project_label.setText("自动保存位置：" + str(project.directory))
        from dialogue_generation import atomic_text
        atomic_text(self.projects_root / "last-project.txt", str(project.directory))
        for row in project.rows:
            if row.status == "done" and not project.audio_path(row):
                row.status, row.error, row.audio = "pending", "原音频缺失或损坏，请继续生成", {}
        project.save()
        self.render()
        if project.rows:
            self.table.setCurrentCell(0, 2)

    def refresh_voices(self):
        items = [("按角色自动选择", "auto")]
        try:
            self.runtime = VoiceRuntime(dict(self.options, mode=self.project.mode, fallback_to_system=False))
            if self.project.mode == "system":
                items.extend((label, "system:" + identity) for identity, (label, *_) in PRESETS.items())
                note = "系统中文声 · 沿用已有角色绑定；年龄声线是调节音高和语速的试制预设。"
            elif self.runtime.system_mode:
                raise ValueError("共享配音设置未选择角色模型，请先在原工具的“参考音频与音量”中设置。")
            else:
                items.extend((name, "character:" + name) for name in self.runtime.profiles)
                note = "角色声线 · 沿用已有参考音频；首次生成较慢，重复台词会检查并复用缓存。"
            self.mode_note.setText(note + " 切换生成方式会重置本项目的声线选择。")
        except Exception as exc:
            self.runtime = None
            self.mode_note.setText("声线尚未就绪：" + str(exc))
        for combo in (self.voice, self.batch_voice):
            combo.blockSignals(True)
            combo.clear()
            for label, identity in items:
                combo.addItem(label, identity)
            combo.blockSignals(False)

    def change_mode(self):
        if not self.project or self.busy():
            return
        self.save_editor()
        self.project.set_mode(self.mode.currentData())
        self.refresh_voices()
        self.project.save()
        self.render()
        self.load_editor(self.table.currentRow())

    def render(self):
        self.rendering = True
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.project.rows))
        for index, row in enumerate(self.project.rows):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable)
            check.setCheckState(QtCore.Qt.Checked if row.selected else QtCore.Qt.Unchecked)
            self.table.setItem(index, 0, check)
            reason = row.review_reason()
            review = "待核对" if reason else ("人工已确认" if row.confirmed else "OCR 已复核")
            chosen = self.voice.itemText(self.voice.findData(row.voice)) or "声线不可用"
            actual = row.audio.get("voice", "")
            default = row.audio.get("default_voice", False)
            if not actual and self.runtime:
                try:
                    profile, default = select_voice(self.runtime, row)
                    actual = profile.speaker
                except ValueError:
                    actual = "请重新选择声线"
            if default:
                actual += "（默认）"
            state = "无需配音" if row.punctuation_only else STATE_NAMES.get(row.status, row.status)
            if row.status == "done" and row.audio and not row.punctuation_only:
                state += f" · {row.audio['duration_seconds']:.1f}秒"
            values = [row.speaker or "未指定", row.text, review, chosen, state, actual]
            for column, value in enumerate(values, 1):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(row.error if column == 5 and row.error else reason if column == 3 and reason else value)
                if column == 3 and reason:
                    item.setForeground(QtGui.QColor("#a65c13"))
                if column == 5 and row.status == "failed":
                    item.setForeground(QtGui.QColor("#b33e35"))
                self.table.setItem(index, column, item)
        self.table.blockSignals(False)
        self.rendering = False
        ready = sum(row.selected and not row.punctuation_only and not row.review_reason() for row in self.project.rows)
        held = sum(row.selected and not row.punctuation_only and bool(row.review_reason()) for row in self.project.rows)
        self.count_label.setText(f"共 {len(self.project.rows)} 条 · 可生成 {ready} 条 · 待核对 {held} 条")
        self.editor.setEnabled(not self.busy() and self.editor_row is not None)
        self.undo_button.setEnabled(not self.busy() and bool(self.project.deleted_batches))

    def current_changed(self, row, _column, _previous, _previous_column):
        if self.rendering:
            return
        self.save_editor()
        self.load_editor(row)

    def load_editor(self, index):
        self.rendering = True
        self.editor_row = self.project.rows[index] if 0 <= index < len(self.project.rows) else None
        row = self.editor_row
        self.speaker.setText(row.speaker if row else "")
        self.text.setPlainText(row.text if row else "")
        voice_index = self.voice.findData(row.voice) if row else 0
        if row and voice_index < 0:
            self.voice.addItem("已失效：" + row.voice, row.voice)
            voice_index = self.voice.count()-1
        self.voice.setCurrentIndex(max(0, voice_index))
        self.confirmed.setChecked(bool(row and row.confirmed))
        evidence = ""
        if row:
            evidence = ("仅含标点，无需配音。" if row.punctuation_only else "")
            evidence += row.review_reason() or "文字已核对；仍可按原图检查。"
            review = row.evidence.get("review_text")
            if review and review != row.text:
                evidence += " 复核候选：" + str(review)[:180]
        self.evidence.setText(evidence)
        self.editor.setEnabled(row is not None and not self.busy())
        self.rendering = False

    def unconfirm(self, *_):
        if not self.rendering:
            self.confirmed.setChecked(False)

    def queue_save(self, *_):
        if not self.rendering and self.editor_row and not self.busy():
            self.edit_timer.start(450)

    def save_editor(self):
        if not self.project or self.rendering or self.busy() or not self.editor_row:
            return True
        self.edit_timer.stop()
        row = self.editor_row
        row.edit(text=self.text.toPlainText().strip(), speaker=self.speaker.text().strip(), voice=self.voice.currentData())
        row.set_confirmed(self.confirmed.isChecked())
        try:
            self.project.save()
            self.render()
            return True
        except Exception as exc:
            self.status.setText("项目未能保存：" + str(exc))
            return False

    def refresh_selection(self, index):
        # A queued autosave must never write the old editor into another row.
        self.edit_timer.stop()
        self.editor_row = None
        self.render()
        index = min(index, len(self.project.rows)-1)
        if index >= 0:
            self.table.setCurrentCell(index, 2)
            self.table.scrollToItem(self.table.item(index, 2))
        self.load_editor(index)

    def add_dialogue(self):
        if self.busy() or not self.save_editor():
            return
        voices = [(self.voice.itemText(i), self.voice.itemData(i)) for i in range(self.voice.count())
                  if not self.voice.itemText(i).startswith("已失效：")]
        speakers = [row.speaker for row in self.project.rows]
        speakers += list(self.runtime.profiles) if self.runtime else []
        try:
            self.manual_dialog = ManualDialogueDialog(voices, speakers, self)
            self.manual_dialog.finished.connect(self.close_after_child)
            if self.manual_dialog.exec() == QtWidgets.QDialog.Accepted:
                row = self.project.add_dialogue(self.manual_dialog.text.toPlainText(),
                                                self.manual_dialog.speaker.currentText(),
                                                self.manual_dialog.voice.currentData())
                self.refresh_selection(len(self.project.rows)-1)
                self.project.save()
                self.status.setText("已添加台词并保存。" + ("仅含标点，无需配音。" if row.punctuation_only else "可直接生成或继续编辑。"))
        except Exception as exc:
            self.status.setText("手动台词未能保存：" + str(exc))
        finally:
            self.manual_dialog = None
            self.close_after_child()

    def delete_current(self):
        row = self.editor_row
        self.delete_rows([row.id] if row else [])

    def delete_checked(self):
        self.delete_rows([row.id for row in self.project.rows if row.selected])

    def delete_rows(self, ids):
        if self.busy() or not self.save_editor():
            return
        if not ids:
            self.status.setText("请先选择一行，或勾选要删除的台词。")
            return
        try:
            index = max(0, self.table.currentRow())
            count = self.project.delete_rows(ids)
            self.refresh_selection(index)
            self.project.save()
            self.status.setText(f"已删除 {count} 条台词，可点击“撤销删除”恢复。原始记录及音频文件保留。")
        except Exception as exc:
            self.status.setText("删除修改未能保存，可撤销后重试：" + str(exc))

    def undo_delete(self):
        if self.busy() or not self.save_editor():
            return
        try:
            identities = self.project.undo_delete()
            if not identities:
                self.status.setText("没有可撤销的删除操作。")
                return
            index = next(i for i, row in enumerate(self.project.rows) if row.id == identities[0])
            self.refresh_selection(index)
            self.project.save()
            self.status.setText(f"已恢复 {len(identities)} 条台词。可连续撤销之前的删除操作。")
        except Exception as exc:
            self.status.setText("恢复修改未能保存：" + str(exc))

    def check_changed(self, item):
        if self.rendering or item.column() != 0 or self.busy():
            return
        self.project.rows[item.row()].selected = item.checkState() == QtCore.Qt.Checked
        self.save_editor()
        self.project.save()
        self.render()

    def check_all(self, checked):
        self.save_editor()
        for row in self.project.rows:
            row.selected = checked
        self.project.save()
        self.render()

    def confirm_checked(self):
        self.save_editor()
        for row in self.project.rows:
            if row.selected and row.text.strip():
                row.set_confirmed(True)
        self.project.save()
        self.render()
        self.load_editor(self.table.currentRow())
        self.status.setText("勾选台词已标记为人工确认。生成将使用表格中的文字和角色。")

    def apply_voice(self):
        self.save_editor()
        for row in self.project.rows:
            if row.selected:
                row.edit(voice=self.batch_voice.currentData())
        self.project.save()
        self.render()
        self.load_editor(self.table.currentRow())

    def import_path(self, path):
        self.save_editor()
        try:
            result = self.project.import_records(path)
            self.project.save()
            self.render()
            if self.table.currentRow() < 0 and self.project.rows:
                self.table.setCurrentCell(0, 2)
            self.load_editor(self.table.currentRow())
            message = f"已导入：新增 {result['added']} 条，更新 {result['updated']} 条，重复跳过 {result['unchanged']} 条。"
            if result.get("deleted"):
                message += f" {result['deleted']} 条已删除台词仍保留在删除记录中。"
            if result["warnings"]:
                message += f" 有 {len(result['warnings'])} 行格式异常：" + result["warnings"][0]
            self.status.setText(message)
        except Exception as exc:
            self.status.setText("未能导入：" + str(exc))

    def import_latest(self):
        path = latest_record(self.root / "records")
        if path:
            self.import_path(path)
        else:
            self.status.setText("还没有识别记录。可先在 OCR 工具中记录台词，或直接粘贴台词。")

    def choose_import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 dialogue.jsonl 或 revisions.jsonl",
                                                       str(self.root / "records"), "OCR 记录 (*.jsonl)")
        if path:
            self.import_path(path)

    def paste_text(self):
        if self.busy():
            return
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "粘贴台词", "每行一条，可写“角色：台词”。请先核对，添加后视为人工确认。")
        if ok and text.strip():
            self.save_editor()
            count = self.project.add_text(text)
            self.project.save()
            self.render()
            self.table.setCurrentCell(len(self.project.rows)-count, 2)
            self.status.setText(f"已添加 {count} 条台词。可修改角色并指定声线。")

    def new_project(self):
        try:
            self.use_project(GenerationProject.create(self.projects_root, self.mode.currentData()))
            self.load_editor(-1)
            self.status.setText("已新建项目。之前的项目及音频仍保存在输出文件夹中。")
        except Exception as exc:
            self.status.setText("新建失败：" + str(exc))

    def open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开生成项目 project.json", str(self.projects_root), "生成项目 (project.json)")
        if path:
            try:
                self.use_project(GenerationProject.load(path))
                self.status.setText("已载入项目。继续生成时会核对当前声线和模型，自动跳过已完成且仍有效的音频。")
            except Exception as exc:
                self.status.setText("打开失败：" + str(exc))

    def open_output(self):
        if self.project:
            os.startfile(self.project.directory)

    def close_after_child(self, *_):
        if self.closing:
            QtCore.QTimer.singleShot(0, self.close)

    def open_ocr(self):
        if self.busy():
            return
        self.save_editor()
        try:
            from generator_ocr import OcrCollectionDialog
            self.ocr_dialog = OcrCollectionDialog(self.root, self)
            self.ocr_dialog.collected.connect(self.receive_ocr)
            self.ocr_dialog.finished.connect(self.close_after_child)
            self.ocr_dialog.exec()
            self.render()
            self.load_editor(self.table.currentRow())
            self.status.setText(f"采集窗口已关闭，当前项目有 {len(self.project.rows)} 条台词。核对后即可批量生成。")
        except Exception as exc:
            self.status.setText("识别窗口未打开：" + str(exc))
        finally:
            self.ocr_dialog = None
            self.close_after_child()

    def open_playback(self):
        if self.busy():
            return
        self.save_editor()
        try:
            from generator_playback import PreparedPlaybackDialog
            self.playback_dialog = PreparedPlaybackDialog(self.root, self.project, self.options, self)
            self.playback_dialog.finished.connect(self.close_after_child)
            self.playback_dialog.exec()
            self.status.setText('自动播放窗口已关闭。缺少音频的台词可先采集、核对并生成，再继续自动播放。')
        except Exception as exc:
            self.status.setText('自动播放未打开：'+str(exc))
        finally:
            self.playback_dialog = None
            self.close_after_child()

    def receive_ocr(self, event):
        try:
            path = Path(event.get("source") or event.get("final_file", "")).resolve()
            if not path.is_relative_to((self.root / "records").resolve()):
                raise ValueError("识别记录不属于当前工具的采集目录")
            if "record" in event:
                self.project.receive_record(event["record"], path)
            elif "final_file" in event:
                self.project.import_records(path)
            self.project.save()
            self.render()
        except Exception as exc:
            self.status.setText("识别记录未能加入项目：" + str(exc))
            if self.ocr_dialog:
                self.ocr_dialog.stop_work()
                self.ocr_dialog.status.setText("项目保存失败，已停止采集。原始记录仍保留：" + str(exc))

    def add_voice(self):
        if self.busy():
            return
        self.save_editor()
        try:
            from generator_voices import AddVoiceDialog
            self.voice_dialog = AddVoiceDialog(self.options, self)
            self.voice_dialog.finished.connect(self.close_after_child)
            self.voice_dialog.exec()
            if self.voice_dialog.result_profile:
                name = self.voice_dialog.result_profile["speaker"]
                self.refresh_voices()
                if self.runtime:
                    for row in self.project.rows:
                        if row.voice == "auto" and row.audio:
                            profile, _ = select_voice(self.runtime, row)
                            if profile.speaker != row.audio.get("voice"):
                                row.audio, row.status, row.error = {}, "pending", ""
                self.project.save()
                self.render()
                self.load_editor(self.table.currentRow())
                self.status.setText(f"已添加“{name}”。在“已有角色声线”模式中选择它，或按角色名和别名自动匹配。")
        except Exception as exc:
            self.status.setText("添加声线未完成：" + str(exc))
        finally:
            self.voice_dialog = None
            self.close_after_child()

    def system_voice_settings(self):
        if self.busy():
            return
        self.save_editor()
        try:
            from system_voice_dialog import SystemVoiceDialog
            from dialogue_generation import atomic_text
            self.voice_dialog = SystemVoiceDialog(self.options, self)
            self.voice_dialog.finished.connect(self.close_after_child)
            if self.voice_dialog.exec() == QtWidgets.QDialog.Accepted:
                config_path = self.root / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8-sig"))
                updates = {}
                for key in ("system_default_voice", "system_character_voices"):
                    value = self.voice_dialog.result_options[key]
                    config.setdefault("dubbing", {})[key] = value
                    updates[key] = value
                atomic_text(config_path, json.dumps(config, ensure_ascii=False, indent=2))
                self.options.update(updates)
                # Auto-routed exports may now use another voice; preserve files but refresh assignments.
                if self.project.mode == "system":
                    for row in self.project.rows:
                        if row.voice == "auto":
                            row.audio, row.status, row.error = {}, "pending", ""
                self.project.save()
                self.refresh_voices()
                self.render()
                self.load_editor(self.table.currentRow())
                self.status.setText("系统角色声线已保存，自动选择声线的台词可继续生成。")
        except Exception as exc:
            self.status.setText("系统声线设置未完成：" + str(exc))
        finally:
            self.voice_dialog = None
            self.close_after_child()

    def show_source(self):
        row = self.editor_row
        if not row or row.source == "manual":
            self.status.setText("手动添加的台词没有 OCR 原图。")
            return
        folder = Path(row.source).resolve()
        path = (folder / row.evidence.get("body_image", "")).resolve()
        if not path.is_relative_to(folder) or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            self.status.setText("当前记录没有可用的正文原图。")
            return
        picture = QtGui.QPixmap(str(path))
        if picture.isNull():
            self.status.setText("原图未找到。请保留识别记录目录中的 images 文件夹。")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("OCR 正文原图 · " + row.source_id)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel()
        label.setPixmap(picture.scaled(1100, 500, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        layout.addWidget(label)
        text = QtWidgets.QLabel(row.text)
        text.setWordWrap(True)
        layout.addWidget(text)
        dialog.exec()

    def set_busy(self, busy):
        for control in self.idle_controls:
            control.setEnabled(not busy)
        self.table.setEnabled(not busy)
        self.editor.setEnabled(not busy and self.editor_row is not None)

    def start_generation(self, *, failed_only=False, force=False):
        if self.busy():
            return
        self.save_editor()
        if not self.runtime:
            self.refresh_voices()
            if not self.runtime:
                self.status.setText("声线尚未就绪，请按生成方式下方提示检查设置。")
                return
        rows = ([self.editor_row] if force and self.editor_row else [] if force else
                [r for r in self.project.rows if r.selected and (not failed_only or r.status == "failed")])
        if not any(not row.punctuation_only and not row.review_reason() for row in rows):
            self.status.setText("没有可生成的台词。请勾选并核对文字；纯省略号等标点无需配音。")
            return
        self.worker = GenerationWorker(self.project, self.options, [row.id for row in rows], force, self)
        self.worker.update.connect(self.generation_update)
        self.worker.result.connect(self.generation_result)
        self.worker.error.connect(self.status.setText)
        self.worker.finished.connect(self.worker_finished)
        self.progress.setRange(0, len(rows))
        self.progress.setValue(0)
        self.set_busy(True)
        self.stop_generation.setEnabled(True)
        self.status.setText("正在准备批量生成，生成期间不会播放声音…")
        self.worker.start()

    def generation_update(self, event):
        if not self.closing:
            self.status.setText(event["message"])
        if "total" in event:
            self.progress.setRange(0, event["total"])
            self.progress.setValue(event["current"])
        self.render()

    def generation_result(self, result):
        self.status.setText(f"本次完成 {result['completed']} 条（缓存 {result['cached']} 条），保留已完成 {result['resumed']} 条；"
                            f"失败 {result['failed']} 条，待核对 {result['held']} 条，无需配音 {result.get('silent',0)} 条。" +
                            ("已停止，剩余台词可继续生成。" if self.worker.runner.stop_event.is_set() else "可选择一行试听。"))

    def cancel_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.stop_generation.setEnabled(False)
            self.status.setText("正在停止当前生成；已完成音频会保留…")

    def worker_finished(self):
        self.stop_generation.setEnabled(False)
        self.stop_preview.setEnabled(False)
        self.set_busy(False)
        self.render()
        if self.closing:
            self.close()

    def start_preview(self):
        if self.busy():
            return
        self.save_editor()
        row = self.editor_row
        if row and row.punctuation_only:
            self.status.setText("当前台词仅含标点，无需生成或试听配音。")
            return
        path = self.project.audio_path(row) if row else None
        if not path:
            self.status.setText("当前行还没有完整音频，或文字/声线已修改，请先生成。")
            return
        try:
            # Playback of saved WAV files does not need a running model service.
            runtime = VoiceRuntime(dict(self.options, mode="system"))
            self.preview_worker = AudioPreviewWorker(path, runtime.config, self)
            self.preview_worker.message.connect(self.status.setText)
            self.preview_worker.finished.connect(self.worker_finished)
            self.set_busy(True)
            self.stop_preview.setEnabled(True)
            self.status.setText("正在试听：" + (row.speaker or "未指定角色") + " · " + row.audio["voice"])
            self.preview_worker.start()
        except Exception as exc:
            self.status.setText("试听失败：" + str(exc))

    def cancel_preview(self):
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.cancel()
            self.stop_preview.setEnabled(False)
            self.status.setText("正在停止试听…")

    def reject(self):
        self.close()

    def changeEvent(self, event):
        super().changeEvent(event)
        if (event.type() == QtCore.QEvent.WindowStateChange
                and event.oldState() & QtCore.Qt.WindowMinimized
                and not self.isMinimized()):
            for dialog in (self.ocr_dialog, self.playback_dialog):
                if dialog and dialog.isVisible() and dialog.isMinimized():
                    dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
                    dialog.raise_()
                    dialog.activateWindow()

    def closeEvent(self, event):
        for dialog in (self.ocr_dialog, self.playback_dialog, self.voice_dialog, self.manual_dialog):
            if dialog and dialog.isVisible():
                self.closing = True
                dialog.close()
                event.ignore()
                return
        if self.busy():
            self.closing = True
            self.cancel_generation()
            self.cancel_preview()
            event.ignore()
            return
        self.save_editor()
        if self.project_lock:
            self.project_lock.unlock()
        self.window_lock.unlock()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="台词批量生成与试听")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--smoke-ms", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    application.setStyle("Fusion")
    font = application.font()
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"])
    font.setPointSize(10)
    application.setFont(font)
    try:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
        window = GeneratorWindow(config.get("dubbing", {}), project_path=args.project)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(None, "台词生成工具未打开", str(exc))
        return 1
    window.show()
    if args.smoke_ms:
        QtCore.QTimer.singleShot(args.smoke_ms, window.close)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
