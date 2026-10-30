from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..audio import AudioPlayer
from ..capture import create_capture_source
from ..capture.windows_list import list_window_titles
from ..config import AppConfig, AppPaths, CLONING_BACKENDS, endpoint_for_backend, save_config
from ..diagnostics import collect_diagnostics
from ..models import PipelineSnapshot, Rect, VoiceProfile
from ..ocr import RapidOcrEngine
from ..pipeline import DubPipeline, crop_image
from ..text_processing import merge_ocr_lines
from ..voice_profiles import VoiceProfileStore
from .bridge import PipelineBridge
from .hotkeys import HotkeyManager
from .region_selector import RegionSelector
from .voice_profile_dialog import VoiceProfileDialog
from .speech_preview import SpeechPreview


class StatusOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setMinimumWidth(320)
        self.text.setStyleSheet(
            "QLabel { color: white; background: rgba(16, 22, 30, 210); "
            "border: 1px solid rgba(120, 180, 220, 180); border-radius: 6px; padding: 10px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text)

    def update_status(self, snapshot: PipelineSnapshot) -> None:
        state = "已停止" if not snapshot.running else ("已暂停" if snapshot.paused else "监听中")
        latency = (
            f"{snapshot.last_latency_seconds:.2f}s"
            if snapshot.last_latency_seconds is not None
            else "--"
        )
        self.text.setText(
            f"{state}  队列 {snapshot.queue_depth}  延迟 {latency}\n"
            f"{snapshot.last_speaker or '旁白'}：{snapshot.last_text or snapshot.message}"
        )
        screen = QApplication.primaryScreen()
        if screen:
            area = screen.availableGeometry()
            self.adjustSize()
            self.move(area.right() - self.width() - 24, area.top() + 24)


class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths, config: AppConfig, voice_store: VoiceProfileStore) -> None:
        super().__init__()
        self.paths = paths
        self.config = config
        self.voice_store = voice_store
        self.pipeline: DubPipeline | None = None
        self._pipeline_stopping = False
        self._pipeline_stop_timer = QTimer(self)
        self._pipeline_stop_timer.setInterval(100)
        self._pipeline_stop_timer.timeout.connect(self._finish_pipeline_stop)
        self.hotkeys: HotkeyManager | None = None
        self.preview: SpeechPreview | None = None
        self._preview_cancel_requested = False
        self.bridge = PipelineBridge()
        self.bridge.snapshot.connect(self._update_snapshot)
        self.bridge.error.connect(self._show_error_status)
        self.bridge.pause_requested.connect(self._toggle_pause)
        self.bridge.replay_requested.connect(self._replay)
        self.bridge.jump_requested.connect(self._jump_latest)
        self.overlay = StatusOverlay()
        self.setWindowTitle("异环 AI 配音")
        self.resize(1080, 760)
        self._build_ui()
        self._load_controls()
        self._refresh_profiles()
        self._refresh_devices()
        self._refresh_diagnostics()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_realtime_tab(), "实时配音")
        tabs.addTab(self._build_capture_tab(), "捕获与识别")
        tabs.addTab(self._build_voice_tab(), "声线与音频")
        tabs.addTab(self._build_diagnostics_tab(), "诊断")
        self.setCentralWidget(tabs)

    def _build_realtime_tab(self) -> QWidget:
        page = QWidget()
        self.status_label = QLabel("已停止")
        self.status_label.setStyleSheet("font-size: 22px; font-weight: 600;")
        self.message_label = QLabel("尚未启动")
        self.message_label.setWordWrap(True)
        self.last_text = QPlainTextEdit()
        self.last_text.setReadOnly(True)
        self.last_text.setMaximumHeight(150)
        self.stats_label = QLabel("延迟 -- | 队列 0 | 跳过 0")
        self.preview_text = QPlainTextEdit("你好，欢迎来到异环。")
        self.preview_text.setMaximumHeight(85)
        self.preview_text.setPlaceholderText("输入一句话，不开游戏也可以试听")
        self.preview_button = QPushButton("试听这句话")
        self.preview_button.clicked.connect(lambda: self._preview_speech())
        self.preview_regenerate_button = QPushButton("重新生成并试听")
        self.preview_regenerate_button.setToolTip("重新生成当前台词的所有分段；生成失败的分段保留原音频。")
        self.preview_regenerate_button.clicked.connect(lambda: self._preview_speech(force=True))
        self.preview_stop_button = QPushButton("停止试听")
        self.preview_stop_button.setEnabled(False)
        self.preview_stop_button.clicked.connect(self._cancel_preview)
        self.preview_status = QLabel("先在“声线与音频”选择模式和角色。普通试听会复用已生成的音频；想重做时点击“重新生成并试听”。角色克隆可能需要几十秒。")
        self.preview_status.setWordWrap(True)
        self.overlay_checkbox = QCheckBox("显示悬浮状态")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.toggled.connect(self._toggle_overlay)

        style = QApplication.style()
        self.start_button = QPushButton(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "启动")
        self.stop_button = QPushButton(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "停止")
        self.pause_button = QPushButton(style.standardIcon(QStyle.StandardPixmap.SP_MediaPause), "暂停")
        self.replay_button = QPushButton(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "重播")
        self.jump_button = QPushButton(style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward), "跳到最新")
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.replay_button.clicked.connect(self._replay)
        self.jump_button.clicked.connect(self._jump_latest)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)

        controls = QHBoxLayout()
        for button in (
            self.start_button,
            self.stop_button,
            self.pause_button,
            self.replay_button,
            self.jump_button,
        ):
            controls.addWidget(button)
        controls.addStretch(1)
        controls.addWidget(self.overlay_checkbox)

        layout = QVBoxLayout(page)
        layout.addWidget(self.status_label)
        layout.addWidget(self.message_label)
        layout.addWidget(self.last_text)
        layout.addWidget(self.stats_label)
        layout.addLayout(controls)
        layout.addWidget(QLabel("文字试听"))
        layout.addWidget(self.preview_text)
        preview_controls = QHBoxLayout()
        preview_controls.addWidget(self.preview_button)
        preview_controls.addWidget(self.preview_regenerate_button)
        preview_controls.addWidget(self.preview_stop_button)
        layout.addLayout(preview_controls)
        layout.addWidget(self.preview_status)
        layout.addStretch(1)
        return page

    def _build_capture_tab(self) -> QWidget:
        page = QWidget()
        self.source_combo = QComboBox()
        self.source_combo.addItem("直接捕获游戏窗口", "direct")
        self.source_combo.addItem("OBS 虚拟摄像头", "obs")
        self.window_title = QLineEdit()
        self.window_title.setPlaceholderText("游戏窗口标题")
        self.choose_window_button = QPushButton("选择窗口")
        self.choose_window_button.clicked.connect(self._choose_window)
        window_row = QWidget()
        window_layout = QHBoxLayout(window_row)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(self.window_title, 1)
        window_layout.addWidget(self.choose_window_button)
        self.virtual_camera_index = QSpinBox()
        self.virtual_camera_index.setRange(0, 9)
        self.sample_fps = QDoubleSpinBox()
        self.sample_fps.setRange(0.5, 15.0)
        self.sample_fps.setSingleStep(0.5)
        self.stability_frames = QSpinBox()
        self.stability_frames.setRange(1, 10)
        self.dedupe_history = QSpinBox()
        self.dedupe_history.setRange(1, 30)
        self.match_threshold = QDoubleSpinBox()
        self.match_threshold.setRange(0.0, 1.0)
        self.match_threshold.setSingleStep(0.01)
        self.resolution_width = QSpinBox()
        self.resolution_width.setRange(320, 7680)
        self.resolution_height = QSpinBox()
        self.resolution_height.setRange(240, 4320)

        self.dialogue_rect_label = QLabel("未配置")
        self.speaker_rect_label = QLabel("未配置（按旁白朗读）")
        select_dialogue = QPushButton("框选对白区域")
        select_speaker = QPushButton("框选角色名区域")
        select_dialogue.clicked.connect(lambda: self._select_region("dialogue"))
        select_speaker.clicked.connect(lambda: self._select_region("speaker"))
        test_ocr = QPushButton("单帧 OCR 测试")
        test_ocr.clicked.connect(self._test_ocr)

        form = QFormLayout()
        form.addRow("画面来源", self.source_combo)
        form.addRow("游戏窗口", window_row)
        form.addRow("虚拟摄像头编号", self.virtual_camera_index)
        form.addRow("采样帧率", self.sample_fps)
        form.addRow("稳定帧数", self.stability_frames)
        form.addRow("去重历史", self.dedupe_history)
        form.addRow("角色匹配阈值", self.match_threshold)
        form.addRow("基准分辨率", self._pair(self.resolution_width, self.resolution_height))
        form.addRow("对白区域", self._rect_row(self.dialogue_rect_label, select_dialogue))
        form.addRow("角色名区域", self._rect_row(self.speaker_rect_label, select_speaker))
        form.addRow("", test_ocr)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_voice_tab(self) -> QWidget:
        page = QWidget()
        self.profile_table = QTableWidget(0, 4)
        self.profile_table.setHorizontalHeaderLabels(["角色", "别名", "参考音频", "状态"])
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.profile_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.profile_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        add_profile = QPushButton("导入声线")
        delete_profile = QPushButton("删除声线")
        reload_profiles = QPushButton("重新载入")
        add_profile.clicked.connect(self._import_profile)
        delete_profile.clicked.connect(self._delete_profile)
        reload_profiles.clicked.connect(self._reload_profiles)
        profile_controls = QHBoxLayout()
        profile_controls.addWidget(add_profile)
        profile_controls.addWidget(delete_profile)
        profile_controls.addWidget(reload_profiles)
        profile_controls.addStretch(1)
        voice_group = QGroupBox("声线档案")
        voice_layout = QVBoxLayout(voice_group)
        voice_layout.addWidget(self.profile_table)
        voice_layout.addLayout(profile_controls)

        self.output_device = QComboBox()
        refresh_devices = QPushButton("刷新设备")
        refresh_devices.clicked.connect(self._refresh_devices)
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(self.output_device, 1)
        device_layout.addWidget(refresh_devices)
        self.output_volume = QDoubleSpinBox()
        self.output_volume.setRange(0.0, 1.5)
        self.output_volume.setSingleStep(0.05)
        self.pending_limit = QSpinBox()
        self.pending_limit.setRange(0, 10)
        self.first_segment_limit = QSpinBox()
        self.first_segment_limit.setRange(0, 300)
        self.first_segment_limit.setSpecialValueText("关闭首段短切")
        self.segment_limit = QSpinBox()
        self.segment_limit.setRange(4, 300)
        self.prefetch_window = QSpinBox()
        self.prefetch_window.setRange(1, 3)
        self.prefetch_window.setToolTip("仍然每次只生成一段；增加的是提前准备的段数。")
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("系统中文朗读（离线，无需模型）", "windows-sapi")
        self.backend_combo.addItem("GPT-SoVITS（角色音色，需要模型和参考音频）", "gpt-sovits")
        self.backend_combo.addItem("Qwen3-TTS（本地声音克隆，需要参考音频）", "qwen3-tts")
        self.backend_combo.currentIndexChanged.connect(self._update_backend_controls)
        self.fake_tts = QCheckBox("测试音模式（只发提示音，不朗读文字）")
        self.fake_tts.toggled.connect(self._update_backend_controls)
        self.tts_endpoint = QLineEdit()
        self.tts_token = QLineEdit()
        self.tts_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.ffmpeg_path = QLineEdit()
        self.ffprobe_path = QLineEdit()
        self.obs_executable = QLineEdit()
        audio_form = QFormLayout()
        audio_form.addRow("输出设备", device_row)
        audio_form.addRow("音量", self.output_volume)
        audio_form.addRow("待播队列上限", self.pending_limit)
        audio_form.addRow("角色配音首段字数", self.first_segment_limit)
        audio_form.addRow("后续分段字数", self.segment_limit)
        audio_form.addRow("角色预生成段数", self.prefetch_window)
        audio_form.addRow("朗读方式", self.backend_combo)
        audio_form.addRow("联调选项", self.fake_tts)
        audio_form.addRow("TTS 工作进程", self.tts_endpoint)
        audio_form.addRow("本地令牌", self.tts_token)
        audio_form.addRow("FFmpeg", self._file_row(self.ffmpeg_path, "选择 FFmpeg"))
        audio_form.addRow("FFprobe", self._file_row(self.ffprobe_path, "选择 FFprobe"))
        audio_form.addRow("OBS", self._file_row(self.obs_executable, "选择 OBS"))
        save_button = QPushButton("保存设置")
        save_button.clicked.connect(self._save_controls)
        audio_form.addRow("", save_button)
        audio_group = QGroupBox("音频与运行环境")
        audio_group.setLayout(audio_form)

        layout = QVBoxLayout(page)
        layout.addWidget(voice_group, 1)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(audio_group)
        settings_scroll.setMinimumHeight(260)
        layout.addWidget(settings_scroll, 2)
        return page

    def _build_diagnostics_tab(self) -> QWidget:
        page = QWidget()
        self.diagnostics_table = QTableWidget(0, 3)
        self.diagnostics_table.setHorizontalHeaderLabels(["项目", "状态", "详情"])
        self.diagnostics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.diagnostics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        refresh = QPushButton("重新检查")
        refresh.clicked.connect(self._refresh_diagnostics)
        data_folder = QPushButton("打开数据目录")
        data_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.root))))
        logs_folder = QPushButton("打开日志目录")
        logs_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.logs))))
        controls = QHBoxLayout()
        controls.addWidget(refresh)
        controls.addWidget(data_folder)
        controls.addWidget(logs_folder)
        controls.addStretch(1)
        layout = QVBoxLayout(page)
        layout.addWidget(self.diagnostics_table, 1)
        layout.addLayout(controls)
        return page

    def _pair(self, left: QWidget, right: QWidget) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(left)
        layout.addWidget(QLabel("×"))
        layout.addWidget(right)
        return widget

    def _rect_row(self, label: QLabel, button: QPushButton) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label, 1)
        layout.addWidget(button)
        return widget

    def _file_row(self, line_edit: QLineEdit, title: str) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("浏览")
        button.clicked.connect(lambda: self._browse_executable(line_edit, title))
        layout.addWidget(button)
        return widget

    def _load_controls(self) -> None:
        config = self.config
        source_index = self.source_combo.findData(config.capture.source)
        self.source_combo.setCurrentIndex(max(0, source_index))
        self.window_title.setText(config.capture.window_title)
        self.virtual_camera_index.setValue(config.capture.virtual_camera_index)
        self.sample_fps.setValue(config.sample_fps)
        self.stability_frames.setValue(config.stability_frames)
        self.dedupe_history.setValue(config.dedupe_history)
        self.match_threshold.setValue(config.speaker_match_threshold)
        self.resolution_width.setValue(config.capture.resolution_width)
        self.resolution_height.setValue(config.capture.resolution_height)
        self._update_rect_labels()
        self.output_volume.setValue(config.output_volume)
        self.pending_limit.setValue(config.pending_queue_limit)
        self.first_segment_limit.setValue(config.first_segment_max_chars)
        self.segment_limit.setValue(config.segment_max_chars)
        self.prefetch_window.setValue(config.prefetch_window)
        self.fake_tts.setChecked(config.use_fake_tts)
        self.backend_combo.setCurrentIndex(max(0, self.backend_combo.findData(config.tts_backend)))
        self.tts_endpoint.setText(config.tts_endpoint)
        self.tts_token.setText(config.tts_token)
        self.ffmpeg_path.setText(config.ffmpeg_path)
        self.ffprobe_path.setText(config.ffprobe_path)
        self.obs_executable.setText(config.obs_executable)
        self._update_backend_controls()

    def _save_controls(self) -> None:
        config = self.config
        config.capture.source = str(self.source_combo.currentData())
        config.capture.window_title = self.window_title.text().strip()
        config.capture.virtual_camera_index = self.virtual_camera_index.value()
        config.sample_fps = self.sample_fps.value()
        config.stability_frames = self.stability_frames.value()
        config.dedupe_history = self.dedupe_history.value()
        config.speaker_match_threshold = self.match_threshold.value()
        config.capture.resolution_width = self.resolution_width.value()
        config.capture.resolution_height = self.resolution_height.value()
        config.output_device = self.output_device.currentData()
        config.output_volume = self.output_volume.value()
        config.pending_queue_limit = self.pending_limit.value()
        config.first_segment_max_chars = self.first_segment_limit.value()
        config.segment_max_chars = self.segment_limit.value()
        config.prefetch_window = self.prefetch_window.value()
        config.use_fake_tts = self.fake_tts.isChecked()
        config.tts_backend = str(self.backend_combo.currentData())
        config.tts_endpoint = self.tts_endpoint.text().strip()
        config.tts_token = self.tts_token.text()
        config.ffmpeg_path = self.ffmpeg_path.text().strip()
        config.ffprobe_path = self.ffprobe_path.text().strip()
        config.obs_executable = self.obs_executable.text().strip()
        save_config(config, self.paths)
        self.message_label.setText("设置已保存" + ("，停止后重新启动生效" if self.pipeline else ""))

    def _update_backend_controls(self) -> None:
        if not hasattr(self, "tts_endpoint"):
            return
        backend = str(self.backend_combo.currentData())
        self.tts_endpoint.setText(endpoint_for_backend(backend, self.tts_endpoint.text()))
        cloning = backend in CLONING_BACKENDS and not self.fake_tts.isChecked()
        self.tts_endpoint.setEnabled(cloning)
        self.tts_token.setEnabled(cloning)
        self.tts_endpoint.setPlaceholderText(
            "http://127.0.0.1:9882" if backend == "qwen3-tts" else "http://127.0.0.1:9881"
        )

    def _preview_speech(self, *, force: bool = False) -> None:
        if self.preview is not None:
            return
        if self.pipeline is not None:
            self.preview_status.setText(
                "上一项配音正在结束，请稍后试听。" if self._pipeline_stopping else "请先停止实时配音，再试听。"
            )
            return
        text = self.preview_text.toPlainText().strip()
        if not text:
            self.preview_status.setText("请输入要朗读的文字。")
            return
        limit = 300 if self.backend_combo.currentData() == "qwen3-tts" else 500
        if len(text) > limit:
            self.preview_status.setText(f"试听请限制在 {limit} 字以内。")
            return
        self._save_controls()
        profile = VoiceProfile(speaker=self.config.narrator_profile)
        if self.config.tts_backend in CLONING_BACKENDS and not self.config.use_fake_tts:
            row = self.profile_table.currentRow()
            name = self.profile_table.item(row, 0).text() if row >= 0 else self.config.narrator_profile
            selected = self.voice_store.get(name)
            if selected is None:
                self.preview_status.setText("请先导入并选中角色参考声线，或切换到系统中文朗读。")
                return
            self.config.warmup_speaker = selected.speaker
            save_config(self.config, self.paths)
            errors = selected.validate(self.paths.profiles)
            if not selected.enabled:
                errors.append("该声线已停用")
            if errors:
                self.preview_status.setText("参考声线不可用：" + "；".join(errors))
                return
            from dataclasses import replace
            profile = replace(selected, reference_audio=str(selected.resolve_reference(self.paths.profiles)))
        self._preview_cancel_requested = False
        self.preview = SpeechPreview(self.config, text, profile, self.paths.cache / "preview.wav", self, force=force)
        self.preview.result.connect(self._update_preview_status)
        self.preview.finished.connect(self._preview_finished)
        self.preview_button.setEnabled(False)
        self.preview_regenerate_button.setEnabled(False)
        self.preview_stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.preview_status.setText(
            f"正在用{profile.speaker}重新生成并试听；可点击停止试听。"
            if force else f"正在准备{profile.speaker}的试听；已有音频会直接播放，可点击停止试听。"
        )
        self.preview.start()

    def _update_preview_status(self, message: str) -> None:
        if not self._preview_cancel_requested:
            self.preview_status.setText(message)

    def _preview_finished(self) -> None:
        self.preview_button.setEnabled(self.pipeline is None)
        self.preview_regenerate_button.setEnabled(self.pipeline is None)
        self.preview_stop_button.setEnabled(False)
        self.start_button.setEnabled(self.pipeline is None)
        if self.preview is not None:
            self.preview.deleteLater()
            self.preview = None
        if self._preview_cancel_requested:
            self.preview_status.setText("试听已停止。")
            self._preview_cancel_requested = False

    def _cancel_preview(self) -> None:
        if self.preview is not None:
            self._preview_cancel_requested = True
            self.preview.cancel()
            self.preview_stop_button.setEnabled(False)
            self.preview_status.setText("正在停止试听…")

    def _select_region(self, field: str) -> None:
        if self.pipeline is not None:
            QMessageBox.information(self, "区域校准", "请先停止实时配音，再调整识别区域。")
            return
        self._save_controls()
        self._show_region_selector(field)

    def _show_region_selector(self, field: str) -> None:
        capture = None
        try:
            capture = create_capture_source(self.config.capture, fps=self.config.sample_fps)
            capture.start()
            frame = capture.read(timeout=4.0)
            if frame is None:
                raise RuntimeError("没有捕获到画面，请确保所选游戏窗口可见。")
        except Exception as exc:
            QMessageBox.warning(self, "无法框选", str(exc))
            return
        finally:
            if capture is not None:
                capture.stop()
        selector = RegionSelector(frame.image, self)
        result = selector.exec()
        if result and selector.selected is not None:
            self.config.capture.resolution_width = frame.width
            self.config.capture.resolution_height = frame.height
            self.resolution_width.setValue(frame.width)
            self.resolution_height.setValue(frame.height)
            if field == "dialogue":
                self.config.capture.dialogue_rect = selector.selected
            else:
                self.config.capture.speaker_rect = selector.selected
            self._update_rect_labels()
            self._save_controls()

    def _update_rect_labels(self) -> None:
        self.dialogue_rect_label.setText(self._format_rect(self.config.capture.dialogue_rect))
        self.speaker_rect_label.setText(self._format_rect(self.config.capture.speaker_rect) if not self.config.capture.speaker_rect.is_empty else "未配置（按旁白朗读）")

    @staticmethod
    def _format_rect(rect: Rect) -> str:
        return "未配置" if rect.is_empty else f"{rect.left}, {rect.top}, {rect.width}×{rect.height}"

    def _choose_window(self) -> None:
        titles = list_window_titles()
        if not titles:
            QMessageBox.information(self, "窗口列表", "未获取到窗口列表，请手动输入标题。")
            return
        from PySide6.QtWidgets import QInputDialog

        title, accepted = QInputDialog.getItem(self, "选择窗口", "游戏窗口", titles, 0, False)
        if accepted:
            self.window_title.setText(title)

    def _browse_executable(self, target: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Executable (*.exe);;All Files (*.*)")
        if path:
            target.setText(path)

    def _start(self) -> None:
        if self.pipeline is not None:
            self.message_label.setText("上一项配音任务尚未结束，请先停止并等待完成。")
            return
        if self.preview is not None:
            self.preview_status.setText("请先停止试听，再启动实时配音。")
            return
        self._save_controls()
        try:
            self.pipeline = DubPipeline(
                AppConfig.from_dict(self.config.to_dict()),
                self.paths,
                self.voice_store,
                on_snapshot=self.bridge.snapshot.emit,
                on_error=self.bridge.error.emit,
            )
            self.pipeline.start()
        except Exception as exc:
            if self.pipeline is not None:
                self._stop()
            QMessageBox.critical(self, "无法启动", str(exc))
            return
        self.hotkeys = HotkeyManager(
            {
                "<f8>": self.bridge.pause_requested.emit,
                "<f9>": self.bridge.replay_requested.emit,
                "<f10>": self.bridge.jump_requested.emit,
            }
        )
        try:
            self.hotkeys.start()
        except Exception as exc:
            self.message_label.setText(f"配音已启动，快捷键不可用：{exc}；可使用窗口按钮。")
        self.start_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.preview_regenerate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.status_label.setText("监听中")
        if self.overlay_checkbox.isChecked():
            self.overlay.show()

    def _stop(self) -> None:
        pipeline = self.pipeline
        if pipeline is not None and not self._pipeline_stopping:
            self._pipeline_stopping = True
            pipeline.close()
        if self.hotkeys is not None:
            self.hotkeys.stop()
            self.hotkeys = None
        self._finish_pipeline_stop()

    def _finish_pipeline_stop(self) -> None:
        pipeline = self.pipeline
        waiting = pipeline is not None and any(thread.is_alive() for thread in pipeline._threads)
        if waiting:
            self._pipeline_stopping = True
            if not self._pipeline_stop_timer.isActive():
                self._pipeline_stop_timer.start()
        else:
            self.pipeline = None
            self._pipeline_stopping = False
            self._pipeline_stop_timer.stop()
        self.start_button.setEnabled(not waiting and self.preview is None)
        self.preview_button.setEnabled(not waiting and self.preview is None)
        self.preview_regenerate_button.setEnabled(not waiting and self.preview is None)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.status_label.setText("正在停止" if waiting else "已停止")
        waiting_message = "正在结束上一项配音任务，请稍候…"
        if waiting:
            self.message_label.setText(waiting_message)
        elif self.message_label.text() == waiting_message:
            self.message_label.setText("已停止，可以重新启动或试听。")
        self.overlay.hide()

    def _toggle_pause(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.toggle_pause()
        self.pause_button.setText("继续" if self.pipeline.snapshot().paused else "暂停")

    def _replay(self) -> None:
        if self.pipeline is not None:
            self.pipeline.replay_last()

    def _jump_latest(self) -> None:
        if self.pipeline is not None:
            self.pipeline.jump_to_latest()

    def _import_profile(self) -> None:
        dialog = VoiceProfileDialog(self)
        if dialog.exec() != VoiceProfileDialog.DialogCode.Accepted:
            return
        speaker, aliases, audio, prompt, speed = dialog.values()
        try:
            self.voice_store.import_reference(
                speaker=speaker,
                source_audio=audio,
                prompt_text=prompt,
                aliases=aliases,
                speed=speed,
                ffmpeg_path=self.ffmpeg_path.text().strip(),
                ffprobe_path=self.ffprobe_path.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self._refresh_profiles()
        if self.pipeline is not None:
            self.pipeline.reload_profiles()

    def _delete_profile(self) -> None:
        row = self.profile_table.currentRow()
        if row < 0:
            return
        speaker = self.profile_table.item(row, 0).text()
        if self.voice_store.delete(speaker):
            self._refresh_profiles()
            if self.pipeline is not None:
                self.pipeline.reload_profiles()

    def _reload_profiles(self) -> None:
        self._refresh_profiles()
        if self.pipeline is not None:
            self.pipeline.reload_profiles()

    def _refresh_profiles(self) -> None:
        profiles = self.voice_store.list()
        self.profile_table.setRowCount(len(profiles))
        for row, profile in enumerate(profiles):
            reference = profile.resolve_reference(self.paths.profiles)
            values = [
                profile.speaker,
                "、".join(profile.aliases),
                reference.name,
                "可用" if reference.is_file() else "缺少音频",
            ]
            for column, value in enumerate(values):
                self.profile_table.setItem(row, column, QTableWidgetItem(value))

    def _refresh_devices(self) -> None:
        current = self.config.output_device
        self.output_device.clear()
        self.output_device.addItem("系统默认", None)
        for device in AudioPlayer.list_devices():
            self.output_device.addItem(f"{device.name} [{device.host_api}]", device.index)
        index = self.output_device.findData(current)
        self.output_device.setCurrentIndex(max(0, index))

    def _refresh_diagnostics(self) -> None:
        items = collect_diagnostics(self.config, self.voice_store)
        self.diagnostics_table.setRowCount(len(items))
        for row, item in enumerate(items):
            for column, value in enumerate((item.name, item.status, item.detail)):
                self.diagnostics_table.setItem(row, column, QTableWidgetItem(value))

    def _test_ocr(self) -> None:
        self._save_controls()
        capture = None
        try:
            capture = create_capture_source(self.config.capture, fps=self.config.sample_fps)
            capture.start()
            frame = capture.read(timeout=3.0)
            if frame is None:
                raise RuntimeError("没有读取到画面")
            engine = RapidOcrEngine()
            dialogue = engine.recognize(crop_image(frame.image, self.config.capture.dialogue_rect))
            speaker = engine.recognize(crop_image(frame.image, self.config.capture.speaker_rect)) if not self.config.capture.speaker_rect.is_empty else []
            text = merge_ocr_lines(line.text for line in dialogue)
            speaker_text = merge_ocr_lines(line.text for line in speaker) or "旁白"
            QMessageBox.information(self, "OCR 测试", f"角色：{speaker_text}\n\n{text or '未识别到文字'}")
        except Exception as exc:
            QMessageBox.critical(self, "OCR 测试失败", str(exc))
        finally:
            if capture is not None:
                capture.stop()

    def _toggle_overlay(self, visible: bool) -> None:
        if not visible:
            self.overlay.hide()
        elif self.pipeline is not None:
            self.overlay.show()

    def _update_snapshot(self, snapshot: PipelineSnapshot) -> None:
        if not snapshot.running and self.pipeline is not None and not self.pipeline.snapshot().running:
            if self.hotkeys is not None:
                self.hotkeys.stop()
                self.hotkeys = None
            self._pipeline_stopping = True
        self.status_label.setText("已停止" if not snapshot.running else ("已暂停" if snapshot.paused else "监听中"))
        self.message_label.setText(snapshot.message)
        self.last_text.setPlainText(snapshot.last_text)
        latency = (
            f"{snapshot.last_latency_seconds:.2f}s"
            if snapshot.last_latency_seconds is not None
            else "--"
        )
        self.stats_label.setText(
            f"延迟 {latency} | 队列 {snapshot.queue_depth} | 跳过 {snapshot.skipped_count}"
        )
        self.overlay.update_status(snapshot)
        if self._pipeline_stopping:
            self._finish_pipeline_stop()

    def _show_error_status(self, message: str) -> None:
        self.message_label.setText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.preview is not None and self.preview.isRunning():
            self.preview.cancel()
            if not self.preview.wait(1500):
                self.preview_status.setText("正在停止试听，请稍后再次关闭。")
                event.ignore()
                return
        self._save_controls()
        self._stop()
        if self.pipeline is not None:
            event.ignore()
            return
        event.accept()
