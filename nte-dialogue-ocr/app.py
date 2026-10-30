"""Local Chinese dialogue OCR for Neverness to Everness (Windows)."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import subprocess
import queue

import cv2
import numpy as np

from capture_windows import enable_dpi_awareness, list_game_windows
from capture_session import CaptureSession, snapshot
from dialogue_store import Recorder, STATUS_LABELS

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"


def load_config():
    source = CONFIG if CONFIG.exists() else ROOT / "config.example.json"
    return json.loads(source.read_text(encoding="utf-8"))


def save_config(config):
    temp = CONFIG.with_suffix(".tmp")
    temp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(CONFIG)


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(".png", image)
    if not okay:
        raise RuntimeError("无法保存截图")
    encoded.tofile(str(path))


def check_geometry(frame, config, client_size=None):
    reference = config.get("reference_size")
    width, height = client_size or (frame.shape[1], frame.shape[0])
    if reference and abs(width / height / (reference[0] / reference[1]) - 1) > .015:
        raise RuntimeError("游戏画面比例已改变，请重新框选文字区域")


def run_loop(window, config, stop, callback, seconds=None, on_capture=None, presentation_queue=None):
    from ocr_engines import make_ocr
    from frame_gate import FrameStability
    if not config.get("calibrated") or not config.get("body_roi"):
        raise RuntimeError("尚未根据实际对话画面校准正文区域")
    if window.is_launcher:
        raise RuntimeError("当前是启动器，请进入游戏后刷新窗口列表")
    callback({"status_text": "正在加载本地中文模型…"})
    engine = make_ocr(config)
    stability=FrameStability(config.get('render_stable_seconds',.45))
    recorder = Recorder(ROOT, engine.session_id)
    capture = CaptureSession(window, backend=config.get("capture_backend", "wgc"),
                             interval_ms=round(1000/config.get("capture_fps", 5)),
                             rois=[config["body_roi"], config.get("speaker_roi")],
                             require_foreground=not (config.get("capture_backend", "wgc") == "wgc"
                                                    and config.get("wgc_background", False))).start()
    if on_capture:
        on_capture(capture)
    last_sequence = 0
    began = time.monotonic()
    end = began+seconds if seconds else float("inf")
    status_at = -float("inf")
    publication = None
    import psutil
    process=psutil.Process()
    cpu_started=sum(process.cpu_times()[:2])
    rss_values=[]
    resource_at=0
    def save_presentations():
        if presentation_queue is None:
            return
        while True:
            try:
                acknowledgement = presentation_queue.get_nowait()
            except queue.Empty:
                break
            with (recorder.directory/'presentation.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(acknowledgement,ensure_ascii=False)+'\n')
    try:
        while not stop.is_set() and time.monotonic() < end:
            save_presentations()
            now = time.monotonic()
            if now-resource_at>=.5:
                rss_values.append(process.memory_info().rss)
                resource_at=now
            if publication is not None:
                event, evidence = publication
                callback({"event": event, "status_text": "识别中 · "+STATUS_LABELS[event["status"]]})
                record = recorder.upsert(event, evidence)
                callback({"event": record, "persisted": True})
                publication = None
                continue
            if engine.pending_ids:
                identity = engine.pending_ids[0]
                # Review always consumes the original immutable crop bound to
                # this ID, even if the game has already moved to another line.
                publication = (engine.verify(identity), None)
                continue
            if not capture.ready_for_ocr():
                engine.pause(now=now)
                stability.reset()
                if now-status_at >= .5:
                    callback({"status_text": capture.status_text})
                    status_at = now
                stop.wait(.05)
                continue
            packet = capture.read(last_sequence)
            if packet is None:
                stop.wait(.03)
                continue
            last_sequence = packet.sequence
            check_geometry(packet.image, config, packet.client_size)
            body,_=engine.region_image(packet.image,engine.body_roi,packet.client_size,packet.origin)
            if not stability.ready(body,packet.captured_at):
                if now-status_at>=.5:
                    callback({'status_text':'正在等待文字显示完整…'})
                    status_at=now
                continue
            result = engine.process_frame(packet.image, now=now, client_size=packet.client_size,
                                          origin=packet.origin, captured_at=packet.captured_at,
                                          reuse_body=stability.unchanged)
            stability.commit()
            event = result.get("event")
            if event:
                event["capture_backend"] = capture.backend_name
                event["source_kind"] = "live_game_capture"
                event["transparent_overlays"] = list(packet.transparent_overlays)
                event["visibility_note"] = ("WGC 窗口画面；桌面遮挡不参与取图" if capture.backend_name == "wgc" else
                    ("存在鼠标穿透覆盖层，请以原图核对" if packet.transparent_overlays else "普通窗口遮挡检查通过"))
                event["evaluation_batch"] = config.get("evaluation_batch")
                evidence = engine.evidence(event["dialogue_id"])
                if evidence is None and event.get("revision_reason") == "speaker_stabilized":
                    name, offset = engine.region_image(packet.image, engine.speaker_roi, packet.client_size, packet.origin)
                    evidence = {"speaker": name}
                publication = (event, evidence)
            elif now-status_at >= .5:
                callback({"status_text": "识别中 · 等待新台词" if result["status"] == "blank" else "识别中 · 台词未变化"})
                status_at = now
    finally:
        # Stopping or losing focus must not discard an already computed
        # candidate/revision and its original evidence.
        try:
            if publication is not None:
                event, evidence = publication
                recorder.upsert(event, evidence)
        finally:
            capture.stop()
            save_presentations()
        metrics = dict(capture.metrics(), duration_seconds=time.monotonic()-began,
                       dialogue_count=len(recorder.records), verification_threshold=engine.min_confidence,
                       ocr_profile=engine.ocr_profile, candidate_model=engine.candidate_model,
                       review_model=engine.review_model, evaluation_batch=config.get("evaluation_batch"),
                       accuracy_measured=False,
                       cpu_percent_one_core=100*(sum(process.cpu_times()[:2])-cpu_started)/max(.001,time.monotonic()-began),
                       rss_peak_mb=max(rss_values)/1024**2 if rss_values else None,
                       resource_note='包含本进程的截图和识别；界面模式也包含界面。100% CPU 表示一个逻辑核心')
        (recorder.directory/"session.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def gui(autoconnect=False):
    from PySide6 import QtCore, QtGui, QtWidgets
    from ocr_core import crop_roi
    from game_launch import ensure_game, is_admin, relaunch_admin, activate_game

    def pixmap(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888).copy())

    class SelectRegion(QtWidgets.QDialog):
        def __init__(self, frame, title, parent):
            super().__init__(parent)
            self.setWindowTitle(title + "：拖动框选，松开后保存")
            source = pixmap(frame)
            available = self.screen().availableGeometry()
            source = source.scaled(int(available.width() * .85), int(available.height() * .8), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label = QtWidgets.QLabel(self)
            self.label.setPixmap(source)
            self.label.setFixedSize(source.size())
            self.setFixedSize(source.size())
            self.rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
            self.origin = None
            self.roi = None
            self.setCursor(QtCore.Qt.CrossCursor)
        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.LeftButton:
                self.origin = event.position().toPoint()
                self.rubber.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
                self.rubber.show()
        def mouseMoveEvent(self, event):
            if self.origin is not None:
                self.rubber.setGeometry(QtCore.QRect(self.origin, event.position().toPoint()).normalized().intersected(self.rect()))
        def mouseReleaseEvent(self, event):
            if self.origin is None:
                return
            rect = self.rubber.geometry()
            if rect.width() >= 20 and rect.height() >= 8:
                self.roi = [rect.x()/self.width(), rect.y()/self.height(), (rect.x()+rect.width())/self.width(), (rect.y()+rect.height())/self.height()]
                self.accept()
            self.origin = None

    class Worker(QtCore.QThread):
        message = QtCore.Signal(object)
        def __init__(self, window, config, parent):
            super().__init__(parent)
            self.window = window
            self.capture = None
            self.presentation_queue = queue.SimpleQueue()
            self.presented = set()
            self.config = dict(config)
            self.stop_event = threading.Event()
        def run(self):
            try:
                run_loop(self.window, self.config, self.stop_event, self.message.emit,
                         on_capture=lambda capture: setattr(self, "capture", capture),
                         presentation_queue=self.presentation_queue)
            except Exception as exc:
                self.message.emit({"error": str(exc)})

    class LaunchWorker(QtCore.QThread):
        message = QtCore.Signal(str)
        preview = QtCore.Signal(object)
        connected = QtCore.Signal(object)
        failed = QtCore.Signal(str)
        def __init__(self, config, parent):
            super().__init__(parent)
            self.config = dict(config)
            self.stop_event = threading.Event()
        def run(self):
            try:
                result = ensure_game(self.config, self.stop_event, self.message.emit,
                                     timeout=self.config.get("launch_timeout_seconds", 180), preview=self.preview.emit)
                self.connected.emit(result)
            except Exception as exc:
                self.failed.emit(str(exc))

    class SnapshotWorker(QtCore.QThread):
        ready = QtCore.Signal(object)
        failed = QtCore.Signal(str)
        def __init__(self, window, backend, parent):
            super().__init__(parent)
            self.window, self.backend = window, backend
            self.stop_event = threading.Event()
        def run(self):
            try:
                activate_game(self.window, self.stop_event)
                frame = snapshot(self.window, backend=self.backend, timeout=10, stop=self.stop_event)
                if not self.stop_event.is_set():
                    self.ready.emit(frame)
            except Exception as exc:
                self.failed.emit(str(exc))

    class Main(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.config = load_config()
            self.worker = None
            self.launch_worker = None
            self.launch_result = None
            self.snapshot_worker = None
            self.history_items = {}
            self.latest_dialogue_id = None
            self.setWindowTitle("异环 · 对话文字识别")
            self.resize(860, 720)
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel("异环对话识别")
            title.setStyleSheet("font-size:24px; font-weight:600; padding:8px 0")
            layout.addWidget(title)
            layout.addWidget(QtWidgets.QLabel("本地识别并记录对话。首次使用先停在对话画面，框选正文。"))
            row = QtWidgets.QHBoxLayout()
            self.connect_button = QtWidgets.QPushButton("自动打开游戏并连接")
            self.connect_button.clicked.connect(self.connect_game)
            row.addWidget(self.connect_button, 1)
            self.path_button = QtWidgets.QPushButton("选择启动器")
            self.path_button.clicked.connect(self.choose_launcher)
            row.addWidget(self.path_button)
            self.backend_choice = QtWidgets.QComboBox()
            self.backend_choice.addItem("画面获取：WGC", "wgc")
            self.backend_choice.addItem("画面获取：MSS", "mss")
            self.backend_choice.addItem("画面获取：PrintWindow", "printwindow")
            self.backend_choice.setCurrentIndex(max(0, self.backend_choice.findData(self.config.get("capture_backend", "wgc"))))
            self.backend_choice.currentIndexChanged.connect(self.set_backend)
            row.addWidget(self.backend_choice)
            layout.addLayout(row)
            from ocr_engines import PROFILES
            self.ocr_choice = QtWidgets.QComboBox()
            for key, (label, profile) in PROFILES.items():
                self.ocr_choice.addItem(label, key)
            self.ocr_choice.setCurrentIndex(max(0, self.ocr_choice.findData(self.config.get("ocr_mode", "rapid_dual"))))
            self.ocr_choice.currentIndexChanged.connect(self.set_ocr_mode)
            layout.addWidget(self.ocr_choice)
            self.background_choice = QtWidgets.QCheckBox("游戏在后台时继续识别（WGC）")
            self.background_choice.setChecked(self.config.get("wgc_background", False))
            self.background_choice.setEnabled(self.config.get("capture_backend", "wgc") == "wgc")
            self.background_choice.toggled.connect(self.set_background)
            layout.addWidget(self.background_choice)
            row = QtWidgets.QHBoxLayout()
            self.windows = QtWidgets.QComboBox()
            row.addWidget(self.windows, 1)
            refresh = QtWidgets.QPushButton("刷新游戏窗口")
            refresh.clicked.connect(self.refresh)
            row.addWidget(refresh)
            layout.addLayout(row)
            row = QtWidgets.QHBoxLayout()
            self.edit_buttons = []
            for label, key in (("框选正文", "body_roi"), ("框选角色名（可选）", "speaker_roi")):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda checked=False, k=key: self.select(k))
                row.addWidget(button)
                self.edit_buttons.append(button)
            clear_name = QtWidgets.QPushButton("不识别角色名")
            clear_name.clicked.connect(self.clear_speaker)
            row.addWidget(clear_name)
            self.edit_buttons.append(clear_name)
            layout.addLayout(row)
            self.preview = QtWidgets.QLabel("尚未校准：等待实际游戏对话画面")
            self.preview.setAlignment(QtCore.Qt.AlignCenter)
            self.preview.setMinimumHeight(120)
            self.preview.setStyleSheet("background:#17202a; color:#dce4ed; padding:8px")
            layout.addWidget(self.preview)
            calibration_preview = ROOT / "validation" / "stage2" / "calibrated-body.png"
            if calibration_preview.exists():
                pm = QtGui.QPixmap(str(calibration_preview))
                self.preview.setPixmap(pm.scaled(800, 150, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.status = QtWidgets.QLabel("已校准，可启动" if self.config.get("calibrated") else "等待校准正文区域")
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            row = QtWidgets.QHBoxLayout()
            self.start_button = QtWidgets.QPushButton("启动识别")
            self.start_button.clicked.connect(self.start)
            self.stop_button = QtWidgets.QPushButton("停止识别")
            self.stop_button.clicked.connect(self.stop)
            self.stop_button.setEnabled(False)
            records = QtWidgets.QPushButton("打开识别记录")
            records.clicked.connect(self.open_records)
            review_button = QtWidgets.QPushButton("真实台词验收")
            review_button.clicked.connect(self.open_review)
            for button in (self.start_button, self.stop_button, records, review_button):
                row.addWidget(button)
            layout.addLayout(row)
            self.current = QtWidgets.QPlainTextEdit()
            self.current.setReadOnly(True)
            self.current.setPlaceholderText("当前识别结果会显示在这里")
            self.current.setMaximumHeight(130)
            layout.addWidget(self.current)
            self.history = QtWidgets.QListWidget()
            self.history.setWordWrap(True)
            layout.addWidget(self.history, 1)
            layout.addWidget(QtWidgets.QLabel("WGC 可后台识别；最小化时暂停。已复核只表示两次检查通过，仍可能有误。"))
            self.preview_timer = QtCore.QTimer(self)
            self.preview_timer.timeout.connect(self.show_live_preview)
            self.preview_timer.start(round(1000/self.config.get("preview_fps", 2)))
            self.refresh()
            if autoconnect:
                QtCore.QTimer.singleShot(300, self.connect_game)

        def is_busy(self):
            return bool((self.worker and self.worker.isRunning()) or
                        (self.launch_worker and self.launch_worker.isRunning()) or
                        (self.snapshot_worker and self.snapshot_worker.isRunning()))

        def set_backend(self):
            self.config["capture_backend"] = self.backend_choice.currentData()
            save_config(self.config)
            self.background_choice.setEnabled(self.config["capture_backend"] == "wgc")

        def set_ocr_mode(self):
            self.config["ocr_mode"] = self.ocr_choice.currentData()
            save_config(self.config)

        def set_background(self, enabled):
            self.config["wgc_background"] = bool(enabled)
            save_config(self.config)

        def choose_launcher(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择官方启动器 NTEGame.exe",
                                                           self.config.get("launcher_path", ""), "启动器 (*.exe)")
            if path:
                from game_launch import _launcher_candidate
                candidate = _launcher_candidate(path)
                if not candidate:
                    self.status.setText("请选择 NTEGame.exe、NTEGlobalGame.exe 或 NTETWGame.exe")
                    return
                self.config["launcher_path"] = candidate
                save_config(self.config)
                self.status.setText("启动器路径已保存")

        def connect_game(self):
            if self.is_busy():
                return
            try:
                # A running game needs only screen read access. Do not prompt
                # for elevation or send input just to attach the OCR worker.
                games = [w for w in list_game_windows() if not w.is_launcher]
                if games:
                    self.refresh()
                    for index in range(self.windows.count()):
                        if self.windows.itemData(index).hwnd == games[0].hwnd:
                            self.windows.setCurrentIndex(index)
                            break
                    self.status.setText("已连接现有游戏；切回剧情对话后开始采集")
                    if self.config.get("calibrated"):
                        self.start()
                    return
                if not is_admin():
                    self.status.setText("正在申请 Windows 管理员授权，以便操作官方启动器…")
                    QtWidgets.QApplication.processEvents()
                    relaunch_admin(["--autoconnect"])
                    self.close()
                    return
                self.launch_result = None
                self.launch_worker = LaunchWorker(self.config, self)
                self.launch_worker.message.connect(self.status.setText)
                self.launch_worker.preview.connect(self.show_launch_preview)
                self.launch_worker.connected.connect(self.game_connected)
                self.launch_worker.failed.connect(lambda error: self.status.setText("连接未完成：" + error))
                self.launch_worker.finished.connect(self.launch_finished)
                self.set_controls_busy(True)
                self.launch_worker.start()
            except Exception as exc:
                self.status.setText(str(exc))

        def game_connected(self, result):
            if self.launch_worker.stop_event.is_set():
                self.status.setText("自动连接已取消")
                return
            self.launch_result = result
            report = result["report"]
            if report.get("launcher_path"):
                self.config["launcher_path"] = report["launcher_path"]
            self.config["game_title"] = result["window"].title
            save_config(self.config)
            self.preview.setPixmap(pixmap(result["frame"]).scaled(800, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.status.setText("已进入游戏内部 · 可校准文字区域或开始识别")

        def show_launch_preview(self, frame):
            if self.launch_worker and not self.launch_worker.stop_event.is_set():
                self.preview.setPixmap(pixmap(frame).scaled(800, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        def launch_finished(self):
            self.set_controls_busy(False)
            self.refresh()
            if (self.launch_result is not None and not self.launch_worker.stop_event.is_set()
                    and self.config.get("calibrated") and self.config.get("startup_acceptance") == "accepted"):
                QtCore.QTimer.singleShot(100, self.start)

        def set_controls_busy(self, busy):
            self.start_button.setEnabled(not busy)
            self.stop_button.setEnabled(busy)
            self.windows.setEnabled(not busy)
            self.connect_button.setEnabled(not busy)
            self.path_button.setEnabled(not busy)
            self.backend_choice.setEnabled(not busy)
            self.ocr_choice.setEnabled(not busy)
            self.background_choice.setEnabled(not busy and self.config.get("capture_backend") == "wgc")
            for button in self.edit_buttons:
                button.setEnabled(not busy)

        def refresh(self):
            if self.is_busy():
                return
            self.windows.clear()
            for window in list_game_windows():
                suffix = "（启动器，请先进入游戏）" if window.is_launcher else f" · {window.width}×{window.height}"
                self.windows.addItem(window.title + suffix, window)

        def target(self):
            window = self.windows.currentData()
            if not window:
                raise RuntimeError("没有找到《异环》窗口，请进入游戏后刷新")
            if window.is_launcher:
                raise RuntimeError("当前只有启动器。请点击开始游戏，并停在剧情对话画面后刷新")
            return window

        def show_live_preview(self):
            if self.worker and self.worker.capture:
                packet = self.worker.capture.peek()
                if packet is not None:
                    self.preview.setPixmap(pixmap(packet.image).scaled(800, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        def select(self, key):
            if self.is_busy():
                return
            try:
                window = self.target()
                self.status.setText("正在后台获取校准截图，请稍候…")
                self.snapshot_worker = SnapshotWorker(window, self.config.get("capture_backend", "wgc"), self)
                self.snapshot_worker.ready.connect(lambda frame: self.select_ready(frame, key, window))
                self.snapshot_worker.failed.connect(lambda error: self.status.setText("截图未完成："+error))
                self.snapshot_worker.finished.connect(lambda: self.set_controls_busy(False))
                self.set_controls_busy(True)
                self.snapshot_worker.start()
            except Exception as exc:
                self.status.setText(str(exc))

        def select_ready(self, frame, key, window):
            try:
                if self.snapshot_worker.stop_event.is_set():
                    return
                if key == "speaker_roi" and self.config.get("body_roi"):
                    check_geometry(frame, self.config)
                dialog = SelectRegion(frame, "正文" if key == "body_roi" else "角色名", self)
                if dialog.exec() == QtWidgets.QDialog.Accepted:
                    if key == "body_roi":
                        previous_size = self.config.get("reference_size")
                        if previous_size and abs(frame.shape[1]/frame.shape[0]/(previous_size[0]/previous_size[1])-1) > .015:
                            self.config["speaker_roi"] = None
                    self.config[key] = dialog.roi
                    self.config["reference_size"] = [frame.shape[1], frame.shape[0]]
                    self.config["game_title"] = window.title
                    self.config["calibrated"] = bool(self.config.get("body_roi"))
                    save_config(self.config)
                    if self.config.get("body_roi"):
                        body = crop_roi(frame, self.config["body_roi"])
                        self.preview.setPixmap(pixmap(body).scaled(800, 150, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                    self.status.setText("区域已保存，可启动识别")
            except Exception as exc:
                self.status.setText(str(exc))

        def clear_speaker(self):
            self.config["speaker_roi"] = None
            save_config(self.config)
            self.status.setText("已关闭角色名识别")

        def start(self):
            if self.is_busy():
                return
            try:
                window = self.target()
                if not self.config.get("calibrated"):
                    raise RuntimeError("请先框选实际对话正文区域")
                self.worker = Worker(window, self.config, self)
                self.worker.message.connect(self.update_result)
                self.worker.finished.connect(self.stopped)
                self.worker.start()
                self.set_controls_busy(True)
            except Exception as exc:
                self.status.setText(str(exc))

        def stop(self):
            if self.snapshot_worker and self.snapshot_worker.isRunning():
                self.snapshot_worker.stop_event.set()
                self.status.setText("正在取消截图…")
            if self.launch_worker and self.launch_worker.isRunning():
                self.launch_worker.stop_event.set()
                self.status.setText("正在取消自动连接…")
            if self.worker:
                self.worker.stop_event.set()
                self.status.setText("正在结束当前识别…")

        def stopped(self):
            self.set_controls_busy(False)
            if not self.status.text().startswith("错误"):
                self.status.setText("已停止，记录已保存")

        def update_result(self, result):
            if "error" in result:
                self.status.setText("错误："+result["error"])
                return
            if result.get("status_text"):
                self.status.setText(result["status_text"])
            record = result.get("event")
            if record is None:
                return
            identity = record["dialogue_id"]
            label = STATUS_LABELS.get(record["status"], record["status"])
            name_state = STATUS_LABELS.get(record.get("speaker_status"), "未启用")
            display = f"{record.get('speaker', '')} · 正文{label} · 人名{name_state} · 修订 {record['revision']}\n{record['body']}"
            if record.get("review_text") is not None and record["review_text"] != record["original_text"]:
                display += "\n复核候选："+record["review_text"]
            if record.get('source_clipped'):
                display += '\n文字靠近裁剪边缘，可能不完整；请重新框选正文。'
            if record.get('coverage_warning') or record.get('review_coverage_warning'):
                display += '\n文字框附近还有未覆盖的笔画，请看原图核对是否漏字。'
            if record['status']=='verified':
                display += '\n已复核表示模型检查通过，尚未经人工逐字确认。'
            if identity not in self.history_items:
                item = QtWidgets.QListWidgetItem(display)
                self.history.addItem(item)
                self.history_items[identity] = item
                self.latest_dialogue_id = identity
            else:
                self.history_items[identity].setText(display)
            if identity == self.latest_dialogue_id:
                self.current.setPlainText(display)
            saved = "原图和修订已保存。" if result.get("persisted") else "正在保存原图和修订。"
            self.history_items[identity].setToolTip("已复核表示模型及文字覆盖检查通过，不等于人工确认无误。"+saved)
            if self.worker and record.get('captured_monotonic') is not None:
                key=(identity,record['revision'])
                if key not in self.worker.presented:
                    self.worker.presented.add(key)
                    self.worker.presentation_queue.put(dict(dialogue_id=identity,revision=record['revision'],
                        status=record['status'],ui_applied_latency_ms=round((time.monotonic()-record['captured_monotonic'])*1000,3),
                        note='采样到界面文字控件更新；不等于显示器刷新完成'))

        def open_records(self):
            path = ROOT / "records"
            path.mkdir(exist_ok=True)
            os.startfile(path)

        def open_review(self):
            subprocess.Popen([str(Path(sys.executable).with_name("pythonw.exe")), str(ROOT/"review_samples.py")], cwd=ROOT)

        def closeEvent(self, event):
            if self.snapshot_worker and self.snapshot_worker.isRunning():
                self.snapshot_worker.stop_event.set()
                if not self.snapshot_worker.wait(4000):
                    event.ignore()
                    return
            if self.launch_worker and self.launch_worker.isRunning():
                self.launch_worker.stop_event.set()
                if not self.launch_worker.wait(5000):
                    event.ignore()
                    self.status.setText("正在结束画面连接，请稍后关闭")
                    return
            if self.worker and self.worker.isRunning():
                self.stop()
                if not self.worker.wait(5000):
                    event.ignore()
                    self.status.setText("等待当前识别结束后，请再次关闭")
                    return
            event.accept()

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    instance_lock = QtCore.QLockFile(str(ROOT / "app.lock"))
    if not instance_lock.tryLock(5000):
        QtWidgets.QMessageBox.information(None, "异环对话识别", "识别窗口已在运行，请使用已打开的窗口。")
        return 0
    application.setStyle("Fusion")
    font = application.font()
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "SimSun"])
    font.setPointSize(10)
    application.setFont(font)
    main = Main()
    main.show()
    return application.exec()


def main():
    enable_dpi_awareness()
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--autoconnect", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--backend", choices=["mss", "wgc", "printwindow"])
    args = parser.parse_args()
    config = load_config()
    if args.backend:
        config["capture_backend"] = args.backend
    if args.launch:
        from game_launch import ensure_game
        result = ensure_game(config, progress=lambda message: print(message, flush=True),
                             timeout=config.get("launch_timeout_seconds", 180))
        print(json.dumps(result["report"], ensure_ascii=True), flush=True)
        return 0
    if args.list:
        print(json.dumps([vars(w) | {"is_launcher": w.is_launcher} for w in list_game_windows()], ensure_ascii=True, indent=2))
        return 0
    if args.snapshot or args.headless:
        windows = [w for w in list_game_windows() if not w.is_launcher]
        if not windows:
            raise RuntimeError("尚未找到实际游戏窗口（启动器不算游戏画面）")
        if args.snapshot:
            frame = snapshot(windows[0], backend=config.get("capture_backend", "wgc"))
            write_png(Path(args.snapshot), frame)
            print(json.dumps({"size": [frame.shape[1], frame.shape[0]], "window": vars(windows[0])}, ensure_ascii=True))
        else:
            run_loop(windows[0], config, threading.Event(), lambda result: print(json.dumps(result, ensure_ascii=True), flush=True), args.seconds)
        return 0
    return gui(args.autoconnect)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except Exception:
        log = ROOT / "error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        raise
