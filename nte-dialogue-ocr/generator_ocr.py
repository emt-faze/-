"""Collect real OCR revisions into the generator without starting speech."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading

from PySide6 import QtCore, QtGui, QtWidgets

from dialogue_generation import atomic_text


def save_ocr_settings(root, updates):
    allowed = {"body_roi", "speaker_roi", "reference_size", "calibrated", "game_title", "capture_backend"}
    if not set(updates).issubset(allowed):
        raise ValueError("不支持的识别设置")
    path = Path(root) / "config.json"
    # Refresh before patching; never overwrite voice or unrelated settings.
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    config.update(updates)
    atomic_text(path, json.dumps(config, ensure_ascii=False, indent=2))
    return config


def pixmap(frame):
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                              QtGui.QImage.Format_RGB888).copy())


class CollectionWorker(QtCore.QThread):
    message = QtCore.Signal(object)

    def __init__(self, window, config, root, parent=None, *, runner=None, playback_events=False):
        super().__init__(parent)
        self.window, self.config, self.root = window, copy.deepcopy(config), Path(root)
        self.config.setdefault("dubbing", {})["enabled"] = False
        self.stop_event = threading.Event()
        self.capture = None
        self.directory = None
        self.runner = runner
        self.playback_events = playback_events

    def cancel(self):
        self.stop_event.set()

    def run(self):
        from app import run_loop
        frame_watch = None
        def captured(capture):
            nonlocal frame_watch
            self.capture = capture
            if self.playback_events:
                frame_watch = PlaybackFrameWatch(capture, self.config, self.message.emit)
                frame_watch.start()
        def session(directory):
            self.directory = Path(directory)
            self.message.emit({"session": str(self.directory)})
        def record(value):
            self.message.emit({"record": copy.deepcopy(value), "source": str(self.directory)})
        def status(value):
            if "status_text" in value:
                self.message.emit({"status_text": value["status_text"]})
        def observation(value):
            self.message.emit({"observation": {key: value.get(key) for key in
                                               ("body", "speaker", "monotonic_seconds", "captured_monotonic",
                                                "body_confidence", "speaker_confidence", "source_clipped",
                                                "coverage_warning", "review_coverage_warning", "render_stable")}})
        try:
            hooks = {}
            if self.playback_events:
                hooks = dict(on_state=lambda state: self.message.emit({"capture_state": state}),
                             on_visual_observation=lambda *args: frame_watch.arm(*args) if frame_watch else None)
            (self.runner or run_loop)(self.window, self.config, self.stop_event, status,
                                      on_capture=captured,
                                      on_record=record, on_session=session, on_observation=observation,
                                      records_root=self.root, **hooks)
        except Exception as exc:
            self.message.emit({"error": str(exc)})
        finally:
            if frame_watch:
                frame_watch.close()
            # run_loop also flushes an already-computed candidate while stopping.
            if self.directory and (self.directory / "dialogue.jsonl").is_file():
                self.message.emit({"final_file": str(self.directory / "dialogue.jsonl")})


class PlaybackFrameWatch:
    """Notice page changes even while OCR is occupied with an older frame.

    This only inspects already captured pixels. It neither takes screenshots nor
    recognizes text, and it never starts audio. Changes to recognized subtitle
    glyphs invalidate the old observation independently of a slow OCR review.
    """
    def __init__(self, capture, config, callback):
        from frame_gate import SubtitleChangeGuard
        self.capture, self.config, self.callback = capture, config, callback
        self.stop = threading.Event()
        self.guard = SubtitleChangeGuard()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, name='prepared-frame-watch', daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()

    def arm(self, image, observation, offset):
        with self.lock:
            self.guard.arm(image,observation,offset)

    def run(self):
        from ocr_core import roi_to_pixels
        sequence, available = 0, False
        try:
            while not self.stop.is_set():
                packet = self.capture.peek() if self.capture.ready_for_ocr() else None
                if packet is None:
                    if available:
                        self.callback({'capture_state': 'unavailable'})
                    available = False
                    with self.lock:
                        self.guard.reset()
                elif packet.sequence > sequence:
                    available, sequence = True, packet.sequence
                    width, height = packet.client_size
                    left, top, right, bottom = roi_to_pixels((height, width), self.config['body_roi'])
                    x, y = packet.origin
                    body = packet.image[max(0, top-y):bottom-y, max(0, left-x):right-x]
                    if not body.size:
                        raise ValueError('台词区域不在当前画面内，请重新框选')
                    with self.lock:
                        changed=self.guard.changed(body,packet.captured_at)
                    if changed:
                        self.callback({'frame_changed': packet.captured_at})
                self.stop.wait(.05)
        except Exception as exc:
            if not self.stop.is_set():
                self.callback({'error': '字幕画面监测失败：'+str(exc)})


class SnapshotWorker(QtCore.QThread):
    ready = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, window, backend, parent=None):
        super().__init__(parent)
        self.window, self.backend = window, backend
        self.stop_event = threading.Event()

    def cancel(self):
        self.stop_event.set()

    def run(self):
        from capture_session import snapshot
        try:
            frame = snapshot(self.window, backend=self.backend, timeout=10, stop=self.stop_event)
            if not self.stop_event.is_set():
                self.ready.emit(frame)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error.emit(str(exc))


class RegionDialog(QtWidgets.QDialog):
    def __init__(self, frame, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("框选" + name + " · 拖动后松开保存，Esc 取消")
        available = self.screen().availableGeometry()
        image = pixmap(frame).scaled(int(available.width()*.85), int(available.height()*.8),
                                     QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label = QtWidgets.QLabel(self)
        self.label.setPixmap(image)
        self.label.setFixedSize(image.size())
        self.setFixedSize(image.size())
        self.rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.origin, self.roi = None, None
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
            self.roi = [rect.x()/self.width(), rect.y()/self.height(),
                        (rect.x()+rect.width())/self.width(), (rect.y()+rect.height())/self.height()]
            self.accept()
        self.origin = None


class OcrCollectionDialog(QtWidgets.QDialog):
    collected = QtCore.Signal(object)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8-sig"))
        self.worker = self.snapshot_worker = None
        self.closing = False
        self.received = {}
        self.latest_observation = None
        self.latest_record = None
        self.setWindowTitle("识别游戏台词 · 加入当前生成项目")
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.resize(840, 650)
        layout = QtWidgets.QVBoxLayout(self)
        self.note = note = QtWidgets.QLabel("先进入游戏的对话画面，再开始采集。角色、台词及复核结果会自动加入当前项目。\n"
                               "采集时不生成或播放配音；完成后停止采集，返回列表核对和批量生成。")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.windows = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("刷新游戏窗口")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.windows, 1)
        row.addWidget(self.refresh_button)
        self.backend = QtWidgets.QComboBox()
        for label, value in (("WGC（支持后台）", "wgc"), ("MSS（需游戏在前台）", "mss"), ("PrintWindow", "printwindow")):
            self.backend.addItem(label, value)
        self.backend.setCurrentIndex(max(0, self.backend.findData(self.config.get("capture_backend", "wgc"))))
        self.backend.currentIndexChanged.connect(self.change_backend)
        row.addWidget(self.backend)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.body_button = QtWidgets.QPushButton("框选台词区域")
        self.speaker_button = QtWidgets.QPushButton("框选角色名区域")
        self.clear_speaker = QtWidgets.QPushButton("不识别角色名")
        self.body_button.clicked.connect(lambda: self.calibrate("body_roi"))
        self.speaker_button.clicked.connect(lambda: self.calibrate("speaker_roi"))
        self.clear_speaker.clicked.connect(self.remove_speaker_region)
        for button in (self.body_button, self.speaker_button, self.clear_speaker):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.region_note = QtWidgets.QLabel()
        self.region_note.setWordWrap(True)
        layout.addWidget(self.region_note)
        self.preview = QtWidgets.QLabel("开始后显示识别区域；首次使用可先框选台词区域。")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(160)
        self.preview.setStyleSheet("background:#172c43; color:#edf5fc; padding:6px;")
        layout.addWidget(self.preview, 1)
        self.current = QtWidgets.QPlainTextEdit()
        self.current.setReadOnly(True)
        self.current.setPlaceholderText("最新识别到的角色与台词")
        self.current.setMaximumHeight(140)
        layout.addWidget(self.current)
        self.count = QtWidgets.QLabel("本次尚未采集台词")
        layout.addWidget(self.count)
        self.status = QtWidgets.QLabel("沿用已有识别设置。请确认选中的是游戏窗口。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton("开始采集到当前项目")
        self.stop = QtWidgets.QPushButton("停止采集")
        self.return_button = QtWidgets.QPushButton("返回台词列表")
        self.start.clicked.connect(self.start_collection)
        self.stop.clicked.connect(self.stop_work)
        self.return_button.clicked.connect(self.close)
        self.stop.setEnabled(False)
        row.addWidget(self.start)
        row.addWidget(self.stop)
        row.addStretch()
        row.addWidget(self.return_button)
        layout.addLayout(row)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.show_preview)
        self.timer.start(500)
        self.refresh()
        self.update_region_note()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange and self.isMinimized():
            owner = self.parentWidget()
            if owner is not None and not owner.window().isMinimized():
                owner.window().showMinimized()

    def busy(self):
        return any(worker and worker.isRunning() for worker in (self.worker, self.snapshot_worker))

    def set_busy(self, value):
        for widget in (self.windows, self.refresh_button, self.backend, self.body_button,
                       self.speaker_button, self.clear_speaker, self.start):
            widget.setEnabled(not value)
        self.stop.setEnabled(value)

    def refresh(self):
        from capture_windows import list_game_windows
        if self.busy():
            return
        current = self.windows.currentData()
        self.windows.clear()
        for window in list_game_windows():
            if not window.is_launcher:
                self.windows.addItem(f"{window.title.strip()} · {window.width}×{window.height}", window)
                if current and window.hwnd == current.hwnd:
                    self.windows.setCurrentIndex(self.windows.count()-1)
        if not self.windows.count():
            self.status.setText("尚未找到游戏窗口。请手动进入游戏，然后点击“刷新游戏窗口”。")

    def target(self):
        window = self.windows.currentData()
        if window is None or window.is_launcher:
            raise ValueError("请先进入游戏并刷新窗口列表，再选择游戏窗口")
        return window

    def change_backend(self):
        try:
            self.config = save_ocr_settings(self.root, {"capture_backend": self.backend.currentData()})
        except Exception as exc:
            self.status.setText("设置未保存：" + str(exc))

    def update_region_note(self):
        self.region_note.setText(("已载入正文区域" if self.config.get("calibrated") else "请先框选正文区域") +
                                 ("和角色名区域。" if self.config.get("speaker_roi") else "；当前未识别角色名。") +
                                 " 文字裁切或画面比例变化时请重新框选。")

    def remove_speaker_region(self):
        try:
            self.config = save_ocr_settings(self.root, {"speaker_roi": None})
            self.update_region_note()
        except Exception as exc:
            self.status.setText("设置未保存：" + str(exc))

    def calibrate(self, key):
        if self.busy():
            return
        try:
            window = self.target()
            self.snapshot_worker = SnapshotWorker(window, self.backend.currentData(), self)
            self.snapshot_worker.ready.connect(lambda frame: self.region_ready(frame, key, window))
            self.snapshot_worker.error.connect(lambda error: self.status.setText("截图未完成：" + error))
            self.snapshot_worker.finished.connect(self.finished_work)
            self.set_busy(True)
            self.status.setText("正在获取游戏画面，请保持窗口可用…")
            self.snapshot_worker.start()
        except Exception as exc:
            self.status.setText(str(exc))

    def region_ready(self, frame, key, window):
        if self.closing or self.snapshot_worker.stop_event.is_set():
            return
        from app import check_geometry
        try:
            if key == "speaker_roi" and self.config.get("body_roi"):
                check_geometry(frame, self.config)
            dialog = RegionDialog(frame, "台词正文" if key == "body_roi" else "角色名", self)
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                updates = {key: dialog.roi, "reference_size": [frame.shape[1], frame.shape[0]],
                           "calibrated": bool(dialog.roi if key == "body_roi" else self.config.get("body_roi")),
                           "game_title": window.title}
                previous = self.config.get("reference_size")
                if key == "body_roi" and previous and abs(frame.shape[1]/frame.shape[0]/(previous[0]/previous[1])-1) > .015:
                    updates["speaker_roi"] = None
                self.config = save_ocr_settings(self.root, updates)
                self.update_region_note()
                self.status.setText("识别区域已保存。可以开始采集。")
        except Exception as exc:
            self.status.setText("区域未保存：" + str(exc))

    def start_collection(self):
        if self.busy():
            return
        try:
            window = self.target()
            self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8-sig"))
            if not self.config.get("calibrated") or not self.config.get("body_roi"):
                raise ValueError("请先框选实际对话的正文区域")
            self.worker = self.make_worker(window)
            self.worker.message.connect(self.receive)
            self.worker.finished.connect(self.finished_work)
            self.set_busy(True)
            self.worker.start()
        except Exception as exc:
            self.status.setText(str(exc))

    def make_worker(self, window):
        return CollectionWorker(window, self.config, self.root, self)

    def receive(self, event):
        if "status_text" in event and not self.closing:
            self.status.setText(event["status_text"])
        if "error" in event:
            self.status.setText("识别已停止：" + event["error"])
        if "session" in event:
            self.latest_observation = self.latest_record = None
            self.current.clear()
        if "observation" in event:
            self.latest_observation = event["observation"]
            self.show_current()
        if "record" in event:
            record = event["record"]
            identity = (event["source"], record["dialogue_id"])
            self.received[identity] = record
            self.latest_record = record
            self.show_current()
            self.count.setText(f"本次已采集 {len(self.received)} 条，复核修订会更新同一行。")
        if "record" in event or "final_file" in event:
            self.collected.emit(event)

    def show_current(self):
        current = self.latest_observation if self.latest_observation is not None else self.latest_record
        if not current or not current.get("body"):
            self.current.clear()
            self.current.setPlaceholderText("当前画面未识别到台词；已采集的台词保留在项目中。")
            return
        from ocr_core import _text_key
        key = lambda record: (_text_key(record.get("body") or ""), _text_key(record.get("speaker") or ""))
        record = next((record for record in reversed(list(self.received.values())) if key(record) == key(current)), None)
        from dialogue_store import STATUS_LABELS
        state = STATUS_LABELS.get(record["status"], record["status"]) if record else "当前识别 · 等待保存或复核"
        self.current.setPlainText(f"{current.get('speaker') or '未识别角色'} · {state}\n{current['body']}")

    def show_preview(self):
        if self.worker and self.worker.capture:
            packet = self.worker.capture.peek()
            if packet is not None:
                self.preview.setPixmap(pixmap(packet.image).scaled(self.preview.width()-12, self.preview.height()-12,
                                                                  QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def stop_work(self):
        for worker in (self.worker, self.snapshot_worker):
            if worker and worker.isRunning():
                worker.cancel()
        self.stop.setEnabled(False)
        self.status.setText("正在停止识别并保存最后的记录…")

    def finished_work(self):
        if self.busy():
            return
        self.set_busy(False)
        if self.closing:
            self.close()

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.busy():
            self.closing = True
            self.stop_work()
            event.ignore()
            return
        self.timer.stop()
        event.accept()
