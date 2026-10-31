"""Prepared-project playback using the existing game capture and OCR dialog."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from dialogue_dubbing import voice_paths
from generator_ocr import OcrCollectionDialog, CollectionWorker
from prepared_playback import PreparedLibrary, PreparedPlayback, text_key


def playback_settings(options):
    _, data = voice_paths(options)
    from wuwa_dubber.config import AppConfig
    config = AppConfig.from_dict(json.loads((data/'config.json').read_text(encoding='utf-8-sig')))
    aliases = {}
    conflicts = set()
    for path in (data/'profiles').glob('*/profile.json'):
        try:
            profile = json.loads(path.read_text(encoding='utf-8-sig'))
            canonical = text_key(profile['speaker'])
            for name in [profile['speaker'], *profile.get('aliases', [])]:
                key = text_key(name)
                if key in aliases and aliases[key] != canonical:
                    conflicts.add(key)
                aliases[key] = canonical
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Playing a saved WAV does not require a valid reference clip.
    for key in conflicts:
        aliases.pop(key, None)
    return config, aliases


class PreparedPlaybackDialog(OcrCollectionDialog):
    playback_event = QtCore.Signal(object)

    def __init__(self, root, project, options, parent=None):
        self.project, self.options = project, dict(options)
        self.controller = self.library = self.player = None
        self.accept_playback = False
        self.last_generation = -1
        self.played = 0
        self.unmatched = set()
        self.last_played = None
        super().__init__(root, parent)
        self.setWindowTitle('自动识别字幕并播放 · 当前项目')
        self.resize(960, 780)
        self.note.setText('先在项目中生成台词音频，再进入游戏对话画面并开始。识别到完整字幕后播放对应音频。\n'
                          '字幕切换会中断上一句，同一句持续显示只播放一次。未匹配的台词会提示并保留识别记录。')
        self.project_note = QtWidgets.QLabel('播放项目：'+str(project.directory))
        self.project_note.setWordWrap(True)
        self.layout().insertWidget(1, self.project_note)
        self.playback_status = QtWidgets.QLabel('播放尚未开始。仅使用本项目已完成且仍有效的音频。')
        self.playback_status.setWordWrap(True)
        self.playback_status.setStyleSheet('color:#126d66; font-weight:600; padding:5px;')
        self.layout().insertWidget(self.layout().count()-1, self.playback_status)
        controls = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton('暂停自动播放')
        self.pause_button.setCheckable(True)
        self.pause_button.setToolTip('继续后，从当前字幕对应音频的开头播放。')
        self.pause_button.toggled.connect(self.pause_playback)
        self.skip_button = QtWidgets.QPushButton('跳过当前句')
        self.skip_button.clicked.connect(lambda: self.controller.skip() if self.controller else None)
        self.replay_button = QtWidgets.QPushButton('重播当前句')
        self.replay_button.clicked.connect(lambda: self.controller.replay() if self.controller else None)
        for button in (self.pause_button, self.skip_button, self.replay_button):
            button.setEnabled(False)
            controls.addWidget(button)
        controls.addStretch()
        self.layout().insertLayout(self.layout().count()-1, controls)
        self.start.setText('开始识别并播放')
        self.stop.setText('停止识别和播放')
        self.current.setPlaceholderText('识别到的当前字幕与匹配结果会显示在这里')
        self.playback_event.connect(self.receive_playback)
        self.monitor = QtCore.QTimer(self)
        self.monitor.timeout.connect(self.monitor_playback)
        self.monitor.start(50)
        self.update_count()

    def busy(self):
        return super().busy() or bool(self.controller and self.controller.alive)

    def start_collection(self):
        if self.busy():
            return
        try:
            config, aliases = playback_settings(self.options)
            self.library = PreparedLibrary(self.project, aliases)
            self.update_count()
            if not self.library.lines:
                raise ValueError('当前项目没有可播放的完整音频，请返回列表核对并生成台词')
            import sounddevice as sd
            from wuwa_dubber.audio.player import AudioPlayer
            sd.check_output_settings(device=config.output_device, channels=1)
            self.player = AudioPlayer(config.output_device, volume=config.output_volume, latency='low')
            self.last_generation = -1
            self.played, self.unmatched = 0, set()
            self.accept_playback = True
            super().start_collection()
            if not self.worker or not self.worker.isRunning():
                self.finished_work()
        except Exception as exc:
            self.accept_playback = False
            self.status.setText('未能开始：'+str(exc))

    def make_worker(self, window):
        self.controller = PreparedPlayback(self.library, self.player, self.playback_event.emit)
        try:
            worker = CollectionWorker(window, self.config, self.root, self, playback_events=True)
        except Exception:
            self.controller.close(0)
            raise
        for button in (self.pause_button, self.skip_button, self.replay_button):
            button.setEnabled(True)
        self.playback_status.setText('等待新字幕，匹配后直接播放本地音频。')
        return worker

    def receive(self, event):
        super().receive(event)
        self.update_count()
        if not self.accept_playback or not self.controller:
            return
        if 'error' in event:
            self.stop_work()
            self.status.setText('识别已停止：'+event['error'])
            return
        if event.get('capture_state') == 'unavailable':
            self.controller.invalidate(unavailable=True)
        if 'frame_changed' in event:
            self.controller.invalidate(captured_at=event['frame_changed'])
        if 'observation' in event:
            self.controller.observe(event['observation'])
        # Persisted revisions and stop-flush records can describe an old page.
        # Only fresh frame observations above may start playback.

    def receive_playback(self, event):
        if (not self.accept_playback or event['generation'] < self.last_generation
                or (self.controller and event['generation'] < self.controller.generation)):
            return
        self.last_generation = event['generation']
        self.playback_status.setText(event['message'])
        if event['state'] == 'playing':
            self.played += 1
            self.last_played = event['row_id']
        elif event['state'] == 'unmatched':
            self.unmatched.add(text_key(event.get('recognized')))
        self.update_count()

    def update_count(self):
        available = len(self.library.lines) if self.library else sum(
            row.status == 'done' and not row.review_reason() and bool(self.project.audio_path(row, verify=False))
            for row in self.project.rows)
        self.count.setText(f'当前项目可播放 {available} 条 · 本次播放 {self.played} 次 · 未匹配 {len(self.unmatched)} 句')

    def pause_playback(self, paused):
        self.pause_button.setText('继续自动播放' if paused else '暂停自动播放')
        if self.controller:
            self.controller.pause(paused)

    def stop_work(self):
        self.accept_playback = False
        if self.controller:
            self.controller.stop()
        for button in (self.pause_button, self.skip_button, self.replay_button):
            button.setEnabled(False)
        super().stop_work()
        self.status.setText('正在停止识别和播放…')
        self.playback_status.setText('播放已停止。项目中的台词和音频保留。')

    def finished_work(self):
        if self.worker and not self.worker.isRunning() and self.controller:
            self.accept_playback = False
            self.controller.stop()
        if not self.busy():
            self.accept_playback = False
            for button in (self.pause_button, self.skip_button, self.replay_button):
                button.setEnabled(False)
            self.pause_button.blockSignals(True)
            self.pause_button.setChecked(False)
            self.pause_button.setText('暂停自动播放')
            self.pause_button.blockSignals(False)
            if self.status.text() == '正在停止识别和播放…':
                self.status.setText('识别和播放已停止，可以返回台词列表或重新开始。')
        super().finished_work()

    def monitor_playback(self):
        if self.controller and (not self.worker or not self.worker.isRunning()):
            self.finished_work()

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted():
            self.accept_playback = False
            self.monitor.stop()
