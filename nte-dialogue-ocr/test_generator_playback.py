import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets

from capture_session import FramePacket
from capture_windows import GameWindow
from dialogue_generation import GenerationProject
from generator_ocr import CollectionWorker, PlaybackFrameWatch
from generator_playback import PreparedPlaybackDialog
from test_prepared_playback import Player, completed_line
from wuwa_dubber.config import AppConfig


class PreparedPlaybackUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(calibrated=True, body_roi=[.1,.8,.9,.9], capture_backend='wgc',
                           dubbing={'enabled':True,'mode':'character'})
        (self.root/'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        self.before_config = (self.root/'config.json').read_bytes()
        self.project = GenerationProject.create(self.root/'projects', mode='character')
        self.row, self.path = completed_line(self.project, '下午好，准备好了吗？')
        self.project.save()
        self.before_project = (self.project.directory/'project.json').read_bytes()
        self.game = GameWindow(123, '异环', 'UnrealWindow', 1920, 1080, 55, r'D:\HTGame.exe')
        self.commands = queue.Queue()
        self.player = Player()
        self.calls = 0
        self.release_stop = threading.Event()
        self.release_stop.set()
        def runner(window, config, stop, callback, **hooks):
            self.calls += 1
            while not stop.wait(.01):
                try:
                    kind, value = self.commands.get_nowait()
                except queue.Empty:
                    continue
                hooks[kind](value)
            self.release_stop.wait(3)
            # A stop-flush revision must not restart audio.
            hooks['on_record'](dict(dialogue_id='old', revision=1, status='verified',
                                    body=self.row.text, speaker=self.row.speaker))
        self.patches = [
            patch('capture_windows.list_game_windows', return_value=[self.game]),
            patch('generator_playback.playback_settings', return_value=(AppConfig(), {})),
            patch('generator_playback.CollectionWorker', side_effect=lambda *a,**kw:CollectionWorker(*a,runner=runner,**kw)),
            patch('sounddevice.check_output_settings'),
            patch('wuwa_dubber.audio.player.AudioPlayer', return_value=self.player),
            patch('wuwa_dubber.tts.client.create_tts_client', side_effect=AssertionError('Playback must not load TTS')),
        ]
        for item in self.patches:
            item.start()
        self.dialog = PreparedPlaybackDialog(self.root, self.project, {})
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self):
        self.release_stop.set()
        self.dialog.close()
        self.wait(lambda:not self.dialog.busy())
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def wait(self, condition):
        end = time.monotonic()+4
        while not condition() and time.monotonic()<end:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertTrue(condition())

    def observe(self, text=None):
        self.commands.put(('on_observation', dict(body=text or self.row.text, speaker=self.row.speaker,
                           body_confidence=.99, speaker_confidence=.99, captured_monotonic=time.monotonic())))
        time.sleep(.015)

    def play(self):
        self.dialog.start.click()
        self.wait(lambda:self.calls>0)
        self.observe()
        self.observe()
        self.wait(lambda:bool(self.player.played))

    def test_start_pause_skip_replay_stop_without_model_or_data_changes(self):
        self.play()
        self.assertFalse(self.dialog.start.isEnabled())
        self.assertIn('1 次', self.dialog.count.text())
        self.dialog.pause_button.click()
        self.wait(lambda:self.player.active==0)
        self.observe()
        self.assertEqual(len(self.player.played), 1)
        self.dialog.pause_button.click()
        self.wait(lambda:len(self.player.played)==2)
        self.dialog.skip_button.click()
        self.wait(lambda:self.player.active==0)
        self.observe()
        self.observe()
        self.dialog.replay_button.click()
        self.wait(lambda:len(self.player.played)==3)
        self.dialog.stop.click()
        self.wait(lambda:not self.dialog.busy() and self.dialog.start.isEnabled())
        self.assertTrue(self.dialog.start.isEnabled())
        self.assertFalse(self.dialog.pause_button.isEnabled())
        self.assertEqual((self.root/'config.json').read_bytes(), self.before_config)
        self.assertEqual((self.project.directory/'project.json').read_bytes(), self.before_project)

    def test_persisted_old_revision_cannot_play(self):
        self.dialog.start.click()
        self.wait(lambda:self.calls>0)
        self.commands.put(('on_record', dict(dialogue_id='old', revision=2, status='verified',
                                            body=self.row.text, speaker=self.row.speaker)))
        self.wait(lambda:bool(self.dialog.received))
        self.assertEqual(self.player.played, [])

    def test_close_waits_for_capture_flush_and_never_replays(self):
        self.play()
        self.release_stop.clear()
        self.dialog.close()
        self.assertTrue(self.dialog.closing)
        self.assertTrue(self.dialog.isVisible())
        self.wait(lambda:self.player.active==0)
        self.release_stop.set()
        self.wait(lambda:not self.dialog.isVisible())
        self.assertEqual(len(self.player.played), 1)

    def test_missing_audio_does_not_start_capture(self):
        self.path.unlink()
        self.dialog.start.click()
        self.assertEqual(self.calls, 0)
        self.assertFalse(self.dialog.busy())
        self.assertIn('没有可播放', self.dialog.status.text())

    def test_capture_error_stops_audio_and_is_retryable(self):
        self.play()
        self.dialog.receive({'error':'画面连接已断开'})
        self.wait(lambda:not self.dialog.busy() and self.dialog.start.isEnabled())
        self.assertEqual(self.player.active, 0)
        self.assertIn('画面连接已断开', self.dialog.status.text())
        self.dialog.start.click()
        self.wait(lambda:self.calls==2)
        self.observe()
        self.observe()
        self.wait(lambda:len(self.player.played)==2)

    def test_stale_audio_ui_event_is_ignored_after_page_change(self):
        self.play()
        old = self.dialog.controller.generation
        self.dialog.receive({'frame_changed':time.monotonic()})
        self.app.processEvents()
        self.dialog.receive_playback(dict(state='completed', generation=old, message='过期的播放结果'))
        self.assertNotIn('过期', self.dialog.playback_status.text())


class FrameWatchTests(unittest.TestCase):
    def test_page_watch_runs_independently_and_stops_on_capture_loss(self):
        class Capture:
            packet = None
            error = False
            def ready_for_ocr(self):
                if self.error:
                    raise RuntimeError('capture lost')
                return self.packet is not None
            def peek(self):
                return self.packet
        capture, events = Capture(), []
        watch = PlaybackFrameWatch(capture, dict(body_roi=[0,0,1,1],render_stable_seconds=.45), events.append)
        first = np.zeros((30,100), bool)
        first[5:20,5:25] = True
        second = np.zeros_like(first)
        second[5:20,50:80] = True
        with patch('frame_gate.glyph_mask', side_effect=lambda image:image):
            capture.packet = FramePacket(0,.1,first,(100,30),(0,0),(0,0,100,30),0)
            watch.arm(first,dict(body='上一句',body_confidence=.99,captured_monotonic=.1,
                                body_lines=[dict(bbox=[[5,5],[25,5],[25,20],[5,20]])]),(0,0))
            watch.start()
            try:
                for i, mask in enumerate([first]*4+[second]*4, 1):
                    capture.packet = FramePacket(i, i*.2, mask, (100,30), (0,0), (0,0,100,30), 0)
                    time.sleep(.07)
                changes = [e['frame_changed'] for e in events if 'frame_changed' in e]
                self.assertEqual(changes, [1.0])
                capture.packet = None
                time.sleep(.08)
                self.assertIn({'capture_state':'unavailable'}, events)
                capture.error = True
                time.sleep(.08)
                self.assertTrue(any('error' in e for e in events))
            finally:
                watch.close()
        self.assertFalse(watch.thread.is_alive())


if __name__ == '__main__':
    unittest.main(verbosity=2)
