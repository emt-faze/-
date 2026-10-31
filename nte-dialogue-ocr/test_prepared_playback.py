import hashlib
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'wuwa-dubber/src'))
from dialogue_generation import GenerationProject
from prepared_playback import PreparedLibrary, PreparedPlayback, text_key


def completed_line(project, text, speaker='伊洛伊', voice=None, sample=1000):
    row = project.add_dialogue(text, speaker, 'character:'+(voice or speaker))
    path = project.directory/'audio'/(row.id+'.wav')
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), 'wb') as out:
        out.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        out.writeframes(int(sample).to_bytes(2,'little',signed=True)*2400)
    row.status = 'done'
    row.audio = dict(file=path.relative_to(project.directory).as_posix(), voice=voice or speaker,
                     input_key=row.input_key(project.mode), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return row, path


class Player:
    def __init__(self):
        self.played, self.cancelled = [], []
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0

    def play(self, path, *, cancel_event, on_started=None):
        if cancel_event.is_set():
            return
        if on_started:
            on_started()
        self.played.append(path)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            while not self.release.wait(.005) and not cancel_event.is_set():
                pass
            if cancel_event.is_set():
                self.cancelled.append(path)
        finally:
            self.active -= 1


class PreparedPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = GenerationProject.create(self.root, mode='character')
        self.first, self.first_path = completed_line(self.project, '下午好，准备好了吗？')
        self.second, self.second_path = completed_line(self.project, '我们现在出发吧。', sample=2000)
        self.player, self.events = Player(), []
        self.now = 100.
        self.controllers = []

    def tearDown(self):
        self.player.release.set()
        for controller in self.controllers:
            self.assertTrue(controller.close(2))
        self.temp.cleanup()

    def controller(self, **options):
        controller = PreparedPlayback(PreparedLibrary(self.project, {'奥涅伊洛伊':'伊洛伊'}), self.player,
                                      self.events.append, clock=lambda:self.now, **options)
        self.controllers.append(controller)
        return controller

    def observe(self, controller, text=None, **values):
        self.now += .1
        event = dict(body=self.first.text if text is None else text, speaker='伊洛伊',
                     body_confidence=.99, speaker_confidence=.99, captured_monotonic=self.now)
        event.update(values)
        controller.observe(event)
        return event

    def stable(self, controller, text=None, **values):
        self.observe(controller, text, **values)
        return self.observe(controller, text, **values)

    def wait(self, condition):
        until = time.monotonic()+3
        while not condition() and time.monotonic()<until:
            time.sleep(.005)
        self.assertTrue(condition())

    def test_matching_ignores_punctuation_spaces_but_not_missing_words(self):
        library = PreparedLibrary(self.project)
        line, _ = library.match('下午好\n 准备好了吗!', '伊洛伊')
        self.assertEqual(line.row_id, self.first.id)
        for text in ('下午好', '下午好准备了吗', '下午不好准备好了吗'):
            self.assertIsNone(library.match(text, '伊洛伊')[0])

    def test_deleted_unreviewed_edited_and_damaged_rows_are_excluded(self):
        self.project.delete_rows([self.first.id])
        self.second.confirmed = False
        self.assertFalse(PreparedLibrary(self.project).lines)
        self.second.confirmed = True
        self.second_path.write_bytes(b'corrupt')
        self.assertFalse(PreparedLibrary(self.project).lines)

    def test_duplicate_text_uses_character_voice_and_alias_without_guessing(self):
        other, _ = completed_line(self.project, self.first.text, '误识别角色', voice='明音凛', sample=3000)
        library = PreparedLibrary(self.project, {'奥涅伊洛伊':'伊洛伊'})
        self.assertIsNone(library.match(self.first.text)[0])
        self.assertEqual(library.match(self.first.text, '奥涅伊洛伊')[0].row_id, self.first.id)
        self.assertEqual(library.match(self.first.text, '明音凛')[0].row_id, other.id)
        self.assertIsNone(library.match(self.first.text, '娜娜莉')[0])

    def test_multiple_takes_of_same_voice_remain_ambiguous(self):
        completed_line(self.project, self.first.text, sample=3000)
        self.assertIsNone(PreparedLibrary(self.project).match(self.first.text, '伊洛伊')[0])

    def test_two_fresh_observations_play_once_without_synthesis(self):
        controller = self.controller()
        with patch('wuwa_dubber.tts.client.create_tts_client', side_effect=AssertionError('No model calls')):
            event = self.observe(controller)
            controller.observe(event)
            self.assertEqual(self.player.played, [])
            self.observe(controller)
            self.wait(lambda:len(self.player.played)==1)
            self.player.release.set()
            self.wait(lambda:self.player.active==0)
            self.stable(controller)
            self.assertEqual(self.player.played, [self.first_path])

    def test_full_match_from_render_stable_capture_starts_without_a_second_ocr_pass(self):
        controller=self.controller()
        event=self.observe(controller,render_stable=True)
        self.wait(lambda:len(self.player.played)==1)
        controller.observe(event)
        self.assertEqual(self.player.played,[self.first_path])

    def test_subtitle_change_interrupts_previous_before_new_audio(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:self.player.active==1)
        self.observe(controller, self.second.text)
        self.wait(lambda:self.first_path in self.player.cancelled)
        self.observe(controller, self.second.text)
        self.wait(lambda:len(self.player.played)==2)
        self.assertEqual(self.player.played[-1], self.second_path)
        self.assertEqual(self.player.max_active, 1)

    def test_blank_or_ellipsis_stops_audio_and_allows_later_reappearance(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:len(self.player.played)==1)
        self.observe(controller, '……', body_confidence=0)
        self.wait(lambda:self.player.active==0)
        self.stable(controller)
        self.wait(lambda:len(self.player.played)==2)

    def test_unknown_caption_stops_current_without_enqueuing_speech(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:self.player.active==1)
        self.stable(controller, '尚未生成的另一句。')
        self.wait(lambda:self.player.active==0)
        self.assertEqual(len(self.player.played), 1)
        self.assertTrue(any(e['state']=='unmatched' for e in self.events))

    def test_stale_out_of_order_low_confidence_and_cropped_frames_never_play(self):
        for changes in ({'captured_monotonic':80.}, {'body_confidence':float('nan')},
                        {'body_confidence':.5}, {'source_clipped':True}, {'coverage_warning':True}):
            controller = self.controller()
            self.stable(controller, **changes)
            self.assertEqual(self.player.played, [])
            controller.close(1)

    def test_render_change_rejects_previous_frames_and_late_results(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:self.player.active==1)
        controller.invalidate(captured_at=self.now+.5)
        self.wait(lambda:self.player.active==0)
        self.observe(controller, self.second.text)
        self.observe(controller, self.second.text)
        self.assertEqual(len(self.player.played), 1)
        self.now += 1
        self.stable(controller, self.second.text)
        self.wait(lambda:len(self.player.played)==2)

    def test_only_latest_pending_line_survives_slow_file_loading(self):
        third, third_path = completed_line(self.project, '已经到下一页了。', sample=3000)
        controller = self.controller()
        entered, release = threading.Event(), threading.Event()
        original = Path.read_bytes
        def slow(path):
            if path == self.first_path:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test timed out')
            return original(path)
        try:
            with patch.object(Path, 'read_bytes', slow):
                self.stable(controller)
                self.assertTrue(entered.wait(2))
                self.stable(controller, self.second.text)
                self.stable(controller, third.text)
                release.set()
                self.wait(lambda:bool(self.player.played))
                self.assertEqual(self.player.played, [third_path])
        finally:
            release.set()

    def test_pause_tracks_latest_and_resume_plays_current_page(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:self.player.active==1)
        controller.pause()
        self.stable(controller, self.second.text)
        self.wait(lambda:self.player.active==0)
        self.assertEqual(len(self.player.played), 1)
        controller.pause(False)
        self.wait(lambda:len(self.player.played)==2)
        self.assertEqual(self.player.played[-1], self.second_path)

    def test_skip_does_not_repeat_until_explicit_replay(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:self.player.active==1)
        controller.skip()
        self.wait(lambda:self.player.active==0)
        self.stable(controller)
        self.assertEqual(len(self.player.played), 1)
        controller.replay()
        self.wait(lambda:len(self.player.played)==2)

    def test_unavailable_capture_stops_and_requires_new_observations(self):
        controller = self.controller()
        old = self.stable(controller)
        self.wait(lambda:self.player.active==1)
        controller.invalidate(unavailable=True)
        self.wait(lambda:self.player.active==0)
        controller.observe(old)
        self.assertEqual(len(self.player.played), 1)
        self.stable(controller)
        self.wait(lambda:len(self.player.played)==2)

    def test_audio_changed_after_indexing_is_rejected_without_deletion(self):
        controller = self.controller()
        self.first_path.write_bytes(b'changed')
        self.stable(controller)
        self.wait(lambda:any(e['state']=='error' for e in self.events))
        self.assertEqual(self.player.played, [])
        self.assertEqual(self.first_path.read_bytes(), b'changed')

    def test_slow_file_loading_cannot_start_expired_audio(self):
        controller = self.controller()
        original = Path.read_bytes
        def slow(path):
            result = original(path)
            self.now += 10
            return result
        with patch.object(Path, 'read_bytes', slow):
            self.stable(controller)
            self.wait(lambda:any(e['state']=='stale' for e in self.events))
        self.assertEqual(self.player.played, [])
        self.stable(controller)
        self.wait(lambda:len(self.player.played)==1)

    def test_duplicate_frame_notification_does_not_erase_confirmation(self):
        controller = self.controller()
        controller.invalidate(captured_at=self.now)
        self.observe(controller)
        controller.invalidate(captured_at=self.now-.1)
        self.observe(controller)
        self.wait(lambda:len(self.player.played)==1)

    def test_close_cancels_active_and_ignores_future_observations(self):
        controller = self.controller()
        self.stable(controller)
        self.wait(lambda:self.player.active==1)
        self.assertTrue(controller.close(1))
        self.stable(controller, self.second.text)
        self.assertEqual(self.player.played, [self.first_path])

    def test_cancelled_before_audio_start_is_not_marked_played(self):
        controller = self.controller()
        entered,release=threading.Event(),threading.Event()
        original=Path.read_bytes
        calls=[]
        def delayed(path):
            calls.append(path)
            if len(calls)==1:
                entered.set()
                release.wait(3)
            return original(path)
        try:
            with patch.object(Path,'read_bytes',delayed):
                self.stable(controller)
                self.assertTrue(entered.wait(2))
                self.assertFalse(any(e['state']=='playing' for e in self.events))
                controller.invalidate(captured_at=self.now)
                self.stable(controller)
                release.set()
                self.wait(lambda:len(self.player.played)==1)
            self.stable(controller)
            self.assertEqual(self.player.played,[self.first_path])
            self.assertEqual(sum(e['state']=='playing' for e in self.events),1)
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main(verbosity=2)
