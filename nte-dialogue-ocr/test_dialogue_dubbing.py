"""Event-to-audio integration: revisions, stale requests, export and voices."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import tempfile
import time
import unittest
import uuid
from unittest.mock import Mock, patch
import wave

from dialogue_dubbing import DubbingController, VoiceRuntime, join_audio, voice_paths

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'wuwa-dubber/src'))
from wuwa_dubber.config import AppConfig
from wuwa_dubber.models import VoiceProfile
from wuwa_dubber.speech_session import SpeechSession
from wuwa_dubber.speaker_matcher import SpeakerMatcher


def write_wave(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        output.writeframes(b'\0\0' * 2400)


class Client:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def synthesize(self, utterance, profile, path):
        self.calls.append((utterance.text, profile.speaker))
        self.entered.set()
        if self.block and len(self.calls) == 1:
            # Deliberately ignores cancellation: a late response must not play.
            if not self.release.wait(3):
                raise TimeoutError('test producer timed out')
        if self.fail:
            raise RuntimeError('test synthesis error')
        write_wave(path)
        return .01

    def cancel(self):
        pass


class Player:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, path, *, cancel_event):
        if not cancel_event.is_set():
            self.played.append(path)

    def stop(self):
        self.stopped += 1


class Runtime:
    select_profile = VoiceRuntime.select_profile

    def __init__(self, directory):
        self.config = AppConfig(tts_backend='windows-sapi', narrator_profile='通用旁白')
        self.profiles = {name: VoiceProfile(name) for name in ['伊洛伊', '娜娜莉', '通用旁白']}
        self.profiles['伊洛伊'].aliases = ['奥涅伊洛伊']
        self.matcher = SpeakerMatcher(list(self.profiles.values()))
        self.client, self.player = Client(), Player()
        self.cache_dir = directory / 'cache'

    def session(self):
        return SpeechSession(self.config, self.cache_dir, self.client, self.player)


def event(identity='one', **updates):
    return dict(dialogue_id=identity, body='你好。', speaker='伊洛伊', status='verified',
                speaker_status='verified', revision=1, body_confidence=.99,
                source_kind='automated_fixture') | updates


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dubbing-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.runtime = Runtime(self.directory)
        self.events = []
        self.controller = DubbingController(self.runtime, self.events.append)
        self.addCleanup(self.cleanup_controller)
        self.controller.set_directory(self.directory / 'records')

    def cleanup_controller(self):
        self.runtime.client.release.set()
        self.assertTrue(self.controller.close(4))

    def wait_for(self, condition):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.01)
        self.fail('Timed out waiting for audio pipeline')

    def completed(self):
        return [x['speech'] for x in self.events if x['speech']['state'] == 'completed']

    def test_review_same_id_and_persisted_event_play_once_and_export(self):
        self.controller.accept(event(status='candidate', revision=0))
        self.assertEqual(self.runtime.client.calls, [])
        self.controller.accept(event())
        self.controller.accept(event())
        self.controller.accept(event(revision=2, revision_reason='speaker_stabilized'))
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(self.runtime.client.calls, [('你好。', '伊洛伊')])
        report = self.completed()[0]
        self.assertEqual(report['source_kind'], 'automated_fixture')
        with wave.open(report['audio_path']) as audio:
            self.assertEqual(audio.getnframes(), 2400)
        self.assertTrue((self.directory / 'records/audio.jsonl').is_file())

    def test_alias_and_unknown_and_uncertain_names_do_not_inherit_previous(self):
        for record, name, fallback in [
            (event(speaker='奥涅伊洛伊'), '伊洛伊', False),
            (event(speaker='明音凛'), '通用旁白', True),
            (event(speaker='娜娜莉', speaker_status='needs_review'), '通用旁白', True),
            (event(speaker='', speaker_status='disabled'), '通用旁白', True),
        ]:
            actual, defaulted = self.runtime.select_profile(record)
            self.assertEqual((actual.speaker, defaulted), (name, fallback))

    def test_late_response_is_discarded_and_only_latest_pending_page_is_spoken(self):
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        began = time.monotonic()
        self.controller.accept(event('two', body='中间页。'))
        self.controller.accept(event('three', body='最后一页。'))
        self.controller.accept(event('one', revision=4))
        self.assertLess(time.monotonic() - began, .3)
        self.runtime.client.release.set()
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(self.completed()[0]['dialogue_id'], 'three')
        self.assertEqual([t for t, _ in self.runtime.client.calls], ['你好。', '最后一页。'])
        self.assertEqual(len(self.runtime.player.played), 1)
        self.assertFalse((self.directory / 'records/audio/one.wav').exists())

    def test_stop_suppresses_late_audio_and_future_events(self):
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        self.controller.stop()
        self.controller.accept(event('two'))
        self.runtime.client.release.set()
        self.assertTrue(self.controller.close(3))
        self.assertEqual(self.runtime.player.played, [])
        self.assertEqual(self.completed(), [])

    def test_new_candidate_cancels_previous_even_before_review_completes(self):
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        self.controller.accept(event('two', status='candidate', revision=0))
        self.runtime.client.release.set()
        self.wait_for(lambda: self.controller._active is None)
        self.assertEqual(self.runtime.player.played, [])
        self.controller.accept(event('two', body='下一句。'))
        self.wait_for(lambda: len(self.completed()) == 1)

    def test_blank_or_unavailable_capture_cancels_and_revision_does_not_replay(self):
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        self.controller.suspend()
        self.controller.accept(event(revision=3))
        self.runtime.client.release.set()
        self.wait_for(lambda: self.controller._active is None)
        self.assertEqual(self.runtime.player.played, [])

    def test_pause_continue_skips_old_page_and_reads_next(self):
        self.controller.pause(True)
        self.controller.accept(event())
        self.assertEqual(self.runtime.client.calls, [])
        self.controller.pause(False)
        self.controller.accept(event('two', body='继续。'))
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(self.runtime.client.calls, [('继续。', '伊洛伊')])

    def test_temporary_capture_loss_resumes_only_after_fresh_matching_observation(self):
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        self.controller.suspend(resume_when_visible=True)
        self.controller.suspend(resume_when_visible=True)
        self.controller.accept(event(revision=3))
        self.runtime.client.release.set()
        self.wait_for(lambda: self.controller._active is None)
        self.assertEqual(self.runtime.player.played, [])
        self.controller.confirm_visible({'body': '不同的台词', 'speaker': '伊洛伊'})
        self.assertEqual(len(self.runtime.client.calls), 1)
        self.controller.confirm_visible({'body': '你好。', 'speaker': '娜娜莉'})
        self.assertEqual(len(self.runtime.client.calls), 1)
        self.controller.confirm_visible({'body': '你好。', 'speaker': '伊洛伊'})
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(len(self.runtime.player.played), 1)
        self.assertEqual(sum(e['speech']['message'] == '画面暂不可用，已停止当前配音' for e in self.events), 1)

    def test_completed_caption_does_not_replay_on_capture_recovery(self):
        self.controller.accept(event())
        self.wait_for(lambda: len(self.completed()) == 1)
        self.wait_for(lambda: self.controller._active is None)
        self.controller.suspend(resume_when_visible=True)
        self.controller.confirm_visible({'body': '你好。', 'speaker': '伊洛伊'})
        self.assertEqual(len(self.runtime.client.calls), 1)
        self.assertEqual(self.controller._pending, None)

    def test_manual_skip_clears_capture_recovery(self):
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        self.controller.suspend(resume_when_visible=True)
        self.controller.skip()
        self.runtime.client.release.set()
        self.wait_for(lambda: self.controller._active is None)
        self.controller.confirm_visible({'body': '你好。', 'speaker': '伊洛伊'})
        self.assertEqual(self.runtime.player.played, [])

    def test_skip_before_review_suppresses_that_id(self):
        self.controller.accept(event(status='candidate', revision=0))
        self.controller.skip()
        self.controller.accept(event())
        self.controller.accept(event('two', body='下一句。'))
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(self.runtime.client.calls, [('下一句。', '伊洛伊')])

    def test_same_text_on_separate_page_reuses_shared_cache(self):
        self.controller.accept(event())
        self.wait_for(lambda: len(self.completed()) == 1)
        self.controller.accept(event('two'))
        self.wait_for(lambda: len(self.completed()) == 2)
        self.assertEqual(len(self.runtime.client.calls), 1)
        self.assertEqual(self.completed()[1]['cache_hits'], 1)
        self.assertEqual(len(self.runtime.player.played), 2)

    def test_failed_synthesis_can_be_retried_without_repeating_ocr(self):
        self.runtime.client.fail = True
        self.controller.accept(event())
        self.wait_for(lambda: any(e['speech']['state'] == 'error' for e in self.events))
        self.wait_for(lambda: self.controller._active is None)
        self.runtime.client.fail = False
        self.controller.retry()
        self.wait_for(lambda: len(self.completed()) == 1)

    def test_clipped_and_unverified_and_low_confidence_text_is_held(self):
        for index, change in enumerate([{'source_clipped': True}, {'coverage_warning': True},
                                        {'review_coverage_warning': True}, {'status': 'needs_review'},
                                        {'body_confidence': .7}]):
            self.controller.accept(event(f'held-{index}', **change))
        self.assertEqual(self.runtime.client.calls, [])

    def test_model_failure_uses_explicit_system_fallback_and_exports_actual_voice(self):
        self.runtime.fallback_enabled = True
        self.runtime.client.fail = True
        backup = Runtime(self.directory / 'backup')
        backup.player = self.runtime.player
        self.runtime.fallback_session = lambda: (VoiceProfile('系统中文朗读'), backup.session())
        self.controller.accept(event())
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(self.completed()[0]['voice'], '系统中文朗读')
        self.assertTrue(self.completed()[0]['fallback'])
        self.assertEqual(backup.client.calls, [('你好。', '系统中文朗读')])
        self.assertEqual(len(self.runtime.player.played), 1)
        self.assertTrue(any(e['speech']['state'] == 'fallback' for e in self.events))

    def test_external_cancellation_never_triggers_system_fallback(self):
        self.runtime.fallback_enabled = True
        self.runtime.fallback_session = Mock()
        self.runtime.client.block = True
        self.controller.accept(event())
        self.assertTrue(self.runtime.client.entered.wait(2))
        self.controller.stop()
        self.runtime.client.release.set()
        self.assertTrue(self.controller.close(3))
        self.runtime.fallback_session.assert_not_called()

    def test_unsafe_output_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.accept(event('../escape'))

    def test_candidate_policy_is_opt_in_and_still_rejects_clipping(self):
        self.controller.policy = 'candidate'
        self.controller.accept(event('bad', status='candidate', source_clipped=True))
        self.assertEqual(self.runtime.client.calls, [])
        self.controller.accept(event(status='candidate', speaker_status='candidate'))
        self.wait_for(lambda: len(self.completed()) == 1)
        self.assertEqual(self.runtime.client.calls, [('你好。', '通用旁白')])

    def test_export_cancel_does_not_publish_partial_wav(self):
        path = self.directory / 'source.wav'
        write_wave(path)
        stop = threading.Event()
        stop.set()
        output = self.directory / 'cancelled.wav'
        self.assertIsNone(join_audio([path], output, stop))
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.wav.tmp').exists())


class WiringTests(unittest.TestCase):
    def test_connected_runner_passes_real_ocr_callbacks_and_closes_on_error(self):
        import app
        runtime = Mock()
        controller = Mock()
        observed = []
        def loop(*args, **kwargs):
            kwargs['on_session'](Path('records/session'))
            kwargs['on_record'](event())
            kwargs['on_state']('blank')
            raise RuntimeError('capture lost')
        with patch('dialogue_dubbing.VoiceRuntime', return_value=runtime), \
             patch('dialogue_dubbing.DubbingController', return_value=controller), \
             patch('app.run_loop', side_effect=loop):
            with self.assertRaisesRegex(RuntimeError, 'capture lost'):
                app.run_connected(Mock(), {'dubbing': {'enabled': True}}, threading.Event(),
                                  observed.append, on_dubber=observed.append)
        runtime.prepare.assert_called_once()
        controller.accept.assert_called_once()
        controller.suspend.assert_called_once()
        controller.close.assert_called_once()
        self.assertIn(controller, observed)

    def test_stopping_during_model_start_never_starts_capture_or_audio(self):
        import app
        stop = threading.Event()
        runtime = Mock()
        runtime.prepare.side_effect = lambda *_: stop.set()
        with patch('dialogue_dubbing.VoiceRuntime', return_value=runtime), \
             patch('dialogue_dubbing.DubbingController') as controller, patch('app.run_loop') as loop:
            app.run_connected(Mock(), {'dubbing': {'enabled': True}}, stop, Mock())
        loop.assert_not_called()
        controller.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
