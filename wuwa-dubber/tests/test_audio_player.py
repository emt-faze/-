from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import patch
import wave

from wuwa_dubber.audio.player import AudioPlayer


def test_started_notification_and_abort_on_cancellation(tmp_path:Path):
    path=tmp_path/'audio.wav'
    with wave.open(str(path),'wb') as out:
        out.setparams((1,2,24000,0,'NONE','not compressed'))
        out.writeframes(b'\x10\x00'*2400)
    events=[]
    stream=SimpleNamespace(active=False)
    def play(*args,**kwargs):
        assert kwargs['latency']=='low'
        stream.active=True
        events.append('output_started')
    def abort():
        stream.active=False
        events.append('abort_buffer')
    stream.abort=abort
    device=SimpleNamespace(play=play,get_stream=lambda:stream,stop=lambda:events.append('close_stream'))
    cancel=threading.Event()
    def started():
        events.append('notified_started')
        cancel.set()
    with patch.dict('sys.modules',sounddevice=device):
        AudioPlayer(latency='low').play(path,cancel_event=cancel,on_started=started)
    assert events[-4:]==['output_started','notified_started','abort_buffer','close_stream']


def test_cancelled_before_play_does_not_report_started(tmp_path:Path):
    cancel=threading.Event()
    cancel.set()
    events=[]
    AudioPlayer().play(tmp_path/'missing.wav',cancel_event=cancel,on_started=lambda:events.append('started'))
    assert not events
