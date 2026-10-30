from wuwa_dubber.models import Utterance
from wuwa_dubber.playback_queue import PlaybackQueue


def test_queue_drops_oldest_pending_when_limit_exceeded() -> None:
    queue = PlaybackQueue(pending_limit=2)
    first = Utterance(text="一")
    second = Utterance(text="二")
    third = Utterance(text="三")
    assert queue.enqueue(first) == ()
    assert queue.enqueue(second) == ()
    assert queue.enqueue(third) == (first,)
    assert [item.text for item in queue.pending] == ["二", "三"]


def test_queue_jump_to_latest_records_skipped() -> None:
    queue = PlaybackQueue(pending_limit=5)
    queue.enqueue(Utterance(text="一"))
    queue.enqueue(Utterance(text="二"))
    queue.enqueue(Utterance(text="三"))
    latest = queue.jump_to_latest()
    assert latest is not None and latest.text == "三"
    assert [item.text for item in queue.skipped] == ["一", "二"]
    assert [item.text for item in queue.pending] == ["三"]


def test_zero_pending_limit_can_still_handoff_to_idle_player() -> None:
    queue = PlaybackQueue(pending_limit=0)
    first = Utterance(text="第一句")
    assert queue.enqueue(first) == ()
    assert queue.pop_next() is first
    dropped = Utterance(text="忙碌时不排队")
    assert queue.enqueue(dropped) == (dropped,)
    queue.finish_current()
    last = Utterance(text="空闲时可以读")
    assert queue.enqueue(last) == ()
    assert queue.pop_next() is last
