from motionguard.config import load_config
from motionguard.events import EventManager


def test_confirmed_fall_resumes_only_on_same_id():
    cfg = dict(load_config(), fall_pose_grace_sec=.27, confirm_sec=[.06]*7)
    manager = EventManager(cfg)
    high = [.99, 0, 0, 0, 0, 0, 0]
    manager.update(1, 0, high)
    manager.update(1, .067, high)
    manager.interrupt(1)
    assert not manager.update(1, .267, high, adequate=False)
    assert manager.active(1) == ["fall"]
    manager.update(2, .267, high)
    assert not manager.active(2)
    manager.update(1, .30, high)
    closed = manager.flush()
    assert len(closed) == 1
    assert closed[0]["detected_time_sec"] == .067
    assert closed[0]["end_time_sec"] == .30


def test_long_gap_closes_fall_and_short_gap_breaks_unconfirmed_evidence():
    cfg = dict(load_config(), fall_pose_grace_sec=.27, confirm_sec=[.06]*7)
    manager = EventManager(cfg)
    high = [.99, 0, 0, 0, 0, 0, 0]
    manager.update(1, 0, high)
    manager.update(1, .04, high, adequate=False)
    manager.update(1, .08, high)
    assert not manager.active(1)
    manager.update(1, .15, high)
    manager.interrupt(1)
    closed = manager.update(1, .43, high)
    assert closed[0]["end_time_sec"] == .15
    assert closed[0]["end_reason"] == "observation_lost"
    assert not manager.active(1)
