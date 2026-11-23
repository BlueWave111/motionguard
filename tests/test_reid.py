import numpy as np
from motionguard.reid import TrackReidentifier

def test_reidentification_requires_unambiguous_appearance_and_location():
    frame = np.zeros((100, 160, 3), np.uint8)
    frame[:, :65] = (0, 0, 255)
    frame[:, 65:110] = (0, 255, 0)
    frame[:, 110:] = (0, 0, 255)
    a, b, far = (np.array(x) for x in ([12, 12, 50, 85], [66, 12, 103, 85], [112, 12, 150, 85]))
    tracker = TrackReidentifier(.27)
    tracker.step(frame, [(10, a, None), (11, b, None)], 0.)
    tracker.step(frame, [(11, b, None)], .067)
    poses, links = tracker.step(frame, [(12, a, None), (11, b, None), (13, far, None)], .134)
    assert [p[0] for p in poses] == [1, 2, 3]
    assert [(raw, pid) for raw, pid, _ in links] == [(12, 1)]
    poses, _ = tracker.step(frame, [(10, a, None), (12, a, None)], .2)
    assert len({p[0] for p in poses}) == 2


def test_reidentification_rejects_ambiguity_and_expired_tracks():
    frame = np.full((100, 100, 3), 80, np.uint8)
    a = np.array([10, 10, 50, 90])
    b = np.array([20, 10, 60, 90])
    tracker = TrackReidentifier(.27)
    tracker.step(frame, [(1, a, None), (2, b, None)], 0.)
    poses, links = tracker.step(frame, [(3, a, None)], .133)
    assert not links and poses[0][0] == 3
    poses, links = tracker.step(frame, [(3, a, None)], .5)
    assert not links and poses[0][0] == 4

def test_inference_retains_history_across_reidentified_raw_id(tmp_path, monkeypatch):
    import json
    from motionguard.config import load_config
    from motionguard.video import infer

    cfg = load_config()
    cfg["min_history"] = 3
    box = np.array([10, 10, 65, 95])
    points = np.column_stack((np.linspace(12, 60, 17), np.linspace(12, 90, 17), np.ones(17)))

    class Pose:
        def __init__(self, cfg):
            self.index = 0

        def step(self, frame):
            step = self.index
            self.index += 1
            if step in (3, 4):
                return []
            return [(10 if step < 3 else 22, box, points)]

    class Predictor:
        def __init__(self, *args):
            self.cfg = dict(cfg)
            self.artifact = {}

        def predict(self, *args):
            return np.zeros(7)

    def frames(*args):
        for step in range(6):
            yield step/15, np.full((120, 160, 3), (0, 0, 255), np.uint8)

    monkeypatch.setattr("motionguard.video.frames", frames)
    monkeypatch.setattr("motionguard.video.PoseStream", Pose)
    monkeypatch.setattr("motionguard.train.Predictor", Predictor)
    for enabled in (False, True):
        out = tmp_path / str(enabled)
        infer("fixture", "unused.pt", out, pose_gap_sec=.27, reid_gap_sec=.27 if enabled else None)
        rows = [json.loads(line) for line in (out/"scores.jsonl").read_text().splitlines()]
        resumed = next(row for row in rows if row["time_sec"] == 5/15 and row["bbox_xyxy"])
        assert (resumed["scores"] is not None) == enabled
        assert resumed["person_id"] == (1 if enabled else 22)
        if enabled:
            event = json.loads((out/"reid.jsonl").read_text())
            assert event["raw_id"] == 22 and event["person_id"] == 1
