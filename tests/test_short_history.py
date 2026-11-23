import numpy as np

def test_short_history_override_scores_before_checkpoint_minimum(tmp_path, monkeypatch):
    import json
    from motionguard.config import load_config
    from motionguard.video import infer

    cfg = load_config()
    cfg["min_history"] = 8
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
    for minimum in (None, 3):
        out = tmp_path / str(minimum)
        infer("fixture", "unused.pt", out, min_history=minimum)
        rows = [json.loads(line) for line in (out/"scores.jsonl").read_text().splitlines()]
        assert rows[1]["scores"] is None
        assert (rows[2]["scores"] is not None) == (minimum == 3)
    import pytest
    with pytest.raises(ValueError, match="min_history"):
        infer("fixture", "unused.pt", tmp_path, min_history=0)
