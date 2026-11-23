import json
import numpy as np
import pytest
cv2=pytest.importorskip('cv2')
from motionguard.config import load_config
from motionguard.events import EventManager
from motionguard.video import ClipStore,infer
from motionguard.demo import make_demo
from motionguard.train import fit_baseline


def test_clip_export_and_metadata(tmp_path):
    cfg=load_config(); cfg.update(sample_hz=10.,pre_roll_sec=.2,post_roll_sec=.2,max_clip_sec=1.)
    store=ClipStore(tmp_path,cfg)
    for i in range(5): store.push(i/10,np.zeros((48,64,3),np.uint8))
    m=EventManager(cfg); m.update(1,.2,[.9,0,0,0,0,0,0]); m.update(1,.4,[.9,0,0,0,0,0,0])
    state=m.states[(1,0)]; store.begin(1,0,state)
    event=m.forget(1)[0]; store.close_event(event)
    for i in range(5,9): store.push(i/10,np.zeros((48,64,3),np.uint8))
    store.finish()
    r=json.loads((tmp_path/'events.jsonl').read_text())
    assert r['post_roll_complete'] and (tmp_path/r['clip_path']).stat().st_size>0
    cap=cv2.VideoCapture(str(tmp_path/r['clip_path'])); ok,_=cap.read(); cap.release()
    assert ok


def test_video_pipeline_with_mock_pose(tmp_path,monkeypatch):
    # Real feature extraction, model, state handling, video I/O. Only YOLO is mocked.
    cfg=load_config(); cfg.update(window=32,min_history=4,stride=4)
    manifest=make_demo(tmp_path/'data',cfg); checkpoint=tmp_path/'model.joblib'
    fit_baseline(manifest,checkpoint,cfg)
    # This test substitutes the pose source; permit its local synthetic fixture only.
    import joblib
    a=joblib.load(checkpoint); a["demo_only"]=False; joblib.dump(a,checkpoint)
    video=tmp_path/'input.mp4'
    w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30.,(160,120))
    for i in range(60): w.write(np.zeros((120,160,3),np.uint8))
    w.release()
    class FakePose:
        def __init__(self,cfg): pass
        def step(self,frame):
            k=np.ones((17,3),np.float32); k[:,0]=np.linspace(40,90,17); k[:,1]=np.linspace(10,100,17); k[:,2]=.95
            return [(1,np.array([30,5,110,115]),k)]
    monkeypatch.setattr('motionguard.video.PoseStream',FakePose)
    infer(str(video),checkpoint,tmp_path/'output')
    rows=[json.loads(s) for s in (tmp_path/'output'/'scores.jsonl').read_text().splitlines()]
    assert len(rows)==30
    assert rows[0]['status']=='insufficient_data'
    assert any(r['scores'] is not None for r in rows)


def test_two_falls_keep_independent_ids_and_boxes(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.update(min_history=2, confirm_sec=[0.0]*7, pre_roll_sec=0.1, post_roll_sec=0.1)
    boxes = {1: [10, 10, 65, 100], 2: [90, 10, 145, 100]}

    class TestPredictor:
        def __init__(self, *args):
            self.cfg = cfg
            self.artifact = {}

        def predict(self, *args):
            return np.array([.95, 0, 0, 0, 0, 0, 0])

    class TestPose:
        def __init__(self, *args):
            self.index = 0

        def step(self, frame):
            ids = [2, 1] if self.index % 2 else [1, 2]
            if self.index >= 4:
                ids = [2]
            self.index += 1
            observations = []
            for person in ids:
                box = np.array(boxes[person])
                if self.index > 2:
                    box = box + [1, 0, 1, 0]
                points = np.column_stack((np.linspace(box[0], box[2], 17),
                                          np.linspace(12, 95, 17), np.ones(17)))
                observations.append((person, box, points))
            return observations

    def test_frames(*args):
        for step in range(12):
            yield step / cfg["sample_hz"], np.zeros((120, 160, 3), np.uint8)

    monkeypatch.setattr("motionguard.train.Predictor", TestPredictor)
    monkeypatch.setattr("motionguard.video.PoseStream", TestPose)
    monkeypatch.setattr("motionguard.video.frames", test_frames)
    infer("fixture", "unused.pt", tmp_path, camera_id="multi")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2
    by_person = {event["person_id"]: event for event in events}
    assert set(by_person) == {1, 2}
    assert by_person[1]["end_reason"] == "observation_lost"
    assert by_person[2]["end_reason"] == "stream_end"
    assert by_person[1]["detected_time_sec"] == by_person[2]["detected_time_sec"]
    for person, event in by_person.items():
        assert event["camera_id"] == "multi"
        assert event["bbox_xyxy_at_detection"] == boxes[person]
        assert (tmp_path / event["clip_path"]).stat().st_size > 0
    assert by_person[1]["clip_path"] != by_person[2]["clip_path"]


@pytest.mark.parametrize("missing_detection", [False, True])
def test_pose_gap_retains_history_only_for_same_track(tmp_path, monkeypatch, missing_detection):
    cfg = load_config()
    cfg.update(min_history=3)

    class TestPredictor:
        def __init__(self, *args):
            self.cfg = dict(cfg)
            self.artifact = {}

        def predict(self, *args):
            return np.zeros(7)

    class TestPose:
        def __init__(self, *args):
            self.index = 0

        def step(self, frame):
            step = self.index
            self.index += 1
            # The first person has a short pose gap, then returns with the
            # same ID. A later new ID must establish its own history.
            person = 1 if step < 7 else 3
            output = []
            for pid in (person, 2):
                x = 10 if pid != 2 else 90
                gap_frame = 3 <= step <= 5 and pid == 1
                if gap_frame and missing_detection:
                    continue
                confidence = .05 if gap_frame else 1.
                points = np.column_stack((np.linspace(x, x+50, 17),
                                          np.linspace(12, 95, 17), np.full(17, confidence)))
                output.append((pid, np.array([x, 10, x+55, 100]), points))
            return output

    def test_frames(*args):
        for step in range(9):
            yield step / cfg["sample_hz"], np.zeros((120,160,3), np.uint8)

    monkeypatch.setattr("motionguard.train.Predictor", TestPredictor)
    monkeypatch.setattr("motionguard.video.PoseStream", TestPose)
    monkeypatch.setattr("motionguard.video.frames", test_frames)
    for gap in (None, .27):
        folder = tmp_path / str(gap)
        infer("fixture", "unused.pt", folder, pose_gap_sec=gap)
        rows = [json.loads(line) for line in (folder/"scores.jsonl").read_text().splitlines()]
        rows = {(round(r["time_sec"]*15),r["person_id"]):r for r in rows}
        assert rows[3,1]["scores"] is None
        assert (rows[6,1]["scores"] is not None) == (gap is not None)
        assert rows[6,2]["scores"] is not None
        assert rows[7,3]["scores"] is None and rows[8,3]["scores"] is None
