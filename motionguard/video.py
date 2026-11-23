"""YOLO adapter, per-person buffers and bounded event video writing."""
import json
import time
import warnings
from collections import deque
from pathlib import Path
import numpy as np
from .features import FeatureExtractor
from .events import EventManager
from .config import LABELS, FEATURE_CONFIG_KEYS


def crop_person(frame,box,cfg):
    import cv2
    h,w=frame.shape[:2]; x1,y1,x2,y2=map(float,box)
    margin=cfg["crop_margin"]; dx=(x2-x1)*margin; dy=(y2-y1)*margin
    x1=max(0,int(x1-dx)); x2=min(w,int(x2+dx))
    y1=max(0,int(y1-dy)); y2=min(h,int(y2+dy))
    if x2<=x1 or y2<=y1: return np.zeros((cfg["crop_size"],cfg["crop_size"],3),np.uint8)
    # Store RGB, not OpenCV BGR.
    return cv2.cvtColor(cv2.resize(frame[y1:y2,x1:x2],(cfg["crop_size"],cfg["crop_size"])),cv2.COLOR_BGR2RGB)


def frames(source,cfg):
    import cv2
    source=int(source) if str(source).isdigit() else source
    live=isinstance(source,int) or str(source).lower().startswith(("rtsp://","http://","https://"))
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise ValueError(f"Cannot open source: {source}")
    fps=float(cap.get(cv2.CAP_PROP_FPS)); fps=fps if np.isfinite(fps) and fps>0 else 30.
    if not live and fps<cfg["sample_hz"]:
        warnings.warn("Source FPS below sample_hz: validate model on this frame rate")
    origin=time.monotonic(); index=0; next_time=0.
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            timestamp=time.monotonic()-origin if live else index/fps
            index+=1
            if timestamp+1e-8<next_time: continue
            next_time += 1/cfg["sample_hz"]
            if next_time < timestamp: next_time = timestamp + 1/cfg["sample_hz"]
            yield timestamp,frame
    finally: cap.release()


class PoseStream:
    def __init__(self,cfg):
        from ultralytics import YOLO
        self.cfg=cfg; self.model=YOLO(cfg["pose_model"])
    def step(self,frame):
        result=self.model.track(frame,persist=True,tracker=self.cfg["tracker"],
            device=self.cfg["device"],imgsz=self.cfg["image_size"],
            conf=self.cfg["detection_conf"],max_det=self.cfg["max_people"],verbose=False)[0]
        if result.keypoints is None or result.boxes is None or result.boxes.id is None: return []
        boxes=result.boxes.xyxy.cpu().numpy(); ids=result.boxes.id.int().cpu().numpy()
        points=result.keypoints.data.cpu().numpy()
        if points.shape[1:]!=(17,3): raise ValueError("This feature schema requires COCO 17x3 pose output")
        return [(int(pid),box,k) for pid,box,k in zip(ids,boxes,points)]


class ClipStore:
    """One bounded-length clip per confirmed event; metadata records truncation.

    Camera frames are shared in a bounded ring. Event clips are streamed to disk,
    never retained as an unbounded Python list. Full camera context is preserved.
    """
    def __init__(self,out,cfg,camera_id="cam01"):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.cfg=cfg; self.camera_id=camera_id
        self.ring=deque(); self.jobs={}; self.counter=0; self.timestamp=0.
        self.log=(self.out/"events.jsonl").open("w",encoding="utf-8")
    def push(self,t,frame):
        self.timestamp=t; self.ring.append((t,frame.copy()))
        keep=self.cfg["pre_roll_sec"]+max(self.cfg["confirm_sec"])+1.
        while self.ring and t-self.ring[0][0]>keep: self.ring.popleft()
        for key,job in list(self.jobs.items()):
            if job["writer"] is not None:
                if t-job["first"]<=self.cfg["max_clip_sec"]:
                    self._write(job,t,frame)
                else:
                    job["writer"].release(); job["writer"]=None; job["capped"]=True
            if job["event"] is not None and t>=job["finish"]:
                self._finish(key)
    def _write(self,job,t,frame):
        # Resample onto a constant-FPS clip timeline without speeding up dropped frames.
        while job["next"]<=t+1e-8:
            job["writer"].write(frame); job["last"]=job["next"]
            job["next"]+=1/self.cfg["sample_hz"]
    def begin(self,pid,c,state):
        import cv2
        key=(pid,c,state.start)
        if key in self.jobs or not self.ring: return
        self.counter+=1; name=f"event_{self.counter:06d}_person{pid}_{LABELS[c]}.mp4"
        chosen=[(t,f) for t,f in self.ring if t>=state.start-self.cfg["pre_roll_sec"]]
        if not chosen: chosen=[self.ring[-1]]
        h,w=chosen[0][1].shape[:2]
        writer=cv2.VideoWriter(str(self.out/name),cv2.VideoWriter_fourcc(*"mp4v"),self.cfg["sample_hz"],(w,h))
        if not writer.isOpened(): raise RuntimeError("Cannot open MP4 writer; check OpenCV codec support")
        job={"writer":writer,"path":name,"first":chosen[0][0],"last":chosen[0][0],
             "next":chosen[0][0],"event":None,"finish":None,"capped":False}
        self.jobs[key]=job
        for t,f in chosen: self._write(job,t,f)
    def close_event(self,event):
        c=LABELS.index(event["label"]); key=(event["person_id"],c,event["start_time_sec"])
        if key not in self.jobs:
            event.update({"clip_path":None,"clip_unavailable":True}); self.log.write(json.dumps(event)+"\n"); self.log.flush(); return
        job=self.jobs[key]; job["event"]=event
        job["finish"]=event["end_time_sec"]+self.cfg["post_roll_sec"]
        if self.timestamp>=job["finish"]: self._finish(key)
    def _finish(self,key):
        job=self.jobs.pop(key)
        if job["writer"] is not None: job["writer"].release()
        event=job["event"]
        if event is None: raise RuntimeError("Event must be closed before clip finalization")
        event.update({"clip_path":job["path"],"clip_start_sec":job["first"],"clip_end_sec":job["last"],
            "clip_capped":job["capped"],"post_roll_complete":job["last"]+1/self.cfg["sample_hz"]>=job["finish"],
            "video_scope":"full_camera_frame"})
        self.log.write(json.dumps(event)+"\n"); self.log.flush()
    def finish(self):
        for key in list(self.jobs): self._finish(key)
        self.log.close()


def extract(source,out,cfg,save_crops=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    pose=PoseStream(cfg); tracks={}; counts={}; extractors={}; last={}
    # Flush inactive tracks; each NPZ is a contiguous track segment, not an identity.
    def flush(pid):
        rows=tracks.pop(pid,[])
        if not rows: return
        counts[pid]=counts.get(pid,0)+1
        payload={"features":np.stack([r[0] for r in rows]),"timestamps":np.array([r[1] for r in rows]),
                 "quality":np.array([r[2] for r in rows],bool),"keypoints":np.stack([r[3] for r in rows]),
                 "bboxes":np.stack([r[4] for r in rows]),"person_id":np.array(pid),
                 "feature_config":np.array(json.dumps({k:cfg[k] for k in FEATURE_CONFIG_KEYS})),
                 "schema_version":np.array(1)}
        if save_crops: payload["crops"]=np.stack([r[5] for r in rows])
        np.savez_compressed(out/f"person{pid}_segment{counts[pid]:04d}.npz",**payload)
        extractors.pop(pid,None); last.pop(pid,None)
    try:
        for t,frame in frames(source,cfg):
            for pid in list(tracks):
                if t-last[pid]>cfg["reset_gap_sec"]: flush(pid)
            detections=pose.step(frame)
            seen={pid for pid,box,k in detections}
            for missing_pid in list(tracks):
                if missing_pid not in seen: flush(missing_pid)
            for pid,box,k in detections:
                fe=extractors.setdefault(pid,FeatureExtractor(cfg))
                f,q,ok=fe.step(k,box,t,frame.shape[1],frame.shape[0])
                crop=crop_person(frame,box,cfg) if save_crops else None
                tracks.setdefault(pid,[]).append((f,t,ok,k,box,crop)); last[pid]=t
                # Bound memory for long streams; segments must share subject/source split.
                if len(tracks[pid])>=int(cfg["sample_hz"]*60): flush(pid)
    finally:
        for pid in list(tracks): flush(pid)
        (out/"extraction_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")


def infer(source,checkpoint,out,device="cpu",camera_id="cam01",display=False,pose_gap_sec=None,reid_gap_sec=None,min_history=None,fall_pose_grace_sec=None):
    import cv2
    from .train import Predictor
    predictor=Predictor(checkpoint,device)
    if predictor.artifact.get("demo_only"):
        raise ValueError("Synthetic demo checkpoint cannot be used for video inference; train on reviewed real data")
    cfg=predictor.cfg; cfg["device"]=device
    if pose_gap_sec is not None and not 0 <= pose_gap_sec <= cfg["reset_gap_sec"]:
        raise ValueError("pose_gap_sec must be between zero and the track reset timeout")
    if pose_gap_sec is not None:
        cfg["pose_gap_sec"] = pose_gap_sec
    if min_history is not None:
        if isinstance(min_history, bool) or not isinstance(min_history, int) or not 1 <= min_history <= cfg["window"]:
            raise ValueError("min_history must be an integer between 1 and the checkpoint window size")
        cfg["min_history"] = min_history
    if fall_pose_grace_sec is not None:
        if not 0 <= fall_pose_grace_sec <= cfg["reset_gap_sec"]:
            raise ValueError("fall_pose_grace_sec must be between zero and the track reset timeout")
        cfg["fall_pose_grace_sec"] = fall_pose_grace_sec
    reid = None
    if reid_gap_sec is not None:
        if not 0 < reid_gap_sec <= cfg["reset_gap_sec"]:
            raise ValueError("reid_gap_sec must be positive and within the track reset timeout")
        from .reid import TrackReidentifier
        reid = TrackReidentifier(reid_gap_sec, retention_sec=cfg["reset_gap_sec"])
        cfg["reid_gap_sec"] = reid_gap_sec
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/"run_config.json").write_text(json.dumps({"cfg":cfg,"labels":LABELS,"checkpoint":str(checkpoint)},indent=2),encoding="utf-8")
    # Bound the time between real observations when a sampled detection is missed.
    observation_grace_sec = min(0.15, cfg["reset_gap_sec"])
    pose=PoseStream(cfg); manager=EventManager(cfg,camera_id); clips=ClipStore(out,cfg,camera_id)
    history_gap_sec = observation_grace_sec if pose_gap_sec is None else pose_gap_sec
    tracks={}; scorelog=(out/"scores.jsonl").open("w",encoding="utf-8")
    detection_boxes = {}
    reid_log = (out/"reid.jsonl").open("w", encoding="utf-8") if reid else None

    def close_event(event):
        key = (event["person_id"], LABELS.index(event["label"]), event["start_time_sec"])
        event["bbox_xyxy_at_detection"] = detection_boxes.pop(key, None)
        clips.close_event(event)

    try:
        for t,frame in frames(source,cfg):
            clips.push(t,frame)
            for pid in list(tracks):
                if t-tracks[pid]["last"]>cfg["reset_gap_sec"]:
                    for event in manager.forget(pid): close_event(event)
                    del tracks[pid]
            detections=pose.step(frame)
            if reid is not None:
                detections, links = reid.step(frame, detections, t)
                for raw, pid, similarity in links:
                    reid_log.write(json.dumps({"time_sec": t, "raw_id": raw,
                                              "person_id": pid, "similarity": similarity})+"\n")
                    # Keep the ID history, but do not compute velocities across the break.
                    if pid in tracks:
                        tracks[pid]["extractor"] = FeatureExtractor(cfg)
                        manager.interrupt(pid)
            seen={pid for pid,box,k in detections}
            for pid,tr in tracks.items():
                missing = pid not in seen
                if not missing and not tr["missing"]:
                    continue
                if not tr["missing"]:
                    tr["missing"] = True
                    tr["extractor"] = FeatureExtractor(cfg)
                    manager.interrupt(pid)
                if t - tr["last"] > history_gap_sec + 1e-8:
                    tr["features"].clear()
                    tr["crops"].clear()
                if t - tr["last"] > observation_grace_sec + 1e-8:
                    for event in manager.update(pid,t,np.zeros(7),False):
                        close_event(event)
                if missing:
                    scorelog.write(json.dumps({"time_sec":t,"camera_id":camera_id,"person_id":pid,"status":"insufficient_data","bbox_xyxy":None,
                        "keypoint_valid_fraction":0.,"scores":None,"events":manager.active(pid)})+"\n")
            for pid,box,k in detections:
                if pid not in tracks:
                    tracks[pid]={"extractor":FeatureExtractor(cfg),"features":deque(maxlen=cfg["window"]),
                                 "crops":deque(maxlen=cfg["window"]),"last":t,"missing":False,"last_good":None}
                tr=tracks[pid]; tr["last"]=t; tr["missing"]=False
                feat,quality,ok=tr["extractor"].step(k,box,t,frame.shape[1],frame.shape[0])
                scores=np.zeros(7); status="insufficient_data"
                last_good = tr["last_good"]
                expired = pose_gap_sec is not None and last_good is not None and t-last_good > pose_gap_sec + 1e-8
                if expired or (not ok and pose_gap_sec is None):
                    tr["features"].clear(); tr["crops"].clear()
                if ok:
                    tr["last_good"] = t
                    tr["features"].append(feat)
                    if cfg["use_rgb"]: tr["crops"].append(crop_person(frame,box,cfg))
                ready=ok and len(tr["features"])>=cfg["min_history"]
                if ready:
                    scores=predictor.predict(list(tr["features"]),list(tr["crops"]) if cfg["use_rgb"] else None)
                for event in manager.update(pid,t,scores,ready): close_event(event)
                for (p,c),state in manager.states.items():
                    if p==pid and state.start is not None:
                        key = (pid, c, state.start)
                        if key not in detection_boxes:
                            detection_boxes[key] = [float(value) for value in box]
                        clips.begin(pid,c,state)
                labels=manager.active(pid)
                if ready: status="event" if labels else ("candidate" if any(s.candidate is not None for (p,c),s in manager.states.items() if p==pid) else "normal")
                scorelog.write(json.dumps({"time_sec":t,"camera_id":camera_id,"person_id":pid,"status":status,
                    "bbox_xyxy":[float(value) for value in box],
                    "keypoint_valid_fraction":quality,"scores":dict(zip(LABELS,map(float,scores))) if ready else None,
                    "events":labels})+"\n")
                if display:
                    x1,y1,x2,y2=map(int,box); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"ID {pid}: {','.join(labels) or status}",(x1,max(y1-5,15)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            if display:
                cv2.imshow("MotionGuard",frame)
                if cv2.waitKey(1)&255==ord('q'): break
    finally:
        for event in manager.flush(): close_event(event)
        clips.finish(); scorelog.close()
        if reid_log is not None: reid_log.close()
        if display: cv2.destroyAllWindows()
