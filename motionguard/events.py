from dataclasses import dataclass
from .config import LABELS

@dataclass
class State:
    candidate: float | None = None
    start: float | None = None
    detected: float | None = None
    low_since: float | None = None
    peak: float = 0.
    peak_time: float | None = None
    last_high: float | None = None
    last_adequate: float | None = None
    interrupted: bool = False

class EventManager:
    """Independent causal hysteresis per person/label; unknown closes at last observation."""
    def __init__(self,cfg,camera_id="cam01"):
        self.cfg=cfg; self.camera_id=camera_id; self.states={}

    def update(self, person_id, timestamp, scores, adequate=True):
        closed=[]
        for c,label in enumerate(LABELS):
            key=(person_id,c); state=self.states.setdefault(key,State())
            grace = self.cfg.get("fall_pose_grace_sec", 0.) if label == "fall" else 0.
            if grace and state.start is not None and (not adequate or state.interrupted):
                if timestamp-state.last_adequate > grace + 1e-8:
                    closed.append(self._close(person_id,c,state,state.last_high,"observation_lost"))
                    state = self.states[key] = State()
                elif not adequate:
                    state.interrupted = True
                    state.low_since = None
                    continue
            if not adequate:
                if state.start is not None:
                    closed.append(self._close(person_id,c,state,state.last_high,"observation_lost"))
                self.states[key]=State()
                continue
            state.last_adequate = timestamp
            state.interrupted = False
            score=float(scores[c])
            if state.start is None:
                if score>=self.cfg["on_threshold"][c]:
                    if state.candidate is None:
                        state.candidate=timestamp; state.peak=score; state.peak_time=timestamp
                    if score>state.peak: state.peak=score; state.peak_time=timestamp
                    if timestamp-state.candidate+1e-8>=self.cfg["confirm_sec"][c]:
                        state.start=state.candidate; state.detected=timestamp; state.last_high=timestamp
                else:
                    self.states[key]=State()
            else:
                if score>state.peak: state.peak=score; state.peak_time=timestamp
                if score<self.cfg["off_threshold"][c]:
                    if state.low_since is None: state.low_since=timestamp
                    if timestamp-state.low_since+1e-8>=self.cfg["release_sec"]:
                        closed.append(self._close(person_id,c,state,state.low_since,"threshold_release"))
                        self.states[key]=State()
                else:
                    state.low_since=None; state.last_high=timestamp
        return closed

    def interrupt(self, person_id):
        """Break consecutive evidence while allowing a confirmed event to continue."""
        for c in range(len(LABELS)):
            key = (person_id, c)
            state = self.states.get(key)
            if state is None:
                continue
            if state.start is None:
                del self.states[key]
            else:
                state.low_since = None
                state.interrupted = True

    def _close(self,pid,c,s,end,reason):
        return {"camera_id":self.camera_id,"person_id":int(pid),"label":LABELS[c],
                "start_time_sec":s.start,"detected_time_sec":s.detected,
                "end_time_sec":max(s.start,end if end is not None else s.start),
                "prediction_score":s.peak,"score_aggregation":"peak_sigmoid",
                "peak_time_sec":s.peak_time,"end_reason":reason}

    def active(self,pid):
        return [LABELS[c] for (p,c),s in self.states.items() if p==pid and s.start is not None]

    def forget(self,pid,reason="track_lost"):
        events=[]
        for key in list(self.states):
            if key[0]!=pid: continue
            s=self.states.pop(key)
            if s.start is not None: events.append(self._close(pid,key[1],s,s.last_high,reason))
        return events

    def flush(self):
        result=[]
        for pid in set(k[0] for k in self.states): result.extend(self.forget(pid,"stream_end"))
        return result
