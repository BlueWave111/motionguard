"""Causal 2D pose features. Missing values never create synthetic velocities."""
import numpy as np

ANGLE_TRIPLES = [(5,7,9),(6,8,10),(11,13,15),(12,14,16),(5,11,13),(6,12,14)]
PAIRS = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last = None
        self.last_seen = np.full(17, -np.inf)
        self.xy = np.zeros((17, 2), dtype=np.float64)
        self.scale = None

    def step(self, keypoints, bbox, timestamp, width, height):
        k = np.asarray(keypoints, dtype=np.float64)
        if k.shape != (17,3):
            raise ValueError("Expected 17x3 COCO keypoints")
        box = np.asarray(bbox, dtype=float)
        valid = np.isfinite(k).all(1) & (k[:,2] >= self.cfg["keypoint_conf"])
        if self.last is not None and timestamp <= self.last["t"]:
            raise ValueError("Timestamps must increase within a track")
        if self.last is not None and timestamp-self.last["t"] > self.cfg["reset_gap_sec"]:
            self.last = None
            self.last_seen[:] = -np.inf
            self.scale = None
        self.xy[valid] = k[valid,:2]
        self.last_seen[valid] = timestamp
        age = timestamp-self.last_seen
        usable = valid | (age <= self.cfg["max_hold_sec"])
        raw = self.xy.copy()
        raw[~usable] = 0
        root_valid = usable[11] and usable[12]
        root = raw[[11,12]].mean(0) if root_valid else (box[:2]+box[2:])/2
        h = max(box[3]-box[1], 1.)
        self.scale = h if self.scale is None else .8*self.scale+.2*h
        rel = (raw-root)/(self.scale+1e-6)
        rel[~usable] = 0
        glob = raw/np.array([max(width,1),max(height,1)])
        glob[~usable] = 0
        velocity = np.zeros((17,2)); acceleration = np.zeros((17,2))
        global_velocity = np.zeros((17,2))
        if self.last is not None:
            dt = timestamp-self.last["t"]
            consecutive = valid & self.last["valid"]
            velocity[consecutive] = (rel-self.last["rel"])[consecutive]/dt
            global_velocity[consecutive] = (glob-self.last["glob"])[consecutive]/dt
            accel_valid = consecutive & self.last["velocity_valid"]
            acceleration[accel_valid] = (velocity-self.last["v"])[accel_valid]/dt
        else:
            consecutive = np.zeros(17, bool)
        angles=[]; angle_masks=[]
        for a,b,c in ANGLE_TRIPLES:
            ok = bool(valid[[a,b,c]].all())
            u,v=raw[a]-raw[b],raw[c]-raw[b]
            denom=np.linalg.norm(u)*np.linalg.norm(v)
            ok = ok and denom > 1e-8
            angles.append(np.arccos(np.clip(np.dot(u,v)/denom,-1,1))/np.pi if ok else 0.)
            angle_masks.append(float(ok))
        shoulder_ok=valid[[5,6,11,12]].all()
        torso=raw[[5,6]].mean(0)-raw[[11,12]].mean(0)
        tilt=np.arctan2(torso[0],-torso[1])/np.pi if shoulder_ok else 0.
        asym=[]; asym_mask=[]
        for a,b in PAIRS:
            ok=valid[[a,b]].all()
            asym.extend([rel[a,1]-rel[b,1],np.linalg.norm(velocity[a])-np.linalg.norm(velocity[b])] if ok else [0.,0.])
            asym_mask.append(float(ok))
        distances=[]; distance_masks=[]
        # Wrist to nose, chest and hip midpoint. No claim about anatomical pain.
        for wrist in (9,10):
            for target in ((0,), (5,6), (11,12)):
                ok=valid[[wrist,*target]].all()
                distances.append(np.linalg.norm(rel[wrist]-rel[list(target)].mean(0)) if ok else 0.)
                distance_masks.append(float(ok))
        confidence=np.where(valid,np.clip(k[:,2],0,1),0)
        boxf=[box[0]/width,box[1]/height,(box[2]-box[0])/width,h/height,
              (box[2]-box[0])/h,float(root_valid),tilt,float(shoulder_ok)]
        feat=np.concatenate([rel.ravel(),glob.ravel(),confidence,valid.astype(float),
            np.minimum(age,10.)/10.,velocity.ravel(),global_velocity.ravel(),
            acceleration.ravel(),consecutive.astype(float),angles,angle_masks,
            asym,asym_mask,distances,distance_masks,boxf]).astype(np.float32)
        feat=np.nan_to_num(feat,nan=0,posinf=0,neginf=0)
        feat=np.clip(feat,-100,100)
        self.last={"t":timestamp,"valid":valid,"rel":rel,"glob":glob,"v":velocity,
                   "velocity_valid":consecutive}
        quality=float(valid.mean())
        adequate=(quality>=self.cfg["min_valid_fraction"] and h>=self.cfg["min_person_height"])
        return feat, quality, adequate

def summary(x):
    """Window statistics for baseline; x contains only real, non-padded history."""
    x=np.asarray(x,dtype=np.float32)
    slope=(x[-1]-x[0])/max(len(x)-1,1)
    return np.concatenate([x.mean(0),x.std(0),x.min(0),x.max(0),
                           np.percentile(x,25,axis=0),np.percentile(x,75,axis=0),slope]).astype(np.float32)
