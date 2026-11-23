"""Explicit, leakage-checked manifests and dense interval supervision."""
import json
from pathlib import Path
import numpy as np
from .config import LABELS, FEATURE_CONFIG_KEYS


def read_manifest(path):
    path=Path(path).resolve(); entries=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        for k in ("path","split","subject_id","source_id"):
            if k not in r: raise ValueError(f"Manifest row missing {k}")
        if r["split"] not in ("train","val","test"): raise ValueError("Invalid split")
        r["path"]=str((path.parent/r["path"]).resolve())
        entries.append(r)
    if not entries: raise ValueError("Empty manifest")
    # Both a person and a source video must belong to exactly one split.
    for key in ("path","subject_id","source_id"):
        seen={}
        for r in entries:
            if r[key] in seen and seen[r[key]]!=r["split"]:
                raise ValueError(f"Split leakage in {key}: {r[key]}")
            seen[r[key]]=r["split"]
    return entries


def load_record(path):
    with np.load(path,allow_pickle=False) as z:
        r={k:z[k] for k in z.files}
    for key in ("features","timestamps","quality"):
        if key not in r: raise ValueError(f"Missing {key}: {path}")
    n=len(r["features"])
    if n==0 or r["features"].ndim!=2: raise ValueError("Empty or invalid features")
    if any(len(r[k])!=n for k in ("timestamps","quality")): raise ValueError("Length mismatch")
    if np.any(np.diff(r["timestamps"])<=0): raise ValueError("Non-increasing timestamps")
    if not np.isfinite(r["features"]).all(): raise ValueError("Non-finite features")
    if "labels" in r:
        if r["labels"].shape!=(n,7): raise ValueError("Expected Nx7 labels")
        if not np.isin(r["labels"],[-1,0,1]).all(): raise ValueError("Labels must be -1,0,1")
    return r


def annotate(record_path,annotation_path,out):
    r=load_record(record_path)
    a=json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    # Unknown defaults to -1. reviewed_labels explicitly authorizes negatives.
    y=np.full((len(r["timestamps"]),7),-1,dtype=np.float32)
    for label in a.get("reviewed_labels",[]):
        if label not in LABELS: raise ValueError(f"Unknown label {label}")
        y[:,LABELS.index(label)]=0
    for e in a.get("events",[]):
        if e["label"] not in LABELS: raise ValueError("Unknown event label")
        if e["end"]<=e["start"]: raise ValueError("Invalid interval")
        mask=(r["timestamps"]>=e["start"]) & (r["timestamps"]<e["end"])
        y[mask,LABELS.index(e["label"])]=1
    r["labels"]=y
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**r)


def windows(entries,cfg,stride=None):
    result=[]
    for e in entries:
        r=load_record(e["path"])
        if "labels" not in r: raise ValueError("Training requires annotated records")
        if "feature_config" in r:
            stored=json.loads(str(r["feature_config"]))
            for key in FEATURE_CONFIG_KEYS:
                if stored.get(key)!=cfg[key]: raise ValueError(f"Extraction/training config mismatch: {key}")
        # Split on time gaps and every inadequate frame, matching online resets.
        start_segment = 0
        for end in range(len(r["features"])):
            if end and r["timestamps"][end]-r["timestamps"][end-1]>cfg["reset_gap_sec"]:
                start_segment=end
            if not r["quality"][end]:
                start_segment=end+1
                continue
            if end-start_segment+1 < cfg["min_history"]:
                continue
            if (end-start_segment-cfg["min_history"]+1) % (stride or cfg["stride"]):
                continue
            start=max(start_segment,end-cfg["window"]+1)
            result.append((r,start,end,e))
    return result


def padded(x,length):
    out=np.zeros((length,*x.shape[1:]),dtype=x.dtype)
    out[-len(x):]=x
    return out
