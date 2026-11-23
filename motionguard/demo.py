"""Artificial feature-level smoke demo: does NOT simulate clinical validation."""
import json
from pathlib import Path
import numpy as np
from .config import LABELS
from .features import FeatureExtractor


def make_demo(out,cfg):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(cfg["seed"])
    # Feature dimension derived from real extractor, while demo signals are artificial.
    k=np.zeros((17,3)); k[:,:2]=rng.uniform(50,200,(17,2)); k[:,2]=.9
    dim=len(FeatureExtractor(cfg).step(k,[20,20,250,450],0.,640,480)[0])
    rows=[]
    for split in ("train","val","test"):
        for subject in range(3):
            n=560; t=np.arange(n)/cfg["sample_hz"]
            y=np.zeros((n,7),np.float32)
            for c in range(7):
                start=40+c*65; y[start:start+38,c]=1
            y[145:170,0]=1; y[145:170,4]=1
            x=rng.normal(0,.15,(n,dim)).astype(np.float32)
            for c in range(7):
                x[:,c]=y[:,c]*2+rng.normal(0,.1,n)
            name=f"{split}_{subject}.npz"
            np.savez_compressed(out/name,features=x,timestamps=t,quality=np.ones(n,bool),labels=y,demo_only=np.array(True))
            rows.append({"path":name,"split":split,"subject_id":f"{split}_s{subject}","source_id":f"{split}_v{subject}"})
    manifest=out/"manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows)+"\n",encoding="utf-8")
    return manifest
