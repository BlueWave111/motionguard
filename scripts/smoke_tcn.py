"""Reproduce small CPU training checks; generated checkpoints remain demo-only."""
from pathlib import Path
import argparse
import numpy as np
import torch
from motionguard.config import load_config
from motionguard.demo import make_demo
from motionguard.train import fit_tcn,evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--rgb',action='store_true')
    parser.add_argument('--out',default='runs/tcn_smoke'); args=parser.parse_args()
    torch.set_num_threads(2)
    cfg=load_config(); cfg.update(window=32,min_history=16,channels=8,dilations=[1,2],epochs=1,
                                  batch_size=4,stride=16,use_rgb=args.rgb,crop_size=24)
    out=Path(args.out); manifest=make_demo(out/'data',cfg)
    if args.rgb:
        for p in (out/'data').glob('*.npz'):
            with np.load(p) as z: r={k:z[k] for k in z.files}
            r['crops']=np.zeros((len(r['features']),24,24,3),np.uint8)
            np.savez_compressed(p,**r)
    fit_tcn(manifest,out/'DEMO_ONLY_tcn.pt',cfg)
    evaluate(manifest,out/'DEMO_ONLY_tcn.pt',out/'DEMO_ONLY_test.json')
    print('Artificial training/save/reload/evaluation complete. No real action validation.')
