import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from .config import load_config

def training_destination(command, explicit=None):
    """Reserve a unique run directory so automatic saves cannot overwrite another run."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = Path("runs") / command.replace("train-", "")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    return (folder / ("model.pt" if command != "train-baseline" else "model.joblib")).resolve()

def main(argv=None):
    p=argparse.ArgumentParser(description="MotionGuard: per-person temporal multilabel motion research")
    sub=p.add_subparsers(dest="command",required=True)
    ex=sub.add_parser("extract",help="Extract tracked pose records from a video/camera")
    ex.add_argument("--source",required=True); ex.add_argument("--out",required=True)
    ex.add_argument("--config"); ex.add_argument("--save-crops",action="store_true")
    an=sub.add_parser("annotate",help="Attach reviewed intervals to one track NPZ")
    an.add_argument("--record",required=True); an.add_argument("--annotations",required=True); an.add_argument("--out",required=True)
    for name in ("train-baseline", "train-tcn", "train-cnn-lstm", "train-stgcn"):
        tr = sub.add_parser(name)
        tr.add_argument("--manifest", required=True)
        tr.add_argument("--out", help="Optional checkpoint path; defaults to a unique directory under runs/")
        tr.add_argument("--config")
        if name != "train-baseline":
            tr.add_argument("--activity-intervals", help="Reviewed ADL interval JSONL for hard-negative training")
    ev=sub.add_parser("evaluate"); ev.add_argument("--manifest",required=True); ev.add_argument("--checkpoint",required=True)
    ev.add_argument("--out",required=True); ev.add_argument("--device",default="cpu")
    inf=sub.add_parser("infer"); inf.add_argument("--source",required=True); inf.add_argument("--checkpoint",required=True)
    inf.add_argument("--out",required=True); inf.add_argument("--device",default="cpu"); inf.add_argument("--camera-id",default="cam01")
    inf.add_argument("--display",action="store_true")
    inf.add_argument("--pose-gap-sec",type=float,help="Preserve same-track history over short pose gaps")
    inf.add_argument("--reid-gap-sec", type=float, help="Try recovering a recently lost person ID using position and appearance")
    inf.add_argument("--min-history", type=int, help="Override the checkpoint minimum number of valid pose samples")
    inf.add_argument("--fall-pose-grace-sec", type=float, help="Keep confirmed falls active through short same-ID pose gaps")
    ee=sub.add_parser("evaluate-events")
    ee.add_argument("--ground-truth",required=True); ee.add_argument("--predictions",required=True)
    ee.add_argument("--reviewed-camera-hours",type=float,required=True); ee.add_argument("--iou",type=float,default=.3)
    ee.add_argument("--out",required=True)
    demo=sub.add_parser("demo"); demo.add_argument("--out",default="runs/demo")
    a=p.parse_args(argv)
    if a.command=="extract":
        from .video import extract
        extract(a.source,a.out,load_config(a.config),a.save_crops)
    elif a.command=="annotate":
        from .data import annotate
        annotate(a.record,a.annotations,a.out)
    elif a.command in ("train-baseline", "train-tcn", "train-cnn-lstm", "train-stgcn"):
        from .train import fit_baseline, fit_cnn_lstm, fit_tcn, fit_stgcn
        fn = {"train-baseline": fit_baseline, "train-tcn": fit_tcn, "train-cnn-lstm": fit_cnn_lstm, "train-stgcn": fit_stgcn}[a.command]
        cfg = load_config(a.config)
        if cfg["activity_aux_weight"] and (a.command != "train-tcn" or not getattr(a, "activity_intervals", None)):
            p.error("activity_aux_weight requires train-tcn and --activity-intervals")
        if getattr(a, "activity_intervals", None):
            if cfg["adl_negative_weight"] <= 1 and not cfg["activity_aux_weight"]:
                p.error("--activity-intervals requires hard-negative weighting or activity_aux_weight")
            cfg["activity_intervals"] = str(Path(a.activity_intervals).resolve())
        destination=training_destination(a.command,a.out)
        print(f"Checkpoint destination: {destination}",flush=True)
        fn(a.manifest,str(destination),cfg)
        print(f"Training complete. Model saved: {destination}",flush=True)
    elif a.command=="evaluate":
        from .train import evaluate
        print(json.dumps(evaluate(a.manifest,a.checkpoint,a.out,a.device),indent=2))
    elif a.command=="infer":
        from .video import infer
        options = {"pose_gap_sec": a.pose_gap_sec}
        if a.reid_gap_sec is not None:
            options["reid_gap_sec"] = a.reid_gap_sec
        if a.min_history is not None:
            options["min_history"] = a.min_history
        if a.fall_pose_grace_sec is not None:
            options["fall_pose_grace_sec"] = a.fall_pose_grace_sec
        infer(a.source,a.checkpoint,a.out,a.device,a.camera_id,a.display,**options)
    elif a.command=="evaluate-events":
        from .event_metrics import event_evaluate
        print(json.dumps(event_evaluate(a.ground_truth,a.predictions,a.reviewed_camera_hours,a.out,a.iou),indent=2))
    elif a.command=="demo":
        from .demo import make_demo
        from .train import fit_baseline,evaluate
        cfg=load_config(); cfg["stride"]=4
        out=Path(a.out); manifest=make_demo(out/"synthetic_features",cfg)
        checkpoint=out/"DEMO_ONLY_baseline.joblib"
        fit_baseline(manifest,checkpoint,cfg)
        import joblib
        artifact=joblib.load(checkpoint); artifact["demo_only"]=True; joblib.dump(artifact,checkpoint)
        result=evaluate(manifest,checkpoint,out/"DEMO_ONLY_metrics.json")
        print("SYNTHETIC FEATURE DEMO ONLY. Do not use this checkpoint on real video.")
        print(json.dumps(result,indent=2))

if __name__=="__main__": main()

