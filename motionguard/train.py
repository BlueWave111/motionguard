import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support,average_precision_score,roc_auc_score
from .data import read_manifest,windows,padded
from .features import summary
from .config import LABELS


def metrics(y,p,thresholds):
    report={}
    for c,label in enumerate(LABELS):
        mask=y[:,c]>=0; target=y[mask,c].astype(int); score=p[mask,c]
        if not len(target): report[label]={"n_reviewed":0}; continue
        pred=score>=thresholds[c]
        prec,rec,f1,_=precision_recall_fscore_support(target,pred,average="binary",zero_division=0)
        both=len(np.unique(target))==2
        report[label]={"n_reviewed":len(target),"positives":int(target.sum()),
            "precision":float(prec),"recall":float(rec),"f1":float(f1),
            "pr_auc_ap":float(average_precision_score(target,score)) if both else None,
            "roc_auc":float(roc_auc_score(target,score)) if both else None,
            "tp":int(((target==1)&pred).sum()),"fp":int(((target==0)&pred).sum()),
            "fn":int(((target==1)&~pred).sum()),"tn":int(((target==0)&~pred).sum())}
    return report


def tune(y,p,active=None):
    thresholds=[1.] * len(LABELS)
    for c in range(len(LABELS)) if active is None else np.flatnonzero(active):
        mask=y[:,c]>=0; target=y[mask,c]; scores=p[mask,c]
        if len(np.unique(target))<2: raise ValueError(f"Validation needs positives and negatives for {LABELS[c]}")
        candidates=np.linspace(.1,.9,33)
        f=[]
        for t in candidates:
            pred=scores>=t; tp=((target==1)&pred).sum()
            f.append(2*tp/max(2*tp+((target==0)&pred).sum()+((target==1)&~pred).sum(),1))
        thresholds[c]=float(candidates[np.argmax(f)])
    return thresholds


def _partition(manifest,cfg):
    entries=read_manifest(manifest)
    train=windows([e for e in entries if e["split"]=="train"],cfg)
    val=windows([e for e in entries if e["split"]=="val"],cfg)
    if not train or not val: raise ValueError("Need usable train and validation windows")
    return train,val


def fit_baseline(manifest,out,cfg):
    train,val=_partition(manifest,cfg)
    def arrays(ws):
        return np.stack([summary(r["features"][a:b+1]) for r,a,b,_ in ws]),np.stack([r["labels"][b] for r,a,b,_ in ws])
    x,y=arrays(train); vx,vy=arrays(val)
    scaler=StandardScaler().fit(x); x=scaler.transform(x); vx=scaler.transform(vx)
    models=[]
    for c in range(7):
        known=y[:,c]>=0
        if len(np.unique(y[known,c]))!=2: raise ValueError(f"Training needs both classes: {LABELS[c]}")
        m=LogisticRegression(class_weight="balanced",max_iter=2000,random_state=cfg["seed"])
        m.fit(x[known],y[known,c]); models.append(m)
    p=np.stack([m.predict_proba(vx)[:,list(m.classes_).index(1)] for m in models],axis=1)
    thresholds=tune(vy,p)
    artifact={"kind":"baseline","demo_only":any(bool(r.get("demo_only",False)) for r,a,b,e in train),"models":models,"scaler":scaler,"cfg":cfg,
              "labels":LABELS,"thresholds":thresholds,"feature_dim":train[0][0]["features"].shape[1]}
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(artifact,out)
    out.with_suffix(".validation.json").write_text(json.dumps(metrics(vy,p,thresholds),indent=2),encoding="utf-8")
    return artifact


def fit_tcn(manifest,out,cfg):
    return _fit_temporal(manifest,out,cfg,"tcn")


def fit_stgcn(manifest,out,cfg):
    return _fit_temporal(manifest,out,cfg,"stgcn")

def fit_cnn_lstm(manifest,out,cfg):
    return _fit_temporal(manifest,out,cfg,"cnn-lstm")


def adl_hard_negatives(record):
    """Select reviewed non-falls resembling falls in OmniFall pose records."""
    labels = record.get("activity_labels")
    if labels is None:
        return np.zeros(len(record["labels"]), dtype=bool)
    features = record["features"]
    if labels.shape != (len(features),) or features.shape[1] != 288:
        raise ValueError("ADL weighting requires one activity label per frame and MotionGuard pose features")
    # OmniFall labels 3-6 cover sitting and lying transitions and states.
    seated_or_lying = np.isin(labels, (3, 4, 5, 6))
    # Label 9 is "other". A valid, strongly inclined torso selects candidates
    # for bending without treating the entire "other" class as bending.
    bending_like = (labels == 9) & (features[:, -1] == 1) & (np.abs(features[:, -2]) >= 1 / 6)
    return (record["labels"][:, LABELS.index("fall")] == 0) & (seated_or_lying | bending_like)


def reviewed_adl_intervals(path, include_label=False):
    intervals = defaultdict(list)
    activities = {"bend", "sit_down", "sitting", "lie_down", "lying", "walking", "standing"}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train" or row["label"] not in activities:
                continue
            start, end = float(row["start"]), float(row["end"])
            if not np.isfinite([start, end]).all() or not 0 <= start < end:
                raise ValueError(f"Invalid reviewed ADL interval on line {number}")
            if not include_label and row["label"] in {"walking", "standing"}:
                continue
            interval = (start, end, row["label"]) if include_label else (start, end)
            intervals[row["source_id"]].append(interval)
    return intervals


def pose_dropout_indices(length, min_history, max_frames, rng):
    """Keep the current observation and remove one interior run of pose samples."""
    removable = min(max_frames, length - max(2, min_history))
    if removable <= 0:
        return slice(None)
    dropped = rng.integers(1, removable + 1)
    start = rng.integers(1, length - dropped)
    return np.r_[0:start, start + dropped:length]


def _fit_temporal(manifest,out,cfg,kind):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from .models import TEMPORAL_MODELS
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    train,val=_partition(manifest,cfg)
    # Fit scaler on unique raw training frames only, never validation/test.
    training_records = {e["path"]: (r, e["source_id"]) for r, _, _, e in train}
    records = {path: record for path, (record, _) in training_records.items()}
    scaler=StandardScaler().fit(np.concatenate([r["features"][r["quality"].astype(bool)] for r in records.values()]))
    ys=np.concatenate([r["labels"][r["quality"].astype(bool)] for r in records.values()])
    adl_weight = cfg.get("adl_negative_weight", 1.)
    active = np.array([label == "fall" or (kind == "tcn" and not cfg.get("fall_only", False)) for label in LABELS])
    aux_weight = cfg.get("activity_aux_weight", 0.)
    activity_targets = {}
    if aux_weight:
        if kind != "tcn" or not cfg.get("activity_intervals"):
            raise ValueError("Activity supervision requires TCN and reviewed activity intervals")
        reviewed = reviewed_adl_intervals(cfg["activity_intervals"], include_label=True)
        groups = {"bend": 0, "sit_down": 1, "sitting": 1, "lie_down": 2, "lying": 2}
        aux_labels = cfg.get("activity_aux_labels", ["bend", "sit", "lie"])
        groups.update(walking=3, standing=4)
        label_names = {"bend": "bend", "sit_down": "sit", "sitting": "sit",
                       "lie_down": "lie", "lying": "lie", "walking": "walking", "standing": "standing"}
        groups = {label: aux_labels.index(name) for label, name in label_names.items() if name in aux_labels}
        counts = np.zeros(len(aux_labels), dtype=int)
        for path, (record, source_id) in training_records.items():
            ts = record["timestamps"]
            fall = record["labels"][:, 0]
            quality = record["quality"].astype(bool)
            target = np.full((len(ts), len(aux_labels)), -1, dtype=np.float32)
            for start, end, label in reviewed.get(source_id, ()):
                if label not in groups:
                    continue
                selected = (ts >= start) & (ts < end) & quality
                # Support states may overlap a fall transition. Fall targets remain independent.
                target[selected, groups[label]] = 1
            counts += (target == 1).sum(axis=0)
            # Legacy posture classes retain their reviewed supervision. New support
            # classes receive supervision only where explicitly annotated.
            posture = [aux_labels.index(name) for name in ("bend", "sit", "lie") if name in aux_labels]
            known = (target[:, posture] == 1).any(axis=1) | ((fall == 1) & quality)
            for column in posture:
                target[known & (target[:, column] < 0), column] = 0
            if "walking" in aux_labels and "standing" in aux_labels:
                walk, stand = aux_labels.index("walking"), aux_labels.index("standing")
                target[(target[:, walk] == 1) & (target[:, stand] < 0), stand] = 0
                target[(target[:, stand] == 1) & (target[:, walk] < 0), walk] = 0
            activity_targets[path] = target
        if not counts.all():
            raise ValueError("Activity supervision needs usable training examples for every configured auxiliary label")
        print(json.dumps({"aux_positive_frames": dict(zip(aux_labels, counts.tolist()))}), flush=True)
    hard_negatives = {}
    if adl_weight > 1:
        hard_negatives = {path: adl_hard_negatives(record) for path, record in records.items()}
        if cfg.get("activity_intervals") and cfg.get("reviewed_activity_hard_negatives", True):
            intervals = reviewed_adl_intervals(cfg["activity_intervals"])
            reviewed_count = added_count = 0
            for path, (record, source_id) in training_records.items():
                timestamps = record["timestamps"]
                reviewed = np.zeros(len(timestamps), dtype=bool)
                for start, end in intervals.get(source_id, ()):
                    reviewed |= (timestamps >= start) & (timestamps < end)
                reviewed &= record["quality"].astype(bool) & (record["labels"][:, 0] == 0)
                reviewed_count += int(reviewed.sum())
                added_count += int((reviewed & ~hard_negatives[path]).sum())
                hard_negatives[path] |= reviewed
            if not reviewed_count:
                raise ValueError("No reviewed ADL intervals overlap usable training non-fall frames")
            print(json.dumps({"reviewed_adl_frames": reviewed_count,
                              "added_hard_negative_frames": added_count}), flush=True)
        if not any((mask & records[path]["quality"].astype(bool)).any() for path, mask in hard_negatives.items()):
            raise ValueError("ADL weighting requires reviewed activity labels and usable pose samples")
    pos=(ys==1).sum(0); neg=(ys==0).sum(0)
    missing = active & ((pos == 0) | (neg == 0))
    if missing.any():
        raise ValueError(f"Training needs positive and negative examples for {np.array(LABELS)[missing].tolist()}")
    dropout_probability = cfg.get("pose_dropout_probability", 0.)
    dropout_rng = np.random.default_rng(cfg["seed"])
    class Windows(Dataset):
        def __init__(self, ws, training):
            self.ws = ws
            self.training = training
        def __len__(self): return len(self.ws)
        def __getitem__(self,i):
            r,a,b,entry=self.ws[i]
            keep = slice(None)
            if self.training and dropout_probability and dropout_rng.random() < dropout_probability:
                keep = pose_dropout_indices(b-a+1, cfg["min_history"],
                                            cfg.get("pose_dropout_max_frames", 3), dropout_rng)
            features = r["features"][a:b+1][keep]
            n = len(features)
            x=padded(scaler.transform(features).astype(np.float32),cfg["window"])
            y=np.full((cfg["window"],7),-1,np.float32); y[-n:]=r["labels"][a:b+1][keep]
            y[:, ~active] = -1
            rgb=np.empty((0,),np.float32)
            if cfg["use_rgb"]:
                if "crops" not in r: raise ValueError("RGB training requires crops in every record")
                rgb=padded(r["crops"][a:b+1][keep],cfg["window"]).transpose(0,3,1,2).astype(np.float32)/255.
            hard = np.zeros(cfg["window"], dtype=bool)
            if entry["path"] in hard_negatives:
                hard[-n:] = hard_negatives[entry["path"]][a:b+1][keep]
            activity = np.full((cfg["window"], len(cfg.get("activity_aux_labels", ["bend", "sit", "lie"]))), -1, dtype=np.float32)
            if self.training and aux_weight:
                activity[-n:] = activity_targets[entry["path"]][a:b+1][keep]
            return tuple(torch.from_numpy(value) for value in (x, y, rgb, hard, activity))
    loaders=[DataLoader(Windows(ws, i == 0),batch_size=cfg["batch_size"],shuffle=i==0,num_workers=0) for i,ws in enumerate((train,val))]
    device=torch.device(cfg["device"]); model=TEMPORAL_MODELS[kind](scaler.n_features_in_,cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=1e-4)
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip(neg/np.maximum(pos,1),.1,20),dtype=torch.float32,device=device),reduction="none")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"kind": kind, "active_labels": active.tolist(),
                "demo_only": any(bool(r.get("demo_only", False)) for r in records.values()),
                "cfg": cfg, "labels": LABELS, "feature_dim": scaler.n_features_in_,
                "mean": scaler.mean_, "scale": scaler.scale_}
    epoch_dir = out.with_suffix(".epochs")
    if cfg.get("save_epoch_checkpoints"):
        epoch_dir.mkdir()
    best=float("inf"); best_state=None; history=[]
    for epoch in range(cfg["epochs"]):
        scores=[]
        for phase,loader in enumerate(loaders):
            model.train(phase==0); total=count=0
            for x,y,rgb,hard,activity in loader:
                x=x.to(device); y=y.to(device); mask=y>=0
                with torch.set_grad_enabled(phase==0):
                    image = rgb.to(device) if cfg["use_rgb"] else None
                    if phase == 0 and aux_weight:
                        logits, activity_logits = model(x, image, return_activity=True)
                    else:
                        logits = model(x, image)
                    targets = y.clamp(min=0)
                    losses = lossfn(logits, targets)
                    if kind == "cnn-lstm":
                        correct_probability = torch.where(targets == 1, logits.sigmoid(), (-logits).sigmoid())
                        losses = losses * (1 - correct_probability).square()
                    if phase == 0 and hard_negatives:
                        losses[:, :, LABELS.index("fall")] *= 1 + (adl_weight - 1) * hard.to(device)
                    loss = (losses * mask).sum() / mask.sum().clamp(min=1)
                    if phase == 0 and aux_weight:
                        activity = activity.to(device)
                        known = activity >= 0
                        activity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            activity_logits, activity.clamp(min=0), reduction="none")
                        loss = loss + aux_weight * (activity_loss * known).sum() / known.sum().clamp(min=1)
                    if phase==0:
                        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                total+=loss.item()*mask.sum().item(); count+=mask.sum().item()
            scores.append(total/max(count,1))
        history.append({"epoch":epoch+1,"train_loss":scores[0],"val_loss":scores[1]})
        print(json.dumps(history[-1]),flush=True)
        if cfg.get("save_epoch_checkpoints"):
            torch.save(dict(artifact, state_dict=model.state_dict(), epoch=epoch+1,
                            thresholds=[cfg["on_threshold"][i] if enabled else 1.
                                        for i, enabled in enumerate(active)]),
                       epoch_dir / f"epoch-{epoch+1:02}.pt")
        if scores[1]<best:
            best=scores[1]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval(); yy=[]; pp=[]
    with torch.no_grad():
        for x,y,rgb,_,_ in loaders[1]:
            logits=model(x.to(device),rgb.to(device) if cfg["use_rgb"] else None)
            pp.append(logits[:,-1].sigmoid().cpu().numpy()); yy.append(y[:,-1].numpy())
    y,p=np.concatenate(yy),np.concatenate(pp); thresholds=tune(y,p,active)
    torch.save(dict(artifact, state_dict=best_state, thresholds=thresholds), out)
    out.with_suffix(".validation.json").write_text(json.dumps({"history":history,"metrics":metrics(y,p,thresholds)},indent=2),encoding="utf-8")


class Predictor:
    def __init__(self,path,device="cpu"):
        self.device=device
        if str(path).endswith(".joblib"):
            self.artifact=joblib.load(path)
        else:
            import torch
            from .models import TEMPORAL_MODELS
            # Load only artifacts produced by this project and trusted by you.
            self.artifact=torch.load(path,map_location="cpu",weights_only=False)
            a=self.artifact
            kind = a.get("kind", "tcn")
            if kind not in TEMPORAL_MODELS:
                raise ValueError(f"Unsupported temporal checkpoint: {kind}")
            self.model=TEMPORAL_MODELS[kind](a["feature_dim"],a["cfg"]).to(device)
            self.model.load_state_dict(a["state_dict"]); self.model.eval()
        if self.artifact["labels"]!=LABELS: raise ValueError("Checkpoint label order differs")
        self.active_labels = np.asarray(self.artifact.get("active_labels", [True] * len(LABELS)), dtype=bool)
        self.cfg=dict(self.artifact["cfg"])
        self.cfg["on_threshold"]=self.artifact["thresholds"]
        self.cfg["off_threshold"]=[min(off,on*.7) for off,on in zip(self.cfg["off_threshold"],self.cfg["on_threshold"])]
    def predict(self,features,crops=None):
        a=self.artifact; x=np.asarray(features,np.float32)[-self.cfg["window"]:]
        if x.shape[1]!=a["feature_dim"]: raise ValueError("Feature schema mismatch")
        if a["kind"]=="baseline":
            z=a["scaler"].transform(summary(x)[None])
            return np.array([m.predict_proba(z)[0,list(m.classes_).index(1)] for m in a["models"]])
        import torch
        x=(x-a["mean"])/a["scale"]
        x=torch.tensor(padded(x.astype(np.float32),self.cfg["window"])[None],device=self.device)
        rgb=None
        if self.cfg["use_rgb"]:
            if crops is None: raise ValueError("RGB checkpoint requires crops")
            images=padded(np.asarray(crops)[-self.cfg["window"]:],self.cfg["window"])
            rgb=torch.tensor(images.transpose(0,3,1,2)[None],dtype=torch.float32,device=self.device)/255.
        with torch.no_grad():
            scores = self.model(x,rgb)[0,-1].sigmoid().cpu().numpy()
        scores[~self.active_labels] = 0
        return scores


def evaluate(manifest,checkpoint,out,device="cpu"):
    predictor=Predictor(checkpoint,device); cfg=predictor.cfg
    entries=[e for e in read_manifest(manifest) if e["split"]=="test"]
    ws=windows(entries,cfg)
    if not ws: raise ValueError("No usable test windows")
    y=[]; p=[]
    for r,a,b,_ in ws:
        y.append(r["labels"][b]); p.append(predictor.predict(r["features"][a:b+1],r.get("crops",None)[a:b+1] if "crops" in r else None))
    y = np.array(y)
    y[:, ~predictor.active_labels] = -1
    result={"evaluation_unit":"sampled_window_endpoint","threshold_source":"validation",
            "metrics":metrics(y,np.array(p),cfg["on_threshold"])}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


