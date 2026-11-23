from pathlib import Path
import yaml

LABELS = ["fall", "stumble", "collapse_to_sit", "cough_sneeze",
          "holding_suspected_pain_area", "vomit_like", "seizure_like"]

FEATURE_CONFIG_KEYS = ["sample_hz", "keypoint_conf", "max_hold_sec", "reset_gap_sec",
                       "min_valid_fraction", "min_person_height", "crop_size", "crop_margin"]

DEFAULT = {
    "pose_model": "yolo11n-pose.pt", "tracker": "bytetrack.yaml",
    "device": "cpu", "image_size": 640, "detection_conf": 0.25,
    "sample_hz": 15.0, "window": 128, "min_history": 16,
    "keypoint_conf": 0.3, "max_hold_sec": 0.2, "reset_gap_sec": 0.75,
    "min_valid_fraction": 0.55, "min_person_height": 40,
    "crop_size": 96, "crop_margin": 0.15,
    "use_rgb": False, "channels": 128, "dilations": [1, 2, 4, 8, 16],
    "dropout": 0.1, "seed": 42, "epochs": 30, "batch_size": 16,
    "learning_rate": 0.001, "stride": 8,
    "fall_only": False, "adl_negative_weight": 1.0,
    "pose_dropout_probability": 0.0, "pose_dropout_max_frames": 3,
    "save_epoch_checkpoints": False, "activity_aux_weight": 0.0, "activity_aux_labels": ["bend", "sit", "lie"], "reviewed_activity_hard_negatives": True,
    "on_threshold": [0.65] * 7, "off_threshold": [0.4] * 7,
    "confirm_sec": [0.13, 0.13, 0.2, 0.07, 0.6, 0.2, 0.4],
    "release_sec": 0.3, "pre_roll_sec": 3.0, "post_roll_sec": 3.0,
    "max_clip_sec": 30.0, "max_people": 32,
}

def load_config(path=None):
    cfg = dict(DEFAULT)
    if path:
        supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(supplied) - set(DEFAULT)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(supplied)
    for name in ("on_threshold", "off_threshold", "confirm_sec"):
        if len(cfg[name]) != len(LABELS):
            raise ValueError(f"{name} must have 7 values")
    for lo, hi in zip(cfg["off_threshold"], cfg["on_threshold"]):
        if not 0 <= lo < hi <= 1:
            raise ValueError("Require 0 <= off_threshold < on_threshold <= 1")
    positive = ["sample_hz", "window", "min_history", "reset_gap_sec", "crop_size",
                "channels", "batch_size", "epochs", "stride", "max_clip_sec", "max_people"]
    if any(cfg[k] <= 0 for k in positive):
        raise ValueError("Rates, sizes and training counts must be positive")
    if not 1 <= cfg["adl_negative_weight"] < float("inf"):
        raise ValueError("adl_negative_weight must be finite and at least 1")
    aux_labels = cfg["activity_aux_labels"]
    if not isinstance(aux_labels, list) or not aux_labels or len(set(aux_labels)) != len(aux_labels) or not set(aux_labels) <= {"bend", "sit", "lie", "walking", "standing"}:
        raise ValueError("Invalid activity_aux_labels")
    if not 0 <= cfg["activity_aux_weight"] < float("inf"):
        raise ValueError("activity_aux_weight must be finite and nonnegative")
    if not 0 <= cfg["pose_dropout_probability"] <= 1:
        raise ValueError("pose_dropout_probability must be between 0 and 1")
    gap_frames = cfg["pose_dropout_max_frames"]
    if isinstance(gap_frames, bool) or not isinstance(gap_frames, int) or gap_frames < 1:
        raise ValueError("pose_dropout_max_frames must be a positive integer")
    if cfg["pose_dropout_probability"] and (gap_frames + 1) / cfg["sample_hz"] > cfg["reset_gap_sec"]:
        raise ValueError("Simulated pose dropout must fit inside the track reset timeout")
    if cfg["min_history"] > cfg["window"]:
        raise ValueError("min_history must not exceed window")
    if not 0 <= cfg["min_valid_fraction"] <= 1:
        raise ValueError("Invalid valid fraction")
    return cfg
