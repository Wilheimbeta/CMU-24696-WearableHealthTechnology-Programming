#!/usr/bin/env python
"""
Experiment 8/8.5/8.6: Evaluate HAR models on custom-collected Xsens data.

Mode A  (cross):   Use HHAR-trained models to classify custom data (no retraining).
Mode B  (loso):    Leave-one-subject-out within custom data only.
Mode C  (reverse): Train on custom data, test on all HHAR subjects (reverse of Mode A).

Custom data has 5 classes and 6 channels from 4 Xsens sensors:
Acc_X/Y/Z + Roll/Pitch/Yaw.

Usage:
  # Mode A: cross-dataset (HHAR model -> custom data)
  python eval_custom_data.py --model hart --mode cross --device-type all --condition all --experiment-tag exp8a_hart_cross

  # Mode B: LOSO within custom data
  python eval_custom_data.py --model hart --mode loso --device-type all --condition all --experiment-tag exp8b_hart_loso

  # Mode C: reverse cross-dataset (custom data -> HHAR)
  python eval_custom_data.py --model hart --mode reverse --device-type all --condition all --experiment-tag exp86_run170_hart_all_ctrl

  # Mode D: cross-condition (train on one condition, test on another)
  python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross-condition --device-type phone --train-condition control --test-condition uncontrolled --channels 6 --experiment-tag exp87_limuX_phone_ctrl2unc
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
HART_DIR = (BASE_DIR / "code"
            / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main"
            / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main")
LIMU_DIR = (BASE_DIR / "code"
            / "LIMU-BERT_Experience-main" / "LIMU-BERT_Experience-main")
SSL_DIR = (BASE_DIR / "code"
           / "ssl-wearables-main" / "ssl-wearables-main")
RESNET_BASE_DIR = BASE_DIR / "code" / "resnet-baseline"
RESULTS_DIR = BASE_DIR / "loso_results"
BEST_MODELS_DIR = RESULTS_DIR / "best_models"
CUSTOM_DATA_DIR = BASE_DIR / "custom_eval_data"

HHAR_LABELS_5 = ["Sitting", "Standing", "Walking", "Upstairs", "Downstairs"]
CUSTOM_LABELS = HHAR_LABELS_5.copy()
DEVICE_NAMES = {0: "right_wrist", 1: "left_wrist", 2: "right_pocket", 3: "left_pocket"}
CONDITION_ALIASES = {
    "control": "control",
    "controll": "control",
    "uncontrol": "uncontrolled",
    "uncontrolled": "uncontrolled",
    "all": "all",
}


def _resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)

HHAR_DIR_MAP_NOBIKE = {
    "phone": "hhar_nobike",
    "watch": "hhar_watch_nobike",
    "all": "hhar_all_nobike",
}


def _safe_torch_save(obj, path, retries=3, delay=0.5):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            tmp = p.with_suffix(".tmp"); torch.save(obj, str(tmp)); tmp.replace(p); return
        except (RuntimeError, OSError):
            if attempt < retries - 1: time.sleep(delay * (attempt + 1))
            else: raise


def load_custom_data(model_family: str, condition: str, device_type: str) -> dict:
    tag = f"{CONDITION_ALIASES.get(condition, condition)}_{device_type}"
    if model_family == "hart":
        base = CUSTOM_DATA_DIR / tag / "hart"
    elif model_family == "limu":
        base = CUSTOM_DATA_DIR / tag / "limu"
    else:
        base = CUSTOM_DATA_DIR / tag / "ssl"
    if not base.exists():
        print(f"ERROR: {base} not found. Run prepare_custom_data.py first.")
        sys.exit(1)
    data = {}
    for key in ["X", "X3", "X6", "Y", "pid", "D"]:
        p = base / f"{key}.npy"
        if p.exists():
            data[key] = np.load(str(p))
    return data


def fmt(v):
    if v is None: return ""
    return f"{float(v)*100:.2f}%"


def evaluate_predictions(y_true, y_pred, label_names, subject_ids=None, device_ids=None):
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist()
    report = classification_report(y_true, y_pred, labels=list(range(len(label_names))),
                                   target_names=label_names, output_dict=True, zero_division=0)
    per_class = {}
    for i, name in enumerate(label_names):
        if name in report:
            per_class[name] = {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }

    per_subject = {}
    if subject_ids is not None:
        for sid in sorted(np.unique(subject_ids)):
            mask = subject_ids == sid
            if mask.sum() == 0: continue
            per_subject[int(sid)] = {
                "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
                "f1_macro": round(float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 4),
                "n_samples": int(mask.sum()),
            }

    per_device = {}
    if device_ids is not None:
        for did in sorted(np.unique(device_ids)):
            mask = device_ids == did
            if mask.sum() == 0: continue
            per_device[DEVICE_NAMES.get(int(did), f"dev_{did}")] = {
                "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
                "f1_macro": round(float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 4),
                "n_samples": int(mask.sum()),
            }

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "per_class_f1": per_class,
        "confusion_matrix": cm,
        "per_subject": per_subject,
        "per_device": per_device,
        "n_samples": int(len(y_true)),
    }


def save_exp8_results(result, output_dir, experiment_tag, model_name):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_tag}_" if experiment_tag else ""
    path = output_dir / f"{prefix}{model_name}_custom_{ts}.json"
    if not isinstance(result.get("experiment_table_metadata"), dict):
        result["experiment_table_metadata"] = _build_experiment_table_metadata(result, experiment_tag)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved -> {path}")
    try:
        from summarize_experiment_results import update_for_experiment_tag
        out_csv = update_for_experiment_tag(experiment_tag, results_dir=output_dir)
        if out_csv:
            print(f"  Experiment summary CSV -> {out_csv}")
    except Exception as e:
        print(f"  (Experiment summary CSV skipped: {e})")
    return path


def print_summary(metrics, mode_label=""):
    print(f"\n{'='*60}")
    print(f"  {mode_label} Summary")
    print(f"{'='*60}")
    print(f"  Accuracy   : {fmt(metrics.get('accuracy'))}")
    print(f"  F1 weighted: {fmt(metrics.get('f1_weighted'))}")
    print(f"  F1 macro   : {fmt(metrics.get('f1_macro'))}")
    if metrics.get("per_class_f1"):
        print("  Per-class:")
        for cls, m in metrics["per_class_f1"].items():
            print(f"    {cls:<12} F1={fmt(m.get('f1'))}  P={fmt(m.get('precision'))}  R={fmt(m.get('recall'))}")
    if metrics.get("per_subject"):
        print("  Per-subject:")
        for sid, m in metrics["per_subject"].items():
            print(f"    Subject {sid}: Acc={fmt(m['accuracy'])}  F1m={fmt(m['f1_macro'])}  n={m['n_samples']}")
    if metrics.get("per_device"):
        print("  Per-device:")
        for dev, m in metrics["per_device"].items():
            print(f"    {dev:<14} Acc={fmt(m['accuracy'])}  F1m={fmt(m['f1_macro'])}  n={m['n_samples']}")
    print(f"{'='*60}")


def _to_model_channels(X: np.ndarray, channels: int) -> np.ndarray:
    """
    Keep tensors at 6 dims for model compatibility.
    If channels=3, zero the last three channels.
    """
    if channels == 6:
        return X
    X_out = X.copy()
    X_out[:, :, 3:] = 0.0
    return X_out


def _extract_run_number(experiment_tag: str | None):
    if not experiment_tag:
        return None
    m = re.search(r"(?:^|_)run(\d+)(?:_|$)", str(experiment_tag), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _infer_architecture(model: str) -> str:
    model = str(model).lower()
    if "limu-bert" in model:
        return "LIMU-BERT-X"
    if "ssl-wearables" in model:
        return "SSL-Wearables"
    if "resnet-baseline" in model:
        return "ResNet-Baseline"
    if "hart" in model:
        return "HART"
    return model


def _infer_additional_pretraining(pretrained) -> str:
    text = str(pretrained or "").lower()
    if not text:
        return "None"
    if "watch_finetune" in text or "watchft" in text or "watch_scratch" in text or "watchscratch" in text:
        return "PAMAP2 + WISDM (Watch only)"
    if "phone_finetune" in text or "phoneft" in text or "phone_scratch" in text or "phonescratch" in text:
        return "SBHAR + WISDM (Phone only)"
    if "wisdm_all" in text or "wisdmft" in text or "wisdmscratch" in text:
        return "WISDM (Watch and Phone)"
    return "None"


def _build_experiment_table_metadata(result: dict, experiment_tag: str | None) -> dict:
    exp_tag = str(experiment_tag or "")
    is_exp85 = exp_tag.lower().startswith("exp85")
    mode = str(result.get("mode", "")).lower()
    condition = str(result.get("condition", "")).lower()
    test_map = {"control": "Our Controlled", "uncontrolled": "Our Uncontrolled", "all": "Our All"}
    test_ds_by_condition = test_map.get(condition, "Custom Data")
    if mode == "cross":
        train_ds = "HHAR"
        test_ds = test_ds_by_condition
        metrics = result.get("test_metrics", {})
        n_tests = metrics.get("repeat_summary", {}).get("num_repeats", 1) if isinstance(metrics, dict) else 1
        testing_method = "Cross"
    elif mode == "reverse":
        train_ds = test_ds_by_condition
        test_ds = "HHAR"
        n_tests = 1
        testing_method = "Cross"
    else:
        train_ds = "None" if is_exp85 else "Custom Data"
        test_ds = test_ds_by_condition
        n_tests = result.get("num_folds", 1)
        testing_method = "LOSO"
    return {
        "Architecture Type": _infer_architecture(result.get("model", "")),
        "Device Type": str(result.get("device_type", "")).capitalize(),
        "Training Dataset": train_ds,
        "Testing Dataset": test_ds,
        "Imported Pretrained Weights": "Yes" if result.get("pretrained") else "No",
        "3ch vs 6ch data": f"{result.get('channels')}ch" if result.get("channels") else "",
        "Data Fraction": "100%",
        "Additional Pretraining dataset": _infer_additional_pretraining(result.get("pretrained")),
        "Testing Method": testing_method,
        "# of Tests": int(n_tests) if n_tests else 1,
        "# Run": _extract_run_number(experiment_tag) or "",
    }


def _load_optional_loso_pretrained(model, args, model_family: str, device) -> bool:
    """
    Load optional pretrained weights for LOSO fine-tuning.
    Uses partial shape-matched loading to tolerate classifier-head mismatches.
    """
    import torch

    if not args.pretrained:
        return True
    ckpt = _resolve_pretrained_path(args.pretrained, model_family)
    if ckpt is None or not Path(ckpt).exists():
        print(f"ERROR: Pretrained checkpoint not found for LOSO: {args.pretrained}")
        return False
    print(f"  Loading LOSO pretrained: {ckpt}")

    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        print("ERROR: Unsupported checkpoint format for LOSO pretrained loading.")
        return False

    model_state = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state)
    print(f"  Loaded {loaded}/{len(model_state)} tensors for LOSO init")
    return True


def _pick_ckpt(candidates, device_type: str, prefer_tokens=None):
    """
    Prefer checkpoints that match the requested HHAR device split.
    Falls back to newest file among candidates.
    """
    if not candidates:
        return None
    prefer_tokens = [t.lower() for t in (prefer_tokens or [])]
    dt = device_type.lower()

    def _score(path: Path):
        name = path.name.lower()
        score = 0
        if f"_{dt}_" in name or name.endswith(f"_{dt}.pt") or name.endswith(f"_{dt}.weights.h5"):
            score += 100
        for tok in prefer_tokens:
            if tok and tok in name:
                score += 10
        return (score, path.stat().st_mtime)

    return max(candidates, key=_score)


def _resolve_pretrained_path(user_path: str | None, model_family: str):
    """Resolve a pretrained checkpoint path, with model-specific fallbacks.

    Tries (in order):
      1. The path as-given (if it exists).
      2. Resolved against project root (if it exists).
      3. Model-specific fallbacks under code/<model_dir>/.
    Falls back to the original resolved path if nothing found (caller should
    check .exists()).
    Note: model_family fallbacks are tried even when the input path is
    absolute, because main() pre-resolves --pretrained to an absolute path
    rooted at the project, which may not exist if the file actually lives
    inside code/<model>/weights or code/<model>/model_check_point.
    """
    if not user_path:
        return None
    raw = Path(user_path)
    p = _resolve_project_path(user_path)
    if p.exists():
        return p

    # Use the basename / project-relative tail for the model-specific fallbacks
    # so absolute paths like "D:\CodeWHT\weights\limu_bert_x" still work.
    name = raw.name
    stem = raw.stem
    # Strip a leading project-root prefix if present (so relative-tail like
    # "weights/limu_bert_x" can be used under LIMU_DIR / SSL_DIR).
    try:
        rel_tail = p.relative_to(BASE_DIR)
    except ValueError:
        rel_tail = raw

    if model_family == "limu":
        cands = [
            LIMU_DIR / rel_tail,
            Path(str(LIMU_DIR / rel_tail) + ".pt"),
            LIMU_DIR / "weights" / name,
            LIMU_DIR / "weights" / (stem + ".pt"),
            Path(str(p) + ".pt"),
        ]
    elif model_family == "ssl":
        cands = [
            SSL_DIR / rel_tail,
            SSL_DIR / "model_check_point" / name,
        ]
    elif model_family == "hart":
        cands = [HART_DIR / rel_tail]
    elif model_family == "resnet":
        cands = [RESNET_BASE_DIR / rel_tail]
    else:
        cands = []

    for cand in cands:
        if cand.exists():
            return cand
    return p


def _aggregate_cross_repeats(metrics_runs):
    if len(metrics_runs) == 1:
        return metrics_runs[0]
    accs = [m["accuracy"] for m in metrics_runs]
    f1w = [m["f1_weighted"] for m in metrics_runs]
    f1m = [m["f1_macro"] for m in metrics_runs]
    out = copy.deepcopy(metrics_runs[0])
    out["accuracy"] = float(np.mean(accs))
    out["f1_weighted"] = float(np.mean(f1w))
    out["f1_macro"] = float(np.mean(f1m))
    all_cls = set()
    for m in metrics_runs:
        all_cls.update(m.get("per_class_f1", {}).keys())
    if all_cls and "per_class_f1" in out:
        for cls in all_cls:
            f1_vals = [m.get("per_class_f1", {}).get(cls, {}).get("f1")
                       for m in metrics_runs]
            f1_vals = [v for v in f1_vals if v is not None]
            rec_vals = [m.get("per_class_f1", {}).get(cls, {}).get("recall")
                        for m in metrics_runs]
            rec_vals = [v for v in rec_vals if v is not None]
            if cls not in out["per_class_f1"]:
                out["per_class_f1"][cls] = {}
            if f1_vals:
                out["per_class_f1"][cls]["f1"] = float(np.mean(f1_vals))
            if rec_vals:
                out["per_class_f1"][cls]["recall"] = float(np.mean(rec_vals))
    out["repeat_summary"] = {
        "num_repeats": len(metrics_runs),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1_weighted": float(np.mean(f1w)),
        "std_f1_weighted": float(np.std(f1w)),
        "mean_f1_macro": float(np.mean(f1m)),
        "std_f1_macro": float(np.std(f1m)),
    }
    out["repeat_runs"] = metrics_runs
    return out


def load_hhar_test_data(model_family: str, device_type: str, channels: int,
                        limu_seq_len: int = 20):
    """Load all HHAR subjects (no-bike, 5-class) as test data for reverse eval."""
    dt = device_type
    nobike_tag = "_nobike"

    if model_family == "hart":
        import hickle as hkl
        hhar_suffix = {
            "all": f"HHAR{nobike_tag}",
            "phone": f"HHAR_phone{nobike_tag}",
            "watch": f"HHAR_watch{nobike_tag}",
        }
        hhar_dir = HART_DIR / "datasets" / "datasetStandardized" / hhar_suffix[dt]
        hhar_map_name = f"hart_subject_map{'_' + dt if dt != 'all' else ''}{nobike_tag}.json"
        hhar_map = BASE_DIR / hhar_map_name
        if not hhar_map.exists():
            print(f"ERROR: {hhar_map} not found. Run prepare_hhar_data.py --no-bike first.")
            sys.exit(1)
        with open(hhar_map) as f:
            sm = json.load(f)
        n_hhar = len(sm)
        X = np.vstack([hkl.load(str(hhar_dir / f"UserData{i}.hkl")) for i in range(n_hhar)])
        y = np.hstack([hkl.load(str(hhar_dir / f"UserLabel{i}.hkl")) for i in range(n_hhar)])
        if channels == 3:
            d_new = np.zeros_like(X); d_new[:, :, :3] = X[:, :, :3]; X = d_new

    elif model_family == "limu":
        hhar_dir_map = {
            "phone": f"hhar{nobike_tag}",
            "watch": f"hhar_watch{nobike_tag}",
            "all": f"hhar_all{nobike_tag}",
        }
        hhar_dir = LIMU_DIR / "dataset" / hhar_dir_map[dt]
        data_path = hhar_dir / "data_20_120.npy"
        label_path = hhar_dir / "label_20_120.npy"
        if not data_path.exists() or not label_path.exists():
            print(f"ERROR: {data_path} not found. Run prepare_hhar_data.py --no-bike first.")
            sys.exit(1)
        X = np.load(str(data_path)).astype(np.float32)
        y = np.load(str(label_path)).astype(np.float32)[:, 0, 2].astype(np.int64)
        X[:, :, :3] /= 9.8
        X = _to_model_channels(X, channels)
        if limu_seq_len < X.shape[1]:
            s = (X.shape[1] - limu_seq_len) // 2
            X = X[:, s:s + limu_seq_len, :]

    else:  # ssl or resnet
        hhar_dir_map = {
            "phone": f"hhar{nobike_tag}",
            "watch": f"hhar_watch{nobike_tag}",
            "all": f"hhar_all{nobike_tag}",
        }
        hhar_dir = SSL_DIR / "data" / "downstream" / hhar_dir_map[dt]
        x_file = hhar_dir / ("X6.npy" if channels == 6 else "X.npy")
        y_file = hhar_dir / "Y.npy"
        if not x_file.exists() or not y_file.exists():
            print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py --no-bike first.")
            sys.exit(1)
        X = np.load(str(x_file)).astype(np.float32)
        y = np.load(str(y_file)).astype(np.int64)
        X = np.transpose(X, (0, 2, 1))  # (N, C, T)

    print(f"  HHAR test data loaded: X={X.shape}, y={y.shape}, classes={sorted(np.unique(y).tolist())}")
    return X, y


def _train_full_hhar_limu_ckpt(args, ckpt_path: Path, init_weights_path: Path | None = None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(LIMU_DIR))
    from models import BERTClassifier, ClassifierGRU
    from config import PretrainModelConfig, ClassifierModelConfig

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data_dir = LIMU_DIR / "dataset" / HHAR_DIR_MAP_NOBIKE[args.device_type]
    data_path = data_dir / "data_20_120.npy"
    label_path = data_dir / "label_20_120.npy"
    if not data_path.exists() or not label_path.exists():
        print(f"ERROR: {data_path} not found. Run prepare_hhar_data.py --no-bike first.")
        return None

    X_all = np.load(str(data_path)).astype(np.float32)
    y_all = np.load(str(label_path)).astype(np.float32)[:, 0, 2].astype(np.int64)
    X_all[:, :, :3] /= 9.8
    X_all = _to_model_channels(X_all, args.channels)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=args.seed, stratify=y_all
    )
    seq_len = args.limu_seq_len
    data_seq_len = X_train.shape[1]

    class _CropDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, target_len, random_crop=True):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
            self.target_len = target_len
            self.random_crop = random_crop
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx]
            if self.target_len < x.size(0):
                if self.random_crop:
                    s = torch.randint(0, x.size(0) - self.target_len, (1,)).item()
                else:
                    s = (x.size(0) - self.target_len) // 2
                x = x[s:s + self.target_len]
            return x, self.y[idx]

    def _make_loader(X, y, shuffle=False, random_crop=False):
        if seq_len < data_seq_len:
            ds = _CropDataset(X, y, seq_len, random_crop=random_crop)
        else:
            ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=args.cross_train_batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(X_train, y_train, shuffle=True, random_crop=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False, random_crop=False)

    bert_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6, n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
    cls_cfg = ClassifierModelConfig(seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1], rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[], flat_num=0, num_attn=0, num_head=0, atten_hidden=0, num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)
    classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=5)
    model = BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False).to(device)
    if init_weights_path is not None and Path(init_weights_path).exists():
        print(f"  Init LIMU weights from: {Path(init_weights_path).name}")
        state = torch.load(str(init_weights_path), map_location=device)
        if isinstance(state, dict):
            msd = model.state_dict()
            loaded = 0
            for k, v in state.items():
                if k in msd and msd[k].shape == v.shape:
                    msd[k] = v; loaded += 1
            model.load_state_dict(msd)
            print(f"    Loaded {loaded}/{len(state)} LIMU init tensors")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"  Training HHAR full-data LIMU checkpoint: {ckpt_path.name}")
    best_state, best_f1, no_improve = None, -1.0, 0
    pbar = tqdm(range(args.cross_train_epochs), desc="    HHAR LIMU train", unit="ep")
    for epoch in pbar:
        model.train()
        tr_loss = 0.0
        tr_steps = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx, True), by)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item())
            tr_steps += 1

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device), False)
                vp.append(logits.argmax(1).cpu().numpy())
                vt.append(by.numpy())
        vf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average="macro", zero_division=0)
        avg_loss = tr_loss / max(tr_steps, 1)
        pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_f1*100:.2f}%")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _safe_torch_save(best_state, ckpt_path)
            tqdm.write(f"    [OK] Saved LIMU best -> {ckpt_path.name} (F1m={vf1*100:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 10:
            tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x10)")
            break
    pbar.close()
    return ckpt_path if ckpt_path.exists() else None


def _train_full_hhar_resnet_ckpt(args, ckpt_path: Path, init_weights_path: Path | None = None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(RESNET_BASE_DIR))
    from resnet1d_baseline import ResNet1DBaseline

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data_dir = SSL_DIR / "data" / "downstream" / HHAR_DIR_MAP_NOBIKE[args.device_type]
    x_file = data_dir / ("X6.npy" if args.channels == 6 else "X.npy")
    y_file = data_dir / "Y.npy"
    if not x_file.exists() or not y_file.exists():
        print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py --no-bike first.")
        return None

    X_all = np.load(str(x_file)).astype(np.float32)
    y_all = np.load(str(y_file)).astype(np.int64)
    X_all = np.transpose(X_all, (0, 2, 1))  # (N, C, T)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=args.seed, stratify=y_all
    )

    def _make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=args.cross_train_batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)

    model = ResNet1DBaseline(n_channels=args.channels, num_classes=5, kernel_size=5).to(device)
    if init_weights_path is not None and Path(init_weights_path).exists():
        print(f"  Init ResNet weights from: {Path(init_weights_path).name}")
        state = torch.load(str(init_weights_path), map_location=device)
        if isinstance(state, dict):
            msd = model.state_dict()
            loaded = 0
            for k, v in state.items():
                if k in msd and msd[k].shape == v.shape:
                    msd[k] = v; loaded += 1
            model.load_state_dict(msd)
            print(f"    Loaded {loaded}/{len(state)} ResNet init tensors")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    print(f"  Training HHAR full-data ResNet checkpoint: {ckpt_path.name}")
    best_state, best_f1, no_improve = None, -1.0, 0
    pbar = tqdm(range(args.cross_train_epochs), desc="    HHAR ResNet train", unit="ep")
    for epoch in pbar:
        model.train()
        tr_loss = 0.0
        tr_steps = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item())
            tr_steps += 1

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device))
                vp.append(logits.argmax(1).cpu().numpy())
                vt.append(by.numpy())
        vf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average="macro", zero_division=0)
        avg_loss = tr_loss / max(tr_steps, 1)
        pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_f1*100:.2f}%")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _safe_torch_save(best_state, ckpt_path)
            tqdm.write(f"    [OK] Saved ResNet best -> {ckpt_path.name} (F1m={vf1*100:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 10:
            tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x10)")
            break
    pbar.close()
    return ckpt_path if ckpt_path.exists() else None


def _train_full_hhar_hart_ckpt(args, ckpt_path: Path, init_weights_path: Path | None = None):
    """Train HART on full HHAR (no-bike, 5-class) for cross-dataset eval."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(HART_DIR))
    import tensorflow as tf
    import model as hart_model
    import hickle as hkl

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.backend.clear_session()
    tf.random.set_seed(args.seed); np.random.seed(args.seed)

    dt = args.device_type
    hhar_suffix = {
        "all": "HHAR_nobike",
        "phone": "HHAR_phone_nobike",
        "watch": "HHAR_watch_nobike",
    }
    hhar_dir = HART_DIR / "datasets" / "datasetStandardized" / hhar_suffix[dt]
    hhar_map_name = f"hart_subject_map{'_' + dt if dt != 'all' else ''}_nobike.json"
    hhar_map = BASE_DIR / hhar_map_name
    if not hhar_map.exists():
        print(f"ERROR: {hhar_map} not found. Run prepare_hhar_data.py --no-bike first.")
        return None
    with open(hhar_map) as f:
        sm = json.load(f)
    n_hhar = len(sm)
    X_all = np.vstack([hkl.load(str(hhar_dir / f"UserData{i}.hkl")) for i in range(n_hhar)])
    y_all = np.hstack([hkl.load(str(hhar_dir / f"UserLabel{i}.hkl")) for i in range(n_hhar)])
    if args.channels == 3:
        d_new = np.zeros_like(X_all); d_new[:, :, :3] = X_all[:, :, :3]; X_all = d_new

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=args.seed, stratify=y_all
    )
    y_train_oh = tf.keras.utils.to_categorical(y_train, 5)
    y_val_oh = tf.keras.utils.to_categorical(y_val, 5)

    mdl = hart_model.HART((128, 6), 5)
    mdl.build(input_shape=(None, 128, 6))
    if init_weights_path is not None and Path(init_weights_path).exists():
        try:
            print(f"  Init HART weights from: {Path(init_weights_path).name}")
            mdl.load_weights(str(init_weights_path))
        except Exception as e:
            print(f"  WARNING: failed to load HART init weights: {e}")

    optimizer = tf.keras.optimizers.Adam(5e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    mdl.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    batch_size = 256
    epochs = args.cross_train_epochs
    print(f"  Training HHAR full-data HART checkpoint: {ckpt_path.name}")
    best_f1, no_improve = -1.0, 0
    patience = 10
    for epoch in range(epochs):
        mdl.fit(X_train, y_train_oh, batch_size=batch_size, epochs=1,
                verbose=0, shuffle=True)
        vp = np.argmax(mdl.predict(X_val, verbose=0), axis=-1)
        vf1 = f1_score(y_val, vp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            mdl.save_weights(str(ckpt_path))
            print(f"    [OK] Saved HART best -> {ckpt_path.name} (F1m={vf1*100:.2f}%) ep{epoch+1}")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1} (no improve x{patience})")
            break
    return ckpt_path if ckpt_path.exists() else None


def _train_full_hhar_ssl_ckpt(args, ckpt_path: Path, init_weights_path: Path | None = None):
    """Train ssl-wearables ResNet on full HHAR (no-bike, 5-class) for cross-dataset eval."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(SSL_DIR))
    from sslearning.models.accNet import Resnet

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data_dir = SSL_DIR / "data" / "downstream" / HHAR_DIR_MAP_NOBIKE[args.device_type]
    x_file = data_dir / ("X6.npy" if args.channels == 6 else "X.npy")
    y_file = data_dir / "Y.npy"
    if not x_file.exists() or not y_file.exists():
        print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py --no-bike first.")
        return None

    X_all = np.load(str(x_file)).astype(np.float32)
    y_all = np.load(str(y_file)).astype(np.int64)
    X_all = np.transpose(X_all, (0, 2, 1))  # (N, C, T)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=args.seed, stratify=y_all
    )

    def _make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=args.cross_train_batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)

    model = Resnet(output_size=5, n_channels=args.channels, is_eva=True,
                   resnet_version=1, epoch_len=10).to(device)
    if init_weights_path is not None and Path(init_weights_path).exists():
        print(f"  Init SSL weights from: {Path(init_weights_path).name}")
        try:
            pre = torch.load(str(init_weights_path), map_location=device)
            if isinstance(pre, dict):
                head = next(iter(pre)).split(".")[0]
                if head == "module":
                    pre = {k.partition("module.")[2]: v for k, v in pre.items()}
                msd = model.state_dict()
                loaded = 0
                for k, v in pre.items():
                    if k in msd and k.split(".")[0] != "classifier" and msd[k].shape == v.shape:
                        msd[k] = v; loaded += 1
                model.load_state_dict(msd)
                print(f"    Loaded {loaded}/{len(pre)} SSL init tensors (classifier skipped)")
        except Exception as e:
            print(f"  WARNING: failed to load SSL init weights: {e}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    print(f"  Training HHAR full-data SSL checkpoint: {ckpt_path.name}")
    best_state, best_f1, no_improve = None, -1.0, 0
    pbar = tqdm(range(args.cross_train_epochs), desc="    HHAR SSL train", unit="ep")
    for epoch in pbar:
        model.train()
        tr_loss = 0.0; tr_steps = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()); tr_steps += 1
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                vp.append(model(bx.to(device)).argmax(1).cpu().numpy())
                vt.append(by.numpy())
        vf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average="macro", zero_division=0)
        avg_loss = tr_loss / max(tr_steps, 1)
        pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_f1*100:.2f}%")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _safe_torch_save(best_state, ckpt_path)
            tqdm.write(f"    [OK] Saved SSL best -> {ckpt_path.name} (F1m={vf1*100:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 10:
            tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x10)")
            break
    pbar.close()
    return ckpt_path if ckpt_path.exists() else None


def cross_eval_hart(args, custom_data):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(HART_DIR))
    import tensorflow as tf
    import model as hart_model

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    X_test = _to_model_channels(custom_data["X"], args.channels)
    y_test = custom_data["Y"]
    P_test = custom_data["pid"]
    D_test = custom_data["D"]

    init_weights = _resolve_pretrained_path(getattr(args, "pretrained", None), "hart")
    exp_tag = args.experiment_tag or "run"
    ckpt = BEST_MODELS_DIR / f"hart_fullhhar_{exp_tag}_{args.device_type}_{args.channels}ch_seed{args.seed}.weights.h5"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trained = _train_full_hhar_hart_ckpt(args, ckpt, init_weights_path=init_weights)
    if trained is None or not ckpt.exists():
        print(f"ERROR: Failed to train HART HHAR checkpoint: {ckpt}"); return None
    print(f"  Loading HART checkpoint: {ckpt.name}")
    tf.keras.backend.clear_session()
    mdl = hart_model.HART((128, 6), 5)
    mdl.build(input_shape=(None, 128, 6))
    mdl.load_weights(str(ckpt))

    probs = mdl.predict(X_test, verbose=0)
    y_pred_5cls = np.argmax(probs, axis=-1)
    metrics = evaluate_predictions(y_test, y_pred_5cls, CUSTOM_LABELS, P_test, D_test)
    return metrics


def cross_eval_limu(args, custom_data, retrain_hhar=True):
    import torch
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(LIMU_DIR))
    from models import BERTClassifier, ClassifierGRU
    from config import PretrainModelConfig, ClassifierModelConfig

    X_test = _to_model_channels(custom_data["X"].astype(np.float32), args.channels)
    X_test[:, :, :3] /= 9.8
    y_test = custom_data["Y"]
    P_test = custom_data["pid"]
    D_test = custom_data["D"]

    seq_len = args.limu_seq_len
    bert_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6, n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
    cls_cfg = ClassifierModelConfig(seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1], rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[], flat_num=0, num_attn=0, num_head=0, atten_hidden=0, num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)
    classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=5)
    model = BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False).to(device)

    init_weights = _resolve_pretrained_path(getattr(args, "pretrained", None), "limu")
    exp_tag = args.experiment_tag or "run"
    ckpt = BEST_MODELS_DIR / f"limubert_fullhhar_{exp_tag}_{args.device_type}_seq{seq_len}_{args.channels}ch_seed{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if retrain_hhar or not ckpt.exists():
        trained_ckpt = _train_full_hhar_limu_ckpt(args, ckpt, init_weights_path=init_weights)
        if trained_ckpt is None:
            print("ERROR: Failed to train full-HHAR LIMU checkpoint"); return None
    if not ckpt.exists():
        print(f"ERROR: LIMU checkpoint not found: {ckpt}"); return None
    print(f"  Loading LIMU checkpoint: {ckpt.name}")
    state = torch.load(str(ckpt), map_location=device)
    # Allow loading base LIMU pretrained weights with partial tensor match.
    if isinstance(state, dict):
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded}/{len(state)} LIMU tensors")
    else:
        print("ERROR: Unsupported LIMU checkpoint format"); return None
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset
    if seq_len < X_test.shape[1]:
        s = (X_test.shape[1] - seq_len) // 2
        X_test = X_test[:, s:s+seq_len, :]
    ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    all_pred = []
    with torch.no_grad():
        for bx, _ in loader:
            logits = model(bx.to(device), False)
            all_pred.append(logits.argmax(1).cpu().numpy())
    y_pred_5cls = np.concatenate(all_pred)
    metrics = evaluate_predictions(y_test, y_pred_5cls, CUSTOM_LABELS, P_test, D_test)
    return metrics


def cross_eval_ssl(args, custom_data, retrain_hhar=True):
    import torch
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(SSL_DIR))
    from sslearning.models.accNet import Resnet

    n_ch = args.channels
    X_key = "X6" if n_ch == 6 else "X3"
    X_test = custom_data[X_key].astype(np.float32)
    X_test = np.transpose(X_test, (0, 2, 1))
    y_test = custom_data["Y"]
    P_test = custom_data["pid"]
    D_test = custom_data["D"]

    model = Resnet(output_size=5, n_channels=n_ch, is_eva=True, resnet_version=1, epoch_len=10).to(device)
    init_weights = _resolve_pretrained_path(getattr(args, "pretrained", None), "ssl")
    exp_tag = args.experiment_tag or "run"
    ckpt = BEST_MODELS_DIR / f"ssl_fullhhar_{exp_tag}_{args.device_type}_{n_ch}ch_seed{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if retrain_hhar or not ckpt.exists():
        trained_ckpt = _train_full_hhar_ssl_ckpt(args, ckpt, init_weights_path=init_weights)
        if trained_ckpt is None:
            print("ERROR: Failed to train full-HHAR SSL checkpoint"); return None
    if not ckpt.exists():
        print(f"ERROR: SSL checkpoint not found: {ckpt}"); return None
    print(f"  Loading SSL checkpoint: {ckpt.name}")
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_pred = []
    with torch.no_grad():
        for bx, _ in loader:
            all_pred.append(model(bx.to(device)).argmax(1).cpu().numpy())
    y_pred_5cls = np.concatenate(all_pred)
    metrics = evaluate_predictions(y_test, y_pred_5cls, CUSTOM_LABELS, P_test, D_test)
    return metrics


def cross_eval_resnet(args, custom_data, retrain_hhar=True):
    import torch
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(RESNET_BASE_DIR))
    from resnet1d_baseline import ResNet1DBaseline

    n_ch = args.channels
    X_key = "X6" if n_ch == 6 else "X3"
    X_test = custom_data[X_key].astype(np.float32)
    X_test = np.transpose(X_test, (0, 2, 1))
    y_test = custom_data["Y"]
    P_test = custom_data["pid"]
    D_test = custom_data["D"]

    model = ResNet1DBaseline(n_channels=n_ch, num_classes=5, kernel_size=5).to(device)
    init_weights = _resolve_pretrained_path(getattr(args, "pretrained", None), "resnet")
    exp_tag = args.experiment_tag or "run"
    ckpt = BEST_MODELS_DIR / f"resbase_fullhhar_{exp_tag}_{args.device_type}_{args.channels}ch_seed{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if retrain_hhar or not ckpt.exists():
        trained_ckpt = _train_full_hhar_resnet_ckpt(args, ckpt, init_weights_path=init_weights)
        if trained_ckpt is None:
            print("ERROR: Failed to train full-HHAR ResNet checkpoint"); return None
    if not ckpt.exists():
        print(f"ERROR: ResNet checkpoint not found: {ckpt}"); return None
    print(f"  Loading ResNet checkpoint: {ckpt.name}")
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_pred = []
    with torch.no_grad():
        for bx, _ in loader:
            all_pred.append(model(bx.to(device)).argmax(1).cpu().numpy())
    y_pred_5cls = np.concatenate(all_pred)
    metrics = evaluate_predictions(y_test, y_pred_5cls, CUSTOM_LABELS, P_test, D_test)
    return metrics


def loso_eval_hart(args, custom_data):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(HART_DIR))
    import tensorflow as tf
    import model as hart_model

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    X_all = _to_model_channels(custom_data["X"], args.channels)
    Y_all = custom_data["Y"]
    P_all = custom_data["pid"]
    D_all = custom_data["D"]
    subjects = sorted(np.unique(P_all).tolist())
    num_classes = len(CUSTOM_LABELS)
    epochs = args.epochs or 100
    batch_size = args.batch_size or 64
    lr = 5e-3

    fold_results = []
    for fold_idx, test_subj in enumerate(subjects):
        print(f"\n  LOSO Fold {fold_idx+1}/{len(subjects)}: test={test_subj}")
        tf.keras.backend.clear_session()
        np.random.seed(args.seed); tf.random.set_seed(args.seed)
        test_mask = P_all == test_subj
        X_train_raw, y_train_raw = X_all[~test_mask], Y_all[~test_mask]
        X_test, y_test = X_all[test_mask], Y_all[test_mask]
        D_test = D_all[test_mask]
        if len(X_test) == 0: continue
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_raw, y_train_raw, test_size=0.15, random_state=args.seed, stratify=y_train_raw)

        mdl = hart_model.HART((128, 6), num_classes)
        optimizer = tf.keras.optimizers.Adam(lr)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, tf.one_hot(y_train, num_classes)))
                    .shuffle(len(X_train), seed=args.seed).batch(batch_size).prefetch(tf.data.AUTOTUNE))
        best_val_f1, no_improve, patience = -1.0, 0, 10
        pbar = tqdm(range(epochs), desc=f"    Fold {fold_idx+1} train", unit="ep")
        for epoch in pbar:
            tr_loss = 0.0
            tr_steps = 0
            for bx, by in train_ds:
                with tf.GradientTape() as tape:
                    loss = loss_fn(by, mdl(bx, training=True))
                mdl.optimizer = optimizer
                optimizer.apply_gradients(zip(tape.gradient(loss, mdl.trainable_variables), mdl.trainable_variables))
                tr_loss += float(loss.numpy())
                tr_steps += 1
            vp = np.argmax(mdl.predict(X_val, verbose=0), axis=-1)
            vf1 = f1_score(y_val, vp, average="macro", zero_division=0)
            avg_loss = tr_loss / max(tr_steps, 1)
            pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_val_f1*100:.2f}%")
            if vf1 > best_val_f1:
                best_val_f1 = vf1; best_weights = mdl.get_weights(); no_improve = 0
                tqdm.write(f"    [OK] Epoch {epoch+1}: best val F1m={vf1*100:.2f}%")
            else:
                no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x{patience})")
                break
        pbar.close()
        if best_val_f1 > -1: mdl.set_weights(best_weights)
        y_pred = np.argmax(mdl.predict(X_test, verbose=0), axis=-1)
        fold_metrics = evaluate_predictions(y_test, y_pred, CUSTOM_LABELS, device_ids=D_test)
        fold_metrics["subject"] = int(test_subj)
        fold_results.append(fold_metrics)
        print(f"    Acc={fmt(fold_metrics['accuracy'])} F1m={fmt(fold_metrics['f1_macro'])}")

    return _aggregate_loso(fold_results)


def loso_eval_pytorch(args, custom_data, model_factory, model_family):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_family in ("ssl", "resnet"):
        n_ch = args.channels
        X_key = "X6" if n_ch == 6 else "X3"
        X_all = np.transpose(custom_data[X_key].astype(np.float32), (0, 2, 1))
    elif model_family == "limu":
        X_all = _to_model_channels(custom_data["X"].astype(np.float32), args.channels)
        X_all[:, :, :3] /= 9.8
        seq_len = args.limu_seq_len
        if seq_len < X_all.shape[1]:
            s = (X_all.shape[1] - seq_len) // 2
            X_all = X_all[:, s:s+seq_len, :]
    else:
        X_all = custom_data["X"].astype(np.float32)

    Y_all = custom_data["Y"]
    P_all = custom_data["pid"]
    D_all = custom_data["D"]
    subjects = sorted(np.unique(P_all).tolist())
    num_classes = len(CUSTOM_LABELS)
    epochs = args.epochs or 100
    batch_size = args.batch_size or 64
    lr = 1e-3

    fold_results = []
    for fold_idx, test_subj in enumerate(subjects):
        print(f"\n  LOSO Fold {fold_idx+1}/{len(subjects)}: test={test_subj}")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        test_mask = P_all == test_subj
        X_tr_raw, y_tr_raw = X_all[~test_mask], Y_all[~test_mask]
        X_test, y_test = X_all[test_mask], Y_all[test_mask]
        D_test_fold = D_all[test_mask]
        if len(X_test) == 0: continue
        idx = np.arange(len(X_tr_raw)); np.random.shuffle(idx)
        vn = max(1, int(len(idx) * 0.15))
        X_val, y_val = X_tr_raw[idx[:vn]], y_tr_raw[idx[:vn]]
        X_train, y_train = X_tr_raw[idx[vn:]], y_tr_raw[idx[vn:]]

        model = model_factory(num_classes).to(device)
        if not _load_optional_loso_pretrained(model, args, model_family, device):
            return {
                "num_folds": 0,
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "mean_f1_macro": 0.0,
                "std_f1_macro": 0.0,
                "per_fold": [],
            }
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                                  batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
                                batch_size=batch_size, shuffle=False, num_workers=0)

        best_val_f1, best_state, no_improve, patience = -1.0, None, 0, 10
        pbar = tqdm(range(epochs), desc=f"    Fold {fold_idx+1} train", unit="ep")
        for epoch in pbar:
            model.train()
            tr_loss = 0.0
            tr_steps = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx) if model_family != "limu" else model(bx, True), by)
                loss.backward(); optimizer.step()
                tr_loss += float(loss.item())
                tr_steps += 1
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    logits = model(bx.to(device)) if model_family != "limu" else model(bx.to(device), False)
                    vp.append(logits.argmax(1).cpu().numpy()); vt.append(by.numpy())
            vf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average="macro", zero_division=0)
            avg_loss = tr_loss / max(tr_steps, 1)
            pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_val_f1*100:.2f}%")
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                tqdm.write(f"    [OK] Epoch {epoch+1}: best val F1m={vf1*100:.2f}%")
            else:
                no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x{patience})")
                break
        pbar.close()

        if best_state: model.load_state_dict(best_state)
        model.eval(); tp = []
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
                                 batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for bx, _ in test_loader:
                logits = model(bx.to(device)) if model_family != "limu" else model(bx.to(device), False)
                tp.append(logits.argmax(1).cpu().numpy())
        y_pred = np.concatenate(tp)
        fold_metrics = evaluate_predictions(y_test, y_pred, CUSTOM_LABELS, device_ids=D_test_fold)
        fold_metrics["subject"] = int(test_subj)
        fold_results.append(fold_metrics)
        print(f"    Acc={fmt(fold_metrics['accuracy'])} F1m={fmt(fold_metrics['f1_macro'])}")
        del model; torch.cuda.empty_cache()

    return _aggregate_loso(fold_results)


def _aggregate_loso(fold_results):
    accs = [f["accuracy"] for f in fold_results]
    f1s = [f["f1_macro"] for f in fold_results]
    f1w = [f.get("f1_weighted", 0.0) for f in fold_results]
    return {
        "num_folds": len(fold_results),
        "mean_accuracy":    float(np.mean(accs)) if accs else 0.0,
        "std_accuracy":     float(np.std(accs))  if accs else 0.0,
        "mean_f1_weighted": float(np.mean(f1w))  if f1w  else 0.0,
        "std_f1_weighted":  float(np.std(f1w))   if f1w  else 0.0,
        "mean_f1_macro":    float(np.mean(f1s))  if f1s  else 0.0,
        "std_f1_macro":     float(np.std(f1s))   if f1s  else 0.0,
        "per_fold": fold_results,
    }


def reverse_eval_hart(args, custom_data):
    """Train HART on all custom subjects, test on HHAR."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(HART_DIR))
    import tensorflow as tf
    import model as hart_model

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    X_all = _to_model_channels(custom_data["X"], args.channels)
    Y_all = custom_data["Y"]
    num_classes = len(CUSTOM_LABELS)
    epochs = args.epochs or 100
    batch_size = args.batch_size or 64
    lr = 5e-3

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, Y_all, test_size=0.1, random_state=args.seed, stratify=Y_all)

    tf.keras.backend.clear_session()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    mdl = hart_model.HART((128, 6), num_classes)

    if args.pretrained:
        ckpt = _resolve_pretrained_path(args.pretrained, "hart")
        if ckpt and Path(ckpt).exists():
            print(f"  Loading pretrained HART weights: {Path(ckpt).name}")
            mdl.build(input_shape=(None, 128, 6))
            try:
                mdl.load_weights(str(ckpt))
                print("  Pretrained weights loaded")
            except Exception as e:
                print(f"  WARNING: Could not load pretrained weights: {e}")
        else:
            print(f"  WARNING: Pretrained checkpoint not found: {args.pretrained}")

    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, tf.one_hot(y_train, num_classes)))
                .shuffle(len(X_train), seed=args.seed).batch(batch_size).prefetch(tf.data.AUTOTUNE))

    best_val_f1, no_improve, patience = -1.0, 0, 10
    best_weights = None
    pbar = tqdm(range(epochs), desc="    Reverse HART train", unit="ep")
    for epoch in pbar:
        tr_loss = 0.0
        tr_steps = 0
        for bx, by in train_ds:
            with tf.GradientTape() as tape:
                loss = loss_fn(by, mdl(bx, training=True))
            mdl.optimizer = optimizer
            optimizer.apply_gradients(zip(tape.gradient(loss, mdl.trainable_variables), mdl.trainable_variables))
            tr_loss += float(loss.numpy())
            tr_steps += 1
        vp = np.argmax(mdl.predict(X_val, verbose=0), axis=-1)
        vf1 = f1_score(y_val, vp, average="macro", zero_division=0)
        avg_loss = tr_loss / max(tr_steps, 1)
        pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_val_f1*100:.2f}%")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_weights = mdl.get_weights()
            no_improve = 0
            tqdm.write(f"    [OK] Epoch {epoch+1}: best val F1m={vf1*100:.2f}%")
        else:
            no_improve += 1
        if no_improve >= patience:
            tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x{patience})")
            break
    pbar.close()
    if best_weights is not None:
        mdl.set_weights(best_weights)

    X_test, y_test = load_hhar_test_data("hart", args.device_type, args.channels)
    y_pred = np.argmax(mdl.predict(X_test, verbose=0), axis=-1)
    return evaluate_predictions(y_test, y_pred, HHAR_LABELS_5)


def reverse_eval_pytorch(args, custom_data, model_factory, model_family):
    """Train a PyTorch model on all custom subjects, test on HHAR."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_family in ("ssl", "resnet"):
        n_ch = args.channels
        X_key = "X6" if n_ch == 6 else "X3"
        X_all = np.transpose(custom_data[X_key].astype(np.float32), (0, 2, 1))
    elif model_family == "limu":
        X_all = _to_model_channels(custom_data["X"].astype(np.float32), args.channels)
        X_all[:, :, :3] /= 9.8
        seq_len = args.limu_seq_len
        if seq_len < X_all.shape[1]:
            s = (X_all.shape[1] - seq_len) // 2
            X_all = X_all[:, s:s + seq_len, :]
    else:
        X_all = custom_data["X"].astype(np.float32)

    Y_all = custom_data["Y"]
    num_classes = len(CUSTOM_LABELS)
    epochs = args.epochs or 100
    batch_size = args.batch_size or 64
    lr = 1e-3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    idx = np.arange(len(X_all))
    np.random.shuffle(idx)
    vn = max(1, int(len(idx) * 0.1))
    X_val, y_val = X_all[idx[:vn]], Y_all[idx[:vn]]
    X_train, y_train = X_all[idx[vn:]], Y_all[idx[vn:]]

    model = model_factory(num_classes).to(device)
    if not _load_optional_loso_pretrained(model, args, model_family, device):
        return None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
                            batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_f1, best_state, no_improve, patience = -1.0, None, 0, 10
    pbar = tqdm(range(epochs), desc="    Reverse train", unit="ep")
    for epoch in pbar:
        model.train()
        tr_loss = 0.0
        tr_steps = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx) if model_family != "limu" else model(bx, True), by)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item())
            tr_steps += 1
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device)) if model_family != "limu" else model(bx.to(device), False)
                vp.append(logits.argmax(1).cpu().numpy())
                vt.append(by.numpy())
        vf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average="macro", zero_division=0)
        avg_loss = tr_loss / max(tr_steps, 1)
        pbar.set_postfix_str(f"loss={avg_loss:.4f} val_f1m={vf1*100:.2f}% best={best_val_f1*100:.2f}%")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            tqdm.write(f"    [OK] Epoch {epoch+1}: best val F1m={vf1*100:.2f}%")
        else:
            no_improve += 1
        if no_improve >= patience:
            tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x{patience})")
            break
    pbar.close()

    if best_state:
        model.load_state_dict(best_state)

    X_test, y_test = load_hhar_test_data(model_family, args.device_type, args.channels, args.limu_seq_len)
    model.eval()
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
                             batch_size=batch_size, shuffle=False, num_workers=0)
    all_pred = []
    with torch.no_grad():
        for bx, _ in test_loader:
            logits = model(bx.to(device)) if model_family != "limu" else model(bx.to(device), False)
            all_pred.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(all_pred)
    metrics = evaluate_predictions(y_test, y_pred, HHAR_LABELS_5)
    del model
    torch.cuda.empty_cache()
    return metrics


def cross_condition_eval_pytorch(args, train_data, test_data, model_factory, model_family):
    """Train on one condition's custom data, test on another condition's."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _prep(data):
        if model_family in ("ssl", "resnet"):
            X_key = "X6" if args.channels == 6 else "X3"
            X = np.transpose(data[X_key].astype(np.float32), (0, 2, 1))
        elif model_family == "limu":
            X = _to_model_channels(data["X"].astype(np.float32), args.channels)
            X[:, :, :3] /= 9.8
            seq_len = args.limu_seq_len
            if seq_len < X.shape[1]:
                s = (X.shape[1] - seq_len) // 2
                X = X[:, s:s + seq_len, :]
        else:
            X = data["X"].astype(np.float32)
        return X, data["Y"]

    X_train_all, y_train_all = _prep(train_data)
    X_test, y_test = _prep(test_data)
    num_classes = len(CUSTOM_LABELS)
    epochs = args.epochs or 100
    batch_size = args.batch_size or 64
    lr = 1e-3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    idx = np.arange(len(X_train_all))
    np.random.shuffle(idx)
    vn = max(1, int(len(idx) * 0.1))
    X_val, y_val = X_train_all[idx[:vn]], y_train_all[idx[:vn]]
    X_train, y_train = X_train_all[idx[vn:]], y_train_all[idx[vn:]]

    model = model_factory(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
                            batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_f1, best_state, no_improve, patience = -1.0, None, 0, 10
    pbar = tqdm(range(epochs), desc="    Cross-cond train", unit="ep")
    for epoch in pbar:
        model.train()
        tr_loss, tr_steps = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx) if model_family != "limu" else model(bx, True), by)
            loss.backward(); optimizer.step()
            tr_loss += float(loss.item()); tr_steps += 1
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device)) if model_family != "limu" else model(bx.to(device), False)
                vp.append(logits.argmax(1).cpu().numpy()); vt.append(by.numpy())
        vf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average="macro", zero_division=0)
        pbar.set_postfix_str(f"loss={tr_loss/max(tr_steps,1):.4f} val_f1m={vf1*100:.2f}% best={best_val_f1*100:.2f}%")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            tqdm.write(f"    [OK] Epoch {epoch+1}: best val F1m={vf1*100:.2f}%")
        else:
            no_improve += 1
        if no_improve >= patience:
            tqdm.write(f"    Early stop at epoch {epoch+1} (no improve x{patience})"); break
    pbar.close()

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
                             batch_size=batch_size, shuffle=False, num_workers=0)
    all_pred = []
    with torch.no_grad():
        for bx, _ in test_loader:
            logits = model(bx.to(device)) if model_family != "limu" else model(bx.to(device), False)
            all_pred.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(all_pred)

    D_test = test_data.get("D")
    metrics = evaluate_predictions(y_test, y_pred, CUSTOM_LABELS, device_ids=D_test)
    print(f"\n  Cross-condition: train={len(X_train_all)} test={len(X_test)} "
          f"Acc={metrics['accuracy']*100:.2f}% F1m={metrics['f1_macro']*100:.2f}%")
    del model; torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Exp8: Evaluate on custom-collected data")
    parser.add_argument("--model", required=True, choices=["hart", "limu-bert", "ssl-wearables", "resnet-baseline"])
    parser.add_argument("--mode", required=True, choices=["cross", "loso", "reverse", "cross-condition"])
    parser.add_argument("--device-type", default="all", choices=["watch", "phone", "all"])
    parser.add_argument("--condition", default="all", choices=["control", "uncontrol", "uncontrolled", "all"])
    parser.add_argument("--train-condition", default=None, choices=["control", "uncontrol", "uncontrolled"],
                        help="For cross-condition mode: condition to train on.")
    parser.add_argument("--test-condition", default=None, choices=["control", "uncontrol", "uncontrolled"],
                        help="For cross-condition mode: condition to test on.")
    parser.add_argument("--channels", type=int, default=6, choices=[3, 6])
    parser.add_argument("--limu-seq-len", type=int, default=20, choices=[20, 120])
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Optional checkpoint to load (cross mode or LOSO fine-tuning init).")
    parser.add_argument("--experiment-tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cross-train-epochs", type=int, default=80,
                        help="When cross checkpoint missing, train full-HHAR model for this many epochs.")
    parser.add_argument("--cross-train-batch-size", type=int, default=128,
                        help="Batch size for full-HHAR training in cross mode (LIMU/ResNet).")
    parser.add_argument("--custom-test-repeats", type=int, default=1,
                        help="Deprecated: use --repeat-seeds instead. Retained for backward compat; ignored when --repeat-seeds > 1.")
    parser.add_argument("--repeat-seeds", type=int, default=1,
                        help="In cross mode, repeat the full HHAR train + custom test pipeline this many times "
                             "with seeds (base_seed, base_seed+1, ...). Aggregates mean/std across seeds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()
    args.output_dir = str(_resolve_project_path(args.output_dir))
    if args.pretrained:
        args.pretrained = str(_resolve_project_path(args.pretrained))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EXP8: Custom Data Evaluation")
    print(f"  Model     : {args.model}")
    print(f"  Mode      : {args.mode}")
    print(f"  Device    : {args.device_type}")
    print(f"  Condition : {args.condition}")
    print(f"  Channels  : {args.channels}")
    print(f"  Pretrained: {args.pretrained or 'auto/default'}")
    print(f"  Seed      : {args.seed}")
    print(f"{'='*60}")

    args.condition = CONDITION_ALIASES.get(args.condition, args.condition)
    model_family = {"hart": "hart", "limu-bert": "limu", "ssl-wearables": "ssl", "resnet-baseline": "resnet"}[args.model]

    if args.mode == "cross-condition":
        custom_data = None
    else:
        custom_data = load_custom_data(model_family if model_family != "resnet" else "ssl", args.condition, args.device_type)

    if args.mode == "cross":
        metrics_runs = []
        seed_list = []
        n_seeds = max(1, int(args.repeat_seeds))
        base_seed = int(args.seed)
        for rep_idx in range(n_seeds):
            cur_seed = base_seed + rep_idx
            args.seed = cur_seed
            seed_list.append(cur_seed)
            if n_seeds > 1:
                print(f"\n  Cross repeat {rep_idx+1}/{n_seeds} (seed={cur_seed}): train HHAR -> test custom")
            if args.model == "hart":
                m = cross_eval_hart(args, custom_data)
            elif args.model == "limu-bert":
                m = cross_eval_limu(args, custom_data, retrain_hhar=True)
            elif args.model == "ssl-wearables":
                m = cross_eval_ssl(args, custom_data, retrain_hhar=True)
            else:
                m = cross_eval_resnet(args, custom_data, retrain_hhar=True)
            if m is None:
                return
            m["seed"] = cur_seed
            metrics_runs.append(m)
        args.seed = base_seed
        metrics = _aggregate_cross_repeats(metrics_runs)
        print_summary(metrics, "Cross-dataset (HHAR -> Custom)")
        result = {
            "experiment": "EXP8_custom_data",
            "mode": "cross",
            "model": f"{args.model}-{args.channels}ch",
            "experiment_tag": args.experiment_tag,
            "pretrained": args.pretrained,
            "device_type": args.device_type,
            "condition": args.condition,
            "channels": args.channels,
            "seeds": seed_list,
            "test_metrics": metrics,
        }
        if n_seeds > 1 and "repeat_summary" in metrics:
            rs = metrics["repeat_summary"]
            # Build per-class recall mean/std from the per-seed runs.
            per_class_recall_ms: dict = {}
            all_cls = set()
            for run in metrics_runs:
                all_cls.update(run.get("per_class_f1", {}).keys())
            for cls in all_cls:
                rec_vals = [run.get("per_class_f1", {}).get(cls, {}).get("recall")
                            for run in metrics_runs]
                rec_vals = [v for v in rec_vals if v is not None]
                if rec_vals:
                    per_class_recall_ms[cls] = {
                        "mean": float(np.mean(rec_vals)),
                        "std":  float(np.std(rec_vals)),
                        "n":    len(rec_vals),
                    }
            result["repeated_seed_summary"] = {
                "runs": n_seeds,
                "seeds": seed_list,
                "metrics_mean_std": {
                    "accuracy": {"mean": rs.get("mean_accuracy"), "std": rs.get("std_accuracy")},
                    "f1_weighted": {"mean": rs.get("mean_f1_weighted"), "std": rs.get("std_f1_weighted")},
                    "f1_macro": {"mean": rs.get("mean_f1_macro"), "std": rs.get("std_f1_macro")},
                },
                "per_class_recall_mean_std": per_class_recall_ms,
            }
    elif args.mode == "loso":
        if args.model == "hart":
            metrics = loso_eval_hart(args, custom_data)
        elif args.model == "limu-bert":
            import torch
            sys.path.insert(0, str(LIMU_DIR))
            from models import BERTClassifier, ClassifierGRU
            from config import PretrainModelConfig, ClassifierModelConfig
            seq_len = args.limu_seq_len
            def limu_factory(nc):
                bert_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6, n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
                cls_cfg = ClassifierModelConfig(seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1], rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[], flat_num=0, num_attn=0, num_head=0, atten_hidden=0, num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)
                classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=nc)
                return BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False)
            metrics = loso_eval_pytorch(args, custom_data, limu_factory, "limu")
        elif args.model == "ssl-wearables":
            sys.path.insert(0, str(SSL_DIR))
            from sslearning.models.accNet import Resnet
            n_ch = args.channels
            def ssl_factory(nc): return Resnet(output_size=nc, n_channels=n_ch, is_eva=True, resnet_version=1, epoch_len=10)
            metrics = loso_eval_pytorch(args, custom_data, ssl_factory, "ssl")
        else:
            sys.path.insert(0, str(RESNET_BASE_DIR))
            from resnet1d_baseline import ResNet1DBaseline
            n_ch = args.channels
            def resnet_factory(nc): return ResNet1DBaseline(n_channels=n_ch, num_classes=nc, kernel_size=5)
            metrics = loso_eval_pytorch(args, custom_data, resnet_factory, "resnet")

        print(f"\n{'='*60}")
        print(f"  LOSO Results: {args.model}")
        print(f"  Mean Accuracy:    {fmt(metrics['mean_accuracy'])} +/- {fmt(metrics['std_accuracy'])}")
        print(f"  Mean F1 weighted: {fmt(metrics.get('mean_f1_weighted', 0.0))} +/- {fmt(metrics.get('std_f1_weighted', 0.0))}")
        print(f"  Mean F1 macro:    {fmt(metrics['mean_f1_macro'])} +/- {fmt(metrics['std_f1_macro'])}")
        print(f"{'='*60}")
        result = {
            "experiment": "EXP8_custom_data",
            "mode": "loso",
            "model": f"{args.model}-{args.channels}ch",
            "experiment_tag": args.experiment_tag,
            "pretrained": args.pretrained,
            "device_type": args.device_type,
            "condition": args.condition,
            "channels": args.channels,
            "seed": args.seed,
            **metrics,
        }

    elif args.mode == "reverse":
        n_seeds = max(1, int(args.repeat_seeds))
        base_seed = int(args.seed)
        seed_list = []
        metrics_runs = []
        for rep_idx in range(n_seeds):
            cur_seed = base_seed + rep_idx
            args.seed = cur_seed
            seed_list.append(cur_seed)
            if n_seeds > 1:
                print(f"\n  Reverse repeat {rep_idx+1}/{n_seeds} (seed={cur_seed}): train Custom -> test HHAR")
            if args.model == "hart":
                m = reverse_eval_hart(args, custom_data)
            elif args.model == "limu-bert":
                import torch
                sys.path.insert(0, str(LIMU_DIR))
                from models import BERTClassifier, ClassifierGRU
                from config import PretrainModelConfig, ClassifierModelConfig
                seq_len = args.limu_seq_len
                def limu_factory(nc):
                    bert_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6, n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
                    cls_cfg = ClassifierModelConfig(seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1], rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[], flat_num=0, num_attn=0, num_head=0, atten_hidden=0, num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)
                    classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=nc)
                    return BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False)
                m = reverse_eval_pytorch(args, custom_data, limu_factory, "limu")
            elif args.model == "ssl-wearables":
                sys.path.insert(0, str(SSL_DIR))
                from sslearning.models.accNet import Resnet
                n_ch = args.channels
                def ssl_factory(nc): return Resnet(output_size=nc, n_channels=n_ch, is_eva=True, resnet_version=1, epoch_len=10)
                m = reverse_eval_pytorch(args, custom_data, ssl_factory, "ssl")
            else:
                sys.path.insert(0, str(RESNET_BASE_DIR))
                from resnet1d_baseline import ResNet1DBaseline
                n_ch = args.channels
                def resnet_factory(nc): return ResNet1DBaseline(n_channels=n_ch, num_classes=nc, kernel_size=5)
                m = reverse_eval_pytorch(args, custom_data, resnet_factory, "resnet")
            if m is None:
                print("ERROR: Reverse evaluation failed.")
                return
            m["seed"] = cur_seed
            metrics_runs.append(m)
        args.seed = base_seed
        metrics = _aggregate_cross_repeats(metrics_runs)
        print_summary(metrics, "Reverse Cross-dataset (Custom -> HHAR)")
        result = {
            "experiment": "EXP86_reverse_custom_data",
            "mode": "reverse",
            "model": f"{args.model}-{args.channels}ch",
            "experiment_tag": args.experiment_tag,
            "pretrained": args.pretrained,
            "device_type": args.device_type,
            "condition": args.condition,
            "channels": args.channels,
            "seeds": seed_list,
            "test_metrics": metrics,
        }
        if n_seeds > 1 and "repeat_summary" in metrics:
            rs = metrics["repeat_summary"]
            per_class_recall_ms: dict = {}
            all_cls = set()
            for run in metrics_runs:
                all_cls.update(run.get("per_class_f1", {}).keys())
            for cls in all_cls:
                rec_vals = [run.get("per_class_f1", {}).get(cls, {}).get("recall")
                            for run in metrics_runs]
                rec_vals = [v for v in rec_vals if v is not None]
                if rec_vals:
                    per_class_recall_ms[cls] = {
                        "mean": float(np.mean(rec_vals)),
                        "std":  float(np.std(rec_vals)),
                        "n":    len(rec_vals),
                    }
            result["repeated_seed_summary"] = {
                "runs": n_seeds,
                "seeds": seed_list,
                "metrics_mean_std": {
                    "accuracy":    {"mean": rs.get("mean_accuracy"),    "std": rs.get("std_accuracy")},
                    "f1_weighted": {"mean": rs.get("mean_f1_weighted"), "std": rs.get("std_f1_weighted")},
                    "f1_macro":    {"mean": rs.get("mean_f1_macro"),    "std": rs.get("std_f1_macro")},
                },
                "per_class_recall_mean_std": per_class_recall_ms,
            }

    elif args.mode == "cross-condition":
        tc_train = CONDITION_ALIASES.get(args.train_condition or "", args.train_condition)
        tc_test  = CONDITION_ALIASES.get(args.test_condition or "", args.test_condition)
        if not tc_train or not tc_test:
            print("ERROR: --train-condition and --test-condition are required for cross-condition mode.")
            return
        if tc_train == tc_test:
            print(f"ERROR: train and test conditions must differ (both are '{tc_train}').")
            return

        model_family_key = model_family if model_family != "resnet" else "ssl"
        train_data = load_custom_data(model_family_key, tc_train, args.device_type)
        test_data  = load_custom_data(model_family_key, tc_test, args.device_type)

        cond_map = {"control": "Controlled", "uncontrolled": "Uncontrolled"}
        train_label = cond_map.get(tc_train, tc_train)
        test_label  = cond_map.get(tc_test, tc_test)
        print(f"\n  Cross-condition: {train_label} -> {test_label}  ({args.device_type})")

        n_seeds = max(1, int(args.repeat_seeds))
        base_seed = int(args.seed)
        seed_list = []
        metrics_runs = []
        for rep_idx in range(n_seeds):
            cur_seed = base_seed + rep_idx
            args.seed = cur_seed
            seed_list.append(cur_seed)
            if n_seeds > 1:
                print(f"\n  Cross-condition repeat {rep_idx+1}/{n_seeds} (seed={cur_seed}): train {train_label} -> test {test_label}")
            if args.model == "limu-bert":
                import torch
                sys.path.insert(0, str(LIMU_DIR))
                from models import BERTClassifier, ClassifierGRU
                from config import PretrainModelConfig, ClassifierModelConfig
                seq_len = args.limu_seq_len
                def limu_factory(nc):
                    bert_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6, n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
                    cls_cfg = ClassifierModelConfig(seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1], rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[], flat_num=0, num_attn=0, num_head=0, atten_hidden=0, num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)
                    classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=nc)
                    return BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False)
                m = cross_condition_eval_pytorch(args, train_data, test_data, limu_factory, "limu")
            elif args.model == "ssl-wearables":
                sys.path.insert(0, str(SSL_DIR))
                from sslearning.models.accNet import Resnet
                n_ch = args.channels
                def ssl_factory(nc): return Resnet(output_size=nc, n_channels=n_ch, is_eva=True, resnet_version=1, epoch_len=10)
                m = cross_condition_eval_pytorch(args, train_data, test_data, ssl_factory, "ssl")
            elif args.model == "resnet-baseline":
                sys.path.insert(0, str(RESNET_BASE_DIR))
                from resnet1d_baseline import ResNet1DBaseline
                n_ch = args.channels
                def resnet_factory(nc): return ResNet1DBaseline(n_channels=n_ch, num_classes=nc, kernel_size=5)
                m = cross_condition_eval_pytorch(args, train_data, test_data, resnet_factory, "resnet")
            else:
                print(f"ERROR: cross-condition not implemented for {args.model}")
                return
            if m is None:
                print("ERROR: Cross-condition evaluation failed.")
                return
            m["seed"] = cur_seed
            metrics_runs.append(m)
        args.seed = base_seed
        metrics = _aggregate_cross_repeats(metrics_runs)
        direction = f"Our {train_label} -> Our {test_label}"
        print_summary(metrics, f"Cross-condition ({direction})")
        result = {
            "experiment": "EXP87_cross_condition",
            "mode": "cross",
            "direction": direction,
            "model": f"{args.model}-{args.channels}ch",
            "experiment_tag": args.experiment_tag,
            "device_type": args.device_type,
            "train_condition": tc_train,
            "test_condition": tc_test,
            "condition": f"{tc_train}->{tc_test}",
            "channels": args.channels,
            "seeds": seed_list,
            "test_metrics": metrics,
        }
        if n_seeds > 1 and "repeat_summary" in metrics:
            rs = metrics["repeat_summary"]
            per_class_recall_ms: dict = {}
            all_cls = set()
            for run in metrics_runs:
                all_cls.update(run.get("per_class_f1", {}).keys())
            for cls in all_cls:
                rec_vals = [run.get("per_class_f1", {}).get(cls, {}).get("recall")
                            for run in metrics_runs]
                rec_vals = [v for v in rec_vals if v is not None]
                if rec_vals:
                    per_class_recall_ms[cls] = {
                        "mean": float(np.mean(rec_vals)),
                        "std":  float(np.std(rec_vals)),
                        "n":    len(rec_vals),
                    }
            result["repeated_seed_summary"] = {
                "runs": n_seeds,
                "seeds": seed_list,
                "metrics_mean_std": {
                    "accuracy":    {"mean": rs.get("mean_accuracy"),    "std": rs.get("std_accuracy")},
                    "f1_weighted": {"mean": rs.get("mean_f1_weighted"), "std": rs.get("std_f1_weighted")},
                    "f1_macro":    {"mean": rs.get("mean_f1_macro"),    "std": rs.get("std_f1_macro")},
                },
                "per_class_recall_mean_std": per_class_recall_ms,
            }

    model_tag = f"{args.model}-{args.channels}ch-{args.device_type}"
    save_exp8_results(result, args.output_dir, args.experiment_tag or "exp8", model_tag)


if __name__ == "__main__":
    main()
