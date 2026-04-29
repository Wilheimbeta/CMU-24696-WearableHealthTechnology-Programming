#!/usr/bin/env python
"""
=============================================================================
Unified LOSO Training Script for HART, LIMU-BERT, and ssl-wearables
=============================================================================

Supports benchmark comparison experiments with phone/watch separation,
partial data training, and channel selection.

Usage examples:
  # ---- Baselines (from scratch, full data) ----
  python train_loso.py --model hart --device-type phone
  python train_loso.py --model hart --device-type watch
  python train_loso.py --model limu-bert --device-type phone
  python train_loso.py --model ssl-wearables --device-type watch --channels 3

  # ---- LIMU-BERT-X (seq_len=20) with pretrained weights ----
  python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone

  # ---- Pretrained + half data ----
  python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.5

  # ---- Single subject fine-tuning ----
  python train_loso.py --model hart --device-type phone --train-subjects 1

  # ---- HART accel-only vs accel+gyro ----
  python train_loso.py --model hart --channels 3 --experiment-tag exp3_hart_accel
  python train_loso.py --model hart --channels 6 --experiment-tag exp3_hart_accel_gyro

  # ---- Experiment tags for organised results ----
  python train_loso.py --model hart --device-type phone --experiment-tag exp1b_hart_phone_baseline
"""

import os, sys, json, time, argparse, copy, re
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths =====================
BASE_DIR    = Path(__file__).resolve().parent
HART_DIR    = (BASE_DIR / "code"
               / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main"
               / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main")
LIMU_DIR    = (BASE_DIR / "code"
               / "LIMU-BERT_Experience-main" / "LIMU-BERT_Experience-main")
SSL_DIR     = (BASE_DIR / "code"
               / "ssl-wearables-main" / "ssl-wearables-main")
RESNET_BASE_DIR = BASE_DIR / "code" / "resnet-baseline"
RESULTS_DIR     = BASE_DIR / "loso_results"
BEST_MODELS_DIR = RESULTS_DIR / "best_models"

ACTIVITY_LABELS = ['Sitting', 'Standing', 'Walking',
                   'Upstairs', 'Downstairs', 'Biking']
ACTIVITY_LABELS_NOBIKE = ['Sitting', 'Standing', 'Walking',
                          'Upstairs', 'Downstairs']
USER_NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


def _safe_torch_save(obj, path, retries=3, delay=0.5):
    """torch.save with retry for Windows file-locking (error 32)."""
    import torch
    p = Path(path)
    for attempt in range(retries):
        try:
            tmp = p.with_suffix('.tmp')
            torch.save(obj, str(tmp))
            tmp.replace(p)
            return
        except (RuntimeError, OSError):
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


# ===================== Results =====================
def _incremental_results_path(output_dir, experiment_tag, model_name, pretrained_tag):
    """Deterministic path for incremental fold results (no timestamp)."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    tag = "pretrained" if pretrained_tag else "scratch"
    prefix = f"{experiment_tag}_" if experiment_tag else ""
    return output_dir / f"{prefix}{model_name}_{tag}_loso_progress.json"


def _extract_run_number(experiment_tag):
    if not experiment_tag:
        return None
    m = re.search(r"(?:^|_)run(\d+)(?:_|$)", str(experiment_tag), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _infer_additional_pretraining(pretrained):
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


def _build_experiment_table_metadata(model_name, experiment_tag, summary, extra_meta=None):
    extra_meta = extra_meta or {}
    protocol = str(extra_meta.get('protocol') or summary.get('protocol') or 'loso').lower()
    is_loso_in = protocol == "leave_one_in" or protocol.startswith("leave_") and protocol.endswith("_in")
    if is_loso_in:
        K = extra_meta.get('train_subjects') or summary.get('train_subjects') or 1
        testing_method = f"Leave-{K}-In"
    else:
        testing_method = "LOSO"
    pretrained = summary.get('pretrained')
    channels = summary.get('channels')
    data_fraction = summary.get('data_fraction', 1.0)
    try:
        data_fraction_txt = f"{float(data_fraction) * 100:.1f}%".replace(".0%", "%")
    except Exception:
        data_fraction_txt = ""
    n_tests = summary.get('repeated_seed_summary', {}).get('runs')
    if n_tests is None:
        n_tests = summary.get('num_folds') or (len(summary.get('per_fold', [])) if isinstance(summary.get('per_fold'), list) else 1)
    return {
        "Architecture Type": model_name,
        "Device Type": str(summary.get('device_type', '')).capitalize(),
        "Training Dataset": "HHAR",
        "Testing Dataset": "HHAR",
        "Imported Pretrained Weights": "Yes" if pretrained else "No",
        "3ch vs 6ch data": f"{channels}ch" if channels else "",
        "Data Fraction": data_fraction_txt,
        "Additional Pretraining dataset": _infer_additional_pretraining(pretrained),
        "Testing Method": testing_method,
        "# of Tests": int(n_tests) if n_tests else 1,
        "# Run": _extract_run_number(experiment_tag) or "",
    }


def is_fold_completed(output_dir, experiment_tag, model_name, pretrained_tag, subject_id):
    """Check if a fold already has a test result in the progress JSON."""
    path = _incremental_results_path(output_dir, experiment_tag, model_name, pretrained_tag)
    if not path.exists():
        return False
    with open(path) as f:
        data = json.load(f)
    for fold in data.get('per_fold', []):
        if fold.get('subject') == subject_id and fold.get('accuracy') is not None:
            return True
    return False


def print_resume_status(subjects, output_dir, experiment_tag, model_name,
                        pretrained_tag, ckpt_pattern):
    """Print per-fold resume status: completed / checkpoint / new."""
    prog_path = _incremental_results_path(output_dir, experiment_tag, model_name, pretrained_tag)
    completed_subjects = set()
    if prog_path.exists():
        with open(prog_path) as f:
            data = json.load(f)
        for fold in data.get('per_fold', []):
            s = fold.get('subject')
            if fold.get('accuracy') is not None:
                completed_subjects.add(s)

    print(f"\n  Resume status for '{experiment_tag or 'run'}':")
    print(f"  {'Fold':<8} {'Subject':<10} {'Status':<14} {'Checkpoint'}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*40}")
    for i, subj in enumerate(subjects):
        ckpt = Path(str(ckpt_pattern).format(fold_idx=i))
        if subj in completed_subjects:
            status = "COMPLETED"
            ckpt_str = "(result in progress JSON)"
        elif ckpt.exists():
            import os as _os
            size_kb = _os.path.getsize(ckpt) / 1024
            status = "HAS CKPT"
            ckpt_str = f"{ckpt.name} ({size_kb:.0f} KB)"
        else:
            status = "NEW"
            ckpt_str = "(will train from scratch)"
        print(f"  {i+1:<8} {str(subj):<10} {status:<14} {ckpt_str}")
    n_done = len(completed_subjects)
    print(f"\n  Summary: {n_done}/{len(subjects)} folds completed, "
          f"{sum(1 for i,s in enumerate(subjects) if s not in completed_subjects and Path(str(ckpt_pattern).format(fold_idx=i)).exists())} with checkpoint, "
          f"{sum(1 for i,s in enumerate(subjects) if s not in completed_subjects and not Path(str(ckpt_pattern).format(fold_idx=i)).exists())} new")
    print()


def save_fold_result(fold_result, output_dir, experiment_tag, model_name,
                     pretrained_tag, extra_meta=None):
    """Save/append a single fold result to the incremental JSON file."""
    path = _incremental_results_path(output_dir, experiment_tag, model_name, pretrained_tag)
    # Load existing
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        folds = data.get('per_fold', [])
    else:
        folds = []
    # Replace if same subject already exists, otherwise append
    subj = fold_result['subject']
    folds = [f for f in folds if f.get('subject') != subj] + [fold_result]
    # Compute running stats
    accs  = [r['accuracy'] for r in folds if r.get('accuracy') is not None]
    f1s_m = [r['f1_macro'] for r in folds if r.get('f1_macro') is not None]
    summary = {
        'model': model_name,
        'experiment_tag': experiment_tag,
        'pretrained': pretrained_tag,
        'folds_completed': len(accs),
        'mean_accuracy': float(np.mean(accs)) if accs else 0,
        'mean_f1_macro': float(np.mean(f1s_m)) if f1s_m else 0,
        'per_fold': folds,
    }
    if extra_meta:
        summary.update(extra_meta)
    summary['experiment_table_metadata'] = _build_experiment_table_metadata(
        model_name, experiment_tag, summary, extra_meta=extra_meta
    )
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    >> Fold result saved ({len(accs)} folds so far) -> {path}")


def save_results(results, model_name, output_dir, pretrained_tag,
                 experiment_tag=None, extra_meta=None):
    """Save final LOSO results to JSON (timestamped) and print summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "pretrained" if pretrained_tag else "scratch"
    prefix = f"{experiment_tag}_" if experiment_tag else ""
    fname = f"{prefix}{model_name}_{tag}_loso_{ts}.json"

    accs  = [r['accuracy'] for r in results if r['accuracy'] is not None]
    f1s_w = [r['f1_weighted'] for r in results if r['f1_weighted'] is not None]
    f1s_m = [r['f1_macro'] for r in results if r['f1_macro'] is not None]

    summary = {
        'model': model_name,
        'pretrained': pretrained_tag,
        'experiment_tag': experiment_tag,
        'num_folds': len(results),
        'mean_accuracy': float(np.mean(accs)) if accs else 0,
        'std_accuracy': float(np.std(accs)) if accs else 0,
        'mean_f1_weighted': float(np.mean(f1s_w)) if f1s_w else 0,
        'std_f1_weighted': float(np.std(f1s_w)) if f1s_w else 0,
        'mean_f1_macro': float(np.mean(f1s_m)) if f1s_m else 0,
        'std_f1_macro': float(np.std(f1s_m)) if f1s_m else 0,
        'per_fold': results,
    }
    if extra_meta:
        summary.update(extra_meta)
    summary['experiment_table_metadata'] = _build_experiment_table_metadata(
        model_name, experiment_tag, summary, extra_meta=extra_meta
    )

    path = output_dir / fname
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Clean up incremental file
    progress_path = _incremental_results_path(
        output_dir, experiment_tag, model_name, pretrained_tag)
    if progress_path.exists():
        progress_path.unlink()

    protocol = (extra_meta or {}).get('protocol')
    print("\n" + "=" * 60)
    is_leave_in = protocol == 'leave_one_in' or (protocol.startswith('leave_') and protocol.endswith('_in'))
    title = "Leave-K-In Results" if is_leave_in else "LOSO Results"
    print(f"  {title}: {model_name} ({tag})")
    if experiment_tag:
        print(f"  Experiment   : {experiment_tag}")
    print("=" * 60)
    print(f"  Folds completed : {len(accs)}/{len(results)}")
    if is_leave_in:
        print("  Per test-subject accuracy:")
        for row in results:
            tr = row.get('train_subjects') or row.get('train_subject')
            te = row.get('subject')
            acc = row.get('accuracy')
            f1m = row.get('f1_macro')
            if acc is None:
                continue
            f1m_str = f"{f1m*100:.2f}%" if f1m is not None else "NA"
            print(f"    train={tr} -> test={te}: Acc={acc*100:.2f}%  F1m={f1m_str}")
    else:
        print(f"  Mean Accuracy   : {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
        print(f"  Mean F1 weighted: {np.mean(f1s_w)*100:.2f}% +/- {np.std(f1s_w)*100:.2f}%")
        print(f"  Mean F1 macro   : {np.mean(f1s_m)*100:.2f}% +/- {np.std(f1s_m)*100:.2f}%")
    print(f"  Results saved to: {path}")
    print("=" * 60)

    try:
        from summarize_experiment_results import update_for_experiment_tag
        out_csv = update_for_experiment_tag(experiment_tag, results_dir=output_dir)
        if out_csv:
            print(f"  Experiment summary CSV -> {out_csv}")
    except Exception as e:
        print(f"  (Experiment summary CSV skipped: {e})")

    return summary


def save_repeated_results(run_summaries, output_dir, experiment_tag, model_name, extra_meta=None):
    """Save repeated-seed summary JSON for LOSO runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_tag}_" if experiment_tag else ""
    fname = f"{prefix}{model_name}_repeated_loso_{ts}.json"

    mean_accs = [r.get('mean_accuracy', 0.0) for r in run_summaries]
    mean_f1w = [r.get('mean_f1_weighted', 0.0) for r in run_summaries]
    mean_f1m = [r.get('mean_f1_macro', 0.0) for r in run_summaries]

    per_fold_agg = []
    subject_ids = []
    for run in run_summaries:
        for fold in run.get('per_fold', []):
            subject_ids.append(fold.get('subject'))
    for subject in sorted(set(subject_ids), key=str):
        folds = [f for run in run_summaries for f in run.get('per_fold', []) if f.get('subject') == subject]
        if not folds:
            continue
        fold_row = {
            'subject': subject,
            'accuracy': float(np.mean([f.get('accuracy', 0.0) for f in folds])),
            'f1_weighted': float(np.mean([f.get('f1_weighted', 0.0) for f in folds])),
            'f1_macro': float(np.mean([f.get('f1_macro', 0.0) for f in folds])),
        }
        all_cls = set()
        for f in folds:
            all_cls.update(f.get('per_class_f1', {}).keys())
        if all_cls:
            pc = {}
            for cls in all_cls:
                rec = [f.get('per_class_f1', {}).get(cls, {}).get('recall')
                       for f in folds]
                rec = [v for v in rec if v is not None]
                f1v = [f.get('per_class_f1', {}).get(cls, {}).get('f1')
                       for f in folds]
                f1v = [v for v in f1v if v is not None]
                entry = {}
                if rec:
                    entry['recall'] = float(np.mean(rec))
                if f1v:
                    entry['f1'] = float(np.mean(f1v))
                if entry:
                    pc[cls] = entry
            if pc:
                fold_row['per_class_f1'] = pc
        per_fold_agg.append(fold_row)

    summary = {
        'model': model_name,
        'experiment_tag': experiment_tag,
        'repeated_seed_summary': {
            'runs': len(run_summaries),
            'seeds': [r.get('seed') for r in run_summaries],
            'metrics_mean_std': {
                'mean_accuracy': {'mean': float(np.mean(mean_accs)), 'std': float(np.std(mean_accs)), 'n': len(mean_accs)},
                'mean_f1_weighted': {'mean': float(np.mean(mean_f1w)), 'std': float(np.std(mean_f1w)), 'n': len(mean_f1w)},
                'mean_f1_macro': {'mean': float(np.mean(mean_f1m)), 'std': float(np.std(mean_f1m)), 'n': len(mean_f1m)},
            },
        },
        'num_folds': run_summaries[-1].get('num_folds', len(per_fold_agg)),
        'mean_accuracy': float(np.mean(mean_accs)),
        'std_accuracy': float(np.std(mean_accs)),
        'mean_f1_weighted': float(np.mean(mean_f1w)),
        'std_f1_weighted': float(np.std(mean_f1w)),
        'mean_f1_macro': float(np.mean(mean_f1m)),
        'std_f1_macro': float(np.std(mean_f1m)),
        'per_fold': per_fold_agg,
    }
    if extra_meta:
        summary.update(extra_meta)
    summary['experiment_table_metadata'] = _build_experiment_table_metadata(
        model_name, experiment_tag, summary, extra_meta=extra_meta
    )

    path = output_dir / fname
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Repeated-run summary saved to: {path}")
    try:
        from summarize_experiment_results import update_for_experiment_tag
        out_csv = update_for_experiment_tag(experiment_tag, results_dir=output_dir)
        if out_csv:
            print(f"  Experiment summary CSV -> {out_csv}")
    except Exception as e:
        print(f"  (Experiment summary CSV skipped: {e})")
    return summary


def subsample_training_data(X, y, fraction, seed=42):
    """Randomly subsample training data to simulate limited labels."""
    if fraction >= 1.0:
        return X, y
    rng = np.random.RandomState(seed)
    n = len(X)
    n_keep = max(1, int(n * fraction))
    idx = rng.choice(n, n_keep, replace=False)
    idx.sort()
    print(f"    Data fraction {fraction}: using {n_keep}/{n} training samples")
    return X[idx], y[idx]


DEVICE_IDX_TO_NAME = {
    0: 'nexus4', 1: 'lgwatch', 2: 's3', 3: 's3mini', 4: 'gear', 5: 'samsungold'
}


def _resolve_project_path(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)

def evaluate_fold(y_true, y_pred, subject_id, n_train, n_test, no_bike=False,
                  device_ids=None):
    """Compute all metrics for a single LOSO fold (shared by all models).

    If device_ids is provided (array same length as y_true), also computes
    per-device accuracy and F1 macro.
    """
    labels = ACTIVITY_LABELS_NOBIKE if no_bike else ACTIVITY_LABELS
    num_cls = len(labels)
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(num_cls))).tolist()
    report = classification_report(
        y_true, y_pred, labels=list(range(num_cls)),
        target_names=labels, output_dict=True, zero_division=0)
    per_class = {}
    for i in range(num_cls):
        name = labels[i]
        if name in report:
            per_class[name] = {
                'precision': round(report[name]['precision'], 4),
                'recall':    round(report[name]['recall'], 4),
                'f1':        round(report[name]['f1-score'], 4),
                'support':   int(report[name]['support']),
            }

    print(f"  Subject {subject_id}: Acc={acc*100:.2f}%, "
          f"F1w={f1_w*100:.2f}%, F1m={f1_m*100:.2f}%")

    per_device = {}
    if device_ids is not None and len(device_ids) == len(y_true):
        for dev_id in sorted(np.unique(device_ids)):
            mask = device_ids == dev_id
            if mask.sum() == 0:
                continue
            dev_name = DEVICE_IDX_TO_NAME.get(int(dev_id), f"device_{dev_id}")
            d_acc = accuracy_score(y_true[mask], y_pred[mask])
            d_f1m = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
            per_device[dev_name] = {
                'accuracy': round(float(d_acc), 4),
                'f1_macro': round(float(d_f1m), 4),
                'n_samples': int(mask.sum()),
            }
            print(f"    {dev_name}: Acc={d_acc*100:.1f}%, F1m={d_f1m*100:.1f}% ({mask.sum()} samples)")

    result = {
        'subject': subject_id,
        'accuracy': float(acc),
        'f1_weighted': float(f1_w),
        'f1_macro': float(f1_m),
        'per_class_f1': per_class,
        'confusion_matrix': cm,
        'n_train': n_train,
        'n_test': n_test,
    }
    if per_device:
        result['per_device'] = per_device
    return result


# =====================================================================
#  HART  (TensorFlow / Keras)
# =====================================================================
def train_hart_loso(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress ptxas/XLA warnings
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sys.path.insert(0, str(HART_DIR))
    import tensorflow as tf
    import model as hart_model
    import hickle as hkl

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  [HART] TF {tf.__version__}, GPUs detected: {len(gpus)}"
          f"{'  >>>  WARNING: training on CPU! <<<' if len(gpus) == 0 else ''}")

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 256
    grad_accum = getattr(args, 'grad_accum', 1)
    lr         = 5e-3
    no_bike    = getattr(args, 'no_bike', False)
    num_classes = 5 if no_bike else 6
    n_channels = args.channels  # 3 or 6
    input_shape = (128, n_channels)
    use_mobilehart = (args.model == 'mobilehart')
    model_label = 'MobileHART' if use_mobilehart else 'HART'
    ckpt_prefix = 'mobilehart' if use_mobilehart else 'hart'

    # --- Resolve data directory ---
    dt = args.device_type
    nobike_tag = '_nobike' if no_bike else ''
    suffix_map = {'all': f'HHAR{nobike_tag}', 'phone': f'HHAR_phone{nobike_tag}',
                  'watch': f'HHAR_watch{nobike_tag}'}
    data_dir = HART_DIR / "datasets" / "datasetStandardized" / suffix_map[dt]
    map_name = f"hart_subject_map{'_' + dt if dt != 'all' else ''}{nobike_tag}.json"
    map_path = BASE_DIR / map_name

    if n_channels == 3:
        suffix_3ch_map = {'all': f'HHAR_3ch{nobike_tag}',
                          'phone': f'HHAR_phone_3ch{nobike_tag}',
                          'watch': f'HHAR_watch_3ch{nobike_tag}'}
        data_dir_3ch = HART_DIR / "datasets" / "datasetStandardized" / suffix_3ch_map[dt]
        map_name_3ch = f"hart_subject_map{'_' + dt if dt != 'all' else ''}_3ch{nobike_tag}.json"
        map_path_3ch = BASE_DIR / map_name_3ch
        if data_dir_3ch.exists() and map_path_3ch.exists():
            data_dir = data_dir_3ch
            map_path = map_path_3ch
            print(f"  Using dedicated HART 3ch dataset: {data_dir.name}")
        else:
            print("  NOTE: Dedicated HART 3ch dataset not found; falling back to 6ch-prepared data with gyro channels zeroed.")

    if not map_path.exists():
        print(f"ERROR: {map_path} not found. Run prepare_hhar_data.py first.")
        sys.exit(1)
    with open(map_path) as f:
        subject_map = json.load(f)

    n_clients = len(subject_map)
    client_data, client_label, client_user, client_devid = [], [], [], []
    dev_index_path = data_dir / "deviceIndex.hkl"
    has_dev_index = dev_index_path.exists()
    if has_dev_index:
        dev_index_all = hkl.load(str(dev_index_path))
    for i in range(n_clients):
        d = hkl.load(str(data_dir / f"UserData{i}.hkl"))
        l = hkl.load(str(data_dir / f"UserLabel{i}.hkl"))
        if n_channels == 3:
            d_new = np.zeros_like(d)
            d_new[:, :, :3] = d[:, :, :3]
            d = d_new
        client_data.append(d)
        client_label.append(l)
        client_user.append(subject_map[str(i)])
        if has_dev_index:
            client_devid.append(dev_index_all[i])

    subjects = sorted(set(client_user))
    print(f"\n{model_label} [{dt}]: {n_clients} clients, {len(subjects)} subjects, "
          f"{n_channels}ch, frac={args.data_fraction}")

    exp_tag_g = args.experiment_tag or 'run'
    if getattr(args, 'resume', False):
        ckpt_pat = str(BEST_MODELS_DIR / f"{ckpt_prefix}_{exp_tag_g}_fold{{fold_idx}}.weights.h5")
        print_resume_status(subjects, args.output_dir, args.experiment_tag,
                            f'{ckpt_prefix}-{n_channels}ch-{dt}', None, ckpt_pat)

    results = []
    for fold_idx, test_subject in enumerate(subjects):
        if fold_idx < getattr(args, 'start_fold', 0):
            print(f"\n  Fold {fold_idx+1}/{len(subjects)}: test='{test_subject}' -- SKIPPED (--start-fold {args.start_fold})")
            results.append({'subject': test_subject, 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        print(f"\n{'='*50}")
        print(f"  LOSO Fold {fold_idx+1}/{len(subjects)}: test='{test_subject}'")
        print(f"{'='*50}")

        tf.keras.backend.clear_session()
        tf.random.set_seed(args.seed); np.random.seed(args.seed)

        # Split data
        train_d, train_l, test_d, test_l, test_dev = [], [], [], [], []
        train_subjects_added = set()
        for i in range(n_clients):
            if client_user[i] == test_subject:
                test_d.append(client_data[i])
                test_l.append(client_label[i])
                if has_dev_index:
                    test_dev.append(client_devid[i])
            else:
                if getattr(args, 'train_subjects', None) is not None:
                    if len(train_subjects_added) < args.train_subjects or client_user[i] in train_subjects_added:
                        train_subjects_added.add(client_user[i])
                        train_d.append(client_data[i])
                        train_l.append(client_label[i])
                else:
                    train_d.append(client_data[i])
                    train_l.append(client_label[i])

        X_train_raw = np.vstack(train_d); y_train_raw = np.hstack(train_l)
        X_test  = np.vstack(test_d);  y_test  = np.hstack(test_l)
        D_test  = np.hstack(test_dev) if test_dev else None

        if len(X_test) == 0:
            results.append({'subject': test_subject, 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        X_train_sub, y_train_sub = subsample_training_data(
            X_train_raw, y_train_raw, args.data_fraction, seed=args.seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_sub, y_train_sub, test_size=0.1, random_state=args.seed,
            stratify=y_train_sub)

        # Checkpoint path
        exp_tag = args.experiment_tag or 'run'
        ckpt_path = str(BEST_MODELS_DIR / f"{ckpt_prefix}_{exp_tag}_fold{fold_idx}.weights.h5")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # --- Resume: skip fold if already has test result in progress JSON ---
        _tag_name = f'{ckpt_prefix}-{n_channels}ch-{dt}'
        if getattr(args, 'resume', False) and is_fold_completed(
                args.output_dir, args.experiment_tag, _tag_name, None, test_subject):
            print(f"    [OK] Fold already completed, skipping (use progress JSON)")
            # Load result from progress JSON
            _prog = _incremental_results_path(args.output_dir, args.experiment_tag, _tag_name, None)
            with open(_prog) as _f:
                _prev = json.load(_f)
            for _r in _prev['per_fold']:
                if _r.get('subject') == test_subject:
                    results.append(_r); break
            continue
        else:
            # --- Training (from scratch or resume from checkpoint) ---
            if use_mobilehart:
                mdl = hart_model.mobileHART_XS(input_shape, num_classes)
            else:
                mdl = hart_model.HART(input_shape, num_classes)
            if getattr(args, 'resume', False) and os.path.exists(ckpt_path):
                # Load partial checkpoint and continue training
                mdl.build(input_shape=(None,) + tuple(input_shape))
                mdl.load_weights(ckpt_path)
                print(f"    [OK] Resuming training from checkpoint -> {ckpt_path}")
            optimizer = tf.keras.optimizers.Adam(lr)
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

            from sklearn.utils import class_weight
            cw = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {j: cw[j] for j in range(len(cw))}
            sample_weights = np.array([class_weights[int(y)] for y in y_train],
                                      dtype=np.float32)

            y_train_oh = tf.one_hot(y_train, num_classes)
            train_ds = (tf.data.Dataset.from_tensor_slices(
                            (X_train, y_train_oh, sample_weights))
                        .shuffle(len(X_train), seed=42)
                        .batch(batch_size).prefetch(tf.data.AUTOTUNE))

            best_val_f1, best_val_loss, no_improve, patience = -1.0, float('inf'), 0, 15
            num_steps = int(np.ceil(len(X_train) / batch_size))
            pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                        bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                total_loss = 0.0
                accum_grads = [tf.zeros_like(v) for v in mdl.trainable_variables]
                for step, (bx, by, bw) in enumerate(train_ds):
                    with tf.GradientTape() as tape:
                        pred = mdl(bx, training=True)
                        loss = loss_fn(by, pred, sample_weight=bw) / grad_accum
                    grads = tape.gradient(loss, mdl.trainable_variables)
                    accum_grads = [a + g for a, g in zip(accum_grads, grads)]
                    if (step + 1) % grad_accum == 0 or (step + 1) == num_steps:
                        optimizer.apply_gradients(
                            zip(accum_grads, mdl.trainable_variables))
                        accum_grads = [tf.zeros_like(v) for v in mdl.trainable_variables]
                    total_loss += float(loss) * grad_accum

                pred_probs = mdl.predict(X_val, verbose=0)
                yp_val = np.argmax(pred_probs, axis=-1)
                v_f1  = f1_score(y_val, yp_val, average='macro', zero_division=0)
                v_acc = accuracy_score(y_val, yp_val)
                v_loss = float(loss_fn(
                    tf.keras.utils.to_categorical(y_val, num_classes),
                    pred_probs).numpy())
                f1_up = v_f1 > best_val_f1
                loss_up = v_loss < best_val_loss
                if f1_up:
                    best_val_f1 = v_f1
                    mdl.save_weights(ckpt_path)
                    tqdm.write(f"    [OK] Saved best model (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
                if loss_up:
                    best_val_loss = v_loss
                if f1_up or loss_up:
                    no_improve = 0
                else:
                    no_improve += 1
                avg_loss = total_loss / num_steps
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} Acc={v_acc*100:.1f}% F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
                if no_improve >= patience:
                    pbar.close()
                    print(f"    Early stop at epoch {epoch+1}"); break
            else:
                pbar.close()

            mdl.load_weights(ckpt_path)

        # --- Test evaluation (always runs) ---
        y_pred = np.argmax(mdl.predict(X_test, verbose=0), axis=-1)
        fold_res = evaluate_fold(
            y_test, y_pred, test_subject,
            len(X_train), len(X_test), no_bike,
            device_ids=D_test)
        results.append(fold_res)
        save_fold_result(fold_res, args.output_dir, args.experiment_tag,
                         f'{ckpt_prefix}-{n_channels}ch-{dt}', None,
                         extra_meta={'device_type': dt, 'channels': n_channels})

        del mdl; import gc; gc.collect()

    tag_name = f'{ckpt_prefix}-{n_channels}ch-{dt}'
    return save_results(results, tag_name, args.output_dir, None,
                        experiment_tag=args.experiment_tag,
                        extra_meta={'device_type': dt, 'channels': n_channels,
                                    'data_fraction': args.data_fraction,
                                    'train_subjects': getattr(args, 'train_subjects', None),
                                    'no_bike': no_bike, 'num_classes': num_classes,
                                    'seed': args.seed})


# =====================================================================
#  LIMU-BERT  (PyTorch)
# =====================================================================
def train_limu_bert_loso(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [LIMU-BERT] PyTorch {torch.__version__}, device={device}"
          f"{'  >>>  WARNING: training on CPU! <<<' if device.type == 'cpu' else ''}")

    sys.path.insert(0, str(LIMU_DIR))
    from models import BERTClassifier, ClassifierGRU
    from config import PretrainModelConfig, ClassifierModelConfig

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 128
    grad_accum = getattr(args, 'grad_accum', 1)
    lr         = 1e-3
    no_bike    = getattr(args, 'no_bike', False)
    num_classes = 5 if no_bike else 6
    label_index = 2

    # --- Resolve data directory ---
    dt = args.device_type
    nobike_tag = '_nobike' if no_bike else ''
    dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
               'all': f'hhar_all{nobike_tag}'}
    data_dir = LIMU_DIR / "dataset" / dir_map[dt]
    data_path  = data_dir / "data_20_120.npy"
    label_path = data_dir / "label_20_120.npy"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run prepare_hhar_data.py first.")
        sys.exit(1)

    raw_data   = np.load(str(data_path)).astype(np.float32)
    raw_labels = np.load(str(label_path)).astype(np.float32)
    user_ids   = raw_labels[:, 0, 0].astype(int)
    act_labels = raw_labels[:, 0, label_index].astype(int)
    subjects   = sorted(np.unique(user_ids).tolist())
    dev_file   = data_dir / "D.npy"
    D_all_limu = np.load(str(dev_file)).astype(np.int64) if dev_file.exists() else None

    data_norm = raw_data.copy()
    data_norm[:, :, :3] /= 9.8

    seq_len = getattr(args, 'limu_seq_len', 120)
    variant = "LIMU-BERT-X" if seq_len == 20 else "LIMU-BERT"
    print(f"\n{variant} [{dt}] (seq_len={seq_len}): {len(raw_data)} samples, "
          f"{len(subjects)} subjects, frac={args.data_fraction}")

    bert_cfg = PretrainModelConfig(
        hidden=72, hidden_ff=144, feature_num=6,
        n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
    cls_cfg = ClassifierModelConfig(
        seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1],
        rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[],
        flat_num=0, num_attn=0, num_head=0, atten_hidden=0,
        num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)

    pretrained_path = None
    if args.pretrained:
        p = _resolve_project_path(args.pretrained)
        if not p.is_absolute() and not p.exists():
            p = LIMU_DIR / p
        if not str(p).endswith('.pt'):
            p_pt = Path(str(p) + '.pt')
            if p_pt.exists(): p = p_pt
        pretrained_path = p
        if not pretrained_path.exists():
            pretrained_path = LIMU_DIR / "weights" / p.name
            if not pretrained_path.exists():
                pretrained_path = LIMU_DIR / "weights" / (p.stem + '.pt')
        print(f"  Pretrained weights: {pretrained_path}")

    model_tag = f'limu-bert{"x" if seq_len == 20 else ""}-{dt}'
    exp_tag_g = args.experiment_tag or 'run'
    _ptag_g = str(pretrained_path) if pretrained_path else None
    if getattr(args, 'resume', False):
        ckpt_pat = str(BEST_MODELS_DIR / f"limubert_{exp_tag_g}_fold{{fold_idx}}.pt")
        print_resume_status(subjects, args.output_dir, args.experiment_tag,
                            model_tag, _ptag_g, ckpt_pat)

    results = []
    if getattr(args, 'leave_one_in', False):
        from itertools import combinations as _combinations
        K = int(getattr(args, 'train_subjects', None) or 1)
        if K < 1 or K >= len(subjects):
            raise ValueError(f"--train-subjects must be in [1, {len(subjects)-1}] for leave-one-in with {len(subjects)} subjects (got {K})")
        train_combos = [tuple(int(s) for s in c) for c in _combinations(sorted(subjects), K)]
        print(f"\n  Using leave-{K}-in protocol ({len(train_combos)} train-subject combos of {K} subjects)")
        for fold_idx, train_combo in enumerate(train_combos):
            combo_str = "_".join(str(s) for s in train_combo)
            if fold_idx < getattr(args, 'start_fold', 0):
                print(f"\n  Fold {fold_idx+1}/{len(train_combos)}: train=({combo_str}) -- SKIPPED (--start-fold {args.start_fold})")
                continue

            print(f"\n{'='*50}")
            print(f"  Leave-{K}-In Fold {fold_idx+1}/{len(train_combos)}: train=({combo_str})")
            print(f"{'='*50}")

            torch.manual_seed(args.seed); np.random.seed(args.seed)
            train_mask = np.isin(user_ids, np.array(train_combo))
            X_train_all = data_norm[train_mask]
            y_train_all = act_labels[train_mask]
            if len(X_train_all) < 2:
                print("  WARNING: not enough train samples for this combo, skipping fold")
                continue

            X_train_all, y_train_all = subsample_training_data(
                X_train_all, y_train_all, args.data_fraction, seed=args.seed)

            idx = np.arange(len(X_train_all)); np.random.shuffle(idx)
            val_n = max(1, int(len(idx) * 0.1))
            if val_n >= len(idx):
                val_n = max(1, len(idx) - 1)
            X_val, y_val     = X_train_all[idx[:val_n]], y_train_all[idx[:val_n]]
            X_train, y_train = X_train_all[idx[val_n:]], y_train_all[idx[val_n:]]
            if len(X_train) == 0:
                print("  WARNING: empty train split after val split, skipping fold")
                continue

            class _CropDataset(torch.utils.data.Dataset):
                def __init__(self, X, y, target_len, random_crop=True):
                    self.X = torch.from_numpy(X).float()
                    self.y = torch.from_numpy(y).long()
                    self.target_len = target_len
                    self.random_crop = random_crop
                def __len__(self): return len(self.X)
                def __getitem__(self, idx):
                    x = self.X[idx]
                    if self.target_len < x.size(0):
                        if self.random_crop:
                            s = torch.randint(0, x.size(0) - self.target_len, (1,)).item()
                        else:
                            s = (x.size(0) - self.target_len) // 2
                        x = x[s:s + self.target_len]
                    return x, self.y[idx]

            data_seq_len = X_train.shape[1]
            def make_loader(X, y, shuffle=False, random_crop=False):
                if seq_len < data_seq_len:
                    ds = _CropDataset(X, y, seq_len, random_crop=random_crop)
                else:
                    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

            train_loader = make_loader(X_train, y_train, shuffle=True, random_crop=True)
            val_loader   = make_loader(X_val, y_val, random_crop=False)

            exp_tag = args.experiment_tag or 'run'
            ckpt_path = BEST_MODELS_DIR / f"limubert_{exp_tag}_K{K}_trainfold{fold_idx}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=num_classes)
            model = BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False)
            if pretrained_path and pretrained_path.exists():
                print(f"  Loading pretrained: {pretrained_path}")
                state_dict = model.state_dict()
                pretrained = torch.load(str(pretrained_path), map_location=device)
                loaded = 0
                for k, v in pretrained.items():
                    if k in state_dict and state_dict[k].shape == v.shape:
                        state_dict[k] = v; loaded += 1
                model.load_state_dict(state_dict)
                print(f"    Loaded {loaded}/{len(pretrained)} weight tensors")

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            best_val_f1, best_val_loss, best_state, no_improve, patience = -1.0, float('inf'), None, 0, 15
            pbar = tqdm(range(epochs), desc="    Training", unit="ep", bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                model.train(); total_loss = 0
                optimizer.zero_grad()
                for step, (bx, by) in enumerate(train_loader):
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(model(bx, True), by) / grad_accum
                    loss.backward()
                    if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                        optimizer.step(); optimizer.zero_grad()
                    total_loss += loss.item() * grad_accum

                model.eval(); vp, vt, vl = [], [], 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx_d, by_d = bx.to(device), by.to(device)
                        logits = model(bx_d, False)
                        vp.append(logits.argmax(1).cpu().numpy())
                        vt.append(by.numpy())
                        vl += criterion(logits, by_d).item()
                vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
                v_f1 = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
                v_acc = accuracy_score(vt_cat, vp_cat)
                v_loss = vl / max(len(val_loader), 1)
                f1_up = v_f1 > best_val_f1
                loss_up = v_loss < best_val_loss
                if f1_up:
                    best_val_f1 = v_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    _safe_torch_save(best_state, ckpt_path)
                    tqdm.write(f"    [OK] Saved best model (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
                if loss_up:
                    best_val_loss = v_loss
                if f1_up or loss_up:
                    no_improve = 0
                else:
                    no_improve += 1
                avg_loss = total_loss / len(train_loader)
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} Acc={v_acc*100:.1f}% F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
                if no_improve >= patience:
                    pbar.close()
                    print(f"    Early stop at epoch {epoch+1}"); break
            else:
                pbar.close()
            if best_state:
                model.load_state_dict(best_state)

            for test_subj in subjects:
                if int(test_subj) in train_combo:
                    continue
                test_mask = user_ids == test_subj
                X_test = data_norm[test_mask]
                y_test = act_labels[test_mask]
                D_test_limu = D_all_limu[test_mask] if D_all_limu is not None else None
                if len(X_test) == 0:
                    continue
                test_loader = make_loader(X_test, y_test, random_crop=False)
                model.eval(); tp, tt = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        tp.append(model(bx.to(device), False).argmax(1).cpu().numpy())
                        tt.append(by.numpy())
                tp, tt = np.concatenate(tp), np.concatenate(tt)
                fold_res = evaluate_fold(
                    tt, tp, int(test_subj), len(X_train), len(X_test), no_bike,
                    device_ids=D_test_limu)
                fold_res['train_subjects'] = list(train_combo)
                results.append(fold_res)
                print(f"    Train=({combo_str}) -> Test={test_subj}: Acc={fold_res['accuracy']*100:.2f}%")

            del model, optimizer, best_state
            torch.cuda.empty_cache(); import gc; gc.collect()

        return save_results(results, model_tag, args.output_dir,
                            str(pretrained_path) if pretrained_path else None,
                            experiment_tag=args.experiment_tag,
                            extra_meta={'device_type': dt, 'data_fraction': args.data_fraction,
                                        'train_subjects': K,
                                        'protocol': f'leave_{K}_in',
                                        'no_bike': no_bike, 'num_classes': num_classes,
                                        'seq_len': seq_len, 'seed': args.seed})

    for fold_idx, test_subj in enumerate(subjects):
        if fold_idx < getattr(args, 'start_fold', 0):
            print(f"\n  Fold {fold_idx+1}/{len(subjects)}: test={test_subj} -- SKIPPED (--start-fold {args.start_fold})")
            results.append({'subject': int(test_subj), 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        print(f"\n{'='*50}")
        print(f"  LOSO Fold {fold_idx+1}/{len(subjects)}: test={test_subj}")
        print(f"{'='*50}")

        torch.manual_seed(args.seed); np.random.seed(args.seed)

        test_mask  = user_ids == test_subj
        train_mask = ~test_mask

        if getattr(args, 'train_subjects', None) is not None:
            available_train_subjects = sorted(np.unique(user_ids[train_mask]))
            keep_subjects = available_train_subjects[:args.train_subjects]
            train_mask = train_mask & np.isin(user_ids, keep_subjects)

        X_train_all = data_norm[train_mask]
        y_train_all = act_labels[train_mask]
        X_test      = data_norm[test_mask]
        y_test      = act_labels[test_mask]
        D_test_limu = D_all_limu[test_mask] if D_all_limu is not None else None

        if len(X_test) == 0:
            results.append({'subject': int(test_subj), 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        # Apply data fraction
        X_train_all, y_train_all = subsample_training_data(
            X_train_all, y_train_all, args.data_fraction, seed=args.seed)

        # Train/val split
        idx = np.arange(len(X_train_all)); np.random.shuffle(idx)
        val_n = max(1, int(len(idx) * 0.1))
        X_val, y_val     = X_train_all[idx[:val_n]], y_train_all[idx[:val_n]]
        X_train, y_train = X_train_all[idx[val_n:]], y_train_all[idx[val_n:]]

        class _CropDataset(torch.utils.data.Dataset):
            """When model seq_len < data length, crop subsequences."""
            def __init__(self, X, y, target_len, random_crop=True):
                self.X = torch.from_numpy(X).float()
                self.y = torch.from_numpy(y).long()
                self.target_len = target_len
                self.random_crop = random_crop
            def __len__(self): return len(self.X)
            def __getitem__(self, idx):
                x = self.X[idx]
                if self.target_len < x.size(0):
                    if self.random_crop:
                        s = torch.randint(0, x.size(0) - self.target_len, (1,)).item()
                    else:
                        s = (x.size(0) - self.target_len) // 2
                    x = x[s:s + self.target_len]
                return x, self.y[idx]

        data_seq_len = X_train.shape[1]
        def make_loader(X, y, shuffle=False, random_crop=False):
            if seq_len < data_seq_len:
                ds = _CropDataset(X, y, seq_len, random_crop=random_crop)
            else:
                ds = TensorDataset(torch.from_numpy(X).float(),
                                   torch.from_numpy(y).long())
            return DataLoader(ds, batch_size=batch_size,
                              shuffle=shuffle, num_workers=0)

        train_loader = make_loader(X_train, y_train, shuffle=True, random_crop=True)
        val_loader   = make_loader(X_val, y_val, random_crop=False)
        test_loader  = make_loader(X_test, y_test, random_crop=False)

        # Checkpoint path
        exp_tag = args.experiment_tag or 'run'
        ckpt_path = BEST_MODELS_DIR / f"limubert_{exp_tag}_fold{fold_idx}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden,
                                   output=num_classes)
        model = BERTClassifier(bert_cfg, classifier=classifier,
                               frozen_bert=False)

        # --- Resume: skip fold if already has test result in progress JSON ---
        _ptag = str(pretrained_path) if pretrained_path else None
        _tag_name = model_tag
        if getattr(args, 'resume', False) and is_fold_completed(
                args.output_dir, args.experiment_tag, _tag_name, _ptag, int(test_subj)):
            print(f"    [OK] Fold already completed, skipping")
            _prog = _incremental_results_path(args.output_dir, args.experiment_tag, _tag_name, _ptag)
            with open(_prog) as _f:
                _prev = json.load(_f)
            for _r in _prev['per_fold']:
                if _r.get('subject') == int(test_subj):
                    results.append(_r); break
            continue
        else:
            # --- Training (from scratch, pretrained, or resume from checkpoint) ---
            if getattr(args, 'resume', False) and ckpt_path.exists():
                best_state = torch.load(str(ckpt_path), map_location=device)
                model.load_state_dict(best_state)
                print(f"    [OK] Resuming training from checkpoint -> {ckpt_path}")
            elif pretrained_path and pretrained_path.exists():
                print(f"  Loading pretrained: {pretrained_path}")
                state_dict = model.state_dict()
                pretrained = torch.load(str(pretrained_path), map_location=device)
                loaded = 0
                for k, v in pretrained.items():
                    if k in state_dict and state_dict[k].shape == v.shape:
                        state_dict[k] = v; loaded += 1
                model.load_state_dict(state_dict)
                print(f"    Loaded {loaded}/{len(pretrained)} weight tensors")

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            best_val_f1, best_val_loss, best_state, no_improve, patience = -1.0, float('inf'), None, 0, 15
            pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                        bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                model.train(); total_loss = 0
                optimizer.zero_grad()
                for step, (bx, by) in enumerate(train_loader):
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(model(bx, True), by) / grad_accum
                    loss.backward()
                    if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                        optimizer.step(); optimizer.zero_grad()
                    total_loss += loss.item() * grad_accum

                model.eval(); vp, vt, vl = [], [], 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx_d, by_d = bx.to(device), by.to(device)
                        logits = model(bx_d, False)
                        vp.append(logits.argmax(1).cpu().numpy())
                        vt.append(by.numpy())
                        vl += criterion(logits, by_d).item()
                vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
                v_f1 = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
                v_acc = accuracy_score(vt_cat, vp_cat)
                v_loss = vl / max(len(val_loader), 1)
                f1_up = v_f1 > best_val_f1
                loss_up = v_loss < best_val_loss
                if f1_up:
                    best_val_f1 = v_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    _safe_torch_save(best_state, ckpt_path)
                    tqdm.write(f"    [OK] Saved best model (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
                if loss_up:
                    best_val_loss = v_loss
                if f1_up or loss_up:
                    no_improve = 0
                else:
                    no_improve += 1
                avg_loss = total_loss / len(train_loader)
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} Acc={v_acc*100:.1f}% F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
                if no_improve >= patience:
                    pbar.close()
                    print(f"    Early stop at epoch {epoch+1}"); break
            else:
                pbar.close()

            if best_state: model.load_state_dict(best_state)

        # --- Test evaluation (always runs) ---
        model = model.to(device)
        model.eval(); tp, tt = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                tp.append(model(bx.to(device), False).argmax(1).cpu().numpy())
                tt.append(by.numpy())
        tp, tt = np.concatenate(tp), np.concatenate(tt)
        fold_res = evaluate_fold(
            tt, tp, int(test_subj), len(X_train), len(X_test), no_bike,
            device_ids=D_test_limu)
        results.append(fold_res)
        _ptag = str(pretrained_path) if pretrained_path else None
        save_fold_result(fold_res, args.output_dir, args.experiment_tag,
                         model_tag, _ptag,
                         extra_meta={'device_type': dt, 'seq_len': seq_len})

        del model, optimizer, best_state
        torch.cuda.empty_cache(); import gc; gc.collect()

    return save_results(results, model_tag, args.output_dir,
                        str(pretrained_path) if pretrained_path else None,
                        experiment_tag=args.experiment_tag,
                        extra_meta={'device_type': dt, 'data_fraction': args.data_fraction,
                                    'train_subjects': getattr(args, 'train_subjects', None),
                                    'no_bike': no_bike, 'num_classes': num_classes,
                                    'seq_len': seq_len, 'seed': args.seed})


# =====================================================================
#  ssl-wearables  (PyTorch)
# =====================================================================
def train_ssl_wearables_loso(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [ssl-wearables] PyTorch {torch.__version__}, device={device}"
          f"{'  >>>  WARNING: training on CPU! <<<' if device.type == 'cpu' else ''}")

    sys.path.insert(0, str(SSL_DIR))
    from sslearning.models.accNet import Resnet

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 64
    grad_accum = getattr(args, 'grad_accum', 1)
    lr         = 1e-4
    no_bike    = getattr(args, 'no_bike', False)
    num_classes = 5 if no_bike else 6
    input_size  = 300
    n_channels  = args.channels

    # --- Resolve data directory ---
    dt = args.device_type
    nobike_tag = '_nobike' if no_bike else ''
    dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
               'all': f'hhar_all{nobike_tag}'}
    data_dir = SSL_DIR / "data" / "downstream" / dir_map[dt]

    if n_channels == 6:
        x_file = data_dir / "X6.npy"
        if not x_file.exists():
            print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py first.")
            sys.exit(1)
        print(f"  Using 6-channel data (acc+gyro) [{dt}]")
    else:
        x_file = data_dir / "X.npy"
        if not x_file.exists():
            print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py first.")
            sys.exit(1)
        print(f"  Using 3-channel data (acc only) [{dt}]")

    X_all = np.load(str(x_file)).astype(np.float32)
    Y_all = np.load(str(data_dir / "Y.npy")).astype(np.int64)
    P_all = np.load(str(data_dir / "pid.npy")).astype(np.int64)
    dev_file = data_dir / "D.npy"
    D_all = np.load(str(dev_file)).astype(np.int64) if dev_file.exists() else None

    if X_all.shape[1] != input_size:
        from scipy.interpolate import interp1d
        to, tn = np.linspace(0,1,X_all.shape[1]), np.linspace(0,1,input_size)
        X_all = interp1d(to, X_all, kind='linear', axis=1,
                         assume_sorted=True)(tn).astype(np.float32)

    X_all = np.transpose(X_all, (0, 2, 1))  # (N, C, T)
    subjects = sorted(np.unique(P_all).tolist())

    print(f"\nssl-wearables [{dt}]: {len(X_all)} samples, {n_channels}ch, "
          f"{len(subjects)} subjects, frac={args.data_fraction}")

    pretrained_path = None
    if args.pretrained:
        p = _resolve_project_path(args.pretrained)
        if not p.is_absolute() and not p.exists():
            p = SSL_DIR / p
        pretrained_path = p
        print(f"  Pretrained: {pretrained_path}")

    def load_ssl_weights(wpath, mdl, dev):
        pre = torch.load(str(wpath), map_location=dev)
        pre2 = copy.deepcopy(pre)
        head = next(iter(pre2)).split(".")[0]
        if head == "module":
            pre2 = {k.partition("module.")[2]: v for k, v in pre2.items()}
        md = mdl.state_dict()
        filt, skip = {}, []
        for k, v in pre2.items():
            if k not in md: continue
            if k.split(".")[0] == "classifier": continue
            if md[k].shape != v.shape: skip.append(k); continue
            filt[k] = v
        md.update(filt); mdl.load_state_dict(md)
        print(f"    Loaded {len(filt)} tensors" +
              (f", skipped {len(skip)}" if skip else ""))

    exp_tag_g = args.experiment_tag or 'run'
    _ptag_g = str(pretrained_path) if pretrained_path else None
    if getattr(args, 'resume', False):
        ckpt_pat = str(BEST_MODELS_DIR / f"ssl_{exp_tag_g}_fold{{fold_idx}}.pt")
        print_resume_status(subjects, args.output_dir, args.experiment_tag,
                            f'ssl-wearables-{n_channels}ch-{dt}', _ptag_g, ckpt_pat)

    results = []
    if getattr(args, 'leave_one_in', False):
        from itertools import combinations as _combinations
        K = int(getattr(args, 'train_subjects', None) or 1)
        if K < 1 or K >= len(subjects):
            raise ValueError(f"--train-subjects must be in [1, {len(subjects)-1}] for leave-one-in with {len(subjects)} subjects (got {K})")
        train_combos = [tuple(int(s) for s in c) for c in _combinations(sorted(subjects), K)]
        print(f"\n  Using leave-{K}-in protocol ({len(train_combos)} train-subject combos of {K} subjects)")
        for fold_idx, train_combo in enumerate(train_combos):
            combo_str = "_".join(str(s) for s in train_combo)
            if fold_idx < getattr(args, 'start_fold', 0):
                print(f"\n  Fold {fold_idx+1}/{len(train_combos)}: train=({combo_str}) -- SKIPPED (--start-fold {args.start_fold})")
                continue

            print(f"\n{'='*50}")
            print(f"  Leave-{K}-In Fold {fold_idx+1}/{len(train_combos)}: train=({combo_str})")
            print(f"{'='*50}")

            torch.manual_seed(args.seed); np.random.seed(args.seed)
            train_mask = np.isin(P_all, np.array(train_combo))
            X_tr_all = X_all[train_mask]; y_tr_all = Y_all[train_mask]
            if len(X_tr_all) < 2:
                print("  WARNING: not enough train samples for this combo, skipping fold")
                continue

            X_tr_all, y_tr_all = subsample_training_data(X_tr_all, y_tr_all, args.data_fraction, seed=args.seed)
            idx = np.arange(len(X_tr_all)); np.random.shuffle(idx)
            vn = max(1, int(len(idx) * 0.1))
            if vn >= len(idx):
                vn = max(1, len(idx) - 1)
            X_val, y_val = X_tr_all[idx[:vn]], y_tr_all[idx[:vn]]
            X_train, y_train = X_tr_all[idx[vn:]], y_tr_all[idx[vn:]]
            if len(X_train) == 0:
                print("  WARNING: empty train split after val split, skipping fold")
                continue

            def make_loader(X, y, shuffle=False):
                ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

            train_loader = make_loader(X_train, y_train, shuffle=True)
            val_loader = make_loader(X_val, y_val)

            exp_tag = args.experiment_tag or 'run'
            ckpt_path = BEST_MODELS_DIR / f"ssl_{exp_tag}_K{K}_trainfold{fold_idx}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            model = Resnet(output_size=num_classes, n_channels=n_channels, is_eva=True, resnet_version=1, epoch_len=10)
            if pretrained_path and pretrained_path.exists():
                print(f"  Loading pretrained: {pretrained_path}")
                load_ssl_weights(pretrained_path, model, device)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
            import collections
            counter = collections.Counter(y_train.tolist())
            wts = [1.0 / (counter.get(i, 1) / len(y_train)) for i in range(num_classes)]
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(wts).to(device))

            best_val_f1, best_val_loss, patience, no_improve, best_state = -1.0, float('inf'), 10, 0, None
            pbar = tqdm(range(epochs), desc="    Training", unit="ep", bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                model.train(); tl = 0
                optimizer.zero_grad()
                for step, (bx, by) in enumerate(train_loader):
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(model(bx), by) / grad_accum
                    loss.backward()
                    if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                        optimizer.step(); optimizer.zero_grad()
                    tl += loss.item() * grad_accum

                model.eval(); vp, vt, vl = [], [], 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx_d, by_d = bx.to(device), by.to(device)
                        logits = model(bx_d)
                        vp.append(logits.argmax(1).cpu().numpy())
                        vt.append(by.numpy())
                        vl += criterion(logits, by_d).item()
                vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
                v_f1 = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
                v_acc = accuracy_score(vt_cat, vp_cat)
                v_loss = vl / max(len(val_loader), 1)
                f1_up = v_f1 > best_val_f1
                loss_up = v_loss < best_val_loss
                if f1_up:
                    best_val_f1 = v_f1
                    best_state = copy.deepcopy(model.state_dict())
                    _safe_torch_save(best_state, ckpt_path)
                    tqdm.write(f"    [OK] Saved best model (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
                if loss_up:
                    best_val_loss = v_loss
                if f1_up or loss_up:
                    no_improve = 0
                else:
                    no_improve += 1
                avg_loss = tl / len(train_loader)
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} Acc={v_acc*100:.1f}% F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
                if no_improve >= patience:
                    pbar.close()
                    print(f"    Early stop at epoch {epoch+1}"); break
            else:
                pbar.close()
            if best_state:
                model.load_state_dict(best_state)

            for test_subj in subjects:
                if int(test_subj) in train_combo:
                    continue
                test_mask = P_all == test_subj
                X_test = X_all[test_mask]; y_test = Y_all[test_mask]
                D_test_ssl = D_all[test_mask] if D_all is not None else None
                if len(X_test) == 0:
                    continue
                test_loader = make_loader(X_test, y_test)
                model = model.to(device)
                model.eval(); tp, tt = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        tp.append(model(bx.to(device)).argmax(1).cpu().numpy())
                        tt.append(by.numpy())
                tp, tt = np.concatenate(tp), np.concatenate(tt)
                fold_res = evaluate_fold(
                    tt, tp, int(test_subj), len(X_train), len(X_test), no_bike,
                    device_ids=D_test_ssl)
                fold_res['train_subjects'] = list(train_combo)
                results.append(fold_res)
                print(f"    Train=({combo_str}) -> Test={test_subj}: Acc={fold_res['accuracy']*100:.2f}%")

            del model
            if 'optimizer' in dir():
                del optimizer
            torch.cuda.empty_cache(); import gc; gc.collect()

        tag_name = f'ssl-wearables-{n_channels}ch-{dt}'
        return save_results(results, tag_name, args.output_dir,
                            str(pretrained_path) if pretrained_path else None,
                            experiment_tag=args.experiment_tag,
                            extra_meta={'device_type': dt, 'channels': n_channels,
                                        'data_fraction': args.data_fraction,
                                        'train_subjects': K,
                                        'protocol': f'leave_{K}_in',
                                        'no_bike': no_bike, 'num_classes': num_classes,
                                        'seed': args.seed})

    for fold_idx, test_subj in enumerate(subjects):
        if fold_idx < getattr(args, 'start_fold', 0):
            print(f"\n  Fold {fold_idx+1}/{len(subjects)}: test={test_subj} -- SKIPPED (--start-fold {args.start_fold})")
            results.append({'subject': int(test_subj), 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        print(f"\n{'='*50}")
        print(f"  LOSO Fold {fold_idx+1}/{len(subjects)}: test={test_subj}")
        print(f"{'='*50}")

        torch.manual_seed(args.seed); np.random.seed(args.seed)
        test_mask  = P_all == test_subj
        train_mask = ~test_mask

        if getattr(args, 'train_subjects', None) is not None:
            available_train_subjects = sorted(np.unique(P_all[train_mask]))
            keep_subjects = available_train_subjects[:args.train_subjects]
            train_mask = train_mask & np.isin(P_all, keep_subjects)

        X_tr_all = X_all[train_mask]; y_tr_all = Y_all[train_mask]
        X_test   = X_all[test_mask];  y_test   = Y_all[test_mask]
        D_test_ssl = D_all[test_mask] if D_all is not None else None

        if len(X_test) == 0:
            results.append({'subject': int(test_subj), 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        # Subsample channels-first data: need to transpose, subsample, transpose back
        # Actually subsample works on first axis so it's fine
        X_tr_all, y_tr_all = subsample_training_data(X_tr_all, y_tr_all, args.data_fraction, seed=args.seed)

        idx = np.arange(len(X_tr_all)); np.random.shuffle(idx)
        vn = max(1, int(len(idx) * 0.1))
        X_val, y_val     = X_tr_all[idx[:vn]], y_tr_all[idx[:vn]]
        X_train, y_train = X_tr_all[idx[vn:]], y_tr_all[idx[vn:]]

        def make_loader(X, y, shuffle=False):
            ds = TensorDataset(torch.from_numpy(X).float(),
                               torch.from_numpy(y).long())
            return DataLoader(ds, batch_size=batch_size,
                              shuffle=shuffle, num_workers=0)

        train_loader = make_loader(X_train, y_train, shuffle=True)
        val_loader   = make_loader(X_val, y_val)
        test_loader  = make_loader(X_test, y_test)

        # Checkpoint path
        exp_tag = args.experiment_tag or 'run'
        ckpt_path = BEST_MODELS_DIR / f"ssl_{exp_tag}_fold{fold_idx}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        model = Resnet(output_size=num_classes, n_channels=n_channels,
                       is_eva=True, resnet_version=1, epoch_len=10)

        # --- Resume: skip fold if already has test result in progress JSON ---
        _ptag = str(pretrained_path) if pretrained_path else None
        _tag_name = f'ssl-wearables-{n_channels}ch-{dt}'
        if getattr(args, 'resume', False) and is_fold_completed(
                args.output_dir, args.experiment_tag, _tag_name, _ptag, int(test_subj)):
            print(f"    [OK] Fold already completed, skipping")
            _prog = _incremental_results_path(args.output_dir, args.experiment_tag, _tag_name, _ptag)
            with open(_prog) as _f:
                _prev = json.load(_f)
            for _r in _prev['per_fold']:
                if _r.get('subject') == int(test_subj):
                    results.append(_r); break
            continue
        else:
            # --- Training (from scratch, pretrained, or resume from checkpoint) ---
            if getattr(args, 'resume', False) and ckpt_path.exists():
                best_state = torch.load(str(ckpt_path), map_location=device)
                model.load_state_dict(best_state)
                print(f"    [OK] Resuming training from checkpoint -> {ckpt_path}")
            elif pretrained_path and pretrained_path.exists():
                print(f"  Loading pretrained: {pretrained_path}")
                load_ssl_weights(pretrained_path, model, device)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

            import collections
            counter = collections.Counter(y_train.tolist())
            wts = [1.0 / (counter.get(i, 1) / len(y_train)) for i in range(num_classes)]
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(wts).to(device))

            best_val_f1, best_val_loss, patience, no_improve, best_state = -1.0, float('inf'), 10, 0, None
            pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                        bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                model.train(); tl = 0
                optimizer.zero_grad()
                for step, (bx, by) in enumerate(train_loader):
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(model(bx), by) / grad_accum
                    loss.backward()
                    if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                        optimizer.step(); optimizer.zero_grad()
                    tl += loss.item() * grad_accum

                model.eval(); vp, vt, vl = [], [], 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx_d, by_d = bx.to(device), by.to(device)
                        logits = model(bx_d)
                        vp.append(logits.argmax(1).cpu().numpy())
                        vt.append(by.numpy())
                        vl += criterion(logits, by_d).item()
                vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
                v_f1 = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
                v_acc = accuracy_score(vt_cat, vp_cat)
                v_loss = vl / max(len(val_loader), 1)
                f1_up = v_f1 > best_val_f1
                loss_up = v_loss < best_val_loss
                if f1_up:
                    best_val_f1 = v_f1
                    best_state = copy.deepcopy(model.state_dict())
                    _safe_torch_save(best_state, ckpt_path)
                    tqdm.write(f"    [OK] Saved best model (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
                if loss_up:
                    best_val_loss = v_loss
                if f1_up or loss_up:
                    no_improve = 0
                else:
                    no_improve += 1
                avg_loss = tl / len(train_loader)
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} Acc={v_acc*100:.1f}% F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
                if no_improve >= patience:
                    pbar.close()
                    print(f"    Early stop at epoch {epoch+1}"); break
            else:
                pbar.close()

            if best_state: model.load_state_dict(best_state)

        # --- Test evaluation (always runs) ---
        model = model.to(device)
        model.eval(); tp, tt = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                tp.append(model(bx.to(device)).argmax(1).cpu().numpy())
                tt.append(by.numpy())
        tp, tt = np.concatenate(tp), np.concatenate(tt)
        fold_res = evaluate_fold(
            tt, tp, int(test_subj), len(X_train), len(X_test), no_bike,
            device_ids=D_test_ssl)
        results.append(fold_res)
        _ptag = str(pretrained_path) if pretrained_path else None
        save_fold_result(fold_res, args.output_dir, args.experiment_tag,
                         f'ssl-wearables-{n_channels}ch-{dt}', _ptag,
                         extra_meta={'device_type': dt, 'channels': n_channels})

        del model
        if 'optimizer' in dir(): del optimizer
        torch.cuda.empty_cache(); import gc; gc.collect()

    tag_name = f'ssl-wearables-{n_channels}ch-{dt}'
    return save_results(results, tag_name, args.output_dir,
                        str(pretrained_path) if pretrained_path else None,
                        experiment_tag=args.experiment_tag,
                        extra_meta={'device_type': dt, 'channels': n_channels,
                                    'data_fraction': args.data_fraction,
                                    'train_subjects': getattr(args, 'train_subjects', None),
                                    'no_bike': no_bike, 'num_classes': num_classes,
                                    'seed': args.seed})


# =====================================================================
#  ResNet Baseline  (PyTorch) -- basic 1D ResNet, no SSL pretraining
# =====================================================================
def train_resnet_baseline_loso(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, str(RESNET_BASE_DIR))
    from resnet1d_baseline import ResNet1DBaseline

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [ResNet-Baseline] PyTorch {torch.__version__}, device={device}"
          f"{'  >>>  WARNING: training on CPU! <<<' if device.type == 'cpu' else ''}")

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 64
    grad_accum = getattr(args, 'grad_accum', 1)
    lr         = 1e-4
    no_bike    = getattr(args, 'no_bike', False)
    num_classes = 5 if no_bike else 6
    input_size  = 300          # same as ssl-wearables
    n_channels  = args.channels

    # --- Resolve data directory (re-use ssl-wearables preprocessed data) ---
    dt = args.device_type
    nobike_tag = '_nobike' if no_bike else ''
    dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
               'all': f'hhar_all{nobike_tag}'}
    data_dir = SSL_DIR / "data" / "downstream" / dir_map[dt]

    if n_channels == 6:
        x_file = data_dir / "X6.npy"
        if not x_file.exists():
            print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py first.")
            sys.exit(1)
        print(f"  Using 6-channel data (acc+gyro) [{dt}]")
    else:
        x_file = data_dir / "X.npy"
        if not x_file.exists():
            print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py first.")
            sys.exit(1)
        print(f"  Using 3-channel data (acc only) [{dt}]")

    X_all = np.load(str(x_file)).astype(np.float32)
    Y_all = np.load(str(data_dir / "Y.npy")).astype(np.int64)
    P_all = np.load(str(data_dir / "pid.npy")).astype(np.int64)
    dev_file = data_dir / "D.npy"
    D_all = np.load(str(dev_file)).astype(np.int64) if dev_file.exists() else None

    if X_all.shape[1] != input_size:
        from scipy.interpolate import interp1d
        to, tn = np.linspace(0, 1, X_all.shape[1]), np.linspace(0, 1, input_size)
        X_all = interp1d(to, X_all, kind='linear', axis=1,
                         assume_sorted=True)(tn).astype(np.float32)

    X_all = np.transpose(X_all, (0, 2, 1))   # (N, C, T)
    subjects = sorted(np.unique(P_all).tolist())

    print(f"\nResNet-Baseline [{dt}]: {len(X_all)} samples, {n_channels}ch, "
          f"{len(subjects)} subjects, frac={args.data_fraction}")

    tag_name = f'resnet-baseline-{n_channels}ch-{dt}'
    ckpt_prefix = 'resbase'

    exp_tag_g = args.experiment_tag or 'run'
    if getattr(args, 'resume', False):
        ckpt_pat = str(BEST_MODELS_DIR / f"{ckpt_prefix}_{exp_tag_g}_fold{{fold_idx}}.pt")
        print_resume_status(subjects, args.output_dir, args.experiment_tag,
                            tag_name, None, ckpt_pat)

    results = []
    for fold_idx, test_subj in enumerate(subjects):
        if fold_idx < getattr(args, 'start_fold', 0):
            print(f"\n  Fold {fold_idx+1}/{len(subjects)}: test={test_subj}"
                  f" -- SKIPPED (--start-fold {args.start_fold})")
            results.append({'subject': int(test_subj), 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        print(f"\n{'='*50}")
        print(f"  LOSO Fold {fold_idx+1}/{len(subjects)}: test={test_subj}")
        print(f"{'='*50}")

        torch.manual_seed(args.seed); np.random.seed(args.seed)
        test_mask  = P_all == test_subj
        train_mask = ~test_mask

        if getattr(args, 'train_subjects', None) is not None:
            available_train_subjects = sorted(np.unique(P_all[train_mask]))
            keep_subjects = available_train_subjects[:args.train_subjects]
            train_mask = train_mask & np.isin(P_all, keep_subjects)

        X_tr_all = X_all[train_mask]; y_tr_all = Y_all[train_mask]
        X_test   = X_all[test_mask];  y_test   = Y_all[test_mask]
        D_test_res = D_all[test_mask] if D_all is not None else None

        if len(X_test) == 0:
            results.append({'subject': int(test_subj), 'accuracy': None,
                            'f1_weighted': None, 'f1_macro': None})
            continue

        X_tr_all, y_tr_all = subsample_training_data(
            X_tr_all, y_tr_all, args.data_fraction, seed=args.seed)

        idx = np.arange(len(X_tr_all)); np.random.shuffle(idx)
        vn = max(1, int(len(idx) * 0.1))
        X_val, y_val     = X_tr_all[idx[:vn]], y_tr_all[idx[:vn]]
        X_train, y_train = X_tr_all[idx[vn:]], y_tr_all[idx[vn:]]

        def make_loader(X, y, shuffle=False):
            ds = TensorDataset(torch.from_numpy(X).float(),
                               torch.from_numpy(y).long())
            return DataLoader(ds, batch_size=batch_size,
                              shuffle=shuffle, num_workers=0)

        train_loader = make_loader(X_train, y_train, shuffle=True)
        val_loader   = make_loader(X_val, y_val)
        test_loader  = make_loader(X_test, y_test)

        # Checkpoint path
        exp_tag = args.experiment_tag or 'run'
        ckpt_path = BEST_MODELS_DIR / f"{ckpt_prefix}_{exp_tag}_fold{fold_idx}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        model = ResNet1DBaseline(n_channels=n_channels,
                                 num_classes=num_classes, kernel_size=5)

        # --- Resume: skip fold if already completed ---
        if getattr(args, 'resume', False) and is_fold_completed(
                args.output_dir, args.experiment_tag, tag_name, None,
                int(test_subj)):
            print(f"    [OK] Fold already completed, skipping")
            _prog = _incremental_results_path(
                args.output_dir, args.experiment_tag, tag_name, None)
            with open(_prog) as _f:
                _prev = json.load(_f)
            for _r in _prev['per_fold']:
                if _r.get('subject') == int(test_subj):
                    results.append(_r); break
            continue
        else:
            # --- Resume from checkpoint if available ---
            if getattr(args, 'resume', False) and ckpt_path.exists():
                best_state = torch.load(str(ckpt_path), map_location=device)
                model.load_state_dict(best_state)
                print(f"    [OK] Resuming training from checkpoint -> {ckpt_path}")

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         amsgrad=True)

            import collections
            counter = collections.Counter(y_train.tolist())
            wts = [1.0 / (counter.get(i, 1) / len(y_train))
                   for i in range(num_classes)]
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(wts).to(device))

            best_val_f1, best_val_loss, patience, no_improve, best_state = -1.0, float('inf'), 10, 0, None
            pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                        bar_format="{l_bar}{bar:20}{r_bar}")
            for epoch in pbar:
                model.train(); tl = 0
                optimizer.zero_grad()
                for step, (bx, by) in enumerate(train_loader):
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(model(bx), by) / grad_accum
                    loss.backward()
                    if ((step + 1) % grad_accum == 0
                            or (step + 1) == len(train_loader)):
                        optimizer.step(); optimizer.zero_grad()
                    tl += loss.item() * grad_accum

                model.eval(); vp, vt, vl = [], [], 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx_d, by_d = bx.to(device), by.to(device)
                        logits = model(bx_d)
                        vp.append(logits.argmax(1).cpu().numpy())
                        vt.append(by.numpy())
                        vl += criterion(logits, by_d).item()
                vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
                v_f1 = f1_score(vt_cat, vp_cat, average='macro',
                                zero_division=0)
                v_acc = accuracy_score(vt_cat, vp_cat)
                v_loss = vl / max(len(val_loader), 1)
                f1_up = v_f1 > best_val_f1
                loss_up = v_loss < best_val_loss
                if f1_up:
                    best_val_f1 = v_f1
                    best_state = copy.deepcopy(model.state_dict())
                    _safe_torch_save(best_state, ckpt_path)
                    tqdm.write(f"    [OK] Saved best model "
                               f"(F1m={v_f1*100:.1f}%) -> {ckpt_path}")
                if loss_up:
                    best_val_loss = v_loss
                if f1_up or loss_up:
                    no_improve = 0
                else:
                    no_improve += 1
                avg_loss = tl / len(train_loader)
                pbar.set_postfix_str(
                    f"loss={avg_loss:.3f} Acc={v_acc*100:.1f}% "
                    f"F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
                if no_improve >= patience:
                    pbar.close()
                    print(f"    Early stop at epoch {epoch+1}"); break
            else:
                pbar.close()

            if best_state:
                model.load_state_dict(best_state)

        # --- Test evaluation (always runs) ---
        model = model.to(device)
        model.eval(); tp, tt = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                tp.append(model(bx.to(device)).argmax(1).cpu().numpy())
                tt.append(by.numpy())
        tp, tt = np.concatenate(tp), np.concatenate(tt)
        fold_res = evaluate_fold(
            tt, tp, int(test_subj), len(X_train), len(X_test), no_bike,
            device_ids=D_test_res)
        results.append(fold_res)
        save_fold_result(fold_res, args.output_dir, args.experiment_tag,
                         tag_name, None,
                         extra_meta={'device_type': dt, 'channels': n_channels})

        del model
        if 'optimizer' in dir():
            del optimizer
        torch.cuda.empty_cache(); import gc; gc.collect()

    return save_results(results, tag_name, args.output_dir, None,
                        experiment_tag=args.experiment_tag,
                        extra_meta={'device_type': dt, 'channels': n_channels,
                                    'data_fraction': args.data_fraction,
                                    'train_subjects': getattr(args, 'train_subjects', None),
                                    'no_bike': no_bike, 'num_classes': num_classes,
                                    'seed': args.seed})


# =====================================================================
#  Main CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified LOSO training with benchmark comparison support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark Experiments:
  Exp1: Pretraining vs supervised (Transformer, phone, accel+gyro)
  Exp2: BERT for smartwatch? (Transformer, watch, accel+gyro)
  Exp3: HART accel vs accel+gyro self-comparison
  Exp4: Pretraining vs supervised (CNN, watch, accel only)
  Exp5: HARNet for smartphone? (CNN, phone, accel only)

Examples:
  python train_loso.py --model hart --device-type phone --experiment-tag exp1b
  python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.5 --experiment-tag exp1a
  python train_loso.py --model hart --channels 3 --experiment-tag exp3_accel_only
  python train_loso.py --model ssl-wearables --device-type watch --channels 3 --experiment-tag exp4b
""")
    parser.add_argument('--model', required=True,
                        choices=['hart', 'mobilehart', 'limu-bert',
                                 'ssl-wearables', 'resnet-baseline'])
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: HART=256, LIMU=128, SSL=64)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--channels', type=int, default=6, choices=[3, 6],
                        help='Input channels: 3=acc only, 6=acc+gyro')
    parser.add_argument('--device-type', type=str, default='all',
                        choices=['phone', 'watch', 'all'],
                        help='HHAR device type: phone, watch, or all')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='Fraction of training data (0.0-1.0, default=1.0)')
    parser.add_argument('--train-subjects', type=int, default=None,
                        help='Restrict training to this many subjects (e.g., 1 for single-subject fine-tuning)')
    parser.add_argument('--leave-one-in', action='store_true',
                        help='Leave-one-in protocol: train on one subject, test on each remaining subject')
    parser.add_argument('--experiment-tag', type=str, default=None,
                        help='Tag for organising results (e.g. exp1a)')
    parser.add_argument('--no-bike', action='store_true',
                        help='Use 5-class data (exclude bike activity)')
    parser.add_argument('--limu-seq-len', type=int, default=120, choices=[20, 120],
                        help='LIMU-BERT sequence length: 120=LIMU-BERT (full), 20=LIMU-BERT-X (random crop)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split, initialization, and subsampling')
    parser.add_argument('--repeat-seeds', type=int, default=1,
                        help='Run the same configuration across consecutive seeds starting from --seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip folds that already have saved best-model checkpoints')
    parser.add_argument('--start-fold', type=int, default=0,
                        help='Start from this fold index (0-based, e.g. --start-fold 3 skips folds 0,1,2)')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()
    args.output_dir = str(_resolve_project_path(args.output_dir))
    if args.pretrained:
        args.pretrained = str(_resolve_project_path(args.pretrained))

    if args.gpu == '-1': args.gpu = None

    # Determine effective defaults for display
    default_bs = {'hart': 256, 'mobilehart': 256, 'limu-bert': 128,
                  'ssl-wearables': 64, 'resnet-baseline': 64}
    eff_bs = args.batch_size or default_bs.get(args.model, 128)
    eff_epochs = args.epochs or 200

    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Device type : {args.device_type}")
    print(f"  Channels    : {args.channels} ({'acc+gyro' if args.channels == 6 else 'acc only'})")
    print(f"  Epochs      : {eff_epochs}")
    print(f"  Batch size  : {eff_bs}" + (f"  (effective: {eff_bs * args.grad_accum} with grad_accum={args.grad_accum})" if args.grad_accum > 1 else ""))
    print(f"  Grad accum  : {args.grad_accum}")
    print(f"  Data frac   : {args.data_fraction}")
    if getattr(args, 'train_subjects', None):
        print(f"  Train subjs : {args.train_subjects}")
    print(f"  Pretrained  : {args.pretrained or 'None (scratch)'}")
    if args.model == 'limu-bert':
        print(f"  LIMU seq_len: {args.limu_seq_len} ({'LIMU-BERT-X' if args.limu_seq_len == 20 else 'LIMU-BERT'})")
    print(f"  Seed       : {args.seed}")
    print(f"  Repeats    : {args.repeat_seeds}")
    print(f"  GPU         : {args.gpu if args.gpu else 'CPU only'}")
    print(f"  No bike     : {args.no_bike}")
    print(f"  Leave-one-in: {args.leave_one_in}")
    print(f"  Resume      : {args.resume}")
    print(f"  Start fold  : {args.start_fold}" + (f"  (skipping folds 0-{args.start_fold-1})" if args.start_fold > 0 else ""))
    print(f"  Exp tag     : {args.experiment_tag or 'none'}")
    print("=" * 60)

    dispatch = {
        'hart': train_hart_loso,
        'mobilehart': train_hart_loso,
        'limu-bert': train_limu_bert_loso,
        'ssl-wearables': train_ssl_wearables_loso,
        'resnet-baseline': train_resnet_baseline_loso,
    }

    run_summaries = []
    base_seed = int(args.seed)
    for seed_offset in range(max(1, args.repeat_seeds)):
        args.seed = base_seed + seed_offset
        print(f"\n  >>> Running seed {args.seed}")
        run_summaries.append(dispatch[args.model](args))

    if len(run_summaries) > 1:
        model_name = run_summaries[-1].get('model')
        K = getattr(args, 'train_subjects', None)
        is_leave_in = bool(getattr(args, 'leave_one_in', False))
        extra_meta = {
            'pretrained': run_summaries[-1].get('pretrained'),
            'device_type': args.device_type,
            'channels': args.channels,
            'data_fraction': args.data_fraction,
            'train_subjects': K,
            'no_bike': args.no_bike,
            'seed': None,
        }
        if is_leave_in and K:
            extra_meta['protocol'] = f'leave_{int(K)}_in'
        if args.model == 'limu-bert':
            extra_meta['seq_len'] = args.limu_seq_len
        save_repeated_results(run_summaries, args.output_dir, args.experiment_tag, model_name, extra_meta=extra_meta)

if __name__ == '__main__':
    main()
