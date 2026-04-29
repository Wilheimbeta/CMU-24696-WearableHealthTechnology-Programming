#!/usr/bin/env python
"""
=============================================================================
Cross-Dataset Evaluation -- EXP5/6, EXP5.1/6.1
=============================================================================

Train each model on one dataset and evaluate on the other (no LOSO).

  Default  : Train HHAR, test <target>  (Exp 5 / 5.1)
  --reverse: Train <target>, test HHAR  (Exp 6 / 6.1)

Target datasets (--target-dataset):
  wisdm              WISDM              (Exp 5 / 6)
  pamap2_sbhar       PAMAP2+SBHAR 8+8   (Exp 5.1 / 6.1, aligned, default)
  pamap2_sbhar_full  PAMAP2+SBHAR 8+30  (Exp 5.1 / 6.1, full)

Two class modes:
  Default (3-class): filtered to sit/stand/walk.
  --with-stairs    : 5 classes (sit/stand/walk/upstair/downstairs),
                     main metrics on 3 classes, supplementary stairs analysis.

Usage:
  python eval_cross_dataset.py --model hart --device-type phone --channels 6 --experiment-tag exp5_hart_phone --target-dataset wisdm
  python eval_cross_dataset.py --model hart --device-type phone --channels 6 --reverse --experiment-tag exp6_hart_phone --target-dataset wisdm
  python eval_cross_dataset.py --model hart --device-type phone --channels 6 --target-dataset pamap2_sbhar --experiment-tag exp51_hart_phone
  python eval_cross_dataset.py --model hart --device-type phone --channels 6 --target-dataset pamap2_sbhar --reverse --experiment-tag exp61_hart_phone
"""

import os, sys, json, time, argparse, copy, collections, re
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths (same as train_loso.py) =====================
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

LABELS_5CLS = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
LABELS_3CLS = ['Sitting', 'Standing', 'Walking']

# ===================== Target Dataset Config =====================
# Maps --target-dataset value to model-specific directory names.
# "pamap2_sbhar" defaults to aligned 8+8; use "pamap2_sbhar_full" for 8+30.
_TARGET_DS_CFG = {
    'wisdm': {
        'hart':  {'all': 'WISDM',          'phone': 'WISDM_phone',          'watch': 'WISDM_watch'},
        'limu':  {'phone': 'wisdm',         'watch': 'wisdm_watch',          'all': 'wisdm_all'},
        'ssl':   {'phone': 'wisdm',         'watch': 'wisdm_watch',          'all': 'wisdm_all'},
        'map_prefix':     'wisdm',
        'display_name':   'WISDM',
        'prepare_script': 'prepare_wisdm_data.py',
    },
    'pamap2_sbhar': {
        'hart':  {'all': 'PAMAP2_SBHAR_aligned',       'phone': 'PAMAP2_SBHAR_aligned_phone',       'watch': 'PAMAP2_SBHAR_aligned_watch'},
        'limu':  {'phone': 'pamap2_sbhar_aligned',      'watch': 'pamap2_sbhar_aligned_watch',       'all': 'pamap2_sbhar_aligned_all'},
        'ssl':   {'phone': 'pamap2_sbhar_aligned',      'watch': 'pamap2_sbhar_aligned_watch',       'all': 'pamap2_sbhar_aligned_all'},
        'map_prefix':     'pamap2_sbhar_aligned',
        'display_name':   'PAMAP2+SBHAR (8+8)',
        'prepare_script': 'prepare_pamap2_sbhar_data.py --aligned',
    },
    'pamap2_sbhar_full': {
        'hart':  {'all': 'PAMAP2_SBHAR',                'phone': 'PAMAP2_SBHAR_phone',               'watch': 'PAMAP2_SBHAR_watch'},
        'limu':  {'phone': 'pamap2_sbhar',               'watch': 'pamap2_sbhar_watch',               'all': 'pamap2_sbhar_all'},
        'ssl':   {'phone': 'pamap2_sbhar',               'watch': 'pamap2_sbhar_watch',               'all': 'pamap2_sbhar_all'},
        'map_prefix':     'pamap2_sbhar',
        'display_name':   'PAMAP2+SBHAR (8+30)',
        'prepare_script': 'prepare_pamap2_sbhar_data.py',
    },
}

VALID_TARGET_DATASETS = list(_TARGET_DS_CFG.keys())


def _get_target_cfg(args):
    """Return the target-dataset config dict from args."""
    td = getattr(args, 'target_dataset', 'wisdm')
    if td not in _TARGET_DS_CFG:
        print(f"ERROR: unknown --target-dataset '{td}'. "
              f"Choices: {VALID_TARGET_DATASETS}")
        sys.exit(1)
    return _TARGET_DS_CFG[td]


def _resolve_project_path(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)


def _resolve_limu_pretrained(path_like):
    p = _resolve_project_path(path_like)
    if not p.is_absolute() and not p.exists():
        p = LIMU_DIR / p
    if not str(p).endswith('.pt'):
        p_pt = Path(str(p) + '.pt')
        if p_pt.exists():
            p = p_pt
    if not p.exists():
        p2 = LIMU_DIR / "weights" / p.name
        if p2.exists():
            p = p2
        else:
            p3 = LIMU_DIR / "weights" / (p.stem + '.pt')
            if p3.exists():
                p = p3
    return p


def _resolve_ssl_pretrained(path_like):
    p = _resolve_project_path(path_like)
    if not p.exists():
        p2 = SSL_DIR / Path(path_like)
        if p2.exists():
            p = p2
        else:
            p3 = SSL_DIR / "model_check_point" / Path(path_like).name
            if p3.exists():
                p = p3
    return p


def _infer_additional_pretraining(pretrained):
    text = str(pretrained or "").lower()
    if not text:
        return "None"
    if "wisdm_all" in text:
        return "WISDM all"
    if "wisdm_watch" in text:
        return "WISDM watch"
    if "wisdm_phone" in text:
        return "WISDM phone"
    if "watch_finetune" in text:
        return "PAMAP2+WISDM watch"
    if "phone_finetune" in text:
        return "SBHAR+WISDM phone"
    return "None"


# ===================== Helpers =====================
def _safe_torch_save(obj, path, retries=3, delay=0.5):
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


def set_global_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def summarize_metric_stats(values):
    arr = np.array(values, dtype=float)
    return {
        'mean': float(np.mean(arr)) if len(arr) else 0.0,
        'std': float(np.std(arr)) if len(arr) else 0.0,
        'n': int(len(arr)),
    }


def _extract_run_number(experiment_tag):
    if not experiment_tag:
        return None
    m = re.search(r"(?:^|_)run(\d+)(?:_|$)", str(experiment_tag), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _build_experiment_table_metadata(result):
    model_text = str(result.get('model', '')).lower()
    if "limu-bert" in model_text:
        arch = "LIMU-BERT-X" if "20" in model_text or "x" in model_text else "LIMU-BERT"
    elif "ssl-wearables" in model_text:
        arch = "SSL-Wearables"
    elif "resnet-baseline" in model_text or "resbase" in model_text:
        arch = "ResNet-Baseline"
    elif "hart" in model_text:
        arch = "HART"
    else:
        arch = str(result.get('model', ''))
    direction = str(result.get('direction', ''))
    train_ds, test_ds = "", ""
    if "->" in direction:
        train_ds, test_ds = [x.strip() for x in direction.split("->", 1)]
    data_fraction = result.get('data_fraction', 1.0)
    try:
        data_fraction_txt = f"{float(data_fraction) * 100:.1f}%".replace(".0%", "%")
    except Exception:
        data_fraction_txt = ""
    n_tests = result.get('repeated_seed_summary', {}).get('runs', 1)
    return {
        "Architecture Type": arch,
        "Device Type": str(result.get('device_type', '')).capitalize(),
        "Training Dataset": train_ds,
        "Testing Dataset": test_ds,
        "Imported Pretrained Weights": "Yes" if result.get('pretrained') else "No",
        "3ch vs 6ch data": f"{result.get('channels')}ch" if result.get('channels') else "",
        "Data Fraction": data_fraction_txt,
        "Additional Pretraining dataset": _infer_additional_pretraining(result.get('pretrained')),
        "Testing Method": "Cross",
        "# of Tests": int(n_tests) if n_tests else 1,
        "# Run": _extract_run_number(result.get('experiment_tag')) or "",
    }


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


def aggregate_cross_runs(results):
    summary = {'runs': len(results), 'seeds': [r.get('seed') for r in results]}
    metric_names = ['accuracy', 'f1_weighted', 'f1_macro']
    summary['test_metrics_mean_std'] = {
        name: summarize_metric_stats([r.get('test_metrics', {}).get(name, 0.0) for r in results])
        for name in metric_names
    }
    per_class_f1, per_class_recall = {}, {}
    for cls in LABELS_5CLS:
        f1_vals = [r.get('test_metrics', {}).get('per_class_f1', {}).get(cls, {}).get('f1')
                   for r in results]
        f1_vals = [v for v in f1_vals if v is not None]
        if f1_vals:
            per_class_f1[cls] = summarize_metric_stats(f1_vals)
        rec_vals = [r.get('test_metrics', {}).get('per_class_f1', {}).get(cls, {}).get('recall')
                    for r in results]
        rec_vals = [v for v in rec_vals if v is not None]
        if rec_vals:
            per_class_recall[cls] = summarize_metric_stats(rec_vals)
    if per_class_f1:
        summary['per_class_f1_mean_std'] = per_class_f1
    if per_class_recall:
        summary['per_class_recall_mean_std'] = per_class_recall
    return summary


def print_cross_summary(result):
    tm = result.get('test_metrics', {})
    print("\n" + "=" * 60)
    print("  Cross-dataset Summary")
    print("=" * 60)
    print(f"  Accuracy   : {tm.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  F1 weighted: {tm.get('f1_weighted', 0.0) * 100:.2f}%")
    print(f"  F1 macro   : {tm.get('f1_macro', 0.0) * 100:.2f}%")
    print(f"  n_train/test: {tm.get('n_train', 0)} / {tm.get('n_test', 0)}")
    if result.get('stairs_analysis'):
        print(f"  Stairs analysis: {result['stairs_analysis'].get('prediction_pct', {})}")
    print("=" * 60)


def print_repeated_cross_summary(summary):
    print("\n" + "=" * 60)
    print("  Repeated-run Summary")
    print("=" * 60)
    for name, stats in summary.get('test_metrics_mean_std', {}).items():
        print(f"  {name:<12} mean={stats['mean']*100:.2f}%  std={stats['std']*100:.2f}%  n={stats['n']}")
    print("=" * 60)


def filter_to_3class(X, y):
    """Keep only sit(0), stand(1), walk(2).  Labels already 0-indexed."""
    mask = y <= 2
    return X[mask], y[mask]


def evaluate_test(y_true, y_pred, label_names, n_train, n_test,
                   test_name='WISDM'):
    """Compute cross-dataset test metrics."""
    num_cls = len(label_names)
    cls_labels = list(range(num_cls))
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, labels=cls_labels, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, labels=cls_labels, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=cls_labels).tolist()
    report = classification_report(
        y_true, y_pred, labels=list(range(num_cls)),
        target_names=label_names, output_dict=True, zero_division=0)
    per_class = {}
    for i, name in enumerate(label_names):
        if name in report:
            per_class[name] = {
                'precision': round(report[name]['precision'], 4),
                'recall':    round(report[name]['recall'], 4),
                'f1':        round(report[name]['f1-score'], 4),
                'support':   int(report[name]['support']),
            }
    print(f"\n  {test_name} Test: Acc={acc*100:.2f}%, F1w={f1_w*100:.2f}%, "
          f"F1m={f1_m*100:.2f}%  (n_train={n_train}, n_test={n_test})")
    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1_w),
        'f1_macro': float(f1_m),
        'per_class_f1': per_class,
        'confusion_matrix': cm,
        'n_train': n_train,
        'n_test': n_test,
    }


def save_cross_results(result, output_dir, experiment_tag, model_name,
                        test_name='wisdm'):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_tag}_" if experiment_tag else ""
    fname = f"{prefix}{model_name}_cross_{test_name}_{ts}.json"
    path = output_dir / fname
    if not isinstance(result.get('experiment_table_metadata'), dict):
        result['experiment_table_metadata'] = _build_experiment_table_metadata(result)
    with open(path, 'w') as f:
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


# =====================================================================
#  HART  (TensorFlow / Keras)
# =====================================================================
def cross_eval_hart(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    print(f"  [HART] TF {tf.__version__}, GPUs: {len(gpus)}")

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 256
    lr         = 5e-3
    n_channels = args.channels
    with_stairs = args.with_stairs
    num_classes = 5 if with_stairs else 3
    labels = LABELS_5CLS if with_stairs else LABELS_3CLS
    dt = args.device_type

    # --- Load HHAR training data (all subjects, no-bike) ---
    nobike_tag = '_nobike'
    hhar_suffix = {'all': f'HHAR{nobike_tag}', 'phone': f'HHAR_phone{nobike_tag}',
                   'watch': f'HHAR_watch{nobike_tag}'}
    hhar_dir = HART_DIR / "datasets" / "datasetStandardized" / hhar_suffix[dt]
    hhar_map_name = f"hart_subject_map{'_' + dt if dt != 'all' else ''}{nobike_tag}.json"
    hhar_map = BASE_DIR / hhar_map_name
    if not hhar_map.exists():
        print(f"ERROR: {hhar_map} not found. Run prepare_hhar_data.py --no-bike first.")
        sys.exit(1)
    with open(hhar_map) as f:
        sm = json.load(f)
    n_hhar = len(sm)
    X_hhar = np.vstack([hkl.load(str(hhar_dir / f"UserData{i}.hkl")) for i in range(n_hhar)])
    y_hhar = np.hstack([hkl.load(str(hhar_dir / f"UserLabel{i}.hkl")) for i in range(n_hhar)])

    if n_channels == 3:
        d_new = np.zeros_like(X_hhar); d_new[:, :, :3] = X_hhar[:, :, :3]; X_hhar = d_new

    # --- Load target dataset test data ---
    tgt = _get_target_cfg(args)
    other_dir = HART_DIR / "datasets" / "datasetStandardized" / tgt['hart'][dt]
    other_map_name = f"{tgt['map_prefix']}_subject_map{'_' + dt if dt != 'all' else ''}.json"
    other_map = BASE_DIR / other_map_name
    if not other_map.exists():
        print(f"ERROR: {other_map} not found. Run {tgt['prepare_script']} first.")
        sys.exit(1)
    with open(other_map) as f:
        wsm = json.load(f)
    n_wisdm = len(wsm)
    X_wisdm = np.vstack([hkl.load(str(other_dir / f"UserData{i}.hkl")) for i in range(n_wisdm)])
    y_wisdm = np.hstack([hkl.load(str(other_dir / f"UserLabel{i}.hkl")) for i in range(n_wisdm)])

    if n_channels == 3:
        d_new = np.zeros_like(X_wisdm); d_new[:, :, :3] = X_wisdm[:, :, :3]; X_wisdm = d_new

    # --- Class filtering ---
    # When --with-stairs: keep ALL 5 classes in BOTH train and test sets so that
    # evaluate_test computes real per-class recall for Upstairs/Downstairs.
    # When NOT --with-stairs: filter both to 3 classes (sit/stand/walk).
    if not with_stairs:
        X_hhar, y_hhar = filter_to_3class(X_hhar, y_hhar)
        X_wisdm, y_wisdm = filter_to_3class(X_wisdm, y_wisdm)
    # else: keep all 5 classes; evaluate_test will be called with LABELS_5CLS

    reverse = getattr(args, 'reverse', False)
    tgt_display = tgt['display_name']
    if reverse:
        X_hhar, y_hhar, X_wisdm, y_wisdm = X_wisdm, y_wisdm, X_hhar, y_hhar
    train_name = tgt_display if reverse else 'HHAR'
    test_name  = 'HHAR' if reverse else tgt_display
    test_tag   = 'hhar' if reverse else args.target_dataset

    # --- Optional stairs analysis on stair-only samples (kept for backwards compat) ---
    X_stairs_wisdm = None
    if with_stairs:
        stairs_mask_only = (y_wisdm == 3) | (y_wisdm == 4)
        if stairs_mask_only.sum() > 0:
            X_stairs_wisdm = X_wisdm[stairs_mask_only]

    # Apply data fraction to training set
    data_fraction = getattr(args, 'data_fraction', 1.0)
    if data_fraction < 1.0:
        X_hhar, y_hhar = subsample_training_data(X_hhar, y_hhar, data_fraction, seed=args.seed)

    print(f"\n  HART [{dt}] {n_channels}ch, classes={num_classes}")
    print(f"  {train_name}: {len(X_hhar)} samples | {test_name} test: {len(X_wisdm)} samples")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_hhar, y_hhar, test_size=0.1, random_state=args.seed, stratify=y_hhar)

    # Build and train
    tf.keras.backend.clear_session()
    tf.random.set_seed(args.seed); np.random.seed(args.seed)
    mdl = hart_model.HART((128, 6), num_classes)

    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {j: cw[j] for j in range(len(cw))}
    sw = np.array([cw_dict[int(y)] for y in y_train], dtype=np.float32)

    y_train_oh = tf.one_hot(y_train, num_classes)
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train_oh, sw))
                .shuffle(len(X_train), seed=args.seed)
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    exp_tag = args.experiment_tag or 'run'
    ckpt_path = str(BEST_MODELS_DIR / f"hart_{exp_tag}_cross.weights.h5")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

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
                loss = loss_fn(by, pred, sample_weight=bw)
            grads = tape.gradient(loss, mdl.trainable_variables)
            accum_grads = [a + g for a, g in zip(accum_grads, grads)]
            if (step + 1) % 1 == 0 or (step + 1) == num_steps:
                optimizer.apply_gradients(zip(accum_grads, mdl.trainable_variables))
                accum_grads = [tf.zeros_like(v) for v in mdl.trainable_variables]
            total_loss += float(loss)

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
            tqdm.write(f"    Saved best (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
        if loss_up:
            best_val_loss = v_loss
        if f1_up or loss_up:
            no_improve = 0
        else:
            no_improve += 1
        pbar.set_postfix_str(
            f"loss={total_loss/num_steps:.3f} Acc={v_acc*100:.1f}% "
            f"F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
        if no_improve >= patience:
            pbar.close(); print(f"    Early stop at epoch {epoch+1}"); break
    else:
        pbar.close()

    mdl.load_weights(ckpt_path)

    # --- Evaluate on test set ---
    print(f"\n  Evaluating on {test_name} ({len(X_wisdm)} samples) ...")
    y_pred = np.argmax(mdl.predict(X_wisdm, verbose=0), axis=-1)
    test_metrics = evaluate_test(
        y_wisdm, y_pred, labels, len(X_train), len(X_wisdm),
        test_name=test_name)

    # --- Stairs analysis ---
    stairs_result = None
    if with_stairs and X_stairs_wisdm is not None and len(X_stairs_wisdm) > 0:
        sp = np.argmax(mdl.predict(X_stairs_wisdm, verbose=0), axis=-1)
        dist = {LABELS_5CLS[i]: int((sp == i).sum()) for i in range(5)}
        total_s = len(sp)
        dist_pct = {k: round(v / total_s * 100, 1) for k, v in dist.items()}
        stairs_result = {'n_samples': total_s, 'prediction_counts': dist,
                         'prediction_pct': dist_pct}
        print(f"\n  Stairs analysis ({total_s} samples): {dist_pct}")

    td = args.target_dataset
    exp_label = f"EXP_cross_{td}_reverse" if reverse else f"EXP_cross_{td}"
    result = {
        'experiment': exp_label,
        'target_dataset': td,
        'direction': f'{train_name} -> {test_name}',
        'model': f'hart-{n_channels}ch-{dt}',
        'experiment_tag': exp_tag,
        'seed': args.seed,
        'device_type': dt, 'channels': n_channels, 'data_fraction': args.data_fraction,
        'with_stairs': with_stairs, 'num_classes': num_classes,
        'train_samples': len(X_train), 'val_samples': len(X_val),
        'best_val_f1_macro': float(best_val_f1),
        'test_metrics': test_metrics,
    }
    if stairs_result:
        result['stairs_analysis'] = stairs_result
    save_cross_results(result, args.output_dir, exp_tag,
                       f'hart-{n_channels}ch-{dt}', test_name=test_tag)
    return result


# =====================================================================
#  LIMU-BERT / LIMU-BERT-X  (PyTorch)
# =====================================================================
def cross_eval_limu(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [LIMU-BERT] PyTorch {torch.__version__}, device={device}")

    sys.path.insert(0, str(LIMU_DIR))
    from models import BERTClassifier, ClassifierGRU
    from config import PretrainModelConfig, ClassifierModelConfig

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 128
    lr         = 1e-3
    seq_len    = getattr(args, 'limu_seq_len', 120)
    with_stairs = args.with_stairs
    num_classes = 5 if with_stairs else 3
    labels = LABELS_5CLS if with_stairs else LABELS_3CLS
    dt = args.device_type

    # --- Load HHAR ---
    nobike_tag = '_nobike'
    hhar_dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
                    'all': f'hhar_all{nobike_tag}'}
    hhar_dir = LIMU_DIR / "dataset" / hhar_dir_map[dt]
    hhar_data  = np.load(str(hhar_dir / "data_20_120.npy")).astype(np.float32)
    hhar_lbl   = np.load(str(hhar_dir / "label_20_120.npy")).astype(np.float32)
    y_hhar     = hhar_lbl[:, 0, 2].astype(int)
    hhar_data[:, :, :3] /= 9.8

    # --- Load target dataset ---
    tgt = _get_target_cfg(args)
    wisdm_dir = LIMU_DIR / "dataset" / tgt['limu'][dt]
    if not (wisdm_dir / "data_20_120.npy").exists():
        print(f"ERROR: {wisdm_dir} not found. Run {tgt['prepare_script']} first.")
        sys.exit(1)
    wisdm_data = np.load(str(wisdm_dir / "data_20_120.npy")).astype(np.float32)
    wisdm_lbl  = np.load(str(wisdm_dir / "label_20_120.npy")).astype(np.float32)
    y_wisdm    = wisdm_lbl[:, 0, 2].astype(int)
    wisdm_data[:, :, :3] /= 9.8

    # --- Class filtering ---
    # See notes in cross_eval_hart: keep all 5 classes when with_stairs.
    if not with_stairs:
        hhar_data, y_hhar = filter_to_3class(hhar_data, y_hhar)
        wisdm_data, y_wisdm = filter_to_3class(wisdm_data, y_wisdm)

    reverse = getattr(args, 'reverse', False)
    tgt_display = tgt['display_name']
    if reverse:
        hhar_data, y_hhar, wisdm_data, y_wisdm = wisdm_data, y_wisdm, hhar_data, y_hhar
    train_name = tgt_display if reverse else 'HHAR'
    test_name  = 'HHAR' if reverse else tgt_display
    test_tag   = 'hhar' if reverse else args.target_dataset

    # --- Optional stairs analysis on stair-only samples (kept for backwards compat) ---
    X_stairs_wisdm = None
    if with_stairs:
        stairs_mask_only = (y_wisdm == 3) | (y_wisdm == 4)
        if stairs_mask_only.sum() > 0:
            X_stairs_wisdm = wisdm_data[stairs_mask_only]

    # Apply data fraction to training set
    data_fraction = getattr(args, 'data_fraction', 1.0)
    if data_fraction < 1.0:
        hhar_data, y_hhar = subsample_training_data(hhar_data, y_hhar, data_fraction, seed=args.seed)

    variant = "LIMU-BERT-X" if seq_len == 20 else "LIMU-BERT"
    print(f"\n  {variant} [{dt}] seq_len={seq_len}, classes={num_classes}")
    print(f"  {train_name}: {len(hhar_data)} | {test_name} test: {len(wisdm_data)}")

    # Model config
    bert_cfg = PretrainModelConfig(
        hidden=72, hidden_ff=144, feature_num=6,
        n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
    cls_cfg = ClassifierModelConfig(
        seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1],
        rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[],
        flat_num=0, num_attn=0, num_head=0, atten_hidden=0,
        num_linear=1, linear_io=[[10, 3]], activ=False, dropout=True)

    # Train/val split
    np.random.seed(args.seed)
    idx = np.arange(len(hhar_data)); np.random.shuffle(idx)
    vn = max(1, int(len(idx) * 0.1))
    X_val, y_val     = hhar_data[idx[:vn]], y_hhar[idx[:vn]]
    X_train, y_train = hhar_data[idx[vn:]], y_hhar[idx[vn:]]

    # DataLoaders (handle seq_len=20 cropping)
    class _CropDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, target_len, random_crop=True):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
            self.tl = target_len; self.rc = random_crop
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            x = self.X[i]
            if self.tl < x.size(0):
                s = (torch.randint(0, x.size(0) - self.tl, (1,)).item()
                     if self.rc else (x.size(0) - self.tl) // 2)
                x = x[s:s + self.tl]
            return x, self.y[i]

    data_seq_len = X_train.shape[1]
    def make_loader(X, y, shuffle=False, random_crop=False):
        if seq_len < data_seq_len:
            ds = _CropDataset(X, y, seq_len, random_crop=random_crop)
        else:
            ds = TensorDataset(torch.from_numpy(X).float(),
                               torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True, random_crop=True)
    val_loader   = make_loader(X_val, y_val, random_crop=False)

    # Build model
    torch.manual_seed(args.seed)
    classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=num_classes)
    model = BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False)
    pretrained_path = None
    if args.pretrained:
        pretrained_path = _resolve_limu_pretrained(args.pretrained)
        print(f"  Pretrained weights: {pretrained_path}")
        if not pretrained_path.exists():
            print(f"ERROR: pretrained weights not found: {pretrained_path}")
            sys.exit(1)
        state_dict = model.state_dict()
        pretrained = torch.load(str(pretrained_path), map_location=device)
        loaded = 0
        for k, v in pretrained.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
                loaded += 1
        model.load_state_dict(state_dict)
        print(f"    Loaded {loaded}/{len(pretrained)} weight tensors")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    exp_tag = args.experiment_tag or 'run'
    ckpt_path = BEST_MODELS_DIR / f"limubert_{exp_tag}_cross.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1, best_val_loss, no_improve, patience, best_state = -1.0, float('inf'), 0, 15, None
    pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                bar_format="{l_bar}{bar:20}{r_bar}")
    for epoch in pbar:
        model.train(); tl = 0
        optimizer.zero_grad()
        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx, True), by)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            tl += loss.item()

        model.eval(); vp, vt, vl = [], [], 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx_d, by_d = bx.to(device), by.to(device)
                logits = model(bx_d, False)
                vp.append(logits.argmax(1).cpu().numpy())
                vt.append(by.numpy())
                vl += criterion(logits, by_d).item()
        vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
        v_f1  = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
        v_acc = accuracy_score(vt_cat, vp_cat)
        v_loss = vl / max(len(val_loader), 1)
        f1_up = v_f1 > best_val_f1
        loss_up = v_loss < best_val_loss
        if f1_up:
            best_val_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _safe_torch_save(best_state, ckpt_path)
            tqdm.write(f"    Saved best (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
        if loss_up:
            best_val_loss = v_loss
        if f1_up or loss_up:
            no_improve = 0
        else:
            no_improve += 1
        pbar.set_postfix_str(
            f"loss={tl/len(train_loader):.3f} Acc={v_acc*100:.1f}% "
            f"F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
        if no_improve >= patience:
            pbar.close(); print(f"    Early stop at epoch {epoch+1}"); break
    else:
        pbar.close()

    if best_state:
        model.load_state_dict(best_state)

    # --- Evaluate on test set ---
    print(f"\n  Evaluating on {test_name} ({len(wisdm_data)} samples) ...")
    wisdm_test_loader = make_loader(wisdm_data, y_wisdm, random_crop=False)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for bx, by in wisdm_test_loader:
            tp.append(model(bx.to(device), False).argmax(1).cpu().numpy())
            tt.append(by.numpy())
    tp, tt = np.concatenate(tp), np.concatenate(tt)
    test_metrics = evaluate_test(tt, tp, labels, len(X_train), len(wisdm_data),
                                  test_name=test_name)

    # --- Stairs analysis ---
    stairs_result = None
    if with_stairs and X_stairs_wisdm is not None and len(X_stairs_wisdm) > 0:
        dummy_y = np.zeros(len(X_stairs_wisdm), dtype=np.int64)
        stairs_loader = make_loader(X_stairs_wisdm, dummy_y, random_crop=False)
        sp_all = []
        with torch.no_grad():
            for bx, _ in stairs_loader:
                sp_all.append(model(bx.to(device), False).argmax(1).cpu().numpy())
        sp = np.concatenate(sp_all)
        dist = {LABELS_5CLS[i]: int((sp == i).sum()) for i in range(5)}
        total_s = len(sp)
        dist_pct = {k: round(v / total_s * 100, 1) for k, v in dist.items()}
        stairs_result = {'n_samples': total_s, 'prediction_counts': dist,
                         'prediction_pct': dist_pct}
        print(f"\n  Stairs analysis ({total_s} samples): {dist_pct}")

    model_tag = f'limu-bert{"x" if seq_len == 20 else ""}-{dt}'
    td = args.target_dataset
    exp_label = f"EXP_cross_{td}_reverse" if reverse else f"EXP_cross_{td}"
    result = {
        'experiment': exp_label,
        'target_dataset': td,
        'direction': f'{train_name} -> {test_name}',
        'model': model_tag, 'experiment_tag': exp_tag, 'seed': args.seed,
        'device_type': dt, 'channels': args.channels, 'seq_len': seq_len, 'data_fraction': args.data_fraction,
        'pretrained': str(pretrained_path) if pretrained_path else None,
        'with_stairs': with_stairs, 'num_classes': num_classes,
        'train_samples': len(X_train), 'val_samples': len(X_val),
        'best_val_f1_macro': float(best_val_f1),
        'test_metrics': test_metrics,
    }
    if stairs_result:
        result['stairs_analysis'] = stairs_result
    save_cross_results(result, args.output_dir, exp_tag, model_tag,
                       test_name=test_tag)
    return result

    del model; torch.cuda.empty_cache()


# =====================================================================
#  ssl-wearables  (PyTorch)
# =====================================================================
def cross_eval_ssl(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [ssl-wearables] PyTorch {torch.__version__}, device={device}")

    sys.path.insert(0, str(SSL_DIR))
    from sslearning.models.accNet import Resnet

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 64
    lr         = 1e-4
    n_channels = args.channels
    input_size = 300
    with_stairs = args.with_stairs
    num_classes = 5 if with_stairs else 3
    labels = LABELS_5CLS if with_stairs else LABELS_3CLS
    dt = args.device_type

    # --- Load HHAR ---
    nobike_tag = '_nobike'
    hhar_dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
                    'all': f'hhar_all{nobike_tag}'}
    hhar_dir = SSL_DIR / "data" / "downstream" / hhar_dir_map[dt]
    x_file = hhar_dir / ("X6.npy" if n_channels == 6 else "X.npy")
    if not x_file.exists():
        print(f"ERROR: {x_file} not found. Run prepare_hhar_data.py first."); sys.exit(1)
    X_hhar = np.load(str(x_file)).astype(np.float32)
    y_hhar = np.load(str(hhar_dir / "Y.npy")).astype(np.int64)
    if X_hhar.shape[1] != input_size:
        from scipy.interpolate import interp1d as _interp
        to, tn = np.linspace(0,1,X_hhar.shape[1]), np.linspace(0,1,input_size)
        X_hhar = _interp(to, X_hhar, kind='linear', axis=1, assume_sorted=True)(tn).astype(np.float32)
    X_hhar = np.transpose(X_hhar, (0, 2, 1))  # (N, C, T)

    # --- Load target dataset ---
    tgt = _get_target_cfg(args)
    wisdm_dir = SSL_DIR / "data" / "downstream" / tgt['ssl'][dt]
    wx_file = wisdm_dir / ("X6.npy" if n_channels == 6 else "X.npy")
    if not wx_file.exists():
        print(f"ERROR: {wx_file} not found. Run {tgt['prepare_script']} first."); sys.exit(1)
    X_wisdm = np.load(str(wx_file)).astype(np.float32)
    y_wisdm = np.load(str(wisdm_dir / "Y.npy")).astype(np.int64)
    if X_wisdm.shape[1] != input_size:
        from scipy.interpolate import interp1d as _interp
        to, tn = np.linspace(0,1,X_wisdm.shape[1]), np.linspace(0,1,input_size)
        X_wisdm = _interp(to, X_wisdm, kind='linear', axis=1, assume_sorted=True)(tn).astype(np.float32)
    X_wisdm = np.transpose(X_wisdm, (0, 2, 1))  # (N, C, T)

    # --- Class filtering ---
    # See notes in cross_eval_hart: keep all 5 classes when with_stairs.
    if not with_stairs:
        X_hhar, y_hhar = filter_to_3class(X_hhar, y_hhar)
        X_wisdm, y_wisdm = filter_to_3class(X_wisdm, y_wisdm)

    reverse = getattr(args, 'reverse', False)
    tgt_display = tgt['display_name']
    if reverse:
        X_hhar, y_hhar, X_wisdm, y_wisdm = X_wisdm, y_wisdm, X_hhar, y_hhar

    # --- Optional stairs analysis on stair-only samples (kept for backwards compat) ---
    X_stairs_wisdm = None
    if with_stairs:
        stairs_mask_only = (y_wisdm == 3) | (y_wisdm == 4)
        if stairs_mask_only.sum() > 0:
            X_stairs_wisdm = X_wisdm[stairs_mask_only]
    train_name = tgt_display if reverse else 'HHAR'
    test_name  = 'HHAR' if reverse else tgt_display
    test_tag   = 'hhar' if reverse else args.target_dataset

    # Apply data fraction to training set
    data_fraction = getattr(args, 'data_fraction', 1.0)
    if data_fraction < 1.0:
        X_hhar, y_hhar = subsample_training_data(X_hhar, y_hhar, data_fraction, seed=args.seed)

    print(f"\n  ssl-wearables [{dt}] {n_channels}ch, classes={num_classes}")
    print(f"  {train_name}: {len(X_hhar)} | {test_name} test: {len(X_wisdm)}")

    # Train/val split
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    idx = np.arange(len(X_hhar)); np.random.shuffle(idx)
    vn = max(1, int(len(idx) * 0.1))
    X_val, y_val     = X_hhar[idx[:vn]], y_hhar[idx[:vn]]
    X_train, y_train = X_hhar[idx[vn:]], y_hhar[idx[vn:]]

    def make_loader(X, y, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                        torch.from_numpy(y).long()),
                          batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val, y_val)

    model = Resnet(output_size=num_classes, n_channels=n_channels,
                   is_eva=True, resnet_version=1, epoch_len=10)
    pretrained_path = None
    if args.pretrained:
        pretrained_path = _resolve_ssl_pretrained(args.pretrained)
        print(f"  Pretrained: {pretrained_path}")
        if not pretrained_path.exists():
            print(f"ERROR: pretrained weights not found: {pretrained_path}")
            sys.exit(1)
        pre = torch.load(str(pretrained_path), map_location=device)
        pre2 = copy.deepcopy(pre)
        head = next(iter(pre2)).split(".")[0]
        if head == "module":
            pre2 = {k.partition("module.")[2]: v for k, v in pre2.items()}
        md = model.state_dict()
        filt, skip = {}, []
        for k, v in pre2.items():
            if k not in md:
                continue
            if k.split(".")[0] == "classifier":
                continue
            if md[k].shape != v.shape:
                skip.append(k)
                continue
            filt[k] = v
        md.update(filt)
        model.load_state_dict(md)
        print(f"    Loaded {len(filt)} tensors" + (f", skipped {len(skip)}" if skip else ""))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    counter = collections.Counter(y_train.tolist())
    wts = [1.0 / (counter.get(i, 1) / len(y_train)) for i in range(num_classes)]
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(wts).to(device))

    exp_tag = args.experiment_tag or 'run'
    ckpt_path = BEST_MODELS_DIR / f"ssl_{exp_tag}_cross.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1, best_val_loss, no_improve, patience, best_state = -1.0, float('inf'), 0, 10, None
    pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                bar_format="{l_bar}{bar:20}{r_bar}")
    for epoch in pbar:
        model.train(); tl = 0
        optimizer.zero_grad()
        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            tl += loss.item()

        model.eval(); vp, vt, vl = [], [], 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx_d, by_d = bx.to(device), by.to(device)
                logits = model(bx_d)
                vp.append(logits.argmax(1).cpu().numpy())
                vt.append(by.numpy())
                vl += criterion(logits, by_d).item()
        vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
        v_f1  = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
        v_acc = accuracy_score(vt_cat, vp_cat)
        v_loss = vl / max(len(val_loader), 1)
        f1_up = v_f1 > best_val_f1
        loss_up = v_loss < best_val_loss
        if f1_up:
            best_val_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())
            _safe_torch_save(best_state, ckpt_path)
            tqdm.write(f"    Saved best (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
        if loss_up:
            best_val_loss = v_loss
        if f1_up or loss_up:
            no_improve = 0
        else:
            no_improve += 1
        pbar.set_postfix_str(
            f"loss={tl/len(train_loader):.3f} Acc={v_acc*100:.1f}% "
            f"F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
        if no_improve >= patience:
            pbar.close(); print(f"    Early stop at epoch {epoch+1}"); break
    else:
        pbar.close()
    if best_state:
        model.load_state_dict(best_state)

    # --- Evaluate on test set ---
    print(f"\n  Evaluating on {test_name} ({len(X_wisdm)} samples) ...")
    test_loader = make_loader(X_wisdm, y_wisdm)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            tp.append(model(bx.to(device)).argmax(1).cpu().numpy())
            tt.append(by.numpy())
    tp, tt = np.concatenate(tp), np.concatenate(tt)
    test_metrics = evaluate_test(tt, tp, labels, len(X_train), len(X_wisdm),
                                  test_name=test_name)

    # --- Stairs analysis ---
    stairs_result = None
    if with_stairs and X_stairs_wisdm is not None and len(X_stairs_wisdm) > 0:
        dummy_y = np.zeros(len(X_stairs_wisdm), dtype=np.int64)
        sl = make_loader(X_stairs_wisdm, dummy_y)
        sp_all = []
        with torch.no_grad():
            for bx, _ in sl:
                sp_all.append(model(bx.to(device)).argmax(1).cpu().numpy())
        sp = np.concatenate(sp_all)
        dist = {LABELS_5CLS[i]: int((sp == i).sum()) for i in range(5)}
        total_s = len(sp)
        dist_pct = {k: round(v / total_s * 100, 1) for k, v in dist.items()}
        stairs_result = {'n_samples': total_s, 'prediction_counts': dist,
                         'prediction_pct': dist_pct}
        print(f"\n  Stairs analysis ({total_s} samples): {dist_pct}")

    tag_name = f'ssl-wearables-{n_channels}ch-{dt}'
    td = args.target_dataset
    exp_label = f"EXP_cross_{td}_reverse" if reverse else f"EXP_cross_{td}"
    result = {
        'experiment': exp_label,
        'target_dataset': td,
        'direction': f'{train_name} -> {test_name}',
        'model': tag_name, 'experiment_tag': exp_tag, 'seed': args.seed,
        'device_type': dt, 'channels': n_channels, 'data_fraction': args.data_fraction,
        'pretrained': str(pretrained_path) if pretrained_path else None,
        'with_stairs': with_stairs, 'num_classes': num_classes,
        'train_samples': len(X_train), 'val_samples': len(X_val),
        'best_val_f1_macro': float(best_val_f1),
        'test_metrics': test_metrics,
    }
    if stairs_result:
        result['stairs_analysis'] = stairs_result
    save_cross_results(result, args.output_dir, exp_tag, tag_name,
                       test_name=test_tag)
    return result

    del model; torch.cuda.empty_cache()


# =====================================================================
#  ResNet-Baseline  (PyTorch)
# =====================================================================
def cross_eval_resnet(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, str(RESNET_BASE_DIR))
    from resnet1d_baseline import ResNet1DBaseline

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [ResNet-Baseline] PyTorch {torch.__version__}, device={device}")

    epochs     = args.epochs or 200
    batch_size = args.batch_size or 64
    lr         = 1e-4
    n_channels = args.channels
    input_size = 300
    with_stairs = args.with_stairs
    num_classes = 5 if with_stairs else 3
    labels = LABELS_5CLS if with_stairs else LABELS_3CLS
    dt = args.device_type

    # --- Load HHAR (same dir as ssl-wearables) ---
    nobike_tag = '_nobike'
    hhar_dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
                    'all': f'hhar_all{nobike_tag}'}
    hhar_dir = SSL_DIR / "data" / "downstream" / hhar_dir_map[dt]
    x_file = hhar_dir / ("X6.npy" if n_channels == 6 else "X.npy")
    if not x_file.exists():
        print(f"ERROR: {x_file} not found."); sys.exit(1)
    X_hhar = np.load(str(x_file)).astype(np.float32)
    y_hhar = np.load(str(hhar_dir / "Y.npy")).astype(np.int64)
    if X_hhar.shape[1] != input_size:
        from scipy.interpolate import interp1d as _interp
        to, tn = np.linspace(0,1,X_hhar.shape[1]), np.linspace(0,1,input_size)
        X_hhar = _interp(to, X_hhar, kind='linear', axis=1, assume_sorted=True)(tn).astype(np.float32)
    X_hhar = np.transpose(X_hhar, (0, 2, 1))

    # --- Load target dataset ---
    tgt = _get_target_cfg(args)
    wisdm_dir = SSL_DIR / "data" / "downstream" / tgt['ssl'][dt]
    wx_file = wisdm_dir / ("X6.npy" if n_channels == 6 else "X.npy")
    if not wx_file.exists():
        print(f"ERROR: {wx_file} not found. Run {tgt['prepare_script']} first."); sys.exit(1)
    X_wisdm = np.load(str(wx_file)).astype(np.float32)
    y_wisdm = np.load(str(wisdm_dir / "Y.npy")).astype(np.int64)
    if X_wisdm.shape[1] != input_size:
        from scipy.interpolate import interp1d as _interp
        to, tn = np.linspace(0,1,X_wisdm.shape[1]), np.linspace(0,1,input_size)
        X_wisdm = _interp(to, X_wisdm, kind='linear', axis=1, assume_sorted=True)(tn).astype(np.float32)
    X_wisdm = np.transpose(X_wisdm, (0, 2, 1))

    # --- Class filtering ---
    # See notes in cross_eval_hart: keep all 5 classes when with_stairs.
    if not with_stairs:
        X_hhar, y_hhar = filter_to_3class(X_hhar, y_hhar)
        X_wisdm, y_wisdm = filter_to_3class(X_wisdm, y_wisdm)

    reverse = getattr(args, 'reverse', False)
    tgt_display = tgt['display_name']
    if reverse:
        X_hhar, y_hhar, X_wisdm, y_wisdm = X_wisdm, y_wisdm, X_hhar, y_hhar

    # --- Optional stairs analysis on stair-only samples (kept for backwards compat) ---
    X_stairs_wisdm = None
    if with_stairs:
        stairs_mask_only = (y_wisdm == 3) | (y_wisdm == 4)
        if stairs_mask_only.sum() > 0:
            X_stairs_wisdm = X_wisdm[stairs_mask_only]
    train_name = tgt_display if reverse else 'HHAR'
    test_name  = 'HHAR' if reverse else tgt_display
    test_tag   = 'hhar' if reverse else args.target_dataset

    # Apply data fraction to training set
    data_fraction = getattr(args, 'data_fraction', 1.0)
    if data_fraction < 1.0:
        X_hhar, y_hhar = subsample_training_data(X_hhar, y_hhar, data_fraction, seed=args.seed)

    print(f"\n  ResNet-Baseline [{dt}] {n_channels}ch, classes={num_classes}")
    print(f"  {train_name}: {len(X_hhar)} | {test_name} test: {len(X_wisdm)}")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    idx = np.arange(len(X_hhar)); np.random.shuffle(idx)
    vn = max(1, int(len(idx) * 0.1))
    X_val, y_val     = X_hhar[idx[:vn]], y_hhar[idx[:vn]]
    X_train, y_train = X_hhar[idx[vn:]], y_hhar[idx[vn:]]

    def make_loader(X, y, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                        torch.from_numpy(y).long()),
                          batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val, y_val)

    model = ResNet1DBaseline(n_channels=n_channels,
                             num_classes=num_classes, kernel_size=5)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    counter = collections.Counter(y_train.tolist())
    wts = [1.0 / (counter.get(i, 1) / len(y_train)) for i in range(num_classes)]
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(wts).to(device))

    exp_tag = args.experiment_tag or 'run'
    ckpt_path = BEST_MODELS_DIR / f"resbase_{exp_tag}_cross.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1, best_val_loss, no_improve, patience, best_state = -1.0, float('inf'), 0, 10, None
    pbar = tqdm(range(epochs), desc="    Training", unit="ep",
                bar_format="{l_bar}{bar:20}{r_bar}")
    for epoch in pbar:
        model.train(); tl = 0
        optimizer.zero_grad()
        for step, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            tl += loss.item()

        model.eval(); vp, vt, vl = [], [], 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx_d, by_d = bx.to(device), by.to(device)
                logits = model(bx_d)
                vp.append(logits.argmax(1).cpu().numpy())
                vt.append(by.numpy())
                vl += criterion(logits, by_d).item()
        vp_cat, vt_cat = np.concatenate(vp), np.concatenate(vt)
        v_f1  = f1_score(vt_cat, vp_cat, average='macro', zero_division=0)
        v_acc = accuracy_score(vt_cat, vp_cat)
        v_loss = vl / max(len(val_loader), 1)
        f1_up = v_f1 > best_val_f1
        loss_up = v_loss < best_val_loss
        if f1_up:
            best_val_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())
            _safe_torch_save(best_state, ckpt_path)
            tqdm.write(f"    Saved best (F1m={v_f1*100:.1f}%) -> {ckpt_path}")
        if loss_up:
            best_val_loss = v_loss
        if f1_up or loss_up:
            no_improve = 0
        else:
            no_improve += 1
        pbar.set_postfix_str(
            f"loss={tl/len(train_loader):.3f} Acc={v_acc*100:.1f}% "
            f"F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
        if no_improve >= patience:
            pbar.close(); print(f"    Early stop at epoch {epoch+1}"); break
    else:
        pbar.close()
    if best_state:
        model.load_state_dict(best_state)

    # --- Evaluate on test set ---
    print(f"\n  Evaluating on {test_name} ({len(X_wisdm)} samples) ...")
    test_loader = make_loader(X_wisdm, y_wisdm)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            tp.append(model(bx.to(device)).argmax(1).cpu().numpy())
            tt.append(by.numpy())
    tp, tt = np.concatenate(tp), np.concatenate(tt)
    test_metrics = evaluate_test(tt, tp, labels, len(X_train), len(X_wisdm),
                                  test_name=test_name)

    # --- Stairs analysis ---
    stairs_result = None
    if with_stairs and X_stairs_wisdm is not None and len(X_stairs_wisdm) > 0:
        dummy_y = np.zeros(len(X_stairs_wisdm), dtype=np.int64)
        sl = make_loader(X_stairs_wisdm, dummy_y)
        sp_all = []
        with torch.no_grad():
            for bx, _ in sl:
                sp_all.append(model(bx.to(device)).argmax(1).cpu().numpy())
        sp = np.concatenate(sp_all)
        dist = {LABELS_5CLS[i]: int((sp == i).sum()) for i in range(5)}
        total_s = len(sp)
        dist_pct = {k: round(v / total_s * 100, 1) for k, v in dist.items()}
        stairs_result = {'n_samples': total_s, 'prediction_counts': dist,
                         'prediction_pct': dist_pct}
        print(f"\n  Stairs analysis ({total_s} samples): {dist_pct}")

    tag_name = f'resnet-baseline-{n_channels}ch-{dt}'
    td = args.target_dataset
    exp_label = f"EXP_cross_{td}_reverse" if reverse else f"EXP_cross_{td}"
    result = {
        'experiment': exp_label,
        'target_dataset': td,
        'direction': f'{train_name} -> {test_name}',
        'model': tag_name, 'experiment_tag': exp_tag, 'seed': args.seed,
        'device_type': dt, 'channels': n_channels, 'data_fraction': args.data_fraction,
        'with_stairs': with_stairs, 'num_classes': num_classes,
        'train_samples': len(X_train), 'val_samples': len(X_val),
        'best_val_f1_macro': float(best_val_f1),
        'test_metrics': test_metrics,
    }
    if stairs_result:
        result['stairs_analysis'] = stairs_result
    save_cross_results(result, args.output_dir, exp_tag, tag_name,
                       test_name=test_tag)
    return result

    del model; torch.cuda.empty_cache()


# =====================================================================
#  Main CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Cross-dataset evaluation -- HHAR <-> target dataset')
    parser.add_argument('--model', required=True,
                        choices=['hart', 'limu-bert', 'ssl-wearables', 'resnet-baseline'])
    parser.add_argument('--device-type', type=str, default='all',
                        choices=['phone', 'watch', 'all'],
                        help='Device type (matched across both datasets)')
    parser.add_argument('--channels', type=int, default=6, choices=[3, 6])
    parser.add_argument('--limu-seq-len', type=int, default=120, choices=[20, 120],
                        help='LIMU-BERT seq length: 120=LIMU-BERT, 20=LIMU-BERT-X')
    parser.add_argument('--target-dataset', type=str, default='pamap2_sbhar',
                        choices=VALID_TARGET_DATASETS,
                        help='Target (non-HHAR) dataset.  '
                             'pamap2_sbhar = aligned 8+8 (default), '
                             'pamap2_sbhar_full = 8+30, wisdm = WISDM')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse direction: train <target>, test HHAR')
    parser.add_argument('--with-stairs', action='store_true',
                        help='5-class + supplementary stairs analysis')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split and initialization')
    parser.add_argument('--repeat-seeds', type=int, default=1,
                        help='Run the same configuration across consecutive seeds starting from --seed')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='Fraction of training data (0.0-1.0, default=1.0)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (supported by limu-bert and ssl-wearables)')
    parser.add_argument('--experiment-tag', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()
    args.output_dir = str(_resolve_project_path(args.output_dir))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tgt_cfg = _get_target_cfg(args)
    tgt_name = tgt_cfg['display_name']
    mode = "5-class + stairs" if args.with_stairs else "3-class (sit/stand/walk)"
    direction = f"{tgt_name} -> HHAR" if args.reverse else f"HHAR -> {tgt_name}"
    print(f"\n{'='*60}")
    print(f"  Cross-Dataset  {direction}")
    print(f"  Target DS  : {args.target_dataset} ({tgt_name})")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {args.device_type}")
    print(f"  Channels   : {args.channels}")
    print(f"  Mode       : {mode}")
    print(f"  Seed start : {args.seed}")
    print(f"  Repeats    : {args.repeat_seeds}")
    print(f"  Pretrained : {args.pretrained or 'None (scratch)'}")
    if args.model == 'limu-bert':
        print(f"  Seq len    : {args.limu_seq_len}")
    print(f"{'='*60}\n")

    dispatch = {
        'hart':             cross_eval_hart,
        'limu-bert':        cross_eval_limu,
        'ssl-wearables':    cross_eval_ssl,
        'resnet-baseline':  cross_eval_resnet,
    }
    run_results = []
    base_seed = int(args.seed)
    for seed_offset in range(max(1, args.repeat_seeds)):
        args.seed = base_seed + seed_offset
        print(f"\n  >>> Running seed {args.seed}")
        result = dispatch[args.model](args)
        print_cross_summary(result)
        run_results.append(result)

    if len(run_results) > 1:
        summary = aggregate_cross_runs(run_results)
        print_repeated_cross_summary(summary)
        final = copy.deepcopy(run_results[-1])
        final['seed'] = None
        final['repeated_seed_summary'] = summary
        pcf1 = summary.get('per_class_f1_mean_std', {})
        pcrec = summary.get('per_class_recall_mean_std', {})
        merged_pc = {}
        for cls in set(list(pcf1.keys()) + list(pcrec.keys())):
            merged_pc[cls] = {
                'f1': pcf1[cls]['mean'] if cls in pcf1 else None,
                'recall': pcrec[cls]['mean'] if cls in pcrec else None,
            }
        final['test_metrics'] = {
            'accuracy': summary['test_metrics_mean_std']['accuracy']['mean'],
            'f1_weighted': summary['test_metrics_mean_std']['f1_weighted']['mean'],
            'f1_macro': summary['test_metrics_mean_std']['f1_macro']['mean'],
            'per_class_f1': merged_pc or final['test_metrics'].get('per_class_f1', {}),
            'confusion_matrix': final['test_metrics'].get('confusion_matrix', []),
            'n_train': final['test_metrics'].get('n_train'),
            'n_test': final['test_metrics'].get('n_test'),
        }
        final['std_test_metrics'] = {
            name: stats['std'] for name, stats in summary['test_metrics_mean_std'].items()
        }
        test_ds_tag = 'hhar' if args.reverse else args.target_dataset
        save_cross_results(final, args.output_dir, args.experiment_tag, f"{final['model']}_repeat",
                           test_name=test_ds_tag)


if __name__ == '__main__':
    main()
