#!/usr/bin/env python
"""
=============================================================================
Cross-Dataset Fine-Tuning -- EXP 5.2
=============================================================================

Two-stage experiment with TWO separate fine-tuning protocols:

  Protocol A  --finetune-subjects N
    "Leave-N-in": pick N target subjects, use ALL their data for fine-tuning,
    test on the remaining subjects.  Varies subject coverage.

  Protocol B  --per-subject-fraction F
    "Data-fraction": use ALL target subjects, take F% of each subject's
    data for fine-tuning, test on the remaining (1-F)% per subject.
    Varies data density.

Stage 1 (HHAR supervised training) is cached per (model, device, num_classes)
so it is trained ONCE and reused across all protocols, targets, and seeds
that share the same class count.  Use --with-stairs for 5-class targets.

Usage:
  # Leave-in
  python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 \\
      --device-type phone --target-dataset wisdm --finetune-subjects 1 \\
      --experiment-tag exp52_limubertx_phone_6ch_wisdm_leavein1 --seed 42 --repeat-seeds 3

  # Data-fraction
  python eval_cross_finetune.py --model ssl-wearables --device-type watch \\
      --channels 6 --target-dataset wisdm --per-subject-fraction 0.25 \\
      --experiment-tag exp52_ssl_watch_6ch_wisdm_frac25 --seed 42 --repeat-seeds 3
"""

import os, sys, json, time, argparse, copy, collections, re
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths =====================
BASE_DIR    = Path(__file__).resolve().parent
LIMU_DIR    = (BASE_DIR / "code"
               / "LIMU-BERT_Experience-main" / "LIMU-BERT_Experience-main")
SSL_DIR     = (BASE_DIR / "code"
               / "ssl-wearables-main" / "ssl-wearables-main")
RESULTS_DIR     = BASE_DIR / "loso_results"
BEST_MODELS_DIR = RESULTS_DIR / "best_models"
STAGE1_DIR      = BEST_MODELS_DIR / "stage1_cache"

LABELS_3CLS = ['Sitting', 'Standing', 'Walking']
LABELS_5CLS = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']

# ===================== Target Dataset Config =====================
_TARGET_DS_CFG = {
    'wisdm': {
        'limu':  {'phone': 'wisdm',         'watch': 'wisdm_watch',          'all': 'wisdm_all'},
        'ssl':   {'phone': 'wisdm',         'watch': 'wisdm_watch',          'all': 'wisdm_all'},
        'display_name':   'WISDM',
        'prepare_script': 'prepare_wisdm_data.py',
    },
    'pamap2_sbhar': {
        'limu':  {'phone': 'pamap2_sbhar_aligned',      'watch': 'pamap2_sbhar_aligned_watch',       'all': 'pamap2_sbhar_aligned_all'},
        'ssl':   {'phone': 'pamap2_sbhar_aligned',      'watch': 'pamap2_sbhar_aligned_watch',       'all': 'pamap2_sbhar_aligned_all'},
        'display_name':   'PAMAP2+SBHAR (8+8)',
        'prepare_script': 'prepare_pamap2_sbhar_data.py --aligned',
    },
    'pamap2_sbhar_full': {
        'limu':  {'phone': 'pamap2_sbhar',               'watch': 'pamap2_sbhar_watch',               'all': 'pamap2_sbhar_all'},
        'ssl':   {'phone': 'pamap2_sbhar',               'watch': 'pamap2_sbhar_watch',               'all': 'pamap2_sbhar_all'},
        'display_name':   'PAMAP2+SBHAR (8+30)',
        'prepare_script': 'prepare_pamap2_sbhar_data.py',
    },
}
VALID_TARGET_DATASETS = list(_TARGET_DS_CFG.keys())


def _get_target_cfg(args):
    td = getattr(args, 'target_dataset', 'wisdm')
    if td not in _TARGET_DS_CFG:
        print(f"ERROR: unknown --target-dataset '{td}'. Choices: {VALID_TARGET_DATASETS}")
        sys.exit(1)
    return _TARGET_DS_CFG[td]


def _resolve_project_path(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)


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


def filter_to_3class(X, y):
    mask = y <= 2
    return X[mask], y[mask]


def filter_to_3class_with_ids(X, y, subj_ids):
    mask = y <= 2
    return X[mask], y[mask], subj_ids[mask]


def filter_to_5class(X, y):
    mask = y <= 4
    return X[mask], y[mask]


def filter_to_5class_with_ids(X, y, subj_ids):
    mask = y <= 4
    return X[mask], y[mask], subj_ids[mask]


# ===================== Protocol Split Functions =====================
def split_leave_in(X, y, subject_ids, n_subjects, seed):
    """Protocol A: pick N subjects -> fine-tune (100%), rest -> test."""
    all_subjects = sorted(np.unique(subject_ids).tolist())
    rng = np.random.RandomState(seed)
    ft_subjects = sorted(rng.choice(all_subjects, n_subjects, replace=False).tolist())
    test_subjects = sorted(set(all_subjects) - set(ft_subjects))
    ft_mask = np.isin(subject_ids, ft_subjects)
    print(f"    Leave-{n_subjects}-in: {ft_mask.sum()} fine-tune / "
          f"{(~ft_mask).sum()} test samples")
    print(f"    FT subjects: {ft_subjects}")
    print(f"    Test subjects ({len(test_subjects)}): {test_subjects}")
    return (X[ft_mask], y[ft_mask], X[~ft_mask], y[~ft_mask],
            ft_subjects, test_subjects)


def split_data_fraction(X, y, subject_ids, fraction, seed):
    """Protocol B: from each subject, fraction% -> fine-tune, rest -> test."""
    rng = np.random.RandomState(seed)
    ft_idx, test_idx = [], []
    for subj in np.unique(subject_ids):
        idx = np.where(subject_ids == subj)[0]
        n_ft = max(1, int(len(idx) * fraction))
        perm = rng.permutation(len(idx))
        ft_idx.extend(idx[perm[:n_ft]].tolist())
        if n_ft < len(idx):
            test_idx.extend(idx[perm[n_ft:]].tolist())
    ft_idx = np.sort(np.array(ft_idx))
    test_idx = np.sort(np.array(test_idx)) if test_idx else np.array([], dtype=int)
    n_subj = len(np.unique(subject_ids))
    print(f"    Data-fraction {fraction*100:.0f}%: {len(ft_idx)} fine-tune / "
          f"{len(test_idx)} test ({n_subj} subjects, per-subject stratified)")
    return X[ft_idx], y[ft_idx], X[test_idx], y[test_idx]


def _get_stage1_path(model_tag, device_type, num_classes):
    """Deterministic cache path for Stage 1 — independent of target/seed/protocol."""
    return STAGE1_DIR / f"stage1_{model_tag}_{device_type}_{num_classes}cls.pt"


# ===================== Evaluation & Output =====================
def evaluate_test(y_true, y_pred, label_names, n_train, n_test, test_name='target'):
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
          f"F1m={f1_m*100:.2f}%  (n_finetune={n_train}, n_test={n_test})")
    return {
        'accuracy': float(acc), 'f1_weighted': float(f1_w), 'f1_macro': float(f1_m),
        'per_class_f1': per_class, 'confusion_matrix': cm,
        'n_train': n_train, 'n_test': n_test,
    }


def _build_experiment_table_metadata(result):
    model_text = str(result.get('model', '')).lower()
    if "limu-bert" in model_text:
        arch = "LIMU-BERT-X" if "20" in model_text or "x" in model_text else "LIMU-BERT"
    elif "ssl-wearables" in model_text:
        arch = "SSL-Wearables"
    else:
        arch = str(result.get('model', ''))
    direction = str(result.get('direction', ''))
    train_ds, test_ds = "", ""
    if "->" in direction:
        parts = direction.split("->")
        train_ds = parts[0].strip()
        test_ds = parts[-1].strip()
    protocol = result.get('protocol', '')
    if protocol == 'leave-in':
        proto_txt = f"Leave-{result.get('finetune_subjects', '?')}-in"
    else:
        psf = result.get('per_subject_fraction', 0)
        proto_txt = f"Frac {psf*100:.0f}%/subj"
    n_tests = result.get('repeated_seed_summary', {}).get('runs', 1)
    return {
        "Architecture Type": arch,
        "Device Type": str(result.get('device_type', '')).capitalize(),
        "Training Dataset": train_ds,
        "Testing Dataset": test_ds,
        "Imported Pretrained Weights": "No",
        "3ch vs 6ch data": f"{result.get('channels')}ch" if result.get('channels') else "",
        "Protocol": proto_txt,
        "Data Fraction": proto_txt,
        "Testing Method": "Cross-Finetune",
        "# of Tests": int(n_tests) if n_tests else 1,
        "# Run": _extract_run_number(result.get('experiment_tag')) or "",
    }


def save_results(result, output_dir, experiment_tag, model_name, test_name='target'):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_tag}_" if experiment_tag else ""
    fname = f"{prefix}{model_name}_finetune_{test_name}_{ts}.json"
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


def aggregate_runs(results):
    summary = {'runs': len(results), 'seeds': [r.get('seed') for r in results]}
    metric_names = ['accuracy', 'f1_weighted', 'f1_macro']
    summary['test_metrics_mean_std'] = {
        name: summarize_metric_stats([r.get('test_metrics', {}).get(name, 0.0) for r in results])
        for name in metric_names
    }
    labels = LABELS_5CLS if any(r.get('num_classes') == 5 for r in results) else LABELS_3CLS
    per_class_f1, per_class_recall = {}, {}
    for cls in labels:
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


def print_summary(result):
    tm = result.get('test_metrics', {})
    print("\n" + "=" * 60)
    print(f"  Cross-Finetune Summary  [{result.get('protocol', '')}]")
    print("=" * 60)
    print(f"  Accuracy   : {tm.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  F1 weighted: {tm.get('f1_weighted', 0.0) * 100:.2f}%")
    print(f"  F1 macro   : {tm.get('f1_macro', 0.0) * 100:.2f}%")
    print(f"  n_finetune/test: {tm.get('n_train', 0)} / {tm.get('n_test', 0)}")
    print("=" * 60)


def print_repeated_summary(summary):
    print("\n" + "=" * 60)
    print("  Repeated-run Summary")
    print("=" * 60)
    for name, stats in summary.get('test_metrics_mean_std', {}).items():
        print(f"  {name:<12} mean={stats['mean']*100:.2f}%  std={stats['std']*100:.2f}%  n={stats['n']}")
    print("=" * 60)


# =====================================================================
#  Shared training loop
# =====================================================================
def _train_loop(model, train_loader, val_loader, optimizer, criterion, device,
                epochs, patience, ckpt_path, desc="Training"):
    import torch
    best_val_f1, best_val_loss, no_improve, best_state = -1.0, float('inf'), 0, None
    pbar = tqdm(range(epochs), desc=f"    {desc}", unit="ep",
                bar_format="{l_bar}{bar:20}{r_bar}")
    is_limu = hasattr(model, 'bert')
    for epoch in pbar:
        model.train(); tl = 0
        optimizer.zero_grad()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx, True) if is_limu else model(bx)
            loss = criterion(logits, by)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            tl += loss.item()

        model.eval(); vp, vt, vl = [], [], 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx_d, by_d = bx.to(device), by.to(device)
                logits = model(bx_d, False) if is_limu else model(bx_d)
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
            tqdm.write(f"    Saved best (F1m={v_f1*100:.1f}%) -> {ckpt_path.name}")
        if loss_up:
            best_val_loss = v_loss
        if f1_up or loss_up:
            no_improve = 0
        else:
            no_improve += 1
        pbar.set_postfix_str(
            f"loss={tl/max(len(train_loader),1):.3f} Acc={v_acc*100:.1f}% "
            f"F1m={v_f1*100:.1f}% best={best_val_f1*100:.1f}%")
        if no_improve >= patience:
            pbar.close(); print(f"    Early stop at epoch {epoch+1}"); break
    else:
        pbar.close()
    if best_state:
        model.load_state_dict(best_state)
    return best_state, best_val_f1


# =====================================================================
#  LIMU-BERT / LIMU-BERT-X  — two-stage finetune
# =====================================================================
def cross_finetune_limu(args):
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

    hhar_epochs = args.epochs or 200
    ft_epochs   = args.ft_epochs or 100
    batch_size  = args.batch_size or 128
    lr          = 1e-3
    ft_lr       = 5e-4
    seq_len     = getattr(args, 'limu_seq_len', 120)
    num_classes = args.num_classes
    class_labels = LABELS_5CLS if num_classes == 5 else LABELS_3CLS
    dt          = args.device_type
    protocol    = args.protocol

    # --- Load HHAR ---
    nobike_tag = '_nobike'
    hhar_dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}',
                    'all': f'hhar_all{nobike_tag}'}
    hhar_dir = LIMU_DIR / "dataset" / hhar_dir_map[dt]
    hhar_data = np.load(str(hhar_dir / "data_20_120.npy")).astype(np.float32)
    hhar_lbl  = np.load(str(hhar_dir / "label_20_120.npy")).astype(np.float32)
    y_hhar    = hhar_lbl[:, 0, 2].astype(int)
    hhar_data[:, :, :3] /= 9.8
    if num_classes == 5:
        hhar_data, y_hhar = filter_to_5class(hhar_data, y_hhar)
    else:
        hhar_data, y_hhar = filter_to_3class(hhar_data, y_hhar)

    # --- Load target dataset WITH subject IDs ---
    tgt = _get_target_cfg(args)
    tgt_dir = LIMU_DIR / "dataset" / tgt['limu'][dt]
    if not (tgt_dir / "data_20_120.npy").exists():
        print(f"ERROR: {tgt_dir} not found. Run {tgt['prepare_script']} first.")
        sys.exit(1)
    tgt_data = np.load(str(tgt_dir / "data_20_120.npy")).astype(np.float32)
    tgt_lbl  = np.load(str(tgt_dir / "label_20_120.npy")).astype(np.float32)
    y_tgt    = tgt_lbl[:, 0, 2].astype(int)
    tgt_subj = tgt_lbl[:, 0, 0].astype(int)
    tgt_data[:, :, :3] /= 9.8
    if num_classes == 5:
        tgt_data, y_tgt, tgt_subj = filter_to_5class_with_ids(tgt_data, y_tgt, tgt_subj)
    else:
        tgt_data, y_tgt, tgt_subj = filter_to_3class_with_ids(tgt_data, y_tgt, tgt_subj)

    tgt_display = tgt['display_name']
    all_subjects = sorted(np.unique(tgt_subj).tolist())

    # --- Split target data by protocol ---
    ft_subjects, test_subjects = None, None
    if protocol == 'leave-in':
        n_ft = args.finetune_subjects
        if n_ft >= len(all_subjects):
            print(f"ERROR: --finetune-subjects {n_ft} >= available subjects {len(all_subjects)}")
            sys.exit(1)
        X_ft, y_ft, X_test, y_test, ft_subjects, test_subjects = \
            split_leave_in(tgt_data, y_tgt, tgt_subj, n_ft, args.seed)
    else:
        X_ft, y_ft, X_test, y_test = \
            split_data_fraction(tgt_data, y_tgt, tgt_subj,
                                args.per_subject_fraction, args.seed)

    variant = "LIMU-BERT-X" if seq_len == 20 else "LIMU-BERT"
    print(f"\n  {variant} [{dt}] seq_len={seq_len}, {num_classes}-class, protocol={protocol}")
    print(f"  Stage 1 HHAR: {len(hhar_data)} samples")
    print(f"  Stage 2 fine-tune: {len(X_ft)} | Test: {len(X_test)}")

    # --- Model config ---
    bert_cfg = PretrainModelConfig(
        hidden=72, hidden_ff=144, feature_num=6,
        n_layers=4, n_heads=4, seq_len=seq_len, emb_norm=True)
    cls_cfg = ClassifierModelConfig(
        seq_len=seq_len, input=6, num_rnn=2, num_layers=[2, 1],
        rnn_io=[[6, 20], [20, 10]], num_cnn=0, conv_io=[], pool=[],
        flat_num=0, num_attn=0, num_head=0, atten_hidden=0,
        num_linear=1, linear_io=[[10, num_classes]], activ=False, dropout=True)

    data_seq_len = hhar_data.shape[1]
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

    def make_loader(X, y, shuffle=False, random_crop=False):
        if seq_len < data_seq_len:
            ds = _CropDataset(X, y, seq_len, random_crop=random_crop)
        else:
            ds = TensorDataset(torch.from_numpy(X).float(),
                               torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # ── Stage 1: Train on HHAR (cached) ──
    model_tag = "limubertx" if seq_len == 20 else "limubert"
    cache_path = _get_stage1_path(model_tag, dt, num_classes)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    classifier = ClassifierGRU(cls_cfg, input=bert_cfg.hidden, output=num_classes)
    model = BERTClassifier(bert_cfg, classifier=classifier, frozen_bert=False)
    model = model.to(device)

    stage1_f1 = -1.0
    if cache_path.exists() and not args.force_stage1:
        state = torch.load(str(cache_path), map_location=device)
        model.load_state_dict(state)
        print(f"\n  ── Stage 1: Loaded from cache ({cache_path.name}) ──")
    else:
        print(f"\n  ── Stage 1: Training on HHAR ({len(hhar_data)} samples) ──")
        np.random.seed(42)
        idx = np.arange(len(hhar_data)); np.random.shuffle(idx)
        vn = max(1, int(len(idx) * 0.1))
        hhar_val, hhar_val_y     = hhar_data[idx[:vn]], y_hhar[idx[:vn]]
        hhar_train, hhar_train_y = hhar_data[idx[vn:]], y_hhar[idx[vn:]]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        s1_ckpt = BEST_MODELS_DIR / f"limubert_stage1_train.pt"
        s1_train_loader = make_loader(hhar_train, hhar_train_y, shuffle=True, random_crop=True)
        s1_val_loader   = make_loader(hhar_val, hhar_val_y, random_crop=False)
        stage1_state, stage1_f1 = _train_loop(
            model, s1_train_loader, s1_val_loader, optimizer, criterion, device,
            hhar_epochs, patience=15, ckpt_path=s1_ckpt, desc="Stage1-HHAR")
        _safe_torch_save(stage1_state or model.state_dict(), cache_path)
        print(f"  Stage 1 cached -> {cache_path.name} (F1={stage1_f1*100:.1f}%)")

    # ── Stage 2: Fine-tune on target ──
    print(f"\n  ── Stage 2: Fine-tuning on {tgt_display} ({len(X_ft)} samples) ──")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ft_np = np.arange(len(X_ft)); np.random.shuffle(ft_np)
    ft_vn = max(1, int(len(ft_np) * 0.1))
    X_ft_val, y_ft_val     = X_ft[ft_np[:ft_vn]], y_ft[ft_np[:ft_vn]]
    X_ft_train, y_ft_train = X_ft[ft_np[ft_vn:]], y_ft[ft_np[ft_vn:]]

    ft_optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
    ft_criterion = nn.CrossEntropyLoss()
    ft_train_loader = make_loader(X_ft_train, y_ft_train, shuffle=True, random_crop=True)
    ft_val_loader   = make_loader(X_ft_val, y_ft_val, random_crop=False)

    exp_tag = args.experiment_tag or 'run'
    ft_ckpt = BEST_MODELS_DIR / f"limubert_{exp_tag}_stage2.pt"
    _, stage2_f1 = _train_loop(
        model, ft_train_loader, ft_val_loader, ft_optimizer, ft_criterion, device,
        ft_epochs, patience=10, ckpt_path=ft_ckpt, desc="Stage2-Finetune")
    print(f"  Stage 2 best val F1: {stage2_f1*100:.1f}%")

    # ── Test ──
    print(f"\n  Evaluating on {tgt_display} test ({len(X_test)} samples) ...")
    test_loader = make_loader(X_test, y_test, random_crop=False)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            tp.append(model(bx.to(device), False).argmax(1).cpu().numpy())
            tt.append(by.numpy())
    tp, tt = np.concatenate(tp), np.concatenate(tt)
    test_metrics = evaluate_test(tt, tp, class_labels, len(X_ft), len(X_test),
                                 test_name=tgt_display)

    model_name = f'limu-bert{"x" if seq_len == 20 else ""}-{dt}'
    result = {
        'experiment': f'EXP_cross_finetune_{args.target_dataset}',
        'target_dataset': args.target_dataset,
        'direction': f'HHAR -> finetune {tgt_display} -> test {tgt_display}',
        'protocol': protocol,
        'model': model_name, 'experiment_tag': exp_tag, 'seed': args.seed,
        'device_type': dt, 'channels': 6, 'seq_len': seq_len,
        'num_classes': num_classes,
        'stage1_cached': cache_path.name,
        'stage2_finetune_samples': len(X_ft),
        'stage2_best_val_f1': float(stage2_f1),
        'test_metrics': test_metrics,
    }
    if protocol == 'leave-in':
        result['finetune_subjects'] = args.finetune_subjects
        result['finetune_subject_ids'] = ft_subjects
        result['test_subject_ids'] = test_subjects
        result['data_fraction'] = f"Leave-{args.finetune_subjects}-in"
    else:
        result['per_subject_fraction'] = args.per_subject_fraction
        result['data_fraction'] = args.per_subject_fraction
    save_results(result, args.output_dir, exp_tag, model_name,
                 test_name=args.target_dataset)
    return result


# =====================================================================
#  ssl-wearables — two-stage finetune
# =====================================================================
def cross_finetune_ssl(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  [ssl-wearables] PyTorch {torch.__version__}, device={device}")

    sys.path.insert(0, str(SSL_DIR))
    from sslearning.models.accNet import Resnet

    hhar_epochs = args.epochs or 200
    ft_epochs   = args.ft_epochs or 100
    batch_size  = args.batch_size or 64
    lr          = 1e-4
    ft_lr       = 5e-5
    n_channels  = args.channels
    input_size  = 300
    num_classes = args.num_classes
    class_labels = LABELS_5CLS if num_classes == 5 else LABELS_3CLS
    dt          = args.device_type
    protocol    = args.protocol

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
        to, tn = np.linspace(0, 1, X_hhar.shape[1]), np.linspace(0, 1, input_size)
        X_hhar = _interp(to, X_hhar, kind='linear', axis=1, assume_sorted=True)(tn).astype(np.float32)
    X_hhar = np.transpose(X_hhar, (0, 2, 1))
    if num_classes == 5:
        X_hhar, y_hhar = filter_to_5class(X_hhar, y_hhar)
    else:
        X_hhar, y_hhar = filter_to_3class(X_hhar, y_hhar)

    # --- Load target dataset WITH subject IDs ---
    tgt = _get_target_cfg(args)
    tgt_dir = SSL_DIR / "data" / "downstream" / tgt['ssl'][dt]
    wx_file = tgt_dir / ("X6.npy" if n_channels == 6 else "X.npy")
    if not wx_file.exists():
        print(f"ERROR: {wx_file} not found. Run {tgt['prepare_script']} first."); sys.exit(1)
    X_tgt = np.load(str(wx_file)).astype(np.float32)
    y_tgt = np.load(str(tgt_dir / "Y.npy")).astype(np.int64)
    tgt_subj = np.load(str(tgt_dir / "pid.npy")).astype(np.int64)
    if X_tgt.shape[1] != input_size:
        from scipy.interpolate import interp1d as _interp
        to, tn = np.linspace(0, 1, X_tgt.shape[1]), np.linspace(0, 1, input_size)
        X_tgt = _interp(to, X_tgt, kind='linear', axis=1, assume_sorted=True)(tn).astype(np.float32)
    X_tgt = np.transpose(X_tgt, (0, 2, 1))
    if num_classes == 5:
        X_tgt, y_tgt, tgt_subj = filter_to_5class_with_ids(X_tgt, y_tgt, tgt_subj)
    else:
        X_tgt, y_tgt, tgt_subj = filter_to_3class_with_ids(X_tgt, y_tgt, tgt_subj)

    tgt_display = tgt['display_name']
    all_subjects = sorted(np.unique(tgt_subj).tolist())

    # --- Split target data by protocol ---
    ft_subjects, test_subjects = None, None
    if protocol == 'leave-in':
        n_ft = args.finetune_subjects
        if n_ft >= len(all_subjects):
            print(f"ERROR: --finetune-subjects {n_ft} >= available subjects {len(all_subjects)}")
            sys.exit(1)
        X_ft, y_ft, X_test, y_test, ft_subjects, test_subjects = \
            split_leave_in(X_tgt, y_tgt, tgt_subj, n_ft, args.seed)
    else:
        X_ft, y_ft, X_test, y_test = \
            split_data_fraction(X_tgt, y_tgt, tgt_subj,
                                args.per_subject_fraction, args.seed)

    print(f"\n  ssl-wearables [{dt}] {n_channels}ch, {num_classes}-class, protocol={protocol}")
    print(f"  Stage 1 HHAR: {len(X_hhar)} samples")
    print(f"  Stage 2 fine-tune: {len(X_ft)} | Test: {len(X_test)}")

    def make_loader(X, y, shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                        torch.from_numpy(y).long()),
                          batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # ── Stage 1: Train on HHAR (cached) ──
    model_tag = f"ssl_{n_channels}ch"
    cache_path = _get_stage1_path(model_tag, dt, num_classes)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42); torch.manual_seed(42)
    model = Resnet(output_size=num_classes, n_channels=n_channels,
                   is_eva=True, resnet_version=1, epoch_len=10)
    model = model.to(device)

    stage1_f1 = -1.0
    if cache_path.exists() and not args.force_stage1:
        state = torch.load(str(cache_path), map_location=device)
        model.load_state_dict(state)
        print(f"\n  ── Stage 1: Loaded from cache ({cache_path.name}) ──")
    else:
        print(f"\n  ── Stage 1: Training on HHAR ({len(X_hhar)} samples) ──")
        idx = np.arange(len(X_hhar)); np.random.shuffle(idx)
        vn = max(1, int(len(idx) * 0.1))
        hhar_val, hhar_val_y     = X_hhar[idx[:vn]], y_hhar[idx[:vn]]
        hhar_train, hhar_train_y = X_hhar[idx[vn:]], y_hhar[idx[vn:]]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        counter = collections.Counter(hhar_train_y.tolist())
        wts = [1.0 / (counter.get(i, 1) / len(hhar_train_y)) for i in range(num_classes)]
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(wts).to(device))

        s1_ckpt = BEST_MODELS_DIR / f"ssl_stage1_train.pt"
        s1_train_loader = make_loader(hhar_train, hhar_train_y, shuffle=True)
        s1_val_loader   = make_loader(hhar_val, hhar_val_y)
        stage1_state, stage1_f1 = _train_loop(
            model, s1_train_loader, s1_val_loader, optimizer, criterion, device,
            hhar_epochs, patience=10, ckpt_path=s1_ckpt, desc="Stage1-HHAR")
        _safe_torch_save(stage1_state or model.state_dict(), cache_path)
        print(f"  Stage 1 cached -> {cache_path.name} (F1={stage1_f1*100:.1f}%)")

    # ── Stage 2: Fine-tune on target ──
    print(f"\n  ── Stage 2: Fine-tuning on {tgt_display} ({len(X_ft)} samples) ──")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ft_counter = collections.Counter(y_ft.tolist())
    ft_wts = [1.0 / (ft_counter.get(i, 1) / max(len(y_ft), 1)) for i in range(num_classes)]
    ft_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(ft_wts).to(device))
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr, amsgrad=True)

    ft_np = np.arange(len(X_ft)); np.random.shuffle(ft_np)
    ft_vn = max(1, int(len(ft_np) * 0.1))
    X_ft_val, y_ft_val     = X_ft[ft_np[:ft_vn]], y_ft[ft_np[:ft_vn]]
    X_ft_train, y_ft_train = X_ft[ft_np[ft_vn:]], y_ft[ft_np[ft_vn:]]

    ft_train_loader = make_loader(X_ft_train, y_ft_train, shuffle=True)
    ft_val_loader   = make_loader(X_ft_val, y_ft_val)

    exp_tag = args.experiment_tag or 'run'
    ft_ckpt = BEST_MODELS_DIR / f"ssl_{exp_tag}_stage2.pt"
    _, stage2_f1 = _train_loop(
        model, ft_train_loader, ft_val_loader, ft_optimizer, ft_criterion, device,
        ft_epochs, patience=10, ckpt_path=ft_ckpt, desc="Stage2-Finetune")
    print(f"  Stage 2 best val F1: {stage2_f1*100:.1f}%")

    # ── Test ──
    print(f"\n  Evaluating on {tgt_display} test ({len(X_test)} samples) ...")
    test_loader = make_loader(X_test, y_test)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            tp.append(model(bx.to(device)).argmax(1).cpu().numpy())
            tt.append(by.numpy())
    tp, tt = np.concatenate(tp), np.concatenate(tt)
    test_metrics = evaluate_test(tt, tp, class_labels, len(X_ft), len(X_test),
                                 test_name=tgt_display)

    tag_name = f'ssl-wearables-{n_channels}ch-{dt}'
    result = {
        'experiment': f'EXP_cross_finetune_{args.target_dataset}',
        'target_dataset': args.target_dataset,
        'direction': f'HHAR -> finetune {tgt_display} -> test {tgt_display}',
        'protocol': protocol,
        'model': tag_name, 'experiment_tag': exp_tag, 'seed': args.seed,
        'device_type': dt, 'channels': n_channels,
        'num_classes': num_classes,
        'stage1_cached': cache_path.name,
        'stage2_finetune_samples': len(X_ft),
        'stage2_best_val_f1': float(stage2_f1),
        'test_metrics': test_metrics,
    }
    if protocol == 'leave-in':
        result['finetune_subjects'] = args.finetune_subjects
        result['finetune_subject_ids'] = ft_subjects
        result['test_subject_ids'] = test_subjects
        result['data_fraction'] = f"Leave-{args.finetune_subjects}-in"
    else:
        result['per_subject_fraction'] = args.per_subject_fraction
        result['data_fraction'] = args.per_subject_fraction
    save_results(result, args.output_dir, exp_tag, tag_name,
                 test_name=args.target_dataset)
    return result


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Cross-dataset fine-tuning (Exp 5.2): '
                    'train HHAR -> finetune target -> test target')
    parser.add_argument('--model', required=True,
                        choices=['limu-bert', 'ssl-wearables'])
    parser.add_argument('--device-type', type=str, default='all',
                        choices=['phone', 'watch', 'all'])
    parser.add_argument('--channels', type=int, default=6, choices=[3, 6])
    parser.add_argument('--limu-seq-len', type=int, default=120, choices=[20, 120],
                        help='LIMU-BERT seq length: 120=LIMU-BERT, 20=LIMU-BERT-X')
    parser.add_argument('--target-dataset', type=str, default='wisdm',
                        choices=VALID_TARGET_DATASETS)
    parser.add_argument('--with-stairs', action='store_true',
                        help='5-class (sit/stand/walk/upstairs/downstairs) '
                             'instead of default 3-class')

    proto = parser.add_mutually_exclusive_group(required=True)
    proto.add_argument('--finetune-subjects', type=int, default=None,
                       help='Protocol A (leave-in): number of target subjects '
                            'for fine-tuning, 100%% of their data (e.g. 1 or 5)')
    proto.add_argument('--per-subject-fraction', type=float, default=None,
                       help='Protocol B (data-fraction): fraction of EACH target '
                            "subject's data for fine-tuning, rest for test "
                            '(e.g. 0.10, 0.25, 0.75)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat-seeds', type=int, default=1)
    parser.add_argument('--experiment-tag', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Stage 1 (HHAR) epochs (default: 200)')
    parser.add_argument('--ft-epochs', type=int, default=None,
                        help='Stage 2 (finetune) epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--force-stage1', action='store_true',
                        help='Retrain Stage 1 even if a cached checkpoint exists')
    args = parser.parse_args()
    args.output_dir = str(_resolve_project_path(args.output_dir))

    if args.finetune_subjects is not None:
        args.protocol = 'leave-in'
    else:
        args.protocol = 'data-fraction'

    args.num_classes = 5 if args.with_stairs else 3

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tgt_cfg = _get_target_cfg(args)
    tgt_name = tgt_cfg['display_name']
    proto_str = (f"leave-{args.finetune_subjects}-in (100% data)"
                 if args.protocol == 'leave-in'
                 else f"data-fraction ({args.per_subject_fraction*100:.0f}% per subject)")
    print(f"\n{'='*60}")
    print(f"  Cross-Dataset Fine-Tuning (Exp 5.2)")
    print(f"  HHAR -> finetune {tgt_name} -> test {tgt_name}")
    print(f"  Target DS  : {args.target_dataset} ({tgt_name})")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {args.device_type}")
    print(f"  Channels   : {args.channels}")
    print(f"  Classes    : {args.num_classes} ({'with stairs' if args.with_stairs else '3-class'})")
    print(f"  Protocol   : {proto_str}")
    print(f"  Seed start : {args.seed}")
    print(f"  Repeats    : {args.repeat_seeds}")
    if args.model == 'limu-bert':
        print(f"  Seq len    : {args.limu_seq_len}")
    print(f"{'='*60}\n")

    dispatch = {
        'limu-bert':     cross_finetune_limu,
        'ssl-wearables': cross_finetune_ssl,
    }
    run_results = []
    base_seed = int(args.seed)
    for seed_offset in range(max(1, args.repeat_seeds)):
        args.seed = base_seed + seed_offset
        set_global_seed(args.seed)
        print(f"\n  >>> Running seed {args.seed}")
        result = dispatch[args.model](args)
        print_summary(result)
        run_results.append(result)

    if len(run_results) > 1:
        summary = aggregate_runs(run_results)
        print_repeated_summary(summary)
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
        save_results(final, args.output_dir, args.experiment_tag,
                     f"{final['model']}_repeat", test_name=args.target_dataset)


if __name__ == '__main__':
    main()
