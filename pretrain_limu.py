#!/usr/bin/env python
"""
=============================================================================
Self-Supervised Pretraining for LIMU-BERT-X  (Masked Span Reconstruction)
=============================================================================

Uses the LIMU-BERT masked prediction objective (span masking + MSE) to
pretrain a Transformer encoder on UNLABELED watch IMU data.

Input:   pretrain_data/limu_watch/data_20_120.npy   (N, 120, 6)
Output:  a .pt checkpoint that train_loso.py can load via --pretrained

The data contains 120-step windows.  Preprocess4Sample randomly crops a
seq_len (default 20) sub-window each iteration, matching LIMU-BERT-X usage.

Usage:
    # Finetune on top of existing limu_bert_x.pt weights
    python pretrain_limu.py \\
        --data pretrain_data/limu_watch/data_20_120.npy \\
        --load-weights weights/limu_bert_x.pt \\
        --output pretrain_data/limu_watch_finetune.pt

    # Pretrain from scratch
    python pretrain_limu.py \\
        --data pretrain_data/limu_watch/data_20_120.npy \\
        --output pretrain_data/limu_watch_scratch.pt
"""

import argparse, sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# ===================== LIMU-BERT imports via sys.path =====================
BASE_DIR = Path(__file__).resolve().parent
LIMU_DIR = (BASE_DIR / "code" / "LIMU-BERT_Experience-main"
            / "LIMU-BERT_Experience-main")
sys.path.insert(0, str(LIMU_DIR))

from models import LIMUBertModel4Pretrain          # noqa: E402
from config import PretrainModelConfig, MaskConfig  # noqa: E402
from utils import (LIBERTDataset4Pretrain,          # noqa: E402
                   Preprocess4Normalization,
                   Preprocess4Sample,
                   Preprocess4Mask)


def _resolve_project_path(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)


# ===================== Utility =====================

def _safe_torch_save(obj, path, retries=3, delay=0.5):
    """torch.save with retry for Windows file-locking (error 32)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            tmp = p.with_suffix(".tmp")
            torch.save(obj, str(tmp))
            tmp.replace(p)
            return
        except (RuntimeError, OSError):
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


def _load_weights_partial(model, weight_path, device="cpu"):
    """Load weights with shape-matching (skip mismatches)."""
    sd = torch.load(str(weight_path), map_location=device)
    if "module." in list(sd.keys())[0]:
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd)
    total = len(model_sd)
    print(f"  Loaded {loaded}/{total} tensors from {weight_path}")
    skipped = [(k, tuple(v.shape)) for k, v in sd.items()
               if k not in model_sd or model_sd[k].shape != v.shape]
    if skipped:
        print(f"  Skipped {len(skipped)}: {[s[0] for s in skipped]}")


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(
        description="Self-supervised pretraining for LIMU-BERT-X")
    ap.add_argument("--data", type=str, required=True,
                    help="Path to pretraining .npy  (N, 120, 6)")
    ap.add_argument("--load-weights", type=str, default=None,
                    help="Existing .pt to finetune on (e.g. weights/limu_bert_x.pt)")
    ap.add_argument("--seq-len", type=int, default=20,
                    help="Sequence length for LIMU-BERT-X (default 20)")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Number of pretraining epochs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", type=str, required=True,
                    help="Where to save the pretrained .pt checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Load data ----
    data_path = _resolve_project_path(args.data)
    if not data_path.exists():
        print(f"ERROR: data not found: {data_path}")
        sys.exit(1)

    data = np.load(str(data_path)).astype(np.float32)
    print(f"Data loaded: {data.shape}  (N, seq, features)")

    # ---- Model config ----
    model_cfg = PretrainModelConfig(
        hidden=72, hidden_ff=144, feature_num=6,
        n_layers=4, n_heads=4,
        seq_len=args.seq_len, emb_norm=True)

    mask_cfg = MaskConfig(
        mask_ratio=0.15, mask_alpha=6, max_gram=10,
        mask_prob=0.8, replace_prob=0.0)

    print(f"Model config: seq_len={args.seq_len}, hidden={model_cfg.hidden}, "
          f"layers={model_cfg.n_layers}, heads={model_cfg.n_heads}")

    # ---- Pipeline ----
    pipeline = [
        Preprocess4Normalization(model_cfg.feature_num),
        Preprocess4Sample(model_cfg.seq_len),
        Preprocess4Mask(mask_cfg),
    ]

    # ---- Train / val split (90/10) ----
    n = len(data)
    n_train = int(n * 0.9)
    perm = np.random.permutation(n)
    data_train = data[perm[:n_train]]
    data_val = data[perm[n_train:]]
    print(f"Train: {len(data_train)}, Val: {len(data_val)}")

    train_ds = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    val_ds = LIBERTDataset4Pretrain(data_val, pipeline=pipeline)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ---- Model ----
    model = LIMUBertModel4Pretrain(model_cfg)
    if args.load_weights:
        wp = _resolve_project_path(args.load_weights)
        if not wp.exists():
            wp = LIMU_DIR / args.load_weights
        if not wp.exists():
            wp = LIMU_DIR / "weights" / Path(args.load_weights).name
        if wp.exists():
            _load_weights_partial(model, wp)
        else:
            print(f"WARNING: weight file not found: {args.load_weights}")

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}  Device: {device}")

    # ---- Optimiser ----
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 30

    print(f"\nStarting pretraining for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss_sum, n_train_batches = 0.0, 0
        for batch in train_loader:
            mask_seqs, masked_pos, seqs = [t.to(device) for t in batch]
            optimizer.zero_grad()
            recon = model(mask_seqs, masked_pos)
            loss = criterion(recon, seqs).mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_train_batches += 1

        # --- validate ---
        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                mask_seqs, masked_pos, seqs = [t.to(device) for t in batch]
                recon = model(mask_seqs, masked_pos)
                loss = criterion(recon, seqs).mean()
                val_loss_sum += loss.item()
                n_val_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        val_loss = val_loss_sum / max(n_val_batches, 1)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            _safe_torch_save(best_state, _resolve_project_path(args.output))
            patience_counter = 0

        tag = " *best*" if improved else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"({dt:.1f}s){tag}")

        if not improved:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop: no improvement for {patience} epochs")
                break

    # ---- Final save ----
    if best_state is not None:
        _safe_torch_save(best_state, _resolve_project_path(args.output))
    print(f"\nDone.  Best val loss: {best_val_loss:.5f}")
    print(f"Checkpoint saved: {_resolve_project_path(args.output)}")


if __name__ == "__main__":
    main()
