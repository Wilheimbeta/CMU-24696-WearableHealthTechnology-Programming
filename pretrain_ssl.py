#!/usr/bin/env python
"""
=============================================================================
Multi-Task Self-Supervised Pretraining for HARNet / ssl-wearables
=============================================================================

Uses 4 binary pretext tasks (time reversal, permutation, scale, time warp)
to pretrain a ResNet encoder on UNLABELED phone IMU data.

Input:   pretrain_data/ssl_phone_{3,6}ch/   (per-subject .npy, C x 300 @ 30 Hz)
Output:  a .mdl checkpoint that train_loso.py can load via --pretrained

Usage:
    # Finetune on top of existing mtl_best.mdl weights (3ch)
    python pretrain_ssl.py \\
        --data-dir pretrain_data/ssl_phone_3ch \\
        --load-weights code/ssl-wearables-main/ssl-wearables-main/model_check_point/mtl_best.mdl \\
        --channels 3  --output pretrain_data/ssl_phone_finetune_3ch.mdl

    # Pretrain from scratch (6ch)
    python pretrain_ssl.py \\
        --data-dir pretrain_data/ssl_phone_6ch \\
        --channels 6  --output pretrain_data/ssl_phone_scratch_6ch.mdl
"""

import argparse, sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline

# ===================== ssl-wearables model import =====================
BASE_DIR = Path(__file__).resolve().parent
SSL_DIR = (BASE_DIR / "code" / "ssl-wearables-main"
           / "ssl-wearables-main")
sys.path.insert(0, str(SSL_DIR))

from sslearning.models.accNet import Resnet  # noqa: E402


def _resolve_project_path(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)

# ===================== Task index constants =====================
TIME_REVERSAL = 0
SCALE = 1
PERMUTATION = 2
TIME_WARPED = 3
N_TASKS = 4


# ===================== Transforms (n-channel safe) =====================

def _flip(x):
    """Time reversal.  x: (C, T) → (C, T)."""
    return np.ascontiguousarray(np.flip(x, axis=1))


def _permute(x, n_perm=4, min_seg=10):
    """Segment permutation.  x: (C, T) → (C, T)."""
    T = x.shape[1]
    if T < n_perm * min_seg:
        return x
    while True:
        segs = np.zeros(n_perm + 1, dtype=int)
        segs[-1] = T
        segs[1:-1] = np.sort(
            np.random.randint(min_seg, T - min_seg, n_perm - 1))
        if np.min(segs[1:] - segs[:-1]) > min_seg:
            break
    idx = np.random.permutation(n_perm)
    x_t = x.T  # (T, C)
    out = np.zeros_like(x_t)
    pp = 0
    for ii in range(n_perm):
        chunk = x_t[segs[idx[ii]]:segs[idx[ii] + 1]]
        out[pp:pp + len(chunk)] = chunk
        pp += len(chunk)
    return out.T


def _scale(x, scale_range=0.5, min_diff=0.15):
    """Random per-channel scaling.  x: (C, T) → (C, T)."""
    C = x.shape[0]
    lo, hi = 1 - scale_range, 1 + scale_range
    while True:
        factors = np.random.uniform(lo, hi, size=C).astype(np.float32)
        if all(abs(f - 1.0) >= min_diff for f in factors):
            break
    return x * factors[:, None]


def _time_warp(x, sigma=0.2, knot=4):
    """Random temporal distortion per channel.  x: (C, T) → (C, T)."""
    C, T = x.shape
    xx = np.linspace(0, T - 1, knot + 2)
    x_range = np.arange(T)
    out = np.zeros_like(x)
    for ch in range(C):
        yy = np.random.normal(1.0, sigma, knot + 2)
        cs = CubicSpline(xx, yy)
        tt_cum = np.cumsum(cs(x_range))
        tt_cum = tt_cum * (T - 1) / tt_cum[-1]
        out[ch] = np.interp(x_range, tt_cum, x[ch])
    return out


def generate_ssl_labels(x, positive_ratio=0.5):
    """Apply 4 pretext transforms and return (x_transformed, labels[4])."""
    labels = np.zeros(N_TASKS, dtype=np.int64)

    c = int(np.random.rand() < positive_ratio)
    if c:
        x = _flip(x)
    labels[TIME_REVERSAL] = c

    c = int(np.random.rand() < positive_ratio)
    if c:
        x = _scale(x)
    labels[SCALE] = c

    c = int(np.random.rand() < positive_ratio)
    if c:
        x = _permute(x)
    labels[PERMUTATION] = c

    c = int(np.random.rand() < positive_ratio)
    if c:
        x = _time_warp(x)
    labels[TIME_WARPED] = c

    return x, labels


# ===================== Dataset =====================

class SSLPretrainDataset(Dataset):
    """Loads all epoch data and applies SSL transforms on-the-fly."""

    def __init__(self, epochs):
        """epochs: np.array (N, C, epoch_len)"""
        self.epochs = epochs

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        x = self.epochs[idx].copy()  # (C, T)
        x, labels = generate_ssl_labels(x)
        return (torch.from_numpy(x).float(),
                torch.tensor(labels[TIME_REVERSAL]).long(),
                torch.tensor(labels[SCALE]).long(),
                torch.tensor(labels[PERMUTATION]).long(),
                torch.tensor(labels[TIME_WARPED]).long())


# ===================== Utility =====================

def _safe_torch_save(obj, path, retries=3, delay=0.5):
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
    sd = torch.load(str(weight_path), map_location=device)
    if "module." in list(sd.keys())[0]:
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    loaded, skipped = 0, []
    for k, v in sd.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            skipped.append(k)
            continue
        model_sd[k] = v
        loaded += 1
    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded}/{len(model_sd)} tensors from {weight_path}")
    if skipped:
        print(f"  Skipped (shape mismatch): {skipped}")


def _load_epoch_data(data_dir):
    """Load all per-subject .npy files from data_dir, return (N, C, T)."""
    import pandas as pd
    data_dir = Path(data_dir)
    fl_path = data_dir / "file_list.csv"
    if not fl_path.exists():
        print(f"ERROR: file_list.csv not found in {data_dir}")
        sys.exit(1)
    fl = pd.read_csv(str(fl_path))
    all_epochs = []
    for fpath in fl["file_list"]:
        fpath = Path(str(fpath))
        if not fpath.is_absolute():
            fpath = data_dir / fpath
        elif not fpath.exists():
            # Backward compatibility: old file_list.csv may store absolute
            # paths from another machine; recover by using just the filename.
            fpath = data_dir / fpath.name
        epochs = np.load(str(fpath))  # (N_i, C, T)
        all_epochs.append(epochs)
    combined = np.concatenate(all_epochs, axis=0).astype(np.float32)
    print(f"  Loaded {len(fl)} subjects, {len(combined)} total epochs, "
          f"shape per epoch: {combined.shape[1:]}")
    return combined


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(
        description="Multi-task SSL pretraining for HARNet")
    ap.add_argument("--data-dir", type=str, required=True,
                    help="Dir with per-subject .npy + file_list.csv")
    ap.add_argument("--load-weights", type=str, default=None,
                    help="Existing .mdl to finetune on")
    ap.add_argument("--channels", type=int, default=3, choices=[3, 6],
                    help="Number of input channels (default 3)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", type=str, required=True,
                    help="Where to save the pretrained .mdl checkpoint")
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (epochs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Load data ----
    args.data_dir = str(_resolve_project_path(args.data_dir))
    args.output = str(_resolve_project_path(args.output))
    if args.load_weights:
        args.load_weights = str(_resolve_project_path(args.load_weights))

    print(f"Loading data from {args.data_dir} ...")
    all_epochs = _load_epoch_data(args.data_dir)

    # ---- Train / val split (90/10 random) ----
    n = len(all_epochs)
    n_train = int(n * 0.9)
    perm = np.random.permutation(n)
    train_data = all_epochs[perm[:n_train]]
    val_data = all_epochs[perm[n_train:]]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    train_ds = SSLPretrainDataset(train_data)
    val_ds = SSLPretrainDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ---- Model ----
    model = Resnet(
        output_size=2,
        n_channels=args.channels,
        is_mtl=True,
        epoch_len=10,
    )
    if args.load_weights:
        wp = _resolve_project_path(args.load_weights)
        if not wp.exists():
            wp = SSL_DIR / args.load_weights
        if not wp.exists():
            wp = SSL_DIR / "model_check_point" / Path(args.load_weights).name
        if wp.exists():
            _load_weights_partial(model, wp)
        else:
            print(f"WARNING: weight file not found: {args.load_weights}")

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: Resnet ({args.channels}ch, is_mtl=True), "
          f"{param_count:,} params, device={device}")

    # ---- Optimiser ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nStarting SSL pretraining for {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device)
            aot_y = batch[1].to(device)
            scale_y = batch[2].to(device)
            perm_y = batch[3].to(device)
            tw_y = batch[4].to(device)

            optimizer.zero_grad()
            aot_out, scale_out, perm_out, tw_out = model(x)

            loss = (criterion(aot_out, aot_y)
                    + criterion(scale_out, scale_y)
                    + criterion(perm_out, perm_y)
                    + criterion(tw_out, tw_y)) / N_TASKS
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        # --- validate ---
        model.eval()
        val_loss_sum, n_val = 0.0, 0
        val_correct = np.zeros(N_TASKS)
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                labels = [batch[i + 1].to(device) for i in range(N_TASKS)]

                outs = model(x)
                loss = sum(criterion(outs[i], labels[i])
                           for i in range(N_TASKS)) / N_TASKS
                val_loss_sum += loss.item()
                n_val += 1

                bs = x.size(0)
                val_total += bs
                for i in range(N_TASKS):
                    preds = outs[i].argmax(dim=1)
                    val_correct[i] += (preds == labels[i]).sum().item()

        train_loss = train_loss_sum / max(n_batches, 1)
        val_loss = val_loss_sum / max(n_val, 1)
        accs = val_correct / max(val_total, 1) * 100
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            _safe_torch_save(best_state, args.output)
            patience_counter = 0

        tag = " *best*" if improved else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc(aot/scl/prm/tw)="
              f"{accs[0]:.0f}/{accs[1]:.0f}/{accs[2]:.0f}/{accs[3]:.0f}%  "
              f"({dt:.1f}s){tag}")

        if not improved:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stop: no improvement for {args.patience} epochs")
                break

    if best_state is not None:
        _safe_torch_save(best_state, args.output)
    print(f"\nDone.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved: {args.output}")


if __name__ == "__main__":
    main()
