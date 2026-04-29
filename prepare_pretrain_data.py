#!/usr/bin/env python
"""
=============================================================================
Prepare UNLABELED Pretraining Data for Self-Supervised Learning (Exp 2 & 3)
=============================================================================

This script is SEPARATE from prepare_wisdm_data.py (EXP5 labeled evaluation)
to avoid data leakage and confusion.  NO labels are stored in the outputs.

Datasets used:
  Watch pretrain (LIMU-BERT-X, 6ch): PAMAP2 wrist + WISDM watch
  Phone pretrain (SSL/HARNet):       SBHAR phone  + WISDM phone
  WISDM-all pretrain:                WISDM watch  + WISDM phone (cross-device)

Outputs:
  pretrain_data/limu_watch/data_20_120.npy       (N, 120, 6) accel+gyro @ 20 Hz
  pretrain_data/limu_wisdm_all/data_20_120.npy   (N, 120, 6) accel+gyro @ 20 Hz
  pretrain_data/ssl_phone_3ch/                   per-subject .npy  (N, 3, 300) @ 30 Hz
  pretrain_data/ssl_phone_6ch/                   per-subject .npy  (N, 6, 300) @ 30 Hz
  pretrain_data/ssl_wisdm_all_3ch/               per-subject .npy  (N, 3, 300) @ 30 Hz
  pretrain_data/ssl_wisdm_all_6ch/               per-subject .npy  (N, 6, 300) @ 30 Hz

Units:
  LIMU-BERT:  accel m/s², gyro rad/s  (Preprocess4Normalization divides accel by 9.8)
  SSL:        accel in g's clipped [-3,3], gyro rad/s (matches ssl-wearables pipeline)

Usage:
    python prepare_pretrain_data.py                     # all outputs
    python prepare_pretrain_data.py --limu              # LIMU-BERT watch only
    python prepare_pretrain_data.py --ssl               # SSL phone 3ch only
    python prepare_pretrain_data.py --ssl-6ch           # SSL phone 6ch only
    python prepare_pretrain_data.py --limu-wisdm-all    # LIMU-BERT WISDM watch+phone
    python prepare_pretrain_data.py --ssl-wisdm-all     # SSL WISDM watch+phone 3ch
    python prepare_pretrain_data.py --ssl-wisdm-all-6ch # SSL WISDM watch+phone 6ch
    python prepare_pretrain_data.py --force              # regenerate
"""

import argparse, os, sys, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ========================= Paths =========================
BASE_DIR = Path(__file__).resolve().parent

PAMAP2_ROOT = (BASE_DIR / "pamap2+physical+activity+monitoring"
               / "PAMAP2_Dataset" / "PAMAP2_Dataset")

WISDM_RAW_DIR = (BASE_DIR / "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"
                 / "wisdm-dataset" / "wisdm-dataset" / "raw")

SBHAR_ROOT = BASE_DIR / "SBHAR"

OUT_ROOT = BASE_DIR / "pretrain_data"

# ========================= Utility =========================

def resize_1d(data, target_len):
    """Resize (T, C) array along axis-0 to target_len via linear interpolation."""
    T = data.shape[0]
    if T == target_len:
        return data
    t_orig = np.linspace(0, 1, T, endpoint=True)
    t_new = np.linspace(0, 1, target_len, endpoint=True)
    return interp1d(t_orig, data, kind="linear", axis=0,
                    assume_sorted=True)(t_new).astype(np.float32)


# ========================= WISDM helpers =========================

def _load_wisdm_file(filepath):
    """Parse one WISDM raw txt file (CSV with ; terminator)."""
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue
            try:
                rows.append((int(parts[0]), parts[1].strip(),
                             int(parts[2]),
                             float(parts[3]), float(parts[4]), float(parts[5])))
            except (ValueError, IndexError):
                continue
    if not rows:
        return pd.DataFrame(columns=["subject", "activity", "timestamp",
                                     "x", "y", "z"])
    return pd.DataFrame(rows, columns=["subject", "activity", "timestamp",
                                       "x", "y", "z"])


def _discover_wisdm_subjects(device):
    accel_dir = WISDM_RAW_DIR / device / "accel"
    if not accel_dir.exists():
        return []
    sids = set()
    for f in accel_dir.glob("data_*_accel_*.txt"):
        try:
            sids.add(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(sids)


def _load_wisdm_device(device):
    """Load ALL WISDM 6-ch data for *device* (all activities, no labels).

    Returns dict  {subject_id: list of np.array(T_i, 6)} at native 20 Hz.
    Each array is one continuous activity segment with accel+gyro synchronised.
    Units: accel m/s², gyro rad/s.
    """
    accel_dir = WISDM_RAW_DIR / device / "accel"
    gyro_dir = WISDM_RAW_DIR / device / "gyro"
    if not accel_dir.exists():
        return {}

    subject_segs: dict[int, list[np.ndarray]] = {}

    for afile in sorted(accel_dir.glob("data_*_accel_*.txt")):
        sid = int(afile.stem.split("_")[1])
        gfile = gyro_dir / afile.name.replace("accel", "gyro")

        adf = _load_wisdm_file(afile)
        if adf.empty:
            continue
        gdf = _load_wisdm_file(gfile) if gfile.exists() else pd.DataFrame()

        for _, agroup in adf.groupby("activity"):
            ag = agroup.sort_values("timestamp").reset_index(drop=True)
            acc_xyz = ag[["x", "y", "z"]].values.astype(np.float32)
            acc_ts = ag["timestamp"].values.astype(np.float64)

            if gdf.empty or len(acc_xyz) < 20:
                continue

            ga = gdf[gdf["activity"] == ag["activity"].iloc[0]]
            ga = ga.sort_values("timestamp").reset_index(drop=True)
            if len(ga) < 2:
                continue

            gyro_xyz = ga[["x", "y", "z"]].values.astype(np.float32)
            gyro_ts = ga["timestamp"].values.astype(np.float64)

            t0, t1 = max(acc_ts[0], gyro_ts[0]), min(acc_ts[-1], gyro_ts[-1])
            mask = (acc_ts >= t0) & (acc_ts <= t1)
            if mask.sum() < 20:
                continue
            acc_xyz = acc_xyz[mask]
            acc_ts = acc_ts[mask]
            gyro_interp = np.column_stack([
                np.interp(acc_ts, gyro_ts, gyro_xyz[:, ch])
                for ch in range(3)
            ]).astype(np.float32)

            seg6 = np.concatenate([acc_xyz, gyro_interp], axis=1)
            subject_segs.setdefault(sid, []).append(seg6)

    return subject_segs


# ========================= PAMAP2 helpers =========================
# Hand/wrist IMU columns (0-indexed in the 54-column row):
#   4-6  = accel ±16 g (m/s²)
#   10-12 = gyroscope  (rad/s)
# We do NOT include magnetometer or chest/ankle sensors.
_PAMAP2_COLS = [4, 5, 6, 10, 11, 12]


def _load_pamap2_wrist():
    """Load PAMAP2 hand/wrist 6-ch data from Protocol + Optional.

    Returns dict {subject_id: np.array(T, 6)} at native 100 Hz.
    Units: accel m/s², gyro rad/s.
    """
    result: dict[int, list[np.ndarray]] = {}

    for folder in ("Protocol", "Optional"):
        data_dir = PAMAP2_ROOT / folder
        if not data_dir.exists():
            continue
        for dat_file in sorted(data_dir.glob("subject*.dat")):
            sid = int(dat_file.stem.replace("subject", ""))
            raw = []
            with open(dat_file, "r") as fh:
                for line in fh:
                    vals = line.strip().split()
                    if len(vals) < 54:
                        continue
                    try:
                        row = [float(v) for v in vals]
                    except ValueError:
                        continue
                    raw.append(row)
            if not raw:
                continue
            arr = np.array(raw, dtype=np.float64)
            hand6 = arr[:, _PAMAP2_COLS].astype(np.float32)
            good = ~np.isnan(hand6).any(axis=1)
            hand6 = hand6[good]
            if len(hand6) > 0:
                result.setdefault(sid, []).append(hand6)

    merged = {}
    for sid in sorted(result.keys()):
        merged[sid] = np.vstack(result[sid])
    return merged


# ========================= SBHAR helpers =========================

def _load_sbhar_phone(channels=3):
    """Load SBHAR raw data for all users.

    Returns dict {user_id: np.array(T, channels)} at native 50 Hz.
    Accel already in g's.  Gyro in rad/s.
    """
    raw_dir = SBHAR_ROOT / "RawData"
    if not raw_dir.exists():
        print(f"  SBHAR RawData not found at {raw_dir}")
        return {}

    user_acc: dict[int, list[np.ndarray]] = {}
    user_gyro: dict[int, list[np.ndarray]] = {}

    for acc_file in sorted(raw_dir.glob("acc_exp*_user*.txt")):
        m = re.search(r"acc_exp(\d+)_user(\d+)", acc_file.stem)
        if not m:
            continue
        exp_id, uid = int(m.group(1)), int(m.group(2))
        data = np.loadtxt(str(acc_file), dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 3:
            continue
        user_acc.setdefault(uid, []).append(data)

        if channels == 6:
            gyro_file = raw_dir / f"gyro_exp{exp_id:02d}_user{uid:02d}.txt"
            if gyro_file.exists():
                gdata = np.loadtxt(str(gyro_file), dtype=np.float32)
                if gdata.ndim == 2 and gdata.shape[1] == 3:
                    user_gyro.setdefault(uid, []).append(gdata)

    merged = {}
    for uid in sorted(user_acc.keys()):
        acc_cat = np.vstack(user_acc[uid])
        if channels == 3:
            merged[uid] = acc_cat
        else:
            if uid in user_gyro:
                gyro_cat = np.vstack(user_gyro[uid])
                min_len = min(len(acc_cat), len(gyro_cat))
                merged[uid] = np.concatenate(
                    [acc_cat[:min_len], gyro_cat[:min_len]], axis=1
                )
    return merged


# =====================================================================
#  LIMU-BERT Watch Pretraining Data
# =====================================================================
def prepare_limu_watch(force=False):
    """PAMAP2 wrist (100 Hz → 20 Hz) + WISDM watch (20 Hz)
    → pretrain_data/limu_watch/data_20_120.npy  shape (N, 120, 6)
    """
    print("=" * 65)
    print("  Preparing LIMU-BERT Watch Pretraining Data (6ch, 20 Hz)")
    print("=" * 65)

    out_dir = OUT_ROOT / "limu_watch"
    out_path = out_dir / "data_20_120.npy"

    if not force and out_path.exists():
        d = np.load(str(out_path))
        print(f"  Already exists: {d.shape}  → {out_path}")
        return

    SEQ_LEN = 120
    all_windows: list[np.ndarray] = []

    # ---- PAMAP2 wrist ----
    print("\n  Loading PAMAP2 wrist data ...")
    pamap2 = _load_pamap2_wrist()
    n_pamap = 0
    for sid, data100 in pamap2.items():
        data20 = data100[::5]                       # 100 Hz → 20 Hz
        n_win = len(data20) // SEQ_LEN
        for i in range(n_win):
            all_windows.append(data20[i * SEQ_LEN:(i + 1) * SEQ_LEN])
        n_pamap += n_win
    print(f"    PAMAP2: {len(pamap2)} subjects, {n_pamap} windows")

    # ---- WISDM watch ----
    print("  Loading WISDM watch data ...")
    wisdm = _load_wisdm_device("watch")
    n_wisdm = 0
    for sid, segs in wisdm.items():
        for seg in segs:
            n_win = len(seg) // SEQ_LEN
            for i in range(n_win):
                all_windows.append(seg[i * SEQ_LEN:(i + 1) * SEQ_LEN])
            n_wisdm += n_win
    print(f"    WISDM watch: {len(wisdm)} subjects, {n_wisdm} windows")

    if not all_windows:
        print("  ERROR: no pretraining windows produced!"); return

    data_arr = np.array(all_windows, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), data_arr)
    print(f"\n  ✓ Saved: {data_arr.shape} → {out_path}")
    print(f"    Total: {len(all_windows)} windows "
          f"(PAMAP2={n_pamap}, WISDM={n_wisdm})")


# =====================================================================
#  LIMU-BERT WISDM-All Pretraining Data  (watch + phone combined)
# =====================================================================
def prepare_limu_wisdm_all(force=False):
    """WISDM watch (20 Hz) + WISDM phone (20 Hz)
    → pretrain_data/limu_wisdm_all/data_20_120.npy  shape (N, 120, 6)
    """
    print("=" * 65)
    print("  Preparing LIMU-BERT WISDM-All Pretraining Data (6ch, 20 Hz)")
    print("=" * 65)

    out_dir = OUT_ROOT / "limu_wisdm_all"
    out_path = out_dir / "data_20_120.npy"

    if not force and out_path.exists():
        d = np.load(str(out_path))
        print(f"  Already exists: {d.shape}  → {out_path}")
        return

    SEQ_LEN = 120
    all_windows: list[np.ndarray] = []
    device_counts = {}

    for device_name in ("watch", "phone"):
        print(f"\n  Loading WISDM {device_name} data ...")
        wisdm = _load_wisdm_device(device_name)
        n_win = 0
        for sid, segs in wisdm.items():
            for seg in segs:
                nw = len(seg) // SEQ_LEN
                for i in range(nw):
                    all_windows.append(seg[i * SEQ_LEN:(i + 1) * SEQ_LEN])
                n_win += nw
        device_counts[device_name] = n_win
        print(f"    WISDM {device_name}: {len(wisdm)} subjects, {n_win} windows")

    if not all_windows:
        print("  ERROR: no pretraining windows produced!"); return

    data_arr = np.array(all_windows, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), data_arr)
    print(f"\n  ✓ Saved: {data_arr.shape} → {out_path}")
    print(f"    Total: {len(all_windows)} windows "
          f"(watch={device_counts.get('watch',0)}, "
          f"phone={device_counts.get('phone',0)})")


# =====================================================================
#  SSL WISDM-All Pretraining Data  (watch + phone combined)
# =====================================================================
def prepare_ssl_wisdm_all(channels, force=False):
    """WISDM watch (20 Hz) + WISDM phone (20 Hz)
    → pretrain_data/ssl_wisdm_all_{3,6}ch/  per-subject .npy (N, C, 300) @ 30 Hz
    """
    ch_tag = f"{channels}ch"
    print("\n" + "=" * 65)
    print(f"  Preparing SSL WISDM-All Pretraining Data ({ch_tag}, 30 Hz)")
    print("=" * 65)

    out_dir = OUT_ROOT / f"ssl_wisdm_all_{ch_tag}"
    fl_path = out_dir / "file_list.csv"

    if not force and fl_path.exists():
        fl = pd.read_csv(str(fl_path))
        print(f"  Already exists: {len(fl)} subjects → {out_dir}")
        return

    WINDOW_30 = 300       # 30 Hz × 10 s
    subject_epochs: dict[str, np.ndarray] = {}
    device_counts = {}

    for device_name in ("watch", "phone"):
        print(f"\n  Loading WISDM {device_name} data ...")
        wisdm = _load_wisdm_device(device_name)
        dev_total = 0
        for sid, segs in wisdm.items():
            epochs = []
            for seg in segs:
                if channels == 3:
                    seg_use = seg[:, :3]
                else:
                    seg_use = seg
                seg_norm = seg_use.copy()
                seg_norm[:, :3] = np.clip(seg_norm[:, :3] / 9.8, -3.0, 3.0)

                WIN_20 = 200   # 20 Hz × 10 s
                n_win = len(seg_norm) // WIN_20
                for i in range(n_win):
                    w = seg_norm[i * WIN_20:(i + 1) * WIN_20]
                    w300 = resize_1d(w, WINDOW_30)       # (300, C)
                    epochs.append(w300.T)                # (C, 300)
                dev_total += n_win
            if epochs:
                key = f"wisdm_{device_name}_{sid}"
                subject_epochs[key] = np.array(epochs, dtype=np.float32)
        device_counts[device_name] = dev_total
        print(f"    WISDM {device_name}: {len(wisdm)} subjects, {dev_total} epochs")

    if not subject_epochs:
        print("  ERROR: no pretraining epochs produced!"); return

    out_dir.mkdir(parents=True, exist_ok=True)
    file_list = []
    for key in sorted(subject_epochs.keys()):
        fpath = out_dir / f"{key}.npy"
        np.save(str(fpath), subject_epochs[key])
        file_list.append(fpath.name)

    pd.DataFrame({"file_list": file_list}).to_csv(str(fl_path), index=False)

    total = sum(len(v) for v in subject_epochs.values())
    print(f"\n  ✓ Saved: {len(subject_epochs)} subjects, {total} epochs "
          f"(watch={device_counts.get('watch',0)}, "
          f"phone={device_counts.get('phone',0)})")
    print(f"    → {out_dir}")


# =====================================================================
#  SSL Phone Pretraining Data
# =====================================================================
def prepare_ssl_phone(channels, force=False):
    """SBHAR phone (50 Hz) + WISDM phone (20 Hz)
    → pretrain_data/ssl_phone_{3,6}ch/  per-subject .npy (N, C, 300) @ 30 Hz
    """
    ch_tag = f"{channels}ch"
    print("\n" + "=" * 65)
    print(f"  Preparing SSL Phone Pretraining Data ({ch_tag}, 30 Hz)")
    print("=" * 65)

    out_dir = OUT_ROOT / f"ssl_phone_{ch_tag}"
    fl_path = out_dir / "file_list.csv"

    if not force and fl_path.exists():
        fl = pd.read_csv(str(fl_path))
        print(f"  Already exists: {len(fl)} subjects → {out_dir}")
        return

    WINDOW_30 = 300       # 30 Hz × 10 s
    subject_epochs: dict[str, np.ndarray] = {}

    # ---- SBHAR phone (50 Hz) ----
    print("\n  Loading SBHAR phone data ...")
    sbhar = _load_sbhar_phone(channels)
    sbhar_total = 0
    for uid, data50 in sbhar.items():
        acc_part = data50[:, :3]
        acc_g = np.clip(acc_part, -3.0, 3.0)       # already in g's
        if channels == 6:
            gyro_part = data50[:, 3:]
            data_proc = np.concatenate([acc_g, gyro_part], axis=1)
        else:
            data_proc = acc_g

        WIN_50 = 500   # 50 Hz × 10 s
        n_win = len(data_proc) // WIN_50
        if n_win == 0:
            continue
        epochs = []
        for i in range(n_win):
            w = data_proc[i * WIN_50:(i + 1) * WIN_50]
            w300 = resize_1d(w, WINDOW_30)           # (300, C)
            epochs.append(w300.T)                    # (C, 300)
        subject_epochs[f"sbhar_{uid:02d}"] = np.array(epochs, dtype=np.float32)
        sbhar_total += n_win
    print(f"    SBHAR: {len(sbhar)} users, {sbhar_total} epochs")

    # ---- WISDM phone (20 Hz) ----
    print("  Loading WISDM phone data ...")
    wisdm = _load_wisdm_device("phone")
    wisdm_total = 0
    for sid, segs in wisdm.items():
        epochs = []
        for seg in segs:
            if channels == 3:
                seg_use = seg[:, :3]
            else:
                seg_use = seg
            # normalise accel to g's
            seg_norm = seg_use.copy()
            seg_norm[:, :3] = np.clip(seg_norm[:, :3] / 9.8, -3.0, 3.0)

            WIN_20 = 200   # 20 Hz × 10 s
            n_win = len(seg_norm) // WIN_20
            for i in range(n_win):
                w = seg_norm[i * WIN_20:(i + 1) * WIN_20]
                w300 = resize_1d(w, WINDOW_30)       # (300, C)
                epochs.append(w300.T)                # (C, 300)
            wisdm_total += n_win
        if epochs:
            subject_epochs[f"wisdm_{sid}"] = np.array(epochs, dtype=np.float32)
    print(f"    WISDM phone: {len(wisdm)} subjects, {wisdm_total} epochs")

    if not subject_epochs:
        print("  ERROR: no pretraining epochs produced!"); return

    # ---- save per-subject .npy + file_list.csv ----
    out_dir.mkdir(parents=True, exist_ok=True)
    file_list = []
    for key in sorted(subject_epochs.keys()):
        fpath = out_dir / f"{key}.npy"
        np.save(str(fpath), subject_epochs[key])
        file_list.append(fpath.name)

    pd.DataFrame({"file_list": file_list}).to_csv(str(fl_path), index=False)

    total = sum(len(v) for v in subject_epochs.values())
    print(f"\n  ✓ Saved: {len(subject_epochs)} subjects, {total} epochs "
          f"(SBHAR={sbhar_total}, WISDM={wisdm_total})")
    print(f"    → {out_dir}")


# =====================================================================
#  Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Prepare UNLABELED pretraining data (Exp 2 & 3)")
    ap.add_argument("--limu", action="store_true",
                    help="LIMU-BERT watch data only")
    ap.add_argument("--ssl", action="store_true",
                    help="SSL phone 3ch data only")
    ap.add_argument("--ssl-6ch", action="store_true",
                    help="SSL phone 6ch data only")
    ap.add_argument("--limu-wisdm-all", action="store_true",
                    help="LIMU-BERT WISDM watch+phone combined")
    ap.add_argument("--ssl-wisdm-all", action="store_true",
                    help="SSL WISDM watch+phone 3ch combined")
    ap.add_argument("--ssl-wisdm-all-6ch", action="store_true",
                    help="SSL WISDM watch+phone 6ch combined")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if files exist")
    args = ap.parse_args()

    any_specific = (args.limu or args.ssl or args.ssl_6ch
                    or args.limu_wisdm_all
                    or args.ssl_wisdm_all or args.ssl_wisdm_all_6ch)
    do_all = not any_specific

    # ---- Validate paths ----
    missing = []
    need_pamap2 = do_all or args.limu
    need_wisdm_watch = do_all or args.limu or args.limu_wisdm_all or args.ssl_wisdm_all or args.ssl_wisdm_all_6ch
    need_wisdm_phone = do_all or args.ssl or args.ssl_6ch or args.limu_wisdm_all or args.ssl_wisdm_all or args.ssl_wisdm_all_6ch
    need_sbhar = do_all or args.ssl or args.ssl_6ch

    if need_pamap2 and not PAMAP2_ROOT.exists():
        missing.append(f"PAMAP2: {PAMAP2_ROOT}")
    if need_wisdm_watch and not (WISDM_RAW_DIR / "watch").exists():
        missing.append(f"WISDM watch: {WISDM_RAW_DIR / 'watch'}")
    if need_wisdm_phone and not (WISDM_RAW_DIR / "phone").exists():
        missing.append(f"WISDM phone: {WISDM_RAW_DIR / 'phone'}")
    if need_sbhar and not SBHAR_ROOT.exists():
        missing.append(f"SBHAR: {SBHAR_ROOT}")
    if missing:
        for m in missing:
            print(f"  ERROR: not found: {m}")
        sys.exit(1)

    print(f"Output root: {OUT_ROOT}\n")

    if do_all or args.limu:
        prepare_limu_watch(force=args.force)
    if do_all or args.ssl:
        prepare_ssl_phone(channels=3, force=args.force)
    if do_all or args.ssl_6ch:
        prepare_ssl_phone(channels=6, force=args.force)
    if do_all or args.limu_wisdm_all:
        prepare_limu_wisdm_all(force=args.force)
    if do_all or args.ssl_wisdm_all:
        prepare_ssl_wisdm_all(channels=3, force=args.force)
    if do_all or args.ssl_wisdm_all_6ch:
        prepare_ssl_wisdm_all(channels=6, force=args.force)

    print("\n  Done!")


if __name__ == "__main__":
    main()
