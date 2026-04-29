#!/usr/bin/env python
"""
=============================================================================
WISDM Data Preparation for Cross-Dataset Evaluation (EXP5)
=============================================================================

Reads raw WISDM sensor files and prepares data in the exact same formats
used by each model so that eval_cross_dataset.py can load them identically
to HHAR data.

  HART:          datasetStandardized/WISDM{_phone,_watch}/   (.hkl)
  LIMU-BERT:     dataset/wisdm{,_watch,_all}/                (.npy)
  ssl-wearables: data/downstream/wisdm{,_watch,_all}/        (.npy)

Activities extracted (label index matches HHAR no-bike first 3 classes):
    0 = sitting   (WISDM code 'D')
    1 = standing  (WISDM code 'E')
    2 = walking   (WISDM code 'A')
    3 = stairs    (WISDM code 'C')  -- supplementary, undifferentiated

Usage:
    python prepare_wisdm_data.py               # all models, phone + watch + all
    python prepare_wisdm_data.py --hart        # HART only
    python prepare_wisdm_data.py --limu        # LIMU-BERT only
    python prepare_wisdm_data.py --ssl         # ssl-wearables / ResNet only
    python prepare_wisdm_data.py --phone-only  # phone data only
    python prepare_wisdm_data.py --watch-only  # watch data only
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths =====================
BASE_DIR     = Path(__file__).resolve().parent
WISDM_RAW_DIR = (BASE_DIR / "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"
                 / "wisdm-dataset" / "wisdm-dataset" / "raw")
HART_DIR     = (BASE_DIR / "code"
                / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main"
                / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main")
LIMU_DIR     = (BASE_DIR / "code"
                / "LIMU-BERT_Experience-main" / "LIMU-BERT_Experience-main")
SSL_DIR      = (BASE_DIR / "code"
                / "ssl-wearables-main" / "ssl-wearables-main")

# ===================== WISDM Constants =====================
# Label indices align with HHAR no-bike: sit=0, stand=1, walk=2.
# stairs=3 is WISDM-only (no up/down distinction).
WISDM_ACT_MAP = {
    'D': 0,  # sitting
    'E': 1,  # standing
    'A': 2,  # walking
    'C': 3,  # stairs
}
WISDM_ACT_NAMES = ['sit', 'stand', 'walk', 'stairs']


# ===================== Utility Functions =====================
def resize_array(X, target_length, axis=1):
    orig_len = X.shape[axis]
    t_orig = np.linspace(0, 1, orig_len, endpoint=True)
    t_new  = np.linspace(0, 1, target_length, endpoint=True)
    return interp1d(t_orig, X, kind='linear', axis=axis,
                    assume_sorted=True)(t_new)


def load_wisdm_file(filepath):
    """Load a single WISDM raw txt file (CSV with ; line terminator)."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            try:
                rows.append((int(parts[0]),
                             parts[1].strip(),
                             int(parts[2]),
                             float(parts[3]),
                             float(parts[4]),
                             float(parts[5])))
            except (ValueError, IndexError):
                continue
    if not rows:
        return pd.DataFrame(columns=['subject','activity','timestamp','x','y','z'])
    return pd.DataFrame(rows, columns=['subject','activity','timestamp','x','y','z'])


def discover_subjects(device='phone'):
    """Return sorted list of subject IDs with accel data for *device*."""
    accel_dir = WISDM_RAW_DIR / device / 'accel'
    if not accel_dir.exists():
        return []
    sids = set()
    for f in accel_dir.glob('data_*_accel_*.txt'):
        try:
            sids.add(int(f.stem.split('_')[1]))
        except (IndexError, ValueError):
            pass
    return sorted(sids)


def _devices_for(device_type):
    if device_type == 'phone':
        return ['phone']
    if device_type == 'watch':
        return ['watch']
    return ['phone', 'watch']


def load_subject_activity(sid, device, act_code):
    """Load & align accel+gyro for one (subject, device, activity).

    Returns (data_6ch, data_3ch) as (T,6) and (T,3) float32, or (None,None).
    """
    afile = WISDM_RAW_DIR / device / 'accel' / f'data_{sid}_accel_{device}.txt'
    gfile = WISDM_RAW_DIR / device / 'gyro'  / f'data_{sid}_gyro_{device}.txt'
    if not afile.exists():
        return None, None

    adf = load_wisdm_file(afile)
    aa  = adf[adf['activity'] == act_code].sort_values('timestamp').reset_index(drop=True)
    if len(aa) < 10:
        return None, None

    acc_xyz = aa[['x','y','z']].values.astype(np.float32)
    acc_ts  = aa['timestamp'].values.astype(np.float64)

    gyro_interp = None
    if gfile.exists():
        gdf = load_wisdm_file(gfile)
        ga  = gdf[gdf['activity'] == act_code].sort_values('timestamp').reset_index(drop=True)
        if len(ga) >= 2:
            gyro_xyz = ga[['x','y','z']].values.astype(np.float32)
            gyro_ts  = ga['timestamp'].values.astype(np.float64)
            t0 = max(acc_ts[0], gyro_ts[0])
            t1 = min(acc_ts[-1], gyro_ts[-1])
            mask = (acc_ts >= t0) & (acc_ts <= t1)
            if mask.sum() >= 10:
                acc_xyz = acc_xyz[mask]
                acc_ts  = acc_ts[mask]
                gyro_interp = np.column_stack([
                    np.interp(acc_ts, gyro_ts, gyro_xyz[:, ch])
                    for ch in range(3)
                ]).astype(np.float32)

    data_3ch = acc_xyz
    data_6ch = (np.concatenate([acc_xyz, gyro_interp], axis=1)
                if gyro_interp is not None else None)
    return data_6ch, data_3ch


# =====================================================================
#  HART Data Preparation
# =====================================================================
def prepare_hart_wisdm(device_type, force=False):
    import hickle as hkl

    tag = device_type.upper()
    print("=" * 60)
    print(f"  Preparing WISDM -> HART [{tag}]")
    print("=" * 60)

    suffix_map = {'all': 'WISDM', 'phone': 'WISDM_phone', 'watch': 'WISDM_watch'}
    out_dir  = HART_DIR / "datasets" / "datasetStandardized" / suffix_map[device_type]
    map_name = f"wisdm_subject_map{'_' + device_type if device_type != 'all' else ''}.json"
    map_path = BASE_DIR / map_name

    if not force and out_dir.exists() and map_path.exists():
        with open(map_path) as f:
            sm = json.load(f)
        print(f"  Existing: {len(sm)} subjects -> {out_dir}")
        return

    TARGET_HZ = 50    # effective rate of HHAR after downsampling
    WINDOW    = 128
    STEP      = 64    # 50 % overlap

    devices = _devices_for(device_type)
    all_sids = set()
    for dev in devices:
        all_sids.update(discover_subjects(dev))
    all_sids = sorted(all_sids)
    print(f"  Subjects found: {len(all_sids)}")

    subj_entries = []  # (data, labels, sid)
    for sid in all_sids:
        segs, lbls = [], []
        for dev in devices:
            for act_code, act_label in WISDM_ACT_MAP.items():
                d6, _ = load_subject_activity(sid, dev, act_code)
                if d6 is None:
                    continue
                new_len = int(len(d6) * TARGET_HZ / 20.0)
                if new_len < WINDOW:
                    continue
                d6 = resize_array(d6[np.newaxis], new_len, axis=1)[0]
                for i in range(0, len(d6) - WINDOW + 1, STEP):
                    segs.append(d6[i:i + WINDOW])
                    lbls.append(act_label)
        if segs:
            subj_entries.append((np.array(segs, dtype=np.float32),
                                 np.array(lbls, dtype=np.int64), sid))

    if not subj_entries:
        print(f"  WARNING: no valid windows for [{tag}]"); return

    # Global z-score (WISDM devices are uniform, no per-device split)
    cat = np.vstack([e[0] for e in subj_entries])
    am, astd = cat[:,:,:3].mean(), cat[:,:,:3].std()
    gm, gstd = cat[:,:,3:].mean(), cat[:,:,3:].std()

    out_dir.mkdir(parents=True, exist_ok=True)
    subject_map = {}
    for i, (data, labels, sid) in enumerate(subj_entries):
        normed = data.copy()
        normed[:,:,:3] = (normed[:,:,:3] - am) / (astd + 1e-8)
        normed[:,:,3:] = (normed[:,:,3:] - gm) / (gstd + 1e-8)
        hkl.dump(normed, str(out_dir / f"UserData{i}.hkl"))
        hkl.dump(labels,  str(out_dir / f"UserLabel{i}.hkl"))
        subject_map[str(i)] = str(sid)

    with open(map_path, 'w') as f:
        json.dump(subject_map, f, indent=2)

    total = sum(len(e[1]) for e in subj_entries)
    print(f"  HART WISDM [{tag}]: {len(subj_entries)} subjects, {total} windows -> {out_dir}")
    print(f"  Subject map: {map_path}")


# =====================================================================
#  LIMU-BERT Data Preparation  (data at 20 Hz -- perfect match)
# =====================================================================
def prepare_limu_wisdm(device_type, force=False):
    tag = device_type.upper()
    print("\n" + "=" * 60)
    print(f"  Preparing WISDM -> LIMU-BERT [{tag}]")
    print("=" * 60)

    dir_map = {'phone': 'wisdm', 'watch': 'wisdm_watch', 'all': 'wisdm_all'}
    data_dir   = LIMU_DIR / "dataset" / dir_map[device_type]
    data_path  = data_dir / "data_20_120.npy"
    label_path = data_dir / "label_20_120.npy"

    if not force and data_path.exists() and label_path.exists():
        d = np.load(str(data_path)); l = np.load(str(label_path))
        print(f"  Existing: data={d.shape}, labels={l.shape}")
        return

    SEQ_LEN = 120  # 120 samples @ 20 Hz = 6 s (matches HHAR LIMU-BERT exactly)

    devices = _devices_for(device_type)
    all_sids = set()
    for dev in devices:
        all_sids.update(discover_subjects(dev))
    all_sids  = sorted(all_sids)
    sid2idx   = {s: i for i, s in enumerate(all_sids)}
    print(f"  Subjects found: {len(all_sids)}")

    data_seqs, label_seqs = [], []
    for sid in all_sids:
        for dev in devices:
            for act_code, act_label in WISDM_ACT_MAP.items():
                d6, _ = load_subject_activity(sid, dev, act_code)
                if d6 is None or len(d6) < SEQ_LEN:
                    continue
                n_seq = len(d6) // SEQ_LEN
                for s in range(n_seq):
                    seq = d6[s * SEQ_LEN:(s + 1) * SEQ_LEN]
                    data_seqs.append(seq)
                    lbl = np.zeros((SEQ_LEN, 3), dtype=np.float32)
                    lbl[:, 0] = sid2idx[sid]
                    lbl[:, 2] = act_label
                    label_seqs.append(lbl)

    if not data_seqs:
        print(f"  WARNING: no valid sequences for [{tag}]"); return

    data_arr  = np.array(data_seqs,  dtype=np.float32)
    label_arr = np.array(label_seqs, dtype=np.float32)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(data_path),  data_arr)
    np.save(str(label_path), label_arr)
    acts = sorted(np.unique(label_arr[:, 0, 2]).astype(int).tolist())
    print(f"  Saved: data={data_arr.shape}, labels={label_arr.shape}")
    print(f"  Subjects: {len(all_sids)}, Activity labels present: {acts}")


# =====================================================================
#  ssl-wearables / ResNet-baseline Data Preparation
# =====================================================================
def prepare_ssl_wisdm(device_type, force=False):
    tag = device_type.upper()
    print("\n" + "=" * 60)
    print(f"  Preparing WISDM -> ssl-wearables (3ch + 6ch) [{tag}]")
    print("=" * 60)

    dir_map = {'phone': 'wisdm', 'watch': 'wisdm_watch', 'all': 'wisdm_all'}
    out_dir = SSL_DIR / "data" / "downstream" / dir_map[device_type]
    out_dir.mkdir(parents=True, exist_ok=True)

    x3p, x6p = out_dir / "X.npy", out_dir / "X6.npy"
    yp, pp   = out_dir / "Y.npy", out_dir / "pid.npy"

    if not force and all(p.exists() for p in [x3p, x6p, yp, pp]):
        X3 = np.load(str(x3p)); X6 = np.load(str(x6p))
        print(f"  Existing: X(3ch)={X3.shape}, X6(6ch)={X6.shape}")
        return

    TARGET_HZ  = 30
    WINDOW_LEN = 300  # 10 s @ 30 Hz

    devices = _devices_for(device_type)
    all_sids = set()
    for dev in devices:
        all_sids.update(discover_subjects(dev))
    all_sids = sorted(all_sids)
    sid2idx  = {s: i for i, s in enumerate(all_sids)}
    print(f"  Subjects found: {len(all_sids)}")

    aX3, aX6, aY, aP = [], [], [], []
    for sid in all_sids:
        for dev in devices:
            for act_code, act_label in WISDM_ACT_MAP.items():
                d6, d3 = load_subject_activity(sid, dev, act_code)
                if d3 is None:
                    continue

                # Resample 20 Hz -> 30 Hz
                new_len = int(len(d3) * TARGET_HZ / 20.0)
                if new_len < WINDOW_LEN:
                    continue
                acc_rs = resize_array(d3[np.newaxis], new_len, axis=1)[0]
                acc_rs = np.clip(acc_rs / 9.8, -3.0, 3.0)

                six_rs = None
                if d6 is not None:
                    new_len6 = int(len(d6) * TARGET_HZ / 20.0)
                    if new_len6 >= WINDOW_LEN:
                        six_rs = resize_array(d6[np.newaxis], new_len6, axis=1)[0]
                        six_rs[:, :3] = np.clip(six_rs[:, :3] / 9.8, -3.0, 3.0)

                nw3 = len(acc_rs) // WINDOW_LEN
                nw6 = (len(six_rs) // WINDOW_LEN) if six_rs is not None else 0
                nw  = min(nw3, nw6) if six_rs is not None else nw3
                for w in range(nw):
                    s, e = w * WINDOW_LEN, (w + 1) * WINDOW_LEN
                    aX3.append(acc_rs[s:e])
                    if six_rs is not None:
                        aX6.append(six_rs[s:e])
                    else:
                        pad = np.zeros((WINDOW_LEN, 6), dtype=np.float32)
                        pad[:, :3] = acc_rs[s:e]
                        aX6.append(pad)
                    aY.append(act_label)
                    aP.append(sid2idx[sid])

    if not aX3:
        print(f"  WARNING: no valid windows for [{tag}]"); return

    X3 = np.array(aX3, dtype=np.float32)
    X6 = np.array(aX6, dtype=np.float32)
    Y  = np.array(aY,  dtype=np.int64)
    P  = np.array(aP,  dtype=np.int64)
    for path, arr in [(x3p, X3), (x6p, X6), (yp, Y), (pp, P)]:
        np.save(str(path), arr)
    print(f"  Saved: X(3ch)={X3.shape}, X6(6ch)={X6.shape}, Y={Y.shape}, pid={P.shape}")
    print(f"  Subjects: {len(all_sids)}")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Prepare WISDM data for cross-dataset evaluation (EXP5)')
    parser.add_argument('--hart',       action='store_true', help='HART only')
    parser.add_argument('--limu',       action='store_true', help='LIMU-BERT only')
    parser.add_argument('--ssl',        action='store_true', help='ssl-wearables / ResNet only')
    parser.add_argument('--phone-only', action='store_true', help='Phone data only')
    parser.add_argument('--watch-only', action='store_true', help='Watch data only')
    parser.add_argument('--force',      action='store_true', help='Regenerate even if files exist')
    args = parser.parse_args()

    do_all_models = not (args.hart or args.limu or args.ssl)
    do_phone = not args.watch_only
    do_watch = not args.phone_only

    if not WISDM_RAW_DIR.exists():
        print(f"ERROR: WISDM raw directory not found: {WISDM_RAW_DIR}")
        sys.exit(1)

    print(f"WISDM raw directory: {WISDM_RAW_DIR}")
    print(f"Devices : phone={do_phone}, watch={do_watch}")
    print(f"Models  : hart={do_all_models or args.hart}, "
          f"limu={do_all_models or args.limu}, ssl={do_all_models or args.ssl}\n")

    def _run(prepare_fn):
        if do_phone: prepare_fn('phone', force=args.force)
        if do_watch: prepare_fn('watch', force=args.force)
        if do_phone and do_watch: prepare_fn('all', force=args.force)

    if do_all_models or args.hart:
        _run(prepare_hart_wisdm)
    if do_all_models or args.limu:
        _run(prepare_limu_wisdm)
    if do_all_models or args.ssl:
        _run(prepare_ssl_wisdm)

    print("\n  Done!")


if __name__ == '__main__':
    main()
