#!/usr/bin/env python
"""
=============================================================================
PAMAP2 + SBHAR Data Preparation for Cross-Dataset Evaluation (EXP 5.1 / 6.1)
=============================================================================

Reads raw PAMAP2 (wrist IMU -> "watch") and SBHAR (phone IMU -> "phone") data,
harmonises activity labels to the HHAR 5-class scheme, and outputs data in the
exact formats expected by each model so that eval_cross_dataset.py can load
them identically to HHAR / WISDM data.

Two subject modes:
  Default  : PAMAP2 8 subjects + SBHAR all 30 subjects (full)
  --aligned: PAMAP2 8 subjects + SBHAR first 8 subjects (balanced 8+8)

Output directories (full / aligned):
  HART:          datasetStandardized/PAMAP2_SBHAR{,_phone,_watch}/           (.hkl)
                 datasetStandardized/PAMAP2_SBHAR_aligned{,_phone,_watch}/   (.hkl)
  LIMU-BERT:     dataset/pamap2_sbhar{,_watch,_all}/                         (.npy)
                 dataset/pamap2_sbhar_aligned{,_watch,_all}/                 (.npy)
  ssl-wearables: data/downstream/pamap2_sbhar{,_watch,_all}/                 (.npy)
                 data/downstream/pamap2_sbhar_aligned{,_watch,_all}/         (.npy)

Label harmonisation (-> HHAR 5-class no-bike):
    HHAR idx | HHAR name   | PAMAP2 raw ID          | SBHAR raw ID
    ---------|-------------|------------------------|-------------------------
    0        | sit         | 2  (sitting)           | 4  (SITTING)
    1        | stand       | 3  (standing)          | 5  (STANDING)
    2        | walk        | 4  (walking)           | 1  (WALKING)
    3        | stairsup    | 12 (ascending stairs)  | 2  (WALKING_UPSTAIRS)
    4        | stairsdown  | 13 (descending stairs) | 3  (WALKING_DOWNSTAIRS)

Device mapping:
    PAMAP2 -> watch  (hand/wrist IMU, 100 Hz, accel m/s², gyro rad/s)
    SBHAR  -> phone  (smartphone IMU,  50 Hz, accel g*9.81->m/s², gyro rad/s)

Usage:
    python prepare_pamap2_sbhar_data.py               # full (8+30), all models, phone+watch+all
    python prepare_pamap2_sbhar_data.py --aligned     # balanced (8+8), all models, phone+watch+all
    python prepare_pamap2_sbhar_data.py --hart        # HART only
    python prepare_pamap2_sbhar_data.py --limu        # LIMU-BERT only
    python prepare_pamap2_sbhar_data.py --ssl         # ssl-wearables / ResNet only
    python prepare_pamap2_sbhar_data.py --phone-only  # SBHAR (phone) data only
    python prepare_pamap2_sbhar_data.py --watch-only  # PAMAP2 (watch) data only
    python prepare_pamap2_sbhar_data.py --force       # regenerate even if files exist
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths =====================
BASE_DIR    = Path(__file__).resolve().parent
PAMAP2_ROOT = (BASE_DIR / "pamap2+physical+activity+monitoring"
               / "PAMAP2_Dataset" / "PAMAP2_Dataset")
SBHAR_ROOT  = BASE_DIR / "SBHAR"
HART_DIR    = (BASE_DIR / "code"
               / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main"
               / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main")
LIMU_DIR    = (BASE_DIR / "code"
               / "LIMU-BERT_Experience-main" / "LIMU-BERT_Experience-main")
SSL_DIR     = (BASE_DIR / "code"
               / "ssl-wearables-main" / "ssl-wearables-main")

# ===================== Label Harmonisation =====================
ACTIVITY_NAMES = ['sit', 'stand', 'walk', 'stairsup', 'stairsdown']

PAMAP2_TO_HHAR = {
    2:  0,   # sitting          -> sit
    3:  1,   # standing         -> stand
    4:  2,   # walking          -> walk
    12: 3,   # ascending stairs -> stairsup
    13: 4,   # descending stairs-> stairsdown
}

SBHAR_TO_HHAR = {
    4: 0,    # SITTING           -> sit
    5: 1,    # STANDING          -> stand
    1: 2,    # WALKING           -> walk
    2: 3,    # WALKING_UPSTAIRS  -> stairsup
    3: 4,    # WALKING_DOWNSTAIRS-> stairsdown
}

# PAMAP2 hand/wrist IMU columns (0-indexed in 54-column rows)
_PAMAP2_COLS_ACC  = [4, 5, 6]       # ±16 g accelerometer (m/s²)
_PAMAP2_COLS_GYRO = [10, 11, 12]    # gyroscope (rad/s)
_PAMAP2_ALL_COLS  = _PAMAP2_COLS_ACC + _PAMAP2_COLS_GYRO

# Subject 109 only contains activity 0 (transient) and 24 (rope jumping)
PAMAP2_SKIP_SUBJECTS = {109}

SBHAR_ALIGNED_COUNT = 8  # first N users for balanced 8+8 mode


# ===================== Utility =====================
def resize_array(X, target_length, axis=1):
    """Resample along *axis* via linear interpolation."""
    orig_len = X.shape[axis]
    if orig_len == target_length:
        return X
    t_orig = np.linspace(0, 1, orig_len, endpoint=True)
    t_new  = np.linspace(0, 1, target_length, endpoint=True)
    return interp1d(t_orig, X, kind='linear', axis=axis,
                    assume_sorted=True)(t_new)


# ===================== PAMAP2 Loader (watch) =====================
def load_pamap2_labeled():
    """Load PAMAP2 Protocol+Optional wrist IMU with activity labels.

    Returns {subject_id: [(data_6ch, hhar_label), ...]}.
    data_6ch is (T, 6) float32 [accel_xyz m/s² | gyro_xyz rad/s] at 100 Hz.
    """
    result = {}

    for folder in ("Protocol", "Optional"):
        data_dir = PAMAP2_ROOT / folder
        if not data_dir.exists():
            continue
        for dat_file in sorted(data_dir.glob("subject*.dat")):
            sid = int(dat_file.stem.replace("subject", ""))
            if sid in PAMAP2_SKIP_SUBJECTS:
                continue

            rows = []
            with open(dat_file, "r") as fh:
                for line in fh:
                    vals = line.strip().split()
                    if len(vals) < 54:
                        continue
                    try:
                        rows.append([float(v) for v in vals])
                    except ValueError:
                        continue
            if not rows:
                continue

            arr = np.array(rows, dtype=np.float64)
            activities = arr[:, 1].astype(int)

            for pamap2_id, hhar_label in PAMAP2_TO_HHAR.items():
                mask = activities == pamap2_id
                if mask.sum() < 10:
                    continue
                indices = np.where(mask)[0]
                segments = np.split(indices,
                                    np.where(np.diff(indices) != 1)[0] + 1)
                for seg_idx in segments:
                    if len(seg_idx) < 10:
                        continue
                    data6 = arr[seg_idx][:, _PAMAP2_ALL_COLS].astype(np.float32)
                    good = ~np.isnan(data6).any(axis=1)
                    data6 = data6[good]
                    if len(data6) < 10:
                        continue
                    result.setdefault(sid, []).append((data6, hhar_label))

    return result


# ===================== SBHAR Loader (phone) =====================
def _parse_sbhar_labels():
    """Parse SBHAR/RawData/labels.txt.

    Returns list of (experiment_id, user_id, activity_id, start, end).
    Indices in labels.txt are 1-based inclusive.
    """
    labels_file = SBHAR_ROOT / "RawData" / "labels.txt"
    entries = []
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            entries.append(tuple(int(p) for p in parts))
    return entries


def load_sbhar_labeled(max_subjects=None):
    """Load SBHAR raw acc+gyro with activity labels (5-class only).

    Args:
        max_subjects: If set, keep only the first N user IDs (sorted).

    Returns {user_id: [(data_6ch, hhar_label), ...]}.
    data_6ch is (T, 6) float32 [accel_xyz m/s² | gyro_xyz rad/s] at 50 Hz.
    SBHAR accel is natively in g; multiplied by 9.81 here to match PAMAP2/HHAR.
    """
    raw_dir = SBHAR_ROOT / "RawData"
    labels = _parse_sbhar_labels()

    allowed_users = None
    if max_subjects is not None:
        all_uids = sorted({uid for _, uid, act, _, _ in labels
                           if act in SBHAR_TO_HHAR})
        allowed_users = set(all_uids[:max_subjects])

    file_cache: dict[tuple[int, int], tuple] = {}

    def _get_raw(exp_id, user_id):
        key = (exp_id, user_id)
        if key in file_cache:
            return file_cache[key]
        acc_f  = raw_dir / f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
        gyro_f = raw_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"
        acc = gyro = None
        if acc_f.exists():
            a = np.loadtxt(str(acc_f), dtype=np.float32)
            if a.ndim == 2 and a.shape[1] == 3:
                acc = a
        if gyro_f.exists():
            g = np.loadtxt(str(gyro_f), dtype=np.float32)
            if g.ndim == 2 and g.shape[1] == 3:
                gyro = g
        file_cache[key] = (acc, gyro)
        return acc, gyro

    result = {}
    for exp_id, user_id, act_id, start, end in labels:
        if act_id not in SBHAR_TO_HHAR:
            continue
        if allowed_users is not None and user_id not in allowed_users:
            continue
        hhar_label = SBHAR_TO_HHAR[act_id]

        acc, gyro = _get_raw(exp_id, user_id)
        if acc is None or gyro is None:
            continue

        # 1-based inclusive -> 0-based Python slice
        s, e = start - 1, end
        if s < 0 or e > len(acc) or e > len(gyro):
            continue

        acc_seg  = acc[s:e].copy()
        gyro_seg = gyro[s:e].copy()
        if len(acc_seg) < 10:
            continue

        acc_seg *= 9.81  # g -> m/s²
        data6 = np.concatenate([acc_seg, gyro_seg], axis=1)
        result.setdefault(user_id, []).append((data6, hhar_label))

    return result


# ===================== Device Routing =====================
def _devices_for(device_type):
    if device_type == 'phone':
        return ['phone']
    if device_type == 'watch':
        return ['watch']
    return ['phone', 'watch']


def _load_for_device(device_type, aligned=False):
    """Load labeled data for the given device split.

    Args:
        aligned: If True, limit SBHAR to first 8 users (balanced 8+8).

    Returns {subject_key: [(data_6ch, label), ...]}.
    Keys are prefixed to avoid collisions: 'pamap2_{sid}' or 'sbhar_{uid}'.
    """
    combined = {}
    devices = _devices_for(device_type)

    if 'watch' in devices:
        pamap2 = load_pamap2_labeled()
        for sid, segs in pamap2.items():
            combined[f'pamap2_{sid}'] = segs

    if 'phone' in devices:
        sbhar = load_sbhar_labeled(
            max_subjects=SBHAR_ALIGNED_COUNT if aligned else None)
        for uid, segs in sbhar.items():
            combined[f'sbhar_{uid}'] = segs

    return combined


def _native_hz(subject_key):
    """Return native sampling rate based on subject key prefix."""
    return 100.0 if subject_key.startswith('pamap2_') else 50.0


# =====================================================================
#  HART Data Preparation
# =====================================================================
def prepare_hart(device_type, force=False, aligned=False):
    import hickle as hkl

    mode_tag = "aligned 8+8" if aligned else "full 8+30"
    tag = device_type.upper()
    print("=" * 60)
    print(f"  Preparing PAMAP2+SBHAR -> HART [{tag}] ({mode_tag})")
    print("=" * 60)

    _aln = '_aligned' if aligned else ''
    suffix_map = {
        'all':   f'PAMAP2_SBHAR{_aln}',
        'phone': f'PAMAP2_SBHAR{_aln}_phone',
        'watch': f'PAMAP2_SBHAR{_aln}_watch',
    }
    out_dir  = HART_DIR / "datasets" / "datasetStandardized" / suffix_map[device_type]
    map_name = f"pamap2_sbhar{_aln}_subject_map{'_' + device_type if device_type != 'all' else ''}.json"
    map_path = BASE_DIR / map_name

    if not force and out_dir.exists() and map_path.exists():
        with open(map_path) as f:
            sm = json.load(f)
        print(f"  Existing: {len(sm)} subjects -> {out_dir}")
        return

    TARGET_HZ = 50
    WINDOW    = 128
    STEP      = 64    # 50 % overlap

    all_data = _load_for_device(device_type, aligned=aligned)
    subject_keys = sorted(all_data.keys())
    print(f"  Subjects found: {len(subject_keys)}")

    subj_entries = []
    for skey in subject_keys:
        hz = _native_hz(skey)
        segs, lbls = [], []
        for data6, label in all_data[skey]:
            new_len = int(len(data6) * TARGET_HZ / hz)
            if new_len < WINDOW:
                continue
            d6 = resize_array(data6[np.newaxis], new_len, axis=1)[0]
            for i in range(0, len(d6) - WINDOW + 1, STEP):
                segs.append(d6[i:i + WINDOW])
                lbls.append(label)

        if segs:
            subj_entries.append((np.array(segs, dtype=np.float32),
                                 np.array(lbls, dtype=np.int64), skey))

    if not subj_entries:
        print(f"  WARNING: no valid windows for [{tag}]")
        return

    cat = np.vstack([e[0] for e in subj_entries])
    am, astd = cat[:, :, :3].mean(), cat[:, :, :3].std()
    gm, gstd = cat[:, :, 3:].mean(), cat[:, :, 3:].std()

    out_dir.mkdir(parents=True, exist_ok=True)
    subject_map = {}
    for i, (data, labels, skey) in enumerate(subj_entries):
        normed = data.copy()
        normed[:, :, :3] = (normed[:, :, :3] - am) / (astd + 1e-8)
        normed[:, :, 3:] = (normed[:, :, 3:] - gm) / (gstd + 1e-8)
        hkl.dump(normed, str(out_dir / f"UserData{i}.hkl"))
        hkl.dump(labels, str(out_dir / f"UserLabel{i}.hkl"))
        subject_map[str(i)] = skey

    with open(map_path, 'w') as f:
        json.dump(subject_map, f, indent=2)

    total = sum(len(e[1]) for e in subj_entries)
    print(f"  HART [{tag}]: {len(subj_entries)} subjects, {total} windows -> {out_dir}")
    print(f"  Subject map: {map_path}")


# =====================================================================
#  LIMU-BERT Data Preparation (target 20 Hz, seq_len=120 = 6 s)
# =====================================================================
def prepare_limu(device_type, force=False, aligned=False):
    mode_tag = "aligned 8+8" if aligned else "full 8+30"
    tag = device_type.upper()
    print("\n" + "=" * 60)
    print(f"  Preparing PAMAP2+SBHAR -> LIMU-BERT [{tag}] ({mode_tag})")
    print("=" * 60)

    _aln = '_aligned' if aligned else ''
    dir_map = {
        'phone': f'pamap2_sbhar{_aln}',
        'watch': f'pamap2_sbhar{_aln}_watch',
        'all':   f'pamap2_sbhar{_aln}_all',
    }
    data_dir   = LIMU_DIR / "dataset" / dir_map[device_type]
    data_path  = data_dir / "data_20_120.npy"
    label_path = data_dir / "label_20_120.npy"

    if not force and data_path.exists() and label_path.exists():
        d = np.load(str(data_path))
        l = np.load(str(label_path))
        print(f"  Existing: data={d.shape}, labels={l.shape}")
        return

    SEQ_LEN   = 120   # 120 samples @ 20 Hz = 6 s
    TARGET_HZ = 20.0

    all_data = _load_for_device(device_type, aligned=aligned)
    subject_keys = sorted(all_data.keys())
    sid2idx = {s: i for i, s in enumerate(subject_keys)}
    print(f"  Subjects found: {len(subject_keys)}")

    data_seqs, label_seqs = [], []
    for skey in subject_keys:
        hz = _native_hz(skey)
        for data6, label in all_data[skey]:
            new_len = int(len(data6) * TARGET_HZ / hz)
            if new_len < SEQ_LEN:
                continue
            d6 = resize_array(data6[np.newaxis], new_len, axis=1)[0]
            n_seq = len(d6) // SEQ_LEN
            for s in range(n_seq):
                seq = d6[s * SEQ_LEN:(s + 1) * SEQ_LEN]
                data_seqs.append(seq)
                lbl = np.zeros((SEQ_LEN, 3), dtype=np.float32)
                lbl[:, 0] = sid2idx[skey]
                lbl[:, 2] = label
                label_seqs.append(lbl)

    if not data_seqs:
        print(f"  WARNING: no valid sequences for [{tag}]")
        return

    data_arr  = np.array(data_seqs,  dtype=np.float32)
    label_arr = np.array(label_seqs, dtype=np.float32)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(data_path),  data_arr)
    np.save(str(label_path), label_arr)
    acts = sorted(np.unique(label_arr[:, 0, 2]).astype(int).tolist())
    print(f"  Saved: data={data_arr.shape}, labels={label_arr.shape}")
    print(f"  Subjects: {len(subject_keys)}, Activity labels present: {acts}")


# =====================================================================
#  ssl-wearables / ResNet-baseline Data Preparation
# =====================================================================
def prepare_ssl(device_type, force=False, aligned=False):
    mode_tag = "aligned 8+8" if aligned else "full 8+30"
    tag = device_type.upper()
    print("\n" + "=" * 60)
    print(f"  Preparing PAMAP2+SBHAR -> ssl-wearables (3ch + 6ch) [{tag}] ({mode_tag})")
    print("=" * 60)

    _aln = '_aligned' if aligned else ''
    dir_map = {
        'phone': f'pamap2_sbhar{_aln}',
        'watch': f'pamap2_sbhar{_aln}_watch',
        'all':   f'pamap2_sbhar{_aln}_all',
    }
    out_dir = SSL_DIR / "data" / "downstream" / dir_map[device_type]
    out_dir.mkdir(parents=True, exist_ok=True)

    x3p, x6p = out_dir / "X.npy", out_dir / "X6.npy"
    yp, pp   = out_dir / "Y.npy", out_dir / "pid.npy"

    if not force and all(p.exists() for p in [x3p, x6p, yp, pp]):
        X3 = np.load(str(x3p))
        X6 = np.load(str(x6p))
        print(f"  Existing: X(3ch)={X3.shape}, X6(6ch)={X6.shape}")
        return

    TARGET_HZ  = 30
    WINDOW_LEN = 300   # 10 s @ 30 Hz

    all_data = _load_for_device(device_type, aligned=aligned)
    subject_keys = sorted(all_data.keys())
    sid2idx = {s: i for i, s in enumerate(subject_keys)}
    print(f"  Subjects found: {len(subject_keys)}")

    aX3, aX6, aY, aP = [], [], [], []
    for skey in subject_keys:
        hz = _native_hz(skey)
        for data6, label in all_data[skey]:
            new_len = int(len(data6) * TARGET_HZ / hz)
            if new_len < WINDOW_LEN:
                continue
            d_rs = resize_array(data6[np.newaxis], new_len, axis=1)[0]

            # accel: m/s² -> g, clipped to [-3, 3] (ssl-wearables convention)
            acc_g  = np.clip(d_rs[:, :3] / 9.81, -3.0, 3.0)
            six_rs = np.concatenate([acc_g, d_rs[:, 3:]], axis=1)

            nw = len(d_rs) // WINDOW_LEN
            for w in range(nw):
                s, e = w * WINDOW_LEN, (w + 1) * WINDOW_LEN
                aX3.append(acc_g[s:e])
                aX6.append(six_rs[s:e])
                aY.append(label)
                aP.append(sid2idx[skey])

    if not aX3:
        print(f"  WARNING: no valid windows for [{tag}]")
        return

    X3 = np.array(aX3, dtype=np.float32)
    X6 = np.array(aX6, dtype=np.float32)
    Y  = np.array(aY,  dtype=np.int64)
    P  = np.array(aP,  dtype=np.int64)
    for path, arr in [(x3p, X3), (x6p, X6), (yp, Y), (pp, P)]:
        np.save(str(path), arr)
    print(f"  Saved: X(3ch)={X3.shape}, X6(6ch)={X6.shape}, Y={Y.shape}, pid={P.shape}")
    print(f"  Subjects: {len(subject_keys)}")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Prepare PAMAP2+SBHAR data for cross-dataset evaluation (EXP 5.1/6.1)')
    parser.add_argument('--hart',       action='store_true', help='HART only')
    parser.add_argument('--limu',       action='store_true', help='LIMU-BERT only')
    parser.add_argument('--ssl',        action='store_true', help='ssl-wearables / ResNet only')
    parser.add_argument('--phone-only', action='store_true', help='SBHAR (phone) data only')
    parser.add_argument('--watch-only', action='store_true', help='PAMAP2 (watch) data only')
    parser.add_argument('--aligned',    action='store_true',
                        help='Balanced 8+8 mode (SBHAR limited to first 8 users)')
    parser.add_argument('--force',      action='store_true', help='Regenerate even if files exist')
    args = parser.parse_args()

    do_all_models = not (args.hart or args.limu or args.ssl)
    do_phone = not args.watch_only
    do_watch = not args.phone_only

    missing = []
    if do_watch and not PAMAP2_ROOT.exists():
        missing.append(f"PAMAP2: {PAMAP2_ROOT}")
    if do_phone and not SBHAR_ROOT.exists():
        missing.append(f"SBHAR: {SBHAR_ROOT}")
    if missing:
        for m in missing:
            print(f"ERROR: not found: {m}")
        sys.exit(1)

    mode_str = f"aligned 8+8 (SBHAR first {SBHAR_ALIGNED_COUNT})" if args.aligned else "full 8+30"
    print(f"PAMAP2 root : {PAMAP2_ROOT}")
    print(f"SBHAR root  : {SBHAR_ROOT}")
    print(f"Subject mode: {mode_str}")
    print(f"Devices     : phone(SBHAR)={do_phone}, watch(PAMAP2)={do_watch}")
    print(f"Models      : hart={do_all_models or args.hart}, "
          f"limu={do_all_models or args.limu}, ssl={do_all_models or args.ssl}\n")

    def _run(prepare_fn):
        if do_phone:
            prepare_fn('phone', force=args.force, aligned=args.aligned)
        if do_watch:
            prepare_fn('watch', force=args.force, aligned=args.aligned)
        if do_phone and do_watch:
            prepare_fn('all', force=args.force, aligned=args.aligned)

    if do_all_models or args.hart:
        _run(prepare_hart)
    if do_all_models or args.limu:
        _run(prepare_limu)
    if do_all_models or args.ssl:
        _run(prepare_ssl)

    print("\n  Done!")


if __name__ == '__main__':
    main()
