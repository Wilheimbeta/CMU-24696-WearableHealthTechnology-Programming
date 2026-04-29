#!/usr/bin/env python
"""
=============================================================================
Unified HHAR Data Preparation for HART, LIMU-BERT, and ssl-wearables
=============================================================================

Reads raw HHAR CSV files and prepares data with SEPARATE outputs for
smartphone and smartwatch:

  HART:
    datasetStandardized/HHAR/            (all devices - backward compat)
    datasetStandardized/HHAR_phone/      (smartphone only)
    datasetStandardized/HHAR_watch/      (smartwatch only)

  LIMU-BERT:
    dataset/hhar/                        (phone - existing)
    dataset/hhar_watch/                  (watch - new)

  ssl-wearables:
    data/downstream/hhar/                (phone - existing)
    data/downstream/hhar_watch/          (watch - new)

Usage:
    python prepare_hhar_data.py              # all models, phone + watch
    python prepare_hhar_data.py --hart       # HART only
    python prepare_hhar_data.py --limu       # LIMU-BERT only
    python prepare_hhar_data.py --ssl        # ssl-wearables only
    python prepare_hhar_data.py --phone-only # phone data only
    python prepare_hhar_data.py --watch-only # watch data only
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ===================== Paths =====================
BASE_DIR     = Path(__file__).resolve().parent
HHAR_RAW_DIR = (BASE_DIR / "heterogeneity+activity+recognition"
                / "Activity recognition exp" / "Activity recognition exp")
HART_DIR     = (BASE_DIR / "code"
                / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main"
                / "Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main")
LIMU_DIR     = (BASE_DIR / "code"
                / "LIMU-BERT_Experience-main" / "LIMU-BERT_Experience-main")
SSL_DIR      = (BASE_DIR / "code"
                / "ssl-wearables-main" / "ssl-wearables-main")

# ===================== HHAR Constants =====================
ACTIVITY_LABELS = ['sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'bike']
ACTIVITY_LABELS_NOBIKE = ['sit', 'stand', 'walk', 'stairsup', 'stairsdown']
USER_NAMES      = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
DEVICE_NAMES    = ['nexus4', 'lgwatch', 's3', 's3mini', 'gear', 'samsungold']

DEVICE_INFO = {
    'nexus4':     {'idx': 0, 'type': 'phone', 'window': 512, 'downsample': 4},
    'lgwatch':    {'idx': 1, 'type': 'watch', 'window': 512, 'downsample': 4},
    's3':         {'idx': 2, 'type': 'phone', 'window': 384, 'downsample': 3},
    's3mini':     {'idx': 3, 'type': 'phone', 'window': 256, 'downsample': 2},
    'gear':       {'idx': 4, 'type': 'watch', 'window': 256, 'downsample': 2},
    'samsungold': {'idx': 5, 'type': 'phone', 'window': 128, 'downsample': 1},
}

PHONE_DEVICES = [d for d in DEVICE_NAMES if DEVICE_INFO[d]['type'] == 'phone']
WATCH_DEVICES = [d for d in DEVICE_NAMES if DEVICE_INFO[d]['type'] == 'watch']


# ===================== Utility Functions =====================
def consecutive_segments(indices, min_length):
    if len(indices) == 0:
        return []
    splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [seg for seg in splits if len(seg) >= min_length]

def segment_array(data, window_size, step):
    step = int(step)
    segments = []
    for i in range(0, data.shape[0] - window_size + 1, step):
        segments.append(data[i:i + window_size, :])
    return np.array(segments) if segments else None

def downsample_lowpass(data, factor):
    channels = []
    for ch in range(data.shape[2]):
        channels.append(sp_signal.decimate(data[:, :, ch], factor))
    return np.stack(channels, axis=2)

def resize_array(X, target_length, axis=1):
    orig_len = X.shape[axis]
    t_orig = np.linspace(0, 1, orig_len, endpoint=True)
    t_new  = np.linspace(0, 1, target_length, endpoint=True)
    return interp1d(t_orig, X, kind='linear', axis=axis,
                    assume_sorted=True)(t_new)

def get_devices_for_type(device_type):
    if device_type == 'all':
        return list(DEVICE_NAMES)
    elif device_type == 'phone':
        return PHONE_DEVICES
    elif device_type == 'watch':
        return WATCH_DEVICES
    raise ValueError(f"Unknown device_type: {device_type}")

def load_hhar_csvs(device_type='all'):
    """Load HHAR CSV files for the given device type."""
    print(f"  Loading CSV files for device_type='{device_type}'...")
    if device_type == 'phone':
        acc  = pd.read_csv(HHAR_RAW_DIR / "Phones_accelerometer.csv")
        gyro = pd.read_csv(HHAR_RAW_DIR / "Phones_gyroscope.csv")
    elif device_type == 'watch':
        acc  = pd.read_csv(HHAR_RAW_DIR / "Watch_accelerometer.csv")
        gyro = pd.read_csv(HHAR_RAW_DIR / "Watch_gyroscope.csv")
    else:
        pa = pd.read_csv(HHAR_RAW_DIR / "Phones_accelerometer.csv")
        pg = pd.read_csv(HHAR_RAW_DIR / "Phones_gyroscope.csv")
        wa = pd.read_csv(HHAR_RAW_DIR / "Watch_accelerometer.csv")
        wg = pd.read_csv(HHAR_RAW_DIR / "Watch_gyroscope.csv")
        acc  = pd.concat([pa, wa], ignore_index=True)
        gyro = pd.concat([pg, wg], ignore_index=True)
    print(f"  Loaded {len(acc)} acc rows, {len(gyro)} gyro rows")
    return acc, gyro


# =====================================================================
#  HART Data Preparation
# =====================================================================
def prepare_hart_data(device_type='all', no_bike=False, channels=6):
    import hickle as hkl
    if channels not in (3, 6):
        raise ValueError(f"HART channels must be 3 or 6, got {channels}")
    activity_labels = ACTIVITY_LABELS_NOBIKE if no_bike else ACTIVITY_LABELS
    nobike_tag = '_nobike' if no_bike else ''
    ch_tag = '_3ch' if channels == 3 else ''
    tag = (f"{device_type.upper()} {channels}ch" +
           (' (no bike)' if no_bike else ''))
    print("=" * 60)
    print(f"  Preparing HART data [{tag}]")
    print("=" * 60)

    base_suffix_map = {'all': 'HHAR', 'phone': 'HHAR_phone', 'watch': 'HHAR_watch'}
    suffix_map = {k: f"{v}{ch_tag}{nobike_tag}" for k, v in base_suffix_map.items()}
    out_dir  = HART_DIR / "datasets" / "datasetStandardized" / suffix_map[device_type]
    map_name = f"hart_subject_map{'_' + device_type if device_type != 'all' else ''}{ch_tag}{nobike_tag}.json"
    map_path = BASE_DIR / map_name

    acc_df, gyro_df = load_hhar_csvs(device_type)
    acc_np, gyro_np = acc_df.values, gyro_df.values
    target_devices = get_devices_for_type(device_type)
    print(f"  Target devices: {target_devices}")

    all_data, all_labels, dev_index, subject_map = {}, {}, {}, {}
    device_client_lists = {DEVICE_INFO[d]['idx']: [] for d in target_devices}
    cc = 0

    for dev_name in target_devices:
        di = DEVICE_INFO[dev_name]
        win, ds_fac = di['window'], di['downsample']
        step = win // 2
        for usr_name in USER_NAMES:
            acc_mask = (acc_np[:, 6] == usr_name) & (acc_np[:, 7] == dev_name)
            user_acc = acc_np[acc_mask]
            if len(user_acc) == 0:
                continue
            user_gyro = None
            if channels == 6:
                gyro_mask = (gyro_np[:, 6] == usr_name) & (gyro_np[:, 7] == dev_name)
                user_gyro = gyro_np[gyro_mask]
                if len(user_gyro) == 0:
                    # No gyro data for this user+device — skip (need both acc+gyro)
                    continue

            proc_data, proc_labels = [], []
            for cls_idx, cls_name in enumerate(activity_labels):
                if channels == 6:
                    ref = user_acc if len(user_acc) <= len(user_gyro) else user_gyro
                    cls_indices = np.where(ref[:, 9] == cls_name)[0]
                else:
                    cls_indices = np.where(user_acc[:, 9] == cls_name)[0]
                for seg in consecutive_segments(cls_indices, win):
                    seg_a = segment_array(user_acc[seg][:, 3:6].astype(np.float32), win, step)
                    if channels == 6:
                        seg_g = segment_array(user_gyro[seg][:, 3:6].astype(np.float32), win, step)
                        if seg_a is not None and seg_g is not None:
                            n = min(len(seg_a), len(seg_g))
                            if n > 0:
                                proc_data.append(np.concatenate([seg_a[:n], seg_g[:n]], axis=2))
                                proc_labels.append(np.full(n, cls_idx, dtype=int))
                    elif seg_a is not None and len(seg_a) > 0:
                        proc_data.append(seg_a)
                        proc_labels.append(np.full(len(seg_a), cls_idx, dtype=int))
            if not proc_data:
                continue
            tmp = np.vstack(proc_data).astype(np.float32)
            if ds_fac > 1:
                tmp = downsample_lowpass(tmp, ds_fac)
            all_data[cc]   = tmp
            all_labels[cc] = np.hstack(proc_labels)
            dev_index[cc]  = np.full(len(all_labels[cc]), di['idx'])
            subject_map[cc] = usr_name
            device_client_lists[di['idx']].append(cc)
            cc += 1

    # Remove subjects that don't have all activities even across all devices.
    # (Check per-subject, not per-client, so h@lgwatch missing "sit" is OK
    #  as long as h@gear has "sit".)
    subject_activities = {}
    for idx, usr in subject_map.items():
        if usr not in subject_activities:
            subject_activities[usr] = set()
        subject_activities[usr].update(np.unique(all_labels[idx]).tolist())

    incomplete_subjects = {usr for usr, acts in subject_activities.items()
                           if len(acts) < len(activity_labels)}
    if incomplete_subjects:
        print(f"    Subjects with incomplete activities (removing): {incomplete_subjects}")

    remove = [idx for idx, usr in subject_map.items()
              if usr in incomplete_subjects]
    for idx in remove:
        usr = subject_map[idx]
        print(f"    Removing client {idx} (subject '{usr}', "
              f"only {len(np.unique(all_labels[idx]))} activities in this client, "
              f"subject total: {len(subject_activities[usr])} activities)")
        del all_data[idx]; del all_labels[idx]; del dev_index[idx]; del subject_map[idx]
        for dl in device_client_lists.values():
            if idx in dl: dl.remove(idx)

    if not all_data:
        print(f"  WARNING: No valid clients for [{tag}]")
        return {}

    # Re-index
    sorted_keys = sorted(all_data.keys())
    data_list  = [all_data[k] for k in sorted_keys]
    label_list = [all_labels[k] for k in sorted_keys]
    dev_list   = [dev_index[k] for k in sorted_keys]
    old2new = {old: new for new, old in enumerate(sorted_keys)}
    new_sub_map = {str(old2new[k]): subject_map[k] for k in sorted_keys}
    new_dev_lists = {d: [old2new[x] for x in lst if x in old2new]
                     for d, lst in device_client_lists.items()}

    # Per-device normalisation
    normalised_data = [None] * len(data_list)
    for d_name in target_devices:
        d_idx = DEVICE_INFO[d_name]['idx']
        clients = new_dev_lists.get(d_idx, [])
        if not clients:
            continue
        dev_all = np.vstack([data_list[c] for c in clients])
        am, astd = np.mean(dev_all[:,:,:3]), np.std(dev_all[:,:,:3])
        gm, gstd = None, None
        if channels == 6 and dev_all.shape[2] > 3:
            gm, gstd = np.mean(dev_all[:,:,3:]), np.std(dev_all[:,:,3:])
        for c in clients:
            d = data_list[c].copy()
            d[:,:,:3] = (d[:,:,:3] - am) / (astd + 1e-8)
            if channels == 6 and d.shape[2] > 3:
                d[:,:,3:] = (d[:,:,3:] - gm) / (gstd + 1e-8)
            normalised_data[c] = d

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(data_list)):
        hkl.dump(normalised_data[i], str(out_dir / f"UserData{i}.hkl"))
        hkl.dump(label_list[i], str(out_dir / f"UserLabel{i}.hkl"))
    hkl.dump(np.array(dev_list, dtype=object), str(out_dir / "deviceIndex.hkl"))
    with open(map_path, 'w') as f:
        json.dump(new_sub_map, f, indent=2)

    print(f"  HART [{tag}]: {len(data_list)} clients -> {out_dir}")
    print(f"  Subject mapping: {map_path}")
    print(f"  Subjects: {sorted(set(new_sub_map.values()))}")
    return new_sub_map


# =====================================================================
#  LIMU-BERT Data Preparation
# =====================================================================
def prepare_limu_data(device_type='phone', no_bike=False, force=False):
    nobike_tag = '_nobike' if no_bike else ''
    tag = device_type.upper() + (' (no bike)' if no_bike else '')
    print("\n" + "=" * 60)
    print(f"  Preparing LIMU-BERT data [{tag}]")
    print("=" * 60)

    dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}'}
    data_dir   = LIMU_DIR / "dataset" / dir_map[device_type]
    data_path  = data_dir / "data_20_120.npy"
    label_path = data_dir / "label_20_120.npy"

    if not force and data_path.exists() and label_path.exists():
        data = np.load(str(data_path)); labels = np.load(str(label_path))
        print(f"  Existing: data={data.shape}, labels={labels.shape}")
        print(f"  Users: {np.unique(labels[:,0,0])}, Activities: {np.unique(labels[:,0,2])}")
        return
    if force and data_path.exists():
        print(f"  --force: removing old files and regenerating...")

    print("  Creating from raw CSVs...")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = "Phones" if device_type == 'phone' else "Watch"
    accs  = pd.read_csv(HHAR_RAW_DIR / f"{csv_prefix}_accelerometer.csv") \
                .sort_values('Creation_Time').reset_index(drop=True)
    gyros = pd.read_csv(HHAR_RAW_DIR / f"{csv_prefix}_gyroscope.csv") \
                .sort_values('Creation_Time').reset_index(drop=True)
    if no_bike:
        accs  = accs[accs['gt'] != 'bike'].reset_index(drop=True)
        gyros = gyros[gyros['gt'] != 'bike'].reset_index(drop=True)

    WINDOW_TIME, SEQ_LEN = 50, 120

    def extract_sensor(df, time_idx, time_tag, window_ms):
        idx = time_idx
        while idx < len(df) and abs(df.iloc[idx]['Creation_Time'] - time_tag) < window_ms:
            idx += 1
        if idx == time_idx:
            return None, idx
        slc = df.iloc[time_idx:idx]
        if slc['User'].nunique() > 1 or slc['gt'].nunique() > 1:
            return None, idx
        sensor = slc[['x', 'y', 'z']].mean().values
        label  = slc[['User', 'Model', 'gt']].iloc[0].values
        return np.concatenate([sensor, label]), idx

    time_tag   = min(accs.iloc[0]['Creation_Time'], gyros.iloc[0]['Creation_Time'])
    time_index = [0, 0]
    window_num = 0
    data_seqs, data_temp = [], []

    print("  Processing (this may take a while)...")
    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc_r, ti_a  = extract_sensor(accs, time_index[0], time_tag, WINDOW_TIME * 1e6)
        gyro_r, ti_g = extract_sensor(gyros, time_index[1], time_tag, WINDOW_TIME * 1e6)
        time_index = [ti_a, ti_g]
        if (acc_r is not None and gyro_r is not None
                and np.all(acc_r[-3:] == gyro_r[-3:])):
            time_tag += WINDOW_TIME * 1e6
            window_num += 1
            data_temp.append(np.concatenate([acc_r[:-3], gyro_r[:-3], acc_r[-3:]]))
            if window_num == SEQ_LEN:
                data_seqs.append(np.array(data_temp))
                data_temp.clear()
                window_num = 0
        else:
            if window_num > 0:
                data_temp.clear(); window_num = 0
            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(accs.iloc[time_index[0]]['Creation_Time'],
                               gyros.iloc[time_index[1]]['Creation_Time'])
            else:
                break

    if not data_seqs:
        print(f"  WARNING: No valid sequences for [{tag}]"); return

    raw = np.array(data_seqs)
    labels_raw = raw[:, :, -3:].astype(str)

    # Extract device model indices BEFORE dynamic encoding overwrites col 1.
    # Each 120-step sequence has the same Model throughout.
    device_model_names = labels_raw[:, 0, 1]  # (N,) string device names
    device_idx_map = {d: DEVICE_INFO[d]['idx'] for d in DEVICE_NAMES}
    D = np.array([device_idx_map.get(n, -1) for n in device_model_names],
                 dtype=np.int64)

    # Use FIXED mappings for User (col 0) and Activity (col 2) so that
    # subject/activity indices are consistent with ssl-wearables and HART,
    # regardless of which users/activities actually appear in the data.
    activity_labels = ACTIVITY_LABELS_NOBIKE if no_bike else ACTIVITY_LABELS
    fixed_maps = {
        0: {u: str(i) for i, u in enumerate(USER_NAMES)},       # User
        2: {a: str(i) for i, a in enumerate(activity_labels)},   # Activity
    }
    for col in range(3):
        if col in fixed_maps:
            for orig, mapped in fixed_maps[col].items():
                mask = labels_raw[:, :, col] == orig
                labels_raw[:, :, col][mask] = mapped
        else:
            # col 1 = Model (device) — keep dynamic encoding
            col_vals = labels_raw[:, :, col]
            for i, u in enumerate(np.unique(col_vals)):
                col_vals[col_vals == u] = str(i)

    np.save(str(data_path), raw[:, :, :6].astype(np.float32))
    np.save(str(label_path), labels_raw.astype(np.float32))
    dev_path = data_dir / "D.npy"
    np.save(str(dev_path), D)
    dev_present = sorted(set(DEVICE_NAMES[i] for i in np.unique(D) if i >= 0))
    print(f"  Saved: {raw[:,:,:6].shape} -> {data_path}")
    print(f"  Devices: {dev_present} -> {dev_path}")


# =====================================================================
#  ssl-wearables Data Preparation
# =====================================================================
def prepare_ssl_data(device_type='phone', no_bike=False, force=False):
    activity_labels = ACTIVITY_LABELS_NOBIKE if no_bike else ACTIVITY_LABELS
    nobike_tag = '_nobike' if no_bike else ''
    tag = device_type.upper() + (' (no bike)' if no_bike else '')
    print("\n" + "=" * 60)
    print(f"  Preparing ssl-wearables data (3ch + 6ch) [{tag}]")
    print("=" * 60)

    dir_map = {'phone': f'hhar{nobike_tag}', 'watch': f'hhar_watch{nobike_tag}'}
    out_dir = SSL_DIR / "data" / "downstream" / dir_map[device_type]
    out_dir.mkdir(parents=True, exist_ok=True)

    x3_path, x6_path = out_dir / "X.npy", out_dir / "X6.npy"
    y_path, pid_path  = out_dir / "Y.npy", out_dir / "pid.npy"
    dev_path = out_dir / "D.npy"

    if not force and all(p.exists() for p in [x3_path, x6_path, y_path, pid_path, dev_path]):
        X3, X6 = np.load(str(x3_path)), np.load(str(x6_path))
        Y, P   = np.load(str(y_path)),  np.load(str(pid_path))
        print(f"  Existing: X(3ch)={X3.shape}, X6(6ch)={X6.shape}, Y={Y.shape}")
        return
    if force:
        print(f"  --force: regenerating...")

    csv_prefix = "Phones" if device_type == 'phone' else "Watch"
    print(f"  Loading {csv_prefix}_accelerometer.csv ...")
    acc  = pd.read_csv(HHAR_RAW_DIR / f"{csv_prefix}_accelerometer.csv")
    print(f"  Loading {csv_prefix}_gyroscope.csv ...")
    gyro = pd.read_csv(HHAR_RAW_DIR / f"{csv_prefix}_gyroscope.csv")

    act_map  = {n: i for i, n in enumerate(activity_labels)}
    user_map = {u: i for i, u in enumerate(USER_NAMES)}
    TARGET_HZ, WINDOW_SEC = 30, 10
    WINDOW_LEN = TARGET_HZ * WINDOW_SEC

    all_X3, all_X6, all_Y, all_P, all_D = [], [], [], [], []

    for user_name in USER_NAMES:
        user_acc  = acc[acc['User'] == user_name]
        user_gyro = gyro[gyro['User'] == user_name]
        if len(user_acc) == 0:
            continue
        for dev_name in user_acc['Model'].unique():
            dev_acc  = user_acc[user_acc['Model'] == dev_name]
            dev_gyro = user_gyro[user_gyro['Model'] == dev_name]
            for act_name in dev_acc['gt'].unique():
                if act_name not in act_map:
                    continue
                act_a = dev_acc[dev_acc['gt'] == act_name].sort_values('Creation_Time').reset_index(drop=True)
                act_g = dev_gyro[dev_gyro['gt'] == act_name].sort_values('Creation_Time').reset_index(drop=True)
                acc_xyz = act_a[['x','y','z']].values.astype(np.float32)
                times = act_a['Creation_Time'].values
                if len(times) < 2:
                    continue
                dt_med = np.median(np.diff(times)) / 1e9
                if dt_med <= 0:
                    continue
                orig_hz = 1.0 / dt_med
                has_gyro = len(act_g) > 0
                data_6ch = None
                if has_gyro:
                    gyro_xyz = act_g[['x','y','z']].values.astype(np.float32)
                    ml = min(len(acc_xyz), len(gyro_xyz))
                    data_6ch = np.concatenate([acc_xyz[:ml], gyro_xyz[:ml]], axis=1)

                def _rs(arr, ohz, thz, mout):
                    if abs(ohz - thz) > 2:
                        nl = int(len(arr) * thz / ohz)
                        if nl < mout: return None
                        arr = resize_array(arr[np.newaxis,:,:], nl, axis=1)[0]
                    return arr if len(arr) >= mout else None

                acc_rs = _rs(acc_xyz, orig_hz, TARGET_HZ, WINDOW_LEN)
                if acc_rs is None:
                    continue
                acc_rs = np.clip(acc_rs / 9.8, -3.0, 3.0)
                six_rs = None
                if data_6ch is not None:
                    six_rs = _rs(data_6ch, orig_hz, TARGET_HZ, WINDOW_LEN)
                    if six_rs is not None:
                        six_rs[:, :3] = np.clip(six_rs[:, :3] / 9.8, -3.0, 3.0)

                nw3 = len(acc_rs) // WINDOW_LEN
                nw6 = (len(six_rs) // WINDOW_LEN) if six_rs is not None else 0
                nw  = min(nw3, nw6) if six_rs is not None else nw3
                for w in range(nw):
                    s, e = w * WINDOW_LEN, (w + 1) * WINDOW_LEN
                    all_X3.append(acc_rs[s:e])
                    if six_rs is not None:
                        all_X6.append(six_rs[s:e])
                    else:
                        pad = np.zeros((WINDOW_LEN, 6), dtype=np.float32)
                        pad[:, :3] = acc_rs[s:e]
                        all_X6.append(pad)
                    all_Y.append(act_map[act_name])
                    all_P.append(user_map[user_name])
                    all_D.append(DEVICE_INFO[dev_name]['idx'])

    if not all_X3:
        print(f"  WARNING: No valid windows for [{tag}]"); return

    X3 = np.array(all_X3, dtype=np.float32)
    X6 = np.array(all_X6, dtype=np.float32)
    Y  = np.array(all_Y, dtype=np.int64)
    P  = np.array(all_P, dtype=np.int64)
    D  = np.array(all_D, dtype=np.int64)
    for p, a in [(x3_path, X3), (x6_path, X6), (y_path, Y), (pid_path, P), (dev_path, D)]:
        np.save(str(p), a)
    dev_names_present = sorted(set(DEVICE_NAMES[i] for i in np.unique(D)
                                   if i < len(DEVICE_NAMES)),
                                key=lambda x: DEVICE_INFO[x]['idx'])
    print(f"  Saved: X(3ch)={X3.shape}, X6(6ch)={X6.shape}, Y={Y.shape}, pid={P.shape}, D={D.shape}")
    print(f"  Subjects: {[USER_NAMES[i] for i in np.unique(P)]}")
    print(f"  Devices: {dev_names_present}")


# =====================================================================
#  Combine phone + watch into 'all' dataset
# =====================================================================
def _combine_npy_datasets(base_dir, phone_name, watch_name, all_name,
                          file_names, no_bike, model_tag, force=False):
    """Concatenate phone and watch .npy files into a combined 'all' dataset."""
    nobike_tag = '_nobike' if no_bike else ''
    phone_dir = base_dir / f"{phone_name}{nobike_tag}"
    watch_dir = base_dir / f"{watch_name}{nobike_tag}"
    all_dir   = base_dir / f"{all_name}{nobike_tag}"

    if not all((phone_dir / f).exists() for f in file_names):
        print(f"  SKIP {model_tag} all: phone data not found"); return
    if not all((watch_dir / f).exists() for f in file_names):
        print(f"  SKIP {model_tag} all: watch data not found"); return

    # Skip if all files already exist (unless --force)
    if not force and all((all_dir / f).exists() for f in file_names):
        print(f"\n  {model_tag} all: already exists, skipping (use --force to regenerate)")
        return

    all_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Combining {model_tag} phone + watch -> all")
    for fname in file_names:
        p = np.load(str(phone_dir / fname))
        w = np.load(str(watch_dir / fname))
        combined = np.concatenate([p, w], axis=0)
        np.save(str(all_dir / fname), combined)
        print(f"    {fname}: phone {p.shape} + watch {w.shape} -> {combined.shape}")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare HHAR data (phone/watch separated)")
    parser.add_argument('--hart', action='store_true', help='HART only')
    parser.add_argument('--limu', action='store_true', help='LIMU-BERT only')
    parser.add_argument('--ssl',  action='store_true', help='ssl-wearables only')
    parser.add_argument('--phone-only', action='store_true', help='Phone data only')
    parser.add_argument('--watch-only', action='store_true', help='Watch data only')
    parser.add_argument('--no-bike', action='store_true',
                        help='Exclude bike activity (5 classes instead of 6)')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if output files already exist')
    parser.add_argument('--hart-3ch', action='store_true',
                        help='Also generate dedicated HART 3ch datasets (acc-only).')
    args = parser.parse_args()

    do_all = not (args.hart or args.limu or args.ssl)
    do_phone = not args.watch_only
    do_watch = not args.phone_only

    for f in ["Phones_accelerometer.csv", "Phones_gyroscope.csv",
              "Watch_accelerometer.csv", "Watch_gyroscope.csv"]:
        if not (HHAR_RAW_DIR / f).exists():
            print(f"ERROR: {HHAR_RAW_DIR / f} not found!"); sys.exit(1)

    print(f"Raw HHAR data: {HHAR_RAW_DIR}")
    no_bike = args.no_bike
    force   = args.force
    print(f"Preparing: phone={'YES' if do_phone else 'NO'}, "
          f"watch={'YES' if do_watch else 'NO'}, "
          f"no_bike={'YES' if no_bike else 'NO'}, "
          f"force={'YES' if force else 'NO'}\n")

    if do_all or args.hart:
        prepare_hart_data('all', no_bike=no_bike, channels=6)
        if do_phone: prepare_hart_data('phone', no_bike=no_bike, channels=6)
        if do_watch: prepare_hart_data('watch', no_bike=no_bike, channels=6)
        if args.hart_3ch:
            prepare_hart_data('all', no_bike=no_bike, channels=3)
            if do_phone: prepare_hart_data('phone', no_bike=no_bike, channels=3)
            if do_watch: prepare_hart_data('watch', no_bike=no_bike, channels=3)

    if do_all or args.limu:
        if do_phone: prepare_limu_data('phone', no_bike=no_bike, force=force)
        if do_watch: prepare_limu_data('watch', no_bike=no_bike, force=force)
        if do_phone and do_watch:
            _combine_npy_datasets(
                LIMU_DIR / "dataset", 'hhar', 'hhar_watch', 'hhar_all',
                ['data_20_120.npy', 'label_20_120.npy', 'D.npy'], no_bike, 'LIMU-BERT',
                force=force)

    if do_all or args.ssl:
        if do_phone: prepare_ssl_data('phone', no_bike=no_bike, force=force)
        if do_watch: prepare_ssl_data('watch', no_bike=no_bike, force=force)
        if do_phone and do_watch:
            _combine_npy_datasets(
                SSL_DIR / "data" / "downstream", 'hhar', 'hhar_watch', 'hhar_all',
                ['X.npy', 'X6.npy', 'Y.npy', 'pid.npy', 'D.npy'], no_bike, 'ssl-wearables',
                force=force)

    bike_msg = " (bike excluded, 5 classes)" if no_bike else ""
    print("\n" + "=" * 60)
    print(f"  Data preparation complete! Phone/watch separated.{bike_msg}")
    print("=" * 60)

if __name__ == "__main__":
    main()
