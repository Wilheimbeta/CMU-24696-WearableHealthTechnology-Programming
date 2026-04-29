#!/usr/bin/env python
"""
Prepare custom-collected Xsens IMU data for Experiment 8.

Reads data_combined/{condition}/{subject}/ folders, parses filename metadata, extracts
accelerometer + orientation columns (Acc_X/Y/Z + Roll/Pitch/Yaw), resamples to each model's target rate,
windows the data, and saves in HART / LIMU-BERT / ssl-wearables formats.

Data characteristics:
  - 5 subjects: B, C, D, O, Y
  - 4 sensors per recording: 2 wrist (watch), 2 pocket (phone)
  - 5 activities: sit, stand, walk, stairUP, stairDOWN
  - 2 conditions: control, uncontrolled
  - Acc + orientation (6ch), ~100 Hz, ~2 min per recording

Usage:
    python prepare_custom_data.py                     # all conditions, all devices
    python prepare_custom_data.py --condition control  # control only
    python prepare_custom_data.py --device-type watch  # wrist sensors only
    python prepare_custom_data.py --force              # regenerate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data_combined"
OUT_ROOT = BASE_DIR / "custom_eval_data"

SUBJECTS = ["B", "C", "D", "O", "Y"]

DEVICE_MAP = {
    "10B41587": {"name": "right_wrist", "type": "watch", "idx": 0},
    "10B415A7": {"name": "left_wrist",  "type": "watch", "idx": 1},
    "10B415AD": {"name": "right_pocket", "type": "phone", "idx": 2},
    "10B41584": {"name": "left_pocket",  "type": "phone", "idx": 3},
}

ACTIVITY_MAP = {
    "sit": 0,
    "stand": 1,
    "walk": 2,
    "stairup": 3,
    "stairdown": 4,
}
SUBJECT_MAP = {name: idx for idx, name in enumerate(SUBJECTS)}
CONDITION_MAP = {"control": "control", "controll": "control", "uncontrol": "uncontrolled", "uncontrolled": "uncontrolled"}


def resize_array(X, target_length, axis=0):
    orig_len = X.shape[axis]
    if orig_len == target_length:
        return X
    t_orig = np.linspace(0, 1, orig_len, endpoint=True)
    t_new = np.linspace(0, 1, target_length, endpoint=True)
    return interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(t_new)


def parse_filename(path: Path) -> dict | None:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 7 or parts[0] != "MT":
        return None
    try:
        trial_num = int(parts[2])
    except ValueError:
        return None
    subject_initial = parts[3].upper()
    if subject_initial not in SUBJECT_MAP:
        return None
    activity_raw = parts[4].lower()
    activity = activity_raw.replace("-", "").replace("_", "")
    if activity not in ACTIVITY_MAP:
        return None
    tail = parts[5:-1]
    if not tail:
        return None
    cond_token = tail[0].lower()
    condition = CONDITION_MAP.get(cond_token)
    if condition is None:
        return None
    duration_raw = "_".join(tail[1:]) if len(tail) > 1 else ""
    duration = re.sub(r"-\d+$", "", duration_raw) if duration_raw else ""
    device_id = parts[-1]
    return {
        "trial": int(trial_num),
        "subject_initial": subject_initial,
        "activity": activity_raw,
        "activity_key": activity,
        "activity_label": ACTIVITY_MAP[activity],
        "condition": condition,
        "duration": duration,
        "device_id": device_id,
        "device_info": DEVICE_MAP.get(device_id),
    }


def load_xsens_txt(path: Path) -> np.ndarray | None:
    rows = []
    acc_idx = ay_idx = az_idx = None
    roll_idx = pitch_idx = yaw_idx = None
    with open(path, "r", encoding="utf-8") as f:
        header_found = False
        for line in f:
            line = line.strip()
            if line.startswith("//"):
                continue
            if not header_found:
                if "Acc_X" in line and "Acc_Y" in line and "Acc_Z" in line:
                    header = line.split("\t")
                    index_map = {name: i for i, name in enumerate(header)}
                    acc_idx = index_map.get("Acc_X")
                    ay_idx = index_map.get("Acc_Y")
                    az_idx = index_map.get("Acc_Z")
                    roll_idx = index_map.get("Roll")
                    pitch_idx = index_map.get("Pitch")
                    yaw_idx = index_map.get("Yaw")
                    if acc_idx is None or ay_idx is None or az_idx is None:
                        return None
                    header_found = True
                continue
            parts = line.split("\t")
            try:
                max_required = max(acc_idx, ay_idx, az_idx)
                if len(parts) <= max_required:
                    continue
                ax = float(parts[acc_idx])
                ay = float(parts[ay_idx])
                az = float(parts[az_idx])
                # Use Acc + Roll/Pitch/Yaw as the 6-channel custom input.
                if roll_idx is not None and pitch_idx is not None and yaw_idx is not None:
                    max_rpy = max(roll_idx, pitch_idx, yaw_idx)
                    if len(parts) > max_rpy:
                        roll = float(parts[roll_idx])
                        pitch = float(parts[pitch_idx])
                        yaw = float(parts[yaw_idx])
                    else:
                        roll, pitch, yaw = 0.0, 0.0, 0.0
                else:
                    roll, pitch, yaw = 0.0, 0.0, 0.0
                rows.append([ax, ay, az, roll, pitch, yaw])
            except (ValueError, IndexError):
                continue
    if not rows:
        return None
    return np.array(rows, dtype=np.float32)


def collect_recordings(condition_filter: str = "all", device_type_filter: str = "all") -> list[dict]:
    normalized_condition = CONDITION_MAP.get(condition_filter, condition_filter)
    records = []
    txt_files = sorted(RAW_DIR.glob("*/*/*.txt"))
    for txt_path in txt_files:
        meta = parse_filename(txt_path)
        if meta is None:
            continue
        if normalized_condition != "all" and meta["condition"] != normalized_condition:
            continue
        if meta["device_info"] is None:
            continue
        if device_type_filter != "all" and meta["device_info"]["type"] != device_type_filter:
            continue
        acc_data = load_xsens_txt(txt_path)
        if acc_data is None or len(acc_data) < 100:
            continue
        subject_name = meta["subject_initial"]
        records.append({
            "subject": subject_name,
            "subject_id": SUBJECT_MAP[subject_name],
            "activity": meta["activity"],
            "activity_label": meta["activity_label"],
            "condition": meta["condition"],
            "duration": meta["duration"],
            "trial": meta["trial"],
            "device_id": meta["device_id"],
            "device_name": meta["device_info"]["name"],
            "device_type": meta["device_info"]["type"],
            "device_idx": meta["device_info"]["idx"],
            "acc_data": acc_data,
            "src_hz": 100.0,
            "path": str(txt_path),
        })
    return records


def window_data(data: np.ndarray, src_hz: float, target_hz: float,
                window_len: int, step: int | None = None, n_channels: int = 3) -> list[np.ndarray]:
    if step is None:
        step = window_len // 2
    new_len = int(len(data) * target_hz / src_hz)
    if new_len < window_len:
        return []
    resampled = resize_array(data, new_len, axis=0).astype(np.float32)
    windows = []
    for i in range(0, len(resampled) - window_len + 1, step):
        w = resampled[i:i + window_len]
        if n_channels > w.shape[1]:
            padded = np.zeros((window_len, n_channels), dtype=np.float32)
            padded[:, :w.shape[1]] = w
            windows.append(padded)
        else:
            windows.append(w[:, :n_channels])
    return windows


def build_hart(records: list[dict]) -> dict:
    all_X, all_Y, all_P, all_D = [], [], [], []
    for rec in records:
        wins = window_data(rec["acc_data"], rec["src_hz"], 50.0, 128, step=64, n_channels=6)
        for w in wins:
            all_X.append(w)
            all_Y.append(rec["activity_label"])
            all_P.append(rec["subject_id"])
            all_D.append(rec["device_idx"])
    if not all_X:
        return {}
    X = np.array(all_X, dtype=np.float32)
    am, astd = X[:, :, :3].mean(), X[:, :, :3].std()
    X[:, :, :3] = (X[:, :, :3] - am) / (astd + 1e-8)
    return {
        "X": X,
        "Y": np.array(all_Y, dtype=np.int64),
        "pid": np.array(all_P, dtype=np.int64),
        "D": np.array(all_D, dtype=np.int64),
    }


def build_limu(records: list[dict]) -> dict:
    all_X, all_Y, all_P, all_D = [], [], [], []
    for rec in records:
        wins = window_data(rec["acc_data"], rec["src_hz"], 20.0, 120, step=120, n_channels=6)
        for w in wins:
            all_X.append(w)
            all_Y.append(rec["activity_label"])
            all_P.append(rec["subject_id"])
            all_D.append(rec["device_idx"])
    if not all_X:
        return {}
    return {
        "X": np.array(all_X, dtype=np.float32),
        "Y": np.array(all_Y, dtype=np.int64),
        "pid": np.array(all_P, dtype=np.int64),
        "D": np.array(all_D, dtype=np.int64),
    }


def build_ssl(records: list[dict]) -> dict:
    all_X3, all_X6, all_Y, all_P, all_D = [], [], [], [], []
    for rec in records:
        wins3 = window_data(rec["acc_data"], rec["src_hz"], 30.0, 300, step=300, n_channels=3)
        wins6 = window_data(rec["acc_data"], rec["src_hz"], 30.0, 300, step=300, n_channels=6)
        n = min(len(wins3), len(wins6))
        for i in range(n):
            w3 = np.clip(wins3[i] / 9.8, -3.0, 3.0).astype(np.float32)
            w6 = wins6[i].astype(np.float32)
            w6[:, :3] = np.clip(w6[:, :3] / 9.8, -3.0, 3.0)
            all_X3.append(w3)
            all_X6.append(w6)
            all_Y.append(rec["activity_label"])
            all_P.append(rec["subject_id"])
            all_D.append(rec["device_idx"])
    if not all_X3:
        return {}
    return {
        "X3": np.array(all_X3, dtype=np.float32),
        "X6": np.array(all_X6, dtype=np.float32),
        "Y": np.array(all_Y, dtype=np.int64),
        "pid": np.array(all_P, dtype=np.int64),
        "D": np.array(all_D, dtype=np.int64),
    }


def save_dataset(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in data.items():
        np.save(str(out_dir / f"{key}.npy"), arr)


def prepare(condition: str, device_type: str, force: bool = False) -> None:
    tag = f"{condition}_{device_type}"
    out_base = OUT_ROOT / tag
    marker = out_base / ".done"
    if not force and marker.exists():
        print(f"  {tag}: already exists, skipping (use --force)")
        return

    print(f"\n{'='*60}")
    print(f"  Preparing custom data [{tag}]")
    print(f"{'='*60}")

    records = collect_recordings(condition, device_type)
    if not records:
        print(f"  WARNING: no records for [{tag}]")
        return

    subjects_found = sorted(set(r["subject"] for r in records))
    activities_found = sorted(set(r["activity"] for r in records))
    devices_found = sorted(set(r["device_name"] for r in records))
    print(f"  Subjects : {subjects_found}")
    print(f"  Activities: {activities_found}")
    print(f"  Devices  : {devices_found}")
    print(f"  Recordings: {len(records)}")

    hart_data = build_hart(records)
    if hart_data:
        save_dataset(hart_data, out_base / "hart")
        print(f"  HART: X={hart_data['X'].shape}")

    limu_data = build_limu(records)
    if limu_data:
        save_dataset(limu_data, out_base / "limu")
        print(f"  LIMU: X={limu_data['X'].shape}")

    ssl_data = build_ssl(records)
    if ssl_data:
        save_dataset(ssl_data, out_base / "ssl")
        print(f"  SSL:  X3={ssl_data['X3'].shape}, X6={ssl_data['X6'].shape}")

    metadata = {
        "condition": condition,
        "device_type": device_type,
        "subjects": {name: SUBJECT_MAP[name] for name in subjects_found},
        "activities": ACTIVITY_MAP,
        "devices": {did: info for did, info in DEVICE_MAP.items()
                    if device_type == "all" or info["type"] == device_type},
        "n_recordings": len(records),
        "n_subjects": len(subjects_found),
    }
    with open(out_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    marker.touch()
    print(f"  Saved to: {out_base}")


def main():
    parser = argparse.ArgumentParser(description="Prepare custom Xsens data for Exp8")
    parser.add_argument("--condition", default="all", choices=["control", "uncontrol", "uncontrolled", "all"])
    parser.add_argument("--device-type", default="all", choices=["watch", "phone", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DIR}")
        sys.exit(1)

    print(f"Raw data: {RAW_DIR}")
    print(f"Subjects found: {SUBJECTS}")
    print(f"Condition: {args.condition}, Device: {args.device_type}")

    conditions = ["control", "uncontrolled", "all"] if args.condition == "all" else [CONDITION_MAP.get(args.condition, args.condition)]
    device_types = ["watch", "phone", "all"] if args.device_type == "all" else [args.device_type]

    for cond in conditions:
        for dt in device_types:
            prepare(cond, dt, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
