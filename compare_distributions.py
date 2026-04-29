#!/usr/bin/env python3
"""
compare_distributions.py
========================
Raw-signal distribution analysis: HHAR vs WISDM vs data_combined.

For each of 5 activities (walking, sitting, upstairs, downstairs, standing):
  1. Detect peaks in raw gyroscope magnitude.
  2. Use peak-to-peak intervals as segments for both gyro and accelerometer.
  3. Resample every segment to a common length, compute the average,
     and draw semi-transparent "ghost" traces behind the bold average.
  4. Overlay HHAR / WISDM / data_combined-control / data_combined-uncontrolled.
  5. Show per-sensor-position comparison for data_combined.
  6. Write a text report with summary statistics.

No filtering (band-pass, low-pass) or outlier removal is applied.
This shows the raw data patterns directly.

Usage
-----
    python compare_distributions.py          # uses all defaults
    python compare_distributions.py --help   # see options
"""

from __future__ import annotations

import argparse
import io
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, fftconvolve

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent
HHAR_DIR = (BASE / "heterogeneity+activity+recognition"
            / "Activity recognition exp" / "Activity recognition exp")
WISDM_BASE = (BASE / "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"
              / "wisdm-dataset" / "wisdm-dataset" / "raw")
WISDM_RAW = WISDM_BASE / "phone"
DC_DIR = BASE / "data_combined"
PAMAP2_ROOT = (BASE / "pamap2+physical+activity+monitoring"
               / "PAMAP2_Dataset" / "PAMAP2_Dataset")
SBHAR_ROOT  = BASE / "SBHAR"
OUT_DIR = BASE / "distribution_analysis"

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
ACTIVITIES = ["walking", "sitting", "upstairs", "downstairs", "standing"]

HHAR_GT = {"walk": "walking", "sit": "sitting", "stairsup": "upstairs",
           "stairsdown": "downstairs", "stand": "standing"}
WISDM_CODE = {"A": "walking", "C": "stairs", "D": "sitting", "E": "standing"}

PAMAP2_TO_ACT = {
    2:  "sitting", 3:  "standing", 4:  "walking",
    12: "upstairs", 13: "downstairs",
}
SBHAR_TO_ACT = {
    4: "sitting", 5: "standing", 1: "walking",
    2: "upstairs", 3: "downstairs",
}
_PAMAP2_COLS_ACC  = [4, 5, 6]
_PAMAP2_COLS_GYRO = [10, 11, 12]

COLORS = {
    "HHAR":          "#1f77b4",
    "WISDM":         "#ff7f0e",
    "Controlled":    "#2ca02c",
    "Uncontrolled":  "#d62728",
    "PAMAP2":        "#9467bd",
    "SBHAR":         "#8c564b",
}
POS_CMAP = ["#e377c2", "#17becf", "#bcbd22", "#9467bd"]
SENSOR_POSITION = {
    "87": "right wrist",
    "A7": "left wrist",
    "AD": "right pocket",
    "84": "left pocket",
}
POSITION_TYPE = {
    "87": "watch", "A7": "watch",
    "AD": "phone", "84": "phone",
}

SEGMENT_LEN = 100      # normalized segment length (samples)
# Backward-compatible alias for legacy name.
CYCLE_LEN = SEGMENT_LEN
MAX_GHOST   = 60       # cap ghost traces per dataset per plot
GHOST_ALPHA = 0.25
AVG_LW      = 2.8
DC_FS       = 120.0    # Xsens MTw sampling rate (Hz)
OUTLIER_IQR_MULT = 2.5
TRIM_FRAC = 0.10
MIN_FILTER_CYCLES = 12
NO_SMOOTH = False


# ═══════════════════════════════════════════════════════════════════════
# Data container
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Segments:
    """Normalized segments for one dataset + one activity."""
    accel_norm: list = field(default_factory=list)
    gyro_norm:  list = field(default_factory=list)
    accel_xyz:  list = field(default_factory=list)
    gyro_xyz:   list = field(default_factory=list)
    durations:  list = field(default_factory=list)
    amplitudes: list = field(default_factory=list)
    fs: float = 0.0
    n_raw: int = 0
    segments_raw: int = 0
    segments_removed: int = 0

    @property
    def n(self) -> int:
        return len(self.accel_norm)

    def extend(self, o: "Segments"):
        for attr in ("accel_norm", "gyro_norm", "accel_xyz", "gyro_xyz",
                      "durations", "amplitudes"):
            getattr(self, attr).extend(getattr(o, attr))
        self.n_raw += o.n_raw
        self.segments_raw += o.segments_raw if o.segments_raw > 0 else o.n
        self.segments_removed += o.segments_removed

    # Backward-compatible aliases for legacy "cycle" names.
    @property
    def cycles_raw(self) -> int:
        return self.segments_raw

    @cycles_raw.setter
    def cycles_raw(self, value: int) -> None:
        self.segments_raw = value

    @property
    def cycles_removed(self) -> int:
        return self.segments_removed

    @cycles_removed.setter
    def cycles_removed(self, value: int) -> None:
        self.segments_removed = value


# Backward-compatible alias to minimize churn in existing call sites.
Cycles = Segments


@dataclass
class RawSnippet:
    """A short continuous time-series chunk (no segmenting/averaging)."""
    accel: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    gyro:  np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    fs: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.accel) / self.fs if self.fs > 0 else 0.0


SNIPPET_SECONDS = 8  # target snippet length for time-series plots


def _iqr_mask(x: np.ndarray, mult: float = OUTLIER_IQR_MULT) -> np.ndarray:
    """Return inlier mask based on IQR rule."""
    if len(x) == 0:
        return np.array([], dtype=bool)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-12:
        return np.ones(len(x), dtype=bool)
    lo = q1 - mult * iqr
    hi = q3 + mult * iqr
    return (x >= lo) & (x <= hi)


def _robust_center(stack: np.ndarray, **_kw) -> np.ndarray:
    """Plain mean across segments (no trimming — raw data mode)."""
    if stack.shape[0] == 1:
        return stack[0]
    return np.mean(stack, axis=0)


def _hierarchical_center(subj_map: Dict[str, "Segments"],
                         field: str) -> Optional[np.ndarray]:
    """Mean-of-subject-means: average per-subject first, then across subjects.

    This gives equal weight to each subject (regardless of segment count)
    and preserves cyclic shape much better than a pooled mean.
    """
    subj_avgs = []
    for cyc in subj_map.values():
        arr = getattr(cyc, field, None)
        if arr is None or len(arr) == 0:
            continue
        subj_avgs.append(np.mean(np.asarray(arr), axis=0))
    if not subj_avgs:
        return None
    return np.mean(np.asarray(subj_avgs), axis=0)


def _filter_segment_outliers(cyc: Segments) -> Segments:
    """Drop extreme segment outliers using duration/amplitude/magnitude spread."""
    n = cyc.n
    if cyc.segments_raw <= 0:
        cyc.segments_raw = n
    if n < MIN_FILTER_CYCLES:
        return cyc

    d = np.asarray(cyc.durations, dtype=float)
    a = np.asarray(cyc.amplitudes, dtype=float)
    g_ptp = np.asarray([np.ptp(v) for v in cyc.gyro_norm], dtype=float)
    acc_ptp = np.asarray([np.ptp(v) for v in cyc.accel_norm], dtype=float)
    acc_mean = np.asarray([np.mean(v) for v in cyc.accel_norm], dtype=float)

    mask = (
        _iqr_mask(d) &
        _iqr_mask(a) &
        _iqr_mask(g_ptp) &
        _iqr_mask(acc_ptp) &
        _iqr_mask(acc_mean)
    )

    # Avoid over-pruning small datasets.
    if mask.sum() < max(6, int(0.45 * n)):
        return cyc

    out = Segments(fs=cyc.fs, n_raw=cyc.n_raw)
    keep = np.where(mask)[0]
    out.accel_norm = [cyc.accel_norm[i] for i in keep]
    out.gyro_norm = [cyc.gyro_norm[i] for i in keep]
    out.accel_xyz = [cyc.accel_xyz[i] for i in keep]
    out.gyro_xyz = [cyc.gyro_xyz[i] for i in keep]
    out.durations = [cyc.durations[i] for i in keep]
    out.amplitudes = [cyc.amplitudes[i] for i in keep]
    out.segments_raw = cyc.segments_raw
    out.segments_removed = cyc.segments_removed + (n - len(keep))
    return out


def _filter_activity_dict(ds: Dict[str, Segments]) -> Dict[str, Segments]:
    """Apply segment-level outlier filtering for each activity entry."""
    return {k: _filter_segment_outliers(v) for k, v in ds.items()}


def _filter_nested_activity_dict(ds: Dict[str, Dict[str, Segments]]) -> Dict[str, Dict[str, Segments]]:
    """Apply segment-level outlier filtering to nested {activity:{id:Segments}}."""
    return {act: {sid: _filter_segment_outliers(c) for sid, c in m.items()}
            for act, m in ds.items()}


# ═══════════════════════════════════════════════════════════════════════
# Signal helpers
# ═══════════════════════════════════════════════════════════════════════
def _bp(sig, fs, lo=0.4, hi=6.0, order=4):
    """Band-pass filter used for dynamic-motion gyro peak detection."""
    ny = fs / 2.0
    b, a = butter(order, [max(lo / ny, 0.01), min(hi / ny, 0.99)], "band")
    pad = min(3 * max(len(b), len(a)), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=max(pad, 0)) if pad > 0 else sig

def _lp(sig, fs, cut=5.0, order=4):
    """Low-pass filter used for relatively static motions (sit/stand)."""
    ny = fs / 2.0
    b, a = butter(order, min(cut / ny, 0.99), "low")
    pad = min(3 * max(len(b), len(a)), len(sig) - 1)
    return filtfilt(b, a, sig, padlen=max(pad, 0)) if pad > 0 else sig

def _norm3(xyz: np.ndarray) -> np.ndarray:
    """Compute vector magnitude from 3-axis signal."""
    return np.sqrt((xyz ** 2).sum(axis=1))

def _resamp(seg: np.ndarray, tgt: int = SEGMENT_LEN) -> np.ndarray:
    """Resample one segment to a fixed length so segments become comparable."""
    x0 = np.linspace(0, 1, len(seg))
    x1 = np.linspace(0, 1, tgt)
    if seg.ndim == 1:
        return np.interp(x1, x0, seg)
    return np.column_stack([np.interp(x1, x0, seg[:, c])
                            for c in range(seg.shape[1])])

def _euler_to_angvel(roll, pitch, yaw, fs):
    """Approximate angular velocity from Euler angles (rad/s)."""
    dt = 1.0 / fs
    # Unwrap angle to avoid +/-180 deg discontinuity spikes in finite-difference.
    return np.column_stack([
        np.gradient(np.unwrap(np.radians(a)), dt) for a in (roll, pitch, yaw)
    ])

def _estimate_fs(ts: np.ndarray) -> float:
    """Infer sampling frequency from timestamp differences (unit-agnostic)."""
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 50.0
    med = float(np.median(diffs))
    if med == 0:
        return 50.0
    if med > 1e6:
        return 1e9 / med
    if med > 1e3:
        return 1e6 / med
    if med > 1:
        return 1e3 / med
    return 1.0 / med


def _pick_snippet(accel: np.ndarray, gyro: np.ndarray, fs: float) -> RawSnippet:
    """Pick a continuous chunk from the middle of the recording."""
    n_want = int(SNIPPET_SECONDS * fs)
    n = len(accel)
    if n <= n_want:
        return RawSnippet(accel=accel, gyro=gyro, fs=fs)
    start = (n - n_want) // 2
    return RawSnippet(accel=accel[start:start + n_want],
                      gyro=gyro[start:start + n_want], fs=fs)


# ═══════════════════════════════════════════════════════════════════════
# Segment detection — frequency-aware (1 segment ≈ 1 gait cycle)
# ═══════════════════════════════════════════════════════════════════════
_PERIOD_DEFAULTS = {
    "walking": 0.55, "upstairs": 0.65, "downstairs": 0.60,
    "sitting": 2.0, "standing": 2.0,
}


def _estimate_period(gnorm: np.ndarray, fs: float, activity: str) -> float:
    """Estimate dominant cycle period (seconds) via autocorrelation.

    Returns the period of the first strong autocorrelation peak in
    the 0.3–3.0 s range.  Falls back to a typical gait-period lookup
    if no clear periodicity is found.
    """
    max_samples = int(10 * fs)
    sig = gnorm[:max_samples] - np.mean(gnorm[:max_samples])
    if len(sig) < int(0.6 * fs):
        return _PERIOD_DEFAULTS.get(activity, 1.0)

    corr = fftconvolve(sig, sig[::-1], mode="full")
    corr = corr[len(corr) // 2:]          # keep positive lags only
    if corr[0] > 0:
        corr = corr / corr[0]             # normalize to [0, 1]

    lo = max(int(0.3 * fs), 1)
    hi = min(int(3.0 * fs), len(corr))
    if hi <= lo:
        return _PERIOD_DEFAULTS.get(activity, 1.0)

    acf_slice = corr[lo:hi]
    peaks, props = find_peaks(acf_slice, prominence=0.05)
    if len(peaks) > 0:
        best = peaks[np.argmax(props["prominences"])]
        return (best + lo) / fs            # seconds

    return _PERIOD_DEFAULTS.get(activity, 1.0)


def _detect(gnorm, fs, activity):
    """Detect segment boundaries so that each segment ≈ 1 gait cycle.

    Smooths gnorm first so that double-peak strides (left+right step)
    merge into a single peak per stride.  Then runs peak detection with
    min_distance ≈ 0.8 × T_est and keeps segments within 0.5–1.5 × T_est.
    """
    T_est = _estimate_period(gnorm, fs, activity)

    if NO_SMOOTH:
        # Raw peak detection: no smoothing, no duration filter
        min_d = max(int(0.6 * T_est * fs), 2)
        prom = max(np.std(gnorm) * 0.08,
                   np.median(np.abs(gnorm - np.median(gnorm))) * 0.15,
                   0.001)
        peaks, _ = find_peaks(gnorm, distance=min_d, prominence=prom)
        segs = [(int(peaks[i]), int(peaks[i + 1]))
                for i in range(len(peaks) - 1)]
    else:
        # Smooth to merge step-level double-peaks into one peak per stride.
        smooth_win = max(int(T_est * fs / 3), 3) | 1   # ensure odd
        kernel = np.ones(smooth_win) / smooth_win
        gn_smooth = np.convolve(gnorm, kernel, mode="same")

        min_d = max(int(0.8 * T_est * fs), 2)
        prom = max(np.std(gn_smooth) * 0.15,
                   np.median(np.abs(gn_smooth - np.median(gn_smooth))) * 0.25,
                   0.001)
        peaks, _ = find_peaks(gn_smooth, distance=min_d, prominence=prom)

        lo_dur = 0.5 * T_est
        hi_dur = 1.5 * T_est
        segs = []
        for i in range(len(peaks) - 1):
            d = (peaks[i + 1] - peaks[i]) / fs
            if lo_dur < d < hi_dur:
                segs.append((int(peaks[i]), int(peaks[i + 1])))

    if not segs:
        per = max(int(1.5 * fs), int(T_est * fs))
        return [(i, i + per)
                for i in range(0, max(len(gnorm) - per, 0), per)]
    return segs


def _extract(accel: np.ndarray, gyro: np.ndarray,
             fs: float, activity: str) -> Segments:
    """Segment raw signals using gyro peaks and apply same segments to accel."""
    if len(accel) < int(1.5 * fs):
        return Segments(fs=fs)

    gn = _norm3(gyro)

    segs = _detect(gn, fs, activity)
    an = _norm3(accel)
    cd = Segments(fs=fs, n_raw=len(accel))

    for s, e in segs:
        if e > len(accel) or (e - s) < 5:
            continue
        cd.accel_xyz.append(_resamp(accel[s:e]))
        cd.gyro_xyz.append(_resamp(gyro[s:e]))
        cd.accel_norm.append(_resamp(an[s:e]))
        cd.gyro_norm.append(_resamp(gn[s:e]))
        cd.durations.append((e - s) / fs)
        cd.amplitudes.append(float(np.ptp(gn[s:e])))
    cd.segments_raw = cd.n
    cd.segments_removed = 0
    return cd


# ═══════════════════════════════════════════════════════════════════════
# HHAR loader
# ═══════════════════════════════════════════════════════════════════════
def load_hhar(max_users: int = 4, device: str = "phone") -> Tuple[Dict[str, Segments], Dict[str, Dict[str, Segments]], Dict[str, RawSnippet]]:
    """Load HHAR data for the given device type ('phone' or 'watch')."""
    prefix = "Phones" if device == "phone" else "Watch"
    print(f"  [HHAR] Loading {device} CSV files …")
    result = {a: Segments() for a in ACTIVITIES}
    by_subject: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    snippets: Dict[str, RawSnippet] = {}

    accel_p = HHAR_DIR / f"{prefix}_accelerometer.csv"
    gyro_p  = HHAR_DIR / f"{prefix}_gyroscope.csv"
    if not accel_p.exists() or not gyro_p.exists():
        print(f"    WARNING: HHAR {device} CSVs not found — skipping.")
        return result, by_subject, snippets

    target_gt = set(HHAR_GT.keys())

    def _load(path):
        """Chunked HHAR reader to keep memory use stable."""
        parts = []
        users_seen = set()
        for ch in pd.read_csv(path, chunksize=500_000, low_memory=False):
            ch = ch.dropna(subset=["gt"])
            ch = ch[ch["gt"].isin(target_gt)]
            users_seen |= set(ch["User"].unique())
            if max_users and len(users_seen) > max_users:
                keep = sorted(users_seen)[:max_users]
                ch = ch[ch["User"].isin(keep)]
            parts.append(ch)
        return pd.concat(parts, ignore_index=True)

    print("    Reading accelerometer …")
    adf = _load(accel_p)
    print(f"    {len(adf):,} accel rows loaded")
    print("    Reading gyroscope …")
    gdf = _load(gyro_p)
    print(f"    {len(gdf):,} gyro rows loaded")

    for gt_lab, act in HHAR_GT.items():
        a_act = adf[adf["gt"] == gt_lab]
        g_act = gdf[gdf["gt"] == gt_lab]

        for (user, dev), ag in a_act.groupby(["User", "Device"]):
            gg = g_act[(g_act["User"] == user) & (g_act["Device"] == dev)]
            if len(ag) < 60 or len(gg) < 60:
                continue

            ag = ag.sort_values("Creation_Time")
            gg = gg.sort_values("Creation_Time")

            at = ag["Creation_Time"].values.astype(np.float64)
            av = ag[["x", "y", "z"]].values.astype(np.float64)
            gt_ = gg["Creation_Time"].values.astype(np.float64)
            gv = gg[["x", "y", "z"]].values.astype(np.float64)

            t0 = max(at[0], gt_[0])
            t1 = min(at[-1], gt_[-1])
            if t1 <= t0:
                continue

            fs_a = _estimate_fs(at)
            fs_g = _estimate_fs(gt_)
            fs = max(min(max(fs_a, fs_g), 200.0), 10.0)

            at_diffs = np.diff(at)
            at_diffs = at_diffs[at_diffs > 0]
            median_dt = float(np.median(at_diffs)) if len(at_diffs) > 0 else 1.0
            n = max(int((t1 - t0) / median_dt), 60)
            # Build a common timeline so accel and gyro can be aligned safely.
            t_uni = np.linspace(t0, t1, n)

            accel = np.column_stack([np.interp(t_uni, at, av[:, i])
                                     for i in range(3)])
            gyro = np.column_stack([np.interp(t_uni, gt_, gv[:, i])
                                    for i in range(3)])

            cd = _extract(accel, gyro, fs, act)
            result[act].extend(cd)
            if result[act].fs == 0:
                result[act].fs = fs
            subj_key = f"user-{user}"
            if subj_key not in by_subject[act]:
                by_subject[act][subj_key] = Segments(fs=fs)
            by_subject[act][subj_key].extend(cd)
            if act not in snippets and len(accel) >= int(3 * fs):
                snippets[act] = _pick_snippet(accel, gyro, fs)

    for a in ACTIVITIES:
        print(f"    {a:>12s}: {result[a].n:4d} segments")
    return result, by_subject, snippets


# ═══════════════════════════════════════════════════════════════════════
# WISDM loader
# ═══════════════════════════════════════════════════════════════════════
def _read_wisdm(path: Path) -> pd.DataFrame:
    """Parse one WISDM raw txt file into numeric columns."""
    rows = []
    for line in path.read_text(errors="replace").strip().split("\n"):
        line = line.strip().rstrip(";").rstrip(",")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append(dict(subj=parts[0].strip(), label=parts[1].strip(),
                             ts=float(parts[2].strip()),
                             x=float(parts[3].strip()),
                             y=float(parts[4].strip()),
                             z=float(parts[5].strip())))
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def load_wisdm(max_subj: int = 10,
               include_stairs_in_updown: bool = False,
               device: str = "phone") -> Tuple[Dict[str, Segments], Dict[str, Dict[str, Segments]], Dict[str, RawSnippet]]:
    """Load WISDM raw data for the given device type ('phone' or 'watch')."""
    raw_dir = WISDM_BASE / device
    print(f"  [WISDM] Loading {device} raw data …")
    result = {a: Segments() for a in ACTIVITIES}
    by_subject: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    snippets: Dict[str, RawSnippet] = {}

    accel_dir = raw_dir / "accel"
    gyro_dir  = raw_dir / "gyro"
    if not accel_dir.exists() or not gyro_dir.exists():
        print(f"    WARNING: WISDM {device} raw dir not found — skipping.")
        return result, by_subject, snippets

    a_files = sorted(accel_dir.glob("*.txt"))[:max_subj]
    for af in a_files:
        gf = gyro_dir / af.name.replace("accel", "gyro")
        if not gf.exists():
            continue
        # Extract subject id from filename such as data_1600_accel_phone.txt.
        m = re.match(r"data_(\d+)_", af.name)
        subj_key = f"subj-{m.group(1)}" if m else af.stem
        try:
            adf = _read_wisdm(af)
            gdf = _read_wisdm(gf)
        except Exception:
            continue

        for code, act_raw in WISDM_CODE.items():
            aa = adf[adf["label"] == code].sort_values("ts")
            gg = gdf[gdf["label"] == code].sort_values("ts")
            if len(aa) < 50 or len(gg) < 50:
                continue

            t0 = max(aa["ts"].iloc[0], gg["ts"].iloc[0])
            t1 = min(aa["ts"].iloc[-1], gg["ts"].iloc[-1])
            if t1 <= t0:
                continue

            aa_diffs = np.diff(aa["ts"].values[:200])
            aa_diffs = aa_diffs[aa_diffs > 0]
            if len(aa_diffs) == 0:
                continue
            median_dt_w = float(np.median(aa_diffs))
            if median_dt_w <= 0:
                continue
            n = max(int((t1 - t0) / median_dt_w), 50)
            n = min(n, 100_000)
            # Build a common timeline and align accel/gyro via interpolation.
            t_uni = np.linspace(t0, t1, n)

            accel = np.column_stack([np.interp(t_uni, aa["ts"].values,
                                               aa[c].values) for c in "xyz"])
            gyro  = np.column_stack([np.interp(t_uni, gg["ts"].values,
                                               gg[c].values) for c in "xyz"])
            fs_est = _estimate_fs(aa["ts"].values[:500])
            fs = max(min(fs_est, 25.0), 15.0)

            # WISDM has a single merged stairs label.
            # Optional behavior:
            #   - include_stairs_in_updown=True  -> mirror into both up/down
            #   - include_stairs_in_updown=False -> exclude from up/down analysis
            if act_raw == "stairs":
                if include_stairs_in_updown:
                    targets = ["upstairs", "downstairs"]
                else:
                    targets = []
            else:
                targets = [act_raw]
            for tgt in targets:
                cd = _extract(accel, gyro, fs, tgt)
                result[tgt].extend(cd)
                if result[tgt].fs == 0:
                    result[tgt].fs = fs
                if subj_key not in by_subject[tgt]:
                    by_subject[tgt][subj_key] = Segments(fs=fs)
                by_subject[tgt][subj_key].extend(cd)
                if tgt not in snippets and len(accel) >= int(3 * fs):
                    snippets[tgt] = _pick_snippet(accel, gyro, fs)

    for a in ACTIVITIES:
        print(f"    {a:>12s}: {result[a].n:4d} segments")
    return result, by_subject, snippets


# ═══════════════════════════════════════════════════════════════════════
# data_combined loader
# ═══════════════════════════════════════════════════════════════════════
def _parse_dc_activity(fname: str) -> Optional[str]:
    """Map data_combined filename keywords to unified 5-task labels."""
    fl = fname.lower()
    if "stairup" in fl:    return "upstairs"
    if "stairdown" in fl:  return "downstairs"
    if "walk" in fl:       return "walking"
    if "sit" in fl:        return "sitting"
    if "stand" in fl:      return "standing"
    return None

def _read_dc(path: Path) -> pd.DataFrame:
    """Read one data_combined txt file (skip metadata comment lines)."""
    lines = path.read_text(errors="replace").split("\n")
    data_lines = [l for l in lines if l.strip() and not l.strip().startswith("//")]
    return pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t")


def load_dc(condition: str = "control",
            pos_filter: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Segments]], Dict[str, Dict[str, Segments]], Dict[str, RawSnippet]]:
    """Return (sensor-level, subject-level, snippets).

    pos_filter: if 'phone' or 'watch', only load sensors of that position type.
    """
    label = f"{condition}/{pos_filter}" if pos_filter else condition
    print(f"  [data_combined/{label}] Loading …")
    result: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    by_subject: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    snippets: Dict[str, RawSnippet] = {}

    cdir = DC_DIR / condition
    if not cdir.exists():
        print(f"    WARNING: {cdir} not found — skipping.")
        return result, by_subject, snippets

    for subj_dir in sorted(cdir.iterdir()):
        if not subj_dir.is_dir():
            continue
        for fpath in sorted(subj_dir.glob("*.txt")):
            if "desktop" in fpath.name.lower():
                continue
            act = _parse_dc_activity(fpath.name)
            if act is None:
                continue

            sid = fpath.stem.split("_")[-1]
            if pos_filter:
                code = str(sid)[-2:].upper()
                if POSITION_TYPE.get(code) != pos_filter:
                    continue
            try:
                df = _read_dc(fpath)
            except Exception:
                continue

            # Keep only files that contain required channels.
            needed = ["Acc_X", "Acc_Y", "Acc_Z", "Roll", "Pitch", "Yaw"]
            if not all(c in df.columns for c in needed):
                continue
            for c in needed:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=needed)
            if len(df) < int(1.5 * DC_FS):
                continue

            accel = df[["Acc_X", "Acc_Y", "Acc_Z"]].values.astype(np.float64)
            gyro  = _euler_to_angvel(
                df["Roll"].values, df["Pitch"].values, df["Yaw"].values, DC_FS
            )
            cd = _extract(accel, gyro, DC_FS, act)
            if sid not in result[act]:
                result[act][sid] = Segments(fs=DC_FS)
            result[act][sid].extend(cd)
            subj_key = f"subj-{subj_dir.name}"
            if subj_key not in by_subject[act]:
                by_subject[act][subj_key] = Segments(fs=DC_FS)
            by_subject[act][subj_key].extend(cd)
            if act not in snippets and len(accel) >= int(3 * DC_FS):
                snippets[act] = _pick_snippet(accel, gyro, DC_FS)

    for a in ACTIVITIES:
        total = sum(c.n for c in result[a].values())
        nsens = len(result[a])
        print(f"    {a:>12s}: {total:4d} segments  ({nsens} sensors)")
    return result, by_subject, snippets


def _merge_dc(dc: Dict[str, Dict[str, Segments]]) -> Dict[str, Segments]:
    """Merge all sensor positions into one pooled segment object per activity."""
    merged = {}
    for act in ACTIVITIES:
        m = Segments(fs=DC_FS)
        for c in dc[act].values():
            m.extend(c)
        merged[act] = m
    return merged


def _merge_dc_by_position(dc: Dict[str, Dict[str, Segments]],
                           pos_type: str) -> Dict[str, Segments]:
    """Merge only sensors matching pos_type ('phone' or 'watch')."""
    merged = {}
    for act in ACTIVITIES:
        m = Segments(fs=DC_FS)
        for sid, c in dc[act].items():
            code = str(sid)[-2:].upper()
            if POSITION_TYPE.get(code) == pos_type:
                m.extend(c)
        merged[act] = m
    return merged


def _split_dc_subj_by_position(
        dc_raw: Dict[str, Dict[str, Segments]],
        by_subj: Dict[str, Dict[str, Segments]],
        pos_type: str) -> Dict[str, Dict[str, Segments]]:
    """Return subject-level dict keeping only sensors of the given position type.

    dc_raw is sensor-level {act: {sid: Segments}};
    by_subj is subject-level {act: {subj_key: Segments}} where segments from
    all sensors of that subject are pooled.  We rebuild it from dc_raw so we
    can filter by sensor position.
    """
    out: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    for act in ACTIVITIES:
        for sid, seg in dc_raw[act].items():
            code = str(sid)[-2:].upper()
            if POSITION_TYPE.get(code) != pos_type:
                continue
            key = f"sensor-{sid}"
            if key not in out[act]:
                out[act][key] = Segments(fs=DC_FS)
            out[act][key].extend(seg)
    return out


def _sensor_code_and_label(sensor_id: str) -> tuple[str, str]:
    """Convert sensor id suffix into human-readable code + body position."""
    code = str(sensor_id)[-2:].upper()
    pos = SENSOR_POSITION.get(code, "unknown position")
    return code, f"{code} - {pos}"


# ═══════════════════════════════════════════════════════════════════════
# PAMAP2 loader (watch only — wrist IMU, 100 Hz)
# ═══════════════════════════════════════════════════════════════════════
def load_pamap2(max_subjects: int = 10) -> Tuple[Dict[str, Segments], Dict[str, Dict[str, Segments]], Dict[str, RawSnippet]]:
    """Load PAMAP2 wrist-IMU data (maps to watch position).

    PAMAP2 records at 100 Hz with accel in m/s² and gyro in rad/s.
    """
    print(f"  [PAMAP2] Loading wrist IMU data (watch) …")
    result = {a: Segments() for a in ACTIVITIES}
    by_subject: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    snippets: Dict[str, RawSnippet] = {}

    if not PAMAP2_ROOT.exists():
        print(f"    WARNING: PAMAP2 root not found ({PAMAP2_ROOT}) — skipping.")
        return result, by_subject, snippets

    skip_subjects = {109}
    subjects_loaded = 0

    for folder in ("Protocol", "Optional"):
        data_dir = PAMAP2_ROOT / folder
        if not data_dir.exists():
            continue
        for fpath in sorted(data_dir.glob("subject10*.dat")):
            sid = int(fpath.stem.replace("subject", ""))
            if sid in skip_subjects:
                continue
            if subjects_loaded >= max_subjects:
                break

            rows = []
            for line in fpath.read_text(errors="replace").strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 54:
                    try:
                        rows.append([float(p) for p in parts])
                    except ValueError:
                        continue
            if not rows:
                continue

            arr = np.array(rows, dtype=np.float64)
            activities = arr[:, 1].astype(int)
            subj_key = f"pamap2-{sid}"
            had_data = False

            for pamap2_id, act in PAMAP2_TO_ACT.items():
                mask = activities == pamap2_id
                if mask.sum() < 10:
                    continue
                indices = np.where(mask)[0]
                segments_list = np.split(indices,
                                         np.where(np.diff(indices) != 1)[0] + 1)
                for seg_idx in segments_list:
                    if len(seg_idx) < 50:
                        continue
                    acc_cols = arr[seg_idx][:, _PAMAP2_COLS_ACC].astype(np.float32)
                    gyr_cols = arr[seg_idx][:, _PAMAP2_COLS_GYRO].astype(np.float32)
                    good = ~(np.isnan(acc_cols).any(axis=1) | np.isnan(gyr_cols).any(axis=1))
                    accel = acc_cols[good]
                    gyro  = gyr_cols[good]
                    if len(accel) < 50:
                        continue

                    fs = 100.0
                    cd = _extract(accel, gyro, fs, act)
                    result[act].extend(cd)
                    if result[act].fs == 0:
                        result[act].fs = fs
                    if subj_key not in by_subject[act]:
                        by_subject[act][subj_key] = Segments(fs=fs)
                    by_subject[act][subj_key].extend(cd)
                    if act not in snippets and len(accel) >= int(3 * fs):
                        snippets[act] = _pick_snippet(accel, gyro, fs)
                    had_data = True

            if had_data:
                subjects_loaded += 1

    for a in ACTIVITIES:
        print(f"    {a:>12s}: {result[a].n:4d} segments")
    return result, by_subject, snippets


# ═══════════════════════════════════════════════════════════════════════
# SBHAR loader (phone only — smartphone IMU, 50 Hz)
# ═══════════════════════════════════════════════════════════════════════
def _parse_sbhar_labels_raw():
    """Parse SBHAR/RawData/labels.txt -> list of (exp_id, user_id, act_id, start, end)."""
    labels_file = SBHAR_ROOT / "RawData" / "labels.txt"
    if not labels_file.exists():
        return []
    entries = []
    for line in labels_file.read_text(errors="replace").strip().split("\n"):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            entries.append(tuple(int(p) for p in parts))
        except ValueError:
            continue
    return entries


def load_sbhar(max_subjects: int = 10) -> Tuple[Dict[str, Segments], Dict[str, Dict[str, Segments]], Dict[str, RawSnippet]]:
    """Load SBHAR smartphone-IMU data (maps to phone position).

    SBHAR records at 50 Hz. Accel is in g (multiplied by 9.81 -> m/s²).
    Gyro is in rad/s.
    """
    print(f"  [SBHAR] Loading smartphone IMU data (phone) …")
    result = {a: Segments() for a in ACTIVITIES}
    by_subject: Dict[str, Dict[str, Segments]] = {a: {} for a in ACTIVITIES}
    snippets: Dict[str, RawSnippet] = {}

    raw_dir = SBHAR_ROOT / "RawData"
    if not raw_dir.exists():
        print(f"    WARNING: SBHAR RawData not found ({raw_dir}) — skipping.")
        return result, by_subject, snippets

    labels = _parse_sbhar_labels_raw()
    if not labels:
        print(f"    WARNING: SBHAR labels.txt empty or missing — skipping.")
        return result, by_subject, snippets

    all_uids = sorted({uid for _, uid, act, _, _ in labels if act in SBHAR_TO_ACT})
    keep_uids = set(all_uids[:max_subjects])

    file_cache: dict = {}

    def _get_raw(exp_id, user_id):
        key = (exp_id, user_id)
        if key in file_cache:
            return file_cache[key]
        acc_f  = raw_dir / f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
        gyro_f = raw_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"
        acc = gyro = None
        if acc_f.exists():
            try:
                a = np.loadtxt(str(acc_f), dtype=np.float32)
                if a.ndim == 2 and a.shape[1] == 3:
                    acc = a
            except Exception:
                pass
        if gyro_f.exists():
            try:
                g = np.loadtxt(str(gyro_f), dtype=np.float32)
                if g.ndim == 2 and g.shape[1] == 3:
                    gyro = g
            except Exception:
                pass
        file_cache[key] = (acc, gyro)
        return acc, gyro

    fs = 50.0
    for exp_id, user_id, act_id, start, end in labels:
        if act_id not in SBHAR_TO_ACT:
            continue
        if user_id not in keep_uids:
            continue
        act = SBHAR_TO_ACT[act_id]

        acc, gyro = _get_raw(exp_id, user_id)
        if acc is None or gyro is None:
            continue

        s, e = start - 1, end
        if s < 0 or e > len(acc) or e > len(gyro):
            continue
        accel = acc[s:e].copy() * 9.81
        gyro_seg = gyro[s:e].copy()
        if len(accel) < 50:
            continue

        cd = _extract(accel, gyro_seg, fs, act)
        result[act].extend(cd)
        if result[act].fs == 0:
            result[act].fs = fs
        subj_key = f"sbhar-{user_id}"
        if subj_key not in by_subject[act]:
            by_subject[act][subj_key] = Segments(fs=fs)
        by_subject[act][subj_key].extend(cd)
        if act not in snippets and len(accel) >= int(3 * fs):
            snippets[act] = _pick_snippet(accel, gyro_seg, fs)

    for a in ACTIVITIES:
        print(f"    {a:>12s}: {result[a].n:4d} segments")
    return result, by_subject, snippets


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════
def fig_timeseries(act: str,
                   snippet_map: Dict[str, Dict[str, RawSnippet]],
                   outdir: Path):
    """Plot ~8 sec continuous raw time-series for each dataset (no averaging)."""
    ds_names = ["HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"]
    present = [(name, snippet_map[name][act])
               for name in ds_names
               if name in snippet_map and act in snippet_map[name]]
    if not present:
        return

    ncols = len(present)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 6), squeeze=False)

    for col, (ds_name, snip) in enumerate(present):
        t = np.arange(len(snip.accel)) / snip.fs
        an = _norm3(snip.accel)
        gn = _norm3(snip.gyro)

        axes[0, col].plot(t, an, lw=0.7, color=COLORS.get(ds_name, "#888"))
        axes[0, col].set_title(f"{ds_name}  ({snip.duration:.1f}s)")
        axes[0, col].grid(True, alpha=0.25)
        if col == 0:
            axes[0, col].set_ylabel("Accel norm (m/s²)")

        axes[1, col].plot(t, gn, lw=0.7, color=COLORS.get(ds_name, "#888"))
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].grid(True, alpha=0.25)
        if col == 0:
            axes[1, col].set_ylabel("Angular vel. (rad/s)")

    fig.suptitle(f"{act.upper()} — Raw Time Series (~{SNIPPET_SECONDS}s continuous)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / f"{act}_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ghost(ax, segments, color, label, alpha=GHOST_ALPHA, maxn=MAX_GHOST,
           avg_override=None, shift=0.0):
    """Draw segment ghosts + mean line + 10-90 percentile band."""
    if not segments:
        return
    x = np.linspace(0, 100, SEGMENT_LEN)
    stack = np.asarray(segments)
    avg = avg_override if avg_override is not None else _robust_center(stack)

    idx = (np.random.choice(len(stack), min(maxn, len(stack)), replace=False)
           if len(stack) > maxn else np.arange(len(stack)))
    for i in idx:
        ax.plot(x, stack[i] - shift, color=color, alpha=alpha, lw=0.5, zorder=1)
    ax.plot(x, avg - shift, color=color, lw=AVG_LW, label=label, zorder=3)

    lo, hi = np.percentile(stack, [10, 90], axis=0)
    ax.fill_between(x, lo - shift, hi - shift, color=color, alpha=0.10, zorder=2)


def _split_by_unit(datasets: Dict[str, "Segments"]):
    """Split datasets into gyro (rad/s) group and orientation (degrees) group."""
    gyro_ds = {}
    orient_ds = {}
    for label, cyc in datasets.items():
        if label in ("Controlled", "Uncontrolled"):
            orient_ds[label] = cyc
        else:
            gyro_ds[label] = cyc
    return gyro_ds, orient_ds


def fig_overlay(act: str, datasets: Dict[str, Segments], outdir: Path,
                subject_data: Optional[Dict[str, Dict[str, Segments]]] = None,
                zero_mean: bool = False):
    """Ghost-trace overlay for one activity — accel + angular velocity (all rad/s)."""
    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for label, cyc in datasets.items():
        if cyc.n == 0:
            continue
        col = COLORS.get(label, "#888888")
        ha, hg = None, None
        if subject_data and label in subject_data:
            ha = _hierarchical_center(subject_data[label], "accel_norm")
            hg = _hierarchical_center(subject_data[label], "gyro_norm")
        shift_a, shift_g = 0.0, 0.0
        if zero_mean:
            avg_a = ha if ha is not None else _robust_center(np.asarray(cyc.accel_norm))
            avg_g = hg if hg is not None else _robust_center(np.asarray(cyc.gyro_norm))
            shift_a = float(np.mean(avg_a))
            shift_g = float(np.mean(avg_g))
        _ghost(ax_a, cyc.accel_norm, col, f"{label} (n={cyc.n})",
               avg_override=ha, shift=shift_a)
        _ghost(ax_g, cyc.gyro_norm,  col, f"{label} (n={cyc.n})",
               avg_override=hg, shift=shift_g)

    for ax in (ax_a, ax_g):
        avg_vals = [line.get_ydata() for line in ax.get_lines()
                    if line.get_linewidth() >= AVG_LW - 0.1]
        if avg_vals:
            cat = np.concatenate(avg_vals)
            lo_v, hi_v = float(np.min(cat)), float(np.max(cat))
            margin = max((hi_v - lo_v) * 0.6, 0.5)
            if zero_mean:
                ax.set_ylim(lo_v - margin, hi_v + margin)
            else:
                ax.set_ylim(max(lo_v - margin, 0), hi_v + margin)

    zm = " [zero-mean]" if zero_mean else ""
    unit_a = "Δ m/s²" if zero_mean else "m/s²"
    unit_g = "Δ rad/s" if zero_mean else "rad/s"

    ax_a.set_ylabel(f"Accelerometer magnitude  ({unit_a})")
    ax_a.set_title(f"{act.upper()} — Accel Magnitude{zm}  (ghost traces + average)")
    ax_a.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_a.grid(True, alpha=0.25)

    ax_g.set_ylabel(f"Angular velocity magnitude  ({unit_g})")
    ax_g.set_xlabel("Segment (%)")
    ax_g.set_title(f"{act.upper()} — Angular Velocity Magnitude{zm}")
    ax_g.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_g.grid(True, alpha=0.25)

    if zero_mean:
        for ax in (ax_a, ax_g):
            ax.axhline(0, color="grey", lw=0.6, ls="--", zorder=0)

    fig.tight_layout()
    suffix = "_overlay_zeromean" if zero_mean else "_overlay"
    fig.savefig(outdir / f"{act}{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_axes(act: str, datasets: Dict[str, Segments], outdir: Path,
             subject_data: Optional[Dict[str, Dict[str, Segments]]] = None,
             zero_mean: bool = False):
    """Per-axis (X, Y, Z) average comparison — all gyro now in rad/s."""
    x = np.linspace(0, 100, SEGMENT_LEN)
    axis_lab = ["X", "Y", "Z"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)

    for label, cyc in datasets.items():
        if cyc.n == 0:
            continue
        col = COLORS.get(label, "#888888")
        a_stack = np.asarray(cyc.accel_xyz)
        g_stack = np.asarray(cyc.gyro_xyz)

        ha = _hierarchical_center(subject_data.get(label, {}),
                                  "accel_xyz") if subject_data else None
        hg = _hierarchical_center(subject_data.get(label, {}),
                                  "gyro_xyz") if subject_data else None
        for i in range(3):
            a_avg = ha[:, i] if ha is not None else _robust_center(a_stack[:, :, i])
            g_avg = hg[:, i] if hg is not None else _robust_center(g_stack[:, :, i])
            a_shift = float(np.mean(a_avg)) if zero_mean else 0.0
            g_shift = float(np.mean(g_avg)) if zero_mean else 0.0

            axes[0, i].plot(x, a_avg - a_shift, color=col, lw=AVG_LW, label=label)
            lo, hi = np.percentile(a_stack[:, :, i], [10, 90], axis=0)
            axes[0, i].fill_between(x, lo - a_shift, hi - a_shift, color=col, alpha=0.10)

            axes[1, i].plot(x, g_avg - g_shift, color=col, lw=AVG_LW, label=label)
            lo, hi = np.percentile(g_stack[:, :, i], [10, 90], axis=0)
            axes[1, i].fill_between(x, lo - g_shift, hi - g_shift, color=col, alpha=0.10)

    for row in range(2):
        for i in range(3):
            lines = axes[row, i].get_lines()
            if not lines:
                continue
            all_v = [line.get_ydata() for line in lines]
            cat = np.concatenate(all_v)
            lo_v, hi_v = float(np.min(cat)), float(np.max(cat))
            margin = max((hi_v - lo_v) * 0.6, 0.3)
            axes[row, i].set_ylim(lo_v - margin, hi_v + margin)

    zm = " [zero-mean]" if zero_mean else ""
    unit_a = "Δ m/s²" if zero_mean else "m/s²"
    unit_g = "Δ rad/s" if zero_mean else "rad/s"
    for i, lab in enumerate(axis_lab):
        axes[0, i].set_title(f"Accel  {lab}-axis{zm}")
        axes[1, i].set_title(f"Angular vel.  {lab}-axis{zm}")
        axes[1, i].set_xlabel("Segment (%)")
        for row in range(2):
            axes[row, i].grid(True, alpha=0.2)
            if zero_mean:
                axes[row, i].axhline(0, color="grey", lw=0.6, ls="--", zorder=0)
    axes[0, 0].set_ylabel(unit_a)
    axes[1, 0].set_ylabel(unit_g)
    axes[0, 2].legend(fontsize=8, loc="upper right", framealpha=0.8)

    fig.suptitle(f"{act.upper()} — Per-Axis Average{zm}  (shading = 10-90 %ile)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    suffix = "_axes_zeromean" if zero_mean else "_axes"
    fig.savefig(outdir / f"{act}{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_positions(act: str, dc_dict: Dict[str, Segments], outdir: Path):
    """Overlay different sensor positions from data_combined."""
    if not dc_dict or all(c.n == 0 for c in dc_dict.values()):
        return
    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.linspace(0, 100, SEGMENT_LEN)

    for idx, (sid, cyc) in enumerate(sorted(dc_dict.items())):
        if cyc.n == 0:
            continue
        col = POS_CMAP[idx % len(POS_CMAP)]
        _, label = _sensor_code_and_label(sid)
        _ghost(ax_a, cyc.accel_norm, col, f"Sensor {label} (n={cyc.n})")
        _ghost(ax_g, cyc.gyro_norm,  col, f"Sensor {label} (n={cyc.n})")

    ax_a.set_ylabel("Accel magnitude (m/s²)")
    ax_a.set_title(f"{act.upper()} — Sensor-Position Comparison  (Accelerometer)")
    ax_a.legend(fontsize=9, framealpha=0.85)
    ax_a.grid(True, alpha=0.25)

    ax_g.set_ylabel("Angular velocity magnitude (rad/s)")
    ax_g.set_xlabel("Segment (%)")
    ax_g.set_title(f"{act.upper()} — Sensor-Position Comparison  (Angular Velocity)")
    ax_g.legend(fontsize=9, framealpha=0.85)
    ax_g.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / f"{act}_positions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_subject_overlay(act: str,
                        subject_ds: Dict[str, Dict[str, Segments]],
                        outdir: Path,
                        top_n: int = 8):
    """Overlay per-subject average segments for each dataset panel."""
    datasets = [n for n in ["HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"]
                if n in subject_ds]
    ncols = max(len(datasets), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 10), sharex=True, squeeze=False)
    x = np.linspace(0, 100, SEGMENT_LEN)

    for col, ds_name in enumerate(datasets):
        subj_map = subject_ds.get(ds_name, {})
        # Prioritize subjects with more detected segments for readability.
        items = sorted(
            [(sid, cyc) for sid, cyc in subj_map.items() if cyc.n > 0],
            key=lambda kv: kv[1].n,
            reverse=True,
        )[:top_n]

        for sid, cyc in items:
            lab = f"{sid} (n={cyc.n})"
            axes[0, col].plot(x, _robust_center(np.asarray(cyc.accel_norm)),
                              lw=1.4, alpha=0.85, label=lab)
            axes[1, col].plot(x, _robust_center(np.asarray(cyc.gyro_norm)),
                              lw=1.4, alpha=0.85, label=lab)

        # add hierarchical mean (black) as reference
        if items:
            subj_dict = {sid: c for sid, c in items}
            ha = _hierarchical_center(subj_dict, "accel_norm")
            hg = _hierarchical_center(subj_dict, "gyro_norm")
            if ha is not None:
                axes[0, col].plot(x, ha, color="black", lw=2.2, alpha=0.95, label="all-subj mean")
            if hg is not None:
                axes[1, col].plot(x, hg, color="black", lw=2.2, alpha=0.95, label="all-subj mean")

        axes[0, col].set_title(ds_name)
        axes[0, col].grid(True, alpha=0.25)
        axes[1, col].grid(True, alpha=0.25)
        axes[1, col].set_xlabel("Segment (%)")
        if col == 0:
            axes[0, col].set_ylabel("Accel norm (m/s²)")
            axes[1, col].set_ylabel("Angular vel. (rad/s)")
        if items:
            axes[0, col].legend(fontsize=6, framealpha=0.8, ncol=1, loc="upper right")

    fig.suptitle(f"{act.upper()} — Subject-Level Raw Overlay ({ncols} datasets)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / f"{act}_subject_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


_DYNAMIC_ACTS = ACTIVITIES


def _ylim_from_lines(ax):
    """Set y-limits tightly around plotted lines."""
    lines = ax.get_lines()
    if not lines:
        return
    cat = np.concatenate([l.get_ydata() for l in lines])
    lo, hi = float(np.min(cat)), float(np.max(cat))
    margin = max((hi - lo) * 0.35, 0.3)
    ax.set_ylim(lo - margin, hi + margin)


def fig_dataset_means_all(all_subj: Dict[str, Dict[str, Dict[str, Segments]]],
                          outdir: Path,
                          zero_mean: bool = False):
    """Norm figure: 3 dynamic activities × 2 rows (accel norm + gyro norm)."""
    x = np.linspace(0, 100, SEGMENT_LEN)
    acts = _DYNAMIC_ACTS
    n_act = len(acts)
    fig, axes = plt.subplots(2, n_act, figsize=(4.5 * n_act, 8), sharex=True)

    for col, act in enumerate(acts):
        subject_ds = all_subj.get(act, {})
        for ds_name in ("HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"):
            subj_map = subject_ds.get(ds_name, {})
            if not subj_map:
                continue
            ha = _hierarchical_center(subj_map, "accel_norm")
            hg = _hierarchical_center(subj_map, "gyro_norm")
            if ha is None:
                continue
            if zero_mean:
                ha = ha - np.mean(ha)
                hg = hg - np.mean(hg)
            c = COLORS.get(ds_name, "#888888")
            axes[0, col].plot(x, ha, color=c, lw=2.2, label=ds_name)
            axes[1, col].plot(x, hg, color=c, lw=2.2, label=ds_name)

        axes[0, col].set_title(act.upper(), fontsize=11, fontweight="bold")
        axes[1, col].set_xlabel("Segment (%)")
        for row in range(2):
            _ylim_from_lines(axes[row, col])
            axes[row, col].grid(True, alpha=0.25)
            if zero_mean:
                axes[row, col].axhline(0, color="grey", lw=0.6, ls="--", zorder=0)

    zm = " [zero-mean]" if zero_mean else ""
    unit_a = "Δ m/s²" if zero_mean else "m/s²"
    unit_g = "Δ rad/s" if zero_mean else "rad/s"
    axes[0, 0].set_ylabel(f"Accel norm ({unit_a})")
    axes[1, 0].set_ylabel(f"Angular vel. ({unit_g})")
    axes[0, -1].legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.suptitle(f"Dataset-Level Hierarchical Mean (norm){zm}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    suffix = "_zeromean" if zero_mean else ""
    fig.savefig(outdir / f"dataset_means_all{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_means_xyz(all_subj: Dict[str, Dict[str, Dict[str, Segments]]],
                          outdir: Path,
                          zero_mean: bool = False):
    """Per-axis figures: 3 dynamic activities × 3 axes (X/Y/Z).

    Generates two PNGs: one for accel xyz, one for gyro xyz.
    """
    x = np.linspace(0, 100, SEGMENT_LEN)
    acts = _DYNAMIC_ACTS
    axis_lab = ["X", "Y", "Z"]
    zm = " [zero-mean]" if zero_mean else ""
    suffix = "_zeromean" if zero_mean else ""

    for sensor, field, unit_raw, fname_base in [
        ("Accel", "accel_xyz", "m/s²", "dataset_means_accel_xyz"),
        ("Angular vel.", "gyro_xyz", "rad/s", "dataset_means_gyro_xyz"),
    ]:
        unit = f"Δ {unit_raw}" if zero_mean else unit_raw
        fig, axes = plt.subplots(3, len(acts), figsize=(4.5 * len(acts), 10),
                                 sharex=True)
        for col, act in enumerate(acts):
            subject_ds = all_subj.get(act, {})
            for ds_name in ("HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"):
                subj_map = subject_ds.get(ds_name, {})
                if not subj_map:
                    continue
                h = _hierarchical_center(subj_map, field)
                if h is None:
                    continue
                c = COLORS.get(ds_name, "#888888")
                for row in range(3):
                    val = h[:, row].copy()
                    if zero_mean:
                        val = val - np.mean(val)
                    axes[row, col].plot(x, val, color=c, lw=2.2,
                                        label=ds_name)

            axes[0, col].set_title(act.upper(), fontsize=11, fontweight="bold")
            axes[-1, col].set_xlabel("Segment (%)")
            for row in range(3):
                _ylim_from_lines(axes[row, col])
                axes[row, col].grid(True, alpha=0.25)
                if zero_mean:
                    axes[row, col].axhline(0, color="grey", lw=0.6, ls="--", zorder=0)

        for row, lab in enumerate(axis_lab):
            axes[row, 0].set_ylabel(f"{sensor} {lab} ({unit})")
        axes[0, -1].legend(fontsize=8, loc="upper right", framealpha=0.85)

        fig.suptitle(f"Dataset-Level Hierarchical Mean — {sensor} X/Y/Z{zm}",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(outdir / f"{fname_base}{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def fig_subject_overlay_xyz(act: str,
                            subject_ds: Dict[str, Dict[str, Segments]],
                            outdir: Path,
                            top_n: int = 8):
    """Per-activity per-axis subject overlay: 3 rows (X/Y/Z) × N cols (datasets).

    Mirrors the layout of fig_subject_overlay so each dataset gets its own
    column (and independent Y-axis), avoiding the squash caused by overlaying
    datasets with very different scales.
    """
    x = np.linspace(0, 100, SEGMENT_LEN)
    axis_lab = ["X", "Y", "Z"]
    ds_names = [n for n in ["HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"]
                if n in subject_ds]
    ncols = max(len(ds_names), 1)

    for sensor, field_name, unit, fname_tag in [
        ("Accel", "accel_xyz", "m/s²", "subject_overlay_accel_xyz"),
        ("Angular vel.", "gyro_xyz", "rad/s", "subject_overlay_gyro_xyz"),
    ]:
        fig, axes = plt.subplots(3, ncols, figsize=(5.5 * ncols, 12), sharex=True, squeeze=False)

        for col, ds_name in enumerate(ds_names):
            subj_map = subject_ds.get(ds_name, {})
            items = sorted(
                [(sid, cyc) for sid, cyc in subj_map.items() if cyc.n > 0],
                key=lambda kv: kv[1].n, reverse=True,
            )[:top_n]

            for sid, cyc in items:
                arr = np.asarray(getattr(cyc, field_name, []))
                if arr.ndim != 3 or len(arr) == 0:
                    continue
                subj_avg = np.mean(arr, axis=0)
                lab = f"{sid} (n={cyc.n})"
                for row in range(3):
                    axes[row, col].plot(x, subj_avg[:, row],
                                        lw=1.4, alpha=0.85, label=lab)

            if items:
                subj_dict = {sid: c for sid, c in items}
                h = _hierarchical_center(subj_dict, field_name)
                if h is not None:
                    for row in range(3):
                        axes[row, col].plot(x, h[:, row], color="black",
                                            lw=2.2, alpha=0.95,
                                            label="all-subj mean")

            axes[0, col].set_title(ds_name)
            for row in range(3):
                axes[row, col].grid(True, alpha=0.25)
            axes[-1, col].set_xlabel("Segment (%)")
            if items:
                axes[0, col].legend(fontsize=6, framealpha=0.8, ncol=1,
                                    loc="upper right")

        for row, lab in enumerate(axis_lab):
            axes[row, 0].set_ylabel(f"{sensor} {lab} ({unit})")

        fig.suptitle(f"{act.upper()} — Subject-Level {sensor} X/Y/Z Overlay ({ncols} datasets)",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(outdir / f"{act}_{fname_tag}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def fig_pair_acc_x_walking(pair: List[Tuple[str, Dict[str, Segments]]],
                           outdir: Path,
                           suptitle: str,
                           filename: str,
                           top_n: int = 8) -> None:
    """Side-by-side per-subject overlay (X-axis accel) for two datasets.

    Each panel shows the top-N subjects (by segment count) as semi-transparent
    colored lines plus a thick black mean-of-subject-means line, using the
    exact subject IDs produced by the loaders (no renaming).
    """
    x = np.linspace(0, 100, SEGMENT_LEN)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, squeeze=False)

    for col, (display_name, subj_map) in enumerate(pair):
        ax = axes[0, col]
        items = sorted(
            [(sid, cyc) for sid, cyc in subj_map.items() if cyc.n > 0],
            key=lambda kv: kv[1].n, reverse=True,
        )[:top_n]

        for sid, cyc in items:
            arr = np.asarray(getattr(cyc, "accel_xyz", []))
            if arr.ndim != 3 or len(arr) == 0:
                continue
            subj_avg_x = np.mean(arr, axis=0)[:, 0]
            ax.plot(x, subj_avg_x, lw=1.4, alpha=0.85,
                    label=f"{sid} (n={cyc.n})")

        if items:
            subj_dict = {sid: c for sid, c in items}
            h = _hierarchical_center(subj_dict, "accel_xyz")
            if h is not None:
                ax.plot(x, h[:, 0], color="black", lw=2.2, alpha=0.95,
                        label="all-subj mean")

        ax.set_title(display_name)
        ax.set_xlabel("Segment (%)")
        ax.set_ylabel("Accel X (m/s²)")
        ax.grid(True, alpha=0.25)
        if items:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_summary(all_ds: Dict[str, Dict[str, Segments]], outdir: Path):
    """Create activity-wise boxplots for duration and gyro amplitude."""
    fig, axes = plt.subplots(2, len(ACTIVITIES), figsize=(22, 8),
                             sharey="row")

    for j, act in enumerate(ACTIVITIES):
        box_dur, box_amp, labels, colors = [], [], [], []
        for label in ("HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"):
            cyc = all_ds[act].get(label)
            if cyc is None or cyc.n == 0:
                continue
            box_dur.append(cyc.durations)
            box_amp.append(cyc.amplitudes)
            labels.append(label)
            colors.append(COLORS.get(label, "#888888"))

        if box_dur:
            bp = axes[0, j].boxplot(box_dur, labels=labels, patch_artist=True,
                                    widths=0.6, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)
        axes[0, j].set_title(act.title(), fontsize=10)
        axes[0, j].tick_params(axis="x", rotation=35, labelsize=8)

        if box_amp:
            bp = axes[1, j].boxplot(box_amp, labels=labels, patch_artist=True,
                                    widths=0.6, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)
        axes[1, j].tick_params(axis="x", rotation=35, labelsize=8)

    axes[0, 0].set_ylabel("Segment duration (s)")
    axes[1, 0].set_ylabel("Rotation peak-to-peak amplitude (rad/s)")
    fig.suptitle("Segment Duration & Rotation Amplitude — Raw Distribution Summary",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "summary_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_diagnostic(act: str, datasets: Dict[str, Segments], outdir: Path,
                   subject_data: Optional[Dict[str, Dict[str, Segments]]] = None):
    """Show mean ± std diagnostic curves — all angular velocity in rad/s."""
    x = np.linspace(0, 100, SEGMENT_LEN)

    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    avg_a_vals, avg_g_vals = [], []

    for label, cyc in datasets.items():
        if cyc.n < 3:
            continue
        col = COLORS.get(label, "#888888")
        a_s = np.asarray(cyc.accel_norm)
        g_s = np.asarray(cyc.gyro_norm)
        ha = _hierarchical_center(subject_data.get(label, {}),
                                  "accel_norm") if subject_data else None
        hg = _hierarchical_center(subject_data.get(label, {}),
                                  "gyro_norm") if subject_data else None
        a_c = ha if ha is not None else _robust_center(a_s)
        g_c = hg if hg is not None else _robust_center(g_s)
        avg_a_vals.append(a_c)
        avg_g_vals.append(g_c)
        ax_a.plot(x, a_c, color=col, lw=2, label=label)
        ax_a.fill_between(x, a_c - a_s.std(0), a_c + a_s.std(0), color=col, alpha=0.15)
        ax_g.plot(x, g_c, color=col, lw=2, label=label)
        ax_g.fill_between(x, g_c - g_s.std(0), g_c + g_s.std(0), color=col, alpha=0.15)

    # Y-limits based on average lines so patterns are visible
    for ax, vals in [(ax_a, avg_a_vals), (ax_g, avg_g_vals)]:
        if vals:
            all_v = np.concatenate(vals)
            lo, hi = float(np.min(all_v)), float(np.max(all_v))
            margin = max((hi - lo) * 0.6, 0.5)
            ax.set_ylim(lo - margin, hi + margin)

    ax_a.set_ylabel("Accel norm (m/s²)")
    ax_a.set_title(f"{act.upper()} — Raw Mean ± 1 std")
    ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.2)

    ax_g.set_ylabel("Angular vel. (rad/s)")
    ax_g.set_xlabel("Segment (%)")
    ax_g.legend(fontsize=8); ax_g.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outdir / f"{act}_mean_std.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════
def _rotation_unit_for_label(label: str) -> str:
    """All datasets now use angular velocity in rad/s."""
    return "rad/s"


def _stat_block(cyc: Segments, label: str) -> str:
    """Format one report block with segment count and key distribution stats."""
    if cyc.n == 0:
        return f"  {label:20s}  —  no segments detected\n"
    d = np.asarray(cyc.durations)
    a = np.asarray(cyc.amplitudes)
    an = np.asarray(cyc.accel_norm)
    gn = np.asarray(cyc.gyro_norm)
    rot_unit = _rotation_unit_for_label(label)
    return (
        f"  {label:20s}\n"
        f"    Segments detected    : {cyc.n}\n"
        f"    Segment duration (s) : {d.mean():.3f} +/- {d.std():.3f}  "
        f"[{d.min():.2f} — {d.max():.2f}]\n"
        f"    Rotation amplitude   : {a.mean():.3f} +/- {a.std():.3f}  ({rot_unit})\n"
        f"    Accel-norm mean      : {an.mean():.2f}   "
        f"std-of-mean {an.mean(1).std():.2f}\n"
        f"    Rotation-norm mean   : {gn.mean():.2f}   "
        f"std-of-mean {gn.mean(1).std():.2f}  ({rot_unit})\n"
    )


def _outlier_totals_by_dataset(all_ds: Dict[str, Dict[str, Segments]]) -> Dict[str, dict]:
    """Aggregate outlier filtering counts across all activities."""
    totals = {}
    for name in ("HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"):
        raw = 0
        kept = 0
        for act in ACTIVITIES:
            cyc = all_ds.get(act, {}).get(name)
            if cyc is None:
                continue
            r = cyc.segments_raw if cyc.segments_raw > 0 else cyc.n
            raw += int(r)
            kept += int(cyc.n)
        removed = max(raw - kept, 0)
        ratio = (removed / raw * 100.0) if raw > 0 else 0.0
        totals[name] = {"raw": raw, "kept": kept, "removed": removed, "ratio": ratio}
    return totals


def write_report(all_ds: Dict[str, Dict[str, Segments]],
                 dc_ctrl: Dict[str, Dict[str, Segments]],
                 outdir: Path,
                 include_wisdm_stairs_in_updown: bool = False):
    """Generate human-readable text report with per-activity summaries."""
    lines = []
    lines.append("=" * 72)
    lines.append("  RAW SIGNAL DISTRIBUTION ANALYSIS REPORT")
    lines.append("  HHAR / WISDM / Controlled / Uncontrolled / PAMAP2 / SBHAR")
    lines.append("  (No filtering, no outlier removal — raw data)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  NOTE: WISDM only has a combined 'stairs' label (no up/down")
    lines.append("  distinction).")
    if include_wisdm_stairs_in_updown:
        lines.append("  Current run setting: WISDM stairs is mirrored into both")
        lines.append("  upstairs and downstairs comparisons.")
    else:
        lines.append("  Current run setting: WISDM stairs is excluded from")
        lines.append("  upstairs/downstairs analyses.")
    lines.append("")
    lines.append("  NOTE: data_combined Roll/Pitch/Yaw are converted to angular")
    lines.append("  velocity (rad/s) via time derivative, so all datasets now share")
    lines.append("  the same rotation unit and are directly comparable.")
    lines.append("")
    lines.append("  MODE: Raw data — no filtering, no outlier removal.")

    # Detailed section: one activity at a time, all datasets side by side.
    for act in ACTIVITIES:
        lines.append(f"\n{'─' * 72}")
        lines.append(f"  Activity: {act.upper()}")
        lines.append(f"{'─' * 72}")
        for label in ("HHAR", "WISDM", "Controlled", "Uncontrolled", "PAMAP2", "SBHAR"):
            cyc = all_ds[act].get(label)
            if cyc and cyc.n > 0:
                lines.append(_stat_block(cyc, label))

        if dc_ctrl[act]:
            lines.append("  ── Sensor-position breakdown (data_combined / control) ──")
            for sid, cyc in sorted(dc_ctrl[act].items()):
                _, label = _sensor_code_and_label(sid)
                lines.append(_stat_block(cyc, f"  Sensor {label}"))

    lines.append(f"\n{'=' * 72}")
    lines.append("  KEY OBSERVATIONS")
    lines.append("=" * 72)

    # Key observations section: concise cross-dataset comparisons.
    for act in ACTIVITIES:
        ds = all_ds[act]
        present = {k: v for k, v in ds.items() if v.n > 0}
        if len(present) < 2:
            continue
        durs = {k: np.mean(v.durations) for k, v in present.items()}
        amps = {k: np.mean(v.amplitudes) for k, v in present.items()}

        lines.append(f"\n  {act.upper()}:")
        lines.append(f"    Avg segment duration:  " +
                     "  |  ".join(f"{k}: {v:.3f}s" for k, v in durs.items()))
        lines.append(f"    Avg rotation amplitude (rad/s):  " +
                     "  |  ".join(f"{k}: {v:.3f}" for k, v in amps.items()))

        if "HHAR" in durs and "Controlled" in durs:
            ratio = durs["Controlled"] / max(durs["HHAR"], 1e-9)
            lines.append(f"    Controlled/HHAR duration ratio: {ratio:.2f}")
        if "Controlled" in durs and "Uncontrolled" in durs:
            ratio = durs["Uncontrolled"] / max(durs["Controlled"], 1e-9)
            lines.append(f"    Uncontrolled/Controlled duration ratio: {ratio:.2f}")

    # Sensor-position variability section for data_combined/control.
    if dc_ctrl:
        lines.append(f"\n{'─' * 72}")
        lines.append("  SENSOR-POSITION VARIABILITY (data_combined/control)")
        lines.append(f"{'─' * 72}")
        for act in ACTIVITIES:
            if not dc_ctrl[act]:
                continue
            pos_durs = {_sensor_code_and_label(sid)[1]: np.mean(c.durations)
                        for sid, c in dc_ctrl[act].items() if c.n > 0}
            pos_amps = {_sensor_code_and_label(sid)[1]: np.mean(c.amplitudes)
                        for sid, c in dc_ctrl[act].items() if c.n > 0}
            if pos_durs:
                lines.append(f"\n  {act.upper()}:")
                lines.append(f"    Duration by sensor:  " +
                             "  |  ".join(f"{k}: {v:.3f}s"
                                          for k, v in pos_durs.items()))
                lines.append(f"    Amplitude by sensor: " +
                             "  |  ".join(f"{k}: {v:.3f}"
                                          for k, v in pos_amps.items()))

    text = "\n".join(lines)
    (outdir / "distribution_report.txt").write_text(text, encoding="utf-8")
    return text


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    """Entry point: load data, extract segments, draw plots, and write report."""
    ap = argparse.ArgumentParser(description="Raw signal distribution analysis (segmenting & averaging)")
    ap.add_argument("--max-hhar-users", type=int, default=4)
    ap.add_argument("--max-wisdm-subjects", type=int, default=10)
    ap.add_argument("--no-hhar", action="store_true")
    ap.add_argument("--no-wisdm", action="store_true")
    ap.add_argument(
        "--include-wisdm-stairs-in-updown",
        action="store_true",
        help=("If set, include WISDM merged stairs label in upstairs/downstairs "
              "by mirroring it to both; default is off (stairs excluded)."),
    )
    ap.add_argument("--no-pamap2", action="store_true", help="Skip PAMAP2 (watch)")
    ap.add_argument("--no-sbhar", action="store_true", help="Skip SBHAR (phone)")
    ap.add_argument("--max-pamap2-subjects", type=int, default=10)
    ap.add_argument("--max-sbhar-subjects", type=int, default=10)
    ap.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable gnorm smoothing and duration filter in peak detection.",
    )
    ap.add_argument(
        "--walking-acc-x",
        action="store_true",
        help="Produce only the 5 walking Accel-X comparison figures; skip all other plots and the report.",
    )
    args = ap.parse_args()

    global NO_SMOOTH
    NO_SMOOTH = args.no_smooth

    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smooth_mode = "NO smoothing, no duration filter" if NO_SMOOTH else "smoothed peak detection"
    print("=" * 60)
    print("  Raw Signal Distribution Analysis")
    print(f"  (segmenting by gyro peaks, {smooth_mode})")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────
    _empty = lambda: {a: Segments() for a in ACTIVITIES}
    _empty_subj = lambda: {a: {} for a in ACTIVITIES}

    if args.no_hhar:
        hhar_phone, hhar_phone_subj, hhar_phone_snip = _empty(), _empty_subj(), {}
        hhar_watch, hhar_watch_subj, hhar_watch_snip = _empty(), _empty_subj(), {}
    else:
        hhar_phone, hhar_phone_subj, hhar_phone_snip = load_hhar(args.max_hhar_users, device="phone")
        hhar_watch, hhar_watch_subj, hhar_watch_snip = load_hhar(args.max_hhar_users, device="watch")

    if args.no_wisdm:
        wisdm_phone, wisdm_phone_subj, wisdm_phone_snip = _empty(), _empty_subj(), {}
        wisdm_watch, wisdm_watch_subj, wisdm_watch_snip = _empty(), _empty_subj(), {}
    else:
        wisdm_phone, wisdm_phone_subj, wisdm_phone_snip = load_wisdm(
            args.max_wisdm_subjects,
            include_stairs_in_updown=args.include_wisdm_stairs_in_updown,
            device="phone",
        )
        wisdm_watch, wisdm_watch_subj, wisdm_watch_snip = load_wisdm(
            args.max_wisdm_subjects,
            include_stairs_in_updown=args.include_wisdm_stairs_in_updown,
            device="watch",
        )

    # Load dc once without filter for sensor-position plots
    dc_ctrl_raw, _, dc_ctrl_snip = load_dc("control")
    dc_unctrl_raw, _, dc_unctrl_snip = load_dc("uncontrolled")

    # Load dc per position type — by_subject naturally grouped by real subject
    dc_ctrl_phone_raw,   dc_ctrl_phone_subj,   _ = load_dc("control",     pos_filter="phone")
    dc_ctrl_watch_raw,   dc_ctrl_watch_subj,   _ = load_dc("control",     pos_filter="watch")
    dc_unctrl_phone_raw, dc_unctrl_phone_subj, _ = load_dc("uncontrolled", pos_filter="phone")
    dc_unctrl_watch_raw, dc_unctrl_watch_subj, _ = load_dc("uncontrolled", pos_filter="watch")

    ctrl_phone   = _merge_dc(dc_ctrl_phone_raw)
    ctrl_watch   = _merge_dc(dc_ctrl_watch_raw)
    unctrl_phone = _merge_dc(dc_unctrl_phone_raw)
    unctrl_watch = _merge_dc(dc_unctrl_watch_raw)

    # Load PAMAP2 (watch) and SBHAR (phone)
    if args.no_sbhar:
        sbhar_phone, sbhar_phone_subj, sbhar_phone_snip = _empty(), _empty_subj(), {}
    else:
        sbhar_phone, sbhar_phone_subj, sbhar_phone_snip = load_sbhar(args.max_sbhar_subjects)

    if args.no_pamap2:
        pamap2_watch, pamap2_watch_subj, pamap2_watch_snip = _empty(), _empty_subj(), {}
    else:
        pamap2_watch, pamap2_watch_subj, pamap2_watch_snip = load_pamap2(args.max_pamap2_subjects)

    # ── Dedicated mode: 5 walking Accel-X comparison figures only ─────
    if args.walking_acc_x:
        out = OUT_DIR / "walking_acc_x"
        out.mkdir(parents=True, exist_ok=True)

        fig_pair_acc_x_walking(
            [("HHAR", hhar_phone_subj["walking"]),
             ("WISDM", wisdm_phone_subj["walking"])],
            out,
            "Accelerometer data (X-axis) of phone placements collected from walking activity for HHAR and WISDM",
            "walking_acc_x_HHAR_vs_WISDM_phone.png",
        )
        fig_pair_acc_x_walking(
            [("HHAR", hhar_watch_subj["walking"]),
             ("WISDM", wisdm_watch_subj["walking"])],
            out,
            "Accelerometer data (X-axis) of watch placements collected from walking activity for HHAR and WISDM",
            "walking_acc_x_HHAR_vs_WISDM_watch.png",
        )
        fig_pair_acc_x_walking(
            [("SBHAR (phone)", sbhar_phone_subj["walking"]),
             ("PAMAP2 (watch)", pamap2_watch_subj["walking"])],
            out,
            "Accelerometer data (X-axis) of device placement collected from walking activity - phone for SBHAR and watch for PAMAP2",
            "walking_acc_x_SBHAR_phone_vs_PAMAP2_watch.png",
        )
        fig_pair_acc_x_walking(
            [("Controlled", dc_ctrl_watch_subj["walking"]),
             ("Uncontrolled", dc_unctrl_watch_subj["walking"])],
            out,
            "Accelerometer data (X-axis) of watch placements collected from walking activity",
            "walking_acc_x_controlled_vs_uncontrolled_watch.png",
        )
        fig_pair_acc_x_walking(
            [("Controlled", dc_ctrl_phone_subj["walking"]),
             ("Uncontrolled", dc_unctrl_phone_subj["walking"])],
            out,
            "Accelerometer data (X-axis) of phone placements collected from walking activity",
            "walking_acc_x_controlled_vs_uncontrolled_phone.png",
        )

        print(f"\n  [walking-acc-x] 5 figures saved to: {out}")
        return

    # ── Generate figures per device type ──────────────────────────────
    device_configs = {
        "phone": {
            "hhar": hhar_phone, "hhar_subj": hhar_phone_subj, "hhar_snip": hhar_phone_snip,
            "wisdm": wisdm_phone, "wisdm_subj": wisdm_phone_subj, "wisdm_snip": wisdm_phone_snip,
            "ctrl": ctrl_phone, "unctrl": unctrl_phone,
            "ctrl_subj": dc_ctrl_phone_subj, "unctrl_subj": dc_unctrl_phone_subj,
            "extra": sbhar_phone, "extra_subj": sbhar_phone_subj, "extra_snip": sbhar_phone_snip,
            "extra_name": "SBHAR",
        },
        "watch": {
            "hhar": hhar_watch, "hhar_subj": hhar_watch_subj, "hhar_snip": hhar_watch_snip,
            "wisdm": wisdm_watch, "wisdm_subj": wisdm_watch_subj, "wisdm_snip": wisdm_watch_snip,
            "ctrl": ctrl_watch, "unctrl": unctrl_watch,
            "ctrl_subj": dc_ctrl_watch_subj, "unctrl_subj": dc_unctrl_watch_subj,
            "extra": pamap2_watch, "extra_subj": pamap2_watch_subj, "extra_snip": pamap2_watch_snip,
            "extra_name": "PAMAP2",
        },
    }

    for dev_type, cfg in device_configs.items():
        dev_dir = OUT_DIR / dev_type
        dev_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Generating {dev_type.upper()} plots …")
        print(f"{'='*60}")

        extra_name = cfg.get("extra_name", "")
        snippet_map: Dict[str, Dict[str, RawSnippet]] = {
            "HHAR": cfg["hhar_snip"],
            "WISDM": cfg["wisdm_snip"],
            "Controlled": dc_ctrl_snip,
            "Uncontrolled": dc_unctrl_snip,
        }
        if extra_name and cfg.get("extra_snip"):
            snippet_map[extra_name] = cfg["extra_snip"]

        all_ds: Dict[str, Dict[str, Segments]] = {}
        all_subj_dev: Dict[str, Dict[str, Dict[str, Segments]]] = {}

        for act in ACTIVITIES:
            ds = {
                "HHAR":          cfg["hhar"][act],
                "WISDM":         cfg["wisdm"][act],
                "Controlled":    cfg["ctrl"][act],
                "Uncontrolled":  cfg["unctrl"][act],
            }
            if extra_name:
                ds[extra_name] = cfg["extra"][act]
            all_ds[act] = ds

            subj = {
                "HHAR":         cfg["hhar_subj"][act],
                "WISDM":        cfg["wisdm_subj"][act],
                "Controlled":   cfg["ctrl_subj"][act],
                "Uncontrolled": cfg["unctrl_subj"][act],
            }
            if extra_name:
                subj[extra_name] = cfg["extra_subj"][act]

            print(f"\n  [{dev_type}] Plotting {act} …")
            fig_timeseries(act, snippet_map, dev_dir)
            fig_overlay(act, ds, dev_dir, subject_data=subj)
            fig_overlay(act, ds, dev_dir, subject_data=subj, zero_mean=True)
            fig_diagnostic(act, ds, dev_dir, subject_data=subj)

            has_xyz = all(len(c.accel_xyz) > 0 for c in ds.values() if c.n > 0)
            if has_xyz:
                fig_axes(act, {k: v for k, v in ds.items() if v.n > 0}, dev_dir,
                         subject_data=subj)
                fig_axes(act, {k: v for k, v in ds.items() if v.n > 0}, dev_dir,
                         subject_data=subj, zero_mean=True)

            fig_subject_overlay(act, subj, dev_dir)
            fig_subject_overlay_xyz(act, subj, dev_dir)
            all_subj_dev[act] = subj

        fig_dataset_means_all(all_subj_dev, dev_dir)
        fig_dataset_means_all(all_subj_dev, dev_dir, zero_mean=True)
        fig_dataset_means_xyz(all_subj_dev, dev_dir)
        fig_dataset_means_xyz(all_subj_dev, dev_dir, zero_mean=True)
        fig_summary(all_ds, dev_dir)

        report = write_report(
            all_ds,
            dc_ctrl_raw,
            dev_dir,
            include_wisdm_stairs_in_updown=args.include_wisdm_stairs_in_updown,
        )
        print(f"\n  [{dev_type.upper()}] Report:")
        print(report)

    # Sensor-position comparison (uses all sensors, not split)
    for act in ACTIVITIES:
        fig_positions(act, dc_ctrl_raw[act], OUT_DIR)

    print(f"\n  All outputs saved to:  {OUT_DIR}")
    print(f"    Phone plots: {OUT_DIR / 'phone'}")
    print(f"    Watch plots: {OUT_DIR / 'watch'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
