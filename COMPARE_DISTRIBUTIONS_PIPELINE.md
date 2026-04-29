# `compare_distributions.py` — Pipeline Documentation

## Overview

Cross-dataset raw-signal distribution analysis comparing **up to 6 data sources** across **5 activities**, split by **device position** (phone / watch).

| Source | Rotation data | Sampling rate | Position |
|---|---|---|---|
| **HHAR** | Native gyroscope (rad/s) | ~50–200 Hz (auto-detected) | phone + watch |
| **WISDM** | Native gyroscope (rad/s) | ~20 Hz | phone + watch |
| **data_combined / Control** | Euler angles → angular velocity (rad/s) | 120 Hz (Xsens MTw) | phone + watch |
| **data_combined / Uncontrolled** | Euler angles → angular velocity (rad/s) | 120 Hz (Xsens MTw) | phone + watch |
| **PAMAP2** | Native gyroscope (rad/s) | 100 Hz | **watch only** (wrist IMU) |
| **SBHAR** | Native gyroscope (rad/s) | 50 Hz | **phone only** (smartphone IMU) |

Activities: `walking`, `sitting`, `upstairs`, `downstairs`, `standing`.

> Phone plots show 5 datasets: HHAR, WISDM, Control, Uncontrolled, **SBHAR**.
> Watch plots show 5 datasets: HHAR, WISDM, Control, Uncontrolled, **PAMAP2**.

> data_combined provides Roll/Pitch/Yaw in degrees. The script converts these to angular velocity via `d/dt(unwrap(radians(angle)))`, putting all 4 sources in the same unit (rad/s).

---

## Usage

```bash
python compare_distributions.py              # all defaults
python compare_distributions.py --help       # show all flags
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--max-hhar-users N` | 4 | Cap number of HHAR users loaded |
| `--max-wisdm-subjects N` | 10 | Cap number of WISDM subjects loaded |
| `--max-pamap2-subjects N` | 10 | Cap number of PAMAP2 subjects loaded (watch) |
| `--max-sbhar-subjects N` | 10 | Cap number of SBHAR subjects loaded (phone) |
| `--no-hhar` | off | Skip HHAR entirely |
| `--no-wisdm` | off | Skip WISDM entirely |
| `--no-pamap2` | off | Skip PAMAP2 (watch) entirely |
| `--no-sbhar` | off | Skip SBHAR (phone) entirely |
| `--include-wisdm-stairs-in-updown` | off | Mirror WISDM's merged "stairs" label into both upstairs and downstairs (default: exclude from both) |
| `--no-smooth` | off | Disable gyro-norm smoothing and duration filter in peak detection |

---

## Pipeline Steps

### 1. Data Loading

Each loader returns three structures per activity:

- **Pooled segments** — all segments merged (one `Segments` object per activity)
- **By-subject** — `{subject_id: Segments}` per activity (for hierarchical averaging)
- **Snippets** — one ~8 s continuous raw chunk per activity (for time-series plots)

Loading is done separately for `phone` and `watch` sensor positions:

- **HHAR** — `Phones_*.csv` / `Watch_*.csv`, grouped by `(User, Device)`, accel+gyro time-aligned via interpolation onto a common timeline.
- **WISDM** — per-subject `data_XXXX_accel_phone.txt` + `data_XXXX_gyro_phone.txt` (and watch equivalents). Accel+gyro aligned via interpolation. Note: WISDM has a single `C = stairs` label with no up/down split.
- **data_combined** — per-subject directories under `control/` and `uncontrolled/`, each containing tab-separated txt files with `Acc_X/Y/Z`, `Roll`, `Pitch`, `Yaw`. Four sensor positions per subject:

| Sensor suffix | Body position | Position type |
|---|---|---|
| `87` | Right wrist | watch |
| `A7` | Left wrist | watch |
| `AD` | Right pocket | phone |
| `84` | Left pocket | phone |

- **PAMAP2** (**watch only**) — Wrist/hand IMU from `PAMAP2_Dataset/Protocol/` and `Optional/`. Columns 4–6 = accel (m/s²), columns 10–12 = gyro (rad/s) at 100 Hz. Continuous segments per activity per subject fed through the same peak-detection pipeline. Subject 109 is skipped (no relevant activities).
- **SBHAR** (**phone only**) — Smartphone IMU from `SBHAR/RawData/`. Accel (in g, ×9.81 → m/s²) + gyro (rad/s) at 50 Hz. Labeled segments parsed from `labels.txt` with per-experiment raw acc/gyro files.

### 2. Segment Detection

Goal: split continuous recordings into **one segment per gait cycle** (peak-to-peak on gyro magnitude).

```
raw gyro XYZ → ‖gyro‖ → estimate period (ACF) → smooth → find_peaks → segments
```

1. **Period estimation** — Autocorrelation on gyro magnitude; find the first prominent peak in the 0.3–3.0 s lag range. Falls back to activity-specific defaults (walking: 0.55 s, upstairs: 0.65 s, etc.).
2. **Smoothing** — Moving-average kernel (width ≈ T/3) merges left+right step double-peaks into one peak per stride. Disabled by `--no-smooth`.
3. **Peak detection** — `scipy.signal.find_peaks` with `distance ≈ 0.8 × T_est`, prominence threshold from signal spread. Only segments with duration in `[0.5T, 1.5T]` are kept.
4. **Resampling** — Each segment (accel XYZ, gyro XYZ, accel norm, gyro norm) is resampled to a fixed length of **100 samples** via linear interpolation.

### 3. Averaging

Two methods:

- **Pooled mean** (`_robust_center`) — Simple mean across all segments. Used as fallback.
- **Hierarchical mean** (`_hierarchical_center`) — Mean-of-subject-means: average segments within each subject first, then average across subjects. Gives equal weight per subject regardless of segment count. **Used for all overlay and summary plots.**

### 4. Outlier Filtering

IQR-based filtering on 5 features per segment (only applied when >= 12 segments):

- Segment duration
- Gyro amplitude (peak-to-peak)
- Gyro norm peak-to-peak
- Accel norm peak-to-peak
- Accel norm mean

Safety guard: never prune below max(6, 45% of segments).

---

## Output Structure

```
distribution_analysis/
├── phone/                          # phone-position results
│   ├── distribution_report.txt     # text report
│   │
│   ├── ── Per activity (×5) ──────────────────────────
│   ├── {act}_timeseries.png        # ~8 s raw continuous signal
│   ├── {act}_overlay.png           # ghost-trace overlay (norm)
│   ├── {act}_overlay_zeromean.png  # ghost-trace overlay (zero-mean norm)
│   ├── {act}_mean_std.png          # mean ± 1 std diagnostic
│   ├── {act}_axes.png              # per-axis X/Y/Z average (all datasets overlaid)
│   ├── {act}_axes_zeromean.png     # per-axis X/Y/Z average (zero-mean)
│   ├── {act}_subject_overlay.png           # per-subject norm (2 rows × N cols)
│   ├── {act}_subject_overlay_accel_xyz.png # per-subject accel X/Y/Z (3 rows × N cols)
│   ├── {act}_subject_overlay_gyro_xyz.png  # per-subject gyro X/Y/Z (3 rows × N cols)
│   │
│   ├── ── Cross-activity (all 5 activities) ───────────
│   ├── dataset_means_all.png               # hierarchical mean norms
│   ├── dataset_means_all_zeromean.png
│   ├── dataset_means_accel_xyz.png         # hierarchical mean accel X/Y/Z
│   ├── dataset_means_accel_xyz_zeromean.png
│   ├── dataset_means_gyro_xyz.png          # hierarchical mean gyro X/Y/Z
│   ├── dataset_means_gyro_xyz_zeromean.png
│   └── summary_boxplots.png                # duration & amplitude boxplots
│
├── watch/                          # watch-position results (same structure)
│   └── ...
│
└── {act}_positions.png             # sensor-position comparison (all 4 sensors, control only)
```

---

## Plot Catalog

### Per-Activity Plots (generated for each of the 5 activities)

| Plot | Layout | Description |
|---|---|---|
| **Time series** | 2 rows × N cols | ~8 s continuous raw accel norm + gyro norm; one column per available dataset |
| **Overlay** | 2 rows × 1 col | Ghost traces (up to 60 random segments) + bold hierarchical mean + 10–90 percentile band; accel norm on top, gyro norm on bottom |
| **Overlay (zero-mean)** | same | Same as overlay but each dataset's mean is shifted to zero to compare shape only |
| **Diagnostic** | 2 rows × 1 col | Hierarchical mean ± 1 std band; Y-axis zoomed to mean lines |
| **Per-axis average** | 2 rows × 3 cols | Accel X/Y/Z (top) + Gyro X/Y/Z (bottom); bold hierarchical mean + 10–90 percentile shading; all available datasets overlaid |
| **Per-axis average (zero-mean)** | same | Same, zero-mean shifted |
| **Subject overlay (norm)** | 2 rows × N cols | Each dataset in its own column; per-subject mean lines + black hierarchical mean; accel norm (top), gyro norm (bottom). Phone: 5 cols (with SBHAR), Watch: 5 cols (with PAMAP2) |
| **Subject overlay (accel XYZ)** | 3 rows × N cols | Each dataset in its own column; per-subject mean lines + black hierarchical mean; rows = X / Y / Z |
| **Subject overlay (gyro XYZ)** | 3 rows × N cols | Same as above for angular velocity |

### Cross-Activity Summary Plots (all 5 activities: walking, sitting, upstairs, downstairs, standing)

| Plot | Layout | Description |
|---|---|---|
| **Dataset means (norm)** | 2 rows × N cols | Hierarchical mean for accel norm + gyro norm; one column per activity (currently N=5) |
| **Dataset means XYZ** | 3 rows × N cols | Hierarchical mean per axis; separate figures for accel and gyro; one column per activity (currently N=5) |
| **Summary boxplots** | 2 rows × 5 cols | Segment duration (top) and rotation amplitude (bottom) across all activities |

### Root-Level Plots

| Plot | Description |
|---|---|
| **Sensor positions** | Per-activity ghost overlay comparing the 4 sensor positions (data_combined/control only) |

---

## Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `SEGMENT_LEN` | 100 | Resampled segment length (samples) |
| `MAX_GHOST` | 60 | Max ghost traces drawn per dataset |
| `GHOST_ALPHA` | 0.25 | Ghost trace transparency |
| `DC_FS` | 120.0 Hz | data_combined sampling rate |
| `OUTLIER_IQR_MULT` | 2.5 | IQR multiplier for outlier detection |
| `MIN_FILTER_CYCLES` | 12 | Minimum segments before outlier filtering activates |
| `SNIPPET_SECONDS` | 8 | Target duration for time-series snippets |

---

## Data Notes

- **WISDM stairs ambiguity** — WISDM labels stairs as activity code `C` without up/down distinction. By default this is excluded from upstairs/downstairs analyses. Use `--include-wisdm-stairs-in-updown` to mirror it into both.
- **data_combined Euler→angular velocity** — Roll/Pitch/Yaw angles are unwrapped (to avoid ±180° discontinuities), converted to radians, then differentiated to yield rad/s. This is an approximation that works well for small-angle motions.
- **Hierarchical averaging** — Critical for fair comparison: a subject with 200 segments doesn't dominate one with 20. Mean-of-subject-means gives each subject equal weight.
- **PAMAP2 device mapping** — PAMAP2 uses a wrist-worn IMU, so it maps to the **watch** position. It only appears in watch-position plots.
- **SBHAR device mapping** — SBHAR uses a smartphone IMU, so it maps to the **phone** position. It only appears in phone-position plots. SBHAR accel is natively in g and is multiplied by 9.81 to convert to m/s².
- **Dynamic column count** — Subject overlay plots and time-series plots dynamically adjust their column count based on how many datasets have data for each device type.
