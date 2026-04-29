"""One-shot patcher: add mean_f1_weighted / std_f1_weighted to existing LOSO
JSON files that were produced by eval_custom_data.py before the aggregation
bug was fixed. Reads per_fold[*].f1_weighted and writes mean/std at top level.

Also triggers CSV refresh for exp8 / exp85.

Safe to re-run: only adds missing keys, never overwrites.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOSO_DIR = BASE_DIR / "loso_results"

PATTERNS = [
    "exp8d_*_custom_*.json",
    "exp8f_*_custom_*.json",
    "exp85_*_custom_*.json",
]


def patch_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [SKIP] {path.name}: could not parse ({e})")
        return False

    # Only patch LOSO-mode results.
    if str(data.get("mode", "")).lower() != "loso":
        return False

    # Already patched?
    if "mean_f1_weighted" in data and "std_f1_weighted" in data:
        return False

    per_fold = data.get("per_fold", [])
    f1w = [f.get("f1_weighted") for f in per_fold if isinstance(f, dict)]
    f1w = [v for v in f1w if v is not None]
    if not f1w:
        # No per-fold data to aggregate; nothing to do.
        return False

    data["mean_f1_weighted"] = float(np.mean(f1w))
    data["std_f1_weighted"] = float(np.std(f1w))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return True


def main():
    if not LOSO_DIR.exists():
        print(f"ERROR: {LOSO_DIR} not found")
        return

    total = 0
    patched = 0
    for pat in PATTERNS:
        for p in sorted(LOSO_DIR.glob(pat)):
            total += 1
            if patch_file(p):
                patched += 1
                print(f"  patched: {p.name}")

    print(f"\nScanned {total} LOSO JSONs, patched {patched}.")

    # Refresh CSVs.
    try:
        from summarize_experiment_results import summarize_experiments
        for exp in ("exp8", "exp85"):
            out = summarize_experiments(exp=exp)
            print(f"  refreshed CSV: {out}")
    except Exception as e:
        print(f"  WARNING: failed to refresh CSV ({e})")


if __name__ == "__main__":
    main()
