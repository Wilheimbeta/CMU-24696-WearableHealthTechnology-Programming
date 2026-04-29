#!/usr/bin/env python
"""
Build experiment-level CSV summaries from JSON result files in loso_results/.

Examples:
    python summarize_experiment_results.py --exp exp8
    python summarize_experiment_results.py --exp 5
    python summarize_experiment_results.py --all
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BASE_DIR / "loso_results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "experiment_tables"


def _resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)


def _norm_exp(exp: str | None) -> str | None:
    if not exp:
        return None
    exp = str(exp).strip().lower()
    if not exp:
        return None
    if exp.startswith("exp"):
        exp = exp[3:]
    if not exp.isdigit():
        return None
    return f"exp{int(exp)}"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_tag(path: Path, data: dict[str, Any]) -> str:
    tag = str(data.get("experiment_tag") or "").strip()
    if tag:
        return tag
    m = re.search(r"(exp\d+[a-z]?(?:_[A-Za-z0-9-]+)*)", path.stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return path.stem


def _exp_from_tag(tag: str) -> str | None:
    m = re.match(r"(exp\d+)", tag or "", flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def _run_from_tag(tag: str) -> int | None:
    m = re.search(r"(?:^|_)run(\d+)(?:_|$)", tag or "", flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"exp\d+_run(\d+)", tag or "", flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _model_display(model: str) -> str:
    m = (model or "").lower()
    if "limu-bert" in m:
        return "LIMU-BERT-X" if "x" in m else "LIMU-BERT"
    if "ssl-wearables" in m:
        return "SSL-Wearables"
    if "resnet-baseline" in m:
        return "ResNet-Baseline"
    if "hart" in m:
        return "HART"
    return model or "Unknown"


def _channels(data: dict[str, Any]) -> str:
    ch = data.get("channels")
    if isinstance(ch, int) and ch > 0:
        return f"{ch}ch"
    model = str(data.get("model", ""))
    m = re.search(r"-(\d)ch", model)
    if m:
        return f"{m.group(1)}ch"
    return ""


def _bool_yes_no(v: Any) -> str:
    if v is None:
        return "No"
    if isinstance(v, str):
        return "Yes" if v.strip() else "No"
    return "Yes" if bool(v) else "No"


def _fmt_percent(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return ""


def _infer_kind(data: dict[str, Any]) -> str:
    if str(data.get("experiment", "")).lower().startswith("exp9") or "unknown_eval" in data:
        return "single_unknown"
    if data.get("open_set") or "open_set_metrics" in data:
        return "open_set"
    if str(data.get("mode", "")).lower() == "cross" or "test_metrics" in data:
        return "cross"
    if str(data.get("mode", "")).lower() == "loso" or "per_fold" in data:
        return "loso"
    return "unknown"


def _infer_datasets(data: dict[str, Any], tag: str, kind: str) -> tuple[str, str]:
    exp = _exp_from_tag(tag)
    condition = str(data.get("condition", "")).lower()
    if kind == "cross":
        direction = str(data.get("direction", ""))
        if "->" in direction:
            left, right = [x.strip() for x in direction.split("->", 1)]
            return left, right
        if str(data.get("experiment", "")).lower() == "exp8_custom_data" or exp == "exp8":
            test_map = {
                "control": "Our Controlled",
                "uncontrolled": "Our Uncontrolled",
                "all": "Our All",
            }
            return "HHAR", test_map.get(condition, "Custom Data")
        return "", ""
    if kind == "loso":
        if str(data.get("experiment", "")).lower() == "exp8_custom_data" or exp == "exp8":
            return "Custom Data", "Custom Data"
        return "HHAR", "HHAR"
    if kind == "open_set":
        return "HHAR", "HHAR + Unknown"
    if kind == "single_unknown":
        return "HHAR", ""
    return "", ""


def _infer_additional_pretraining(pretrained: Any, tag: str) -> str:
    text = f"{pretrained or ''} {tag or ''}".lower()
    if not text.strip():
        return "None"
    if "watch_finetune" in text or "watchft" in text or "watch_scratch" in text or "watchscratch" in text:
        return "PAMAP2 + WISDM (Watch only)"
    if "phone_finetune" in text or "phoneft" in text or "phone_scratch" in text or "phonescratch" in text:
        return "SBHAR + WISDM (Phone only)"
    if "wisdm_all" in text or "wisdmft" in text or "wisdmscratch" in text:
        return "WISDM (Watch and Phone)"
    if "limu_bert_x" in text or "mtl_best" in text:
        return "None"
    return "None"


def _infer_test_count(data: dict[str, Any], kind: str) -> int:
    if kind == "cross":
        rr = data.get("test_metrics", {}).get("repeat_summary", {})
        if isinstance(rr, dict) and rr.get("num_repeats"):
            return int(rr["num_repeats"])
        rs = data.get("repeated_seed_summary", {})
        if isinstance(rs, dict) and rs.get("runs"):
            return int(rs["runs"])
        return 1
    if kind == "loso":
        rs = data.get("repeated_seed_summary", {})
        if isinstance(rs, dict) and rs.get("runs"):
            return int(rs["runs"])
        if data.get("num_folds"):
            return int(data["num_folds"])
        return 1
    if kind == "open_set" or kind == "single_unknown":
        rs = data.get("repeated_seed_summary", {})
        if isinstance(rs, dict) and rs.get("runs"):
            return int(rs["runs"])
        return 1
    return 1


def _testing_method(data: dict[str, Any], kind: str) -> str:
    protocol = str(data.get("protocol", "")).lower()
    if protocol == "leave_one_in":
        return "Leave-One-In"
    if kind == "cross":
        return "Cross"
    if kind == "loso":
        return "LOSO"
    if kind == "open_set":
        return "Open-Set"
    if kind == "single_unknown":
        return "Single-Unknown"
    return "Unknown"


_PER_CLASS_LABELS = ["Sitting", "Standing", "Walking", "Upstairs", "Downstairs"]


def _extract_metrics(data: dict[str, Any], kind: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    if kind == "cross":
        tm = data.get("test_metrics", {})
        std_tm = data.get("std_test_metrics", {})
        rr = tm.get("repeat_summary", {}) if isinstance(tm, dict) else {}
        rss = (data.get("repeated_seed_summary", {}) or {}).get("metrics_mean_std", {}) or {}
        std_acc = std_tm.get("accuracy")
        std_f1w = std_tm.get("f1_weighted")
        std_f1m = std_tm.get("f1_macro")
        if std_acc is None:
            std_acc = rr.get("std_accuracy")
        if std_f1w is None:
            std_f1w = rr.get("std_f1_weighted")
        if std_f1m is None:
            std_f1m = rr.get("std_f1_macro")
        # Fallback to repeated_seed_summary.metrics_mean_std if still missing.
        if std_acc is None and isinstance(rss.get("accuracy"), dict):
            std_acc = rss["accuracy"].get("std")
        if std_f1w is None and isinstance(rss.get("f1_weighted"), dict):
            std_f1w = rss["f1_weighted"].get("std")
        if std_f1m is None and isinstance(rss.get("f1_macro"), dict):
            std_f1m = rss["f1_macro"].get("std")
        return (
            tm.get("accuracy"),
            std_acc,
            tm.get("f1_weighted"),
            std_f1w,
            tm.get("f1_macro"),
            std_f1m,
        )
    if kind == "loso":
        return (
            data.get("mean_accuracy"),
            data.get("std_accuracy"),
            data.get("mean_f1_weighted"),
            data.get("std_f1_weighted"),
            data.get("mean_f1_macro"),
            data.get("std_f1_macro"),
        )
    if kind == "open_set":
        m = data.get("open_set_metrics", {})
        rs = data.get("repeated_seed_summary", {}).get("metrics_mean_std", {})
        return (
            m.get("open_set_accuracy"),
            rs.get("open_set_accuracy", {}).get("std"),
            None,
            None,
            m.get("open_set_f1_macro"),
            rs.get("open_set_f1_macro", {}).get("std"),
        )
    return None, None, None, None, None, None


_STAIR_ALIASES = {
    "Upstairs": ("Upstairs", "Upstair"),
    "Downstairs": ("Downstairs",),
}


def _stairs_fallback_pct(data: dict[str, Any], cls: str) -> float | None:
    """Backward-compat fallback: derive stairs recall from stairs_analysis.

    If test_metrics.per_class_f1 has zero or missing support for stair
    classes, read stairs_analysis.prediction_pct and combine Upstair +
    Downstairs predictions as the "stair detection rate" for both stair
    columns (since source datasets typically have a single 'stairs' label).
    Returns a float in [0,1] or None.
    """
    sa = data.get("stairs_analysis")
    if not isinstance(sa, dict):
        return None
    pct = sa.get("prediction_pct")
    if not isinstance(pct, dict):
        return None

    stair_pct = 0.0
    for key in ("Upstairs", "Upstair", "Downstairs"):
        v = pct.get(key)
        if v is None:
            continue
        try:
            stair_pct += float(v)
        except (TypeError, ValueError):
            continue
    if stair_pct <= 0:
        return None
    return stair_pct / 100.0


def _extract_per_class_accuracy(data: dict[str, Any], kind: str) -> dict[str, str]:
    """Extract per-class recall (= per-activity accuracy) from the JSON result.

    Returns a dict mapping column names like 'Acc Sitting' / 'Std Acc Sitting'
    to formatted percentage strings. Works for cross, loso, and finetune
    results. Falls back to stairs_analysis for stair columns when per_class_f1
    lacks stair support (e.g. primary eval was 3-class).
    """
    pc: dict[str, Any] = {}
    pc_std: dict[str, float] = {}
    if kind == "cross":
        pc = data.get("test_metrics", {}).get("per_class_f1", {})
        # Std for cross repeated-seed runs lives in repeated_seed_summary.
        rec_ms = (data.get("repeated_seed_summary", {})
                      .get("per_class_recall_mean_std", {}))
        for cls in _PER_CLASS_LABELS:
            for alias in _STAIR_ALIASES.get(cls, (cls,)):
                if alias in rec_ms and isinstance(rec_ms[alias], dict):
                    s = rec_ms[alias].get("std")
                    if s is not None:
                        pc_std[cls] = float(s)
                        break
    elif kind == "loso":
        per_fold = data.get("per_fold", [])
        if per_fold:
            import numpy as _np
            for cls in _PER_CLASS_LABELS:
                aliases = _STAIR_ALIASES.get(cls, (cls,))
                vals: list[float] = []
                for f in per_fold:
                    if not isinstance(f, dict):
                        continue
                    f_pc = f.get("per_class_f1", {})
                    for alias in aliases:
                        r = f_pc.get(alias, {}).get("recall")
                        if r is not None:
                            vals.append(float(r))
                            break
                if vals:
                    pc.setdefault(cls, {})["recall"] = float(_np.mean(vals))
                    if len(vals) > 1:
                        pc_std[cls] = float(_np.std(vals))
        if not pc:
            pc = data.get("test_metrics", {}).get("per_class_f1", {})

    out: dict[str, str] = {}
    for cls in _PER_CLASS_LABELS:
        recall = None
        for alias in _STAIR_ALIASES.get(cls, (cls,)):
            if alias in pc:
                support = pc[alias].get("support")
                r = pc[alias].get("recall")
                if r is not None and (support is None or support > 0):
                    recall = r
                    break
        if recall is None and cls in ("Upstairs", "Downstairs"):
            recall = _stairs_fallback_pct(data, cls)
        out[f"Acc {cls}"] = _fmt_percent(recall)
        out[f"Std Acc {cls}"] = _fmt_percent(pc_std.get(cls))
    return out


def _build_row(path: Path, data: dict[str, Any], tag: str) -> dict[str, Any]:
    kind = _infer_kind(data)
    model_name = _model_display(str(data.get("model", "")))
    train_ds, test_ds = _infer_datasets(data, tag, kind)
    pretrained = data.get("pretrained")
    accuracy, std_acc, f1w, std_f1w, f1m, std_f1m = _extract_metrics(data, kind)
    data_fraction_text = ""
    meta_proto = (data.get("experiment_table_metadata") or {}).get("Protocol", "")
    if meta_proto:
        data_fraction_text = meta_proto
    else:
        protocol = data.get("protocol")
        if protocol == "data-fraction":
            psf = data.get("per_subject_fraction")
            if psf is not None:
                data_fraction_text = f"Frac {float(psf)*100:.0f}%/subj"
        elif protocol == "leave-in":
            ns = data.get("finetune_subjects")
            if ns is not None:
                data_fraction_text = f"Leave-{ns}-in"
    if not data_fraction_text:
        raw = data.get("data_fraction", 1.0)
        if isinstance(raw, str):
            data_fraction_text = raw
        else:
            try:
                v = float(raw)
                data_fraction_text = f"Frac {v*100:.0f}%/subj" if v < 1.0 else "100%"
            except Exception:
                data_fraction_text = "100%"

    per_class_acc = _extract_per_class_accuracy(data, kind)

    row = {
        "Experiment": (_exp_from_tag(tag) or ""),
        "Architecture Type": model_name,
        "Device Type": str(data.get("device_type", "")).capitalize(),
        "Training Dataset": train_ds,
        "Testing Dataset": test_ds,
        "Imported Pretrained Weights": _bool_yes_no(pretrained),
        "3ch vs 6ch data": _channels(data),
        "Data Fraction": data_fraction_text,
        "Additional Pretraining dataset": _infer_additional_pretraining(pretrained, tag),
        "Testing Method": _testing_method(data, kind),
        "# of Tests": _infer_test_count(data, kind),
        "# Run": _run_from_tag(tag) or "",
        "Experiment Tag": tag,
        "Accuracy": _fmt_percent(accuracy),
        "Std Accuracy": _fmt_percent(std_acc),
        "F1 weighted": _fmt_percent(f1w),
        "Std F1 weighted": _fmt_percent(std_f1w),
        "F1 macro": _fmt_percent(f1m),
        "Std F1 macro": _fmt_percent(std_f1m),
        **per_class_acc,
        "Result File": path.name,
        "Saved At": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta = data.get("experiment_table_metadata", {})
    if isinstance(meta, dict):
        for key in (
            "Architecture Type",
            "Device Type",
            "Training Dataset",
            "Testing Dataset",
            "Testing Dataset (Task)",
            "Imported Pretrained Weights",
            "3ch vs 6ch data",
            "Data Fraction",
            "Additional Pretraining dataset",
            "Testing Method",
            "# of Tests",
            "# Run",
            "Expected Label",
            "AVG Wrongly assigned to Sitting",
            "AVG Wrongly assigned to Standing",
            "AVG Wrongly assigned to Walking",
            "AVG Wrongly assigned to Stairup",
            "AVG Wrongly assigned to Stairdown",
            "SD Wrongly assigned to Sitting",
            "SD Wrongly assigned to Standing",
            "SD Wrongly assigned to Walking",
            "SD Wrongly assigned to Stairup",
            "SD Wrongly assigned to Stairdown",
        ):
            if meta.get(key) not in (None, ""):
                row[key] = meta.get(key)
    return row


def _select_latest(files: list[tuple[Path, dict[str, Any], str]]) -> list[tuple[Path, dict[str, Any], str]]:
    best: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for path, data, tag in files:
        key = tag or path.stem
        prev = best.get(key)
        if prev is None:
            best[key] = (path, data, tag)
            continue
        prev_data = prev[1]
        cur_rep = bool(data.get("repeated_seed_summary"))
        prev_rep = bool(prev_data.get("repeated_seed_summary"))
        if cur_rep and not prev_rep:
            best[key] = (path, data, tag)
            continue
        if cur_rep == prev_rep and path.stat().st_mtime > prev[0].stat().st_mtime:
            best[key] = (path, data, tag)
    return list(best.values())


def summarize_experiments(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    exp: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    latest_only: bool = True,
) -> Path:
    results_dir = _resolve_project_path(results_dir)
    output_dir = _resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_norm = _norm_exp(exp)
    candidates: list[tuple[Path, dict[str, Any], str]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_progress.json"):
            continue
        data = _load_json(path)
        if not data:
            continue
        tag = _infer_tag(path, data)
        exp_tag = _exp_from_tag(tag)
        if exp_norm and exp_tag != exp_norm:
            continue
        candidates.append((path, data, tag))

    if latest_only:
        candidates = _select_latest(candidates)

    rows = [_build_row(path, data, tag) for path, data, tag in candidates]
    rows.sort(key=lambda r: (str(r.get("Experiment")), int(r.get("# Run") or 10**9), str(r.get("Experiment Tag"))))

    import csv

    out_name = f"{exp_norm}_summary.csv" if exp_norm else "all_experiments_summary.csv"
    out_path = output_dir / out_name
    cols = [
        "Experiment",
        "Architecture Type",
        "Device Type",
        "Testing Dataset",
        "Testing Dataset (Task)",
        "Imported Pretrained Weights",
        "3ch vs 6ch data",
        "Data Fraction",
        "Additional Pretraining dataset",
        "Training Dataset",
        "Testing Method",
        "Expected Label",
        "# of Tests",
        "# Run",
        "Experiment Tag",
        "Accuracy",
        "Std Accuracy",
        "F1 weighted",
        "Std F1 weighted",
        "F1 macro",
        "Std F1 macro",
        "Acc Sitting",
        "Std Acc Sitting",
        "Acc Standing",
        "Std Acc Standing",
        "Acc Walking",
        "Std Acc Walking",
        "Acc Upstairs",
        "Std Acc Upstairs",
        "Acc Downstairs",
        "Std Acc Downstairs",
        "AVG Wrongly assigned to Sitting",
        "AVG Wrongly assigned to Standing",
        "AVG Wrongly assigned to Walking",
        "AVG Wrongly assigned to Stairup",
        "AVG Wrongly assigned to Stairdown",
        "SD Wrongly assigned to Sitting",
        "SD Wrongly assigned to Standing",
        "SD Wrongly assigned to Walking",
        "SD Wrongly assigned to Stairup",
        "SD Wrongly assigned to Stairdown",
        "Result File",
        "Saved At",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})

    return out_path


def update_for_experiment_tag(
    experiment_tag: str | None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path | None:
    exp = _exp_from_tag(str(experiment_tag or ""))
    if not exp:
        return None
    return summarize_experiments(results_dir=results_dir, exp=exp, output_dir=output_dir, latest_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment JSON results into CSV.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--exp", default=None, help="Experiment id, e.g. exp8 or 8")
    parser.add_argument("--all", action="store_true", help="Summarize all experiments")
    parser.add_argument("--include-all-runs", action="store_true", help="Keep all JSONs (not only latest per tag)")
    args = parser.parse_args()

    if args.all:
        out = summarize_experiments(
            results_dir=args.results_dir,
            exp=None,
            output_dir=args.output_dir,
            latest_only=not args.include_all_runs,
        )
    else:
        out = summarize_experiments(
            results_dir=args.results_dir,
            exp=args.exp,
            output_dir=args.output_dir,
            latest_only=not args.include_all_runs,
        )
    print(f"Summary CSV saved -> {out}")


if __name__ == "__main__":
    main()
