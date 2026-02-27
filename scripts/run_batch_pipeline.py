#!/usr/bin/env python3
"""Run Spark batch ETL + veracity validation + attribution-surrogate modeling pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ingestion_trust.batch_orchestrator import run_batch_pipeline as run_batch_pipeline_orchestrated  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "metrics" / "baseline_stats.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=600000)
    parser.add_argument("--row-cap", type=int, default=600000)
    parser.add_argument(
        "--approach-track",
        type=str,
        choices=["production_safe", "benchmark_transductive", "both"],
        default="both",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["baseline", "lightgbm_bagging", "all"],
        default="all",
    )
    parser.add_argument("--bag-count", type=int, default=5)
    parser.add_argument("--neg-pos-ratio", type=int, default=200)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-num-leaves", type=int, default=64)
    parser.add_argument("--lgbm-n-estimators", type=int, default=600)
    parser.add_argument("--min-val-positives", type=int, default=100)
    parser.add_argument("--min-test-positives", type=int, default=100)
    parser.add_argument("--row-cap-step", type=int, default=50000)
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        choices=["subsampling_only", "subsampling_plus_weighting"],
        default="subsampling_only",
    )
    parser.add_argument("--calibration-method", type=str, choices=["platt"], default="platt")
    parser.add_argument("--stability-seeds", type=str, default="42,43")
    parser.add_argument("--enable-next-click", action="store_true")
    parser.add_argument("--target-col", type=str, default="is_attributed")
    parser.add_argument("--allow-legacy-fraud-target", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if args.target_col not in {"is_attributed", "is_fraud"}:
        raise ValueError("target_col must be one of: is_attributed, is_fraud")
    if args.target_col == "is_fraud":
        if not args.allow_legacy_fraud_target:
            raise ValueError(
                "Target semantics hardening requires --target-col is_attributed. "
                "Legacy target is_fraud is disabled by default. "
                "If you must run legacy mode, pass --allow-legacy-fraud-target explicitly."
            )
        _legacy_semantics_warning = (
            "Legacy `is_fraud` target semantics were explicitly enabled for compatibility. "
            "This mode is deprecated and must not be used for presentation-safe primary claims."
        )
    if args.min_val_positives <= 0 or args.min_test_positives <= 0:
        raise ValueError("--min-val-positives and --min-test-positives must be > 0")
    if args.row_cap_step <= 0:
        raise ValueError("--row-cap-step must be > 0")
    tracks = (
        ["production_safe", "benchmark_transductive"]
        if args.approach_track == "both"
        else [args.approach_track]
    )
    if args.enable_next_click and "production_safe" in tracks:
        raise ValueError("--enable-next-click cannot be used when production_safe track is selected")

    summary = run_batch_pipeline_orchestrated(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
