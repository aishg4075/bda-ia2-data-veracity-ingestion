#!/usr/bin/env python3
"""Create a small, class-aware sample from a full click-fraud dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ingestion_trust.core import make_small_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Full input CSV")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_click_fraud.csv",
        help="Sample output CSV",
    )
    parser.add_argument("--max-rows", type=int, default=90000, help="Max row count in sample")
    parser.add_argument(
        "--target-size-mb",
        type=float,
        default=15.0,
        help="Approximate maximum file size",
    )
    parser.add_argument(
        "--stratify-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stratify sample by label when possible",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = make_small_sample(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        max_rows=args.max_rows,
        target_size_mb=args.target_size_mb,
        stratify_label=args.stratify_label,
    )
    print(f"Sample created at: {out}")
    print(f"Size MB: {out.stat().st_size / (1024 * 1024):.3f}")


if __name__ == "__main__":
    main()
