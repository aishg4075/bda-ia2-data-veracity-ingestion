#!/usr/bin/env python3
"""Download click-fraud dataset with robust fallback to synthetic generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ingestion_trust.core import load_dataset_with_fallback, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Output directory for downloaded or generated dataset",
    )
    parser.add_argument(
        "--prefer-kaggle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt Kaggle first when credentials are configured",
    )
    parser.add_argument(
        "--open-url",
        action="append",
        default=[],
        help="Open CSV URL candidate (can be passed multiple times)",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=120000,
        help="Synthetic fallback row count",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download / re-generation even if a local file exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_dataset_with_fallback(
        output_dir=args.output_dir,
        prefer_kaggle=args.prefer_kaggle,
        open_urls=args.open_url,
        synthetic_rows=args.synthetic_rows,
        force=args.force,
    )

    metadata_path = PROJECT_ROOT / "artifacts" / "provenance" / "dataset_download_metadata.json"
    save_json(meta, metadata_path)

    print("Dataset resolution complete")
    print(json.dumps(meta, indent=2))
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
