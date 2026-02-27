from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .winner_style import (
    TrackMode,
    add_count_unique_cum_features,
    add_delta_time_features,
    build_competition_like_splits,
    build_rate_features_train_only,
    build_temporal_splits,
    load_talkingdata_frame,
    make_bag_subsample_indices,
    select_row_window_for_support,
    split_support_counts,
    validate_split_integrity,
)
from .modeling_pipeline import WinnerStyleConfig, train_and_evaluate_models, train_lightgbm_bagging
from .trust_plots import plot_data_quality_summary, plot_dq_rule_percentage, plot_fraud_prevalence

DEFAULT_OPEN_DATA_URLS: List[str] = [
    # Lightweight open sample mirrors. These may change; failures are handled gracefully.
    "https://raw.githubusercontent.com/yzhao062/FraudDetection/master/data/creditcard.csv",
]


EXPECTED_COLUMNS: List[str] = [
    "click_time",
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "is_fraud",
    "is_attributed",
    "attributed_time",
]


NUMERIC_BASE_COLUMNS: List[str] = ["ip", "app", "device", "os", "channel", "is_fraud", "is_attributed"]



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Mapping[str, Any], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def hash_record(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def build_merkle_root(record_hashes: Sequence[str]) -> str:
    if not record_hashes:
        return hashlib.sha256(b"EMPTY").hexdigest()
    level = list(record_hashes)
    while len(level) > 1:
        next_level: List[str] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else left
            next_level.append(hashlib.sha256(f"{left}{right}".encode("utf-8")).hexdigest())
        level = next_level
    return level[0]


def _read_last_batch_hash(log_path: Path) -> str:
    if not log_path.exists():
        return "0" * 64
    last = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return "0" * 64
    try:
        return json.loads(last).get("batch_hash", "0" * 64)
    except json.JSONDecodeError:
        return "0" * 64


def append_audit_batch(
    log_path: Path,
    batch_id: str,
    records: Iterable[Mapping[str, Any]],
    batch_metadata: Mapping[str, Any],
    prev_batch_hash: Optional[str] = None,
    include_record_hashes: bool = True,
) -> Dict[str, Any]:
    record_hashes = [hash_record(r) for r in records]
    merkle_root = build_merkle_root(record_hashes)
    prev_hash = prev_batch_hash or _read_last_batch_hash(log_path)
    canonical_meta = canonical_json(batch_metadata)
    record_count = len(record_hashes)
    batch_hash_input = f"{prev_hash}{merkle_root}{canonical_meta}{batch_id}{record_count}"
    batch_hash = hashlib.sha256(batch_hash_input.encode("utf-8")).hexdigest()

    entry: Dict[str, Any] = {
        "batch_id": str(batch_id),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "record_count": record_count,
        "merkle_root": merkle_root,
        "prev_batch_hash": prev_hash,
        "batch_metadata": dict(batch_metadata),
        "batch_hash": batch_hash,
    }
    if include_record_hashes:
        entry["record_hashes"] = record_hashes

    ensure_parent(log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, default=_json_default) + "\n")
    return entry


def verify_audit_log(log_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": True,
        "checked_batches": 0,
        "first_mismatch_index": None,
        "reason": None,
    }
    if not log_path.exists():
        result["success"] = False
        result["reason"] = "audit log missing"
        return result

    prev_hash = "0" * 64
    with log_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                result.update(
                    {
                        "success": False,
                        "first_mismatch_index": idx,
                        "reason": "invalid json line",
                    }
                )
                return result

            if entry.get("prev_batch_hash") != prev_hash:
                result.update(
                    {
                        "success": False,
                        "first_mismatch_index": idx,
                        "reason": "prev hash mismatch",
                    }
                )
                return result

            record_hashes = entry.get("record_hashes", [])
            merkle_expected = build_merkle_root(record_hashes)
            if merkle_expected != entry.get("merkle_root"):
                result.update(
                    {
                        "success": False,
                        "first_mismatch_index": idx,
                        "reason": "merkle root mismatch",
                    }
                )
                return result

            canonical_meta = canonical_json(entry.get("batch_metadata", {}))
            batch_id = str(entry.get("batch_id", ""))
            record_count = int(entry.get("record_count", len(record_hashes)))
            batch_hash_input = (
                f"{entry.get('prev_batch_hash', '')}{entry.get('merkle_root', '')}"
                f"{canonical_meta}{batch_id}{record_count}"
            )
            expected_batch_hash = hashlib.sha256(batch_hash_input.encode("utf-8")).hexdigest()
            if expected_batch_hash != entry.get("batch_hash"):
                result.update(
                    {
                        "success": False,
                        "first_mismatch_index": idx,
                        "reason": "batch hash mismatch",
                    }
                )
                return result

            prev_hash = entry.get("batch_hash", "")
            result["checked_batches"] += 1

    return result


def tamper_log_copy(source_log: Path, tampered_log: Path) -> Path:
    ensure_parent(tampered_log)
    shutil.copy2(source_log, tampered_log)
    lines = tampered_log.read_text(encoding="utf-8").splitlines()
    if not lines:
        return tampered_log

    first = json.loads(lines[0])
    metadata = first.get("batch_metadata", {})
    metadata["tampered"] = True
    first["batch_metadata"] = metadata
    lines[0] = json.dumps(first, sort_keys=True)
    tampered_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tampered_log


def _safe_run(cmd: Sequence[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_commit_hash(repo_dir: Optional[Path]) -> Optional[str]:
    if not repo_dir:
        return None
    code, out, _ = _safe_run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"])
    return out if code == 0 else None


def _try_import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def build_run_metadata(
    dataset_path: Path,
    dataset_source: str,
    dataset_mode: str,
    column_schema: Sequence[Mapping[str, str]],
    repo_dir: Optional[Path] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path) if dataset_path.exists() else None,
        "dataset_mode": dataset_mode,
        "column_schema": list(column_schema),
        "git_commit_hash": _git_commit_hash(repo_dir),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "pyspark_version": _try_import_version("pyspark"),
            "numpy_version": _try_import_version("numpy"),
            "pandas_version": _try_import_version("pandas"),
            "sklearn_version": _try_import_version("sklearn"),
            "xgboost_version": _try_import_version("xgboost"),
        },
    }
    if extra:
        metadata["extra"] = dict(extra)
    return metadata


def detect_dataset_mode(dataset_path: Path, fallback_generated: bool) -> str:
    if fallback_generated:
        return "fallback_synthetic"
    if dataset_path.name.startswith("sample_"):
        return "sample"
    return "full"


def generate_synthetic_click_fraud(
    output_csv: Path,
    n_rows: int = 50_000,
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    base_ts = datetime(2017, 11, 6, 0, 0, 0)
    ip_pool = rng.integers(10000, 65000, size=max(2500, n_rows // 6))
    suspicious_ips = set(rng.choice(ip_pool, size=max(20, len(ip_pool) // 18), replace=False).tolist())

    app_vals = np.array([1, 2, 3, 6, 9, 12, 15, 18, 21, 24], dtype=int)
    device_vals = np.array([1, 2, 3, 4, 5], dtype=int)
    os_vals = np.array([9, 13, 17, 19, 20, 22, 24, 26], dtype=int)
    channel_vals = np.array([111, 112, 113, 121, 134, 145, 233, 255, 303], dtype=int)
    suspicious_channels = {233, 255, 303}

    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        ip = int(rng.choice(ip_pool))
        app = int(rng.choice(app_vals))
        device = int(rng.choice(device_vals))
        os_code = int(rng.choice(os_vals))
        channel = int(rng.choice(channel_vals))

        # Dense bursts create realistic high-velocity behavior.
        minute_offset = int(rng.integers(0, 60 * 24 * 7))
        if ip in suspicious_ips and rng.random() < 0.35:
            minute_offset = int(rng.integers(0, 60 * 12))
        click_time = base_ts + timedelta(minutes=minute_offset, seconds=int(rng.integers(0, 60)))

        risk = 0.01
        risk += 0.22 if ip in suspicious_ips else 0.0
        risk += 0.10 if channel in suspicious_channels else 0.0
        risk += 0.06 if app in {21, 24} else 0.0
        risk += 0.04 if device == 1 and os_code in {9, 13} else 0.0
        risk += float(rng.normal(0.0, 0.015))
        risk = max(0.001, min(0.92, risk))

        is_fraud = int(rng.random() < risk)
        is_attributed = int(0 if is_fraud else (rng.random() < 0.24))

        attributed_time = None
        if is_attributed:
            attributed_time = click_time + timedelta(minutes=int(rng.integers(1, 90)))

        rows.append(
            {
                "click_time": click_time.isoformat(sep=" "),
                "ip": ip,
                "app": app,
                "device": device,
                "os": os_code,
                "channel": channel,
                "is_fraud": is_fraud,
                "is_attributed": is_attributed,
                "attributed_time": attributed_time.isoformat(sep=" ") if attributed_time else "",
            }
        )

    df = pd.DataFrame(rows)
    # Inject a small number of duplicates and null-like rows for validation demonstration.
    if len(df) > 1000:
        dup = df.sample(n=max(5, len(df) // 500), random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)
        null_ix = df.sample(n=max(10, len(df) // 800), random_state=seed + 1).index
        df.loc[null_ix, "channel"] = np.nan

    df.to_csv(output_csv, index=False)
    return output_csv


def _looks_like_click_fraud_schema(df: pd.DataFrame) -> bool:
    required_any = {"click_time", "ip", "app", "device", "os", "channel"}
    return len(required_any.intersection(df.columns)) >= 4


def _normalize_columns_for_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    col_map = {c.lower(): c for c in normalized.columns}

    if "click_time" not in normalized.columns:
        for candidate in ["timestamp", "event_time", "ts"]:
            if candidate in col_map:
                normalized = normalized.rename(columns={col_map[candidate]: "click_time"})
                break

    if "is_fraud" not in normalized.columns:
        if "is_attributed" in normalized.columns:
            normalized["is_fraud"] = 1 - normalized["is_attributed"].astype(int)
        elif "label" in normalized.columns:
            normalized = normalized.rename(columns={"label": "is_fraud"})

    for col in ["ip", "app", "device", "os", "channel", "is_attributed"]:
        if col not in normalized.columns:
            normalized[col] = np.nan

    if "attributed_time" not in normalized.columns:
        normalized["attributed_time"] = ""

    keep = [c for c in EXPECTED_COLUMNS if c in normalized.columns]
    return normalized[keep]


def _attempt_download(url: str, dest_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    try:
        import requests
    except Exception:
        return False, "requests not installed"

    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return False, f"status={r.status_code}"
            ensure_parent(dest_path)
            with dest_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
        return True, "ok"
    except Exception as exc:  # pragma: no cover - environment/network variability
        return False, str(exc)


def _attempt_kaggle_download(output_dir: Path) -> Tuple[bool, Optional[Path], str]:
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        return False, None, "missing ~/.kaggle/kaggle.json"
    if shutil.which("kaggle") is None:
        return False, None, "kaggle CLI not installed"

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "talkingdata-adtracking-fraud-detection.zip"
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "talkingdata-adtracking-fraud-detection",
        "-p",
        str(output_dir),
    ]
    code, _, err = _safe_run(cmd)
    if code != 0:
        return False, None, err or "kaggle download failed"

    if not zip_path.exists():
        # Kaggle may emit multiple files directly. Search for a csv.
        csv_candidates = sorted(output_dir.glob("*.csv"))
        if csv_candidates:
            return True, csv_candidates[0], "downloaded"
        return False, None, "kaggle output missing expected files"

    extract_dir = output_dir / "kaggle_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    csv_candidates = sorted(extract_dir.rglob("*.csv"))
    if not csv_candidates:
        return False, None, "no csv inside kaggle archive"

    # Prefer train-like file for labels.
    preferred = None
    for c in csv_candidates:
        name = c.name.lower()
        if "train" in name:
            preferred = c
            break
    return True, preferred or csv_candidates[0], "downloaded"


def load_dataset_with_fallback(
    output_dir: Path,
    prefer_kaggle: bool = True,
    open_urls: Optional[Sequence[str]] = None,
    synthetic_rows: int = 120_000,
    force: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = output_dir / "click_fraud_full.csv"
    meta: Dict[str, Any] = {
        "dataset_path": str(final_csv),
        "dataset_source": None,
        "dataset_mode": None,
        "logs": [],
    }

    if final_csv.exists() and not force:
        meta["dataset_source"] = "existing_local"
        meta["dataset_mode"] = "full"
        meta["dataset_sha256"] = sha256_file(final_csv)
        return meta

    if prefer_kaggle:
        ok, csv_path, msg = _attempt_kaggle_download(output_dir)
        meta["logs"].append(f"kaggle: {msg}")
        if ok and csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            df = _normalize_columns_for_schema(df)
            if not _looks_like_click_fraud_schema(df):
                meta["logs"].append("kaggle schema does not match expected clickstream columns")
            else:
                df.to_csv(final_csv, index=False)
                meta.update(
                    {
                        "dataset_source": "kaggle:talkingdata-adtracking-fraud-detection",
                        "dataset_mode": "full",
                        "dataset_sha256": sha256_file(final_csv),
                    }
                )
                return meta

    for url in (list(open_urls) if open_urls else DEFAULT_OPEN_DATA_URLS):
        temp_path = output_dir / "downloaded_candidate.csv"
        ok, msg = _attempt_download(url, temp_path)
        meta["logs"].append(f"open_url {url}: {msg}")
        if not ok:
            continue
        try:
            df = pd.read_csv(temp_path)
            df = _normalize_columns_for_schema(df)
            if not _looks_like_click_fraud_schema(df):
                continue
            df.to_csv(final_csv, index=False)
            meta.update(
                {
                    "dataset_source": url,
                    "dataset_mode": "full",
                    "dataset_sha256": sha256_file(final_csv),
                }
            )
            return meta
        except Exception as exc:  # pragma: no cover
            meta["logs"].append(f"parse failure for {url}: {exc}")

    generate_synthetic_click_fraud(final_csv, n_rows=synthetic_rows, seed=seed)
    meta.update(
        {
            "dataset_source": "synthetic_fallback",
            "dataset_mode": "fallback_synthetic",
            "dataset_sha256": sha256_file(final_csv),
        }
    )
    meta["logs"].append("used synthetic fallback")
    return meta


def make_small_sample(
    input_csv: Path,
    output_csv: Path,
    max_rows: int = 80_000,
    target_size_mb: float = 15.0,
    stratify_label: bool = True,
    seed: int = 42,
) -> Path:
    df = pd.read_csv(input_csv)
    if len(df) <= max_rows:
        sample_df = df.copy()
    elif stratify_label and "is_fraud" in df.columns and df["is_fraud"].nunique() > 1:
        frac = min(1.0, max_rows / len(df))
        parts: List[pd.DataFrame] = []
        for label, group in df.groupby("is_fraud"):
            n_take = max(1, int(round(len(group) * frac)))
            parts.append(group.sample(n=min(n_take, len(group)), random_state=seed))
        sample_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        sample_df = sample_df.head(max_rows)
    else:
        sample_df = df.sample(n=max_rows, random_state=seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_csv, index=False)

    # Ensure size stays under target cap.
    while output_csv.stat().st_size > int(target_size_mb * 1024 * 1024) and len(sample_df) > 5000:
        sample_df = sample_df.sample(frac=0.8, random_state=seed).reset_index(drop=True)
        sample_df.to_csv(output_csv, index=False)

    return output_csv



def measure_integrity_overhead(records: List[Mapping[str, Any]], loops: int = 5, warmup: int = 1) -> Dict[str, float]:
    if loops < 1:
        loops = 1

    plain_times: List[float] = []
    hash_times: List[float] = []

    def _plain_pass(rows: List[Mapping[str, Any]]) -> None:
        canonical_rows = [canonical_json(r) for r in rows]
        _ = sum(len(x) for x in canonical_rows)

    def _hash_pass(rows: List[Mapping[str, Any]]) -> None:
        canonical_rows = [canonical_json(r) for r in rows]
        rec_hashes = [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in canonical_rows]
        _ = build_merkle_root(rec_hashes)

    for _ in range(max(0, warmup)):
        _plain_pass(records)
        _hash_pass(records)

    for _ in range(loops):
        t0 = time.perf_counter()
        _plain_pass(records)
        plain_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        _hash_pass(records)
        hash_times.append(time.perf_counter() - t1)

    plain_arr = np.asarray(plain_times, dtype=float)
    hash_arr = np.asarray(hash_times, dtype=float)
    plain_median = float(np.median(plain_arr))
    hash_median = float(np.median(hash_arr))
    avg_plain = float(np.mean(plain_times))
    avg_hash = float(np.mean(hash_times))
    std_plain = float(np.std(plain_arr))
    std_hash = float(np.std(hash_arr))
    iqr_plain = float(np.percentile(plain_arr, 75) - np.percentile(plain_arr, 25))
    iqr_hash = float(np.percentile(hash_arr, 75) - np.percentile(hash_arr, 25))
    throughput_plain = float(len(records) / avg_plain) if avg_plain > 0 else float("inf")
    throughput_hash = float(len(records) / avg_hash) if avg_hash > 0 else float("inf")
    median_throughput_plain = float(len(records) / plain_median) if plain_median > 0 else float("inf")
    median_throughput_hash = float(len(records) / hash_median) if hash_median > 0 else float("inf")
    per_record_overhead_us = float((hash_median - plain_median) * 1e6 / max(len(records), 1))

    return {
        "records": float(len(records)),
        "loops": loops,
        "warmup": warmup,
        "plain_times_sec": plain_times,
        "hash_times_sec": hash_times,
        "avg_plain_sec": avg_plain,
        "avg_hash_sec": avg_hash,
        "median_plain_sec": plain_median,
        "median_hash_sec": hash_median,
        "std_plain_sec": std_plain,
        "std_hash_sec": std_hash,
        "iqr_plain_sec": iqr_plain,
        "iqr_hash_sec": iqr_hash,
        "overhead_sec": float(avg_hash - avg_plain),
        "median_overhead_sec": float(hash_median - plain_median),
        "per_record_overhead_us": per_record_overhead_us,
        "throughput_plain_rps": throughput_plain,
        "throughput_hash_rps": throughput_hash,
        "median_throughput_plain_rps": median_throughput_plain,
        "median_throughput_hash_rps": median_throughput_hash,
    }
