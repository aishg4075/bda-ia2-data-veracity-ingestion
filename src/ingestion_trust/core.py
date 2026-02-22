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


@dataclass
class WinnerStyleConfig:
    track_mode: TrackMode = "production_safe"
    active_target_col: str = "is_attributed"
    bag_count: int = 5
    neg_pos_ratio: int = 200
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 64
    lgbm_n_estimators: int = 600
    calibration_method: Literal["platt"] = "platt"
    imbalance_strategy: Literal["subsampling_only", "subsampling_plus_weighting"] = "subsampling_only"
    min_val_positives: int = 100
    min_test_positives: int = 100
    row_cap_step: int = 50_000
    oof_folds: int = 5
    oof_smoothing_alpha: float = 20.0
    history_carryover_enabled: bool = True
    enable_next_click: bool = False
    random_seed: int = 42


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


def _compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    except Exception:
        metrics["pr_auc"] = float("nan")
    return metrics


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocess, numeric_cols, categorical_cols


def _track_claim_label(track_mode: str) -> str:
    return "Causal / Deployment-safe" if track_mode == "production_safe" else "Kaggle-style Benchmark (Non-causal / Not for production claims)"


def _plot_roc_curves(
    y_true: np.ndarray,
    pred_probs: Dict[str, np.ndarray],
    out_path: Path,
    track_mode: str,
    target_label: str,
) -> None:
    pos = int(np.sum(y_true == 1))
    plt.figure(figsize=(8, 6))
    for name, probs in pred_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        try:
            auc = roc_auc_score(y_true, probs)
            auc_label = f"{auc:.3f}"
        except Exception:
            auc_label = "N/A"
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_label})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({target_label}, positive support={pos})")
    plt.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pr_curves(
    y_true: np.ndarray,
    pred_probs: Dict[str, np.ndarray],
    out_path: Path,
    track_mode: str,
    target_label: str,
) -> None:
    plt.figure(figsize=(8, 6))
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    pos = int(np.sum(y_true == 1))
    for name, probs in pred_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        try:
            ap = average_precision_score(y_true, probs)
            ap_label = f"{ap:.3f}"
        except Exception:
            ap_label = "N/A"
        plt.plot(recall, precision, label=f"{name} (AP={ap_label})")
    plt.hlines(prevalence, xmin=0, xmax=1, colors="gray", linestyles="--", label="Prevalence")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves ({target_label}, positive support={pos})")
    plt.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.legend(loc="best")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_metric_bars(metrics_df: pd.DataFrame, out_path: Path, track_mode: str) -> None:
    metric_cols = ["precision", "recall", "f1", "pr_auc", "mcc"]
    available = [c for c in metric_cols if c in metrics_df.columns]
    if not available:
        return
    plot_df = metrics_df[["model", *available]].melt(id_vars="model", var_name="metric", value_name="score")
    plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    y_min = float(np.nanmin(plot_df["score"])) if not plot_df["score"].isna().all() else 0.0
    y_low = min(-1.0, y_min - 0.05) if y_min < 0 else 0.0
    plt.ylim(y_low, 1.0)
    plt.title("Model Metrics Comparison (Imbalance-focused, is_attributed surrogate target)")
    plt.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.legend(title="Metric")
    for container in ax.containers:
        labels = []
        for val in container.datavalues:
            labels.append("N/A" if pd.isna(val) else f"{val:.3f}")
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
    track_mode: str,
) -> None:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    cm_norm = cm / total if total else np.zeros_like(cm, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_norm, cmap="Blues", annot=False, cbar=True, vmin=0, vmax=max(0.5, float(cm_norm.max())), ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.2f}%)"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", color="black", fontsize=10)
    ax.set_xticklabels(["not_attributed_pred", "attributed_pred"])
    ax.set_yticklabels(["not_attributed_true", "attributed_true"], rotation=0)
    ax.set_title(f"Confusion Matrix: {model_name} (is_attributed surrogate, counts + global %)")
    fig.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _clean_feature_name(name: str) -> str:
    out = name.replace("num__", "").replace("cat__", "")
    out = out.replace("[", "=").replace("]", "")
    out = out.replace("<", " lt ").replace(">", " gt ")
    out = out.replace("x0_", "")
    return textwrap.shorten(out, width=60, placeholder="...")


def _extract_feature_importance(trained_pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    preprocess: ColumnTransformer = trained_pipeline.named_steps["preprocess"]
    model = trained_pipeline.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    metric_label = "importance"
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).ravel()
        metric_label = "abs(coefficient)"
    else:
        return pd.DataFrame(columns=["feature", "importance", "metric_label"])

    imp_df = pd.DataFrame(
        {
            "feature": [ _clean_feature_name(str(name)) for name in feature_names ],
            "importance": importances,
            "metric_label": metric_label,
        }
    )
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
    return imp_df


def _plot_feature_importance(importance_df: pd.DataFrame, model_name: str, out_path: Path, track_mode: str) -> None:
    if importance_df.empty:
        return
    metric_label = str(importance_df["metric_label"].iloc[0]) if "metric_label" in importance_df.columns else "importance"
    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", orient="h")
    if metric_label == "abs(coefficient)":
        plt.title(f"Top Absolute Coefficients ({model_name})")
    else:
        plt.title(f"Top Feature Importances ({model_name})")
    plt.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.xlabel(metric_label)
    plt.ylabel("Feature")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    track_mode: str,
    probability_variant: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 30)
    neg = y_prob[y_true == 0]
    pos = y_prob[y_true == 1]
    if len(neg) > 0:
        ax.hist(neg, bins=bins, alpha=0.6, label=f"not_attributed (n={len(neg)})", density=True)
    if len(pos) > 0:
        ax.hist(pos, bins=bins, alpha=0.6, label=f"attributed (n={len(pos)})", density=True)
    ax.set_title(f"Predicted Probability Distribution (Best Model, {probability_variant})")
    fig.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    ax.set_xlabel("Predicted attributed probability")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    track_mode: str,
    probability_variant: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(prob_pred, prob_true, marker="o", label=f"Model (Brier={brier:.4f})")
    except Exception as exc:
        ax.text(0.5, 0.5, f"Calibration unavailable: {exc}", ha="center", va="center")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_title(f"Calibration Curve (Best Model, {probability_variant})")
    fig.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.legend(loc="best")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        pred = (y_prob >= float(thr)).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, pred)),
                "positives_predicted": int(np.sum(pred == 1)),
            }
        )
    return pd.DataFrame(rows)


def _compute_topk_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: Sequence[int] = (50, 100, 500, 1000),
) -> pd.DataFrame:
    n = int(len(y_true))
    if n == 0:
        return pd.DataFrame(columns=["k", "precision_at_k", "recall_at_k", "lift_at_k", "topk_capture"])
    prevalence = float(np.mean(y_true))
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    pos_total = int(np.sum(y_true == 1))
    rows: List[Dict[str, Any]] = []
    for k in k_values:
        k_eff = min(max(1, int(k)), n)
        top = y_sorted[:k_eff]
        hits = int(np.sum(top == 1))
        precision_k = float(hits / k_eff)
        recall_k = float(hits / pos_total) if pos_total > 0 else 0.0
        lift_k = float(precision_k / prevalence) if prevalence > 0 else float("nan")
        rows.append(
            {
                "k": int(k_eff),
                "hits": hits,
                "precision_at_k": precision_k,
                "recall_at_k": recall_k,
                "lift_at_k": lift_k,
                "topk_capture": recall_k,
            }
        )

    decile_k = max(1, int(round(0.1 * n)))
    top_decile_hits = int(np.sum(y_sorted[:decile_k] == 1))
    top_decile_capture = float(top_decile_hits / pos_total) if pos_total > 0 else 0.0
    out = pd.DataFrame(rows)
    out["top_decile_k"] = int(decile_k)
    out["top_decile_capture"] = float(top_decile_capture)
    out["prevalence"] = prevalence
    return out


def _plot_topk_curves(topk_df: pd.DataFrame, out_pr_path: Path, out_lift_path: Path, track_mode: str) -> None:
    if topk_df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(topk_df["k"], topk_df["precision_at_k"], marker="o", label="Precision@K")
    plt.plot(topk_df["k"], topk_df["recall_at_k"], marker="o", label="Recall@K")
    plt.title("Precision/Recall at K (is_attributed surrogate target)")
    plt.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    ensure_parent(out_pr_path)
    plt.savefig(out_pr_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(topk_df["k"], topk_df["lift_at_k"], marker="o", color="tab:green")
    plt.title("Lift@K (is_attributed surrogate target)")
    plt.suptitle(_track_claim_label(track_mode), y=0.98, fontsize=10)
    plt.xlabel("K")
    plt.ylabel("Lift")
    plt.tight_layout()
    ensure_parent(out_lift_path)
    plt.savefig(out_lift_path, dpi=150)
    plt.close()


def _fit_platt_scaler(y_val: np.ndarray, val_prob: np.ndarray, seed: int = 42) -> LogisticRegression:
    calibrator = LogisticRegression(max_iter=1000, random_state=seed)
    calibrator.fit(val_prob.reshape(-1, 1), y_val.astype(int))
    return calibrator


def _as_track_mode(value: str) -> TrackMode:
    if value not in {"production_safe", "benchmark_transductive"}:
        raise ValueError(f"Unsupported track mode: {value}")
    return value  # type: ignore[return-value]


def _to_selection_tuple(pr_auc: float, mcc: float, f1: float) -> Tuple[float, float, float]:
    def _clean(v: float) -> float:
        return float(v) if pd.notna(v) else -1.0

    return (_clean(pr_auc), _clean(mcc), _clean(f1))


def _prepare_lgbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            ts_sec = (series.astype("int64") // 10**9).astype("float64")
            ts_sec[series.isna()] = -1.0
            out[col] = ts_sec
        elif pd.api.types.is_bool_dtype(series):
            out[col] = series.astype("int8")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            out[col] = series.astype("category").cat.codes.astype("int32")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    return out


def train_lightgbm_bagging(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    config: WinnerStyleConfig,
    model_dir: Path,
) -> Dict[str, Any]:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:
        raise ImportError(
            "LightGBM is required for model_family=lightgbm_bagging/all. "
            "Install dependencies via: pip install -r requirements.txt"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
    combined_lgbm = _prepare_lgbm_frame(combined)
    n_train = len(X_train)
    n_val = len(X_val)
    X_train_lgbm = combined_lgbm.iloc[:n_train].reset_index(drop=True)
    X_val_lgbm = combined_lgbm.iloc[n_train : n_train + n_val].reset_index(drop=True)
    X_test_lgbm = combined_lgbm.iloc[n_train + n_val :].reset_index(drop=True)

    y_train_arr = y_train.astype(int).to_numpy()
    y_val_arr = y_val.astype(int).to_numpy()

    val_probs: List[np.ndarray] = []
    test_probs: List[np.ndarray] = []
    bag_paths: List[str] = []
    bag_rows: List[Dict[str, Any]] = []
    train_times: List[float] = []
    bag_neg_sets: List[set[int]] = []

    for bag_i in range(max(1, int(config.bag_count))):
        bag_seed = int(config.random_seed) + bag_i
        bag_idx = make_bag_subsample_indices(y_train_arr, neg_pos_ratio=int(config.neg_pos_ratio), seed=bag_seed)
        X_bag = X_train_lgbm.iloc[bag_idx]
        y_bag = y_train.iloc[bag_idx]
        bag_neg_sets.append(set(int(i) for i in bag_idx if int(y_train_arr[i]) == 0))

        class_weight = None
        if config.imbalance_strategy == "subsampling_plus_weighting":
            bag_pos = int(np.sum(y_train_arr[bag_idx] == 1))
            bag_neg = int(np.sum(y_train_arr[bag_idx] == 0))
            scale = max(1.0, float(bag_neg / max(1, bag_pos)))
            class_weight = {0: 1.0, 1: scale}

        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=float(config.lgbm_learning_rate),
            num_leaves=int(config.lgbm_num_leaves),
            n_estimators=int(config.lgbm_n_estimators),
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=bag_seed,
            n_jobs=4,
            class_weight=class_weight,
        )

        t0 = time.perf_counter()
        try:
            model.fit(
                X_bag,
                y_bag,
                eval_set=[(X_val_lgbm, y_val_arr)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception:
            model.fit(
                X_bag,
                y_bag,
                eval_set=[(X_val_lgbm, y_val_arr)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        train_times.append(time.perf_counter() - t0)

        val_prob = model.predict_proba(X_val_lgbm)[:, 1]
        test_prob = model.predict_proba(X_test_lgbm)[:, 1]
        val_probs.append(val_prob)
        test_probs.append(test_prob)
        try:
            bag_ap = float(average_precision_score(y_val_arr, val_prob))
        except Exception:
            bag_ap = float("nan")

        bag_path = model_dir / f"bag_{bag_i + 1}.txt"
        model.booster_.save_model(str(bag_path))
        bag_paths.append(str(bag_path))
        bag_rows.append(
            {
                "bag_index": bag_i + 1,
                "seed": bag_seed,
                "sample_size": int(len(bag_idx)),
                "positive_count": int(np.sum(y_train_arr[bag_idx] == 1)),
                "negative_count": int(np.sum(y_train_arr[bag_idx] == 0)),
                "val_pr_auc": bag_ap,
                "model_path": str(bag_path),
            }
        )

    ensemble_val_prob = np.mean(np.vstack(val_probs), axis=0)
    ensemble_test_prob = np.mean(np.vstack(test_probs), axis=0)
    bag_overlap_pairs: List[float] = []
    if len(bag_neg_sets) > 1:
        for i in range(len(bag_neg_sets)):
            for j in range(i + 1, len(bag_neg_sets)):
                a = bag_neg_sets[i]
                b = bag_neg_sets[j]
                denom = max(1, len(a.union(b)))
                bag_overlap_pairs.append(float(len(a.intersection(b)) / denom))
    manifest = {
        "track_mode": config.track_mode,
        "bag_count": int(config.bag_count),
        "neg_pos_ratio": int(config.neg_pos_ratio),
        "imbalance_strategy": str(config.imbalance_strategy),
        "learning_rate": float(config.lgbm_learning_rate),
        "num_leaves": int(config.lgbm_num_leaves),
        "n_estimators": int(config.lgbm_n_estimators),
        "bag_models": bag_rows,
        "bag_negative_jaccard_mean": float(np.mean(bag_overlap_pairs)) if bag_overlap_pairs else 0.0,
        "avg_train_time_sec": float(np.mean(train_times) if train_times else 0.0),
    }
    manifest_path = model_dir / "ensemble_manifest.json"
    save_json(manifest, manifest_path)

    return {
        "val_prob": ensemble_val_prob,
        "test_prob": ensemble_test_prob,
        "manifest_path": manifest_path,
        "bag_model_paths": bag_paths,
        "avg_train_time_sec": float(np.mean(train_times) if train_times else 0.0),
    }


def train_and_evaluate_models(
    curated_df: pd.DataFrame,
    artifacts_metrics_dir: Path,
    figures_dir: Path,
    model_dir: Path,
    seed: int = 42,
    target_col: str = "is_attributed",
    approach_track: TrackMode = "production_safe",
    model_family: Literal["baseline", "lightgbm_bagging", "all"] = "all",
    row_cap: int = 600_000,
    winner_style_config: Optional[WinnerStyleConfig] = None,
    lightgbm_model_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    artifacts_metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_family not in {"baseline", "lightgbm_bagging", "all"}:
        raise ValueError("model_family must be one of: baseline, lightgbm_bagging, all")

    track_mode = _as_track_mode(str(approach_track))
    ws_cfg = winner_style_config or WinnerStyleConfig(track_mode=track_mode, random_seed=seed)
    ws_cfg.track_mode = track_mode
    ws_cfg.active_target_col = target_col

    if ws_cfg.calibration_method != "platt":
        raise ValueError("Only platt calibration is supported in this hardening pass")
    if ws_cfg.imbalance_strategy not in {"subsampling_only", "subsampling_plus_weighting"}:
        raise ValueError("Unsupported imbalance_strategy")
    if track_mode == "production_safe" and ws_cfg.enable_next_click:
        raise ValueError("enable_next_click is only allowed for benchmark_transductive track")

    df = load_talkingdata_frame(curated_df, seed=seed)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    if df[target_col].nunique() < 2:
        raise ValueError("Target column has fewer than 2 classes; cannot train classifier")

    df, split_info, row_window_meta = select_row_window_for_support(
        df=df,
        track_mode=track_mode,
        target_col=target_col,
        row_cap=row_cap,
        row_cap_step=int(ws_cfg.row_cap_step),
        min_val_positives=int(ws_cfg.min_val_positives),
        min_test_positives=int(ws_cfg.min_test_positives),
    )
    idx_train = np.asarray(split_info["train_idx"], dtype=int)
    idx_val = np.asarray(split_info["val_idx"], dtype=int)
    idx_test = np.asarray(split_info["test_idx"], dtype=int)

    df = add_count_unique_cum_features(df, mode=track_mode)
    df = add_delta_time_features(df, mode=track_mode, enable_next_click=ws_cfg.enable_next_click)
    df, encoding_meta = build_rate_features_train_only(
        df,
        split_info=split_info,
        target_col=target_col,
        oof_folds=int(ws_cfg.oof_folds),
        smoothing_alpha=float(ws_cfg.oof_smoothing_alpha),
    )

    click_ns = df["click_time"].astype("int64", copy=False)
    df["click_unix"] = np.where(df["click_time"].notna(), click_ns // 10**9, 0).astype("int64")

    forbidden_exact = {
        target_col,
        "is_fraud",
        "is_attributed",
        "attributed_time",
        "attributed_time_ts",
        "record_hash",
        "batch_hash",
        "audit_pass",
        "click_time",
    }
    forbidden_tokens = ("attributed_time",)

    def _is_forbidden_feature(col_name: str) -> bool:
        if col_name in forbidden_exact:
            return True
        lc = col_name.lower()
        if lc == target_col.lower():
            return True
        if any(tok in lc for tok in forbidden_tokens):
            return True
        return False

    feature_cols = [c for c in df.columns if not _is_forbidden_feature(c)]
    excluded_cols = [c for c in df.columns if _is_forbidden_feature(c)]
    if not feature_cols:
        raise ValueError("No features available after leakage-safe column exclusions")

    X = df[feature_cols]
    y = df[target_col].astype(int)
    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    X_test = X.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_val = y.iloc[idx_val]
    y_test = y.iloc[idx_test]
    y_val_arr = y_val.to_numpy(dtype=int)
    y_test_arr = y_test.to_numpy(dtype=int)

    split_overlap = validate_split_integrity(split_info)
    temporal_order_valid = bool(
        (df.iloc[idx_train]["click_time"].max() <= df.iloc[idx_val]["click_time"].min())
        and (df.iloc[idx_val]["click_time"].max() <= df.iloc[idx_test]["click_time"].min())
    )
    support_counts = split_support_counts(df, split_info, target_col=target_col)
    prevalence = {
        "overall": float(y.mean()),
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    }

    baseline_pred = np.zeros_like(y_test_arr)
    baseline_accuracy = float(accuracy_score(y_test_arr, baseline_pred))

    rows: List[Dict[str, Any]] = []
    metrics_uncal_rows: List[Dict[str, Any]] = []
    metrics_cal_rows: List[Dict[str, Any]] = []
    model_prob_test: Dict[str, Dict[str, np.ndarray]] = {}
    model_prob_val: Dict[str, Dict[str, np.ndarray]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    model_paths: Dict[str, Path] = {}
    selection_rows: List[Dict[str, Any]] = []
    calibration_summary: Dict[str, Any] = {
        "calibration_method": ws_cfg.calibration_method,
        "calibration_split": "temporal_validation",
        "models": {},
    }
    xgb_available = False

    def _record_metrics(
        model_name: str,
        model_family_name: str,
        variant: str,
        val_prob: np.ndarray,
        test_prob: np.ndarray,
        train_time_sec: float,
        inference_time_val_sec: float,
    ) -> Dict[str, Any]:
        val_pred = (val_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_arr, test_pred, labels=[0, 1]).ravel()
        val_metrics = _compute_model_metrics(y_val_arr, val_pred, val_prob)
        test_metrics = _compute_model_metrics(y_test_arr, test_pred, test_prob)
        row = {
            "track_mode": track_mode,
            "target_col": target_col,
            "target_semantics": "is_attributed_surrogate",
            "probability_variant": variant,
            "model_family": model_family_name,
            "model": model_name,
            **test_metrics,
            "val_pr_auc": float(val_metrics.get("pr_auc", float("nan"))),
            "val_mcc": float(val_metrics.get("mcc", float("nan"))),
            "val_f1": float(val_metrics.get("f1", float("nan"))),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "test_pos_support": int(np.sum(y_test_arr == 1)),
            "test_neg_support": int(np.sum(y_test_arr == 0)),
            "train_time_sec": float(train_time_sec),
            "inference_time_val_sec": float(inference_time_val_sec),
        }
        rows.append(row)
        if variant == "uncalibrated":
            metrics_uncal_rows.append(row)
        elif variant == "calibrated":
            metrics_cal_rows.append(row)
        return row

    if model_family in {"baseline", "all"}:
        preprocess, _, _ = _build_preprocessor(X_train)
        models: Dict[str, Any] = {
            "LogisticRegression": LogisticRegression(
                max_iter=700,
                class_weight="balanced",
                random_state=seed,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=260,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        }

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = max(1.0, float(neg / max(1, pos)))
        try:
            from xgboost import XGBClassifier  # type: ignore

            models["XGBoost"] = XGBClassifier(
                n_estimators=260,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=4,
                scale_pos_weight=scale_pos_weight,
            )
            xgb_available = True
        except Exception:
            xgb_available = False

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocess),
                    ("model", model),
                ]
            )
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            infer_time_val = time.perf_counter() - t1
            test_prob = pipeline.predict_proba(X_test)[:, 1]

            row = _record_metrics(
                model_name=model_name,
                model_family_name="baseline",
                variant="uncalibrated",
                val_prob=val_prob,
                test_prob=test_prob,
                train_time_sec=train_time,
                inference_time_val_sec=infer_time_val,
            )
            model_prob_val.setdefault(model_name, {})["uncalibrated"] = val_prob
            model_prob_test.setdefault(model_name, {})["uncalibrated"] = test_prob
            fitted_pipelines[model_name] = pipeline
            selection_rows.append(row)

    if model_family in {"lightgbm_bagging", "all"}:
        lgbm_dir = lightgbm_model_dir or (model_dir / "lightgbm_bagging" / track_mode)
        lgbm_result = train_lightgbm_bagging(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            config=ws_cfg,
            model_dir=lgbm_dir,
        )
        model_name = "LightGBMBagging"
        val_prob_uncal = np.asarray(lgbm_result["val_prob"], dtype=float)
        test_prob_uncal = np.asarray(lgbm_result["test_prob"], dtype=float)
        row_uncal = _record_metrics(
            model_name=model_name,
            model_family_name="lightgbm_bagging",
            variant="uncalibrated",
            val_prob=val_prob_uncal,
            test_prob=test_prob_uncal,
            train_time_sec=float(lgbm_result.get("avg_train_time_sec", 0.0)),
            inference_time_val_sec=0.0,
        )
        model_prob_val.setdefault(model_name, {})["uncalibrated"] = val_prob_uncal
        model_prob_test.setdefault(model_name, {})["uncalibrated"] = test_prob_uncal
        model_paths[model_name] = Path(lgbm_result["manifest_path"])

        calibrator = _fit_platt_scaler(y_val_arr, val_prob_uncal, seed=seed)
        val_prob_cal = calibrator.predict_proba(val_prob_uncal.reshape(-1, 1))[:, 1]
        test_prob_cal = calibrator.predict_proba(test_prob_uncal.reshape(-1, 1))[:, 1]
        row_cal = _record_metrics(
            model_name=model_name,
            model_family_name="lightgbm_bagging",
            variant="calibrated",
            val_prob=val_prob_cal,
            test_prob=test_prob_cal,
            train_time_sec=float(lgbm_result.get("avg_train_time_sec", 0.0)),
            inference_time_val_sec=0.0,
        )
        model_prob_val.setdefault(model_name, {})["calibrated"] = val_prob_cal
        model_prob_test.setdefault(model_name, {})["calibrated"] = test_prob_cal
        selection_rows.append(row_cal)
        calibration_summary["models"][model_name] = {
            "fit_size": {
                "rows": int(len(y_val_arr)),
                "positives": int(np.sum(y_val_arr == 1)),
                "negatives": int(np.sum(y_val_arr == 0)),
            },
            "intercept": float(calibrator.intercept_[0]),
            "coef": float(calibrator.coef_[0][0]),
            "uncalibrated_val_pr_auc": float(row_uncal.get("val_pr_auc", float("nan"))),
            "calibrated_val_pr_auc": float(row_cal.get("val_pr_auc", float("nan"))),
        }

    if not rows:
        raise ValueError("No models were trained. Check model_family configuration.")

    metrics_df = pd.DataFrame(rows).sort_values(["val_pr_auc", "val_mcc", "val_f1"], ascending=False).reset_index(drop=True)
    metrics_uncal_df = pd.DataFrame(metrics_uncal_rows).sort_values(["val_pr_auc", "val_mcc", "val_f1"], ascending=False).reset_index(drop=True) if metrics_uncal_rows else pd.DataFrame()
    metrics_cal_df = pd.DataFrame(metrics_cal_rows).sort_values(["val_pr_auc", "val_mcc", "val_f1"], ascending=False).reset_index(drop=True) if metrics_cal_rows else pd.DataFrame()

    metrics_path = artifacts_metrics_dir / "model_metrics.csv"
    metrics_uncal_path = artifacts_metrics_dir / "model_metrics_uncalibrated.csv"
    metrics_cal_path = artifacts_metrics_dir / "model_metrics_calibrated.csv"
    metrics_df.to_csv(metrics_path, index=False)
    metrics_uncal_df.to_csv(metrics_uncal_path, index=False)
    if not metrics_cal_df.empty:
        metrics_cal_df.to_csv(metrics_cal_path, index=False)
    else:
        metrics_cal_path.write_text("", encoding="utf-8")
    save_json({"rows": metrics_uncal_df.to_dict(orient="records") if not metrics_uncal_df.empty else []}, artifacts_metrics_dir / "model_metrics_uncalibrated.json")
    save_json({"rows": metrics_cal_df.to_dict(orient="records") if not metrics_cal_df.empty else []}, artifacts_metrics_dir / "model_metrics_calibrated.json")

    best_row = max(
        selection_rows,
        key=lambda r: _to_selection_tuple(float(r.get("val_pr_auc", float("nan"))), float(r.get("val_mcc", float("nan"))), float(r.get("val_f1", float("nan")))),
    )
    best_model_name = str(best_row["model"])
    best_model_family = str(best_row["model_family"])
    best_variant = str(best_row["probability_variant"])
    best_val_prob = model_prob_val[best_model_name][best_variant]
    best_test_prob = model_prob_test[best_model_name][best_variant]

    threshold_val_df = _compute_threshold_sweep(y_val_arr, best_val_prob)
    threshold_val_df = threshold_val_df.sort_values(["f1", "mcc", "recall"], ascending=False).reset_index(drop=True)
    selected_threshold = float(threshold_val_df.iloc[0]["threshold"]) if not threshold_val_df.empty else 0.5
    threshold_test_df = _compute_threshold_sweep(y_test_arr, best_test_prob)
    threshold_test_df["track_mode"] = track_mode
    threshold_test_df["model"] = best_model_name
    threshold_test_df["probability_variant"] = best_variant
    threshold_test_df["selected_threshold"] = selected_threshold
    threshold_test_df.to_csv(artifacts_metrics_dir / "threshold_analysis_best_model.csv", index=False)
    threshold_test_df.to_csv(artifacts_metrics_dir / f"threshold_analysis_best_model_{best_variant}.csv", index=False)

    best_pred = (best_test_prob >= selected_threshold).astype(int)
    display_probs: Dict[str, np.ndarray] = {}
    for model_name, variants in model_prob_test.items():
        if "calibrated" in variants:
            display_probs[model_name] = variants["calibrated"]
        else:
            display_probs[model_name] = variants["uncalibrated"]

    def _slug(value: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")

    metrics_plot_path = figures_dir / f"model_metrics_comparison_{track_mode}.png"
    roc_plot_path = figures_dir / f"roc_curves_{track_mode}.png"
    pr_plot_path = figures_dir / f"pr_curves_{track_mode}.png"
    conf_plot_path = figures_dir / f"best_model_confusion_matrix_{track_mode}.png"
    prob_plot_path = figures_dir / f"class_probability_distribution_{track_mode}_{_slug(best_model_name)}.png"
    cal_plot_path = figures_dir / f"calibration_curve_{track_mode}_{_slug(best_model_name)}.png"
    topk_csv_path = artifacts_metrics_dir / f"topk_metrics_{track_mode}_{_slug(best_model_name)}.csv"
    topk_pr_plot = figures_dir / f"precision_recall_at_k_{track_mode}_{_slug(best_model_name)}.png"
    topk_lift_plot = figures_dir / f"lift_at_k_{track_mode}_{_slug(best_model_name)}.png"

    _plot_metric_bars(metrics_df, metrics_plot_path, track_mode=track_mode)
    _plot_roc_curves(y_test_arr, display_probs, roc_plot_path, track_mode=track_mode, target_label="is_attributed surrogate target")
    _plot_pr_curves(y_test_arr, display_probs, pr_plot_path, track_mode=track_mode, target_label="is_attributed surrogate target")
    _plot_confusion_matrix(y_test_arr, best_pred, best_model_name, conf_plot_path, track_mode=track_mode)
    _plot_probability_distribution(y_test_arr, best_test_prob, prob_plot_path, track_mode=track_mode, probability_variant=best_variant)
    _plot_calibration_curve(y_test_arr, best_test_prob, cal_plot_path, track_mode=track_mode, probability_variant=best_variant)

    topk_df = _compute_topk_metrics(y_test_arr, best_test_prob, k_values=(50, 100, 500, 1000))
    topk_df["track_mode"] = track_mode
    topk_df["model"] = best_model_name
    topk_df["probability_variant"] = best_variant
    topk_df.to_csv(topk_csv_path, index=False)
    _plot_topk_curves(topk_df, topk_pr_plot, topk_lift_plot, track_mode=track_mode)

    compat_map = {
        figures_dir / "model_metrics_comparison.png": metrics_plot_path,
        figures_dir / "roc_curves.png": roc_plot_path,
        figures_dir / "pr_curves.png": pr_plot_path,
        figures_dir / "best_model_confusion_matrix.png": conf_plot_path,
        figures_dir / "class_probability_distribution_best_model.png": prob_plot_path,
        figures_dir / "calibration_curve_best_model.png": cal_plot_path,
    }
    for legacy, modern in compat_map.items():
        if modern.exists():
            shutil.copy2(modern, legacy)

    if best_model_name in fitted_pipelines:
        best_pipeline = fitted_pipelines[best_model_name]
        importance_df = _extract_feature_importance(best_pipeline)
        if not importance_df.empty:
            importance_df.to_csv(artifacts_metrics_dir / "best_model_feature_importance.csv", index=False)
            _plot_feature_importance(
                importance_df,
                best_model_name,
                figures_dir / f"best_model_feature_importance_{track_mode}.png",
                track_mode=track_mode,
            )
        model_path = model_dir / "best_model_pipeline.joblib"
        joblib.dump(best_pipeline, model_path)
    else:
        model_path = model_paths.get(best_model_name, model_dir / "best_model_reference.txt")
        if best_model_name in model_paths:
            (model_dir / "best_model_reference.txt").write_text(str(model_path), encoding="utf-8")

    pred_dump: Dict[str, Any] = {"y_test": y_test_arr.astype(int)}
    for model_name, variants in model_prob_test.items():
        for variant_name, probs in variants.items():
            pred_dump[f"{model_name}_{variant_name}_proba"] = probs
    np.savez_compressed(artifacts_metrics_dir / "model_eval_predictions.npz", **pred_dump)

    def _summary_path(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(Path.cwd().resolve()))
        except Exception:
            return str(p)

    feature_count_by_type = {
        "count": int(len([c for c in feature_cols if c.endswith("_count") or c.startswith("freq_")])),
        "unique": int(len([c for c in feature_cols if "unique" in c])),
        "cum": int(len([c for c in feature_cols if c.startswith("cumcount_")])),
        "delta": int(len([c for c in feature_cols if "delta_sec" in c])),
        "rate": int(len([c for c in feature_cols if c.startswith("te_")])),
    }
    forbidden_absent = bool(all(c not in feature_cols for c in forbidden_exact))

    summary = {
        "metrics_path": _summary_path(metrics_path),
        "metrics_uncalibrated_path": _summary_path(metrics_uncal_path),
        "metrics_calibrated_path": _summary_path(metrics_cal_path),
        "best_model_name": best_model_name,
        "best_model_family": best_model_family,
        "best_probability_variant": best_variant,
        "best_model_path": _summary_path(model_path),
        "xgboost_available": xgb_available,
        "target_col": target_col,
        "target_semantics": "is_attributed_surrogate",
        "track_mode": track_mode,
        "split_strategy": split_info.get("strategy"),
        "feature_columns_used": feature_cols,
        "excluded_leakage_columns": sorted(excluded_cols),
        "split_overlap_counts": split_overlap,
        "temporal_order_valid": temporal_order_valid,
        "support_counts": support_counts,
        "class_prevalence": prevalence,
        "all_negative_baseline_accuracy": baseline_accuracy,
        "selected_threshold": float(selected_threshold),
        "row_window_policy": row_window_meta,
        "encoding_metadata": encoding_meta,
        "feature_diagnostics": {
            "chronological_pass_used": bool(track_mode == "production_safe"),
            "history_carryover_enabled": bool(ws_cfg.history_carryover_enabled),
            "forbidden_feature_names_absent": forbidden_absent,
            "feature_count_by_type": feature_count_by_type,
        },
        "calibration_summary": calibration_summary,
        "ranking_outputs": {
            "topk_metrics_path": _summary_path(topk_csv_path),
            "lift_plot_path": _summary_path(topk_lift_plot),
            "precision_recall_plot_path": _summary_path(topk_pr_plot),
        },
        "plot_paths": {
            "model_metrics": _summary_path(metrics_plot_path),
            "roc": _summary_path(roc_plot_path),
            "pr": _summary_path(pr_plot_path),
            "confusion": _summary_path(conf_plot_path),
            "probability_distribution": _summary_path(prob_plot_path),
            "calibration_curve": _summary_path(cal_plot_path),
        },
        "compatibility_copies": {str(k): _summary_path(v) for k, v in compat_map.items()},
        "winner_style_config": {
            "active_target_col": ws_cfg.active_target_col,
            "bag_count": int(ws_cfg.bag_count),
            "neg_pos_ratio": int(ws_cfg.neg_pos_ratio),
            "imbalance_strategy": str(ws_cfg.imbalance_strategy),
            "lgbm_learning_rate": float(ws_cfg.lgbm_learning_rate),
            "lgbm_num_leaves": int(ws_cfg.lgbm_num_leaves),
            "lgbm_n_estimators": int(ws_cfg.lgbm_n_estimators),
            "calibration_method": str(ws_cfg.calibration_method),
            "oof_folds": int(ws_cfg.oof_folds),
            "oof_smoothing_alpha": float(ws_cfg.oof_smoothing_alpha),
            "enable_next_click": bool(ws_cfg.enable_next_click),
            "min_val_positives": int(ws_cfg.min_val_positives),
            "min_test_positives": int(ws_cfg.min_test_positives),
            "row_cap_step": int(ws_cfg.row_cap_step),
            "history_carryover_enabled": bool(ws_cfg.history_carryover_enabled),
        },
    }
    save_json(summary, artifacts_metrics_dir / "model_run_summary.json")

    return summary


def plot_data_quality_summary(
    validation_summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if validation_summary_df.empty:
        return
    plot_df = validation_summary_df.copy()
    plot_df = plot_df.sort_values("invalid_count", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=plot_df, x="rule", y="invalid_count", hue="rule", palette="viridis", legend=False)
    for idx, row in plot_df.iterrows():
        pct = float(row.get("invalid_percent", 0.0))
        ax.text(
            idx,
            float(row["invalid_count"]) + max(1.0, 0.01 * float(plot_df["invalid_count"].max())),
            f"{int(row['invalid_count'])}\\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Data Quality Violations by Rule (Count + % of records)")
    plt.xlabel("Validation Rule")
    plt.ylabel("Invalid Record Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dq_rule_percentage(
    validation_summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if validation_summary_df.empty:
        return
    plot_df = validation_summary_df.copy().sort_values("invalid_percent", ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="rule", y="invalid_percent", hue="rule", palette="crest", legend=False)
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, float(row["invalid_percent"]) + 0.05, f"{float(row['invalid_percent']):.2f}%", ha="center", fontsize=9)
    plt.title("Top Data Quality Rule Failures by Percentage")
    plt.xlabel("Validation Rule")
    plt.ylabel("Invalid Percentage (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fraud_prevalence(
    df: pd.DataFrame,
    out_path: Path,
    target_col: str = "is_attributed",
) -> None:
    if target_col not in df.columns:
        return
    counts = df[target_col].value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "label": ["not_attributed" if i == 0 else "attributed" for i in counts.index],
            "count": counts.values,
        }
    )
    if set(plot_df["label"]) != {"not_attributed", "attributed"}:
        for lbl in ["not_attributed", "attributed"]:
            if lbl not in set(plot_df["label"]):
                plot_df = pd.concat([plot_df, pd.DataFrame([{"label": lbl, "count": 0}])], ignore_index=True)
        plot_df = plot_df.sort_values("label").reset_index(drop=True)
    total = float(plot_df["count"].sum())
    plot_df["percent"] = plot_df["count"] / total * 100.0 if total else 0.0
    plot_df["plot_count"] = plot_df["count"].clip(lower=0.8)
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=plot_df, x="label", y="plot_count", hue="label", palette="Set2", legend=False)
    ax.set_yscale("log")
    ymax = float(plot_df["count"].max()) if not plot_df.empty else 1.0
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(
            idx,
            float(max(row["plot_count"], 0.8)) * 1.05,
            f"{int(row['count'])}\\n({float(row['percent']):.3f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Class Prevalence: is_attributed surrogate target (log scale)")
    plt.xlabel("Class")
    plt.ylabel("Count (log scale)")
    plt.ylim(bottom=0.8, top=max(2.0, ymax * 1.4))
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


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
