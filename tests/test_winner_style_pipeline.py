from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ingestion_trust.core import WinnerStyleConfig, train_and_evaluate_models
from ingestion_trust.winner_style import (
    add_delta_time_features,
    build_rate_features_train_only,
    build_temporal_splits,
    load_talkingdata_frame,
    make_bag_subsample_indices,
    select_row_window_for_support,
    validate_split_integrity,
)


def _make_sample_df(n_rows: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2017-11-06 00:00:00")
    click_time = [base + pd.Timedelta(seconds=int(i * 13)) for i in range(n_rows)]
    ip = rng.integers(10_000, 10_800, size=n_rows)
    app = rng.integers(1, 25, size=n_rows)
    device = rng.integers(1, 6, size=n_rows)
    os_code = rng.integers(9, 27, size=n_rows)
    channel = rng.integers(100, 400, size=n_rows)

    attr_signal = (
        (ip % 17 == 0).astype(float) * 0.18
        + (channel % 13 == 0).astype(float) * 0.12
        + (app % 7 == 0).astype(float) * 0.07
    )
    noise = rng.normal(0.02, 0.03, size=n_rows)
    attr_prob = np.clip(0.03 + attr_signal + noise, 0.001, 0.95)
    is_attributed = (rng.random(n_rows) < attr_prob).astype(int)
    is_fraud_legacy = (1 - is_attributed).astype(int)

    attributed_time = []
    for t, y in zip(click_time, is_attributed):
        if int(y) == 1:
            attributed_time.append((t + pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"))
        else:
            attributed_time.append("")

    return pd.DataFrame(
        {
            "click_time": click_time,
            "ip": ip,
            "app": app,
            "device": device,
            "os": os_code,
            "channel": channel,
            "is_fraud": is_fraud_legacy,
            "is_attributed": is_attributed,
            "attributed_time": attributed_time,
        }
    )


def _resolve_output_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def test_temporal_split_integrity() -> None:
    df = load_talkingdata_frame(_make_sample_df(500))
    split_info = build_temporal_splits(df)
    overlap = validate_split_integrity(split_info)
    assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}

    train_idx = split_info["train_idx"]
    val_idx = split_info["val_idx"]
    test_idx = split_info["test_idx"]
    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()
    assert df.iloc[train_idx]["click_time"].max() <= df.iloc[val_idx]["click_time"].min()
    assert df.iloc[val_idx]["click_time"].max() <= df.iloc[test_idx]["click_time"].min()


def test_row_cap_policy_meets_positive_support() -> None:
    prepared = load_talkingdata_frame(_make_sample_df(1600, seed=13))
    selected_df, split_info, meta = select_row_window_for_support(
        df=prepared,
        track_mode="production_safe",
        target_col="is_attributed",
        row_cap=400,
        row_cap_step=200,
        min_val_positives=20,
        min_test_positives=20,
    )
    assert len(selected_df) >= 400
    assert meta["support_counts"]["val_pos"] >= 20
    assert meta["support_counts"]["test_pos"] >= 20
    assert meta["row_cap_strategy"] in {"chronological_head", "trailing_retry"}
    assert meta["start_timestamp"] is not None
    assert meta["end_timestamp"] is not None
    assert validate_split_integrity(split_info) == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_bag_sampling_keeps_all_positives_and_respects_ratio() -> None:
    y = np.array([1] * 40 + [0] * 4000, dtype=int)
    idx_a = make_bag_subsample_indices(y, neg_pos_ratio=20, seed=42)
    idx_b = make_bag_subsample_indices(y, neg_pos_ratio=20, seed=43)

    pos_idx = set(np.flatnonzero(y == 1).tolist())
    sampled_pos = set(i for i in idx_a.tolist() if y[i] == 1)
    sampled_neg_count = int(np.sum(y[idx_a] == 0))

    assert sampled_pos == pos_idx
    assert sampled_neg_count <= 20 * len(pos_idx)
    assert not np.array_equal(np.sort(idx_a), np.sort(idx_b))


def test_oof_target_encoding_blocks_self_contribution() -> None:
    df = _make_sample_df(600, seed=31)
    df["ip"] = np.arange(100000, 100000 + len(df), dtype=int)
    prepared = load_talkingdata_frame(df)
    split_info = build_temporal_splits(prepared)
    encoded_df, meta = build_rate_features_train_only(
        prepared,
        split_info=split_info,
        target_col="is_attributed",
        oof_folds=5,
        smoothing_alpha=20.0,
    )

    train_idx = np.asarray(split_info["train_idx"], dtype=int)
    global_rate = float(meta["global_rate"])
    te_ip_train = encoded_df.iloc[train_idx]["te_ip"].to_numpy(dtype=float)
    assert np.allclose(te_ip_train, global_rate)
    assert bool(meta["temporal_mode"]) is True
    assert meta["self_target_contribution"] == "blocked_by_design"


def test_next_click_guard_and_track_specific_outputs(tmp_path: Path) -> None:
    df = _make_sample_df(1800, seed=19)
    prepared = load_talkingdata_frame(df)
    prod_df = add_delta_time_features(prepared, mode="production_safe", enable_next_click=True)
    bench_df = add_delta_time_features(prepared, mode="benchmark_transductive", enable_next_click=True)
    assert "next_click_delta_sec_ip" not in prod_df.columns
    assert "next_click_delta_sec_ip" in bench_df.columns

    with pytest.raises(ValueError):
        train_and_evaluate_models(
            curated_df=prepared,
            artifacts_metrics_dir=tmp_path / "metrics_guard",
            figures_dir=tmp_path / "figures_guard",
            model_dir=tmp_path / "models_guard",
            approach_track="production_safe",
            model_family="baseline",
            target_col="is_attributed",
            row_cap=1200,
            winner_style_config=WinnerStyleConfig(
                track_mode="production_safe",
                enable_next_click=True,
                min_val_positives=20,
                min_test_positives=20,
                row_cap_step=100,
                random_seed=11,
            ),
        )

    for track in ["production_safe", "benchmark_transductive"]:
        summary = train_and_evaluate_models(
            curated_df=prepared,
            artifacts_metrics_dir=tmp_path / "metrics" / track,
            figures_dir=tmp_path / "figures" / track,
            model_dir=tmp_path / "models" / track,
            approach_track=track,
            model_family="baseline",
            target_col="is_attributed",
            row_cap=1200,
            winner_style_config=WinnerStyleConfig(
                track_mode=track,
                bag_count=2,
                neg_pos_ratio=40,
                min_val_positives=20,
                min_test_positives=20,
                row_cap_step=100,
                random_seed=11,
            ),
        )
        assert summary["track_mode"] == track
        metrics_path = tmp_path / "metrics" / track / "model_metrics.csv"
        assert metrics_path.exists()
        df_metrics = pd.read_csv(metrics_path)
        assert "track_mode" in df_metrics.columns
        assert set(df_metrics["track_mode"].astype(str).unique().tolist()) == {track}

        assert (tmp_path / "metrics" / track / "model_metrics_uncalibrated.csv").exists()
        assert (tmp_path / "metrics" / track / "model_metrics_calibrated.csv").exists()
        assert (tmp_path / "metrics" / track / "threshold_analysis_best_model.csv").exists()

        topk_path = _resolve_output_path(str(summary["ranking_outputs"]["topk_metrics_path"]))
        pr_k_path = _resolve_output_path(str(summary["ranking_outputs"]["precision_recall_plot_path"]))
        lift_path = _resolve_output_path(str(summary["ranking_outputs"]["lift_plot_path"]))
        assert topk_path.exists()
        assert pr_k_path.exists()
        assert lift_path.exists()

        for _, raw_path in summary.get("plot_paths", {}).items():
            p = _resolve_output_path(str(raw_path))
            assert p.exists()

        assert bool(summary.get("feature_diagnostics", {}).get("forbidden_feature_names_absent", False)) is True


def test_lightgbm_calibration_outputs_when_available(tmp_path: Path) -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed in current environment")

    prepared = load_talkingdata_frame(_make_sample_df(2000, seed=29))
    summary = train_and_evaluate_models(
        curated_df=prepared,
        artifacts_metrics_dir=tmp_path / "metrics_lgbm" / "production_safe",
        figures_dir=tmp_path / "figures_lgbm" / "production_safe",
        model_dir=tmp_path / "models_lgbm" / "production_safe",
        approach_track="production_safe",
        model_family="lightgbm_bagging",
        target_col="is_attributed",
        row_cap=1400,
        winner_style_config=WinnerStyleConfig(
            track_mode="production_safe",
            bag_count=2,
            neg_pos_ratio=40,
            min_val_positives=20,
            min_test_positives=20,
            row_cap_step=100,
            calibration_method="platt",
            imbalance_strategy="subsampling_only",
            random_seed=17,
        ),
        lightgbm_model_dir=tmp_path / "models_lgbm" / "lightgbm_bagging" / "production_safe",
    )

    metrics_dir = tmp_path / "metrics_lgbm" / "production_safe"
    uncal = metrics_dir / "model_metrics_uncalibrated.csv"
    cal = metrics_dir / "model_metrics_calibrated.csv"
    assert uncal.exists()
    assert cal.exists()
    assert cal.stat().st_size > 0
    assert (metrics_dir / "model_metrics_uncalibrated.json").exists()
    assert (metrics_dir / "model_metrics_calibrated.json").exists()
    assert (metrics_dir / "threshold_analysis_best_model.csv").exists()
    assert (metrics_dir / "model_eval_predictions.npz").exists()

    cal_summary = summary.get("calibration_summary", {})
    assert cal_summary.get("calibration_method") == "platt"
    assert "LightGBMBagging" in cal_summary.get("models", {})

    cal_curve = _resolve_output_path(str(summary["plot_paths"]["calibration_curve"]))
    topk_csv = _resolve_output_path(str(summary["ranking_outputs"]["topk_metrics_path"]))
    assert cal_curve.exists()
    assert topk_csv.exists()
