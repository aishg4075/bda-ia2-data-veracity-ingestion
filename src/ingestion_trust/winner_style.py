from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


TrackMode = Literal["production_safe", "benchmark_transductive"]


def load_talkingdata_frame(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    del seed
    out = df.copy()
    if "click_time" not in out.columns:
        raise ValueError("Expected click_time column for winner-style preparation")

    out["click_time"] = pd.to_datetime(out["click_time"], errors="coerce")
    out = out.dropna(subset=["click_time"]).sort_values("click_time").reset_index(drop=True)

    out["day"] = out["click_time"].dt.day.astype("int16")
    out["hour"] = out["click_time"].dt.hour.astype("int8")
    out["minute"] = out["click_time"].dt.minute.astype("int8")
    out["second"] = out["click_time"].dt.second.astype("int8")
    return out


def build_temporal_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Dict[str, Any]:
    if len(df) < 30:
        raise ValueError("Need at least 30 rows to create stable temporal splits")

    idx = np.arange(len(df), dtype=int)
    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * (train_frac + val_frac))
    train_end = max(1, min(train_end, len(df) - 2))
    val_end = max(train_end + 1, min(val_end, len(df) - 1))

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "strategy": "strict_temporal",
    }


def build_competition_like_splits(df: pd.DataFrame) -> Dict[str, Any]:
    if "day" not in df.columns:
        raise ValueError("Expected day column for competition-like split")

    unique_days = sorted(pd.Series(df["day"]).dropna().astype(int).unique().tolist())
    if len(unique_days) < 3:
        out = build_temporal_splits(df)
        out["strategy"] = "competition_like_fallback_temporal"
        return out

    train_days = unique_days[:-2]
    val_day = unique_days[-2]
    test_day = unique_days[-1]

    train_idx = df.index[df["day"].isin(train_days)].to_numpy(dtype=int)
    val_idx = df.index[df["day"] == val_day].to_numpy(dtype=int)
    test_idx = df.index[df["day"] == test_day].to_numpy(dtype=int)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        out = build_temporal_splits(df)
        out["strategy"] = "competition_like_fallback_temporal"
        return out

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "strategy": "competition_day_hour_style",
        "train_days": train_days,
        "val_day": int(val_day),
        "test_day": int(test_day),
    }


def split_support_counts(
    df: pd.DataFrame,
    split_info: Mapping[str, Any],
    target_col: str,
) -> Dict[str, int]:
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    train_idx = np.asarray(split_info.get("train_idx", []), dtype=int)
    val_idx = np.asarray(split_info.get("val_idx", []), dtype=int)
    test_idx = np.asarray(split_info.get("test_idx", []), dtype=int)
    return {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_pos": int(y.iloc[train_idx].sum()) if len(train_idx) else 0,
        "val_pos": int(y.iloc[val_idx].sum()) if len(val_idx) else 0,
        "test_pos": int(y.iloc[test_idx].sum()) if len(test_idx) else 0,
    }


def select_row_window_for_support(
    df: pd.DataFrame,
    track_mode: TrackMode,
    target_col: str,
    row_cap: int,
    row_cap_step: int,
    min_val_positives: int,
    min_test_positives: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    if row_cap <= 0:
        raise ValueError("row_cap must be > 0")
    if row_cap_step <= 0:
        raise ValueError("row_cap_step must be > 0")

    n = len(df)
    if n < 30:
        raise ValueError("Need at least 30 rows for row window selection")

    split_builder = build_temporal_splits if track_mode == "production_safe" else build_competition_like_splits
    start_cap = min(int(row_cap), n)

    attempted: List[Dict[str, Any]] = []
    cap = start_cap
    selected_df: pd.DataFrame | None = None
    selected_split: Dict[str, Any] | None = None
    selected_support: Dict[str, int] | None = None

    while True:
        head_df = df.iloc[:cap].reset_index(drop=True)
        split_info = split_builder(head_df)
        support = split_support_counts(head_df, split_info, target_col=target_col)
        attempted.append({"window": "head", "cap": int(cap), **support})
        if support["val_pos"] >= min_val_positives and support["test_pos"] >= min_test_positives:
            selected_df = head_df
            selected_split = split_info
            selected_support = support
            window_mode = "chronological_head"
            break

        if cap >= n:
            break
        cap = min(n, cap + int(row_cap_step))

    if selected_df is None:
        trailing_cap = min(cap, n)
        trailing_df = df.iloc[n - trailing_cap :].reset_index(drop=True)
        trailing_split = split_builder(trailing_df)
        trailing_support = split_support_counts(trailing_df, trailing_split, target_col=target_col)
        attempted.append({"window": "trailing", "cap": int(trailing_cap), **trailing_support})
        if trailing_support["val_pos"] >= min_val_positives and trailing_support["test_pos"] >= min_test_positives:
            selected_df = trailing_df
            selected_split = trailing_split
            selected_support = trailing_support
            window_mode = "trailing_retry"

    if selected_df is None or selected_split is None or selected_support is None:
        attempted_msg = "; ".join(
            [f"{a['window']} cap={a['cap']} val_pos={a['val_pos']} test_pos={a['test_pos']}" for a in attempted]
        )
        raise ValueError(
            "Unable to satisfy minimum positive support after deterministic window policy. "
            f"Needed val>={min_val_positives}, test>={min_test_positives}. Attempts: {attempted_msg}"
        )

    window_meta = {
        "row_cap_strategy": window_mode,
        "initial_row_cap": int(row_cap),
        "row_cap_step": int(row_cap_step),
        "selected_rows": int(len(selected_df)),
        "start_timestamp": selected_df["click_time"].min().isoformat() if len(selected_df) else None,
        "end_timestamp": selected_df["click_time"].max().isoformat() if len(selected_df) else None,
        "support_counts": selected_support,
        "attempts": attempted,
        "fallback_behavior": "trailing_retry_once",
    }
    return selected_df, selected_split, window_meta


def _prior_unique_count_single(key_vals: Sequence[Any], value_vals: Sequence[Any]) -> np.ndarray:
    state: Dict[Any, set] = {}
    out = np.zeros(len(key_vals), dtype=np.int32)
    for i, (k, v) in enumerate(zip(key_vals, value_vals)):
        s = state.get(k)
        if s is None:
            s = set()
            state[k] = s
        out[i] = len(s)
        s.add(v)
    return out


def _prior_unique_count_pair(key_a: Sequence[Any], key_b: Sequence[Any], value_vals: Sequence[Any]) -> np.ndarray:
    state: Dict[Tuple[Any, Any], set] = {}
    out = np.zeros(len(key_a), dtype=np.int32)
    for i, (a, b, v) in enumerate(zip(key_a, key_b, value_vals)):
        k = (a, b)
        s = state.get(k)
        if s is None:
            s = set()
            state[k] = s
        out[i] = len(s)
        s.add(v)
    return out


def add_count_unique_cum_features(
    df: pd.DataFrame,
    mode: TrackMode,
) -> pd.DataFrame:
    out = df.copy()
    if mode == "production_safe":
        out["ip_click_count"] = out.groupby("ip").cumcount().astype("int32")
        out["app_click_count"] = out.groupby("app").cumcount().astype("int32")
        out["channel_click_count"] = out.groupby("channel").cumcount().astype("int32")
        out["ip_app_count"] = out.groupby(["ip", "app"]).cumcount().astype("int32")
        out["ip_device_os_count"] = out.groupby(["ip", "device", "os"]).cumcount().astype("int32")
        out["ip_day_hour_count"] = out.groupby(["ip", "day", "hour"]).cumcount().astype("int32")
        out["ip_unique_app_count"] = _prior_unique_count_single(out["ip"].tolist(), out["app"].tolist())
        out["ip_app_unique_channel_count"] = _prior_unique_count_pair(
            out["ip"].tolist(), out["app"].tolist(), out["channel"].tolist()
        )
        out["cumcount_ip"] = out["ip_click_count"]
        out["cumcount_ip_app"] = out["ip_app_count"]
        out["cumcount_ip_device_os"] = out["ip_device_os_count"]
        return out

    out["ip_click_count"] = out.groupby("ip")["ip"].transform("size")
    out["app_click_count"] = out.groupby("app")["app"].transform("size")
    out["channel_click_count"] = out.groupby("channel")["channel"].transform("size")
    out["ip_app_count"] = out.groupby(["ip", "app"])["app"].transform("size")
    out["ip_device_os_count"] = out.groupby(["ip", "device", "os"])["os"].transform("size")
    out["ip_day_hour_count"] = out.groupby(["ip", "day", "hour"])["hour"].transform("size")
    out["ip_unique_app_count"] = out.groupby("ip")["app"].transform("nunique")
    out["ip_app_unique_channel_count"] = out.groupby(["ip", "app"])["channel"].transform("nunique")
    out["cumcount_ip"] = out.groupby("ip").cumcount()
    out["cumcount_ip_app"] = out.groupby(["ip", "app"]).cumcount()
    out["cumcount_ip_device_os"] = out.groupby(["ip", "device", "os"]).cumcount()
    return out


def add_delta_time_features(
    df: pd.DataFrame,
    mode: TrackMode,
    enable_next_click: bool = False,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    out["prev_click_delta_sec_ip"] = (
        out.groupby("ip")["click_time"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    )
    out["prev_click_delta_sec_ip_app"] = (
        out.groupby(["ip", "app"])["click_time"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    )

    if mode == "benchmark_transductive" and enable_next_click:
        out["next_click_delta_sec_ip"] = (
            (out.groupby("ip")["click_time"].shift(-1) - out["click_time"])
            .dt.total_seconds()
            .fillna(0.0)
            .clip(lower=0.0)
        )
    return out


def _build_smoothed_stats(
    hist_df: pd.DataFrame,
    cols: Sequence[str],
    target_col: str,
    global_rate: float,
    smoothing_alpha: float,
) -> pd.DataFrame:
    if hist_df.empty:
        return pd.DataFrame(columns=[*cols, "enc_val", "freq_val"])
    grouped = hist_df.groupby(list(cols), as_index=False)[target_col].agg(["sum", "count"]).reset_index()
    grouped["enc_val"] = (grouped["sum"] + float(smoothing_alpha) * float(global_rate)) / (
        grouped["count"] + float(smoothing_alpha)
    )
    grouped = grouped.rename(columns={"count": "freq_val"})
    return grouped[[*cols, "enc_val", "freq_val"]]


def _apply_oof_for_single(
    out: pd.DataFrame,
    train_idx: np.ndarray,
    col: str,
    target_col: str,
    global_rate: float,
    oof_folds: int,
    smoothing_alpha: float,
) -> Tuple[pd.Series, pd.Series]:
    te = pd.Series(np.full(len(out), global_rate, dtype=float), index=out.index)
    freq = pd.Series(np.zeros(len(out), dtype=float), index=out.index)

    fold_edges = np.linspace(0, len(train_idx), num=max(2, int(oof_folds) + 1), dtype=int)
    for fold_i in range(len(fold_edges) - 1):
        start = int(fold_edges[fold_i])
        end = int(fold_edges[fold_i + 1])
        fold_idx = train_idx[start:end]
        if len(fold_idx) == 0:
            continue
        hist_idx = train_idx[:start]
        hist_df = out.iloc[hist_idx] if len(hist_idx) else out.iloc[[]]
        stats = _build_smoothed_stats(hist_df, [col], target_col, global_rate, smoothing_alpha)
        if stats.empty:
            te.iloc[fold_idx] = global_rate
            freq.iloc[fold_idx] = 0.0
            continue
        enc_map = stats.set_index(col)["enc_val"]
        freq_map = stats.set_index(col)["freq_val"]
        vals = out.iloc[fold_idx][col]
        te.iloc[fold_idx] = vals.map(enc_map).fillna(global_rate).astype(float)
        freq.iloc[fold_idx] = vals.map(freq_map).fillna(0.0).astype(float)

    return te, freq


def _apply_oof_for_pair(
    out: pd.DataFrame,
    train_idx: np.ndarray,
    cols: Sequence[str],
    target_col: str,
    global_rate: float,
    oof_folds: int,
    smoothing_alpha: float,
) -> Tuple[pd.Series, pd.Series]:
    te = pd.Series(np.full(len(out), global_rate, dtype=float), index=out.index)
    freq = pd.Series(np.zeros(len(out), dtype=float), index=out.index)

    fold_edges = np.linspace(0, len(train_idx), num=max(2, int(oof_folds) + 1), dtype=int)
    for fold_i in range(len(fold_edges) - 1):
        start = int(fold_edges[fold_i])
        end = int(fold_edges[fold_i + 1])
        fold_idx = train_idx[start:end]
        if len(fold_idx) == 0:
            continue
        hist_idx = train_idx[:start]
        hist_df = out.iloc[hist_idx] if len(hist_idx) else out.iloc[[]]
        stats = _build_smoothed_stats(hist_df, cols, target_col, global_rate, smoothing_alpha)
        fold_rows = out.iloc[fold_idx][list(cols)].copy()
        fold_rows["__row_idx"] = fold_idx
        if not stats.empty:
            merged = fold_rows.merge(stats, on=list(cols), how="left", sort=False)
            merged = merged.sort_values("__row_idx")
            te.iloc[fold_idx] = merged["enc_val"].fillna(global_rate).to_numpy()
            freq.iloc[fold_idx] = merged["freq_val"].fillna(0.0).to_numpy()
        else:
            te.iloc[fold_idx] = global_rate
            freq.iloc[fold_idx] = 0.0

    return te, freq


def build_rate_features_train_only(
    df: pd.DataFrame,
    split_info: Mapping[str, Any],
    target_col: str = "is_attributed",
    oof_folds: int = 5,
    smoothing_alpha: float = 20.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    if target_col not in out.columns:
        raise ValueError(f"Missing target column '{target_col}' for train-only rate features")

    train_idx = np.asarray(split_info.get("train_idx", []), dtype=int)
    val_idx = np.asarray(split_info.get("val_idx", []), dtype=int)
    test_idx = np.asarray(split_info.get("test_idx", []), dtype=int)
    if len(train_idx) == 0:
        raise ValueError("train_idx is empty for leakage-safe encoding")

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(0.0)
    train_df = out.iloc[train_idx]
    global_rate = float(train_df[target_col].mean()) if len(train_df) else 0.0

    single_cols = [c for c in ["ip", "app", "device", "os", "channel", "hour", "day"] if c in out.columns]
    pair_defs: List[Tuple[List[str], str]] = [
        (["ip", "app"], "ip_app"),
        (["app", "channel"], "app_channel"),
        (["ip", "hour"], "ip_hour"),
    ]

    for col in single_cols:
        te_ser, freq_ser = _apply_oof_for_single(
            out=out,
            train_idx=train_idx,
            col=col,
            target_col=target_col,
            global_rate=global_rate,
            oof_folds=oof_folds,
            smoothing_alpha=smoothing_alpha,
        )
        out[f"te_{col}"] = te_ser.astype("float64")
        out[f"freq_{col}_train"] = freq_ser.astype("float64")

        full_stats = _build_smoothed_stats(train_df, [col], target_col, global_rate, smoothing_alpha)
        if not full_stats.empty:
            enc_map = full_stats.set_index(col)["enc_val"]
            freq_map = full_stats.set_index(col)["freq_val"]
            holdout_idx = np.concatenate([val_idx, test_idx]) if len(val_idx) + len(test_idx) > 0 else np.asarray([], dtype=int)
            if len(holdout_idx):
                vals = out.iloc[holdout_idx][col]
                out.iloc[holdout_idx, out.columns.get_loc(f"te_{col}")] = vals.map(enc_map).fillna(global_rate).astype(float)
                out.iloc[holdout_idx, out.columns.get_loc(f"freq_{col}_train")] = vals.map(freq_map).fillna(0.0).astype(float)

    for cols, suffix in pair_defs:
        if any(c not in out.columns for c in cols):
            continue
        te_ser, freq_ser = _apply_oof_for_pair(
            out=out,
            train_idx=train_idx,
            cols=cols,
            target_col=target_col,
            global_rate=global_rate,
            oof_folds=oof_folds,
            smoothing_alpha=smoothing_alpha,
        )
        out[f"te_{suffix}"] = te_ser.astype("float64")
        out[f"freq_{suffix}_train"] = freq_ser.astype("float64")

        full_stats = _build_smoothed_stats(train_df, cols, target_col, global_rate, smoothing_alpha)
        holdout_idx = np.concatenate([val_idx, test_idx]) if len(val_idx) + len(test_idx) > 0 else np.asarray([], dtype=int)
        if len(holdout_idx):
            holdout = out.iloc[holdout_idx][list(cols)].copy()
            holdout["__row_idx"] = holdout_idx
            if not full_stats.empty:
                merged = holdout.merge(full_stats, on=list(cols), how="left", sort=False)
                merged = merged.sort_values("__row_idx")
                out.iloc[holdout_idx, out.columns.get_loc(f"te_{suffix}")] = merged["enc_val"].fillna(global_rate).to_numpy()
                out.iloc[holdout_idx, out.columns.get_loc(f"freq_{suffix}_train")] = merged["freq_val"].fillna(0.0).to_numpy()
            else:
                out.iloc[holdout_idx, out.columns.get_loc(f"te_{suffix}")] = global_rate
                out.iloc[holdout_idx, out.columns.get_loc(f"freq_{suffix}_train")] = 0.0

    meta = {
        "encoding_type": "target_mean_oof_temporal",
        "fold_strategy": "temporal_sequential_folds",
        "oof_folds": int(oof_folds),
        "smoothing_alpha": float(smoothing_alpha),
        "global_rate": float(global_rate),
        "temporal_mode": True,
        "self_target_contribution": "blocked_by_design",
        "single_cols": single_cols,
        "pair_cols": [suffix for cols, suffix in pair_defs if all(c in out.columns for c in cols)],
    }
    return out, meta


def make_bag_subsample_indices(
    y: Iterable[int],
    neg_pos_ratio: int = 200,
    seed: int = 42,
) -> np.ndarray:
    y_arr = np.asarray(list(y), dtype=int)
    if len(y_arr) == 0:
        return np.asarray([], dtype=int)

    pos_idx = np.flatnonzero(y_arr == 1)
    neg_idx = np.flatnonzero(y_arr == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.arange(len(y_arr), dtype=int)

    target_neg = min(len(neg_idx), max(len(pos_idx), int(len(pos_idx) * max(1, neg_pos_ratio))))
    rng = np.random.default_rng(seed)
    sampled_neg = rng.choice(neg_idx, size=target_neg, replace=False)
    merged = np.concatenate([pos_idx, sampled_neg]).astype(int)
    rng.shuffle(merged)
    return merged


def validate_split_integrity(split_info: Mapping[str, Any]) -> Dict[str, int]:
    train_set = set(int(i) for i in split_info.get("train_idx", []))
    val_set = set(int(i) for i in split_info.get("val_idx", []))
    test_set = set(int(i) for i in split_info.get("test_idx", []))
    return {
        "train_val": len(train_set.intersection(val_set)),
        "train_test": len(train_set.intersection(test_set)),
        "val_test": len(val_set.intersection(test_set)),
    }
