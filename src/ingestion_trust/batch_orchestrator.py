from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score

from .core import (
    WinnerStyleConfig,
    append_audit_batch,
    build_run_metadata,
    detect_dataset_mode,
    measure_integrity_overhead,
    plot_data_quality_summary,
    plot_dq_rule_percentage,
    plot_fraud_prevalence,
    save_json,
    sha256_file,
    tamper_log_copy,
    train_and_evaluate_models,
    verify_audit_log,
)
from .pipeline_contracts import (
    BatchPipelineResult,
    ModelingStageResult,
    PipelinePaths,
    TrackExecutionSpec,
    TrustStageResult,
)
from .spark_utils import (
    apply_schema_casting,
    build_baseline_stats,
    compute_veracity_score,
    create_spark_session,
    feature_engineering,
    run_veracity_checks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _save_validation_reports(summary: dict, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, metrics_dir / "validation_report.json")

    rows = []
    invalid_counts = summary.get("invalid_counts_by_rule", {})
    invalid_percent = summary.get("invalid_percent_by_rule", {})
    for rule, count in invalid_counts.items():
        rows.append(
            {
                "rule": rule,
                "invalid_count": int(count),
                "invalid_percent": float(invalid_percent.get(rule, 0.0)),
            }
        )
    pd.DataFrame(rows).to_csv(metrics_dir / "validation_report.csv", index=False)


def _plot_split_prevalence(prevalence: dict, out_path: Path) -> None:
    rows = [
        {"split": split, "prevalence_pct": float(val) * 100.0}
        for split, val in prevalence.items()
    ]
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=df, x="split", y="prevalence_pct", hue="split", palette="crest", legend=False)
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(i, float(row["prevalence_pct"]) + 0.001, f"{row['prevalence_pct']:.4f}%", ha="center", fontsize=9)
    ymax = max(0.01, float(df["prevalence_pct"].max()) * 1.25 if not df.empty else 0.01)
    plt.ylim(0, ymax)
    plt.title("Attributed Class Prevalence by Split (%)")
    plt.xlabel("Split")
    plt.ylabel("Attributed Rate (%)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _resolve_summary_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _parse_stability_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        try:
            seeds.append(int(t))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value in --stability-seeds: '{t}'") from exc
    if not seeds:
        raise ValueError("--stability-seeds must provide at least one integer")
    return sorted(set(seeds))


def _read_best_metrics_row(track_metrics_path: Path, best_model_name: str, best_variant: str) -> dict:
    if not track_metrics_path.exists():
        return {}
    try:
        df = pd.read_csv(track_metrics_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    for col in ["val_pr_auc", "val_mcc", "val_f1"]:
        if col not in df.columns:
            df[col] = np.nan
    mask = (df.get("model", pd.Series(dtype=str)) == best_model_name) & (
        df.get("probability_variant", pd.Series(dtype=str)) == best_variant
    )
    pick = df.loc[mask]
    if pick.empty:
        pick = df.sort_values(["val_pr_auc", "val_mcc", "val_f1"], ascending=False).head(1)
    return pick.iloc[0].to_dict()


def _recompute_metrics_parity(track_metrics_dir: Path, best_model_name: str, best_variant: str) -> dict:
    pred_path = track_metrics_dir / "model_eval_predictions.npz"
    metrics_path = track_metrics_dir / "model_metrics.csv"
    key = f"{best_model_name}_{best_variant}_proba"
    if not pred_path.exists():
        return {"pass": False, "reason": f"missing predictions file: {pred_path}"}
    if not metrics_path.exists():
        return {"pass": False, "reason": f"missing metrics file: {metrics_path}"}

    with np.load(pred_path) as npz:
        if "y_test" not in npz.files:
            return {"pass": False, "reason": "missing y_test in model_eval_predictions.npz"}
        if key not in npz.files:
            return {"pass": False, "reason": f"missing probability key in predictions: {key}"}
        y_true = np.asarray(npz["y_test"], dtype=int)
        y_prob = np.asarray(npz[key], dtype=float)

    y_pred = (y_prob >= 0.5).astype(int)
    recomputed = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    row = _read_best_metrics_row(metrics_path, best_model_name, best_variant)
    if not row:
        return {"pass": False, "reason": "unable to find matching metrics row for parity check"}

    deltas = {}
    for metric in ["pr_auc", "f1", "mcc", "precision", "recall"]:
        logged = float(row.get(metric, np.nan))
        if np.isnan(logged):
            deltas[metric] = None
        else:
            deltas[metric] = float(abs(logged - recomputed[metric]))
    max_delta = max([d for d in deltas.values() if d is not None], default=0.0)
    return {
        "pass": bool(max_delta <= 1e-9),
        "max_abs_delta": float(max_delta),
        "deltas": deltas,
        "recomputed_metrics": recomputed,
    }


def _ensure_legacy_compat_copy(src: Path, dst: Path, mapping: list[dict]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mapping.append(
            {
                "legacy_path": _to_project_relative(dst),
                "canonical_path": _to_project_relative(src),
            }
        )


def _track_best_row(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    for col in ["val_pr_auc", "val_mcc", "val_f1"]:
        if col not in df.columns:
            df[col] = np.nan
    best = df.sort_values(["val_pr_auc", "val_mcc", "val_f1"], ascending=False).iloc[0]
    return best.to_dict()


def _format_best_row_md(track_mode: str, row: dict) -> str:
    if not row:
        return f"- `{track_mode}`: no metrics row available."
    model = str(row.get("model", "n/a"))
    variant = str(row.get("probability_variant", "n/a"))
    pr_auc = row.get("pr_auc", np.nan)
    mcc = row.get("mcc", np.nan)
    f1 = row.get("f1", np.nan)
    return (
        f"- `{track_mode}`: best `{model}` ({variant}), "
        f"PR-AUC={float(pr_auc):.6f}, MCC={float(mcc):.6f}, F1={float(f1):.6f}"
    )


def _build_debug_outputs(
    *,
    artifacts_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    track_summaries: dict[str, dict],
    tracks: list[str],
    args: argparse.Namespace,
    legacy_semantics_warning: str | None,
    stability_summary: dict,
    compat_copy_map: list[dict],
) -> dict:
    debug_dir = artifacts_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    leakage_track_checks: dict[str, dict] = {}
    artifact_trace_rows: list[dict] = []
    correctness_pass = True
    calibrated_path_valid = True
    track_mode_column_valid = True
    no_split_overlap = True
    temporal_order_ok = True
    forbidden_features_ok = True
    causal_mode_ok = True
    topk_outputs_ok = True
    no_pr_auc_regression = True
    threshold_non_flat = True
    pr_auc_regression_details: dict[str, dict] = {}

    for track in tracks:
        summary = track_summaries.get(track, {})
        split_overlap = summary.get("split_overlap_counts", {})
        no_overlap = all(int(v) == 0 for v in split_overlap.values())
        temporal_ok = bool(summary.get("temporal_order_valid", False))
        feature_diag = summary.get("feature_diagnostics", {})
        forbidden_ok = bool(feature_diag.get("forbidden_feature_names_absent", False))
        causal_ok = (
            bool(feature_diag.get("chronological_pass_used", False)) if track == "production_safe" else True
        )
        history_carryover = bool(feature_diag.get("history_carryover_enabled", False))

        track_metrics_path = metrics_dir / track / "model_metrics.csv"
        metrics_track_mode_ok = False
        if track_metrics_path.exists():
            try:
                track_df = pd.read_csv(track_metrics_path)
                metrics_track_mode_ok = (
                    ("track_mode" in track_df.columns)
                    and (not track_df.empty)
                    and set(track_df["track_mode"].astype(str).unique().tolist()) == {track}
                )
            except Exception:
                metrics_track_mode_ok = False

        metrics_uncal_path = metrics_dir / track / "model_metrics_uncalibrated.csv"
        metrics_cal_path = metrics_dir / track / "model_metrics_calibrated.csv"
        uncal_exists = metrics_uncal_path.exists()
        cal_exists = metrics_cal_path.exists()
        cal_nonempty = cal_exists and metrics_cal_path.stat().st_size > 0
        if args.model_family in {"lightgbm_bagging", "all"}:
            calibrated_path_valid = calibrated_path_valid and cal_nonempty

        topk_path_raw = str(summary.get("ranking_outputs", {}).get("topk_metrics_path", ""))
        topk_path = _resolve_summary_path(topk_path_raw) if topk_path_raw else Path("")
        topk_exists = bool(topk_path_raw) and topk_path.exists()

        threshold_path = metrics_dir / track / "threshold_analysis_best_model.csv"
        threshold_has_signal = False
        if threshold_path.exists():
            try:
                thr_df = pd.read_csv(threshold_path)
                if not thr_df.empty and "f1" in thr_df.columns and "mcc" in thr_df.columns:
                    threshold_has_signal = bool(
                        float(pd.to_numeric(thr_df["f1"], errors="coerce").std(ddof=0) or 0.0) > 0.0
                        or float(pd.to_numeric(thr_df["mcc"], errors="coerce").std(ddof=0) or 0.0) > 0.0
                    )
            except Exception:
                threshold_has_signal = False

        pr_auc_ok = True
        best_pr_auc = None
        rf_pr_auc = None
        if track_metrics_path.exists():
            try:
                track_df = pd.read_csv(track_metrics_path)
                if not track_df.empty and "pr_auc" in track_df.columns:
                    sorted_df = track_df.sort_values(["val_pr_auc", "val_mcc", "val_f1"], ascending=False)
                    best_pr_auc = float(sorted_df.iloc[0]["pr_auc"])
                    rf_df = track_df[(track_df["model"] == "RandomForest") & (track_df["probability_variant"] == "uncalibrated")]
                    if not rf_df.empty:
                        rf_pr_auc = float(rf_df.iloc[0]["pr_auc"])
                        pr_auc_ok = bool(best_pr_auc >= rf_pr_auc)
            except Exception:
                pr_auc_ok = False

        parity = _recompute_metrics_parity(
            metrics_dir / track,
            best_model_name=str(summary.get("best_model_name", "")),
            best_variant=str(summary.get("best_probability_variant", "uncalibrated")),
        )

        leakage_track_checks[track] = {
            "target_col": summary.get("target_col"),
            "target_semantics": summary.get("target_semantics"),
            "split_overlap_counts": split_overlap,
            "no_split_overlap": no_overlap,
            "temporal_order_valid": temporal_ok,
            "forbidden_feature_names_absent": forbidden_ok,
            "chronological_pass_used": bool(feature_diag.get("chronological_pass_used", False)),
            "history_carryover_enabled": history_carryover,
            "next_click_allowed": bool(track == "benchmark_transductive"),
            "metrics_track_mode_column_valid": metrics_track_mode_ok,
            "metrics_uncalibrated_exists": uncal_exists,
            "metrics_calibrated_exists": cal_exists,
            "metrics_calibrated_nonempty": cal_nonempty,
            "topk_metrics_exists": topk_exists,
            "threshold_curve_non_flat": threshold_has_signal,
            "no_pr_auc_regression_vs_randomforest": pr_auc_ok,
            "pr_auc_comparison": {"best_pr_auc": best_pr_auc, "randomforest_pr_auc": rf_pr_auc},
            "metrics_recompute_parity": parity,
            "support_counts": summary.get("support_counts", {}),
            "row_window_policy": summary.get("row_window_policy", {}),
        }

        artifact_trace_rows.append(
            {
                "track_mode": track,
                "metrics_path": _to_project_relative(track_metrics_path),
                "metrics_uncalibrated_path": _to_project_relative(metrics_uncal_path),
                "metrics_calibrated_path": _to_project_relative(metrics_cal_path),
                "topk_metrics_path": _to_project_relative(topk_path) if topk_path_raw else None,
                "summary_path": _to_project_relative(metrics_dir / track / "model_run_summary.json"),
            }
        )

        no_split_overlap = no_split_overlap and no_overlap
        temporal_order_ok = temporal_order_ok and temporal_ok
        forbidden_features_ok = forbidden_features_ok and forbidden_ok
        causal_mode_ok = causal_mode_ok and causal_ok
        topk_outputs_ok = topk_outputs_ok and topk_exists
        threshold_non_flat = threshold_non_flat and threshold_has_signal
        no_pr_auc_regression = no_pr_auc_regression and pr_auc_ok
        track_mode_column_valid = track_mode_column_valid and metrics_track_mode_ok
        correctness_pass = correctness_pass and bool(parity.get("pass", False))
        pr_auc_regression_details[track] = {
            "best_pr_auc": best_pr_auc,
            "randomforest_pr_auc": rf_pr_auc,
            "pass": pr_auc_ok,
        }

    semantics_ok = bool(args.target_col == "is_attributed" or args.allow_legacy_fraud_target)
    legacy_active = bool(args.target_col == "is_fraud")
    next_click_guard_ok = not (args.enable_next_click and ("production_safe" in tracks))

    leakage_payload = {
        "target_semantics": {
            "active_target_col": args.target_col,
            "active_target_semantics": "is_attributed_surrogate" if args.target_col == "is_attributed" else "legacy_is_fraud_deprecated",
            "legacy_semantics_opt_in_used": legacy_active,
            "legacy_semantics_warning": legacy_semantics_warning,
        },
        "semantics_checks": {
            "active_supervised_target_is_is_attributed": bool(args.target_col == "is_attributed"),
            "no_default_is_fraud_training_path": bool(args.target_col != "is_fraud" or args.allow_legacy_fraud_target),
            "next_click_guard_valid": next_click_guard_ok,
        },
        "track_checks": leakage_track_checks,
        "acceptance_framework": {
            "correctness": {
                "semantics_ok": semantics_ok,
                "no_split_overlap": no_split_overlap,
                "temporal_order_valid": temporal_order_ok,
                "forbidden_features_absent": forbidden_features_ok,
                "track_mode_column_valid": track_mode_column_valid,
                "calibrated_probability_path_valid": calibrated_path_valid if args.model_family in {"lightgbm_bagging", "all"} else True,
                "metrics_recompute_parity": correctness_pass,
            },
            "performance_targets": {
                "no_pr_auc_regression_vs_randomforest": no_pr_auc_regression,
                "pr_auc_regression_details": pr_auc_regression_details,
                "topk_outputs_non_degenerate": topk_outputs_ok,
                "threshold_curve_non_flat": threshold_non_flat,
                "threshold_sweep_paths_present": all((metrics_dir / track / "threshold_analysis_best_model.csv").exists() for track in tracks),
            },
            "stability_targets": stability_summary,
        },
        "artifact_trace_map": artifact_trace_rows,
        "legacy_compatibility_copies": compat_copy_map,
    }
    save_json(leakage_payload, debug_dir / "leakage_and_causality_checks.json")

    calibration_payload = {
        "calibration_method": args.calibration_method,
        "calibration_split": "temporal_validation",
        "tracks": {},
    }
    for track in tracks:
        summary = track_summaries.get(track, {})
        cal_summary = summary.get("calibration_summary", {})
        cal_plot = summary.get("plot_paths", {}).get("calibration_curve")
        calibration_payload["tracks"][track] = {
            "best_model": summary.get("best_model_name"),
            "best_probability_variant": summary.get("best_probability_variant"),
            "calibration_summary": cal_summary,
            "calibration_curve_path": cal_plot,
            "metrics_uncalibrated_path": summary.get("metrics_uncalibrated_path"),
            "metrics_calibrated_path": summary.get("metrics_calibrated_path"),
        }
    save_json(calibration_payload, debug_dir / "calibration_summary.json")

    track_best = {track: _track_best_row(metrics_dir / track / "model_metrics.csv") for track in tracks}
    track_lines = [
        "# Track Comparison Summary",
        "",
        "Primary claims must use `production_safe` only.",
        "`benchmark_transductive` is a Kaggle-style benchmark and is not valid for deployment-causal claims.",
        "",
        "## Best Model by Track",
        _format_best_row_md("production_safe", track_best.get("production_safe", {})),
        _format_best_row_md("benchmark_transductive", track_best.get("benchmark_transductive", {})),
        "",
        "## Shared Trust Controls (Track-Independent)",
        "- Data quality validation, integrity chain verification, audit overhead, and streaming correctness are shared pipeline controls.",
        "- Modeling-track performance numbers must not be used to claim stronger veracity/integrity control behavior.",
        "",
        "## Artifact Paths",
    ]
    for row in artifact_trace_rows:
        track_lines.append(f"- `{row['track_mode']}` metrics: `{row['metrics_path']}`")
        track_lines.append(f"- `{row['track_mode']}` uncalibrated: `{row['metrics_uncalibrated_path']}`")
        track_lines.append(f"- `{row['track_mode']}` calibrated: `{row['metrics_calibrated_path']}`")
        if row.get("topk_metrics_path"):
            track_lines.append(f"- `{row['track_mode']}` top-K: `{row['topk_metrics_path']}`")
    (debug_dir / "track_comparison_summary.md").write_text("\n".join(track_lines) + "\n", encoding="utf-8")

    hardening_lines = [
        "# Winner-Style Hardening Report",
        "",
        "## Target Semantics",
        f"- Active supervised target: `{args.target_col}`.",
        "- Active semantics label: `is_attributed` surrogate attribution/conversion target.",
        f"- Legacy `is_fraud` opt-in used: `{legacy_active}`.",
    ]
    if legacy_semantics_warning:
        hardening_lines.append(f"- Legacy warning: {legacy_semantics_warning}")
    hardening_lines.extend(
        [
            "",
            "## Leakage + Causality",
            f"- No split overlap: `{no_split_overlap}`.",
            f"- Temporal order valid: `{temporal_order_ok}`.",
            f"- Forbidden features absent: `{forbidden_features_ok}`.",
            f"- Production causal pass enforced: `{causal_mode_ok}`.",
            "",
            "## Calibration",
            f"- Calibration method: `{args.calibration_method}` on temporal validation split.",
            f"- Calibrated path valid: `{calibrated_path_valid if args.model_family in {'lightgbm_bagging', 'all'} else True}`.",
            "",
            "## Row-Cap + Support Policy",
            "- Deterministic chronological window with stepwise expansion and one trailing retry is enabled.",
            f"- Minimum support thresholds: val>={int(args.min_val_positives)}, test>={int(args.min_test_positives)}.",
            "",
            "## Imbalance Strategy",
            f"- Active strategy: `{args.imbalance_strategy}`.",
            "",
            "## Ranking Metrics",
            f"- Top-K outputs generated for tracks: `{topk_outputs_ok}` (K in {{50,100,500,1000}}).",
            "",
            "## Stability",
            f"- Stability seeds: `{args.stability_seeds}`.",
            f"- Stability summary: `{json.dumps(stability_summary, sort_keys=True)}`.",
            "",
            "## Report-Safe Claim Separation",
            "- Shared trust controls (DQ/integrity/audit/stream) are reported independently of modeling-track experiments.",
            "",
            "## Debug Artifact Map",
            f"- `{_to_project_relative(debug_dir / 'leakage_and_causality_checks.json')}`",
            f"- `{_to_project_relative(debug_dir / 'calibration_summary.json')}`",
            f"- `{_to_project_relative(debug_dir / 'track_comparison_summary.md')}`",
            f"- `{_to_project_relative(debug_dir / 'winner_style_hardening_report.md')}`",
        ]
    )
    (debug_dir / "winner_style_hardening_report.md").write_text("\n".join(hardening_lines) + "\n", encoding="utf-8")

    return {
        "debug_dir": debug_dir,
        "leakage_and_causality_checks": debug_dir / "leakage_and_causality_checks.json",
        "calibration_summary": debug_dir / "calibration_summary.json",
        "track_comparison_summary": debug_dir / "track_comparison_summary.md",
        "winner_style_hardening_report": debug_dir / "winner_style_hardening_report.md",
    }




def prepare_pipeline_paths(args: argparse.Namespace) -> PipelinePaths:
    return PipelinePaths(
        project_root=PROJECT_ROOT,
        artifacts_dir=args.artifacts_dir,
        metrics_dir=args.artifacts_dir / "metrics",
        audit_dir=args.artifacts_dir / "audit",
        provenance_dir=args.artifacts_dir / "provenance",
        curated_dir=args.artifacts_dir / "curated",
        debug_dir=args.artifacts_dir / "debug",
        figures_dir=args.figures_dir,
        model_dir=args.model_dir,
    )


def _legacy_warning_and_tracks(args: argparse.Namespace) -> tuple[str | None, list[str]]:
    legacy_semantics_warning = None
    if args.target_col == "is_fraud":
        if not args.allow_legacy_fraud_target:
            raise ValueError(
                "Target semantics hardening requires --target-col is_attributed. "
                "Legacy target is_fraud is disabled by default. "
                "If you must run legacy mode, pass --allow-legacy-fraud-target explicitly."
            )
        legacy_semantics_warning = (
            "Legacy `is_fraud` target semantics were explicitly enabled for compatibility. "
            "This mode is deprecated and must not be used for presentation-safe primary claims."
        )
    tracks = (
        ["production_safe", "benchmark_transductive"]
        if args.approach_track == "both"
        else [args.approach_track]
    )
    return legacy_semantics_warning, tracks


def run_trust_ingestion_stage(args: argparse.Namespace, paths: PipelinePaths) -> TrustStageResult:
    spark = create_spark_session("topic17-batch-pipeline")

    t_etl_start = time.perf_counter()
    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .csv(str(args.input_csv))
    )
    if args.max_rows and args.max_rows > 0:
        raw_df = raw_df.limit(args.max_rows)

    cast_df = apply_schema_casting(raw_df)
    feat_df = feature_engineering(cast_df)

    baseline_meta_path = args.baseline_json.with_suffix(args.baseline_json.suffix + ".meta.json")
    input_fingerprint = sha256_file(args.input_csv)
    baseline_meta = {}
    if baseline_meta_path.exists():
        try:
            baseline_meta = json.loads(baseline_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline_meta = {}

    baseline_matches = (
        args.baseline_json.exists()
        and baseline_meta.get("input_csv") == str(args.input_csv)
        and baseline_meta.get("input_sha256") == input_fingerprint
        and int(baseline_meta.get("max_rows", 0)) == int(args.max_rows)
    )

    if baseline_matches:
        baseline_stats = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    else:
        baseline_stats = build_baseline_stats(feat_df)
        save_json(baseline_stats, args.baseline_json)
        save_json(
            {
                "input_csv": str(args.input_csv),
                "input_sha256": input_fingerprint,
                "max_rows": int(args.max_rows),
                "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            },
            baseline_meta_path,
        )

    validated_df, validation_summary = run_veracity_checks(feat_df, baseline_stats=baseline_stats)
    scored_df = compute_veracity_score(validated_df)
    curated_df = scored_df.filter("is_valid_record = true")

    curated_path = paths.curated_dir / "batch_curated.parquet"
    curated_df.write.mode("overwrite").parquet(str(curated_path))

    etl_time_sec = time.perf_counter() - t_etl_start
    _save_validation_reports(validation_summary, paths.metrics_dir)

    validation_df = pd.read_csv(paths.metrics_dir / "validation_report.csv")
    plot_data_quality_summary(validation_df, paths.figures_dir / "data_quality_violations.png")
    plot_dq_rule_percentage(validation_df, paths.figures_dir / "top_dq_rules_percentage.png")

    curated_pd = curated_df.toPandas()
    compatibility_copy_map: list[dict] = []
    prevalence_plot_path = paths.figures_dir / "attribution_prevalence.png"
    plot_fraud_prevalence(curated_pd, prevalence_plot_path, target_col=args.target_col)
    _ensure_legacy_compat_copy(prevalence_plot_path, paths.figures_dir / "fraud_prevalence.png", compatibility_copy_map)

    return TrustStageResult(
        spark=spark,
        raw_df=raw_df,
        cast_df=cast_df,
        feat_df=feat_df,
        validated_df=validated_df,
        scored_df=scored_df,
        curated_df=curated_df,
        curated_pd=curated_pd,
        curated_path=curated_path,
        validation_summary=validation_summary,
        etl_time_sec=float(etl_time_sec),
        baseline_stats=baseline_stats,
        input_fingerprint=input_fingerprint,
        compatibility_copy_map=compatibility_copy_map,
    )


def run_modeling_stage(
    args: argparse.Namespace,
    paths: PipelinePaths,
    trust_result: TrustStageResult,
    *,
    tracks: list[str],
    legacy_semantics_warning: str | None,
) -> ModelingStageResult:
    t_train_start = time.perf_counter()
    track_summaries: dict[str, dict] = {}
    track_descriptors: list[dict] = []
    combined_metric_frames: list[pd.DataFrame] = []
    compatibility_copy_map = trust_result.compatibility_copy_map

    for track in tracks:
        track_metrics_dir = paths.metrics_dir / track
        track_figures_dir = paths.figures_dir / track
        track_model_dir = paths.model_dir / track
        track_lgbm_dir = paths.model_dir / "lightgbm_bagging" / track
        enable_next = bool(args.enable_next_click and track == "benchmark_transductive")
        ws_cfg = WinnerStyleConfig(
            track_mode=track,
            active_target_col=args.target_col,
            bag_count=args.bag_count,
            neg_pos_ratio=args.neg_pos_ratio,
            lgbm_learning_rate=args.lgbm_learning_rate,
            lgbm_num_leaves=args.lgbm_num_leaves,
            lgbm_n_estimators=args.lgbm_n_estimators,
            calibration_method=args.calibration_method,
            imbalance_strategy=args.imbalance_strategy,
            min_val_positives=args.min_val_positives,
            min_test_positives=args.min_test_positives,
            row_cap_step=args.row_cap_step,
            history_carryover_enabled=True,
            enable_next_click=enable_next,
            random_seed=args.seed,
        )
        track_spec = TrackExecutionSpec(
            track_mode=track,
            metrics_dir=track_metrics_dir,
            figures_dir=track_figures_dir,
            model_dir=track_model_dir,
            lgbm_model_dir=track_lgbm_dir,
            seed=int(args.seed),
            enable_next_click=enable_next,
            winner_style_config=ws_cfg,
        )
        model_summary = train_and_evaluate_models(
            curated_df=trust_result.curated_pd,
            artifacts_metrics_dir=track_spec.metrics_dir,
            figures_dir=track_spec.figures_dir,
            model_dir=track_spec.model_dir,
            seed=track_spec.seed,
            target_col=args.target_col,
            approach_track=track_spec.track_mode,
            model_family=args.model_family,
            row_cap=args.row_cap,
            winner_style_config=track_spec.winner_style_config,
            lightgbm_model_dir=track_spec.lgbm_model_dir,
        )
        track_summaries[track] = model_summary
        prevalence = model_summary.get("class_prevalence", {})
        track_prevalence_plot = track_spec.figures_dir / f"attribution_split_distribution_{track}.png"
        _plot_split_prevalence(prevalence, track_prevalence_plot)
        _ensure_legacy_compat_copy(track_prevalence_plot, track_spec.figures_dir / "fraud_split_distribution.png", compatibility_copy_map)

        track_descriptors.append(
            {
                "track_mode": track,
                "transductive_features_used": bool(track == "benchmark_transductive"),
                "future_context_features_used": bool(enable_next),
                "include_test_distribution_features": bool(track == "benchmark_transductive"),
                "causality_level": "strict_temporal" if track == "production_safe" else "benchmark_transductive",
                "metrics_path": _to_project_relative(track_spec.metrics_dir / "model_metrics.csv"),
                "metrics_uncalibrated_path": _to_project_relative(track_spec.metrics_dir / "model_metrics_uncalibrated.csv"),
                "metrics_calibrated_path": _to_project_relative(track_spec.metrics_dir / "model_metrics_calibrated.csv"),
                "best_model_name": model_summary.get("best_model_name"),
                "best_model_path": model_summary.get("best_model_path"),
                "best_probability_variant": model_summary.get("best_probability_variant"),
                "target_col": args.target_col,
                "target_semantics": "is_attributed_surrogate" if args.target_col == "is_attributed" else "legacy_is_fraud_deprecated",
            }
        )

        track_metrics_path = track_spec.metrics_dir / "model_metrics.csv"
        if track_metrics_path.exists():
            track_df = pd.read_csv(track_metrics_path)
            if not track_df.empty:
                combined_metric_frames.append(track_df)

    stability_summary: dict = {
        "enabled": False,
        "seeds": [],
        "runs": [],
        "delta_between_first_two": None,
    }
    stability_seeds = _parse_stability_seeds(args.stability_seeds)
    if "production_safe" in tracks and len(stability_seeds) >= 2:
        stability_summary["enabled"] = True
        stability_summary["seeds"] = stability_seeds
        stability_model_family = "lightgbm_bagging" if args.model_family == "all" else args.model_family
        for stab_seed in stability_seeds:
            stab_metrics_dir = paths.metrics_dir / "production_safe" / "stability" / f"seed_{stab_seed}"
            stab_figures_dir = paths.figures_dir / "production_safe" / "stability" / f"seed_{stab_seed}"
            stab_model_dir = paths.model_dir / "production_safe" / "stability" / f"seed_{stab_seed}"
            stab_lgbm_dir = paths.model_dir / "lightgbm_bagging" / "production_safe" / "stability" / f"seed_{stab_seed}"
            stab_cfg = WinnerStyleConfig(
                track_mode="production_safe",
                active_target_col=args.target_col,
                bag_count=args.bag_count,
                neg_pos_ratio=args.neg_pos_ratio,
                lgbm_learning_rate=args.lgbm_learning_rate,
                lgbm_num_leaves=args.lgbm_num_leaves,
                lgbm_n_estimators=args.lgbm_n_estimators,
                calibration_method=args.calibration_method,
                imbalance_strategy=args.imbalance_strategy,
                min_val_positives=args.min_val_positives,
                min_test_positives=args.min_test_positives,
                row_cap_step=args.row_cap_step,
                history_carryover_enabled=True,
                enable_next_click=False,
                random_seed=stab_seed,
            )
            stab_summary = train_and_evaluate_models(
                curated_df=trust_result.curated_pd,
                artifacts_metrics_dir=stab_metrics_dir,
                figures_dir=stab_figures_dir,
                model_dir=stab_model_dir,
                seed=stab_seed,
                target_col=args.target_col,
                approach_track="production_safe",
                model_family=stability_model_family,
                row_cap=args.row_cap,
                winner_style_config=stab_cfg,
                lightgbm_model_dir=stab_lgbm_dir,
            )
            best_row = _read_best_metrics_row(
                stab_metrics_dir / "model_metrics.csv",
                best_model_name=str(stab_summary.get("best_model_name", "")),
                best_variant=str(stab_summary.get("best_probability_variant", "uncalibrated")),
            )
            stability_summary["runs"].append(
                {
                    "seed": stab_seed,
                    "model_family": stability_model_family,
                    "best_model_name": stab_summary.get("best_model_name"),
                    "best_probability_variant": stab_summary.get("best_probability_variant"),
                    "val_pr_auc": float(best_row.get("val_pr_auc", np.nan)) if best_row else None,
                    "val_mcc": float(best_row.get("val_mcc", np.nan)) if best_row else None,
                    "val_f1": float(best_row.get("val_f1", np.nan)) if best_row else None,
                    "metrics_path": _to_project_relative(stab_metrics_dir / "model_metrics.csv"),
                }
            )
        if len(stability_summary["runs"]) >= 2:
            a = stability_summary["runs"][0]
            b = stability_summary["runs"][1]
            stability_summary["delta_between_first_two"] = {
                "val_pr_auc_delta_abs": abs(float(a.get("val_pr_auc", np.nan)) - float(b.get("val_pr_auc", np.nan))),
                "val_mcc_delta_abs": abs(float(a.get("val_mcc", np.nan)) - float(b.get("val_mcc", np.nan))),
                "val_f1_delta_abs": abs(float(a.get("val_f1", np.nan)) - float(b.get("val_f1", np.nan))),
            }

    train_time_sec = time.perf_counter() - t_train_start
    if combined_metric_frames:
        pd.concat(combined_metric_frames, ignore_index=True).to_csv(paths.metrics_dir / "model_metrics.csv", index=False)

    primary_track = "production_safe" if "production_safe" in track_summaries else tracks[0]
    model_summary = track_summaries.get(primary_track, {})
    split_plot_path = paths.figures_dir / "attribution_split_distribution.png"
    _plot_split_prevalence(model_summary.get("class_prevalence", {}), split_plot_path)
    _ensure_legacy_compat_copy(split_plot_path, paths.figures_dir / "fraud_split_distribution.png", compatibility_copy_map)
    save_json(
        {
            "approach_track": args.approach_track,
            "model_family": args.model_family,
            "row_cap": int(args.row_cap),
            "target_col": args.target_col,
            "target_semantics": "is_attributed_surrogate" if args.target_col == "is_attributed" else "legacy_is_fraud_deprecated",
            "legacy_semantics_warning": legacy_semantics_warning,
            "calibration_method": args.calibration_method,
            "imbalance_strategy": args.imbalance_strategy,
            "min_val_positives": int(args.min_val_positives),
            "min_test_positives": int(args.min_test_positives),
            "row_cap_step": int(args.row_cap_step),
            "stability_summary": stability_summary,
            "legacy_compatibility_copies": compatibility_copy_map,
            "tracks": track_descriptors,
        },
        paths.provenance_dir / "winner_style_run_metadata.json",
    )

    scored_pd = trust_result.scored_df.select("veracity_score", args.target_col).toPandas()
    if not scored_pd.empty:
        plt.figure(figsize=(9, 5))
        if scored_pd[args.target_col].nunique() > 1:
            sns.histplot(data=scored_pd, x="veracity_score", hue=args.target_col, bins=30, stat="density", common_norm=False, element="step")
        else:
            sns.histplot(data=scored_pd, x="veracity_score", bins=30, stat="density")
        plt.title("Veracity Score Distribution")
        plt.xlabel("Veracity Score (0-1)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "veracity_score_distribution.png", dpi=150)
        plt.close()

    drift_metrics = trust_result.validation_summary.get("drift_metrics", {})
    if drift_metrics:
        drift_rows = []
        for feat, vals in drift_metrics.items():
            drift_rows.append({"feature": feat, "psi": float(vals.get("psi", 0.0)), "kl": float(vals.get("kl", 0.0))})
        drift_df = pd.DataFrame(drift_rows)
        drift_plot_df = drift_df.melt(id_vars="feature", var_name="metric", value_name="value")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=drift_plot_df, x="feature", y="value", hue="metric")
        plt.title("Drift Metric Summary (PSI / KL)")
        plt.xlabel("Feature")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "drift_metric_summary.png", dpi=150)
        plt.close()

    return ModelingStageResult(
        track_summaries=track_summaries,
        track_descriptors=track_descriptors,
        combined_metric_frames=combined_metric_frames,
        primary_track=primary_track,
        primary_track_summary=model_summary,
        train_time_sec=float(train_time_sec),
        stability_summary=stability_summary,
    )


def run_audit_and_overhead_stage(args: argparse.Namespace, paths: PipelinePaths, trust_result: TrustStageResult) -> dict:
    audit_log = paths.audit_dir / "audit_log.jsonl"
    audit_records = [r.asDict(recursive=True) for r in trust_result.curated_df.limit(20000).collect()]
    append_audit_batch(
        log_path=audit_log,
        batch_id=f"batch_{int(time.time())}",
        records=audit_records,
        batch_metadata={
            "stage": "batch_curation",
            "source": str(args.input_csv),
            "rows_used_for_audit": len(audit_records),
        },
    )
    verify_ok = verify_audit_log(audit_log)
    save_json(verify_ok, paths.audit_dir / "verification_report.json")

    tampered_log = tamper_log_copy(audit_log, paths.audit_dir / "audit_log_tampered_demo.jsonl")
    verify_bad = verify_audit_log(tampered_log)
    save_json(verify_bad, paths.audit_dir / "verification_report_tampered.json")

    overhead = measure_integrity_overhead(audit_records[: min(5000, len(audit_records))], loops=5, warmup=1)
    save_json(overhead, paths.metrics_dir / "integrity_overhead.json")

    overhead_df = pd.DataFrame(
        [
            {
                "mode": "without_integrity",
                "throughput_rps": overhead["median_throughput_plain_rps"],
                "throughput_iqr": overhead["iqr_plain_sec"],
                "latency_sec": overhead["median_plain_sec"],
                "latency_iqr": overhead["iqr_plain_sec"],
            },
            {
                "mode": "with_integrity",
                "throughput_rps": overhead["median_throughput_hash_rps"],
                "throughput_iqr": overhead["iqr_hash_sec"],
                "latency_sec": overhead["median_hash_sec"],
                "latency_iqr": overhead["iqr_hash_sec"],
            },
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tp_ax = sns.barplot(data=overhead_df, x="mode", y="throughput_rps", ax=axes[0], hue="mode", palette="mako", legend=False)
    for i, row in overhead_df.reset_index(drop=True).iterrows():
        tp_ax.text(i, float(row["throughput_rps"]) * 1.02, f"{row['throughput_rps']:.1f}", ha="center", fontsize=9)
    axes[0].set_title("Throughput Comparison (Median)")
    axes[0].set_xlabel("Mode")
    axes[0].set_ylabel("Records / second")
    axes[0].tick_params(axis="x", rotation=20)

    lat_ax = sns.barplot(data=overhead_df, x="mode", y="latency_sec", ax=axes[1], hue="mode", palette="rocket", legend=False)
    for i, row in overhead_df.reset_index(drop=True).iterrows():
        lat_ax.text(i, float(row["latency_sec"]) * 1.02, f"{row['latency_sec']:.4f}s", ha="center", fontsize=9)
    axes[1].set_title("Latency Comparison (Median)")
    axes[1].set_xlabel("Mode")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Integrity Overhead (per-record overhead: {overhead['per_record_overhead_us']:.2f} µs)", fontsize=10)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "audit_overhead_throughput_latency.png", dpi=150)
    plt.close(fig)

    return {"verify_ok": verify_ok, "verify_bad": verify_bad, "overhead": overhead}


def run_optional_stream_comparison_stage(args: argparse.Namespace, paths: PipelinePaths, trust_result: TrustStageResult) -> dict | None:
    stream_metrics_path = paths.metrics_dir / "streaming_metrics.csv"
    if not stream_metrics_path.exists():
        return None
    try:
        stream_df = pd.read_csv(stream_metrics_path)
        if stream_df.empty:
            return {"generated": False, "reason": "empty_stream_metrics"}
        batch_records = float(trust_result.validation_summary.get("total_records", len(trust_result.curated_pd)))
        batch_throughput = batch_records / trust_result.etl_time_sec if trust_result.etl_time_sec > 0 else 0.0
        comp_df = pd.DataFrame(
            [
                {"pipeline": "batch", "throughput_rps": batch_throughput, "latency_sec": trust_result.etl_time_sec},
                {"pipeline": "stream_avg", "throughput_rps": float(stream_df["throughput_rps"].mean()), "latency_sec": float(stream_df["latency_sec"].mean())},
            ]
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.barplot(data=comp_df, x="pipeline", y="throughput_rps", ax=axes[0], hue="pipeline", legend=False, palette="deep")
        sns.barplot(data=comp_df, x="pipeline", y="latency_sec", ax=axes[1], hue="pipeline", legend=False, palette="pastel")
        axes[0].set_title("Batch vs Stream Throughput")
        axes[0].set_xlabel("Pipeline")
        axes[0].set_ylabel("Records / second")
        axes[1].set_title("Batch vs Stream Latency")
        axes[1].set_xlabel("Pipeline")
        axes[1].set_ylabel("Seconds")
        plt.tight_layout()
        out_path = paths.figures_dir / "batch_vs_stream_comparison_summary.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return {"generated": True, "path": _to_project_relative(out_path)}
    except Exception as exc:
        return {"generated": False, "reason": str(exc)}


def build_and_save_run_metadata(
    args: argparse.Namespace,
    paths: PipelinePaths,
    trust_result: TrustStageResult,
    modeling_result: ModelingStageResult,
) -> dict:
    download_meta_path = paths.provenance_dir / "dataset_download_metadata.json"
    dataset_source = "user_provided"
    if download_meta_path.exists():
        try:
            dl_meta = json.loads(download_meta_path.read_text(encoding="utf-8"))
            if dl_meta.get("dataset_path") == str(args.input_csv):
                dataset_source = dl_meta.get("dataset_source", "user_provided")
        except json.JSONDecodeError:
            dataset_source = "user_provided"

    column_schema = [{"name": name, "type": dtype} for name, dtype in trust_result.curated_df.dtypes]
    run_metadata = build_run_metadata(
        dataset_path=args.input_csv,
        dataset_source=dataset_source,
        dataset_mode=detect_dataset_mode(args.input_csv, fallback_generated=(dataset_source == "synthetic_fallback")),
        column_schema=column_schema,
        repo_dir=PROJECT_ROOT,
        extra={
            "etl_time_sec": trust_result.etl_time_sec,
            "training_time_sec": modeling_result.train_time_sec,
            "curated_output_parquet": _to_project_relative(trust_result.curated_path),
            "validation_report": _to_project_relative(paths.metrics_dir / "validation_report.json"),
            "model_metrics": _to_project_relative(paths.metrics_dir / "model_metrics.csv"),
            "winner_style_run_metadata": _to_project_relative(paths.provenance_dir / "winner_style_run_metadata.json"),
            "approach_track": args.approach_track,
            "model_family": args.model_family,
            "row_cap": int(args.row_cap),
            "target_col": args.target_col,
            "target_semantics": "is_attributed_surrogate" if args.target_col == "is_attributed" else "legacy_is_fraud_deprecated",
            "calibration_method": args.calibration_method,
            "imbalance_strategy": args.imbalance_strategy,
            "min_val_positives": int(args.min_val_positives),
            "min_test_positives": int(args.min_test_positives),
            "row_cap_step": int(args.row_cap_step),
        },
    )
    save_json(run_metadata, paths.provenance_dir / "run_metadata.json")
    return run_metadata


def build_and_save_batch_summary(
    args: argparse.Namespace,
    paths: PipelinePaths,
    trust_result: TrustStageResult,
    modeling_result: ModelingStageResult,
    debug_outputs: dict,
    audit_result: dict,
    legacy_semantics_warning: str | None,
) -> dict:
    model_summary = modeling_result.primary_track_summary
    summary = {
        "input_csv": str(args.input_csv),
        "target_col": args.target_col,
        "target_semantics": "is_attributed_surrogate" if args.target_col == "is_attributed" else "legacy_is_fraud_deprecated",
        "legacy_semantics_warning": legacy_semantics_warning,
        "approach_track": args.approach_track,
        "model_family": args.model_family,
        "row_cap": int(args.row_cap),
        "primary_track": modeling_result.primary_track,
        "etl_time_sec": trust_result.etl_time_sec,
        "training_time_sec": modeling_result.train_time_sec,
        "curated_parquet": _to_project_relative(trust_result.curated_path),
        "validation_report": _to_project_relative(paths.metrics_dir / "validation_report.json"),
        "model_metrics": _to_project_relative(paths.metrics_dir / "model_metrics.csv"),
        "winner_style_run_metadata": _to_project_relative(paths.provenance_dir / "winner_style_run_metadata.json"),
        "track_summaries": modeling_result.track_summaries,
        "class_prevalence": model_summary.get("class_prevalence", {}),
        "support_counts": model_summary.get("support_counts", {}),
        "split_overlap_counts": model_summary.get("split_overlap_counts", {}),
        "stability_summary": modeling_result.stability_summary,
        "legacy_compatibility_copies": trust_result.compatibility_copy_map,
        "debug_artifacts": {
            "winner_style_hardening_report": _to_project_relative(debug_outputs["winner_style_hardening_report"]),
            "leakage_and_causality_checks": _to_project_relative(debug_outputs["leakage_and_causality_checks"]),
            "calibration_summary": _to_project_relative(debug_outputs["calibration_summary"]),
            "track_comparison_summary": _to_project_relative(debug_outputs["track_comparison_summary"]),
        },
        "shared_trust_controls": {
            "dq_validation_report": _to_project_relative(paths.metrics_dir / "validation_report.json"),
            "audit_verification_report": _to_project_relative(paths.audit_dir / "verification_report.json"),
            "integrity_overhead_report": _to_project_relative(paths.metrics_dir / "integrity_overhead.json"),
            "streaming_metrics": _to_project_relative(paths.metrics_dir / "streaming_metrics.csv"),
        },
        "audit_verification": audit_result["verify_ok"],
        "tamper_verification": audit_result["verify_bad"],
    }
    save_json(summary, paths.metrics_dir / "batch_pipeline_summary.json")
    return summary


def run_batch_pipeline(args: argparse.Namespace) -> dict:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if args.target_col not in {"is_attributed", "is_fraud"}:
        raise ValueError("target_col must be one of: is_attributed, is_fraud")
    if args.min_val_positives <= 0 or args.min_test_positives <= 0:
        raise ValueError("--min-val-positives and --min-test-positives must be > 0")
    if args.row_cap_step <= 0:
        raise ValueError("--row-cap-step must be > 0")

    legacy_semantics_warning, tracks = _legacy_warning_and_tracks(args)
    if args.enable_next_click and "production_safe" in tracks:
        raise ValueError("--enable-next-click cannot be used when production_safe track is selected")

    paths = prepare_pipeline_paths(args)
    for p in [paths.metrics_dir, paths.audit_dir, paths.provenance_dir, paths.curated_dir, paths.debug_dir, paths.figures_dir, paths.model_dir]:
        p.mkdir(parents=True, exist_ok=True)

    trust_result = run_trust_ingestion_stage(args, paths)
    try:
        modeling_result = run_modeling_stage(args, paths, trust_result, tracks=tracks, legacy_semantics_warning=legacy_semantics_warning)
        audit_result = run_audit_and_overhead_stage(args, paths, trust_result)
        run_metadata = build_and_save_run_metadata(args, paths, trust_result, modeling_result)
        debug_outputs = _build_debug_outputs(
            artifacts_dir=paths.artifacts_dir,
            metrics_dir=paths.metrics_dir,
            figures_dir=paths.figures_dir,
            track_summaries=modeling_result.track_summaries,
            tracks=tracks,
            args=args,
            legacy_semantics_warning=legacy_semantics_warning,
            stability_summary=modeling_result.stability_summary,
            compat_copy_map=trust_result.compatibility_copy_map,
        )
        _ = run_optional_stream_comparison_stage(args, paths, trust_result)
        summary = build_and_save_batch_summary(
            args,
            paths,
            trust_result,
            modeling_result,
            debug_outputs,
            audit_result,
            legacy_semantics_warning,
        )
        _batch_result = BatchPipelineResult(
            summary=summary,
            trust_stage=trust_result,
            modeling_stage=modeling_result,
            debug_outputs=debug_outputs,
            run_metadata=run_metadata,
        )
        return _batch_result.summary
    finally:
        try:
            trust_result.spark.stop()
        except Exception:
            pass
