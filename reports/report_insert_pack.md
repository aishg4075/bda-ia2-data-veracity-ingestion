# Report Insert Pack

## 1) Experimental Setup (Final Run)
- Dataset source/mode: Kaggle TalkingData `data/train.csv`, capped to 300,000 rows.
- Supervised target for this implementation is Kaggle `is_attributed` (attribution/conversion proxy), used as a surrogate target for demonstrating veracity/integrity-aware ingestion and classification. It is not a native fraud ground-truth label.
- Software stack (from `artifacts/provenance/run_metadata.json`):
  - Python: 3.11.14 (main, Oct  9 2025, 16:16:55) [Clang 17.0.0 (clang-1700.6.3.2)]
  - PySpark: 3.5.1
  - scikit-learn: 1.5.2
  - XGBoost: 2.1.1
  - Platform: macOS-15.6.1-arm64-arm-64bit

## 2) Data Quality / Veracity Findings
- Top rule violations (count + percent):
  - rule_drift: 60336 (20.11%)
  - rule_outlier: 41819 (13.94%)
  - rule_uniqueness: 7734 (2.58%)
- Total records evaluated in validation snapshot: 300000
- Duplicate-event removals flagged by uniqueness rule: 7734
- Veracity scoring summary source: `reports/figures/veracity_score_distribution.png` and `artifacts/metrics/validation_report.json`.

## 3) Modeling Results (Imbalance-Aware, Dual-Track)
- Best-model selection rule in each track: **PR-AUC -> MCC -> F1** (validation split).
- Track comparison should be reported side-by-side:

| Track | Causality level | Purpose | Primary metrics source |
|---|---|---|---|
| `production_safe` | strict temporal | Main report claims | `artifacts/metrics/production_safe/model_metrics.csv` |
| `benchmark_transductive` | competition-style / transductive benchmark | Reference only | `artifacts/metrics/benchmark_transductive/model_metrics.csv` |

- Consolidated model table (all tracks): `artifacts/metrics/model_metrics.csv`.
- Calibrated/uncalibrated tables (per track):
  - `artifacts/metrics/production_safe/model_metrics_uncalibrated.csv`
  - `artifacts/metrics/production_safe/model_metrics_calibrated.csv`
  - `artifacts/metrics/benchmark_transductive/model_metrics_uncalibrated.csv`
  - `artifacts/metrics/benchmark_transductive/model_metrics_calibrated.csv`
- All-negative baseline accuracy remains a required sanity check.
- Accuracy alone is misleading under severe imbalance; PR-AUC, MCC, and threshold trade-offs are primary interpretation metrics.
- Ranking interpretation for viva:
  - `topk_metrics_<track>_<model>.csv`
  - `precision_recall_at_k_<track>_<model>.png`
  - `lift_at_k_<track>_<model>.png`

## 4) Integrity Validation Findings
- Audit verification (normal chain): True.
- Tamper verification (expected fail): False with first mismatch index 0.
- Overhead summary (fair repeated benchmark):
  - Median throughput without integrity: 77337.38 rec/s
  - Median throughput with integrity: 69763.99 rec/s
  - Per-record overhead: 1.404 µs

## 5) Streaming Findings
- Mode: kafka.
- Metrics source: Spark progress fields + wall-clock instrumentation in `artifacts/metrics/streaming_metrics.csv`.
- Average latency: 2.5604 sec
- Average wall throughput: 873.29 rec/s
- Average processedRowsPerSecond: 15136.29

## 6) Limitations
- `is_attributed` is a surrogate target and not native fraud ground-truth.
- Severe imbalance makes recall highly threshold-sensitive.
- Local-machine execution limits direct extrapolation to multi-node production scale.
- `benchmark_transductive` results can use competition-like feature timing and must not be used for production-causal claims.

## 7) What to Copy Into Final Report
- Core summary metrics: `artifacts/metrics/final_results_summary.json`
- Model table: `artifacts/metrics/model_metrics.csv`
- Production-safe model table: `artifacts/metrics/production_safe/model_metrics.csv`
- Benchmark-only model table: `artifacts/metrics/benchmark_transductive/model_metrics.csv`
- Threshold table: `artifacts/metrics/threshold_analysis_best_model.csv`
- Validation table: `artifacts/metrics/validation_report.csv`
- Hardening traceability:
  - `artifacts/debug/winner_style_hardening_report.md`
  - `artifacts/debug/leakage_and_causality_checks.json`
  - `artifacts/debug/calibration_summary.json`
  - `artifacts/debug/track_comparison_summary.md`
- Figures:
  - `reports/figures/data_quality_violations.png`
  - `reports/figures/attribution_prevalence.png`
  - `reports/figures/attribution_split_distribution.png`
  - `reports/figures/production_safe/model_metrics_comparison_production_safe.png`
  - `reports/figures/production_safe/pr_curves_production_safe.png`
  - `reports/figures/production_safe/best_model_confusion_matrix_production_safe.png`
  - `reports/figures/production_safe/calibration_curve_production_safe_<model>.png`
  - `reports/figures/production_safe/precision_recall_at_k_production_safe_<model>.png`
  - `reports/figures/production_safe/lift_at_k_production_safe_<model>.png`
  - `reports/figures/audit_overhead_throughput_latency.png`
  - `reports/figures/streaming_latency_throughput.png`
