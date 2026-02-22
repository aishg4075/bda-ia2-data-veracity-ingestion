# Presentation Slide Mapping

## Slide 1 — Problem Statement & Caveat
- Asset path(s):
  - `reports/figures/attribution_prevalence.png`
- Talking points:
  - Trust-in-ingestion objective for high-velocity ad events.
  - Caveat: target is Kaggle `is_attributed` surrogate, not native fraud ground-truth.
  - Positive class = attributed event.

## Slide 2 — Architecture Overview
- Asset path(s):
  - `artifacts/metrics/final_results_summary.json`
- Talking points:
  - Batch + streaming + integrity audit chain in one pipeline.
  - Shared validation and feature logic across modes.

## Slide 3 — Dataset & Split Imbalance
- Asset path(s):
  - `reports/figures/attribution_split_distribution.png`
- Talking points:
  - Kaggle `train.csv` capped at 300k rows.
  - Severe imbalance persists across train/val/test.

## Slide 4 — Veracity Rule Findings
- Asset path(s):
  - `reports/figures/data_quality_violations.png`
  - `reports/figures/top_dq_rules_percentage.png`
- Talking points:
  - Top DQ violations by count and percent.
  - Rule-level failures inform ingestion trust controls.

## Slide 5 — Veracity Score and Drift
- Asset path(s):
  - `reports/figures/veracity_score_distribution.png`
  - `reports/figures/drift_metric_summary.png`
- Talking points:
  - Per-record veracity score distribution.
  - PSI/KL drift snapshots for monitored features.

## Slide 6 — Model Comparison (Why PR-AUC/MCC > Accuracy)
- Asset path(s):
  - `reports/figures/production_safe/model_metrics_comparison_production_safe.png`
- Talking points:
  - Best-model criterion: PR-AUC -> MCC -> F1.
  - Accuracy alone is misleading under extreme imbalance.
  - All intended models shown (LR/RF/XGBoost).

## Slide 7 — PR and ROC Curves
- Asset path(s):
  - `reports/figures/production_safe/pr_curves_production_safe.png`
  - `reports/figures/production_safe/roc_curves_production_safe.png`
- Talking points:
  - PR curve is the primary decision curve under imbalance.
  - ROC shown as complementary discrimination view.

## Slide 8 — Confusion Matrix + Threshold Trade-off
- Asset path(s):
  - `reports/figures/production_safe/best_model_confusion_matrix_production_safe.png`
  - `artifacts/metrics/production_safe/threshold_analysis_best_model.csv`
- Talking points:
  - Confusion matrix includes counts and normalized percentages.
  - Threshold sweep shows precision/recall/F1/MCC trade-offs.
  - Default threshold 0.5 highlighted for discussion.

## Slide 9 — Probability & Feature Explanations
- Asset path(s):
  - `reports/figures/production_safe/class_probability_distribution_production_safe_<model>.png`
  - `reports/figures/production_safe/calibration_curve_production_safe_<model>.png`
  - `reports/figures/production_safe/precision_recall_at_k_production_safe_<model>.png`
  - `reports/figures/production_safe/lift_at_k_production_safe_<model>.png`
- Talking points:
  - Class probability separation of the best model.
  - Top feature signals from leakage-safe feature space.

## Slide 10 — Integrity Validation & Overhead
- Asset path(s):
  - `reports/figures/audit_overhead_throughput_latency.png`
- Talking points:
  - Audit chain verification passes on untampered log.
  - Tamper check fails as expected at first mismatch.
  - Overhead benchmark uses fair repeated methodology.

## Slide 11 — Streaming Runtime Evidence
- Asset path(s):
  - `reports/figures/streaming_latency_throughput.png`
  - `reports/figures/batch_vs_stream_comparison_summary.png`
- Talking points:
  - Streaming metrics sourced from Spark progress fields.
  - Latency/throughput trend across micro-batches.
  - Batch vs stream runtime contrast.

## Slide 12 — Final Conclusions & Limits
- Asset path(s):
  - `artifacts/metrics/final_results_summary.csv`
- Talking points:
  - Copy final numbers only from generated artifacts.
  - Limitations: surrogate target semantics, imbalance sensitivity, local-scale constraints.
