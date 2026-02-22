# Final Assets Manifest

Supervised target for this implementation is Kaggle `is_attributed` (attribution/conversion proxy), used as a surrogate target for demonstrating veracity/integrity-aware ingestion and classification. It is not a native fraud ground-truth label.

| Asset path | What it shows | Where to use | Notes |
|---|---|---|---|
| `reports/figures/attribution_prevalence.png` | Target Prevalence (`is_attributed`) | Report results; PPT Slide 1 | Positive class = attributed event (surrogate target) |
| `reports/figures/attribution_split_distribution.png` | Target positive rate by split | Report setup; PPT Slide 3 | Train/val/test imbalance visibility |
| `reports/figures/data_quality_violations.png` | DQ rule violations (count + %) | Report veracity; PPT Slide 4 | Required |
| `reports/figures/top_dq_rules_percentage.png` | Ranked DQ rule failure percentages | Report veracity; PPT Slide 4 | Companion to counts |
| `reports/figures/veracity_score_distribution.png` | Veracity score distribution | Report veracity; PPT Slide 5 | Ingestion trust summary |
| `reports/figures/drift_metric_summary.png` | PSI/KL drift metrics | Report veracity; PPT Slide 5 | Drift snapshot |
| `reports/figures/production_safe/model_metrics_comparison_production_safe.png` | Precision/Recall/F1/PR-AUC/MCC by model | Report modeling; PPT Slide 6 | Use production-safe as primary |
| `reports/figures/production_safe/pr_curves_production_safe.png` | PR curves + prevalence baseline | Report modeling; PPT Slide 7 | Primary curve under imbalance |
| `reports/figures/production_safe/roc_curves_production_safe.png` | ROC curves | Report modeling; PPT Slide 7 | Secondary curve |
| `reports/figures/production_safe/best_model_confusion_matrix_production_safe.png` | Best-model confusion matrix | Report modeling; PPT Slide 8 | Counts + normalized percentages |
| `reports/figures/production_safe/class_probability_distribution_production_safe_<model>.png` | Best-model probability distribution | Report modeling; PPT Slide 9 | Calibrated variant preferred |
| `reports/figures/production_safe/calibration_curve_production_safe_<model>.png` | Calibration curve | Appendix; PPT Slide 9 optional | Platt on temporal validation |
| `reports/figures/production_safe/precision_recall_at_k_production_safe_<model>.png` | Precision@K and Recall@K | Viva/Appendix | Ranking interpretation |
| `reports/figures/production_safe/lift_at_k_production_safe_<model>.png` | Lift@K | Viva/Appendix | Ranking interpretation |
| `reports/figures/audit_overhead_throughput_latency.png` | Integrity overhead benchmark | Report integrity; PPT Slide 10 | Fair benchmark annotation |
| `reports/figures/streaming_latency_throughput.png` | Streaming latency/throughput trends | Report streaming; PPT Slide 11 | Spark progress-backed metrics |
| `reports/figures/batch_vs_stream_comparison_summary.png` | Batch vs stream comparison | Report scalability; PPT Slide 11 | Runtime comparison |
| `artifacts/metrics/model_metrics.csv` | Model metrics table | Report table source | LR/RF/XGBoost |
| `artifacts/metrics/production_safe/model_metrics_uncalibrated.csv` | Uncalibrated production-safe metrics | Appendix | Pre-calibration reference |
| `artifacts/metrics/production_safe/model_metrics_calibrated.csv` | Calibrated production-safe metrics | Report table source | Use for threshold selection/curves |
| `artifacts/metrics/production_safe/topk_metrics_production_safe_<model>.csv` | Top-K ranking metrics | Viva/report appendix | K={50,100,500,1000} |
| `artifacts/metrics/production_safe/threshold_analysis_best_model.csv` | Threshold sweep table | Report table source | Uses best probability variant |
| `artifacts/metrics/validation_report.csv` | Validation table | Report table source | DQ findings |
| `artifacts/metrics/final_results_summary.json` | Canonical final values | Report/PPT numeric source | Source of truth |
| `artifacts/debug/winner_style_hardening_report.md` | Hardening corrections summary | Viva defense | Maps checks to concerns |
| `artifacts/debug/leakage_and_causality_checks.json` | Leakage/causality assertions | Viva defense | Includes split/order/parity checks |
| `artifacts/debug/calibration_summary.json` | Calibration fit + artifact map | Viva defense | Method and support counts |
| `artifacts/debug/track_comparison_summary.md` | Track claim separation summary | Viva defense | Primary vs benchmark usage |
