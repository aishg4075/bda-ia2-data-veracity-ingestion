# Final Debug Validation Checklist

Timestamp (UTC): `2026-02-21T10:58:56.301316+00:00`

| Check | Pass/Fail | Evidence |
|---|---|---|
| Target semantics caveat present across notebook/README/report/PPT assets | PASS | `notebooks/01_veracity_integrity_ingestion.ipynb`, `README.md`, `reports/report_insert_pack.md`, `reports/presentation_slide_mapping.md`, `reports/final_assets_manifest.md` |
| Best-model selection criterion documented as PR-AUC -> MCC -> F1 | PASS | `artifacts/metrics/model_metrics.csv`, `reports/report_insert_pack.md`, `artifacts/debug/leakage_and_causality_checks.json` |
| Threshold tradeoff artifact generated | PASS | `artifacts/metrics/production_safe/threshold_analysis_best_model.csv`, `artifacts/metrics/benchmark_transductive/threshold_analysis_best_model.csv` |
| All required plots present and presentation-ready | PASS | `reports/figures/` |
| No hardcoded absolute paths reintroduced | PASS | path scan on key docs/notebook |
| Leakage/split/metric parity checks remain PASS | PASS | `artifacts/debug/leakage_and_causality_checks.json` |
