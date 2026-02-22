# Final Submission Checklist

| Item | Status | Evidence |
|---|---|---|
| Compliance A–P remains intact | Pass | `notebooks/01_veracity_integrity_ingestion.ipynb`, `scripts/`, `docs/`, `docker/` |
| Sample run completed | Pass | `artifacts/metrics/model_metrics_sample.csv`, `artifacts/metrics/validation_report_sample.json`, `artifacts/provenance/run_metadata_sample.json` |
| Bigger dataset run completed OR fallback documented | Pass | `artifacts/metrics/scaling_status.md`, `artifacts/provenance/dataset_download_metadata.json` |
| Required plots present | Pass | `reports/figures/*.png` including `streaming_latency_throughput.png` |
| Final summary files present | Pass | `artifacts/metrics/final_results_summary.json`, `artifacts/metrics/final_results_summary.csv` |
| Report insert pack present | Pass | `reports/report_insert_pack.md` |
| Presentation mapping present | Pass | `reports/presentation_slide_mapping.md` |
| Notebook headings unchanged | Pass | `notebooks/01_veracity_integrity_ingestion.ipynb` (14 required headings retained in order) |
| No hardcoded absolute paths introduced in finalization docs | Pass | `reports/*.md`, `artifacts/metrics/*.md` use repo-relative paths |
