# Viva Code Walkthrough (Refactored Flow)

This note is a quick code map for explaining the project implementation in viva without walking through every file line-by-line.

## 1) High-Level Call Flow

1. `scripts/run_batch_pipeline.py`
2. `ingestion_trust.batch_orchestrator.run_batch_pipeline(args)`
3. Stage execution inside orchestrator:
   - `run_trust_ingestion_stage(...)`
   - `run_modeling_stage(...)`
   - `run_audit_and_overhead_stage(...)`
   - `run_optional_stream_comparison_stage(...)`
   - metadata/debug/summary writers

## 2) Module Responsibilities

### `scripts/run_batch_pipeline.py`
- Thin CLI adapter only.
- Keeps CLI arguments and validation guardrails (target semantics, next-click restriction, positive-support settings).
- Prints final summary JSON returned by orchestrator.

### `src/ingestion_trust/batch_orchestrator.py`
- Owns the batch execution sequence and report/debug artifact coordination.
- Preserves artifact names/paths and track segregation.
- Generates:
  - metrics summaries,
  - provenance metadata,
  - debug hardening outputs,
  - batch summary JSON.

### `src/ingestion_trust/pipeline_contracts.py`
- Defines dataclasses used between stages:
  - `PipelinePaths`
  - `TrustStageResult`
  - `ModelingStageResult`
  - `BatchPipelineResult`
- Makes the stage interfaces explicit and easier to explain.

### `src/ingestion_trust/spark_utils.py`
- Spark-side ingestion and trust preprocessing:
  - schema casting,
  - feature engineering,
  - baseline stats,
  - veracity checks,
  - drift metrics,
  - veracity scoring.

### `src/ingestion_trust/winner_style.py`
- Winner-style batch modeling preparation with hardening:
  - temporal splits,
  - row-cap selection with minimum positive support,
  - causal feature generation (`production_safe`),
  - optional benchmark transductive features,
  - leakage-safe OOF target/stat encoding,
  - bagging negative subsampling.

### `src/ingestion_trust/modeling_pipeline.py`
- Dedicated modeling engine:
  - baseline models (LR/RF/XGB if available),
  - LightGBM bagging,
  - Platt calibration,
  - PR/ROC/threshold/top-K artifacts,
  - per-track model summary output.

### `src/ingestion_trust/core.py`
- Shared utilities + compatibility facade.
- Still contains:
  - audit chain / Merkle / integrity verification utilities,
  - provenance helpers,
  - dataset loading/fallback utilities,
  - integrity overhead measurement.
- Re-exports moved functions for backward compatibility:
  - `WinnerStyleConfig`
  - `train_lightgbm_bagging`
  - `train_and_evaluate_models`
  - trust plotting helpers

## 3) Where Hardening Guards Live

- Target semantics / legacy target guard:
  - CLI validation in `scripts/run_batch_pipeline.py`
  - duplicated defensive checks in `batch_orchestrator.run_batch_pipeline(...)`
- Production-safe next-click restriction:
  - CLI validation and modeling-layer validation
- Causality / leakage checks:
  - `winner_style.py` + debug validation in orchestrator (`_build_debug_outputs`)
- Calibration after subsampling:
  - `modeling_pipeline.py`
- Track segregation + report-safe labels:
  - `modeling_pipeline.py` and orchestrator debug/provenance writers

## 4) What to Emphasize in Viva

1. Trust controls are the core project (veracity + integrity), not just ML.
2. Modeling branch is hardened and track-separated:
   - `production_safe` for primary claims
   - `benchmark_transductive` for reference only
3. Refactor simplifies explanation:
   - CLI -> orchestrator -> stages -> artifacts
4. Existing outputs/tests remain compatible after refactor.

## 5) Key Artifacts (What the Code Produces)

- Trust/control artifacts:
  - `artifacts/metrics/validation_report.json`
  - `artifacts/audit/verification_report.json`
  - `artifacts/metrics/integrity_overhead.json`
- Modeling artifacts (per track):
  - `artifacts/metrics/<track>/model_metrics*.csv`
  - `artifacts/metrics/<track>/topk_metrics_*.csv`
  - `reports/figures/<track>/*.png`
- Hardening/debug artifacts:
  - `artifacts/debug/leakage_and_causality_checks.json`
  - `artifacts/debug/calibration_summary.json`
  - `artifacts/debug/track_comparison_summary.md`
  - `artifacts/debug/winner_style_hardening_report.md`
