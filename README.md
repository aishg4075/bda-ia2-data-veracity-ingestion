# bda-topic17-ingestion-trust

**Topic 17:** Data Veracity and Integrity in Distributed Systems: Ensuring Trust in Information Ingestion  
**Use case:** High-velocity web advertising clickstream veracity validation + attribution-surrogate modeling.

This repository implements a full, notebook-first case study with:

1. Ingestion-time **veracity/data-quality validation**
2. **Integrity + provenance + tamper-evident audit logging**
3. **Batch vs streaming** comparison using Spark + Kafka

## Project Overview

The pipeline ingests clickstream events, applies explicit data quality rules, computes a per-record veracity score, trains/evaluates attribution-surrogate models (`is_attributed`), and logs tamper-evident audit metadata using hash chaining + Merkle roots.  
The same schema, validation, and feature logic is reused across batch and streaming.

## Repository Structure

```text
bda-topic17-ingestion-trust/
  README.md
  requirements.txt
  .gitignore
  notebooks/
    01_veracity_integrity_ingestion.ipynb
  data/
    sample_click_fraud.csv
    data_sources.md
  artifacts/
    models/
    audit/
    provenance/
    metrics/
    stream/
      output_parquet/
      checkpoints/
  reports/
    figures/
  scripts/
    download_dataset.py
    make_sample.py
    run_batch_pipeline.py
    run_stream_pipeline.py
    kafka_producer.py
    kafka_consumer_debug.py
    generate_references.py
  docker/
    docker-compose.kafka.yml
    README_kafka.md
  docs/
    research_alignment.md
    references.md
  src/
    ingestion_trust/
      __init__.py
      core.py
      winner_style.py
      spark_utils.py
```

## Setup

### 1) Python environment

Recommended: Python **3.10 or 3.11**.

```bash
cd bda-topic17-ingestion-trust
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Spark prerequisites

- Java 8/11/17 installed and on `PATH`
- Local Spark runtime is pulled via `pyspark`

Quick check:

```bash
python -c "import pyspark; print(pyspark.__version__)"
java -version
```

## Data Preparation

Use the included sample for immediate run:

- `data/sample_click_fraud.csv`

Optional: attempt full dataset resolution with fallback strategy:

```bash
python scripts/download_dataset.py --output-dir data --prefer-kaggle
python scripts/make_sample.py --input-csv data/click_fraud_full.csv --output-csv data/sample_click_fraud.csv
```

Behavior:

1. Try Kaggle (if credentials exist)
2. Try open CSV URL(s)
3. Fallback to synthetic clickstream generation

## Run: Notebook (Primary Deliverable)

```bash
jupyter notebook notebooks/01_veracity_integrity_ingestion.ipynb
```

Run cells top-to-bottom. The notebook covers:

- ETL + Spark validation rules
- Veracity scoring and summary metrics
- Model training/evaluation (LR, RF, XGBoost)
- Streaming architecture and Kafka integration steps
- Tamper-evident audit logging + tamper detection demo
- Integrity overhead measurement
- Research alignment and social impact analysis

## Run: Batch CLI (Optional Wrapper)

```bash
python scripts/run_batch_pipeline.py \
  --input-csv data/sample_click_fraud.csv \
  --artifacts-dir artifacts \
  --figures-dir reports/figures \
  --model-dir artifacts/models \
  --target-col is_attributed \
  --approach-track both \
  --model-family all \
  --row-cap 600000 \
  --min-val-positives 100 \
  --min-test-positives 100 \
  --row-cap-step 50000 \
  --imbalance-strategy subsampling_only \
  --calibration-method platt \
  --stability-seeds 42,43 \
  --bag-count 5 \
  --neg-pos-ratio 200 \
  --lgbm-learning-rate 0.05 \
  --lgbm-num-leaves 64 \
  --lgbm-n-estimators 600
```

### Winner-style dual track modes

- `production_safe`: strict temporal splits and causally-safe feature generation (primary interpretation path).
- `benchmark_transductive`: competition-style day/hour split; optional `--enable-next-click` for future-context benchmark only.

Guardrail:
- `--enable-next-click` is blocked whenever `production_safe` is selected.
- If `model-family` includes `lightgbm_bagging` and LightGBM is unavailable, the run fails with an actionable install message.
- Default supervised target is `is_attributed` (surrogate attribution/conversion semantics).
- Legacy `is_fraud` target is disabled by default and only available with explicit `--allow-legacy-fraud-target` opt-in.

## Run: Streaming CLI (Optional Wrapper)

### Kafka mode (default)

1. Start Kafka:

```bash
cd docker
docker-compose -f docker-compose.kafka.yml up -d
cd ..
```

2. Produce events:

```bash
python scripts/kafka_producer.py --input-csv data/sample_click_fraud.csv --topic clickstream_events
```

3. Run stream pipeline:

```bash
python scripts/run_stream_pipeline.py \
  --kafka-bootstrap localhost:9092 \
  --topic clickstream_events \
  --model-path artifacts/models/best_model_pipeline.joblib
```

### Fallback mode (no Kafka)

```bash
python scripts/run_stream_pipeline.py --fallback-file-source data
```

## Outputs and Artifacts

### Metrics and reports

- Validation report JSON/CSV:  
  `artifacts/metrics/validation_report.json`  
  `artifacts/metrics/validation_report.csv`
- Model metrics:  
  `artifacts/metrics/model_metrics.csv`
- Track metrics (dual-track mode):  
  `artifacts/metrics/production_safe/model_metrics.csv`  
  `artifacts/metrics/benchmark_transductive/model_metrics.csv`
- Calibrated/uncalibrated split metrics (per track):  
  `artifacts/metrics/<track>/model_metrics_uncalibrated.csv`  
  `artifacts/metrics/<track>/model_metrics_calibrated.csv`
- Ranking outputs (per track best model):  
  `artifacts/metrics/<track>/topk_metrics_<track>_<model>.csv`
- Streaming metrics:  
  `artifacts/metrics/streaming_metrics.csv`

### Curated and stream data

- Batch curated parquet:  
  `artifacts/curated/batch_curated.parquet`
- Stream outputs:  
  `artifacts/stream/output_parquet/`

### Integrity and provenance

- Audit log JSONL:  
  `artifacts/audit/audit_log.jsonl`
- Verification reports:  
  `artifacts/audit/verification_report.json`  
  `artifacts/audit/verification_report_tampered.json`
- Run metadata/provenance:  
  `artifacts/provenance/run_metadata.json`  
  `artifacts/provenance/winner_style_run_metadata.json`
- Hardening/debug traceability outputs:  
  `artifacts/debug/winner_style_hardening_report.md`  
  `artifacts/debug/leakage_and_causality_checks.json`  
  `artifacts/debug/calibration_summary.json`  
  `artifacts/debug/track_comparison_summary.md`
- LightGBM bagging models (if enabled):  
  `artifacts/models/lightgbm_bagging/production_safe/bag_*.txt`  
  `artifacts/models/lightgbm_bagging/benchmark_transductive/bag_*.txt`

### Figures

Exported plots are saved under:

- `reports/figures/`

## Using Outputs in Report/PPT

### Target Semantics (Final Validated Run)

Supervised target for this implementation is Kaggle `is_attributed` (attribution/conversion proxy), used as a surrogate target for demonstrating veracity/integrity-aware ingestion and classification. It is not a native fraud ground-truth label.

Final validated report values are sourced from `data/train.csv` with a 300,000-row cap and copied from generated artifacts (primarily `artifacts/metrics/final_results_summary.json`).

Use generated artifacts directly for final report and presentation numbers:

- Final consolidated metrics:
  - `artifacts/metrics/final_results_summary.json`
  - `artifacts/metrics/final_results_summary.csv`
- Model and validation tables:
  - `artifacts/metrics/model_metrics.csv`
  - `artifacts/metrics/validation_report.csv`
- Streaming summary:
  - `artifacts/metrics/streaming_metrics.csv`
  - `artifacts/metrics/stream_run_status.md`
- Report writing pack:
  - `reports/report_insert_pack.md`
  - `reports/final_assets_manifest.md`
  - `reports/presentation_slide_mapping.md`

Rerun with larger dataset (when available):

```bash
python scripts/download_dataset.py --output-dir data --prefer-kaggle --force
python scripts/make_sample.py --input-csv data/click_fraud_full.csv --output-csv data/sample_click_fraud.csv --max-rows 200000 --stratify-label
python scripts/run_batch_pipeline.py --input-csv data/sample_click_fraud.csv
```

Expected changes with larger data: model stability typically improves, precision/recall trade-offs can shift, and streaming throughput/latency may change with higher event volume.

### Result provenance

Values in the report/presentation should be copied from generated artifacts, not manually typed from memory.

## Troubleshooting

### Kafka not running / connection refused

- Ensure Docker Desktop is running.
- Start Kafka from `docker/` using `docker-compose -f docker-compose.kafka.yml up -d`.
- Verify `localhost:9092` is reachable.

### Spark not found / Java runtime missing

- Install Java first (`java -version` should work).
- Reinstall dependencies in a clean venv.
- Validate `pyspark` import:

```bash
python -c "from pyspark.sql import SparkSession; print('ok')"
```

### Dataset download failed

- This is expected in restricted environments.
- Use the included sample CSV or run synthetic fallback via `scripts/download_dataset.py`.
- `artifacts/provenance/dataset_download_metadata.json` logs exact failure reasons.

### XGBoost unavailable

- If `xgboost` install fails on your platform, LR and RF still run.
- The notebook/scripts record model availability in output summary files.

## Research References

- `docs/references.md` is generated from provided files only.
- To regenerate when source files are locally available in the repo root:

```bash
python scripts/generate_references.py --xlsx "Reference List BDA IA2.xlsx" --pdf "BDA_GRP12_IA2_LABCA (1).pdf"
```

## Academic Integrity Note

This project is an original implementation for educational use. External datasets, frameworks, and papers must be cited in `docs/references.md` and discussed in `docs/research_alignment.md` without fabricated citations.
