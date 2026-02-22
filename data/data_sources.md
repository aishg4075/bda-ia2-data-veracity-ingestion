# Data Sources and Fallback Notes

## Preferred source

1. **Kaggle**: TalkingData AdTracking Fraud Detection Challenge  
   URL: <https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection>

Notes:
- Requires Kaggle account + accepted competition rules.
- Requires `~/.kaggle/kaggle.json` and Kaggle CLI.
- The dataset label (`is_attributed`) is transformed into `is_fraud = 1 - is_attributed` as a proxy for invalid/non-converting traffic in this implementation.

## Open fallback source candidates

The downloader attempts open CSV URLs (if provided via `--open-url`).

Default candidate included in code:
- <https://raw.githubusercontent.com/yzhao062/FraudDetection/master/data/creditcard.csv>

If an open dataset does not match required clickstream schema, the pipeline normalizes where possible; otherwise it falls back to synthetic data.

## Guaranteed fallback (always available)

When external downloads fail, `scripts/download_dataset.py` generates a synthetic clickstream dataset with the required schema:

- `click_time`
- `ip`
- `app`
- `device`
- `os`
- `channel`
- `is_fraud`
- optional: `is_attributed`, `attributed_time`

Synthetic mode is explicitly labeled as `fallback_synthetic` in provenance metadata and notebook outputs.

## Commands

```bash
cd bda-topic17-ingestion-trust
python scripts/download_dataset.py --output-dir data --prefer-kaggle
python scripts/make_sample.py --input-csv data/click_fraud_full.csv --output-csv data/sample_click_fraud.csv
```
