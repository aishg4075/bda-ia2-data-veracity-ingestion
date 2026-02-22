# Research Alignment: Topic 17

This mapping links implemented modules to literature themes described in the IA2/LabCA report.
Reference numbers below point to `docs/references.md`.

| Implemented Component | What was implemented in this repo | Expected supporting literature theme | Reference mapping status |
|---|---|---|---|
| Declarative ingestion validation rules | Schema checks, completeness, range checks, uniqueness, drift snapshot, outlier checks in Spark | Declarative data quality constraints and automated DQ verification | [3], [6], [8], [12] |
| Incremental drift checks | Batch-by-batch PSI/KL drift metrics against baseline stats | Differential or incremental data quality verification and drift mitigation | [12], [15] |
| Unified batch + stream processing | Shared validation and feature logic in Spark batch and Structured Streaming | Unified processing model for batch and streaming | [4], [5], [13] |
| Tamper-evident logging | Record hashes, Merkle root, chained batch hash, JSONL append-only log, verifier | Verifiable ledgers / tamper-evident audit trails | [25], [26], [27], [29], [30] |
| Provenance metadata | Dataset source, dataset hash, run metadata, env capture, git commit hash | Data provenance and lineage trust frameworks | [18], [19], [31], [32], [33], [34] |
| Click-fraud detection modeling | Logistic Regression, Random Forest, XGBoost, imbalance-aware metrics | Mobile/web ad fraud and invalid traffic detection methods | [20], [21], [22], [23], [24] |
| Integrity overhead measurement | Throughput/latency comparison with and without integrity hashing | Practical overhead of trustworthy logging in distributed systems | [10], [11], [29], [30] |
| Stream framework comparison | Notebook discussion compares implemented Kafka/Spark streaming with batch Spark and references Storm conceptually | Comparative stream processing architecture and cleaning constraints | [4], [5], [13] |

## Notes

1. The current mapping is derived from references extracted from the provided PDF source.
2. If Excel references become accessible, regenerate:

```bash
python scripts/generate_references.py --xlsx "Reference List BDA IA2.xlsx" --pdf "BDA_GRP12_IA2_LABCA (1).pdf"
```
3. Keep citations source-backed only; do not add inferred or fabricated references.
