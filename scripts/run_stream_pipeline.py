#!/usr/bin/env python3
"""Run Spark Structured Streaming pipeline with Kafka (default) or file-source fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ingestion_trust.core import append_audit_batch, save_json  # noqa: E402
from ingestion_trust.spark_utils import (  # noqa: E402
    apply_schema_casting,
    build_baseline_stats,
    compute_veracity_score,
    create_spark_session,
    feature_engineering,
    get_event_schema,
    run_veracity_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kafka-bootstrap", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="clickstream_events")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "best_model_pipeline.joblib",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stream" / "checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "stream" / "output_parquet",
    )
    parser.add_argument("--trigger-seconds", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--fallback-file-source",
        type=Path,
        default=None,
        help="Optional CSV folder for file-stream fallback if Kafka is unavailable",
    )
    return parser.parse_args()


def _append_stream_metrics(metrics_path: Path, row: Dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if metrics_path.exists():
        df.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        df.to_csv(metrics_path, index=False)


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _plot_stream_latency_throughput(metrics_csv: Path, out_path: Path) -> None:
    if not metrics_csv.exists():
        return

    metrics_df = pd.read_csv(metrics_csv)
    if metrics_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(
        data=metrics_df,
        x="batch_id",
        y="latency_sec",
        marker="o",
        ax=axes[0],
        color="#1f77b4",
        label="Latency (sec)",
    )
    if "latency_sec" in metrics_df and not metrics_df["latency_sec"].empty:
        axes[0].axhline(metrics_df["latency_sec"].mean(), linestyle="--", color="gray", linewidth=1, label="Avg latency")
    axes[0].set_title("Streaming Micro-batch Latency")
    axes[0].set_xlabel("Batch ID")
    axes[0].set_ylabel("Seconds")
    axes[0].legend()

    throughput_col = "processedRowsPerSecond" if "processedRowsPerSecond" in metrics_df.columns else "throughput_rps"
    sns.lineplot(
        data=metrics_df,
        x="batch_id",
        y=throughput_col,
        marker="o",
        ax=axes[1],
        color="#2ca02c",
        label=f"{throughput_col}",
    )
    if throughput_col in metrics_df and not metrics_df[throughput_col].empty:
        axes[1].axhline(
            pd.to_numeric(metrics_df[throughput_col], errors="coerce").mean(),
            linestyle="--",
            color="gray",
            linewidth=1,
            label="Avg throughput",
        )
    axes[1].set_title("Streaming Micro-batch Throughput")
    axes[1].set_xlabel("Batch ID")
    axes[1].set_ylabel("Records / second")
    axes[1].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    audit_dir = PROJECT_ROOT / "artifacts" / "audit"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.model_path.exists():
        model = joblib.load(args.model_path)
    else:
        print(f"Warning: model artifact not found at {args.model_path}; inference columns will be omitted")

    source_mode = "kafka" if args.fallback_file_source is None else "file_fallback"
    spark = create_spark_session(
        app_name="topic17-stream-pipeline",
        include_kafka=(source_mode == "kafka"),
    )

    if source_mode == "kafka":
        stream_df = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", args.kafka_bootstrap)
            .option("subscribe", args.topic)
            .option("startingOffsets", "latest")
            .load()
        )
        parsed_df = (
            stream_df.selectExpr("CAST(value AS STRING) as json_payload")
            .selectExpr("from_json(json_payload, 'click_time STRING, ip STRING, app STRING, device STRING, os STRING, channel STRING, is_fraud STRING, is_attributed STRING, attributed_time STRING') as data")
            .select("data.*")
        )
    else:
        parsed_df = (
            spark.readStream.schema(get_event_schema())
            .option("header", True)
            .csv(str(args.fallback_file_source))
        )

    state: Dict[str, Any] = {
        "processed_batches": 0,
        "baseline_stats": None,
    }

    audit_log = audit_dir / "audit_log.jsonl"
    stream_metrics_csv = metrics_dir / "streaming_metrics.csv"
    stream_metrics_alias_csv = metrics_dir / "stream_metrics.csv"
    stream_progress_csv = metrics_dir / "stream_progress_metrics.csv"
    stream_metrics_plot = PROJECT_ROOT / "reports" / "figures" / "streaming_latency_throughput.png"
    for p in [stream_metrics_csv, stream_metrics_alias_csv, stream_progress_csv]:
        if p.exists():
            p.unlink()

    def process_batch(microbatch_df, batch_id: int) -> None:
        start = time.perf_counter()
        count = microbatch_df.count()
        if count == 0:
            return

        cast_df = apply_schema_casting(microbatch_df)
        feat_df = feature_engineering(cast_df)

        if state["baseline_stats"] is None:
            state["baseline_stats"] = build_baseline_stats(feat_df)

        valid_df, _summary = run_veracity_checks(feat_df, baseline_stats=state["baseline_stats"])
        scored_df = compute_veracity_score(valid_df)

        output_pd = scored_df.toPandas()
        if model is not None and not output_pd.empty:
            try:
                pred_prob = model.predict_proba(output_pd)[:, 1]
                output_pd["pred_fraud_proba"] = pred_prob
                output_pd["pred_is_fraud"] = (pred_prob >= 0.5).astype(int)
            except Exception as exc:
                output_pd["pred_error"] = str(exc)

        batch_out_path = args.output_dir / f"batch_{batch_id}.parquet"
        output_pd.to_parquet(batch_out_path, index=False)

        records = output_pd.head(10000).to_dict(orient="records")
        append_audit_batch(
            log_path=audit_log,
            batch_id=f"stream_{batch_id}",
            records=records,
            batch_metadata={
                "source_mode": source_mode,
                "topic": args.topic,
                "records_sampled_for_audit": len(records),
                "microbatch_records": int(count),
            },
        )

        latency_sec = time.perf_counter() - start
        throughput_rps = float(count) / latency_sec if latency_sec > 0 else float("inf")

        metric_row = {
            "batch_id": int(batch_id),
            "records": int(count),
            "latency_sec": float(latency_sec),
            "throughput_rps": float(throughput_rps),
            "source_mode": source_mode,
        }
        _append_stream_metrics(stream_metrics_csv, metric_row)
        _append_stream_metrics(stream_metrics_alias_csv, metric_row)
        state["processed_batches"] += 1

    query = (
        parsed_df.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", str(args.checkpoint_dir))
        .trigger(processingTime=f"{args.trigger_seconds} seconds")
        .start()
    )

    print(f"Streaming query started in {source_mode} mode")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")

    idle_loops = 0
    progress_rows: List[Dict[str, Any]] = []
    seen_progress_batches = set()
    last_seen_progress_batch: Optional[int] = None
    try:
        while query.isActive:
            time.sleep(max(2, args.trigger_seconds))
            if state["processed_batches"] >= args.max_batches:
                query.stop()
                break
            progress = query.lastProgress
            if progress:
                batch_id = progress.get("batchId")
                if batch_id is not None:
                    batch_id = int(batch_id)
                is_new_progress = batch_id is not None and batch_id != last_seen_progress_batch
                if is_new_progress:
                    idle_loops = 0
                    last_seen_progress_batch = batch_id
                else:
                    idle_loops += 1
                if batch_id is not None and batch_id not in seen_progress_batches:
                    seen_progress_batches.add(int(batch_id))
                    duration_ms = progress.get("durationMs", {}) or {}
                    row = {
                        "batch_id": int(batch_id),
                        "timestamp": progress.get("timestamp"),
                        "numInputRows": int(progress.get("numInputRows", 0) or 0),
                        "inputRowsPerSecond": float(progress.get("inputRowsPerSecond", 0.0) or 0.0),
                        "processedRowsPerSecond": float(progress.get("processedRowsPerSecond", 0.0) or 0.0),
                        "triggerExecution_ms": float(duration_ms.get("triggerExecution", 0.0) or 0.0),
                        "getBatch_ms": float(duration_ms.get("getBatch", 0.0) or 0.0),
                        "addBatch_ms": float(duration_ms.get("addBatch", 0.0) or 0.0),
                        "queryPlanning_ms": float(duration_ms.get("queryPlanning", 0.0) or 0.0),
                        "walCommit_ms": float(duration_ms.get("walCommit", 0.0) or 0.0),
                        "source_mode": source_mode,
                    }
                    progress_rows.append(row)
                    _append_stream_metrics(stream_progress_csv, row)
            else:
                idle_loops += 1
            if idle_loops >= 15:
                # Avoid infinite wait when no input is arriving.
                query.stop()
                break
    finally:
        # Merge foreachBatch timings with Spark progress metrics (if any).
        if stream_metrics_csv.exists():
            wall_df = pd.read_csv(stream_metrics_csv)
        else:
            wall_df = pd.DataFrame(columns=["batch_id", "records", "latency_sec", "throughput_rps", "source_mode"])
        if progress_rows:
            prog_df = pd.DataFrame(progress_rows).drop_duplicates(subset=["batch_id"], keep="last")
            merged_df = wall_df.merge(prog_df, on=["batch_id", "source_mode"], how="left")
        else:
            merged_df = wall_df.copy()
        merged_df = merged_df.sort_values("batch_id").reset_index(drop=True)
        if not merged_df.empty:
            merged_df.to_csv(stream_metrics_csv, index=False)
            merged_df.to_csv(stream_metrics_alias_csv, index=False)

        _plot_stream_latency_throughput(stream_metrics_csv, stream_metrics_plot)
        avg_latency = float(pd.to_numeric(merged_df.get("latency_sec", pd.Series(dtype=float)), errors="coerce").mean()) if not merged_df.empty else 0.0
        avg_throughput = float(pd.to_numeric(merged_df.get("throughput_rps", pd.Series(dtype=float)), errors="coerce").mean()) if not merged_df.empty else 0.0
        avg_processed_rps = float(pd.to_numeric(merged_df.get("processedRowsPerSecond", pd.Series(dtype=float)), errors="coerce").mean()) if not merged_df.empty else 0.0
        save_json(
            {
                "processed_batches": state["processed_batches"],
                "stream_metrics_csv": _to_project_relative(stream_metrics_csv),
                "stream_metrics_alias_csv": _to_project_relative(stream_metrics_alias_csv),
                "stream_progress_csv": _to_project_relative(stream_progress_csv),
                "stream_metrics_plot": _to_project_relative(stream_metrics_plot),
                "output_dir": _to_project_relative(args.output_dir),
                "checkpoint_dir": _to_project_relative(args.checkpoint_dir),
                "source_mode": source_mode,
                "average_latency_sec": avg_latency,
                "average_throughput_rps_wall": avg_throughput,
                "average_processedRowsPerSecond": avg_processed_rps,
            },
            metrics_dir / "stream_pipeline_summary.json",
        )
        spark.stop()

    print("Streaming pipeline finished")


if __name__ == "__main__":
    main()
