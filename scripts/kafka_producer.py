#!/usr/bin/env python3
"""Publish clickstream events from CSV into Kafka as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_click_fraud.csv",
    )
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="clickstream_events")
    parser.add_argument("--delay-ms", type=int, default=5, help="Delay between events")
    parser.add_argument("--max-events", type=int, default=0, help="0 means all rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    try:
        from kafka import KafkaProducer
    except Exception as exc:
        raise RuntimeError(
            "kafka-python is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    df = pd.read_csv(args.input_csv)
    if args.max_events and args.max_events > 0:
        df = df.head(args.max_events)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        retries=3,
    )

    sent = 0
    t0 = time.perf_counter()
    for rec in df.to_dict(orient="records"):
        payload: Dict[str, object] = {
            "click_time": rec.get("click_time"),
            "ip": rec.get("ip"),
            "app": rec.get("app"),
            "device": rec.get("device"),
            "os": rec.get("os"),
            "channel": rec.get("channel"),
            "is_fraud": rec.get("is_fraud"),
            "is_attributed": rec.get("is_attributed"),
            "attributed_time": rec.get("attributed_time"),
        }
        producer.send(args.topic, payload)
        sent += 1
        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)

    producer.flush()
    elapsed = time.perf_counter() - t0
    rate = sent / elapsed if elapsed > 0 else float("inf")

    print(
        json.dumps(
            {
                "topic": args.topic,
                "bootstrap_servers": args.bootstrap_servers,
                "events_sent": sent,
                "elapsed_sec": elapsed,
                "throughput_eps": rate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
