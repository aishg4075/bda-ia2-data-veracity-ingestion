#!/usr/bin/env python3
"""Simple Kafka consumer to inspect clickstream messages."""

from __future__ import annotations

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="clickstream_events")
    parser.add_argument("--group-id", type=str, default="topic17-debug-consumer")
    parser.add_argument("--max-messages", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from kafka import KafkaConsumer
    except Exception as exc:
        raise RuntimeError("kafka-python is required. Run: pip install -r requirements.txt") from exc

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=args.timeout_ms,
    )

    count = 0
    for msg in consumer:
        print(json.dumps(msg.value, indent=2))
        count += 1
        if count >= args.max_messages:
            break

    consumer.close()
    print(f"Consumed {count} message(s) from topic '{args.topic}'")


if __name__ == "__main__":
    main()
