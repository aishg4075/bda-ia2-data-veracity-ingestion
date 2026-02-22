from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


def create_spark_session(
    app_name: str,
    master: str = "local[*]",
    include_kafka: bool = False,
    shuffle_partitions: int = 8,
) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).master(master)
    builder = builder.config("spark.sql.session.timeZone", "UTC")
    builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))

    if include_kafka:
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def get_event_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("click_time", T.StringType(), True),
            T.StructField("ip", T.StringType(), True),
            T.StructField("app", T.StringType(), True),
            T.StructField("device", T.StringType(), True),
            T.StructField("os", T.StringType(), True),
            T.StructField("channel", T.StringType(), True),
            T.StructField("is_fraud", T.StringType(), True),
            T.StructField("is_attributed", T.StringType(), True),
            T.StructField("attributed_time", T.StringType(), True),
        ]
    )


def apply_schema_casting(df: DataFrame) -> DataFrame:
    required_cols = [
        "click_time",
        "ip",
        "app",
        "device",
        "os",
        "channel",
        "is_fraud",
        "is_attributed",
        "attributed_time",
    ]

    out = df
    for col in required_cols:
        if col not in out.columns:
            out = out.withColumn(col, F.lit(None).cast("string"))

    out = (
        out.withColumn("click_time_ts", F.to_timestamp("click_time"))
        .withColumn("attributed_time_ts", F.to_timestamp("attributed_time"))
        .withColumn("ip", F.col("ip").cast("long"))
        .withColumn("app", F.col("app").cast("int"))
        .withColumn("device", F.col("device").cast("int"))
        .withColumn("os", F.col("os").cast("int"))
        .withColumn("channel", F.col("channel").cast("int"))
        .withColumn("is_fraud", F.col("is_fraud").cast("int"))
        .withColumn("is_attributed", F.col("is_attributed").cast("int"))
    )

    out = out.withColumn(
        "is_fraud",
        F.when(F.col("is_fraud").isNull() & F.col("is_attributed").isNotNull(), 1 - F.col("is_attributed")).otherwise(
            F.col("is_fraud")
        ),
    )

    return out


def feature_engineering(df: DataFrame) -> DataFrame:
    out = df
    out = (
        out.withColumn("click_hour", F.hour("click_time_ts"))
        .withColumn("click_day", F.dayofmonth("click_time_ts"))
        .withColumn("click_weekday", F.dayofweek("click_time_ts"))
        .withColumn("click_minute", F.minute("click_time_ts"))
        .withColumn("event_hour", F.date_trunc("hour", "click_time_ts"))
    )

    window_ip_hour = Window.partitionBy("ip", "event_hour")
    window_ip = Window.partitionBy("ip")
    window_ip_order = Window.partitionBy("ip").orderBy(F.col("click_time_ts").asc())

    out = out.withColumn("clicks_per_ip_per_hour", F.count(F.lit(1)).over(window_ip_hour))
    out = out.withColumn("clicks_per_ip", F.count(F.lit(1)).over(window_ip))
    out = out.withColumn("prev_click_time", F.lag("click_time_ts").over(window_ip_order))
    out = out.withColumn(
        "inter_click_seconds",
        F.when(
            F.col("prev_click_time").isNull(),
            F.lit(0.0),
        ).otherwise(F.col("click_time_ts").cast("long") - F.col("prev_click_time").cast("long")),
    )

    channel_per_ip = df.groupBy("ip").agg(F.countDistinct("channel").alias("unique_channels_per_ip"))
    out = out.join(channel_per_ip, on="ip", how="left")

    app_channel = df.groupBy("app", "channel").agg(F.count(F.lit(1)).alias("app_channel_freq"))
    out = out.join(app_channel, on=["app", "channel"], how="left")

    return out


def build_baseline_stats(
    df: DataFrame,
    cols: Sequence[str] = ("app", "channel", "device", "os"),
    top_k: int = 32,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    total = df.count()
    if total == 0:
        return {c: {} for c in cols}

    for col in cols:
        rows = (
            df.groupBy(col)
            .count()
            .orderBy(F.desc("count"))
            .limit(top_k)
            .collect()
        )
        dist: Dict[str, float] = {}
        for r in rows:
            key = "NULL" if r[col] is None else str(r[col])
            dist[key] = float(r["count"]) / float(total)
        stats[col] = dist
    return stats


def compute_drift_metrics(
    current_df: DataFrame,
    baseline_stats: Mapping[str, Mapping[str, float]],
    cols: Sequence[str] = ("app", "channel", "device", "os"),
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    eps = 1e-6

    cur_total = current_df.count()
    if cur_total == 0:
        return {c: {"psi": 0.0, "kl": 0.0} for c in cols}

    for col in cols:
        cur_rows = current_df.groupBy(col).count().collect()
        cur_dist: Dict[str, float] = {}
        for r in cur_rows:
            key = "NULL" if r[col] is None else str(r[col])
            cur_dist[key] = float(r["count"]) / float(cur_total)

        base_dist = dict(baseline_stats.get(col, {}))
        keys = set(cur_dist.keys()).union(base_dist.keys())

        psi_val = 0.0
        kl_val = 0.0
        for k in keys:
            p = max(base_dist.get(k, eps), eps)
            q = max(cur_dist.get(k, eps), eps)
            psi_val += (q - p) * math.log(q / p)
            kl_val += q * math.log(q / p)

        metrics[col] = {"psi": float(psi_val), "kl": float(kl_val)}

    return metrics


def run_veracity_checks(
    df: DataFrame,
    baseline_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Tuple[DataFrame, Dict[str, Any]]:
    now_ts = datetime.now(timezone.utc)
    lower_bound = datetime(2000, 1, 1, tzinfo=timezone.utc)
    upper_bound = now_ts + timedelta(days=2)

    out = df

    out = out.withColumn(
        "rule_schema_valid",
        (
            F.col("click_time_ts").isNotNull()
            & F.col("ip").isNotNull()
            & F.col("app").isNotNull()
            & F.col("device").isNotNull()
            & F.col("os").isNotNull()
            & F.col("channel").isNotNull()
            & F.col("is_fraud").isin(0, 1)
        ),
    )

    out = out.withColumn(
        "rule_completeness",
        (
            F.col("click_time_ts").isNotNull()
            & F.col("ip").isNotNull()
            & F.col("app").isNotNull()
            & F.col("device").isNotNull()
            & F.col("os").isNotNull()
            & F.col("channel").isNotNull()
            & F.col("is_fraud").isNotNull()
        ),
    )

    out = out.withColumn(
        "rule_range",
        (
            F.col("click_time_ts").between(F.lit(lower_bound), F.lit(upper_bound))
            & (F.col("ip") >= 0)
            & (F.col("app") >= 0)
            & (F.col("device") >= 0)
            & (F.col("os") >= 0)
            & (F.col("channel") >= 0)
        ),
    )

    dedup_w = Window.partitionBy("ip", "app", "device", "os", "channel", "click_time_ts").orderBy(
        F.col("click_time_ts").asc_nulls_last()
    )
    out = out.withColumn("_dup_rank", F.row_number().over(dedup_w))
    out = out.withColumn("rule_uniqueness", F.col("_dup_rank") == 1)

    if baseline_stats:
        for c in ["app", "channel", "device", "os"]:
            allowed = list(baseline_stats.get(c, {}).keys())
            allowed_clean = [None if x == "NULL" else int(x) for x in allowed if x != "NULL"]
            col_rule = F.col(c).isin([x for x in allowed_clean if x is not None])
            if "NULL" in allowed:
                col_rule = col_rule | F.col(c).isNull()
            out = out.withColumn(f"_drift_{c}", col_rule)
        out = out.withColumn(
            "rule_drift",
            F.col("_drift_app") & F.col("_drift_channel") & F.col("_drift_device") & F.col("_drift_os"),
        )
    else:
        out = out.withColumn("rule_drift", F.lit(True))

    q1, q3 = out.approxQuantile("clicks_per_ip_per_hour", [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr
    out = out.withColumn(
        "rule_outlier",
        F.col("clicks_per_ip_per_hour").between(F.lit(lower_iqr), F.lit(upper_iqr)),
    )

    out = out.withColumn(
        "is_valid_record",
        F.col("rule_schema_valid")
        & F.col("rule_completeness")
        & F.col("rule_range")
        & F.col("rule_uniqueness")
        & F.col("rule_drift")
        & F.col("rule_outlier"),
    )

    total_count = out.count()
    dup_count = out.filter(~F.col("rule_uniqueness")).count()

    rule_cols = [
        "rule_schema_valid",
        "rule_completeness",
        "rule_range",
        "rule_uniqueness",
        "rule_drift",
        "rule_outlier",
    ]
    invalid_counts: Dict[str, int] = {}
    for rc in rule_cols:
        invalid_counts[rc] = out.filter(~F.col(rc)).count()

    baseline_for_drift = baseline_stats if baseline_stats else build_baseline_stats(out)
    drift_metrics = compute_drift_metrics(out, baseline_for_drift)

    summary = {
        "total_records": total_count,
        "duplicates_removed": dup_count,
        "invalid_counts_by_rule": invalid_counts,
        "invalid_percent_by_rule": {
            k: (float(v) / total_count * 100.0 if total_count else 0.0) for k, v in invalid_counts.items()
        },
        "drift_metrics": drift_metrics,
        "outlier_bounds": {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_iqr,
            "upper_bound": upper_iqr,
        },
    }

    drop_cols = [c for c in out.columns if c.startswith("_drift_") or c == "_dup_rank"]
    out = out.drop(*drop_cols)
    return out, summary


def compute_veracity_score(
    df_flags: DataFrame,
    weights: Optional[Mapping[str, float]] = None,
) -> DataFrame:
    rule_cols = [
        "rule_schema_valid",
        "rule_completeness",
        "rule_range",
        "rule_uniqueness",
        "rule_drift",
        "rule_outlier",
    ]
    w = {
        "rule_schema_valid": 0.24,
        "rule_completeness": 0.20,
        "rule_range": 0.18,
        "rule_uniqueness": 0.15,
        "rule_drift": 0.13,
        "rule_outlier": 0.10,
    }
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    total_w = sum(w.values())
    expr = None
    for c in rule_cols:
        term = F.when(F.col(c), F.lit(w[c])).otherwise(F.lit(0.0))
        expr = term if expr is None else (expr + term)

    assert expr is not None
    return df_flags.withColumn("veracity_score", (expr / F.lit(total_w)).cast("double"))
