from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def plot_data_quality_summary(
    validation_summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if validation_summary_df.empty:
        return
    plot_df = validation_summary_df.copy()
    plot_df = plot_df.sort_values("invalid_count", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=plot_df, x="rule", y="invalid_count", hue="rule", palette="viridis", legend=False)
    for idx, row in plot_df.iterrows():
        pct = float(row.get("invalid_percent", 0.0))
        ax.text(
            idx,
            float(row["invalid_count"]) + max(1.0, 0.01 * float(plot_df["invalid_count"].max())),
            f"{int(row['invalid_count'])}\\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Data Quality Violations by Rule (Count + % of records)")
    plt.xlabel("Validation Rule")
    plt.ylabel("Invalid Record Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dq_rule_percentage(
    validation_summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if validation_summary_df.empty:
        return
    plot_df = validation_summary_df.copy().sort_values("invalid_percent", ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="rule", y="invalid_percent", hue="rule", palette="crest", legend=False)
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, float(row["invalid_percent"]) + 0.05, f"{float(row['invalid_percent']):.2f}%", ha="center", fontsize=9)
    plt.title("Top Data Quality Rule Failures by Percentage")
    plt.xlabel("Validation Rule")
    plt.ylabel("Invalid Percentage (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fraud_prevalence(
    df: pd.DataFrame,
    out_path: Path,
    target_col: str = "is_attributed",
) -> None:
    if target_col not in df.columns:
        return
    counts = df[target_col].value_counts().sort_index()
    plot_df = pd.DataFrame(
        {
            "label": ["not_attributed" if i == 0 else "attributed" for i in counts.index],
            "count": counts.values,
        }
    )
    if set(plot_df["label"]) != {"not_attributed", "attributed"}:
        for lbl in ["not_attributed", "attributed"]:
            if lbl not in set(plot_df["label"]):
                plot_df = pd.concat([plot_df, pd.DataFrame([{"label": lbl, "count": 0}])], ignore_index=True)
        plot_df = plot_df.sort_values("label").reset_index(drop=True)
    total = float(plot_df["count"].sum())
    plot_df["percent"] = plot_df["count"] / total * 100.0 if total else 0.0
    plot_df["plot_count"] = plot_df["count"].clip(lower=0.8)
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=plot_df, x="label", y="plot_count", hue="label", palette="Set2", legend=False)
    ax.set_yscale("log")
    ymax = float(plot_df["count"].max()) if not plot_df.empty else 1.0
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(
            idx,
            float(max(row["plot_count"], 0.8)) * 1.05,
            f"{int(row['count'])}\\n({float(row['percent']):.3f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Class Prevalence: is_attributed surrogate target (log scale)")
    plt.xlabel("Class")
    plt.ylabel("Count (log scale)")
    plt.ylim(bottom=0.8, top=max(2.0, ymax * 1.4))
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
