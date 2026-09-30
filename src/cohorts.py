"""
M4b — Cohort retention analysis.

Builds cohort retention curves by first-purchase month, and compares
retention for customers whose first purchase was online vs in store.

Usage:
    python src/cohorts.py              # full dataset
    python src/cohorts.py --sample     # sample mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from plotting import apply_style, save_fig, COLOURS, FIGURES_DIR

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def run_cohorts(sample: bool = False) -> dict:
    """Build cohort retention curves."""

    con = duckdb.connect()
    txn = str(DATA_DIR / "transactions.parquet")

    # ── 1. First purchase date per customer ─────────────────────
    cohort_data = con.execute(f"""
        WITH first_purchase AS (
            SELECT
                customer_id,
                min(t_dat)                                          AS first_date,
                date_trunc('month', min(t_dat))::DATE               AS cohort_month,
                -- first purchase channel
                FIRST(sales_channel_id ORDER BY t_dat, sales_channel_id) AS first_channel
            FROM '{txn}'
            GROUP BY customer_id
        ),
        monthly_activity AS (
            SELECT
                customer_id,
                date_trunc('month', t_dat)::DATE AS activity_month
            FROM '{txn}'
            GROUP BY 1, 2
        )
        SELECT
            fp.cohort_month,
            fp.first_channel,
            ma.activity_month,
            datediff('month', fp.cohort_month, ma.activity_month) AS months_since,
            count(DISTINCT fp.customer_id) AS n_customers
        FROM first_purchase fp
        JOIN monthly_activity ma ON fp.customer_id = ma.customer_id
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 4
    """).fetchdf()

    # ── 2. Build retention matrix ───────────────────────────────
    # Total customers per cohort
    cohort_sizes = con.execute(f"""
        SELECT
            date_trunc('month', min(t_dat))::DATE AS cohort_month,
            count(DISTINCT customer_id) AS cohort_size
        FROM '{txn}'
        GROUP BY customer_id
    """).fetchdf()
    cohort_sizes = cohort_sizes.groupby("cohort_month")["cohort_size"].sum().reset_index()

    # Overall retention
    overall = cohort_data.groupby(["cohort_month", "months_since"])["n_customers"].sum().reset_index()
    overall = overall.merge(cohort_sizes, on="cohort_month")
    overall["retention_pct"] = (overall["n_customers"] / overall["cohort_size"] * 100).round(1)

    # Pivot for heatmap (limit to first 12 months for readability)
    pivot = overall[overall["months_since"] <= 12].pivot(
        index="cohort_month", columns="months_since", values="retention_pct"
    )
    # Only keep cohorts with at least 6 months of data
    pivot = pivot.dropna(thresh=6)

    # ── 3. Retention heatmap ────────────────────────────────────
    if not pivot.empty:
        # Average retention at month 3 and month 6
        avg_m3 = pivot[3].mean() if 3 in pivot.columns else None
        avg_m6 = pivot[6].mean() if 6 in pivot.columns else None

        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
        sns.heatmap(
            pivot,
            annot=True, fmt=".0f", cmap="YlOrRd_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Retention %"},
            vmin=0, vmax=100,
        )
        ax.set_xlabel("Months since first purchase")
        ax.set_ylabel("Cohort (first purchase month)")
        # Format y labels
        ax.set_yticklabels([str(d)[:7] for d in pivot.index], rotation=0, fontsize=7)

        title_parts = []
        if avg_m3:
            title_parts.append(f"3-month retention averages {avg_m3:.0f}%")
        if avg_m6:
            title_parts.append(f"6-month averages {avg_m6:.0f}%")
        ax.set_title(
            " and ".join(title_parts) if title_parts else "Cohort retention heatmap",
            fontsize=11,
        )
        fig.tight_layout()
        save_fig(fig, "cohort_retention_heatmap")
    else:
        avg_m3, avg_m6 = None, None

    # ── 4. Online vs store first-purchase retention ─────────────
    channel_retention = (
        cohort_data.groupby(["first_channel", "months_since"])["n_customers"]
        .sum()
        .reset_index()
    )
    channel_sizes = con.execute(f"""
        SELECT
            FIRST(sales_channel_id ORDER BY t_dat, sales_channel_id) AS first_channel,
            count(DISTINCT customer_id) AS cohort_size
        FROM '{txn}'
        GROUP BY customer_id
    """).fetchdf()
    channel_sizes = channel_sizes.groupby("first_channel")["cohort_size"].sum().reset_index()

    channel_retention = channel_retention.merge(channel_sizes, on="first_channel")
    channel_retention["retention_pct"] = (
        channel_retention["n_customers"] / channel_retention["cohort_size"] * 100
    ).round(1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ch in sorted(channel_retention["first_channel"].unique()):
        ch_data = channel_retention[
            (channel_retention["first_channel"] == ch) &
            (channel_retention["months_since"] <= 12) &
            (channel_retention["months_since"] > 0)
        ]
        label = f"Channel {int(ch)} first"
        colour = COLOURS["primary"] if ch == 1 else COLOURS["accent"]
        ax.plot(ch_data["months_since"], ch_data["retention_pct"],
                marker="o", markersize=4, linewidth=1.8, label=label, color=colour)

    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Retention (%)")
    ax.legend()

    # Compare at month 6
    ch1_m6 = channel_retention[
        (channel_retention["first_channel"] == 1) & (channel_retention["months_since"] == 6)
    ]
    ch2_m6 = channel_retention[
        (channel_retention["first_channel"] == 2) & (channel_retention["months_since"] == 6)
    ]
    if not ch1_m6.empty and not ch2_m6.empty:
        r1 = ch1_m6.iloc[0]["retention_pct"]
        r2 = ch2_m6.iloc[0]["retention_pct"]
        diff = abs(r1 - r2)
        better = "Channel 1" if r1 > r2 else "Channel 2"
        ax.set_title(
            f"{better} first-purchasers retain {diff:.0f}pp better at 6 months "
            f"({r1:.0f}% vs {r2:.0f}%)",
            fontsize=11,
        )
    else:
        ax.set_title("Retention by first-purchase channel", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "cohort_retention_by_channel")

    # ── 5. Save results ─────────────────────────────────────────
    results = {
        "avg_retention_month_3": round(float(avg_m3), 1) if avg_m3 else None,
        "avg_retention_month_6": round(float(avg_m6), 1) if avg_m6 else None,
        "n_cohorts": len(pivot) if not pivot.empty else 0,
        "channel_retention_month_6": {},
    }
    for ch in sorted(channel_retention["first_channel"].unique()):
        ch_m6 = channel_retention[
            (channel_retention["first_channel"] == ch) & (channel_retention["months_since"] == 6)
        ]
        if not ch_m6.empty:
            results["channel_retention_month_6"][f"channel_{int(ch)}"] = float(ch_m6.iloc[0]["retention_pct"])

    results_path = RESULTS_DIR / "cohorts.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → {results_path}")

    # Save retention table
    pivot_save = pivot.copy()
    pivot_save.index = [str(d)[:7] for d in pivot_save.index]
    pivot_save.to_csv(TABLES_DIR / "cohort_retention.csv")
    print(f"  → {TABLES_DIR / 'cohort_retention.csv'}")

    con.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohort retention")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M4b: Cohorts ══")
    results = run_cohorts(sample=args.sample)

    if results["avg_retention_month_6"]:
        print(f"\n  Average 6-month retention: {results['avg_retention_month_6']}%")
    print("══ Done ══\n")
