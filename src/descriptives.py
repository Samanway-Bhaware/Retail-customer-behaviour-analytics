"""
M3 — Descriptive analytics and charts.

Produces:
  1. Weekly revenue index and transaction trend
  2. Online vs store split over time
  3. Product-group mix (revenue share)
  4. Revenue concentration (top 10/20 % + Lorenz curve)
  5. Seasonality (monthly pattern)

Every chart title states the finding, computed from data.

Usage:
    python src/descriptives.py              # full dataset
    python src/descriptives.py --sample     # sample mode
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
import matplotlib.ticker as mticker

from plotting import apply_style, save_fig, COLOURS, PALETTE_SEQ, FIGURES_DIR

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def run_descriptives(sample: bool = False) -> dict:
    """Generate descriptive analytics and charts."""

    con = duckdb.connect()
    txn = str(DATA_DIR / "transactions.parquet")
    art = str(DATA_DIR / "articles.parquet")

    results = {}

    # ══════════════════════════════════════════════════════════════
    # 1. Weekly revenue index + transaction trend
    # ══════════════════════════════════════════════════════════════
    weekly = con.execute(f"""
        SELECT
            date_trunc('week', t_dat)::DATE         AS week,
            sum(price)                               AS revenue,
            count(*)                                 AS n_txns,
            count(DISTINCT customer_id)              AS n_customers
        FROM '{txn}'
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    # Index revenue to first full week = 100
    base_rev = weekly["revenue"].iloc[1]  # second week (first may be partial)
    weekly["rev_index"] = (weekly["revenue"] / base_rev * 100).round(1)

    results["weekly_trend"] = {
        "n_weeks": len(weekly),
        "peak_week": str(weekly.loc[weekly["rev_index"].idxmax(), "week"]),
        "peak_index": float(weekly["rev_index"].max()),
        "trough_week": str(weekly.loc[weekly["rev_index"].idxmin(), "week"]),
        "trough_index": float(weekly["rev_index"].min()),
    }

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.fill_between(weekly["week"], weekly["rev_index"], alpha=0.3, color=COLOURS["primary"])
    ax1.plot(weekly["week"], weekly["rev_index"], color=COLOURS["primary"], linewidth=1.2, label="Revenue index")
    ax1.set_ylabel("Revenue index (week 2 = 100)")
    ax1.set_xlabel("")
    peak_idx = weekly["rev_index"].max()
    ax1.set_title(
        f"Revenue peaks at {peak_idx:.0f}× the baseline week, "
        f"with strong seasonal spikes",
        fontsize=11
    )
    ax1.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, "weekly_revenue_trend")

    # ══════════════════════════════════════════════════════════════
    # 2. Online vs store split over time
    # ══════════════════════════════════════════════════════════════
    channel_weekly = con.execute(f"""
        SELECT
            date_trunc('week', t_dat)::DATE      AS week,
            sales_channel_id,
            sum(price) AS revenue,
            count(*)   AS n_txns
        FROM '{txn}'
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchdf()

    total_by_channel = con.execute(f"""
        SELECT sales_channel_id,
               sum(price) AS revenue,
               round(sum(price) * 100.0 / (SELECT sum(price) FROM '{txn}'), 1) AS pct
        FROM '{txn}'
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    results["channel_split"] = {
        f"channel_{int(r['sales_channel_id'])}": {
            "revenue_pct": float(r["pct"]),
        }
        for _, r in total_by_channel.iterrows()
    }

    # Pivot for stacked area
    pivot = channel_weekly.pivot(index="week", columns="sales_channel_id", values="revenue").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    ch1_pct = results["channel_split"].get("channel_1", {}).get("revenue_pct", 0)
    ch2_pct = results["channel_split"].get("channel_2", {}).get("revenue_pct", 0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        pivot_pct.index,
        [pivot_pct[c] for c in pivot_pct.columns],
        labels=[f"Channel {int(c)}" for c in pivot_pct.columns],
        colors=[COLOURS["primary"], COLOURS["accent"]],
        alpha=0.8,
    )
    ax.set_ylabel("Revenue share (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Channel 1 accounts for {ch1_pct:.0f}% of revenue overall, "
        f"channel 2 for {ch2_pct:.0f}%",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "channel_split_over_time")

    # ══════════════════════════════════════════════════════════════
    # 3. Product-group revenue mix
    # ══════════════════════════════════════════════════════════════
    product_mix = con.execute(f"""
        SELECT
            a.product_group_name,
            sum(t.price) AS revenue,
            round(sum(t.price) * 100.0 / (SELECT sum(price) FROM '{txn}'), 1) AS pct
        FROM '{txn}' t
        LEFT JOIN '{art}' a ON t.article_id = a.article_id
        GROUP BY 1
        ORDER BY revenue DESC
    """).fetchdf()

    results["product_group_mix"] = {
        r["product_group_name"]: float(r["pct"])
        for _, r in product_mix.iterrows()
    }

    top_group = product_mix.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        product_mix["product_group_name"][::-1],
        product_mix["pct"][::-1],
        color=COLOURS["primary"],
        alpha=0.85,
    )
    ax.set_xlabel("Share of revenue (%)")
    ax.set_title(
        f"\"{top_group['product_group_name']}\" dominates at "
        f"{top_group['pct']:.0f}% of revenue",
        fontsize=11,
    )
    for bar, pct in zip(bars, product_mix["pct"][::-1]):
        if pct >= 2:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8, color=COLOURS["dark"])
    fig.tight_layout()
    save_fig(fig, "product_group_mix")

    # ══════════════════════════════════════════════════════════════
    # 4. Revenue concentration (Lorenz curve)
    # ══════════════════════════════════════════════════════════════
    cust_rev = con.execute(f"""
        SELECT customer_id, sum(price) AS total_rev
        FROM '{txn}'
        GROUP BY customer_id
        ORDER BY total_rev
    """).fetchdf()

    rev_sorted = cust_rev["total_rev"].values
    cum_rev = np.cumsum(rev_sorted) / rev_sorted.sum()
    cum_cust = np.arange(1, len(rev_sorted) + 1) / len(rev_sorted)

    # Top 10% and 20% revenue share
    top_10_share = float(1 - cum_rev[int(len(cum_rev) * 0.9)])
    top_20_share = float(1 - cum_rev[int(len(cum_rev) * 0.8)])
    # Gini coefficient
    gini = float(1 - 2 * np.trapz(cum_rev, cum_cust))

    results["revenue_concentration"] = {
        "top_10_pct_share": round(top_10_share * 100, 1),
        "top_20_pct_share": round(top_20_share * 100, 1),
        "gini": round(gini, 3),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cum_cust * 100, cum_rev * 100, color=COLOURS["primary"], linewidth=2, label="Actual")
    ax.plot([0, 100], [0, 100], "--", color=COLOURS["muted"], linewidth=1, label="Perfect equality")
    ax.fill_between(cum_cust * 100, cum_rev * 100, cum_cust * 100, alpha=0.15, color=COLOURS["primary"])
    ax.set_xlabel("Cumulative % of customers (ascending spend)")
    ax.set_ylabel("Cumulative % of revenue")
    ax.set_title(
        f"The top 20% of customers generate {top_20_share*100:.0f}% of revenue\n"
        f"(top 10% generate {top_10_share*100:.0f}%; Gini = {gini:.2f})",
        fontsize=11,
    )
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save_fig(fig, "revenue_concentration_lorenz")

    # ══════════════════════════════════════════════════════════════
    # 5. Seasonality — monthly revenue pattern
    # ══════════════════════════════════════════════════════════════
    monthly = con.execute(f"""
        SELECT
            date_trunc('month', t_dat)::DATE AS month,
            sum(price) AS revenue,
            count(*)   AS n_txns
        FROM '{txn}'
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    # Index to the median month
    median_rev = monthly["revenue"].median()
    monthly["rev_index"] = (monthly["revenue"] / median_rev * 100).round(1)
    monthly["month_name"] = pd.to_datetime(monthly["month"]).dt.strftime("%b %Y")

    peak_month = monthly.loc[monthly["rev_index"].idxmax()]
    results["seasonality"] = {
        "peak_month": str(peak_month["month"]),
        "peak_index": float(peak_month["rev_index"]),
        "n_months": len(monthly),
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    colours = [COLOURS["accent"] if idx == monthly["rev_index"].idxmax()
               else COLOURS["primary"] for idx in monthly.index]
    ax.bar(range(len(monthly)), monthly["rev_index"], color=colours, alpha=0.85)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month_name"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Revenue index (median month = 100)")
    ax.axhline(100, color=COLOURS["muted"], linestyle="--", linewidth=0.8)
    ax.set_title(
        f"Strongest month ({peak_month['month_name']}) is "
        f"{peak_month['rev_index']:.0f}% of the median month",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, "monthly_seasonality")

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    results_path = RESULTS_DIR / "descriptives.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results → {results_path}")

    con.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descriptive analytics")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M3: Descriptives ══")
    results = run_descriptives(sample=args.sample)

    print("\n  Key findings:")
    rc = results["revenue_concentration"]
    print(f"    Top 20% customers → {rc['top_20_pct_share']}% of revenue")
    print(f"    Top 10% customers → {rc['top_10_pct_share']}% of revenue")
    print(f"    Gini coefficient: {rc['gini']}")
    print("══ Done ══\n")
