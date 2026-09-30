"""
M4a — RFM segmentation.

Computes recency, frequency and monetary value at a fixed reference date,
scores into quintiles, and maps to named segments with explicit rules.

Usage:
    python src/segments.py              # full dataset
    python src/segments.py --sample     # sample mode
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

from plotting import apply_style, save_fig, COLOURS, SEGMENT_COLOURS, FIGURES_DIR

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()


def _assign_segment(row: pd.Series, rules: dict) -> str:
    """Assign a customer to the first matching segment (order matters)."""
    r, f, m = row["R_q"], row["F_q"], row["M_q"]
    # Check segments in priority order
    for seg_name in ["Champions", "Loyal", "At_risk", "Promising", "Lapsed"]:
        rule = rules.get(seg_name)
        if rule is None:
            continue
        if (r in rule["recency_q"] and f in rule["frequency_q"] and m in rule["monetary_q"]):
            return seg_name
    return "Other"


def run_segmentation(sample: bool = False) -> dict:
    """Build RFM scores and segment customers."""

    con = duckdb.connect()
    txn = str(DATA_DIR / "transactions.parquet")

    # Reference date = day after last transaction
    max_date = con.execute(f"SELECT max(t_dat) FROM '{txn}'").fetchone()[0]
    ref_date = max_date  # reference date for recency

    print(f"  Reference date: {ref_date}")

    # ── 1. Compute RFM metrics ──────────────────────────────────
    rfm_df = con.execute(f"""
        SELECT
            customer_id,
            datediff('day', max(t_dat), DATE '{ref_date}') AS recency_days,
            count(DISTINCT t_dat)                           AS frequency,
            sum(price)                                      AS monetary
        FROM '{txn}'
        GROUP BY customer_id
    """).fetchdf()

    print(f"  Customers with purchases: {len(rfm_df):,}")

    # ── 2. Score into quintiles ─────────────────────────────────
    n_q = CONFIG["rfm"]["n_quantiles"]
    # Recency: lower is better → invert so quintile 5 = most recent
    rfm_df["R_q"] = pd.qcut(rfm_df["recency_days"], n_q, labels=False, duplicates="drop") + 1
    rfm_df["R_q"] = n_q + 1 - rfm_df["R_q"]  # invert

    rfm_df["F_q"] = pd.qcut(rfm_df["frequency"].rank(method="first"), n_q, labels=False, duplicates="drop") + 1
    rfm_df["M_q"] = pd.qcut(rfm_df["monetary"].rank(method="first"), n_q, labels=False, duplicates="drop") + 1

    # ── 3. Map to named segments ────────────────────────────────
    segment_rules = CONFIG["rfm"]["segments"]
    rfm_df["segment"] = rfm_df.apply(lambda r: _assign_segment(r, segment_rules), axis=1)

    # Check every customer is in exactly one segment
    assert rfm_df["segment"].notna().all(), "Some customers have no segment"
    assert len(rfm_df) == rfm_df["customer_id"].nunique(), "Duplicate customer IDs"

    # ── 4. Segment summary ──────────────────────────────────────
    total_rev = rfm_df["monetary"].sum()
    seg_summary = (
        rfm_df.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
    )
    seg_summary["revenue_share_pct"] = (seg_summary["total_revenue"] / total_rev * 100).round(1)
    seg_summary["customer_share_pct"] = (seg_summary["n_customers"] / len(rfm_df) * 100).round(1)
    seg_summary = seg_summary.sort_values("revenue_share_pct", ascending=False)

    print("\n  Segment summary:")
    for _, r in seg_summary.iterrows():
        print(f"    {r['segment']:12s}  {r['n_customers']:>8,} customers ({r['customer_share_pct']:>5.1f}%)  "
              f"rev share {r['revenue_share_pct']:>5.1f}%")

    # ── 5. Save results ─────────────────────────────────────────
    results = {
        "reference_date": str(ref_date),
        "total_customers": len(rfm_df),
        "segments": {
            r["segment"]: {
                "n_customers": int(r["n_customers"]),
                "customer_share_pct": float(r["customer_share_pct"]),
                "revenue_share_pct": float(r["revenue_share_pct"]),
                "avg_recency_days": round(float(r["avg_recency"]), 1),
                "avg_frequency": round(float(r["avg_frequency"]), 1),
                "avg_monetary_index": round(float(r["avg_monetary"] / total_rev * len(rfm_df) * 100), 1),
            }
            for _, r in seg_summary.iterrows()
        },
        "quintile_boundaries": {
            "recency_days": rfm_df.groupby("R_q")["recency_days"].agg(["min", "max"]).to_dict(),
            "frequency": rfm_df.groupby("F_q")["frequency"].agg(["min", "max"]).to_dict(),
        },
    }

    results_path = RESULTS_DIR / "segments.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → {results_path}")

    # Save segment table
    seg_summary.to_csv(TABLES_DIR / "segments.csv", index=False)
    print(f"  → {TABLES_DIR / 'segments.csv'}")

    # Save RFM data for downstream use
    rfm_df.to_parquet(DATA_DIR / "rfm_segments.parquet", index=False)
    print(f"  → {DATA_DIR / 'rfm_segments.parquet'}")

    # ── 6. Charts ───────────────────────────────────────────────
    # Segment revenue share
    seg_order = seg_summary.sort_values("revenue_share_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_colours = [SEGMENT_COLOURS.get(s, COLOURS["muted"]) for s in seg_order["segment"]]
    bars = ax.barh(seg_order["segment"], seg_order["revenue_share_pct"], color=bar_colours, alpha=0.85)
    for bar, pct, n in zip(bars, seg_order["revenue_share_pct"], seg_order["customer_share_pct"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% rev | {n:.0f}% cust", va="center", fontsize=8)
    ax.set_xlabel("Share of total revenue (%)")

    champion_row = seg_summary[seg_summary["segment"] == "Champions"]
    if not champion_row.empty:
        champ_rev = champion_row.iloc[0]["revenue_share_pct"]
        champ_cust = champion_row.iloc[0]["customer_share_pct"]
        ax.set_title(
            f"Champions are {champ_cust:.0f}% of customers but drive {champ_rev:.0f}% of revenue",
            fontsize=11,
        )
    else:
        ax.set_title("Revenue share by customer segment", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "segment_revenue_share")

    con.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFM Segmentation")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M4a: Segmentation ══")
    run_segmentation(sample=args.sample)
    print("══ Done ══\n")
