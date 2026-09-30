"""
M7 — Basket affinity analysis.

Defines a basket as (customer_id, t_dat). Works at the product-type level.
Computes support, confidence, and lift for pairs, subject to minimum support.
Removes trivial pairs (A -> A).

Lift indicates association, NOT cause.

Usage:
    python src/affinity.py              # full dataset
    python src/affinity.py --sample     # sample mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import duckdb
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting import apply_style, save_fig, COLOURS, FIGURES_DIR

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

def compute_affinity(baskets_df: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """
    Compute support, confidence, and lift for pairs of items in baskets.
    baskets_df has columns ['basket_id', 'item'].
    """
    n_baskets = baskets_df["basket_id"].nunique()
    if n_baskets == 0:
        return pd.DataFrame()
    
    # 1. Item frequencies
    item_counts = baskets_df.groupby("item")["basket_id"].nunique().to_dict()
    
    # 2. Pair frequencies
    # For each basket, get a sorted list of unique items
    basket_items = baskets_df.groupby("basket_id")["item"].unique()
    
    pair_counts = defaultdict(int)
    for items in basket_items:
        # Sort to ensure (A, B) is treated same as (B, A) for support counting
        sorted_items = sorted(list(set(items)))
        for pair in combinations(sorted_items, 2):
            pair_counts[pair] += 1
            
    # 3. Calculate metrics
    results = []
    min_count = max(1, int(min_support * n_baskets))
    
    for (item_A, item_B), pair_count in pair_counts.items():
        if pair_count < min_count:
            continue
            
        support_A = item_counts[item_A] / n_baskets
        support_B = item_counts[item_B] / n_baskets
        support_AB = pair_count / n_baskets
        
        # A -> B
        conf_A_B = support_AB / support_A
        lift_A_B = conf_A_B / support_B
        
        results.append({
            "item_A": item_A,
            "item_B": item_B,
            "pair_count": pair_count,
            "support_pct": support_AB * 100,
            "confidence_A_to_B_pct": conf_A_B * 100,
            "confidence_B_to_A_pct": (support_AB / support_B) * 100,
            "lift": lift_A_B
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("lift", ascending=False)
    return df

def run_affinity(sample: bool = False):
    con = duckdb.connect()
    txn = str(DATA_DIR / "transactions.parquet")
    art = str(DATA_DIR / "articles.parquet")
    
    min_support = CONFIG["affinity"]["min_support"]
    top_n = CONFIG["affinity"]["top_n_pairs"]
    
    print(f"  Min support: {min_support:.4f}")
    
    # Define a basket as customer_id + t_dat, items as product_type_name
    # Group by basket and collect distinct product types
    baskets_df = con.execute(f"""
        SELECT 
            t.customer_id || '_' || t.t_dat AS basket_id,
            a.product_type_name AS item
        FROM '{txn}' t
        JOIN '{art}' a ON t.article_id = a.article_id
    """).fetchdf()
    
    n_baskets = baskets_df["basket_id"].nunique()
    print(f"  Total baskets: {n_baskets:,}")
    
    affinity_df = compute_affinity(baskets_df, min_support)
    
    if affinity_df.empty:
        print("  No pairs found above minimum support.")
        con.close()
        return
        
    affinity_df = affinity_df.head(top_n)
    
    # Save table
    affinity_df.to_csv(TABLES_DIR / "affinity_top_pairs.csv", index=False)
    print(f"  → {TABLES_DIR / 'affinity_top_pairs.csv'}")
    
    # Save chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = affinity_df.apply(lambda r: f"{r['item_A']} + {r['item_B']}", axis=1)
    y_pos = range(len(affinity_df))
    
    ax.barh(y_pos[::-1], affinity_df["lift"], color=COLOURS["secondary"], alpha=0.85)
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Lift (association strength)")
    ax.set_title(
        f"Top {len(affinity_df)} product pairs by lift (min support {min_support*100:.1f}%)",
        fontsize=11
    )
    
    for i, lift in zip(y_pos[::-1], affinity_df["lift"]):
        ax.text(lift + 0.1, i, f"{lift:.1f}x", va="center", fontsize=9)
        
    ax.text(0.5, -0.15, "Note: Lift indicates association (co-occurrence), NOT causation.", 
            transform=ax.transAxes, ha="center", fontsize=9, color=COLOURS["muted"])
            
    fig.tight_layout()
    save_fig(fig, "basket_affinity_lift")
    
    # Save results json
    results = {
        "n_baskets": n_baskets,
        "min_support": min_support,
        "top_pairs": affinity_df.to_dict(orient="records")
    }
    
    results_path = RESULTS_DIR / "affinity.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → {results_path}")
    
    print("\n  Top 5 associated pairs (NOTE: association, not causation):")
    for _, r in affinity_df.head(5).iterrows():
        print(f"    {r['item_A']} + {r['item_B']}: Lift {r['lift']:>4.1f}x (Support {r['support_pct']:>4.1f}%)")
        
    con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basket Affinity")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M7: Affinity ══")
    run_affinity(sample=args.sample)
    print("══ Done ══\n")
