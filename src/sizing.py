"""
M8 — Opportunity sizing.

Defines the target group as customers in top risk decile(s) who are
in high-value segments. Estimates value at risk based on their observed
spend in the test window, and applies retention uplift scenarios.

Never presents revenue in currency (uses an index or share of total).

Usage:
    python src/sizing.py              # full dataset
    python src/sizing.py --sample     # sample mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def run_sizing(sample: bool = False):
    con = duckdb.connect()
    
    txn_file = str(DATA_DIR / "transactions.parquet")
    preds_file = str(DATA_DIR / "test_predictions.parquet")
    rfm_file = str(DATA_DIR / "rfm_segments.parquet")
    
    # Load snapshot dates
    with open(RESULTS_DIR / "snapshots.json") as f:
        snapshots = json.load(f)
    
    T_test = snapshots["snapshots"]["test"]["T"]
    label_end = snapshots["snapshots"]["test"]["label_end"]
    
    target_deciles = tuple(CONFIG["sizing"]["target_deciles"])
    uplift_scenarios = CONFIG["sizing"]["retention_uplift_pp"]
    
    # High-value segments
    target_segments = ('Champions', 'Loyal', 'Promising')
    
    # ── 1. Calculate actual window revenue for test cohort ──────────
    # Revenue during [T_test, label_end)
    cohort_rev = con.execute(f"""
        WITH target_cohort AS (
            SELECT p.customer_id, p.risk_decile, p.lapsed, r.segment
            FROM '{preds_file}' p
            LEFT JOIN '{rfm_file}' r ON p.customer_id = r.customer_id
        ),
        window_spend AS (
            SELECT customer_id, sum(price) as spend
            FROM '{txn_file}'
            WHERE t_dat >= DATE '{T_test}' AND t_dat < DATE '{label_end}'
            GROUP BY customer_id
        )
        SELECT 
            c.customer_id, 
            c.risk_decile, 
            c.lapsed, 
            c.segment, 
            COALESCE(w.spend, 0.0) as actual_spend
        FROM target_cohort c
        LEFT JOIN window_spend w ON c.customer_id = w.customer_id
    """).fetchdf()
    
    total_window_revenue = cohort_rev["actual_spend"].sum()
    
    # Baseline for index
    index_base = CONFIG["sizing"]["revenue_index_base"]
    
    # Normalize revenue to the index base (e.g. 100 = total test window revenue)
    # So if total is 1,000,000, and a group's spend is 10,000, their index value is 1.
    rev_multiplier = index_base / total_window_revenue
    
    # ── 2. Define target group and value at risk ────────────────────
    
    cohort_rev["is_target"] = (
        cohort_rev["risk_decile"].isin(target_deciles) & 
        cohort_rev["segment"].isin(target_segments)
    )
    
    target_group = cohort_rev[cohort_rev["is_target"]]
    n_target_customers = len(target_group)
    
    # The value of the target group in the test window is what they actually spent.
    # Wait, if they are lapsers, their actual spend is 0. If they didn't lapse, they spent >0.
    # The "Value at Risk" is the expected value IF they had not lapsed, minus what they actually spent.
    # Let's estimate the expected value of a retained target customer.
    retained_target = target_group[target_group["lapsed"] == 0]
    avg_retained_spend = retained_target["actual_spend"].mean() if not retained_target.empty else 0.0
    
    # Expected total value if ALL target customers were retained
    potential_total_spend = n_target_customers * avg_retained_spend
    
    # Actual spend of target group (only retained ones contribute)
    actual_target_spend = target_group["actual_spend"].sum()
    
    # Value lost to lapse
    value_lost = potential_total_spend - actual_target_spend
    
    # Expressed as index points
    idx_value_lost = value_lost * rev_multiplier
    idx_avg_retained_spend = avg_retained_spend * rev_multiplier
    
    print(f"  Target group: {n_target_customers:,} high-value, high-risk customers")
    print(f"  Total window revenue (Index): {index_base:.1f}")
    print(f"  Value lost to lapse in target group (Index): {idx_value_lost:.2f}")
    
    # ── 3. Retention Uplift Scenarios ───────────────────────────────
    
    scenarios = []
    
    # Uplift is defined in percentage points (e.g., 2 pp).
    # If we prevent 2% of the target group from lapsing, they become retained.
    for uplift_pp in uplift_scenarios:
        # Number of additional customers retained
        n_saved = n_target_customers * (uplift_pp / 100.0)
        
        # Additional revenue gained
        rev_gained = n_saved * avg_retained_spend
        idx_rev_gained = rev_gained * rev_multiplier
        
        # Share of total window revenue
        share_of_total_pct = (rev_gained / total_window_revenue) * 100
        
        # Break-even cost per CONTACTED customer (assuming we target the whole group)
        # Max cost = (Total Rev Gained) / (Total Contacted)
        # Expressed as % of average retained customer's value
        break_even_pct = (rev_gained / n_target_customers) / avg_retained_spend * 100
        
        scenarios.append({
            "Assumed_Uplift_pp": uplift_pp,
            "Customers_Saved": int(n_saved),
            "Revenue_Recovered_Index": round(idx_rev_gained, 2),
            "Share_of_Total_Revenue_pct": round(share_of_total_pct, 3),
            "Break_Even_Cost_pct_of_LTV": round(break_even_pct, 1)
        })
        
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df.to_csv(TABLES_DIR / "sizing_scenarios.csv", index=False)
    
    print(f"\n  Scenarios:")
    print(scenarios_df.to_string(index=False))
    
    # ── 4. Assumptions Table ────────────────────────────────────────
    
    assumptions = [
        {"Parameter": "Target risk deciles", "Value": str(target_deciles)},
        {"Parameter": "Target segments", "Value": ", ".join(target_segments)},
        {"Parameter": "Average value of retained target customer (Index)", "Value": f"{idx_avg_retained_spend:.4f}"},
        {"Parameter": "Test window duration", "Value": "12 weeks"},
        {"Parameter": "Intervention success", "Value": "See uplift scenarios (assumptions, not findings)"}
    ]
    assumptions_df = pd.DataFrame(assumptions)
    assumptions_df.to_csv(TABLES_DIR / "sizing_assumptions.csv", index=False)
    
    # ── 5. Sensitivity Chart ────────────────────────────────────────
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [str(x) + " pp" for x in scenarios_df["Assumed_Uplift_pp"]], 
        scenarios_df["Revenue_Recovered_Index"], 
        color=COLOURS["primary"], 
        alpha=0.85
    )
    ax.set_xlabel("Assumed Retention Uplift")
    ax.set_ylabel("Revenue Recovered (Index)")
    ax.set_title(
        "Opportunity Sizing: Recovered Revenue by Intervention Success\n"
        "(Note: Uplifts are assumptions, not model findings)",
        fontsize=11
    )
    for i, val in enumerate(scenarios_df["Revenue_Recovered_Index"]):
        ax.text(i, val + (scenarios_df["Revenue_Recovered_Index"].max()*0.02), 
                f"{val:.2f}", ha="center", fontsize=9)
        
    fig.tight_layout()
    save_fig(fig, "sizing_sensitivity")
    
    # ── 6. Save results ─────────────────────────────────────────────
    
    results = {
        "n_target_customers": n_target_customers,
        "value_lost_index": round(idx_value_lost, 2),
        "total_revenue_index": index_base,
        "scenarios": scenarios,
        "assumptions": assumptions
    }
    
    results_path = RESULTS_DIR / "sizing.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → {results_path}")
    
    con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opportunity Sizing")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M8: Sizing ══")
    run_sizing(sample=args.sample)
    print("══ Done ══\n")
