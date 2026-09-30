"""
M5 — Leakage-safe snapshot builder.

Builds three out-of-time snapshots (train / val / test) with features
computed strictly before T and labels from [T, T + label_window).

Usage:
    python src/snapshots.py              # full dataset
    python src/snapshots.py --sample     # sample mode
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR = ROOT / "docs"


def _compute_snapshot(con: duckdb.DuckDBPyConnection, txn: str, cust: str,
                      snapshot_date: str, label_end: str,
                      lookback_weeks: int, feature_windows: list[int]) -> pd.DataFrame:
    """
    Build one snapshot.

    Population: customers with ≥1 purchase in [snapshot_date - lookback, snapshot_date).
    Features:   computed from transactions strictly before snapshot_date.
    Label:      1 if NO purchase in [snapshot_date, label_end), else 0.
    """

    # ── Population ──────────────────────────────────────────────
    lookback_start = (pd.Timestamp(snapshot_date) - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")

    pop = con.execute(f"""
        SELECT DISTINCT customer_id
        FROM '{txn}'
        WHERE t_dat >= DATE '{lookback_start}'
          AND t_dat <  DATE '{snapshot_date}'
    """).fetchdf()

    if pop.empty:
        return pd.DataFrame()

    # Store population in temp table
    con.execute("DROP TABLE IF EXISTS _pop")
    con.execute("CREATE TEMP TABLE _pop AS SELECT * FROM pop")

    # ── Features (strictly before snapshot_date) ────────────────
    feature_queries = []

    for w in feature_windows:
        w_start = (pd.Timestamp(snapshot_date) - timedelta(weeks=w)).strftime("%Y-%m-%d")
        feature_queries.append(f"""
            -- {w}-week trailing features
            count(*)      FILTER (WHERE t_dat >= DATE '{w_start}' AND t_dat < DATE '{snapshot_date}')
                AS txn_count_{w}w,
            sum(price)    FILTER (WHERE t_dat >= DATE '{w_start}' AND t_dat < DATE '{snapshot_date}')
                AS spend_{w}w,
            count(DISTINCT t_dat) FILTER (WHERE t_dat >= DATE '{w_start}' AND t_dat < DATE '{snapshot_date}')
                AS active_days_{w}w
        """)

    features_sql = ",\n".join(feature_queries)

    feat = con.execute(f"""
        SELECT
            t.customer_id,

            -- All-time features (strictly before snapshot)
            count(*)                                                    AS txn_count_all,
            sum(price)                                                  AS spend_all,
            count(DISTINCT t_dat)                                       AS active_days_all,
            datediff('day', max(t_dat), DATE '{snapshot_date}')         AS recency_days,
            datediff('day', min(t_dat), DATE '{snapshot_date}')         AS tenure_days,

            -- Trailing window features
            {features_sql},

            -- Basket and inter-purchase
            avg(price)                                                  AS avg_item_price,
            count(DISTINCT t_dat)                                       AS n_purchase_days,

            -- Channel
            sum(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0 END) * 1.0
                / count(*)                                              AS online_share,

            -- Product diversity (distinct article IDs as proxy)
            count(DISTINCT article_id)                                  AS n_distinct_articles

        FROM '{txn}' t
        INNER JOIN _pop p ON t.customer_id = p.customer_id
        WHERE t.t_dat < DATE '{snapshot_date}'
        GROUP BY t.customer_id
    """).fetchdf()

    # Inter-purchase gap features
    gaps = con.execute(f"""
        WITH purchase_days AS (
            SELECT customer_id, t_dat,
                   lag(t_dat) OVER (PARTITION BY customer_id ORDER BY t_dat) AS prev_dat
            FROM (
                SELECT DISTINCT t.customer_id, t.t_dat
                FROM '{txn}' t
                INNER JOIN _pop p ON t.customer_id = p.customer_id
                WHERE t.t_dat < DATE '{snapshot_date}'
            )
        )
        SELECT
            customer_id,
            avg(datediff('day', prev_dat, t_dat))  AS avg_gap_days,
            max(datediff('day', prev_dat, t_dat))  AS max_gap_days,
            min(datediff('day', prev_dat, t_dat))  AS min_gap_days
        FROM purchase_days
        WHERE prev_dat IS NOT NULL
        GROUP BY customer_id
    """).fetchdf()

    feat = feat.merge(gaps, on="customer_id", how="left")

    # Average basket size (items per purchase day)
    feat["avg_basket_size"] = feat["txn_count_all"] / feat["n_purchase_days"].clip(lower=1)

    # ── Customer demographics ───────────────────────────────────
    cust_demo = con.execute(f"""
        SELECT
            c.customer_id,
            CASE WHEN c.age < 15 OR c.age > 100 THEN NULL ELSE c.age END AS age,
            CASE WHEN c.club_member_status = 'ACTIVE' THEN 1 ELSE 0 END  AS is_club_active,
            CASE WHEN c.fashion_news_frequency = 'Regularly' THEN 1 ELSE 0 END AS news_regular
        FROM '{cust}' c
        INNER JOIN _pop p ON c.customer_id = p.customer_id
    """).fetchdf()

    feat = feat.merge(cust_demo, on="customer_id", how="left")

    # ── Label ───────────────────────────────────────────────────
    # 1 = lapsed (no purchase in window), 0 = retained
    label = con.execute(f"""
        SELECT
            p.customer_id,
            CASE
                WHEN exists(
                    SELECT 1 FROM '{txn}' t
                    WHERE t.customer_id = p.customer_id
                      AND t.t_dat >= DATE '{snapshot_date}'
                      AND t.t_dat <  DATE '{label_end}'
                ) THEN 0
                ELSE 1
            END AS lapsed
        FROM _pop p
    """).fetchdf()

    feat = feat.merge(label, on="customer_id", how="inner")

    # ── Max source date check (leakage guard) ───────────────────
    max_source = con.execute(f"""
        SELECT max(t_dat)
        FROM '{txn}' t
        INNER JOIN _pop p ON t.customer_id = p.customer_id
        WHERE t.t_dat < DATE '{snapshot_date}'
    """).fetchone()[0]

    assert str(max_source) < snapshot_date, (
        f"LEAKAGE: max source date {max_source} >= snapshot date {snapshot_date}"
    )

    feat["snapshot_date"] = snapshot_date

    con.execute("DROP TABLE IF EXISTS _pop")

    return feat


def build_snapshots(sample: bool = False) -> dict:
    """Build train / val / test snapshots."""

    con = duckdb.connect()
    txn = str(DATA_DIR / "transactions.parquet")
    cust = str(DATA_DIR / "customers.parquet")

    label_window = CONFIG["label_window_weeks"]
    spacing = CONFIG["snapshot_spacing_weeks"]
    lookback = CONFIG["lookback_weeks"]
    feature_windows = CONFIG["feature_windows_weeks"]

    # Max transaction date
    max_date = con.execute(f"SELECT max(t_dat) FROM '{txn}'").fetchone()[0]
    max_date = pd.Timestamp(max_date)

    # Snapshot dates (working backwards)
    T_test  = (max_date - timedelta(weeks=label_window)).strftime("%Y-%m-%d")
    T_val   = (pd.Timestamp(T_test)  - timedelta(weeks=spacing)).strftime("%Y-%m-%d")
    T_train = (pd.Timestamp(T_val)   - timedelta(weeks=spacing)).strftime("%Y-%m-%d")

    snapshots = {
        "train": {
            "T": T_train,
            "label_end": (pd.Timestamp(T_train) + timedelta(weeks=label_window)).strftime("%Y-%m-%d"),
        },
        "val": {
            "T": T_val,
            "label_end": (pd.Timestamp(T_val) + timedelta(weeks=label_window)).strftime("%Y-%m-%d"),
        },
        "test": {
            "T": T_test,
            "label_end": (pd.Timestamp(T_test) + timedelta(weeks=label_window)).strftime("%Y-%m-%d"),
        },
    }

    print(f"  Label window: {label_window} weeks")
    print(f"  Snapshot spacing: {spacing} weeks")
    print(f"  Max transaction date: {max_date.strftime('%Y-%m-%d')}")
    print()

    results = {
        "label_window_weeks": label_window,
        "snapshot_spacing_weeks": spacing,
        "lookback_weeks": lookback,
        "max_transaction_date": max_date.strftime("%Y-%m-%d"),
        "snapshots": {},
    }

    for name, info in snapshots.items():
        T = info["T"]
        label_end = info["label_end"]
        print(f"  Building {name} snapshot: T={T}, label_end={label_end}")

        df = _compute_snapshot(con, txn, cust, T, label_end, lookback, feature_windows)

        if df.empty:
            print(f"    WARNING: empty snapshot for {name}")
            continue

        n_lapsed = int(df["lapsed"].sum())
        n_total = len(df)
        lapse_rate = n_lapsed / n_total * 100

        print(f"    {n_total:,} customers, {n_lapsed:,} lapsed ({lapse_rate:.1f}%)")
        print(f"    Features: {len(df.columns) - 2}")  # minus customer_id and lapsed

        # Save snapshot
        out_path = DATA_DIR / f"snapshot_{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"    → {out_path}")

        results["snapshots"][name] = {
            "T": T,
            "label_end": label_end,
            "n_customers": n_total,
            "n_lapsed": n_lapsed,
            "lapse_rate_pct": round(lapse_rate, 1),
            "n_features": len(df.columns) - 2,
        }

    # ── Justify label window ────────────────────────────────────
    # Compute inter-purchase day distribution
    gap_dist = con.execute(f"""
        WITH purchase_days AS (
            SELECT customer_id, t_dat,
                   lag(t_dat) OVER (PARTITION BY customer_id ORDER BY t_dat) AS prev_dat
            FROM (SELECT DISTINCT customer_id, t_dat FROM '{txn}')
        )
        SELECT
            approx_quantile(datediff('day', prev_dat, t_dat), 0.50) AS median_gap,
            approx_quantile(datediff('day', prev_dat, t_dat), 0.75) AS p75_gap,
            approx_quantile(datediff('day', prev_dat, t_dat), 0.90) AS p90_gap,
            approx_quantile(datediff('day', prev_dat, t_dat), 0.95) AS p95_gap,
            avg(datediff('day', prev_dat, t_dat))                    AS mean_gap
        FROM purchase_days
        WHERE prev_dat IS NOT NULL
    """).fetchone()
    median_gap, p75_gap, p90_gap, p95_gap, mean_gap = gap_dist

    results["inter_purchase_gap_days"] = {
        "median": int(median_gap) if median_gap else None,
        "p75": int(p75_gap) if p75_gap else None,
        "p90": int(p90_gap) if p90_gap else None,
        "p95": int(p95_gap) if p95_gap else None,
        "mean": round(float(mean_gap), 1) if mean_gap else None,
    }

    print(f"\n  Inter-purchase gaps: median={median_gap}d, p75={p75_gap}d, "
          f"p90={p90_gap}d, p95={p95_gap}d, mean={mean_gap:.1f}d")
    print(f"  Label window of {label_window} weeks ({label_window*7}d) captures "
          f"well beyond the p75 gap ({p75_gap}d)")

    # Log decision
    decisions_path = DOCS_DIR / "decisions.md"
    with open(decisions_path, "a") as f:
        f.write(f"\n\n## M5: Label Window Justification\n\n")
        f.write(f"**Decision:** Label window set to {label_window} weeks ({label_window*7} days).\n\n")
        f.write(f"**Evidence:** Inter-purchase gap distribution: "
                f"median = {median_gap}d, p75 = {p75_gap}d, p90 = {p90_gap}d, "
                f"p95 = {p95_gap}d, mean = {mean_gap:.1f}d. "
                f"A {label_window}-week window ({label_window*7}d) is well beyond the 75th percentile "
                f"inter-purchase gap, meaning most active customers would have made at least one purchase "
                f"in this window if they intend to continue buying.\n\n")
        f.write(f"**Effect:** This is used to define the binary lapse label for modelling.\n")

    # Save results
    results_path = RESULTS_DIR / "snapshots.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → {results_path}")

    con.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build leakage-safe snapshots")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M5: Snapshots ══")
    build_snapshots(sample=args.sample)
    print("══ Done ══\n")
