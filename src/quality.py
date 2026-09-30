"""
M2 — Data-quality checks and cleaning report.

Each check produces a count, a treatment, and the effect on row counts.
Nothing is silently dropped.

Usage:
    python src/quality.py              # full dataset
    python src/quality.py --sample     # sample mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = ROOT / "docs"


def run_quality_checks(sample: bool = False) -> dict:
    """Run all quality checks and return results dict."""

    con = duckdb.connect()

    txn = str(DATA_DIR / "transactions.parquet")
    cust = str(DATA_DIR / "customers.parquet")
    art = str(DATA_DIR / "articles.parquet")

    checks = []
    decisions = []

    # ── Starting row counts ─────────────────────────────────────
    n_txn = con.execute(f"SELECT count(*) FROM '{txn}'").fetchone()[0]
    n_cust = con.execute(f"SELECT count(*) FROM '{cust}'").fetchone()[0]
    n_art = con.execute(f"SELECT count(*) FROM '{art}'").fetchone()[0]

    print(f"  Starting counts: {n_txn:,} txns, {n_cust:,} customers, {n_art:,} articles")

    # ── 1. Duplicate transaction rows ───────────────────────────
    n_dup = con.execute(f"""
        SELECT count(*) FROM (
            SELECT t_dat, customer_id, article_id, price, sales_channel_id,
                   row_number() OVER (
                       PARTITION BY t_dat, customer_id, article_id, price, sales_channel_id
                       ORDER BY t_dat
                   ) AS rn
            FROM '{txn}'
        ) WHERE rn > 1
    """).fetchone()[0]
    checks.append({
        "check": "Duplicate transaction rows (all columns identical)",
        "count": n_dup,
        "treatment": "Kept — a customer can buy the same article at the same price multiple times on the same day",
        "rows_affected": 0,
    })
    decisions.append(
        "**Duplicate transaction rows**: Found {:,} rows that are exact duplicates across all columns. "
        "Treatment: **kept**, because a customer can legitimately purchase multiple units of the same item "
        "in the same transaction. There is no unique transaction ID to distinguish them.".format(n_dup)
    )

    # ── 2. Missing or implausible age ───────────────────────────
    age_stats = con.execute(f"""
        SELECT
            count(*) FILTER (WHERE age IS NULL)     AS n_null,
            count(*) FILTER (WHERE age < 15)        AS n_under_15,
            count(*) FILTER (WHERE age > 100)       AS n_over_100,
            min(age), max(age),
            approx_quantile(age, 0.5)               AS median_age
        FROM '{cust}'
    """).fetchone()
    n_null_age, n_under_15, n_over_100, min_age, max_age, median_age = age_stats
    checks.append({
        "check": "Missing age",
        "count": n_null_age,
        "pct": round(n_null_age / n_cust * 100, 2),
        "treatment": "Kept as NULL — not imputed; model will use indicator for missing age",
        "rows_affected": 0,
    })
    checks.append({
        "check": "Implausible age (< 15 or > 100)",
        "count": n_under_15 + n_over_100,
        "detail": f"Under 15: {n_under_15:,}, Over 100: {n_over_100:,}, Range: [{min_age}, {max_age}]",
        "treatment": "Ages outside [15, 100] set to NULL — likely data entry errors",
        "rows_affected": n_under_15 + n_over_100,
    })
    decisions.append(
        "**Implausible ages**: {:,} customers have age < 15 or > 100 (range [{}, {}]). "
        "Treatment: **set to NULL**. A fashion retailer's customer base is unlikely to include "
        "infants or centenarians at scale; these are likely data entry errors. "
        "Median age = {:.0f}.".format(n_under_15 + n_over_100, min_age, max_age, median_age)
    )

    # ── 3. Customers with no transactions ───────────────────────
    n_no_txn = con.execute(f"""
        SELECT count(*) FROM '{cust}' c
        WHERE NOT EXISTS (
            SELECT 1 FROM '{txn}' t WHERE t.customer_id = c.customer_id
        )
    """).fetchone()[0]
    checks.append({
        "check": "Customers with no transactions",
        "count": n_no_txn,
        "pct": round(n_no_txn / n_cust * 100, 2),
        "treatment": "Excluded from analysis — no purchase history to analyse",
        "rows_affected": n_no_txn,
    })
    decisions.append(
        "**Customers with no transactions**: {:,} ({:.1f}% of customer table). "
        "Treatment: **excluded from all analysis**. Without purchase history, "
        "these customers cannot be segmented or modelled.".format(
            n_no_txn, n_no_txn / n_cust * 100
        )
    )

    # ── 4. Transactions with unknown customers ──────────────────
    n_unknown_cust = con.execute(f"""
        SELECT count(*) FROM '{txn}' t
        WHERE NOT EXISTS (
            SELECT 1 FROM '{cust}' c WHERE c.customer_id = t.customer_id
        )
    """).fetchone()[0]
    checks.append({
        "check": "Transactions with unknown customer_id (not in customers table)",
        "count": n_unknown_cust,
        "treatment": "Kept — customer metadata will be NULL for these; they can still contribute to descriptives",
        "rows_affected": 0,
    })

    # ── 5. Transactions with unknown articles ───────────────────
    n_unknown_art = con.execute(f"""
        SELECT count(*) FROM '{txn}' t
        WHERE NOT EXISTS (
            SELECT 1 FROM '{art}' a WHERE a.article_id = t.article_id
        )
    """).fetchone()[0]
    checks.append({
        "check": "Transactions with unknown article_id (not in articles table)",
        "count": n_unknown_art,
        "treatment": "Kept — article metadata will be NULL for these; price and date are still valid",
        "rows_affected": 0,
    })

    # ── 6. Price distribution and outliers ──────────────────────
    price_stats = con.execute(f"""
        SELECT
            count(*) FILTER (WHERE price IS NULL)     AS n_null_price,
            count(*) FILTER (WHERE price <= 0)        AS n_non_positive,
            min(price), max(price),
            approx_quantile(price, 0.25)              AS p25,
            approx_quantile(price, 0.50)              AS p50,
            approx_quantile(price, 0.75)              AS p75,
            approx_quantile(price, 0.99)              AS p99,
            avg(price)                                AS mean_price
        FROM '{txn}'
    """).fetchone()
    n_null_price, n_non_pos, min_p, max_p, p25, p50, p75, p99, mean_p = price_stats
    checks.append({
        "check": "Price: null or non-positive",
        "count": n_null_price + n_non_pos,
        "detail": f"NULL: {n_null_price}, ≤ 0: {n_non_pos}",
        "treatment": "Kept — prices are anonymised/normalised; non-positive values flagged but not removed",
        "rows_affected": 0,
    })
    checks.append({
        "check": "Price distribution",
        "count": n_txn,
        "detail": f"min={min_p:.4f}, p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}, p99={p99:.4f}, max={max_p:.4f}, mean={mean_p:.4f}",
        "treatment": "No outlier removal — prices are anonymised and we cannot judge what is 'too high'",
        "rows_affected": 0,
    })
    decisions.append(
        "**Price outliers**: Prices are anonymised/normalised. Distribution: "
        "min={:.4f}, p25={:.4f}, median={:.4f}, p75={:.4f}, p99={:.4f}, max={:.4f}. "
        "Treatment: **no removal**. We cannot judge which prices are implausible "
        "without knowing the normalisation scheme.".format(min_p, p25, p50, p75, p99, max_p)
    )

    # ── 7. Single-purchase customers ────────────────────────────
    single_purchase = con.execute(f"""
        SELECT count(*) FROM (
            SELECT customer_id, count(*) AS n
            FROM '{txn}'
            GROUP BY customer_id
            HAVING count(*) = 1
        )
    """).fetchone()[0]
    total_buying_cust = con.execute(f"""
        SELECT count(DISTINCT customer_id) FROM '{txn}'
    """).fetchone()[0]
    checks.append({
        "check": "Single-purchase customers",
        "count": single_purchase,
        "pct": round(single_purchase / total_buying_cust * 100, 2),
        "treatment": "Kept — included in descriptives and segmentation; will naturally appear in 'Lapsed' segment",
        "rows_affected": 0,
    })

    # ── 8. Date gaps ────────────────────────────────────────────
    date_gaps = con.execute(f"""
        WITH daily AS (
            SELECT t_dat, count(*) AS n_txns
            FROM '{txn}'
            GROUP BY t_dat
            ORDER BY t_dat
        ),
        gaps AS (
            SELECT t_dat,
                   t_dat - lag(t_dat) OVER (ORDER BY t_dat) AS gap_days
            FROM daily
        )
        SELECT count(*) FILTER (WHERE gap_days > 1) AS n_gaps,
               max(gap_days) AS max_gap
        FROM gaps
    """).fetchone()
    n_gaps, max_gap = date_gaps
    # Get the date of the max gap separately
    if max_gap and max_gap > 1:
        max_gap_date = con.execute(f"""
            WITH daily AS (
                SELECT t_dat FROM '{txn}' GROUP BY t_dat ORDER BY t_dat
            ),
            gaps AS (
                SELECT t_dat,
                       t_dat - lag(t_dat) OVER (ORDER BY t_dat) AS gap_days
                FROM daily
            )
            SELECT t_dat FROM gaps WHERE gap_days = {max_gap} LIMIT 1
        """).fetchone()[0]
    else:
        max_gap_date = None
    checks.append({
        "check": "Date gaps (days with no transactions)",
        "count": n_gaps if n_gaps else 0,
        "detail": f"Max gap: {max_gap} days (around {max_gap_date})" if max_gap else "No gaps",
        "treatment": "No action — gaps are expected (e.g. holidays, data collection artefacts)",
        "rows_affected": 0,
    })

    # ── 9. Sales channel distribution ───────────────────────────
    channel_dist = con.execute(f"""
        SELECT sales_channel_id, count(*) AS n, 
               round(count(*) * 100.0 / sum(count(*)) OVER (), 2) AS pct
        FROM '{txn}'
        GROUP BY sales_channel_id
        ORDER BY sales_channel_id
    """).fetchdf()
    checks.append({
        "check": "Sales channel distribution",
        "count": len(channel_dist),
        "detail": "; ".join(f"channel {int(r['sales_channel_id'])}: {r['n']:,.0f} ({r['pct']:.1f}%)" 
                           for _, r in channel_dist.iterrows()),
        "treatment": "No action — used as a feature (online vs store)",
        "rows_affected": 0,
    })

    # ── 10. Club member status ──────────────────────────────────
    club_dist = con.execute(f"""
        SELECT club_member_status, count(*) AS n,
               round(count(*) * 100.0 / sum(count(*)) OVER (), 2) AS pct
        FROM '{cust}'
        GROUP BY club_member_status
        ORDER BY n DESC
    """).fetchdf()
    checks.append({
        "check": "Club member status distribution",
        "count": len(club_dist),
        "detail": "; ".join(f"{r['club_member_status']}: {r['n']:,.0f} ({r['pct']:.1f}%)" 
                           for _, r in club_dist.iterrows()),
        "treatment": "NULL/missing values kept as a separate category for features",
        "rows_affected": 0,
    })

    # ── Summary ─────────────────────────────────────────────────
    total_rows_affected = sum(c.get("rows_affected", 0) for c in checks)
    summary = {
        "starting_counts": {
            "transactions": n_txn,
            "customers": n_cust,
            "articles": n_art,
        },
        "total_rows_flagged_or_cleaned": total_rows_affected,
        "checks": checks,
    }

    # ── Write results ───────────────────────────────────────────
    results_path = RESULTS_DIR / "quality.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  → {results_path}")

    # ── Write CSV table ─────────────────────────────────────────
    table_rows = []
    for c in checks:
        table_rows.append({
            "Check": c["check"],
            "Count": c["count"],
            "Percentage": c.get("pct", ""),
            "Detail": c.get("detail", ""),
            "Treatment": c["treatment"],
            "Rows affected": c.get("rows_affected", 0),
        })
    df = pd.DataFrame(table_rows)
    csv_path = TABLES_DIR / "quality.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # ── Append to decisions.md ──────────────────────────────────
    decisions_path = DOCS_DIR / "decisions.md"
    with open(decisions_path, "a") as f:
        f.write("\n\n## M2: Data-Quality Cleaning Decisions\n\n")
        for i, d in enumerate(decisions, 1):
            f.write(f"{i}. {d}\n\n")
        f.write(f"**Net effect**: {total_rows_affected:,} customer records flagged for cleaning. "
                f"No transaction rows were dropped.\n")
    print(f"  → decisions appended to {decisions_path}")

    con.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-quality checks")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M2: Data Quality ══")
    results = run_quality_checks(sample=args.sample)

    print("\n  Summary of checks:")
    for c in results["checks"]:
        status = "⚠" if c.get("rows_affected", 0) > 0 else "✓"
        print(f"    {status} {c['check']}: {c['count']:,}")

    print("══ Done ══\n")
