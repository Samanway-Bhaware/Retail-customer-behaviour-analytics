"""
M1 — Ingest: CSV → Parquet + schema report.

Usage:
    python src/ingest.py              # full dataset
    python src/ingest.py --sample     # deterministic 5 % customer sample
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Expected source files (verify in this script, do not assume)
SRC_FILES = {
    "transactions": "transactions_train.csv",
    "customers": "customers.csv",
    "articles": "articles.csv",
}


def _customer_sample_ids(con: duckdb.DuckDBPyConnection, frac: float, seed: int) -> list[str]:
    """Return a deterministic sample of customer IDs using a hash-based approach."""
    # Use a hash of the customer_id to get a deterministic sample
    # This avoids ordering issues and is reproducible regardless of row order
    result = con.execute(f"""
        SELECT DISTINCT customer_id
        FROM read_csv_auto('{DATA_DIR / SRC_FILES["customers"]}', header=true)
        WHERE mod(abs(hash(customer_id || '{seed}')), 10000) < {int(frac * 10000)}
    """).fetchall()
    return [r[0] for r in result]


def ingest(sample: bool = False) -> dict:
    """Convert CSVs to Parquet and write schema report."""

    # ── 0. Check source files exist ──────────────────────────────
    for name, fname in SRC_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"ERROR: {path} not found. See data/README.md for download steps.")
            sys.exit(1)

    con = duckdb.connect()

    # ── 1. Inspect raw schema (do NOT assume columns) ───────────
    schema_info = {}
    for name, fname in SRC_FILES.items():
        path = DATA_DIR / fname
        info = con.execute(f"""
            SELECT column_name, column_type
            FROM (DESCRIBE SELECT * FROM read_csv_auto('{path}', header=true))
        """).fetchall()
        row_count = con.execute(f"""
            SELECT count(*) FROM read_csv_auto('{path}', header=true)
        """).fetchone()[0]
        schema_info[name] = {
            "file": fname,
            "columns": {col: dtype for col, dtype in info},
            "row_count": row_count,
        }
        print(f"  {name}: {row_count:,} rows, {len(info)} columns")
        for col, dtype in info:
            print(f"    {col}: {dtype}")

    # ── 2. Transaction date range + distinct counts ─────────────
    txn_path = DATA_DIR / SRC_FILES["transactions"]
    date_stats = con.execute(f"""
        SELECT
            min(t_dat)          AS min_date,
            max(t_dat)          AS max_date,
            count(DISTINCT customer_id) AS n_customers,
            count(DISTINCT article_id)  AS n_articles
        FROM read_csv_auto('{txn_path}', header=true)
    """).fetchone()
    schema_info["transactions"]["min_date"] = str(date_stats[0])
    schema_info["transactions"]["max_date"] = str(date_stats[1])
    schema_info["transactions"]["distinct_customers"] = date_stats[2]
    schema_info["transactions"]["distinct_articles"] = date_stats[3]

    cust_path = DATA_DIR / SRC_FILES["customers"]
    n_cust_total = con.execute(f"""
        SELECT count(DISTINCT customer_id)
        FROM read_csv_auto('{cust_path}', header=true)
    """).fetchone()[0]
    schema_info["customers"]["distinct_customers"] = n_cust_total

    art_path = DATA_DIR / SRC_FILES["articles"]
    n_art_total = con.execute(f"""
        SELECT count(DISTINCT article_id)
        FROM read_csv_auto('{art_path}', header=true)
    """).fetchone()[0]
    schema_info["articles"]["distinct_articles"] = n_art_total

    # ── 3. Sampling ─────────────────────────────────────────────
    sample_frac = CONFIG["sample_fraction"]
    seed = CONFIG["random_seed"]

    if sample:
        print(f"\n  --sample mode: keeping {sample_frac*100:.0f}% of customers")
        sample_ids = _customer_sample_ids(con, sample_frac, seed)
        print(f"  Sampled {len(sample_ids):,} customers")
        schema_info["sample"] = {
            "enabled": True,
            "fraction": sample_frac,
            "n_customers": len(sample_ids),
        }
    else:
        sample_ids = None
        schema_info["sample"] = {"enabled": False}

    # ── 4. Write Parquet files ──────────────────────────────────
    # Customers
    if sample_ids:
        # Create a temp table with sample IDs for efficient filtering
        con.execute("CREATE TEMP TABLE sample_ids (customer_id VARCHAR)")
        con.executemany("INSERT INTO sample_ids VALUES (?)", [(cid,) for cid in sample_ids])

        con.execute(f"""
            COPY (
                SELECT c.*
                FROM read_csv_auto('{cust_path}', header=true) c
                INNER JOIN sample_ids s ON c.customer_id = s.customer_id
            ) TO '{DATA_DIR / "customers.parquet"}' (FORMAT PARQUET)
        """)
    else:
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto('{cust_path}', header=true)
            ) TO '{DATA_DIR / "customers.parquet"}' (FORMAT PARQUET)
        """)

    pq_cust_count = con.execute(f"""
        SELECT count(*) FROM '{DATA_DIR / "customers.parquet"}'
    """).fetchone()[0]
    print(f"  customers.parquet: {pq_cust_count:,} rows")

    # Articles (keep all — articles are small and needed for joins)
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{art_path}', header=true)
        ) TO '{DATA_DIR / "articles.parquet"}' (FORMAT PARQUET)
    """)
    pq_art_count = con.execute(f"""
        SELECT count(*) FROM '{DATA_DIR / "articles.parquet"}'
    """).fetchone()[0]
    print(f"  articles.parquet: {pq_art_count:,} rows")

    # Transactions (sorted by date; filtered to sample customers if applicable)
    if sample_ids:
        con.execute(f"""
            COPY (
                SELECT t.*
                FROM read_csv_auto('{txn_path}', header=true) t
                INNER JOIN sample_ids s ON t.customer_id = s.customer_id
                ORDER BY t.t_dat
            ) TO '{DATA_DIR / "transactions.parquet"}' (FORMAT PARQUET)
        """)
    else:
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{txn_path}', header=true)
                ORDER BY t_dat
            ) TO '{DATA_DIR / "transactions.parquet"}' (FORMAT PARQUET)
        """)

    pq_txn_count = con.execute(f"""
        SELECT count(*) FROM '{DATA_DIR / "transactions.parquet"}'
    """).fetchone()[0]
    print(f"  transactions.parquet: {pq_txn_count:,} rows")

    # ── 5. Reconcile counts ─────────────────────────────────────
    schema_info["parquet_counts"] = {
        "transactions": pq_txn_count,
        "customers": pq_cust_count,
        "articles": pq_art_count,
    }

    if not sample:
        # Full mode: Parquet row counts must match CSV
        for name in ["transactions", "customers", "articles"]:
            csv_count = schema_info[name]["row_count"]
            pq_count = schema_info["parquet_counts"][name]
            assert csv_count == pq_count, (
                f"{name}: CSV has {csv_count:,} rows but Parquet has {pq_count:,}"
            )
        print("\n  ✓ Row counts reconcile between CSV and Parquet")
    else:
        print("\n  ✓ Sample Parquet files written (counts reflect sampled data)")

    # ── 6. Write schema report ──────────────────────────────────
    schema_path = RESULTS_DIR / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema_info, f, indent=2, default=str)
    print(f"\n  Schema report → {schema_path}")

    con.close()
    return schema_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest H&M CSVs → Parquet")
    parser.add_argument("--sample", action="store_true", help="Use 5%% customer sample")
    args = parser.parse_args()

    print("\n══ M1: Ingest ══")
    ingest(sample=args.sample)
    print("══ Done ══\n")
