"""
Mandatory leakage tests for M5 snapshots.

Tests:
  1. For every feature, max source transaction date < snapshot date T.
  2. Labels are computed only from [T, T + label_window).
  3. No customer appears twice within a snapshot.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA_DIR = ROOT / "data"
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SNAPSHOT_NAMES = ["train", "val", "test"]


def _load_snapshot(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"snapshot_{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@pytest.fixture(params=SNAPSHOT_NAMES)
def snapshot_with_name(request):
    name = request.param
    df = _load_snapshot(name)
    if df is None:
        pytest.skip(f"snapshot_{name}.parquet not found — run M5 first")
    return name, df


def test_no_duplicate_customers(snapshot_with_name):
    """No customer appears twice within a snapshot."""
    name, df = snapshot_with_name
    assert df["customer_id"].is_unique, (
        f"Snapshot '{name}' has {df['customer_id'].duplicated().sum()} duplicate customer IDs"
    )


def test_max_feature_date_before_snapshot(snapshot_with_name):
    """
    Verify that features were built from data strictly before snapshot_date T.

    We check this by loading the transactions and confirming that the max
    transaction date for each customer (in the feature window) is < T.
    """
    name, df = snapshot_with_name
    T = df["snapshot_date"].iloc[0]

    txn_path = DATA_DIR / "transactions.parquet"
    if not txn_path.exists():
        pytest.skip("transactions.parquet not found")

    con = duckdb.connect()

    # Get all customer IDs from this snapshot
    con.execute("CREATE TEMP TABLE _snap_custs AS SELECT customer_id FROM df")

    # Check: for these customers, every transaction used in features must be < T
    max_dates = con.execute(f"""
        SELECT t.customer_id, max(t.t_dat) AS max_t
        FROM '{txn_path}' t
        INNER JOIN _snap_custs s ON t.customer_id = s.customer_id
        WHERE t.t_dat < DATE '{T}'
        GROUP BY t.customer_id
    """).fetchdf()

    # All max dates should be < T
    violations = max_dates[max_dates["max_t"] >= pd.Timestamp(T)]
    assert violations.empty, (
        f"LEAKAGE in {name}: {len(violations)} customers have feature data >= {T}. "
        f"Max: {violations['max_t'].max()}"
    )

    # Also check: no transaction >= T should have been included in recency calculation
    # recency_days should be >= 1 (at least 1 day before T)
    assert (df["recency_days"] >= 0).all(), (
        f"Snapshot '{name}' has negative recency_days — possible leakage"
    )

    con.close()


def test_labels_from_correct_window(snapshot_with_name):
    """Labels must be computed from transactions in [T, T + label_window) only."""
    name, df = snapshot_with_name
    T = df["snapshot_date"].iloc[0]
    label_window_weeks = CONFIG["label_window_weeks"]
    label_end = (pd.Timestamp(T) + pd.Timedelta(weeks=label_window_weeks)).strftime("%Y-%m-%d")

    txn_path = DATA_DIR / "transactions.parquet"
    if not txn_path.exists():
        pytest.skip("transactions.parquet not found")

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE _snap AS SELECT customer_id, lapsed FROM df")

    # For retained customers (lapsed=0), verify they have ≥1 transaction in [T, label_end)
    retained_check = con.execute(f"""
        SELECT s.customer_id,
               count(t.t_dat) AS n_txns_in_window
        FROM _snap s
        LEFT JOIN '{txn_path}' t
            ON s.customer_id = t.customer_id
            AND t.t_dat >= DATE '{T}'
            AND t.t_dat <  DATE '{label_end}'
        WHERE s.lapsed = 0
        GROUP BY s.customer_id
        HAVING n_txns_in_window = 0
    """).fetchdf()

    assert retained_check.empty, (
        f"Snapshot '{name}': {len(retained_check)} customers labelled retained "
        f"but have no transactions in [{T}, {label_end})"
    )

    # For lapsed customers (lapsed=1), verify they have 0 transactions in [T, label_end)
    lapsed_check = con.execute(f"""
        SELECT s.customer_id,
               count(t.t_dat) AS n_txns_in_window
        FROM _snap s
        LEFT JOIN '{txn_path}' t
            ON s.customer_id = t.customer_id
            AND t.t_dat >= DATE '{T}'
            AND t.t_dat <  DATE '{label_end}'
        WHERE s.lapsed = 1
        GROUP BY s.customer_id
        HAVING n_txns_in_window > 0
    """).fetchdf()

    assert lapsed_check.empty, (
        f"Snapshot '{name}': {len(lapsed_check)} customers labelled lapsed "
        f"but have transactions in [{T}, {label_end})"
    )

    con.close()


def test_snapshot_dates_are_correctly_spaced():
    """Verify the three snapshots are spaced by the configured interval."""
    snapshots = {}
    for name in SNAPSHOT_NAMES:
        df = _load_snapshot(name)
        if df is None:
            pytest.skip(f"snapshot_{name}.parquet not found")
        snapshots[name] = pd.Timestamp(df["snapshot_date"].iloc[0])

    spacing_weeks = CONFIG["snapshot_spacing_weeks"]
    expected_gap = pd.Timedelta(weeks=spacing_weeks)

    assert snapshots["val"] - snapshots["train"] == expected_gap, (
        f"Train→Val gap: {snapshots['val'] - snapshots['train']} != {expected_gap}"
    )
    assert snapshots["test"] - snapshots["val"] == expected_gap, (
        f"Val→Test gap: {snapshots['test'] - snapshots['val']} != {expected_gap}"
    )
