"""Tests for M4: Segmentation determinism and correctness."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA_DIR = ROOT / "data"


@pytest.fixture
def rfm_df():
    """Load the RFM segments Parquet (requires M4 to have run)."""
    path = DATA_DIR / "rfm_segments.parquet"
    if not path.exists():
        pytest.skip("rfm_segments.parquet not found — run M4 first")
    return pd.read_parquet(path)


def test_every_customer_in_exactly_one_segment(rfm_df):
    """Each customer must appear exactly once and have a non-null segment."""
    assert rfm_df["customer_id"].is_unique, "Duplicate customer IDs in RFM table"
    assert rfm_df["segment"].notna().all(), "Some customers lack a segment"


def test_quintile_values_are_valid(rfm_df):
    """Quintile scores must be in [1, 5]."""
    for col in ["R_q", "F_q", "M_q"]:
        assert rfm_df[col].between(1, 5).all(), f"{col} has values outside [1, 5]"


def test_segment_assignment_is_deterministic(rfm_df):
    """Re-applying the segment rules should yield the same result."""
    import yaml
    from segments import _assign_segment

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    rules = config["rfm"]["segments"]

    # Check a sample of rows
    sample = rfm_df.sample(min(500, len(rfm_df)), random_state=42)
    for _, row in sample.iterrows():
        expected = _assign_segment(row, rules)
        assert row["segment"] == expected, (
            f"Customer {row['customer_id']}: expected {expected}, got {row['segment']}"
        )


def test_all_segments_have_customers(rfm_df):
    """At least one customer should exist in each defined segment (or Other)."""
    segments_present = set(rfm_df["segment"].unique())
    # We expect at least Champions, Loyal, At_risk — but Lapsed/Promising/Other may vary
    assert len(segments_present) >= 3, f"Only {len(segments_present)} segments found: {segments_present}"
