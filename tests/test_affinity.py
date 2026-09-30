"""Tests for M7: Affinity logic."""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from affinity import compute_affinity

def test_lift_calculation():
    """
    Hand-checked example:
    Basket 1: A, B
    Basket 2: A, C
    Basket 3: A, B, C
    Basket 4: D
    Basket 5: B
    
    Total baskets = 5.
    item_counts: A=3, B=3, C=2, D=1
    Pairs:
    (A, B): 2 baskets
    (A, C): 2 baskets
    (B, C): 1 basket
    
    Let's check (A, B):
    support(A) = 3/5 = 0.6
    support(B) = 3/5 = 0.6
    support(A, B) = 2/5 = 0.4
    
    confidence(A -> B) = support(A, B) / support(A) = 0.4 / 0.6 = 0.666...
    lift(A, B) = confidence(A -> B) / support(B) = (2/3) / 0.6 = 1.111...
    """
    
    baskets = pd.DataFrame([
        {"basket_id": 1, "item": "A"},
        {"basket_id": 1, "item": "B"},
        {"basket_id": 2, "item": "A"},
        {"basket_id": 2, "item": "C"},
        {"basket_id": 3, "item": "A"},
        {"basket_id": 3, "item": "B"},
        {"basket_id": 3, "item": "C"},
        {"basket_id": 4, "item": "D"},
        {"basket_id": 5, "item": "B"},
    ])
    
    df = compute_affinity(baskets, min_support=0.0)
    assert not df.empty
    
    # Get A and B pair
    ab = df[((df["item_A"] == "A") & (df["item_B"] == "B")) | ((df["item_A"] == "B") & (df["item_B"] == "A"))].iloc[0]
    
    assert ab["pair_count"] == 2
    assert abs(ab["support_pct"] - 40.0) < 1e-6
    assert abs(ab["confidence_A_to_B_pct"] - 66.666666) < 1e-4
    assert abs(ab["lift"] - 1.111111) < 1e-4
