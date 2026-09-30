"""
M6 — Lapse prediction model.

Three models:
  (a) Recency-only rule baseline
  (b) Logistic regression (standardised)
  (c) XGBoost with class weighting

Train on T_train, tune/calibrate on T_val, evaluate once on T_test.

Usage:
    python src/model.py              # full dataset
    python src/model.py --sample     # sample mode
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting import apply_style, save_fig, COLOURS, FIGURES_DIR

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR = ROOT / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

FEATURE_COLS = None  # will be set dynamically


def _load_snapshot(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"snapshot_{name}.parquet"
    return pd.read_parquet(path)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get feature columns — everything except identifiers and target."""
    exclude = {"customer_id", "lapsed", "snapshot_date"}
    return [c for c in df.columns if c not in exclude]


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> dict:
    """Compute standard classification metrics."""
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def _lift_by_decile(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Compute lift and cumulative gain by decile."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], 10, labels=False, duplicates="drop") + 1
    # Decile 10 = highest predicted risk
    df["decile"] = df["decile"].max() + 1 - df["decile"]  # invert so 1 = highest risk

    summary = (
        df.groupby("decile")
        .agg(
            n_customers=("y_true", "count"),
            n_lapsed=("y_true", "sum"),
            avg_prob=("y_prob", "mean"),
        )
        .reset_index()
    )
    total_lapsed = y_true.sum()
    summary["lapse_rate_pct"] = (summary["n_lapsed"] / summary["n_customers"] * 100).round(1)
    summary["capture_pct"] = (summary["n_lapsed"] / total_lapsed * 100).round(1)
    summary["cum_capture_pct"] = summary["capture_pct"].cumsum().round(1)
    summary["lift"] = (summary["lapse_rate_pct"] / (total_lapsed / len(y_true) * 100)).round(2)

    return summary


def run_model(sample: bool = False) -> dict:
    """Train, calibrate, and evaluate lapse models."""

    seed = CONFIG["random_seed"]

    # ── Load snapshots ──────────────────────────────────────────
    train_df = _load_snapshot("train")
    val_df = _load_snapshot("val")
    test_df = _load_snapshot("test")

    feature_cols = _get_feature_cols(train_df)
    print(f"  Features ({len(feature_cols)}): {feature_cols[:10]}...")

    X_train = train_df[feature_cols].fillna(-1).values
    y_train = train_df["lapsed"].values
    X_val = val_df[feature_cols].fillna(-1).values
    y_val = val_df["lapsed"].values
    X_test = test_df[feature_cols].fillna(-1).values
    y_test = test_df["lapsed"].values

    print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.1f}% lapsed)")
    print(f"  Val:   {len(X_val):,} ({y_val.mean()*100:.1f}% lapsed)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.1f}% lapsed)")

    all_metrics = []

    # ══════════════════════════════════════════════════════════════
    # (a) Recency-only rule baseline
    # ══════════════════════════════════════════════════════════════
    recency_idx = feature_cols.index("recency_days")
    # Probability proxy: normalise recency to [0, 1] — higher recency → higher lapse risk
    recency_test = X_test[:, recency_idx]
    recency_prob = (recency_test - recency_test.min()) / (recency_test.max() - recency_test.min() + 1e-9)

    baseline_metrics = _compute_metrics(y_test, recency_prob, "Recency baseline")
    all_metrics.append(baseline_metrics)
    print(f"\n  (a) Recency baseline — ROC-AUC: {baseline_metrics['roc_auc']}, PR-AUC: {baseline_metrics['pr_auc']}")

    # ══════════════════════════════════════════════════════════════
    # (b) Logistic regression
    # ══════════════════════════════════════════════════════════════
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_C = None
    best_auc = -1
    for C in CONFIG["model"]["logistic_C"]:
        lr = LogisticRegression(C=C, max_iter=1000, random_state=seed, solver="lbfgs")
        lr.fit(X_train_s, y_train)
        val_prob = lr.predict_proba(X_val_s)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        if val_auc > best_auc:
            best_auc = val_auc
            best_C = C

    lr_final = LogisticRegression(C=best_C, max_iter=1000, random_state=seed, solver="lbfgs")
    lr_final.fit(X_train_s, y_train)
    lr_test_prob = lr_final.predict_proba(X_test_s)[:, 1]

    lr_metrics = _compute_metrics(y_test, lr_test_prob, "Logistic regression")
    lr_metrics["best_C"] = best_C
    all_metrics.append(lr_metrics)
    print(f"  (b) Logistic regression (C={best_C}) — ROC-AUC: {lr_metrics['roc_auc']}, PR-AUC: {lr_metrics['pr_auc']}")

    # ══════════════════════════════════════════════════════════════
    # (c) XGBoost
    # ══════════════════════════════════════════════════════════════
    param_grid = CONFIG["model"]["xgb_param_grid"]

    best_params = None
    best_val_auc = -1

    # Small grid search on validation set
    from itertools import product as iter_product
    param_combos = list(iter_product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["n_estimators"],
        param_grid["scale_pos_weight"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
    ))

    print(f"\n  XGBoost grid search: {len(param_combos)} combinations...")

    for md, lr_rate, n_est, spw, ss, csbt in param_combos:
        model = xgb.XGBClassifier(
            max_depth=md, learning_rate=lr_rate, n_estimators=n_est,
            scale_pos_weight=spw, subsample=ss, colsample_bytree=csbt,
            random_state=seed, use_label_encoder=False, eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = {
                "max_depth": md, "learning_rate": lr_rate, "n_estimators": n_est,
                "scale_pos_weight": spw, "subsample": ss, "colsample_bytree": csbt,
            }

    print(f"  Best XGBoost params: {best_params} (val AUC: {best_val_auc:.4f})")

    # Retrain with best params
    xgb_model = xgb.XGBClassifier(
        **best_params, random_state=seed, use_label_encoder=False,
        eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    # Uncalibrated predictions on test
    xgb_test_prob_raw = xgb_model.predict_proba(X_test)[:, 1]
    xgb_raw_metrics = _compute_metrics(y_test, xgb_test_prob_raw, "XGBoost (uncalibrated)")
    xgb_raw_metrics["best_params"] = best_params

    # ── Calibrate on validation set ─────────────────────────────
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    from sklearn.isotonic import IsotonicRegression
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_reg.fit(xgb_val_prob, y_val)

    xgb_test_prob_cal = iso_reg.transform(xgb_test_prob_raw)
    xgb_cal_metrics = _compute_metrics(y_test, xgb_test_prob_cal, "XGBoost (calibrated)")
    xgb_cal_metrics["best_params"] = best_params

    all_metrics.append(xgb_raw_metrics)
    all_metrics.append(xgb_cal_metrics)

    print(f"  (c) XGBoost uncalibrated — ROC-AUC: {xgb_raw_metrics['roc_auc']}, Brier: {xgb_raw_metrics['brier']}")
    print(f"      XGBoost calibrated   — ROC-AUC: {xgb_cal_metrics['roc_auc']}, Brier: {xgb_cal_metrics['brier']}")

    # ══════════════════════════════════════════════════════════════
    # Lift / cumulative gain (calibrated XGBoost)
    # ══════════════════════════════════════════════════════════════
    lift_df = _lift_by_decile(y_test, xgb_test_prob_cal)
    lift_df.to_csv(TABLES_DIR / "lift_by_decile.csv", index=False)

    top_decile_capture = float(lift_df.iloc[0]["capture_pct"])
    print(f"\n  Top decile captures {top_decile_capture:.1f}% of all lapsers")

    # ══════════════════════════════════════════════════════════════
    # Charts
    # ══════════════════════════════════════════════════════════════

    # 1. Calibration curve comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, probs, colour in [
        ("XGBoost raw", xgb_test_prob_raw, COLOURS["accent"]),
        ("XGBoost calibrated", xgb_test_prob_cal, COLOURS["primary"]),
        ("Logistic", lr_test_prob, COLOURS["secondary"]),
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", markersize=4, label=label, color=colour, linewidth=1.5)
    ax.plot([0, 1], [0, 1], "--", color=COLOURS["muted"], linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(
        f"Calibration improves Brier score from {xgb_raw_metrics['brier']:.3f} to {xgb_cal_metrics['brier']:.3f}",
        fontsize=11,
    )
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "calibration_curve")

    # 2. Lift chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(lift_df["decile"], lift_df["capture_pct"], color=COLOURS["primary"], alpha=0.85)
    ax.plot(lift_df["decile"], lift_df["cum_capture_pct"], marker="o", color=COLOURS["danger"],
            linewidth=2, label="Cumulative capture")
    ax.set_xlabel("Decile (1 = highest risk)")
    ax.set_ylabel("% of lapsers captured")
    ax.set_title(
        f"Top decile captures {top_decile_capture:.0f}% of lapsers; "
        f"top 3 deciles capture {lift_df.iloc[:3]['capture_pct'].sum():.0f}%",
        fontsize=11,
    )
    ax.legend()
    ax.set_xticks(lift_df["decile"])
    fig.tight_layout()
    save_fig(fig, "lift_chart")

    # 3. Model comparison table
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    print(f"\n  Model comparison:")
    print(comparison_df[["model", "roc_auc", "pr_auc", "brier"]].to_string(index=False))

    # 4. SHAP analysis
    print("\n  Computing SHAP values...")
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, max_display=15)
        plt.title("SHAP feature importance (association, not causation)", fontsize=11)
        plt.tight_layout()
        save_fig(plt.gcf(), "shap_summary")

        # Top drivers
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(TABLES_DIR / "shap_importance.csv", index=False)

        top_drivers = list(shap_importance.head(5)["feature"])
        print(f"  Top SHAP drivers: {top_drivers}")
        print("  NOTE: SHAP shows feature importance, NOT causal effects.")
    except Exception as e:
        print(f"  SHAP failed: {e}")
        top_drivers = []

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    results = {
        "models": all_metrics,
        "best_model": "XGBoost (calibrated)",
        "best_params": best_params,
        "top_decile_capture_pct": top_decile_capture,
        "top_3_decile_capture_pct": round(float(lift_df.iloc[:3]["capture_pct"].sum()), 1),
        "test_snapshot_used_once": True,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "shap_top_drivers": top_drivers,
        "lift_by_decile": lift_df.to_dict(orient="records"),
    }

    results_path = RESULTS_DIR / "model_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → {results_path}")

    # Save calibrated test probabilities for M8
    test_df_out = test_df[["customer_id"]].copy()
    test_df_out["lapsed"] = y_test
    test_df_out["prob_lapse"] = xgb_test_prob_cal
    test_df_out["risk_decile"] = pd.qcut(
        xgb_test_prob_cal, 10, labels=False, duplicates="drop"
    )
    # Invert so decile 1 = highest risk
    test_df_out["risk_decile"] = test_df_out["risk_decile"].max() - test_df_out["risk_decile"] + 1
    test_df_out.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lapse prediction model")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    print("\n══ M6: Lapse Model ══")
    run_model(sample=args.sample)
    print("══ Done ══\n")
