"""
Consistent chart style for all figures.

Every chart uses the same colour palette, font sizes and layout.
Action titles state the finding, generated from computed values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Palette ─────────────────────────────────────────────────────
COLOURS = {
    "primary": "#2563EB",       # blue-600
    "secondary": "#7C3AED",     # violet-600
    "accent": "#F59E0B",        # amber-500
    "danger": "#EF4444",        # red-500
    "success": "#10B981",       # emerald-500
    "muted": "#94A3B8",         # slate-400
    "dark": "#1E293B",          # slate-800
    "light": "#F8FAFC",         # slate-50
}

SEGMENT_COLOURS = {
    "Champions": "#10B981",
    "Loyal": "#2563EB",
    "Promising": "#7C3AED",
    "At_risk": "#F59E0B",
    "Lapsed": "#EF4444",
}

PALETTE_SEQ = ["#2563EB", "#7C3AED", "#F59E0B", "#EF4444", "#10B981",
               "#06B6D4", "#EC4899", "#8B5CF6", "#F97316", "#14B8A6"]

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def apply_style():
    """Apply the project's chart style globally."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLOURS["muted"],
        "axes.labelcolor": COLOURS["dark"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": COLOURS["dark"],
        "ytick.color": COLOURS["dark"],
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure and close it."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    → {path}")
    return path


def pct_fmt(x: float) -> str:
    """Format a percentage value."""
    return f"{x:.1f}%"


def comma_fmt(x: float) -> str:
    """Format a number with commas."""
    return f"{x:,.0f}"
