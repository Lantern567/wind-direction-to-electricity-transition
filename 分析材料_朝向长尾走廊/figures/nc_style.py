"""Shared Nature-Communications-style matplotlib configuration.

Import this at the top of every figure script:

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from nc_style import PAL, apply_style, panel_label, savefig

All figures are English-only, restrained, colour-blind-safe.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour-blind-safe palette (fixed roles across the whole figure set)
PAL = {
    "baseline":  "#2563EB",   # real / observed / main baseline (blue)
    "highlight": "#DC2626",   # optimised / frontier / attention (red)
    "accent":    "#F59E0B",   # secondary comparison (amber)
    "positive":  "#059669",   # supporting positive series (green)
    "neutral":   "#64748B",   # context / other (slate)
    "light":     "#CBD5E1",   # faint context
    "ink":       "#1E293B",   # text
}

# Region / paradigm consistent colours
REGION_COL = {"europe": "#2563EB", "east_asia": "#DC2626", "us_east": "#059669"}
PARADIGM_COL = {"A": "#2563EB", "B": "#F59E0B", "C": "#059669", "D": "#8B5CF6", "E": "#DC2626"}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 350,
        "savefig.bbox": "tight",
        "axes.axisbelow": True,
        "mathtext.default": "regular",
    })


def panel_label(ax, letter, dx=-0.10, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def savefig(fig, path):
    fig.savefig(path, dpi=350, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved:", path)
