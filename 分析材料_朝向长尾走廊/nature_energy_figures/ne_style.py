"""Shared Nature Energy-style matplotlib configuration."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "blue": "#2166AC",
    "red": "#B2182B",
    "orange": "#EF8A62",
    "green": "#1B7837",
    "purple": "#762A83",
    "grey": "#6B7280",
    "light": "#D7DCE2",
    "pale": "#F3F4F6",
    "ink": "#1F2937",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.45,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stixsans",
        }
    )


def panel_label(ax, letter: str, x: float = -0.11, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["ink"],
    )


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")

