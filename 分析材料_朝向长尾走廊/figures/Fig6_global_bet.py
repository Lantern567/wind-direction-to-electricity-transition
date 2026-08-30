"""Figure 5: transparent national-scale scenario arithmetic."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from nc_style import PAL, apply_style, panel_label, savefig


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if not (REPO / "补算").exists():
    REPO = REPO / "wind-direction-to-electricity-transition"
DATA = REPO / "补算" / "output" / "task8_global_bet.csv"


def main():
    apply_style()
    df = pd.read_csv(DATA)
    df = df[df["corridor_tier"].isin(["high", "mid"])].copy()
    df["risk_TWh"] = df["est_at_risk_GWh"] / 1000
    df = df.sort_values(["corridor_tier", "risk_TWh"], ascending=[True, False])

    colors = {"high": PAL["highlight"], "mid": PAL["accent"]}
    bar_colors = df["corridor_tier"].map(colors)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.7),
        sharey=True,
        gridspec_kw={"width_ratios": (1.0, 1.12), "wspace": 0.12},
    )
    y = range(len(df))
    ax0.barh(y, df["planned_GW"], color=bar_colors, height=0.66)
    ax1.barh(y, df["risk_TWh"], color=bar_colors, height=0.66)
    ax0.set_yticks(list(y), df["country"])
    ax0.invert_yaxis()

    ax0.set_xlabel("Planned offshore-wind capacity (GW)")
    ax1.set_xlabel("Scenario recoverable energy (TWh yr$^{-1}$)")
    ax0.set_xlim(0, df["planned_GW"].max() * 1.18)
    ax1.set_xlim(0, df["risk_TWh"].max() * 1.20)
    for ax in (ax0, ax1):
        ax.grid(axis="y", visible=False)

    for i, value in enumerate(df["planned_GW"]):
        ax0.text(value + 0.8, i, f"{value:.0f}", va="center", fontsize=7)
    for i, value in enumerate(df["risk_TWh"]):
        ax1.text(value + 0.10, i, f"{value:.1f}", va="center", fontsize=7)

    handles = [
        mpatches.Patch(color=colors["high"], label="High-sensitivity corridor"),
        mpatches.Patch(color=colors["mid"], label="Medium-sensitivity corridor"),
    ]
    ax1.legend(handles=handles, loc="lower right")

    total_gw = df["planned_GW"].sum()
    total_twh = df["risk_TWh"].sum()
    ax1.text(
        0.98,
        0.62,
        f"Corridor scenario total\n{total_gw:.0f} GW  |  {total_twh:.1f} TWh yr$^{{-1}}$",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=PAL["ink"],
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": PAL["light"], "lw": 0.7},
    )

    fig.text(
        0.5,
        0.015,
        "Scenario, not forecast: planned capacity × 8,760 h × 0.40 capacity factor × country mean recoverable share.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=PAL["neutral"],
    )
    panel_label(ax0, "a", dx=-0.34, dy=1.04)
    panel_label(ax1, "b", dx=-0.08, dy=1.04)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.96, bottom=0.14)

    savefig(fig, str(HERE / "Fig6_global_bet.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
