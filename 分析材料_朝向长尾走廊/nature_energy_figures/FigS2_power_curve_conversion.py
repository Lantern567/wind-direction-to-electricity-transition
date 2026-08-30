"""Power-curve conversion of orientation recovery into annual energy."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ne_style import COLORS, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = REPO / "补算" / "output"
HEADLINE = [57, 66, 91, 126, 155, 157, 159]
COUNTRY = {
    57: "Viet Nam",
    66: "China",
    91: "China",
    126: "Viet Nam",
    155: "Italy",
    157: "Denmark",
    159: "Viet Nam",
}


def main() -> None:
    apply_style()
    speed = pd.read_csv(OUTPUT / "mechanism_dual_layer_realrose_speed.csv")
    farms = pd.read_csv(OUTPUT / "mechanism_dual_layer_farms.csv")
    speed = speed.loc[speed["farm_id"].isin(HEADLINE)].copy()
    farms = farms.loc[farms["farm_id"].isin(HEADLINE)].copy()

    share_3_10 = (
        speed.loc[speed["ws_bin_ms"].between(3, 10)]
        .groupby("farm_id")["recovery_contribution_pct"]
        .sum()
    )
    if len(share_3_10) != len(HEADLINE):
        raise ValueError("All seven audited wind farms must have speed-bin results.")

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.12, 1]},
        constrained_layout=True,
    )

    ax_a.axvspan(3, 10, color=COLORS["orange"], alpha=0.10, lw=0, zorder=0)
    for farm_id in HEADLINE:
        block = speed.loc[speed["farm_id"].eq(farm_id)].sort_values("ws_bin_ms")
        ax_a.plot(
            block["ws_bin_ms"],
            block["recovery_contribution_pct"],
            color=COLORS["light"],
            lw=0.8,
            alpha=0.9,
            zorder=1,
        )

    highlighted = [57, 126, 157]
    for farm_id, colour in zip(
        highlighted,
        [COLORS["red"], COLORS["orange"], COLORS["blue"]],
    ):
        block = speed.loc[speed["farm_id"].eq(farm_id)].sort_values("ws_bin_ms")
        ax_a.plot(
            block["ws_bin_ms"],
            block["recovery_contribution_pct"],
            color=colour,
            lw=1.45,
            marker="o",
            ms=2.5,
            markevery=2,
            label=f"F{farm_id}, {COUNTRY[farm_id]}",
            zorder=3,
        )
    ax_a.text(
        0.31,
        0.97,
        "83.3–93.9% of recovery",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        color=COLORS["ink"],
    )
    ax_a.set_xlim(2, 25)
    ax_a.set_xticks([3, 5, 8, 10, 15, 20, 25])
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel("Hub-height wind speed (m s$^{-1}$)")
    ax_a.set_ylabel("Contribution to recoverable energy (%)")
    ax_a.legend(loc="upper right", handlelength=1.4, labelspacing=0.45)

    band_columns = [
        "realrose_recovery_share_low_3_6_pct",
        "realrose_recovery_share_partial_7_10_pct",
        "realrose_recovery_share_near_rated_11_14_pct",
        "realrose_recovery_share_saturated_15_25_pct",
    ]
    band_labels = ["3–6", "7–10", "11–14", "15–25 m s$^{-1}$"]
    band_colours = [COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["light"]]
    farms["share_3_10"] = farms["farm_id"].map(share_3_10)
    farms = farms.sort_values("share_3_10")
    values = farms[band_columns].clip(lower=0).to_numpy()
    values = values / values.sum(axis=1, keepdims=True) * 100
    y = np.arange(len(farms))
    left = np.zeros(len(farms))
    for j, (label, colour) in enumerate(zip(band_labels, band_colours)):
        ax_b.barh(
            y,
            values[:, j],
            left=left,
            height=0.68,
            color=colour,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        left += values[:, j]
    for ypos, total in zip(y, farms["share_3_10"]):
        ax_b.text(
            min(total - 1.0, 97.0),
            ypos,
            f"{total:.1f}%",
            ha="right",
            va="center",
            fontsize=6.2,
            color="white" if total < 92 else COLORS["ink"],
            fontweight="bold",
        )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(
        [f"F{int(fid)}  {COUNTRY[int(fid)]}" for fid in farms["farm_id"]]
    )
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Share of recoverable annual energy (%)")
    ax_b.grid(axis="x")
    ax_b.grid(axis="y", visible=False)
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.2,
    )

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    save_figure(fig, HERE / "FigS2_power_curve_conversion.png")
    plt.close(fig)
    print(
        "3–10 m/s recovery share: "
        f"min={share_3_10.min():.6f}%, max={share_3_10.max():.6f}%, n={len(share_3_10)}"
    )


if __name__ == "__main__":
    main()
