"""Spatial distribution and temporal persistence of high orientation recovery."""
from __future__ import annotations

from pathlib import Path

import frykit.plot as fplt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ne_style import COLORS, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DERIVED = REPO / "分析材料_几何主导杠杆" / "data_derived"


def prepare_map(ax, extent, xticks, yticks):
    fplt.set_map_ticks(ax, extent, xticks=xticks, yticks=yticks)
    ax.grid(False)
    ax.set_facecolor("#EDF4F7")
    fplt.add_land(ax, facecolor="#F2F1ED", edgecolor="none", zorder=0)
    fplt.add_countries(
        ax,
        facecolor="none",
        edgecolor="#AEB6BF",
        linewidth=0.35,
        zorder=1,
    )


def plot_farms(ax, data, threshold):
    regular = data.loc[data["gain"] <= 2]
    middle = data.loc[(data["gain"] > 2) & (data["gain"] <= threshold)]
    high = data.loc[data["gain"] > threshold]
    ax.scatter(
        regular["centroid_lon"],
        regular["centroid_lat"],
        s=9,
        color=COLORS["light"],
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    ax.scatter(
        middle["centroid_lon"],
        middle["centroid_lat"],
        s=24,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    ax.scatter(
        high["centroid_lon"],
        high["centroid_lat"],
        s=47,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )


def main() -> None:
    apply_style()
    annual_gain = pd.read_csv(DERIVED / "orientation_gain.csv")
    farm_gain = annual_gain.groupby("farm_id")["gain_pct"].mean().rename("gain")
    metadata = pd.read_csv(DERIVED / "farm_wakeloss_map.csv").set_index("farm_id")
    farms = metadata.join(farm_gain).dropna(subset=["gain"])

    annual_cf = pd.read_csv(
        REPO / "offshore-task2" / "output" / "task2_annual_floris.csv",
        encoding="utf-8-sig",
    )
    annual_cf = annual_cf.loc[annual_cf["wake_model"].eq("gauss")]
    farm_cv = annual_cf.groupby("farm_id")["CF"].agg(
        lambda values: values.std(ddof=0) / values.mean() * 100
        if len(values) >= 5
        else np.nan
    )
    threshold = float(farm_cv.median())

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    east_asia = farms.loc[farms["region"].eq("east_asia")]
    prepare_map(ax_a, [102, 125, 7, 35], [105, 115, 125], [10, 20, 30])
    plot_farms(ax_a, east_asia, threshold)
    annotations_a = [
        (57, "F57  18.2%", (20, 18)),
        (126, "F126  5.2%", (24, -13)),
        (159, "F159  5.2%", (22, 2)),
        (66, "F66  7.8%", (-58, 12)),
        (91, "F91  6.6%", (-52, -16)),
    ]
    for farm_id, label, offset in annotations_a:
        row = farms.loc[farm_id]
        ax_a.annotate(
            label,
            (row["centroid_lon"], row["centroid_lat"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.1,
            color=COLORS["ink"],
            arrowprops={"arrowstyle": "-", "color": COLORS["grey"], "lw": 0.5},
            zorder=5,
        )
    europe = farms.loc[farms["region"].eq("europe")]
    prepare_map(ax_b, [-12, 21, 35, 59], [-10, 0, 10, 20], [40, 50, 60])
    plot_farms(ax_b, europe, threshold)
    annotations_b = [
        (155, "F155  9.6%", (-67, -9)),
        (157, "F157  6.9%", (-72, -20)),
    ]
    for farm_id, label, offset in annotations_b:
        row = farms.loc[farm_id]
        ax_b.annotate(
            label,
            (row["centroid_lon"], row["centroid_lat"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.1,
            color=COLORS["ink"],
            arrowprops={"arrowstyle": "-", "color": COLORS["grey"], "lw": 0.5},
            zorder=5,
        )
    selected = [57, 66, 155, 157]
    line_colours = [COLORS["red"], COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    labels = ["F57, Vietnam", "F66, China", "F155, Italy", "F157, Denmark"]
    for farm_id, colour, label in zip(selected, line_colours, labels):
        block = annual_gain.loc[annual_gain["farm_id"].eq(farm_id)].sort_values("year")
        ax_c.plot(
            block["year"],
            block["gain_pct"],
            marker="o",
            ms=3,
            lw=1.25,
            color=colour,
            label=label,
        )
    ax_c.axhline(threshold, color=COLORS["ink"], ls="--", lw=0.8)
    ax_c.text(
        2024.3,
        threshold + 0.35,
        "Global median observed CF variability",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COLORS["ink"],
    )
    ax_c.set_xlim(2013.5, 2024.5)
    ax_c.set_ylim(-1, 24)
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Orientation recovery (%)")
    ax_c.legend(loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.4)

    legend = [
        Line2D([], [], marker="o", ls="", color=COLORS["red"], ms=5.5,
               label=f">{threshold:.2f}%"),
        Line2D([], [], marker="o", ls="", color=COLORS["orange"], ms=4.5,
               label=f"2–{threshold:.2f}%"),
        Line2D([], [], marker="o", ls="", color=COLORS["light"], ms=3.5,
               label="≤2%"),
    ]
    ax_a.legend(
        handles=legend,
        loc="lower right",
        handletextpad=0.3,
        labelspacing=0.3,
        borderaxespad=0.25,
    )

    panel_label(ax_a, "a", x=-0.09, y=1.04)
    panel_label(ax_b, "b", x=-0.09, y=1.04)
    panel_label(ax_c, "c", x=-0.055, y=1.06)
    save_figure(fig, HERE / "Fig2_spatial_concentration.png")
    plt.close(fig)
    print(f"threshold={threshold:.6f}, highlighted={int((farms['gain'] > threshold).sum())}")


if __name__ == "__main__":
    main()
