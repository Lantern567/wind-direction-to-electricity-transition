"""Redraw Fig. 4 for the Advances in Applied Energy manuscript.

S1, S2 and S3 are presented as parallel, non-nested counterfactuals.  The old
"incremental value" panel is replaced by a direct corridor-versus-other
comparison so that the figure does not imply S1 ⊆ S2 ⊆ S3.  The map is
rendered with frykit and all labels are in English.
"""
from __future__ import annotations

from pathlib import Path

import frykit.plot as fplt
import frykit.shp as fshp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely


HERE = Path(__file__).resolve().parent
BLOCK = HERE.parent
REPO = BLOCK.parent
OUT = BLOCK / "output-new"

BLUE = "#2563EB"
ORANGE = "#D97706"
GREEN = "#059669"
INK = "#1F2937"
NEUTRAL = "#64748B"
LIGHT = "#CBD5E1"
PALE = "#E8EEF5"
SCENARIO_COLORS = {"S1": "#93C5D8", "S2": "#3E95C5", "S3": "#145DA0"}
CORRIDOR_COLORS = {
    "Vietnam": "#E66101",
    "China_strait": "#ECA82C",
    "Italy": "#8C4E99",
    "Denmark": "#3B8EA5",
    "other": "#A7B3C2",
}
SCENARIOS = ["S1", "S2", "S3"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        clip_on=False,
    )


def load_inputs():
    table = pd.read_csv(OUT / "wp7c_scenario_table.csv", encoding="utf-8-sig")
    summary = pd.read_csv(OUT / "wp7c_corridor_summary.csv", encoding="utf-8-sig")
    attribution = pd.read_csv(OUT / "wp7d_scenario_attribution.csv", encoding="utf-8-sig")
    grid = pd.read_csv(OUT / "wp5c_rfd_grid.csv", encoding="utf-8-sig")
    farms = pd.read_csv(
        REPO / "offshore-task0-HuTingxian" / "output" / "task0" / "farms_master.csv",
        encoding="utf-8-sig",
    ).set_index("farm_id")
    return table, summary, attribution, grid, farms


def ocean_grid(grid: pd.DataFrame) -> pd.DataFrame:
    data = grid.dropna(subset=["R"]).copy()
    ocean = fshp.get_ocean()
    mask = shapely.contains_xy(ocean, data["lon"].to_numpy(), data["lat"].to_numpy())
    return data.loc[mask]


def add_map(ax, main, grid, farms) -> None:
    extent = [-100, 160, 0, 70]
    fplt.set_map_ticks(ax, extent, xticks=np.arange(-90, 151, 60), yticks=[0, 30, 60])
    ax.grid(False)
    ax.set_facecolor("#ECF4F7")
    fplt.add_land(ax, facecolor="#F4F3EF", edgecolor="none", zorder=0)
    fplt.add_countries(ax, facecolor="none", edgecolor="#A1AAB8", linewidth=0.3, zorder=1)

    data = ocean_grid(grid)
    ax.scatter(
        data["lon"],
        data["lat"],
        c=data["R"],
        s=5.5,
        cmap="Greys",
        vmin=data["R"].quantile(0.05),
        vmax=data["R"].quantile(0.98),
        linewidths=0,
        alpha=0.35,
        zorder=2,
    )

    for corridor, group in main.groupby("corridor"):
        valid = group.loc[group["farm_id"].isin(farms.index)]
        if valid.empty:
            continue
        locs = farms.loc[valid["farm_id"].astype(int)]
        color = CORRIDOR_COLORS.get(corridor, NEUTRAL)
        size = np.clip(valid["capacity_MW"].to_numpy() / 20.0, 10, 110)
        label = "Other projects" if corridor == "other" else corridor.replace("_", " ")
        ax.scatter(
            locs["centroid_lon"],
            locs["centroid_lat"],
            s=size,
            color=color,
            alpha=0.78 if corridor == "other" else 0.92,
            edgecolor="white",
            linewidth=0.45,
            zorder=4 if corridor == "other" else 5,
            label=label,
        )
    ax.legend(
        loc="lower left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        handletextpad=0.3,
        columnspacing=0.7,
        borderpad=0.25,
    )
    ax.text(
        0.98,
        0.03,
        "Marker area scales with project capacity",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL,
    )
    panel_label(ax, "a", x=-0.07, y=1.05)


def interval_panel(ax, main) -> None:
    labels = ["S1\nOrientation only", "S2\nMatched-template\nre-layout", "S3\nBest of six\ntemplates"]
    for idx, scenario in enumerate(SCENARIOS):
        values = main[f"G_plan_{scenario}"].dropna().to_numpy()
        median = np.median(values)
        p05, p95 = np.percentile(values, [5, 95])
        color = SCENARIO_COLORS[scenario]
        ax.errorbar(
            idx,
            median,
            yerr=[[median - p05], [p95 - median]],
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.2,
            linewidth=1.4,
            capsize=3,
            zorder=3,
        )
        ax.text(idx + 0.08, median, f"{median:+.1f}%", va="center", fontsize=7.2, fontweight="bold", color=color)
    ax.axhline(0, color=INK, linewidth=0.65)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Generation difference vs built layout (%)")
    ax.grid(axis="y", color=LIGHT, linewidth=0.45, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "b")


def corridor_generation_panel(ax, summary) -> None:
    data = summary.loc[summary["corridor"] != "other"].sort_values("capacity_GW", ascending=False)
    x = np.arange(len(data))
    width = 0.24
    for idx, scenario in enumerate(SCENARIOS):
        ax.bar(
            x + (idx - 1) * width,
            data[f"dE_{scenario}_GWh"],
            width=width,
            color=SCENARIO_COLORS[scenario],
            edgecolor="white",
            linewidth=0.35,
            label=scenario,
        )
    ax.axhline(0, color=INK, linewidth=0.65)
    ax.set_xticks(
        x,
        [f"{row.corridor.replace('_', ' ')}\n({row.capacity_GW:.1f} GW)" for row in data.itertuples()],
    )
    ax.set_ylabel("Generation difference (GWh yr$^{-1}$)")
    ax.legend(frameon=False, ncol=3, loc="upper right", columnspacing=0.7, handlelength=1.1)
    ax.grid(axis="y", color=LIGHT, linewidth=0.45, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "c")


def paradigm_panel(ax, attribution) -> None:
    labels = {
        "S_A": "Aligned\n9.4D",
        "S_B0": "Dense\n3.3D",
        "S_B45": "Dense 45°\n3.3D",
        "S_C": "Phased\n5.0D",
        "S_D": "Compact\n4.0D",
        "S_E": "Wide\n11.8D",
    }
    data = attribution.set_index("scenario")
    x = np.arange(len(data))
    for idx, (scenario, row) in enumerate(data.iterrows()):
        color = BLUE if row.med >= 0 else ORANGE
        ax.bar(idx, row.med, width=0.58, color=color, alpha=0.88, edgecolor="white", linewidth=0.35)
        ax.errorbar(
            idx,
            row.med,
            yerr=[[row.med - row.p05], [row.p95 - row.med]],
            fmt="none",
            color=color,
            linewidth=1.0,
            capsize=2.5,
        )
        ax.text(
            idx,
            row.med + (1.2 if row.med >= 0 else -1.2),
            f"{row.med:+.1f}%",
            ha="center",
            va="bottom" if row.med >= 0 else "top",
            fontsize=6.9,
            fontweight="bold",
            color=color,
        )
    ax.axhline(0, color=INK, linewidth=0.65)
    ax.set_xticks(x, [labels[item] for item in data.index])
    ax.set_ylabel("Difference at each template's best orientation (%)")
    ax.set_ylim(-26, 33)
    ax.grid(axis="y", color=LIGHT, linewidth=0.45, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "d")


def spacing_panel(ax, attribution) -> None:
    order = ["S_B0", "S_D", "S_C", "S_A", "S_E"]
    data = attribution.set_index("scenario").loc[order]
    x = data["spacing_D"].to_numpy()
    y = data["med"].to_numpy()
    colors = [BLUE if value >= 0 else ORANGE for value in y]
    ax.plot(x, y, color=LIGHT, linewidth=1.3, zorder=1)
    ax.scatter(x, y, s=46, color=colors, edgecolor="white", linewidth=0.75, zorder=3)
    offsets = {
        "S_B0": (-0.35, 1.4),
        "S_D": (0.45, -2.1),
        "S_C": (0.0, 1.4),
        "S_A": (-0.35, 1.5),
        "S_E": (-0.55, -2.2),
    }
    for scenario, row in data.iterrows():
        dx, dy = offsets[scenario]
        ax.text(
            row.spacing_D + dx,
            row.med + dy,
            f"{scenario}\n{row.spacing_D:.1f}D",
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=6.4,
        )
    ax.axhline(0, color=INK, linewidth=0.65)
    ax.set_xlabel("Minimum turbine spacing (D)")
    ax.set_ylabel("Median generation difference (%)")
    ax.set_xlim(2.6, 12.7)
    ax.set_ylim(-26, 21)
    ax.grid(axis="y", color=LIGHT, linewidth=0.45, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "e")


def corridor_contrast_panel(ax, main) -> None:
    corridor = main["corridor"] != "other"
    x = np.arange(3)
    for mask, offset, color, marker, label in [
        (corridor, -0.10, ORANGE, "o", f"Corridor projects (n={corridor.sum()})"),
        (~corridor, 0.10, BLUE, "s", f"Other projects (n={(~corridor).sum()})"),
    ]:
        medians = []
        lows = []
        highs = []
        for scenario in SCENARIOS:
            values = main.loc[mask, f"G_plan_{scenario}"].dropna().to_numpy()
            median = np.median(values)
            q25, q75 = np.percentile(values, [25, 75])
            medians.append(median)
            lows.append(median - q25)
            highs.append(q75 - median)
        ax.errorbar(
            x + offset,
            medians,
            yerr=np.vstack([lows, highs]),
            fmt=marker + "-",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.0,
            linewidth=1.25,
            capsize=2.2,
            label=label,
        )
    ax.axhline(0, color=INK, linewidth=0.65)
    ax.set_xticks(x, SCENARIOS)
    ax.set_ylabel("Median difference vs built layout (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=LIGHT, linewidth=0.45, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "f")


def main() -> None:
    apply_style()
    table, summary, attribution, grid, farms = load_inputs()
    projects = table.loc[table["n_turb"] >= 10].copy()

    fig = plt.figure(figsize=(12.8, 8.6))
    grid_spec = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.35)
    axes = [fig.add_subplot(grid_spec[row, col]) for row in range(2) for col in range(3)]

    add_map(axes[0], projects, grid, farms)
    interval_panel(axes[1], projects)
    corridor_generation_panel(axes[2], summary)
    paradigm_panel(axes[3], attribution)
    spacing_panel(axes[4], attribution)
    corridor_contrast_panel(axes[5], projects)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.955, bottom=0.085)
    png = HERE / "Fig4d_ADAPEN.png"
    pdf = HERE / "Fig4d_ADAPEN.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
