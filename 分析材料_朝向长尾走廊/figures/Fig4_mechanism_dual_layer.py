"""Publication figure for the wake-to-generation mechanism in §2.3."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "补算" / "output"

farms = pd.read_csv(OUT / "mechanism_dual_layer_farms.csv")
turbines = pd.read_csv(OUT / "mechanism_dual_layer_turbines.csv")
edges = pd.read_csv(OUT / "mechanism_dual_layer_edges.csv")
rose_speed = pd.read_csv(OUT / "mechanism_dual_layer_realrose_speed.csv")

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#2563EB"
ORANGE = "#F59E0B"
RED = "#DC2626"
GREEN = "#059669"
GREY = "#94A3B8"
DARK = "#334155"
LIGHT = "#E2E8F0"

fig = plt.figure(figsize=(7.15, 8.45), constrained_layout=True)
grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95])
axes = [
    fig.add_subplot(grid[0, 0]),
    fig.add_subplot(grid[0, 1]),
    fig.add_subplot(grid[1, 0]),
    fig.add_subplot(grid[1, 1]),
    fig.add_subplot(grid[2, 0]),
    fig.add_subplot(grid[2, 1]),
]


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT, linewidth=0.55, zorder=0)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


# a-b: turbine-level wake states for F126
farm_id = 126
farm_record = farms.loc[farms["farm_id"] == farm_id].iloc[0]
state_limits = []
for state in ("bad", "good"):
    state_turbines = turbines[
        (turbines["farm_id"] == farm_id) & (turbines["state"] == state)
    ]
    state_limits.extend(
        [
            state_turbines["alongwind_D"].abs().max(),
            state_turbines["crosswind_D"].abs().max(),
        ]
    )
limit = float(max(state_limits) * 1.08)
norm = Normalize(vmin=0, vmax=75)
cmap = mpl.colormaps["YlOrRd"]

for ax, state, label, letter in zip(
    axes[:2],
    ("bad", "good"),
    ("Unfavourable inflow", "Favourable inflow"),
    ("a", "b"),
):
    state_turbines = turbines[
        (turbines["farm_id"] == farm_id) & (turbines["state"] == state)
    ]
    state_edges = edges[
        (edges["farm_id"] == farm_id) & (edges["state"] == state)
    ]
    for _, edge in state_edges.iterrows():
        ax.plot(
            [edge["upstream_alongwind_D"], edge["downstream_alongwind_D"]],
            [edge["upstream_crosswind_D"], edge["downstream_crosswind_D"]],
            color=DARK,
            alpha=0.16,
            linewidth=0.45
            + 0.025 * min(edge["pair_velocity_deficit_pct"], 20),
            zorder=1,
        )
    scatter = ax.scatter(
        state_turbines["alongwind_D"],
        state_turbines["crosswind_D"],
        c=state_turbines["velocity_deficit_pct"],
        cmap=cmap,
        norm=norm,
        s=22,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    p90 = state_turbines["velocity_deficit_pct"].quantile(0.9)
    multi = (state_turbines["effective_wake_count"] >= 2).mean() * 100
    efficiency = (
        state_turbines["power_kw"].sum()
        / (len(state_turbines) * 6331.0)
        * 100
    )
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.87,
        f"P90 deficit {p90:.1f}%\n"
        f"Multi-wake turbines {multi:.1f}%\n"
        f"Farm efficiency {efficiency:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=DARK,
        linespacing=1.3,
    )
    ax.annotate(
        "wind",
        xy=(0.95, 0.08),
        xytext=(0.72, 0.08),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0),
        color=BLUE,
        ha="center",
        va="center",
        fontsize=6.2,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Along-wind position (rotor diameters)")
    ax.set_ylabel("Cross-wind position (rotor diameters)")
    clean_axis(ax)
    panel_letter(ax, letter)

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes[1],
    orientation="vertical",
    fraction=0.046,
    pad=0.025,
    aspect=28,
)
cbar.set_label("Velocity deficit at 9 m s$^{-1}$ (%)")
cbar.outline.set_linewidth(0.5)


# c: all-farm turbine-level wake contrast
ax = axes[2]
sample = farms[["A", "delta_p90_deficit_pct", "farm_id"]].dropna()
headline_ids = {57, 66, 91, 126, 155, 157, 159}
headline_mask = sample["farm_id"].isin(headline_ids)
ax.scatter(
    sample.loc[~headline_mask, "delta_p90_deficit_pct"],
    sample.loc[~headline_mask, "A"],
    s=13,
    color=GREY,
    alpha=0.62,
    edgecolor="none",
    zorder=2,
)
ax.scatter(
    sample.loc[headline_mask, "delta_p90_deficit_pct"],
    sample.loc[headline_mask, "A"],
    s=31,
    facecolor="white",
    edgecolor=RED,
    linewidth=1.2,
    zorder=4,
)
rho = spearmanr(sample["delta_p90_deficit_pct"], sample["A"])
ax.text(
    0.04,
    0.94,
    f"Spearman $\\rho$ = {rho.statistic:.3f}\n"
    f"$p$ = {rho.pvalue:.1e}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color=DARK,
)
ax.axhline(5.2, color=DARK, linestyle=":", linewidth=0.8)
ax.text(
    0.98,
    5.2,
    "5.2% noise floor",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=5.9,
    color=DARK,
)
ax.set_yscale("log")
ax.set_xlabel("Reduction in P90 turbine velocity deficit (percentage points)")
ax.set_ylabel("Recoverable AEP share, $A$ (%)")
clean_axis(ax, "both")
panel_letter(ax, "c")


# d: speed-resolved generation recovery for representative high-value farms
ax = axes[3]
line_specs = [
    (57, RED, "F57 Vietnam"),
    (126, BLUE, "F126 Vietnam"),
    (155, GREEN, "F155 Italy"),
]
for field_id, color, label in line_specs:
    field = rose_speed[rose_speed["farm_id"] == field_id]
    ax.plot(
        field["ws_bin_ms"],
        field["recovery_contribution_pct"],
        color=color,
        linewidth=1.6,
        marker="o",
        markersize=2.5,
        label=label,
    )
ax.axvspan(3, 10.5, color=ORANGE, alpha=0.10, linewidth=0)
ax.axvline(11, color=DARK, linestyle="--", linewidth=0.8)
ax.text(
    10.75,
    0.96,
    "rated onset",
    transform=ax.get_xaxis_transform(),
    rotation=90,
    ha="right",
    va="top",
    fontsize=5.8,
    color=DARK,
)
ax.set_xlim(3, 15)
ax.set_xlabel("Free-stream wind speed (m s$^{-1}$)")
ax.set_ylabel("Contribution to recoverable annual energy (% per 1 m s$^{-1}$ bin)")
ax.legend(frameon=False, loc="upper right")
clean_axis(ax, "both")
panel_letter(ax, "d")


# e: energy-conversion window for all seven audited fields
ax = axes[4]
audited = farms[farms["realrose_gain_pct_of_mean"].notna()].sort_values(
    "realrose_gain_pct_of_mean"
)
below_rated = (
    audited["realrose_recovery_share_low_3_6_pct"]
    + audited["realrose_recovery_share_partial_7_10_pct"]
)
near_rated = audited["realrose_recovery_share_near_rated_11_14_pct"]
free_below_rated = (
    audited["free_energy_share_low_3_6_pct"]
    + audited["free_energy_share_partial_7_10_pct"]
)
y = np.arange(len(audited))
ax.barh(
    y,
    below_rated,
    color=ORANGE,
    height=0.67,
    label="Recovered energy at 3–10 m s$^{-1}$",
)
ax.barh(
    y,
    near_rated,
    left=below_rated,
    color=BLUE,
    height=0.67,
    label="Recovered energy at 11–14 m s$^{-1}$",
)
ax.scatter(
    free_below_rated,
    y,
    marker="|",
    s=70,
    linewidth=1.5,
    color=DARK,
    label="Below-rated share of free AEP",
    zorder=4,
)
ax.set_yticks(y, [f"F{int(value)}" for value in audited["farm_id"]])
ax.set_xlim(0, 100)
ax.set_xlabel("Share of energy (%)")
ax.set_ylabel("Audited farm")
ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=1,
    handlelength=1.4,
)
clean_axis(ax, "x")
panel_letter(ax, "e")


# f: forward mechanism prediction against FLORIS + ERA5
ax = axes[5]
audited = audited.copy()
x = audited["A"]
y_pred = audited["realrose_gain_pct_of_mean"]
limit_xy = max(float(x.max()), float(y_pred.max())) * 1.08
ax.plot([0, limit_xy], [0, limit_xy], color=GREY, linestyle="--", linewidth=0.9)
ax.scatter(
    x,
    y_pred,
    s=30,
    color=RED,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)
for _, item in audited.iterrows():
    ax.annotate(
        f"F{int(item.farm_id)}",
        (item["A"], item["realrose_gain_pct_of_mean"]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=5.9,
        color=DARK,
    )
rho7 = spearmanr(x, y_pred)
mae = float(np.mean(np.abs(x - y_pred)))
ax.text(
    0.05,
    0.94,
    f"Spearman $\\rho$ = {rho7.statistic:.2f}\nMAE = {mae:.2f} pp",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color=DARK,
)
ax.set_xlim(0, limit_xy)
ax.set_ylim(0, limit_xy)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("FLORIS + ERA5 recoverable AEP share (%)")
ax.set_ylabel("Forward mechanism estimate (%)")
clean_axis(ax, "both")
panel_letter(ax, "f")

png_path = HERE / "Fig4_mechanism_dual_layer.png"
pdf_path = HERE / "Fig4_mechanism_dual_layer.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved: {png_path}")
print(f"saved: {pdf_path}")
