"""Redraw Fig. 3 for the Advances in Applied Energy manuscript.

The figure separates two questions that were conflated in the previous draft:
(i) whether corridor farms are genuinely more orientation-sensitive under
controlled construction paradigms, and (ii) how accurately an unseen corridor
can be identified. Maps are rendered with frykit and all labels are in English.
"""
from __future__ import annotations

from pathlib import Path

import frykit.plot as fplt
import frykit.shp as fshp
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parent
BLOCK = HERE.parent
REPO = BLOCK.parent
OUT = BLOCK / "output-new"

PARADIGMS = ["S_A", "S_B0", "S_B45", "S_C", "S_D", "S_E"]
PARADIGM_LABELS = [
    "Aligned\n9.4D",
    "Dense\n3.3D",
    "Dense 45°\n3.3D",
    "Phased\n5.0D",
    "Compact\n4.0D",
    "Wide\n11.8D",
]
CORRIDORS = {
    "Vietnam": [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    "China strait": [12, 66, 85, 91, 92, 97, 103, 105],
    "Adriatic–Taranto": [155],
    "Danish straits": [157],
}
CORRIDOR_IDS = {farm_id for ids in CORRIDORS.values() for farm_id in ids}

BLUE = "#2563EB"
ORANGE = "#D97706"
RED = "#C2410C"
GREEN = "#059669"
INK = "#1F2937"
NEUTRAL = "#64748B"
LIGHT = "#CBD5E1"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
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


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def load_inputs():
    cross = np.load(OUT / "wp5c_cross_farms.npz")
    loo = np.load(OUT / "wp6c_loo.npz")
    real = np.load(OUT / "wp7a_real_curves.npz")
    grid = pd.read_csv(OUT / "wp5c_rfd_grid.csv", encoding="utf-8-sig")
    farm_geo = pd.read_csv(
        REPO / "offshore-task0-HuTingxian" / "output" / "task0" / "farms_master.csv",
        encoding="utf-8-sig",
    ).set_index("farm_id")
    return cross, loo, real, grid, farm_geo


def add_sensitivity_panel(ax: plt.Axes, cross) -> None:
    values = cross["A"]
    farm_ids = cross["farm_ids"].astype(int)
    corridor = np.array([farm_id in CORRIDOR_IDS for farm_id in farm_ids])
    x = np.arange(values.shape[1])

    for mask, offset, color, marker, label in [
        (corridor, -0.10, ORANGE, "o", f"Corridor farms (n={corridor.sum()})"),
        (~corridor, 0.10, BLUE, "s", f"Other farms (n={(~corridor).sum()})"),
    ]:
        med = np.nanmedian(values[mask], axis=0)
        q25, q75 = np.nanpercentile(values[mask], [25, 75], axis=0)
        ax.errorbar(
            x + offset,
            med,
            yerr=np.vstack([med - q25, q75 - med]),
            color=color,
            marker=marker,
            markersize=4.7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.35,
            linestyle="none",
            elinewidth=1.0,
            capsize=2.2,
            label=label,
            zorder=3,
        )

    ratios = np.nanmedian(values[corridor], axis=0) / np.nanmedian(values[~corridor], axis=0)
    for idx, ratio in enumerate(ratios):
        ax.text(idx, 8.0, f"{ratio:.1f}×", ha="center", va="bottom", fontsize=7.2, color=RED)

    ax.set_yscale("log")
    ax.set_ylim(0.08, 11.5)
    ax.set_xticks(x, PARADIGM_LABELS)
    ax.set_ylabel("Orientation-response amplitude, A (%)")
    ax.grid(axis="y", which="both", color=LIGHT, linewidth=0.45, alpha=0.65)
    ax.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.995,
        0.03,
        "Points show medians; bars show interquartile ranges",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=NEUTRAL,
    )
    panel_label(ax, "a", x=-0.055, y=1.08)


def ocean_grid(grid: pd.DataFrame) -> pd.DataFrame:
    data = grid.dropna(subset=["F75"]).copy()
    ocean = fshp.get_ocean()
    mask = shapely.contains_xy(ocean, data["lon"].to_numpy(), data["lat"].to_numpy())
    return data.loc[mask].copy()


def prepare_map(ax: plt.Axes, extent, xticks, yticks) -> None:
    fplt.set_map_ticks(ax, extent, xticks=xticks, yticks=yticks)
    ax.grid(False)
    ax.set_facecolor("#ECF4F7")
    fplt.add_land(ax, facecolor="#F4F3EF", edgecolor="none", zorder=0)
    fplt.add_countries(ax, facecolor="none", edgecolor="#9CA3AF", linewidth=0.35, zorder=1)


def add_map_panel(ax: plt.Axes, data, farm_geo, extent, xticks, yticks, region_label):
    prepare_map(ax, extent, xticks, yticks)
    sub = data.loc[
        data["lon"].between(extent[0], extent[1])
        & data["lat"].between(extent[2], extent[3])
    ]
    counts = np.rint(sub["F75"].to_numpy() * 6.0 / 100.0).astype(int)
    cmap = mpl.colormaps["viridis"].resampled(7)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 7.5, 1.0), cmap.N)
    sc = ax.scatter(
        sub["lon"],
        sub["lat"],
        c=counts,
        cmap=cmap,
        norm=norm,
        marker="s",
        s=13,
        linewidths=0,
        alpha=0.92,
        zorder=2,
    )
    ids = [farm_id for farm_id in CORRIDOR_IDS if farm_id in farm_geo.index]
    farms = farm_geo.loc[ids]
    farms = farms.loc[
        farms["centroid_lon"].between(extent[0], extent[1])
        & farms["centroid_lat"].between(extent[2], extent[3])
    ]
    ax.scatter(
        farms["centroid_lon"],
        farms["centroid_lat"],
        marker="*",
        s=55,
        facecolor=RED,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax.text(
        0.02,
        0.98,
        region_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=5,
    )
    return sc


def add_type_match_panel(ax: plt.Axes, cross, real) -> None:
    farm_ids = cross["farm_ids"].astype(int)
    real_series = pd.Series(real["A"], index=real["farm_ids"].astype(int))
    common = np.array([farm_id in real_series.index for farm_id in farm_ids])
    ids = farm_ids[common]
    x = cross["A"][common].mean(axis=1)
    y = real_series.loc[ids].to_numpy()
    corridor = np.array([farm_id in CORRIDOR_IDS for farm_id in ids])

    ax.scatter(
        x[~corridor],
        y[~corridor],
        s=17,
        color=NEUTRAL,
        alpha=0.48,
        edgecolor="none",
        label=f"Other farms (n={(~corridor).sum()})",
        zorder=2,
    )
    ax.scatter(
        x[corridor],
        y[corridor],
        s=28,
        color=ORANGE,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
        label=f"Corridor farms (n={corridor.sum()})",
        zorder=3,
    )
    lo, hi = 0.035, 30
    ax.plot([lo, hi], [lo, hi], color=LIGHT, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.12, 6)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean response across six paradigms, A (%)")
    ax.set_ylabel("Real-layout response, A (%)")
    ax.grid(which="both", color=LIGHT, linewidth=0.4, alpha=0.55)
    rho = spearmanr(x, y).statistic
    ax.text(
        0.04,
        0.95,
        f"Spearman $\\rho$ = {rho:.3f}\n$n$ = {len(x)} farms",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
    )
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "c", x=-0.13, y=1.04)


def auc_from_ranks(y_true: np.ndarray, score: np.ndarray) -> float:
    ranks = rankdata(score, method="average")
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_points(y_true: np.ndarray, score: np.ndarray):
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = np.r_[0.0, tp / y.sum()]
    fpr = np.r_[0.0, fp / (~y).sum()]
    return fpr, tpr


def screening_metrics(y_true: np.ndarray, score: np.ndarray):
    n_select = int(y_true.sum())
    selected = np.argsort(-score, kind="mergesort")[:n_select]
    y_pred = np.zeros(len(y_true), dtype=bool)
    y_pred[selected] = True
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    return balanced_accuracy, recall, fp / (fp + tn)


def add_validation_panel(ax: plt.Axes, loo) -> None:
    series = [
        (
            "Paradigm-mean target",
            loo["lab_para"] >= float(loo["thr_para"]),
            loo["pred_para"],
            BLUE,
            "-",
        ),
        (
            "Real-layout target",
            loo["lab_real"] >= float(loo["thr_real"]),
            loo["pred_real"],
            ORANGE,
            "--",
        ),
    ]
    metrics = []
    for label, truth, score, color, linestyle in series:
        fpr, tpr = roc_points(truth, score)
        auc = auc_from_ranks(truth, score)
        bal, recall, marker_fpr = screening_metrics(truth, score)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=f"{label} (AUC = {auc:.3f})",
        )
        ax.scatter(marker_fpr, recall, s=26, facecolor="white", edgecolor=color, linewidth=1.1, zorder=4)
        metrics.append((label, auc, bal, recall))

    ax.plot([0, 1], [0, 1], color=LIGHT, linestyle=":", linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.grid(color=LIGHT, linewidth=0.4, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    text = (
        "Top-quartile screen\n"
        "Balanced accuracy / recall\n"
        f"Paradigm mean  {metrics[0][2]:.1%} / {metrics[0][3]:.1%}\n"
        f"Real layout       {metrics[1][2]:.1%} / {metrics[1][3]:.1%}"
    )
    ax.text(
        0.97,
        0.05,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.35,
        linespacing=1.22,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.45, "alpha": 0.90, "pad": 2.0},
        zorder=5,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    panel_label(ax, "d", x=-0.13, y=1.04)


def main() -> None:
    apply_style()
    cross, loo, real, grid, farm_geo = load_inputs()
    grid = ocean_grid(grid)

    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=(0.92, 1.05, 1.05), hspace=0.13, wspace=0.24)

    ax_a = fig.add_subplot(gs[0, :])
    add_sensitivity_panel(ax_a, cross)

    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    sc = add_map_panel(
        ax_b1,
        grid,
        farm_geo,
        [101, 126, 5, 33],
        [105, 115, 125],
        [10, 20, 30],
        "East Asia",
    )
    add_map_panel(
        ax_b2,
        grid,
        farm_geo,
        [-2, 23, 35, 61],
        [0, 10, 20],
        [40, 50, 60],
        "Europe",
    )
    panel_label(ax_b1, "b", x=-0.12, y=1.04)
    cbar = fig.colorbar(sc, ax=[ax_b1, ax_b2], orientation="horizontal", fraction=0.055, pad=0.08, aspect=32)
    cbar.set_label("Paradigms retaining top-quartile response (count of 6)", labelpad=4)
    cbar.set_ticks(range(7))
    cbar.outline.set_linewidth(0.5)

    ax_c = fig.add_subplot(gs[2, 0])
    add_type_match_panel(ax_c, cross, real)
    ax_d = fig.add_subplot(gs[2, 1])
    add_validation_panel(ax_d, loo)

    png = HERE / "Fig3d_ADAPEN.png"
    pdf = HERE / "Fig3d_ADAPEN.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
