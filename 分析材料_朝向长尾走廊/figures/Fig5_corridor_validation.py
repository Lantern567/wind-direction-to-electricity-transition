"""Figure 4: corridor footprint, energy weighting, and validation.

The map is rendered with frykit. All labels are English and every statistic is
recomputed from the committed supplementary outputs.
"""
from __future__ import annotations

from pathlib import Path

import frykit.plot as fplt
import frykit.shp as fshp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

from nc_style import PAL, apply_style, panel_label, savefig


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if not (REPO / "补算").exists():
    REPO = REPO / "wind-direction-to-electricity-transition"
SUPP = REPO / "补算"
GEOM = REPO / "分析材料_几何主导杠杆" / "data_derived"


def load_data():
    training = pd.read_csv(SUPP / "output" / "task1_training_data.csv")
    loo = pd.read_csv(SUPP / "output" / "task1_loo_predictions.csv")
    grid = pd.read_csv(SUPP / "output" / "task1_corridor_grid.csv")
    energy_path = SUPP / "corridor_aep_validation.csv"
    if not energy_path.exists():
        energy_path = SUPP / "output" / "corridor_aep_validation.csv"
    energy = pd.read_csv(energy_path)
    energy["recoverable_GWh"] = (
        energy["AEP_kWh"] * energy["A_pred_pct"] / 100.0 / 1e6
    )
    gain = (
        pd.read_csv(GEOM / "orientation_gain.csv")
        .groupby("farm_id", as_index=False)["gain_pct"]
        .mean()
        .rename(columns={"gain_pct": "mean_gain_pct"})
    )
    training = training.merge(gain, on="farm_id", how="left")
    return training, loo, grid, energy


def model_diagnostics(training):
    features = [
        "WCI", "wake_pool", "WCI_x_pool", "spacing_D", "aspect_ratio",
        "n_turb", "log_n", "pc1_share", "ws_mean", "ws_std",
        "weibull_A", "weibull_k", "frac_below_rated",
        "orient_sensitivity", "gain_proxy_raw", "wd_entropy_norm",
        "exp_wake_loss", "WCI_density", "pool_density", "ws_x_WCI",
    ]
    x = training[features].to_numpy()
    y = training["A"].to_numpy()
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
    )
    model.fit(x, y)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring="r2")
    importance = pd.Series(model.feature_importances_, index=features)
    return scores, importance.sort_values(ascending=False)


def prepare_map(ax):
    extent = (-82, 145, 4, 63)
    fplt.set_map_ticks(ax, extent, dx=40, dy=15)
    ax.grid(False)
    ax.set_facecolor("#EAF2F8")
    fplt.add_land(ax, facecolor="#F4F4F2", edgecolor="none", zorder=0)
    fplt.add_countries(
        ax,
        facecolor="none",
        edgecolor="#B8BEC6",
        linewidth=0.35,
        zorder=1,
    )


def ocean_only(data):
    ocean = fshp.get_ocean()
    mask = shapely.contains_xy(ocean, data["lon"], data["lat"])
    return data.loc[mask].copy()


def add_share_map(ax, training, grid):
    prepare_map(ax)

    # The original grid is distance-masked but not land-masked. Restrict the
    # display to ocean cells so the visual encoding matches "offshore".
    offshore = ocean_only(grid)

    norm = mcolors.Normalize(vmin=0, vmax=16)
    sc = ax.scatter(
        offshore["lon"],
        offshore["lat"],
        c=offshore["A_pred_pct"],
        cmap="YlOrRd",
        norm=norm,
        marker="s",
        s=14,
        linewidths=0,
        alpha=0.88,
        zorder=2,
    )

    ax.scatter(
        training["lon"],
        training["lat"],
        s=11,
        facecolor="white",
        edgecolor=PAL["ink"],
        linewidth=0.35,
        alpha=0.8,
        zorder=3,
        label="Observed farms",
    )
    tail = training[training["mean_gain_pct"] > 5.0]
    ax.scatter(
        tail["lon"],
        tail["lat"],
        s=43,
        facecolor="none",
        edgecolor=PAL["highlight"],
        linewidth=1.25,
        zorder=4,
        label=">5% realized gain",
    )

    labels = [
        (105.6, 9.2, "Vietnam monsoon", (7, 10)),
        (121.0, 25.0, "Taiwan Strait", (-58, -1)),
        (17.1, 40.5, "Adriatic", (5, -9)),
        (10.9, 55.3, "Danish straits", (7, -12)),
    ]
    for lon, lat, text, offset in labels:
        ax.annotate(
            text,
            xy=(lon, lat),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.8,
            color=PAL["ink"],
            arrowprops={"arrowstyle": "-", "color": PAL["neutral"], "lw": 0.45},
            zorder=5,
        )

    ax.legend(loc="upper left", ncol=2, handletextpad=0.4, columnspacing=0.8)
    ax.text(
        0.01,
        0.02,
        "1° grid; ocean cells ≤500 km from observed farms",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PAL["neutral"],
    )
    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.012, fraction=0.022, aspect=28)
    cbar.set_label("Predicted recoverable AEP share (%)", fontsize=7.2, labelpad=7)
    cbar.outline.set_linewidth(0.6)


def add_energy_map(ax, energy):
    prepare_map(ax)
    offshore = ocean_only(energy)
    norm = mcolors.Normalize(vmin=0, vmax=400)
    sc = ax.scatter(
        offshore["lon"],
        offshore["lat"],
        c=offshore["recoverable_GWh"],
        cmap="viridis",
        norm=norm,
        marker="s",
        s=14,
        linewidths=0,
        alpha=0.9,
        zorder=2,
    )
    ax.text(
        0.01,
        0.02,
        "Reference project: 64 × IEA 10 MW, 5D square grid",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PAL["neutral"],
    )
    ax.text(
        0.99,
        0.02,
        "Energy weighting, not independent validation",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=PAL["highlight"],
    )
    ax.text(
        0.99,
        0.96,
        f"Ocean-only display: n = {len(offshore):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color=PAL["neutral"],
    )
    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.012, fraction=0.022, aspect=28)
    cbar.set_label(
        "Recoverable energy per reference project (GWh yr$^{-1}$)",
        fontsize=7.2,
        labelpad=7,
    )
    cbar.outline.set_linewidth(0.6)


def add_validation(ax, loo):
    tail = loo["actual_A"] > 5.2
    ax.scatter(
        loo.loc[~tail, "actual_A"],
        loo.loc[~tail, "pred_A"],
        s=18,
        color=PAL["neutral"],
        alpha=0.55,
        linewidth=0,
        label="$A\\leq5.2\\%$",
    )
    ax.scatter(
        loo.loc[tail, "actual_A"],
        loo.loc[tail, "pred_A"],
        s=24,
        color=PAL["highlight"],
        alpha=0.78,
        linewidth=0,
        label="$A>5.2\\%$",
    )
    lim = 26
    ax.plot([0, lim], [0, lim], color=PAL["ink"], lw=0.8, ls="--")
    ax.set(xlim=(0, lim), ylim=(0, lim))
    ax.set_xlabel("Observed recoverable AEP share (%)")
    ax.set_ylabel("Country-holdout prediction (%)")
    rho, p = spearmanr(loo["actual_A"], loo["pred_A"])
    r2 = 1 - np.sum((loo["actual_A"] - loo["pred_A"]) ** 2) / np.sum(
        (loo["actual_A"] - loo["actual_A"].mean()) ** 2
    )
    auc2 = roc_auc_score(loo["actual_A"] > 2.0, loo["pred_A"])
    auc52 = roc_auc_score(tail, loo["pred_A"])
    ax.text(
        0.04,
        0.96,
        "Country holdout\n"
        f"Spearman $\\rho$ = {rho:.3f}\n"
        f"AUC ($A>2\\%$) = {auc2:.3f}\n"
        f"AUC ($A>5.2\\%$) = {auc52:.3f}\n"
        f"Amplitude $R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        color=PAL["ink"],
    )
    ax.legend(loc="lower right", handletextpad=0.35)


def add_importance(ax, scores, importance):
    label_map = {
        "ws_std": "Wind-speed SD",
        "frac_below_rated": "Below-rated fraction",
        "weibull_k": "Weibull k",
        "ws_mean": "Mean wind speed",
        "wd_entropy_norm": "Direction entropy",
        "orient_sensitivity": "Orientation proxy",
        "weibull_A": "Weibull A",
        "gain_proxy_raw": "Gain proxy",
    }
    top = importance.head(8).sort_values()
    labels = [label_map.get(x, x) for x in top.index]
    colors = [PAL["highlight"] if x == "ws_std" else PAL["baseline"] for x in top.index]
    ax.barh(labels, top.values, color=colors, height=0.65)
    ax.set_xlabel("Random-forest feature importance")
    ax.set_xlim(0, 0.72)
    ax.text(
        0.98,
        0.05,
        f"Random 5-fold CV $R^2$\n= {scores.mean():.2f} ± {scores.std():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PAL["ink"],
    )


def main():
    apply_style()
    training, loo, grid, energy = load_data()
    scores, importance = model_diagnostics(training)

    fig = plt.figure(figsize=(7.2, 8.25))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 0.92),
        hspace=0.40,
        wspace=0.48,
    )
    ax_share = fig.add_subplot(gs[0, :])
    ax_energy = fig.add_subplot(gs[1, :])
    ax_val = fig.add_subplot(gs[2, 0])
    ax_imp = fig.add_subplot(gs[2, 1])

    add_share_map(ax_share, training, grid)
    add_energy_map(ax_energy, energy)
    add_validation(ax_val, loo)
    add_importance(ax_imp, scores, importance)
    panel_label(ax_share, "a", dx=-0.055, dy=1.05)
    panel_label(ax_energy, "b", dx=-0.055, dy=1.05)
    panel_label(ax_val, "c", dx=-0.10, dy=1.06)
    panel_label(ax_imp, "d", dx=-0.12, dy=1.06)

    savefig(fig, str(HERE / "Fig5_corridor_validation.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
