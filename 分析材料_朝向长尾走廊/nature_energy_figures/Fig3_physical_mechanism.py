"""Physical determinants and held-out screening of intrinsic sensitivity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from ne_style import COLORS, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = REPO / "补算" / "output"
THRESHOLD = 5.2
HEADLINE = [57, 126, 159, 66, 91, 155, 157]


def main() -> None:
    apply_style()
    metrics = pd.read_csv(OUTPUT / "mechanism_v2_metrics.csv").dropna(subset=["A"]).copy()
    curves = pd.read_csv(OUTPUT / "mechanism_v2_curves.csv")
    held_out = pd.read_csv(OUTPUT / "mechanism_v2_loco_predictions.csv")

    metrics["headline"] = metrics["farm_id"].isin(HEADLINE)
    metrics["concentration"] = 1 - metrics["wd_entropy_norm"]
    metrics["interaction"] = metrics["Lw_range"] * metrics["concentration"]
    metrics["log_sensitivity"] = np.log(metrics["A"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    high_id = 126
    reference_pool = metrics.loc[
        (metrics["A"] < 0.5) & (metrics["lat"] > 51) & (metrics["n_turb"] > 40)
    ]
    reference_id = int(reference_pool.sort_values("n_turb", ascending=False).iloc[0]["farm_id"])
    for farm_id, colour, label in [
        (high_id, COLORS["red"], f"F{high_id}, Mekong corridor"),
        (reference_id, COLORS["blue"], f"F{reference_id}, North Sea"),
    ]:
        block = curves.loc[curves["farm_id"].eq(farm_id)].sort_values("theta_deg")
        row = metrics.loc[metrics["farm_id"].eq(farm_id)].iloc[0]
        ax_a.plot(
            block["theta_deg"],
            block["L_energy"] * 100,
            color=colour,
            lw=1.35,
            label=f"{label}\nrange {row['Lw_range']:.0f} pp; sensitivity {row['A']:.1f}%",
        )
    ax_a.set_xlim(0, 355)
    ax_a.set_xticks([0, 90, 180, 270, 355])
    ax_a.set_ylim(-3, 95)
    ax_a.set_xlabel("Inflow direction (degrees)")
    ax_a.set_ylabel("Energy-weighted wake loss (%)")
    ax_a.legend(loc="upper left", handlelength=1.2, labelspacing=0.6)

    scatter = ax_b.scatter(
        metrics["Lw_range"],
        metrics["A"],
        c=metrics["frac_below_rated"],
        cmap="YlOrRd",
        s=18,
        lw=0.3,
        edgecolor="white",
        vmin=0.5,
        vmax=1.0,
        zorder=3,
    )
    ax_b.scatter(
        metrics.loc[metrics["headline"], "Lw_range"],
        metrics.loc[metrics["headline"], "A"],
        s=50,
        facecolor="none",
        edgecolor=COLORS["red"],
        lw=1.0,
        zorder=4,
    )
    ax_b.axhline(THRESHOLD, color=COLORS["grey"], ls=":", lw=0.8)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Directional wake-loss range (percentage points)")
    ax_b.set_ylabel("Intrinsic orientation sensitivity (%)")
    rho = spearmanr(metrics["Lw_range"], metrics["A"]).statistic
    auc = roc_auc_score(metrics["A"] > THRESHOLD, metrics["Lw_range"])
    ax_b.text(
        0.04,
        0.96,
        f"Spearman $\\rho$ = {rho:.3f}\nAUC = {auc:.3f}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["ink"],
    )
    colorbar = fig.colorbar(scatter, ax=ax_b, pad=0.02, fraction=0.046)
    colorbar.set_label("Share of hours below rated speed", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)

    fit = sm.OLS(
        metrics["log_sensitivity"],
        sm.add_constant(metrics[["Lw_range", "concentration", "interaction"]]),
    ).fit()
    x_grid = np.linspace(
        metrics["Lw_range"].quantile(0.05),
        metrics["Lw_range"].quantile(0.95),
        120,
    )
    concentrations = metrics["concentration"].quantile([0.25, 0.50, 0.75]).to_numpy()
    for value, colour, label in zip(
        concentrations,
        [COLORS["grey"], COLORS["orange"], COLORS["red"]],
        ["Low", "Median", "High"],
    ):
        design = pd.DataFrame(
            {
                "const": np.ones_like(x_grid),
                "Lw_range": x_grid,
                "concentration": np.full_like(x_grid, value),
                "interaction": x_grid * value,
            }
        )
        ax_c.plot(
            x_grid,
            np.exp(fit.predict(design)),
            color=colour,
            lw=1.35,
            label=f"{label} wind-axis concentration",
        )
    ax_c.axhline(THRESHOLD, color=COLORS["grey"], ls=":", lw=0.8)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Directional wake-loss range (percentage points)")
    ax_c.set_ylabel("Modelled intrinsic sensitivity (%)")
    ax_c.text(
        0.04,
        0.96,
        f"Interaction $p$ = {fit.pvalues['interaction']:.4f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["ink"],
    )
    ax_c.legend(loc="lower right", handlelength=1.2, labelspacing=0.5)

    actual = held_out["A_actual"]
    predicted = held_out["A_pred_loco"]
    binary = actual > THRESHOLD
    fpr, tpr, _ = roc_curve(binary, predicted)
    score = roc_auc_score(binary, predicted)
    ax_d.plot(fpr, tpr, color=COLORS["red"], lw=1.45, label=f"Corridor holdout  AUC = {score:.3f}")
    ax_d.plot([0, 1], [0, 1], color=COLORS["grey"], ls="--", lw=0.8)
    ax_d.set_xlabel("False-positive rate")
    ax_d.set_ylabel("True-positive rate")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    held_rho = spearmanr(actual, predicted).statistic
    ax_d.text(
        0.04,
        0.96,
        f"Spearman $\\rho$ = {held_rho:.3f}\nTarget: intrinsic sensitivity >5.2%",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=COLORS["ink"],
    )
    ax_d.legend(loc="lower right", handlelength=1.3)

    for axis, letter in zip(axes.ravel(), "abcd"):
        panel_label(axis, letter)
    save_figure(fig, HERE / "Fig3_physical_mechanism.png")
    plt.close(fig)
    print(
        f"n={len(metrics)}, rho={rho:.6f}, auc={auc:.6f}, "
        f"interaction_p={fit.pvalues['interaction']:.8f}, "
        f"holdout_rho={held_rho:.6f}, holdout_auc={score:.6f}"
    )


if __name__ == "__main__":
    main()

