"""Figure 3: transferable determinants of orientation sensitivity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from nc_style import PAL, apply_style, panel_label, savefig


HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "补算" / "output"
met = pd.read_csv(OUT / "mechanism_v2_metrics.csv").dropna(subset=["A"])
curves = pd.read_csv(OUT / "mechanism_v2_curves.csv")
loco = pd.read_csv(OUT / "mechanism_v2_loco_predictions.csv")
country_holdout = pd.read_csv(OUT / "task1_loo_predictions.csv")

HEADLINE = [57, 126, 159, 66, 91, 155, 157]
FLOOR = 5.2
met["headline"] = met["farm_id"].isin(HEADLINE)
met["narrow"] = 1.0 - met["wd_entropy_norm"]
met["logA"] = np.log(met["A"])


def main():
    apply_style()
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a: directional wake-loss response on two real layouts.
    high_id = 126
    low_pool = met[(met["A"] < 0.5) & (met["lat"] > 51) & (met["n_turb"] > 40)]
    low_id = int(low_pool.sort_values("n_turb", ascending=False).iloc[0]["farm_id"])
    for farm_id, color, label in [
        (high_id, PAL["highlight"], f"F{high_id}, Mekong corridor"),
        (low_id, PAL["baseline"], f"F{low_id}, North Sea"),
    ]:
        block = curves[curves["farm_id"] == farm_id].sort_values("theta_deg")
        row = met[met["farm_id"] == farm_id].iloc[0]
        ax_a.plot(
            block["theta_deg"],
            block["L_energy"] * 100,
            color=color,
            lw=1.4,
            label=f"{label}\nrange {row['Lw_range']:.0f} pp; sensitivity {row['A']:.1f}%",
        )
    ax_a.set_xlim(0, 355)
    ax_a.set_xticks([0, 90, 180, 270, 355])
    ax_a.set_ylim(-3, 95)
    ax_a.set_xlabel("Inflow direction (degrees)")
    ax_a.set_ylabel("Energy-weighted wake loss (%)")
    ax_a.legend(loc="upper left", fontsize=6.2, handlelength=1.2, labelspacing=0.6)

    # b: main determinant and the power-conversion operating window.
    scatter = ax_b.scatter(
        met["Lw_range"],
        met["A"],
        c=met["frac_below_rated"],
        cmap="YlOrRd",
        s=17,
        lw=0.3,
        edgecolor="white",
        vmin=0.5,
        vmax=1.0,
        zorder=3,
    )
    ax_b.scatter(
        met.loc[met["headline"], "Lw_range"],
        met.loc[met["headline"], "A"],
        s=48,
        facecolor="none",
        edgecolor=PAL["highlight"],
        lw=1.1,
        zorder=4,
    )
    ax_b.axhline(FLOOR, color=PAL["neutral"], ls=":", lw=0.8)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Directional wake-loss range (percentage points)")
    ax_b.set_ylabel("Orientation sensitivity, $A$ (%, log scale)")
    rho = spearmanr(met["Lw_range"], met["A"]).statistic
    auc = roc_auc_score((met["A"] > FLOOR).astype(int), met["Lw_range"])
    ax_b.text(
        0.04,
        0.96,
        f"Spearman $\\rho$ = {rho:+.3f}\nAUC ($A>5.2\\%$) = {auc:.3f}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=PAL["ink"],
    )
    colorbar = fig.colorbar(scatter, ax=ax_b, pad=0.02, fraction=0.045)
    colorbar.set_label("Share of hours below rated speed", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)

    # c: interaction with wind-axis concentration.
    met["interaction"] = met["Lw_range"] * met["narrow"]
    fit = sm.OLS(
        met["logA"],
        sm.add_constant(met[["Lw_range", "narrow", "interaction"]]),
    ).fit()
    x_grid = np.linspace(met["Lw_range"].quantile(0.05), met["Lw_range"].quantile(0.95), 120)
    q_values = met["narrow"].quantile([0.25, 0.50, 0.75]).to_numpy()
    for q, color, label in zip(
        q_values,
        [PAL["neutral"], PAL["accent"], PAL["highlight"]],
        ["Low", "Median", "High"],
    ):
        design = pd.DataFrame(
            {
                "const": np.ones_like(x_grid),
                "Lw_range": x_grid,
                "narrow": np.full_like(x_grid, q),
                "interaction": x_grid * q,
            }
        )
        ax_c.plot(
            x_grid,
            np.exp(fit.predict(design)),
            color=color,
            lw=1.5,
            label=f"{label} wind-axis concentration",
        )
    ax_c.axhline(FLOOR, color=PAL["neutral"], ls=":", lw=0.8)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Directional wake-loss range (percentage points)")
    ax_c.set_ylabel("Modelled orientation sensitivity (%, log scale)")
    ax_c.text(
        0.04,
        0.96,
        f"Interaction $p$ = {fit.pvalues['interaction']:.4f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=PAL["ink"],
    )
    ax_c.legend(loc="lower right", fontsize=6.0, handlelength=1.2, labelspacing=0.5)

    # d: independent geographic screening.
    for source, prediction, color, label in [
        (loco, "A_pred_loco", PAL["highlight"], "Physical factors; corridor holdout"),
        (country_holdout, "pred_A", PAL["baseline"], "Random-forest atlas; country holdout"),
    ]:
        actual = source["A_actual"] if "A_actual" in source else source["actual_A"]
        binary = (actual > FLOOR).astype(int)
        fpr, tpr, _ = roc_curve(binary, source[prediction])
        score = roc_auc_score(binary, source[prediction])
        ax_d.plot(fpr, tpr, color=color, lw=1.5, label=f"{label}\nAUC = {score:.3f}")
    ax_d.plot([0, 1], [0, 1], color=PAL["neutral"], ls="--", lw=0.8)
    ax_d.set_xlabel("False-positive rate")
    ax_d.set_ylabel("True-positive rate")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.text(
        0.04,
        0.96,
        "Screening above wind-year variability",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=PAL["ink"],
    )
    ax_d.legend(loc="lower right", fontsize=6.2, handlelength=1.2, labelspacing=0.7)

    for axis, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        panel_label(axis, letter)

    savefig(fig, str(HERE / "Fig3_mechanism_v2.png"))
    plt.close(fig)
    print(f"corridor farm F{high_id}; reference farm F{low_id}")


if __name__ == "__main__":
    main()
