"""Internal-only audit figure for the submitted four-scenario calculation.

Revision 2026-08-06: retargeted at submission ``3e4669c`` (v3).  The v2 blocking
issue shown in the old panel c — farm-years carrying two or three years of
hours — has been fixed, so that panel now shows the replacement blocker: the
submitted ``n_hours`` integrates whole calendar years while the reference chain
integrates effective hours.  Panels e and f were added to demonstrate that the
残余 bias is neither a constant scale factor nor a constant loss term.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ne_style import COLORS, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUBMISSION = REPO / "四场景风速风向分解贡献"
TASK2 = REPO / "offshore-task2" / "output"


def relative_error(submitted: pd.Series, reference: pd.Series) -> pd.Series:
    return (submitted - reference).abs() / reference.abs() * 100


def parity_panel(ax, reference, submitted, error, x_label, y_label) -> None:
    lower = min(reference.min(), submitted.min()) * 0.78
    upper = max(reference.max(), submitted.max()) * 1.28
    ax.scatter(
        reference / 1e9,
        submitted / 1e9,
        c=np.clip(error, 0, 150),
        cmap="OrRd",
        vmin=0,
        vmax=150,
        s=11,
        alpha=0.72,
        edgecolor="none",
        rasterized=True,
    )
    ax.plot(
        [lower / 1e9, upper / 1e9],
        [lower / 1e9, upper / 1e9],
        color=COLORS["ink"],
        ls="--",
        lw=0.8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower / 1e9, upper / 1e9)
    ax.set_ylim(lower / 1e9, upper / 1e9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.text(
        0.04,
        0.96,
        f"Median error = {error.median():.1f}%\n95th percentile = {error.quantile(0.95):.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["ink"],
    )


def main() -> None:
    apply_style()
    submitted = pd.read_csv(SUBMISSION / "four_scenario_aep_farmyear.csv")
    annual = pd.read_csv(TASK2 / "task2_annual_floris.csv")
    annual = annual.loc[annual["wake_model"].eq("gauss")].drop_duplicates(
        subset=["farm_id", "year"]
    )
    counter = pd.read_csv(TASK2 / "task2_counterfactual.csv")

    p11 = submitted.merge(
        annual[["farm_id", "year", "AEP_kWh", "region", "n_hours"]],
        on=["farm_id", "year"],
        how="inner",
        suffixes=("_submitted", "_reference"),
        validate="one_to_one",
    )
    p11["error"] = relative_error(p11["P11_kWh"], p11["AEP_kWh"])
    p11["ratio"] = p11["P11_kWh"] / p11["AEP_kWh"]
    p10 = submitted.merge(
        counter[["farm_id", "year", "AEP_baseWD_kWh"]],
        on=["farm_id", "year"],
        how="inner",
        validate="one_to_one",
    )
    p10["error"] = relative_error(p10["P10_kWh"], p10["AEP_baseWD_kWh"])

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.9), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    parity_panel(
        ax_a,
        p11["AEP_kWh"],
        p11["P11_kWh"],
        p11["error"],
        "Reference actual-climate AEP (TWh yr$^{-1}$)",
        "Submitted $P_{11}$ (TWh yr$^{-1}$)",
    )
    parity_panel(
        ax_b,
        p10["AEP_baseWD_kWh"],
        p10["P10_kWh"],
        p10["error"],
        "Reference fixed-direction AEP (TWh yr$^{-1}$)",
        "Submitted $P_{10}$ (TWh yr$^{-1}$)",
    )

    # c — hours convention: whole calendar year versus effective hours
    gap = p11["n_hours_submitted"] - p11["n_hours_reference"]
    ax_c.scatter(
        p11["n_hours_reference"],
        p11["n_hours_submitted"],
        s=9,
        color=COLORS["blue"],
        alpha=0.42,
        edgecolor="none",
        rasterized=True,
    )
    span = (5900, 8950)
    ax_c.plot(span, span, color=COLORS["ink"], ls="--", lw=0.8)
    ax_c.set_xlim(*span)
    ax_c.set_ylim(*span)
    ax_c.set_xlabel("Reference effective hours (h yr$^{-1}$)")
    ax_c.set_ylabel("Submitted n_hours (h yr$^{-1}$)")
    ax_c.text(
        0.04,
        0.96,
        f"Identical = {int((gap == 0).sum())} / {len(p11)}\n"
        f"Median gap = {gap.median():.0f} h\n"
        f"Coverage = {len(submitted):,} / 1,203",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["ink"],
    )

    # d — region confusion matrix
    region_names = {
        "east_asia": "East Asia",
        "East_Asia": "East Asia",
        "europe": "Europe",
        "Europe": "Europe",
        "us_east": "US East",
        "US_East": "US East",
    }
    p11["reference_region"] = p11["region_reference"].map(region_names).fillna("Missing")
    p11["submitted_region"] = p11["region_submitted"].map(region_names).fillna("Missing")
    row_order = ["East Asia", "Europe", "US East"]
    col_order = ["East Asia", "Europe", "US East", "Missing"]
    confusion = pd.crosstab(p11["reference_region"], p11["submitted_region"]).reindex(
        index=row_order, columns=col_order, fill_value=0
    )
    image = ax_d.imshow(confusion.to_numpy(), cmap="OrRd", vmin=0, vmax=confusion.to_numpy().max())
    threshold = confusion.to_numpy().max() * 0.48
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = int(confusion.iloc[i, j])
            ax_d.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else COLORS["ink"],
            )
    ax_d.set_xticks(np.arange(len(col_order)))
    ax_d.set_xticklabels(col_order, rotation=28, ha="right")
    ax_d.set_yticks(np.arange(len(row_order)))
    ax_d.set_yticklabels(row_order)
    ax_d.set_xlabel("Submitted region")
    ax_d.set_ylabel("Reference region")
    ax_d.grid(False)
    colourbar = fig.colorbar(image, ax=ax_d, pad=0.02, fraction=0.046)
    colourbar.set_label("Farm–year records", fontsize=6.5)
    colourbar.ax.tick_params(labelsize=6)

    # e — the bias scales with farm size, so no single factor can correct it
    bands = pd.cut(p11["n_turb"], [0, 100, 400, 10000], labels=["≤100", "101–400", ">400"])
    stats = p11.groupby(bands, observed=True).agg(
        n=("error", "size"), median_error=("error", "median"), median_ratio=("ratio", "median")
    )
    bars = ax_e.bar(
        np.arange(len(stats)),
        stats["median_error"],
        color=[COLORS["grey"], COLORS["orange"], COLORS["red"]],
        width=0.62,
    )
    ax_e.bar_label(
        bars,
        labels=[f"{v:.1f}%\nratio {r:.3f}\n(n={n})"
                for v, r, n in zip(stats["median_error"], stats["median_ratio"], stats["n"])],
        padding=2,
        fontsize=6.4,
    )
    ax_e.axhline(1.5, color=COLORS["ink"], ls="--", lw=0.8)
    ax_e.text(-0.38, 3.4, "1.5% acceptance", ha="left", fontsize=6.4, color=COLORS["ink"])
    ax_e.set_xticks(np.arange(len(stats)))
    ax_e.set_xticklabels([f"{s} turbines" for s in stats.index])
    ax_e.set_ylabel("Median $P_{11}$ error (%)")
    ax_e.set_ylim(0, 62)

    # f — within one farm the bias reverses sign between years
    f66 = p11.loc[p11["farm_id"].eq(66)].sort_values("year")
    ax_f.bar(f66["year"], f66["ratio"], width=0.58, color=COLORS["blue"])
    ax_f.axhline(1.0, color=COLORS["ink"], ls="--", lw=0.8)
    for year, ratio in zip(f66["year"], f66["ratio"]):
        ax_f.text(year, ratio + 0.04, f"{ratio:.2f}", ha="center", fontsize=6.4,
                  color=COLORS["ink"])
    ax_f.set_xticks(f66["year"].tolist())
    ax_f.set_xticklabels([int(y) for y in f66["year"]])
    ax_f.set_ylim(0, 1.78)
    ax_f.set_xlabel("Year")
    ax_f.set_ylabel("$P_{11}$ / reference AEP, farm F66")
    ax_f.text(
        0.04,
        0.96,
        "Over- then under-estimation\nwithin a single layout",
        transform=ax_f.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["ink"],
    )

    for axis, letter in zip(axes.ravel(), "abcdef"):
        panel_label(axis, letter)
    save_figure(fig, HERE / "FigQA_four_scenario_audit.png")
    plt.close(fig)

    mismatches = int((p11["reference_region"] != p11["submitted_region"]).sum())
    print(
        f"n={len(p11)}; "
        f"P11 median={p11['error'].median():.4f}%, p95={p11['error'].quantile(0.95):.4f}%, "
        f"pass1.5={int((p11['error'] <= 1.5).sum())}; "
        f"P10 median={p10['error'].median():.4f}%, p95={p10['error'].quantile(0.95):.4f}%, "
        f"pass1.5={int((p10['error'] <= 1.5).sum())}; "
        f"region mismatches={mismatches}/{len(p11)}; "
        f"n_hours>9000={(submitted['n_hours'] > 9000).sum()}; "
        f"hours identical={int((gap == 0).sum())}, median gap={gap.median():.0f} h"
    )


if __name__ == "__main__":
    main()
