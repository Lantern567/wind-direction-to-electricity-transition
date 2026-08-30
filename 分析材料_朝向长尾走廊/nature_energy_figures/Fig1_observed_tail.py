"""Observed orientation-recovery tail and interannual-CF comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ne_style import COLORS, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DERIVED = REPO / "分析材料_几何主导杠杆" / "data_derived"


def load_data():
    annual_gain = pd.read_csv(DERIVED / "orientation_gain.csv")
    farm_gain = annual_gain.groupby("farm_id")["gain_pct"].mean().rename("gain")

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
    farm_cv.name = "cf_cv"

    metadata = pd.read_csv(DERIVED / "farm_wakeloss_map.csv").set_index("farm_id")
    data = pd.concat([farm_gain, farm_cv], axis=1).join(metadata[["country"]])
    return annual_gain, data


def main() -> None:
    apply_style()
    annual_gain, data = load_data()
    threshold = float(data["cf_cv"].median())

    ranked = data["gain"].sort_values(ascending=False)
    values = ranked.to_numpy()
    colours = np.where(
        values > threshold,
        COLORS["red"],
        np.where(values > 2, COLORS["orange"], COLORS["light"]),
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )

    ranks = np.arange(1, len(ranked) + 1)
    ax_a.bar(ranks, np.maximum(values, 0), width=1.0, color=colours, linewidth=0)
    negative = values < 0
    ax_a.bar(
        ranks[negative],
        values[negative],
        width=1.0,
        color=COLORS["grey"],
        linewidth=0,
    )
    ax_a.axhline(threshold, color=COLORS["ink"], ls="--", lw=0.9)
    ax_a.axhline(ranked.median(), color=COLORS["grey"], ls=":", lw=0.8)
    ax_a.text(
        169,
        threshold + 0.35,
        f"Global median observed CF variability  {threshold:.2f}%",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=COLORS["ink"],
    )
    ax_a.text(
        169,
        ranked.median() - 0.45,
        f"Median orientation recovery  {ranked.median():.2f}%",
        ha="right",
        va="top",
        fontsize=6.4,
        color=COLORS["grey"],
    )
    for farm_id, dx, dy in [(57, 18, -0.2), (155, 22, -0.1), (66, 25, 0.4)]:
        value = ranked.loc[farm_id]
        rank = int(np.flatnonzero(ranked.index.to_numpy() == farm_id)[0]) + 1
        country = data.loc[farm_id, "country"]
        ax_a.annotate(
            f"F{farm_id}, {country}  {value:.1f}%",
            xy=(rank, value),
            xytext=(rank + dx, value + dy),
            fontsize=6.2,
            color=COLORS["ink"],
            va="center",
            arrowprops={"arrowstyle": "-", "color": COLORS["grey"], "lw": 0.55},
        )
    ax_a.text(
        0.98,
        0.78,
        f"{int((ranked > threshold).sum())} of {len(ranked)} farms above\n"
        f"the global variability benchmark",
        transform=ax_a.transAxes,
        ha="right",
        va="center",
        fontsize=6.5,
        color=COLORS["ink"],
    )
    ax_a.set_xlim(0, 172)
    ax_a.set_ylim(-2, 20)
    ax_a.set_xlabel("Farm rank")
    ax_a.set_ylabel("AEP recovered by correcting built orientation (%)")

    comparison = data.dropna(subset=["cf_cv"]).copy()
    exceeds = comparison["gain"] > comparison["cf_cv"]
    ax_b.scatter(
        comparison.loc[~exceeds, "cf_cv"],
        comparison.loc[~exceeds, "gain"],
        s=18,
        color=COLORS["light"],
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
    )
    ax_b.scatter(
        comparison.loc[exceeds, "cf_cv"],
        comparison.loc[exceeds, "gain"],
        s=31,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    limit = 26
    ax_b.plot([0, limit], [0, limit], color=COLORS["ink"], ls="--", lw=0.8)
    label_offsets = {
        57: (4, 4),
        66: (4, 4),
        157: (4, -7),
        92: (4, 6),
        160: (4, -10),
    }
    for farm_id in comparison.index[exceeds]:
        row = comparison.loc[farm_id]
        ax_b.annotate(
            f"F{farm_id}",
            (row["cf_cv"], row["gain"]),
            xytext=label_offsets.get(int(farm_id), (4, 4)),
            textcoords="offset points",
            fontsize=6.1,
            color=COLORS["ink"],
        )
    ax_b.text(
        0.97,
        0.95,
        f"n = {len(comparison)} farms with ≥5 years\n"
        f"{int(exceeds.sum())} above the 1:1 line",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["ink"],
    )
    ax_b.text(
        0.97,
        0.05,
        "Observed CF variability includes all annual drivers",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=COLORS["grey"],
    )
    ax_b.set_xlim(0, limit)
    ax_b.set_ylim(-2, 20)
    ax_b.set_xlabel("Observed interannual CF variability (%)")
    ax_b.set_ylabel("Orientation recovery (%)")

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    save_figure(fig, HERE / "Fig1_observed_tail.png")
    plt.close(fig)

    print(
        f"farms={len(ranked)}, farm_years={len(annual_gain)}, threshold={threshold:.6f}, "
        f"above_global={int((ranked > threshold).sum())}, "
        f"above_individual={int(exceeds.sum())}"
    )


if __name__ == "__main__":
    main()
