"""Four-scenario attribution design and manuscript inference chain."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ne_style import COLORS, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent


def box(ax, xy, width, height, text, face, edge=COLORS["ink"], fontsize=7):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.75,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.75,
            color=COLORS["grey"],
        )
    )


def main() -> None:
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.45]},
    )
    for axis in (ax_a, ax_b):
        axis.set_axis_off()

    x0, y0, width, height = 0.18, 0.16, 0.31, 0.25
    box(ax_a, (x0, y0 + 0.34), width, height, "$P_{00}$\nbaseline speed\nbaseline direction", COLORS["pale"])
    box(ax_a, (x0 + 0.39, y0 + 0.34), width, height, "$P_{01}$\nbaseline speed\nactual direction", "#FDE8E8")
    box(ax_a, (x0, y0), width, height, "$P_{10}$\nactual speed\nbaseline direction", "#E8F1FA")
    box(ax_a, (x0 + 0.39, y0), width, height, "$P_{11}$\nactual speed\nactual direction", "#EFE8F5")
    ax_a.text(0.525, 0.91, "Wind direction", ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    ax_a.text(0.525, 0.83, "Baseline", ha="center", fontsize=6.5, color=COLORS["grey"])
    ax_a.text(0.77, 0.83, "Actual", ha="center", fontsize=6.5, color=COLORS["grey"])
    ax_a.text(0.035, 0.45, "Wind speed", rotation=90, ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    ax_a.text(0.12, 0.63, "Baseline", rotation=90, ha="center", va="center", fontsize=6.5, color=COLORS["grey"])
    ax_a.text(0.12, 0.28, "Actual", rotation=90, ha="center", va="center", fontsize=6.5, color=COLORS["grey"])
    ax_a.text(
        0.5,
        0.035,
        "$\\phi_v=\\frac{1}{2}[(P_{10}-P_{00})+(P_{11}-P_{01})]$\n"
        "$\\phi_d=\\frac{1}{2}[(P_{01}-P_{00})+(P_{11}-P_{10})]$",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["ink"],
        linespacing=1.45,
    )

    stages = [
        (0.02, "Existing farms\n$G_i>\\phi_{v,i}$"),
        (0.22, "Spatial\nconcentration"),
        (0.42, "Physical\ndrivers"),
        (0.62, "Held-out\nscreening"),
        (0.82, "Energy and\nvalue losses"),
    ]
    faces = ["#FDE8E8", "#FFF1E6", "#EAF3EC", "#E8F1FA", "#EFE8F5"]
    for (x, text), face in zip(stages, faces):
        box(ax_b, (x, 0.43), 0.15, 0.23, text, face, fontsize=7)
    for idx in range(len(stages) - 1):
        arrow(ax_b, (stages[idx][0] + 0.15, 0.545), (stages[idx + 1][0], 0.545))
    ax_b.text(
        0.5,
        0.78,
        "Inference chain",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["ink"],
    )
    ax_b.text(
        0.5,
        0.27,
        "Observed discovery  →  explanation  →  geographic transfer  →  system consequence",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["grey"],
    )
    panel_label(ax_a, "a", x=-0.02, y=1.0)
    panel_label(ax_b, "b", x=-0.02, y=1.0)
    save_figure(fig, HERE / "FigS1_four_scenario_design.png")
    plt.close(fig)
    print("saved four-scenario attribution schematic")


if __name__ == "__main__":
    main()

