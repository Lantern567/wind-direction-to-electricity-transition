import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
SPACING_CSV = (r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition/"
               r"task1_output/task1_paradigm_classification.csv")
spacing = pd.read_csv(SPACING_CSV)["spacing_d"].dropna().to_numpy()
n_farms = spacing.size                     # 171
med = float(np.median(spacing))            # 3.32
frac_below5 = float((spacing < 5).mean())  # 0.848
pct_below5 = frac_below5 * 100.0           # 84.8

# Point-lattice optimisation POC (12 representative farms)
POC = pd.read_csv(r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived/qiming_poc_analysis.csv")
poc_pct = POC["AEP_pct"].to_numpy()
poc_sp  = POC["spacing_D_real"].to_numpy()
poc_n   = POC["n_turb"].to_numpy()
poc_med = float(np.median(poc_pct))        # 6.2
poc_mean = float(np.mean(poc_pct))         # 12.2
poc_max = float(np.max(poc_pct))           # 44.8

OUT = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures/Fig4_recoverable_envelope.png"

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), layout="constrained")
axA, axB, axC = axes

# ======================================================================
# Panel (a) -- recoverable AEP by lever  (horizontal bars)
# ======================================================================
bar_h = 0.46
y_or, y_wflo, y_lit = 1, 2, 3
labels = {
    y_or:   "Orientation\n(rigid rotation, this study)",
    y_wflo: "Point-lattice optimisation\n(this study, 12 farms)",
    y_lit:  "Ideal 5D-array optimisation\n(Warder & Piggott 2024)",
}

# lever 1: orientation (real)
axA.barh(y_or, 0.92, height=bar_h, color=PAL["neutral"], zorder=3)
axA.text(0.92 + 0.25, y_or, "+0.9%", va="center", ha="left",
         fontsize=7, color=PAL["ink"], zorder=4)

# lever 2: point-lattice POC (real): median bar + mean whisker + max callout
axA.barh(y_wflo, poc_med, height=bar_h, color=PAL["highlight"], zorder=3)
axA.errorbar(poc_mean, y_wflo, xerr=[[poc_mean - poc_med], [0]], fmt="none",
             ecolor=PAL["ink"], elinewidth=1.0, capsize=3, capthick=1.0, zorder=5)
axA.plot(poc_mean, y_wflo, marker="D", ms=4.2, color=PAL["ink"], zorder=6)
axA.text(poc_mean + 0.35, y_wflo, "median +6.2%\nmean +12.2%",
         va="center", ha="left", fontsize=6.6, color=PAL["ink"],
         linespacing=1.05, zorder=4)

# lever 3: ideal 5D-array optimisation (literature anchor, 6-7%)
axA.barh(y_lit, 6.5, height=bar_h, color=PAL["accent"], zorder=3)
axA.errorbar(6.5, y_lit, xerr=0.5, fmt="none", ecolor=PAL["ink"],
             elinewidth=1.0, capsize=3, capthick=1.0, zorder=5)
axA.text(6.5 + 0.55, y_lit, "6.5%  (6-7)", va="center", ha="left",
         fontsize=7, color=PAL["ink"], zorder=4)

# category labels above each bar
for y, lab in labels.items():
    axA.text(0.05, y + 0.33, lab, va="bottom", ha="left",
             fontsize=6.5, color=PAL["ink"], linespacing=1.0)

axA.set_xlim(0, 14)
axA.set_ylim(-1.25, 3.95)
axA.set_yticks([])
axA.set_xlabel("Recoverable AEP (%)")
axA.grid(axis="x", zorder=0)
axA.grid(axis="y", visible=False)

annot_a = ("Re-arranging the point lattice recovers 13-22x more "
           "than orientation; the densest farms exceed +40%.")
axA.text(0.5, 0.135, textwrap.fill(annot_a, 46),
         transform=axA.transAxes, ha="center", va="center",
         fontsize=6.1, color=PAL["ink"], linespacing=1.2,
         bbox=dict(boxstyle="round,pad=0.4", fc="white",
                   ec=PAL["light"], lw=0.6, alpha=0.92))
panel_label(axA, "a")

# ======================================================================
# Panel (b) -- real spacing distribution vs ideal 5D baseline
# ======================================================================
xmax = 11.8
bins = np.arange(0, 12.0 + 1e-9, 0.5)
axB.hist(spacing, bins=bins, density=True, color=PAL["baseline"],
         alpha=0.50, edgecolor="white", linewidth=0.4, zorder=2)

grid = np.linspace(0, xmax, 400)
kde = gaussian_kde(spacing)
axB.plot(grid, kde(grid), color=PAL["baseline"], lw=1.6, zorder=4)

axB.axvspan(0, 5, color=PAL["accent"], alpha=0.12, zorder=0)

ytop = axB.get_ylim()[1]
axB.axvline(5, color=PAL["ink"], ls="--", lw=1.2, zorder=5)
axB.text(5.16, ytop * 0.52, "ideal 5D array",
         rotation=90, va="center", ha="left", fontsize=6.6, color=PAL["ink"])
axB.axvline(med, color=PAL["highlight"], ls=":", lw=1.2, zorder=5)
axB.text(med - 0.16, ytop * 0.52,
         "median %.2fD" % med, rotation=90, va="center", ha="right",
         fontsize=6.6, color=PAL["highlight"])

axB.set_xlim(0, xmax)
axB.set_xlabel("Nearest-neighbour spacing (D)")
axB.set_ylabel("Probability density")
axB.grid(axis="y", zorder=0)
axB.grid(axis="x", visible=False)

annot_b = ("%.1f%% of real farms start denser than the ideal 5D "
           "baseline -> a real-minus-ideal gap that lattice "
           "optimisation can exploit" % pct_below5)
axB.text(0.965, 0.935, textwrap.fill(annot_b, 30),
         transform=axB.transAxes, ha="right", va="top",
         fontsize=6.1, color=PAL["ink"], linespacing=1.15,
         bbox=dict(boxstyle="round,pad=0.4", fc="white",
                   ec=PAL["light"], lw=0.6, alpha=0.92))
axB.text(0.965, 0.045, "n = %d farms" % n_farms, transform=axB.transAxes,
         ha="right", va="bottom", fontsize=6.4, color=PAL["neutral"])
panel_label(axB, "b")

# ======================================================================
# Panel (c) -- POC recovered AEP vs real spacing (denser -> more recovery)
# ======================================================================
sizes = poc_n / poc_n.max() * 150 + 18
axC.scatter(poc_sp, poc_pct, s=sizes, color=PAL["highlight"], alpha=0.60,
            edgecolor="white", linewidth=0.5, zorder=3)

# log-fit trend (recovery falls as spacing grows)
coef = np.polyfit(np.log(poc_sp), poc_pct, 1)
xx = np.linspace(poc_sp.min(), poc_sp.max(), 100)
axC.plot(xx, np.polyval(coef, np.log(xx)), color=PAL["ink"], lw=1.3,
         ls="--", zorder=4)

# annotate the two densest, highest-recovery farms
for sp, pc, tag in [(1.7, 43.7, "F83 (1.7D)"), (3.1, 44.8, "F153 (3.1D)")]:
    axC.annotate(tag, xy=(sp, pc), xytext=(sp + 0.5, pc - 3.0),
                 fontsize=6.2, color=PAL["ink"],
                 arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))

axC.axhline(0.92, color=PAL["neutral"], ls=":", lw=1.0, zorder=2)
axC.text(5.3, 2.6, "orientation +0.9%", fontsize=6.2, color=PAL["neutral"],
         ha="right", va="bottom")

axC.set_xlim(1.2, 5.9)
axC.set_ylim(-4, 50)
axC.set_xlabel("Real nearest-neighbour spacing (D)")
axC.set_ylabel("Point-lattice recovered AEP (%)")
axC.grid(True, color="#E2E8F0", linewidth=0.5, alpha=0.8, zorder=0)
axC.text(0.96, 0.60, "marker size\n∝ turbines", transform=axC.transAxes,
         ha="right", va="top", fontsize=6.0, color=PAL["neutral"],
         linespacing=1.1)
panel_label(axC, "c")

# ----------------------------------------------------------------------
savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
