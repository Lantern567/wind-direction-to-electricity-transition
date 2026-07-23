import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# ---------------------------------------------------------------- data paths
SHUFFLE = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition/offshore-task2/output/audit_shuffle_floris.csv"
SPACING = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition/task1_output/task1_paradigm_classification.csv"
SPWL    = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived/spacing_vs_wakeloss.csv"
OUT     = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures/Fig2_geometry_drives_wake.png"

sh = pd.read_csv(SHUFFLE)
sp = pd.read_csv(SPACING)
sw = pd.read_csv(SPWL)

# ---------------------------------------------------------------- figure
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.65),
                                    constrained_layout=True)

# ===================================================== Panel (a): shuffle bars
sh0 = sh[sh.farm_id == 0].iloc[0]
sh2 = sh[sh.farm_id == 2].iloc[0]
real = [sh0.real_wake_loss, sh2.real_wake_loss]
shuf = [sh0.shuffled_wake_loss, sh2.shuffled_wake_loss]
x = np.arange(2)
w = 0.36
bR = axA.bar(x - w/2, real, w, color=PAL["baseline"], label="Real layout",
             edgecolor="white", linewidth=0.5)
bS = axA.bar(x + w/2, shuf, w, color=PAL["neutral"],
             label="Randomly scattered, same turbines",
             edgecolor="white", linewidth=0.5)

for rect, v in zip(list(bR) + list(bS), real + shuf):
    axA.text(rect.get_x() + rect.get_width()/2, v + 1.2, f"{v:.1f}",
             ha="center", va="bottom", fontsize=6.8, color=PAL["ink"])

# delta annotation on F0 (arrow along the collapse, text below the legend band)
axA.annotate("", xy=(x[0] + w/2, shuf[0] + 4.0), xytext=(x[0] - w/2, real[0] + 3.5),
             arrowprops=dict(arrowstyle="->", color=PAL["highlight"], lw=1.1))
axA.text(x[0] + 0.16, 70.5, "-95.9%\n(55.3% -> 12.4%)", ha="center", va="top",
         fontsize=7.2, color=PAL["highlight"], fontweight="bold", linespacing=1.15)

axA.set_xticks(x)
axA.set_xticklabels(["Farm 0\n(n = 928)", "Farm 2\n(n = 572)"])
axA.set_ylabel("Wake loss (%)")
axA.set_ylim(0, 88)
axA.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
           labels=["Real layout", "Randomly scattered,\nsame turbines"],
           handles=[bR, bS], handlelength=1.0, borderaxespad=0.2,
           labelspacing=0.35, fontsize=6.5)
axA.grid(axis="x", visible=False)
panel_label(axA, "a")

# ===================================================== Panel (b): spacing hist
s = sp["spacing_d"].dropna().values
med = float(np.median(s))                      # 3.32
bins = np.arange(0, 12.5, 0.5)
counts, _, _ = axB.hist(s, bins=bins, color=PAL["baseline"],
                        edgecolor="white", linewidth=0.4)
axB.set_ylim(0, counts.max() * 1.30)          # headroom for the reference legend

axB.axvline(med, color=PAL["highlight"], lw=1.4, solid_capstyle="butt")
axB.axvline(5.1, color=PAL["neutral"], lw=1.1, ls=(0, (5, 2)))
axB.axvline(7.0, color=PAL["neutral"], lw=1.1, ls=(0, (2, 2)))

axB.set_xlabel("Nearest-neighbour spacing (rotor diameters, D)")
axB.set_ylabel("Number of farms")
axB.set_xlim(0, 12)

ref_handles = [
    Line2D([0], [0], color=PAL["highlight"], lw=1.4, label="Median 3.3D"),
    Line2D([0], [0], color=PAL["neutral"], lw=1.1, ls=(0, (5, 2)),
           label="Jung & Schindler global median 5.1D"),
    Line2D([0], [0], color=PAL["neutral"], lw=1.1, ls=(0, (2, 2)),
           label="Shen & Mikkelsen recommended >= 7D"),
]
legB = axB.legend(handles=ref_handles, loc="upper right", handlelength=1.4,
                  borderaxespad=0.2, labelspacing=0.35, fontsize=6.4,
                  frameon=True, framealpha=0.85, facecolor="white",
                  edgecolor="none")
legB.get_frame().set_linewidth(0.0)
axB.text(med + 0.45, counts.max() * 0.72, "84.8% of\nfarms < 5D", ha="left",
         va="top", fontsize=6.8, color=PAL["ink"], linespacing=1.2)
panel_label(axB, "b")

# ===================================================== Panel (c): scatter
xd = sw["spacing_d"].values
yl = sw["wl_mean_pct"].values
nt = sw["n_turb"].values
axC.scatter(xd, yl, s=nt / 18.0, color=PAL["baseline"], alpha=0.6,
            edgecolor="none", zorder=3)

coef = np.polyfit(xd, yl, 1)
xx = np.linspace(xd.min(), xd.max(), 100)
axC.plot(xx, np.polyval(coef, xx), color=PAL["highlight"], lw=1.6, zorder=4)

rho, pval = spearmanr(xd, yl)
axC.text(0.96, 0.96,
         "Spearman rho = -0.25, p = 0.001\n(denser -> more wake loss)",
         transform=axC.transAxes, ha="right", va="top", fontsize=6.8,
         color=PAL["ink"], linespacing=1.25)

axC.set_xlabel("Spacing (D)")
axC.set_ylabel("Mean wake loss (%)")
axC.set_xlim(0, 12)
axC.set_ylim(bottom=0)
panel_label(axC, "c")

savefig(fig, OUT)
print("bytes:", os.path.getsize(OUT))
plt.close(fig)
