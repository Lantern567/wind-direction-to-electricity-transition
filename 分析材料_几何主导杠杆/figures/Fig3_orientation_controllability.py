import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"d:/onedrive/01_科研与论文/08_风向建设"
GAIN_CSV = BASE + r"/分析材料_几何主导杠杆/data_derived/orientation_gain.csv"
DEV_CSV  = BASE + r"/wind-direction-to-electricity-transition/Task3-output/s4_farm_deviation.csv"
ROT_CSV  = BASE + r"/wind-direction-to-electricity-transition/offshore-task2/output/audit_rotation_floris.csv"
OUT_PNG  = BASE + r"/分析材料_几何主导杠杆/figures/Fig3_orientation_controllability.png"

# ---------------------------------------------------------------- load data
gain = pd.read_csv(GAIN_CSV)["gain_pct"].to_numpy()
dev  = pd.read_csv(DEV_CSV)["axis_deviation"].dropna().abs().to_numpy()
rot  = pd.read_csv(ROT_CSV)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), constrained_layout=True)
axa, axb, axc, axd = axes.ravel()

# ================================================================ panel a
XLO, XHI = -5, 8
ga = np.clip(gain, XLO, XHI)
bins = np.arange(XLO, XHI + 0.5, 0.5)
axa.axvspan(XLO, 0, color=PAL["light"], alpha=0.30, zorder=0, lw=0)
axa.hist(ga, bins=bins, color=PAL["neutral"], edgecolor="white", linewidth=0.35, zorder=2)
axa.axvline(0.92, color=PAL["highlight"], lw=1.4, zorder=4, label="mean +0.9%")
axa.axvline(0.28, color=PAL["ink"], lw=1.1, ls="--", zorder=4, label="median +0.3%")
axa.set_xlim(XLO, XHI)
axa.set_xlabel("AEP gain from rigid rotation to optimal orientation (%)")
axa.set_ylabel("Farm-years (count)")
axa.legend(loc="upper right", handlelength=1.4, borderaxespad=0.3)
axa.text(0.035, 0.965,
         "n = 1203 farm-years\nwin 64.3%\npaired t  p = 3e-44\n(robust but tiny)",
         transform=axa.transAxes, va="top", ha="left", fontsize=6.8,
         color=PAL["ink"], linespacing=1.35)
panel_label(axa, "a")

# ================================================================ panel b
bbins = np.arange(0, 95, 7.5)
axb.hist(dev, bins=bbins, color=PAL["baseline"], edgecolor="white", linewidth=0.35, zorder=2)
axb.axvline(51.2, color=PAL["ink"], lw=1.2, ls="--", zorder=4, label="median 51.2 deg")
axb.set_xlim(0, 90)
axb.set_xlabel("|Actual - optimal orientation| (deg)")
axb.set_ylabel("Farms (count)")
axb.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
axb.text(0.965, 0.965,
         "n = 167 farms\n70% are > 30 deg\noff the optimum,\nyet lose only ~1%",
         transform=axb.transAxes, va="top", ha="right", fontsize=6.8,
         color=PAL["ink"], linespacing=1.35)
panel_label(axb, "b")

# ================================================================ panel c
countries = [
    ("Italy",           9.57, True,  False),
    ("Vietnam",         4.92, False, False),
    ("Ireland",         2.41, False, False),
    ("United States",   1.41, False, False),
    ("Taiwan",          1.17, False, False),
    ("Germany",         0.91, False, False),
    ("China",           0.77, False, False),
    ("United Kingdom",  0.72, False, False),
    ("Denmark",         0.65, False, False),
    ("Netherlands",     0.38, False, False),
    ("France",         -0.03, False, True),
    ("Japan",          -0.32, False, True),
]
countries.sort(key=lambda r: r[1], reverse=True)
names  = [c[0] + (" *" if c[2] else "") for c in countries]
vals   = [c[1] for c in countries]
ns     = [c[3] for c in countries]
ypos   = np.arange(len(countries))[::-1]  # first = top
cols   = [PAL["neutral"] if is_ns else PAL["positive"] for is_ns in ns]
axc.barh(ypos, vals, color=cols, edgecolor="white", linewidth=0.35, height=0.72, zorder=2)
axc.axvline(0, color=PAL["ink"], lw=0.7, zorder=3)
axc.set_yticks(ypos)
axc.set_yticklabels(names, fontsize=6.6)
axc.set_xlim(-1.6, 11.2)
axc.set_xlabel("Mean AEP gain (%)")
for y, v, is_ns in zip(ypos, vals, ns):
    lab = f"{v:+.2f}" + ("  ns" if is_ns else "")
    if v >= 0:
        axc.text(v + 0.18, y, lab, va="center", ha="left", fontsize=6.2, color=PAL["ink"])
    else:
        axc.text(0.15, y, lab, va="center", ha="left", fontsize=6.2, color=PAL["neutral"])
axc.text(0.985, 0.045, "* small sample (n = 4)",
         transform=axc.transAxes, va="bottom", ha="right", fontsize=6.2, color=PAL["ink"])
panel_label(axc, "c")

# ================================================================ panel d
farm_order = (rot.groupby("farm_id")["wake_efficiency"]
                 .agg(lambda s: s.max() - s.min())
                 .sort_values(ascending=False).index.tolist())[:4]
dcols = {farm_order[0]: PAL["highlight"], farm_order[1]: PAL["baseline"],
         farm_order[2]: PAL["accent"],    farm_order[3]: PAL["positive"]}
for fid in farm_order:
    s = rot[rot.farm_id == fid].sort_values("angle_deg")
    axd.plot(s["angle_deg"], s["wake_efficiency"], color=dcols[fid],
             lw=1.3, marker="o", ms=2.6, mew=0, label=f"Farm {fid}", zorder=3)
axd.set_xlim(0, 170)
axd.set_ylim(0.40, 0.96)
axd.set_xlabel("Rigid rotation angle (deg)")
axd.set_ylabel("Wake efficiency")
axd.legend(loc="upper right", ncol=2, columnspacing=1.0,
           handlelength=1.4, borderaxespad=0.3)
axd.text(0.5, 0.05,
         "Single-direction sensitivity is large, but the real\n"
         "multidirectional wind rose collapses the recoverable\n"
         "gain to ~1% (panel a)",
         transform=axd.transAxes, va="bottom", ha="center", fontsize=6.3,
         color=PAL["ink"], linespacing=1.3,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PAL["light"], lw=0.6, alpha=0.9))
panel_label(axd, "d")

# ---------------------------------------------------------------- save
savefig(fig, OUT_PNG)
print("bytes:", os.path.getsize(OUT_PNG))
