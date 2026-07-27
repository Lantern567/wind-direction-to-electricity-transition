import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_style import PAL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DD = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived"
REPO = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig1_tail_thresholds.png")

og = pd.read_csv(os.path.join(DD, "orientation_gain.csv"))
fg = og.groupby("farm_id")["gain_pct"].mean()

# endogenous rulers, computed from the paper's own data
t2 = pd.read_csv(os.path.join(REPO, "offshore-task2", "output", "task2_annual_floris.csv"),
                 encoding="utf-8-sig")
t2 = t2[t2.wake_model == "gauss"]
cv = t2.groupby("farm_id")["CF"].agg(lambda v: v.std(ddof=0) / v.mean() * 100 if len(v) >= 5 else np.nan)
NOISE = float(np.nanmedian(cv))                                      # 5.17
poc = pd.read_csv(os.path.join(DD, "qiming_poc_analysis.csv"))
GEO = float(poc["AEP_pct"].median())                                 # 6.17

vals = np.sort(fg.to_numpy())[::-1]
n = len(vals)
n_noise = int((vals > NOISE).sum())
n_geo = int((vals > GEO).sum())
n_2 = int((vals > 2).sum())
print("noise=%.2f geo=%.2f | >noise %d, >geo %d, >2%% %d"
      % (NOISE, GEO, n_noise, n_geo, n_2))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained",
                               gridspec_kw={"width_ratios": [1.35, 1.0]})

# ======================================================================
# (a) ranked farm-level gains with the two endogenous rulers
# ======================================================================
x = np.arange(1, n + 1)
cols = [PAL["highlight"] if v > NOISE else (PAL["accent"] if v > 2 else PAL["light"])
        for v in vals]
axA.bar(x, np.maximum(vals, 0), width=1.0, color=cols, linewidth=0)
neg = vals < 0
axA.bar(x[neg], vals[neg], width=1.0, color=PAL["neutral"], linewidth=0)

axA.axhline(NOISE, color=PAL["ink"], ls="--", lw=1.0, zorder=5)
axA.text(169, NOISE + 0.35, "wind-year noise floor  %.1f%%" % NOISE,
         ha="right", fontsize=6.4, color=PAL["ink"])
axA.axhline(GEO, color=PAL["baseline"], ls="-.", lw=1.0, zorder=5)
axA.text(169, GEO + 0.35, "geometry lever, median  %.1f%%" % GEO,
         ha="right", fontsize=6.4, color=PAL["baseline"])
axA.axhline(np.median(vals), color=PAL["neutral"], ls=":", lw=0.9)
axA.text(169, np.median(vals) - 1.35, "median  +%.2f%%" % np.median(vals),
         ha="right", fontsize=6.2, color=PAL["neutral"])

for fid, lab, dx in [(57, "F57 Vietnam +18.2%", 20), (155, "F155 Italy +9.6%", 22),
                     (66, "F66 China +7.8%", 24)]:
    v = fg.loc[fid]
    r = int(np.where(vals == v)[0][0]) + 1
    axA.annotate(lab, xy=(r, v), xytext=(r + dx, v - 0.3), fontsize=6.2,
                 color=PAL["ink"], va="center",
                 arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))

axA.set_xlim(0, 172)
axA.set_ylim(-2, 19.5)
axA.set_xlabel("Farm rank")
axA.set_ylabel("Mean orientation gain (%)")
axA.text(0.985, 0.80,
         "%d farms above the noise floor\n%d farms above the geometry lever\n%d farms above 2%%"
         % (n_noise, n_geo, n_2),
         transform=axA.transAxes, ha="right", va="center", fontsize=6.4,
         color=PAL["ink"], linespacing=1.45,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PAL["light"], lw=0.6))
panel_label(axA, "a")

# ======================================================================
# (b) ECDF with the same rulers
# ======================================================================
xs = np.sort(fg.to_numpy())
ys = np.arange(1, n + 1) / n * 100
axB.step(xs, ys, where="post", color=PAL["baseline"], lw=1.6, zorder=4)

axB.axvline(NOISE, color=PAL["ink"], ls="--", lw=1.0)
axB.axvline(GEO, color=PAL["baseline"], ls="-.", lw=1.0, alpha=0.7)
pct_noise = (fg <= NOISE).mean() * 100
pct_geo = (fg <= GEO).mean() * 100
axB.annotate("%.0f%% of farms below\nthe noise floor" % pct_noise,
             xy=(NOISE, pct_noise), xytext=(7.6, 62), fontsize=6.4,
             color=PAL["ink"], linespacing=1.25,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))
axB.annotate("%.0f%% below the\ngeometry lever" % pct_geo,
             xy=(GEO, pct_geo), xytext=(9.6, 84), fontsize=6.4,
             color=PAL["baseline"], linespacing=1.25,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))

axB.set_xlim(-2, 19.5)
axB.set_ylim(0, 103)
axB.set_xlabel("Mean orientation gain (%)")
axB.set_ylabel("Cumulative share of farms (%)")
panel_label(axB, "b")

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
