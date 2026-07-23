import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import csv, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆"
REPO = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition"
DD   = os.path.join(BASE, "data_derived")
OUT  = os.path.join(BASE, "figures", "Fig6_windspeed_role.png")

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
wsbin = pd.read_csv(os.path.join(DD, "wakeloss_by_wsbin.csv"))
sweep = pd.read_csv(os.path.join(DD, "wakeloss_sweep.csv"))
wstat = pd.read_csv(os.path.join(DD, "farm_ws_stats.csv"))

t2 = pd.read_csv(os.path.join(REPO, "offshore-task2", "output", "task2_annual_floris.csv"),
                 encoding="utf-8-sig")
t2 = t2[t2.wake_model == "gauss"]
cls = pd.read_csv(os.path.join(REPO, "task1_output", "task1_paradigm_classification.csv"),
                  encoding="utf-8-sig")
fm = pd.read_csv(os.path.join(REPO, "offshore-task0-HuTingxian", "output", "task0",
                              "farms_master.csv"), encoding="utf-8-sig")

# farm-level means
farm_wl = t2.groupby("farm_id")["WakeLoss"].mean() * 100
farm_cf = t2.groupby("farm_id")["CF"].mean()
farm_ws = wstat.groupby("farm_id")["ws_mean"].mean()

# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), layout="constrained")
axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# ======================================================================
# Panel (a) -- wake loss vs hub wind speed: observed + controlled sweeps
# ======================================================================
obs = wsbin[(wsbin.ws_bin_m_s >= 3) & (wsbin.ws_bin_m_s <= 20)]
g = obs.groupby("ws_bin_m_s")
ws_x = np.array(sorted(g.groups.keys()))
med = g["wakeloss_pct"].median().reindex(ws_x).to_numpy()
p25 = g["wakeloss_pct"].quantile(0.25).reindex(ws_x).to_numpy()
p75 = g["wakeloss_pct"].quantile(0.75).reindex(ws_x).to_numpy()
esh = g["energy_share"].mean().reindex(ws_x).to_numpy() * 100  # %

axA.fill_between(ws_x, p25, p75, color=PAL["baseline"], alpha=0.15, zorder=2)
axA.plot(ws_x, med, "o-", color=PAL["baseline"], lw=1.6, ms=3.2, zorder=4,
         label="Observed, 171 farms (median, IQR)")

for lab, col, ls, txt in [("standard_8x8_5D", PAL["highlight"], (0, (5, 2)),
                           "Controlled sweep: 8x8 5D array"),
                          ("real_F99", PAL["accent"], (0, (2, 2)),
                           "Controlled sweep: real dense farm")]:
    s = sweep[sweep.layout_label == lab]
    axA.plot(s.ws_m_s, s.wakeloss_pct, color=col, ls=ls, lw=1.4, zorder=3, label=txt)

axA.axvline(11, color=PAL["neutral"], ls=":", lw=1.0, zorder=2)
axA.text(11.25, 74, "rated\n~11 m/s", fontsize=6.2, color=PAL["neutral"],
         va="top", linespacing=1.1)

# energy share as faint bars on twin axis (caveat: low-ws bins carry no energy)
axA2 = axA.twinx()
axA2.bar(ws_x, esh, width=0.7, color=PAL["light"], alpha=0.45, zorder=1)
axA2.set_ylabel("Mean energy share per bin (%)", color=PAL["neutral"], fontsize=7)
axA2.tick_params(axis="y", labelcolor=PAL["neutral"], labelsize=6.5)
axA2.set_ylim(0, 40)
axA2.grid(False)
axA2.spines["top"].set_visible(False)

axA.set_xlim(2.5, 20)
axA.set_ylim(0, 92)
axA.set_xlabel("Hub-height wind speed (m/s)")
axA.set_ylabel("Wake loss (%)")
axA.legend(loc="upper right", fontsize=6.0, handlelength=1.6, labelspacing=0.35,
           borderaxespad=0.2)
axA.text(0.985, 0.42, "high relative loss below 5 m/s,\nbut those bins carry\nalmost no energy (grey bars)",
         transform=axA.transAxes, ha="right", va="top", fontsize=6.0,
         color=PAL["neutral"], linespacing=1.2,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
panel_label(axA, "a")

# ======================================================================
# Panel (b) -- attribution defence: wind collapses, spacing survives
# ======================================================================
# computed on the n=72 subsample with WPD_hist available
sub = cls.dropna(subset=["WPD_hist", "spacing_d"]).copy()
sub = sub[sub.farm_id.isin(farm_wl.index)]
W = sub["WPD_hist"].to_numpy()
S = sub["spacing_d"].to_numpy()
L = farm_wl.reindex(sub.farm_id).to_numpy()

def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])

def resid(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)

r_w_raw = pear(W, L)                      # -0.25
r_w_par = pear(resid(W, S), resid(L, S))  # -0.00
r_s_raw = pear(S, L)                      # -0.52
r_s_par = pear(resid(S, W), resid(L, W))  # -0.47

x = np.array([0, 1, 2.6, 3.6])
vals = [r_w_raw, r_w_par, r_s_raw, r_s_par]
cols = [PAL["neutral"], PAL["light"], PAL["baseline"], PAL["baseline"]]
bars = axB.bar(x, vals, 0.8, color=cols, edgecolor="white", linewidth=0.5, zorder=3)
bars[3].set_alpha(0.65)

for xi, v in zip(x, vals):
    axB.text(xi, v - 0.028, "%.2f" % v, ha="center", va="top", fontsize=7.0,
             color=PAL["ink"])

axB.axhline(0, color=PAL["ink"], lw=0.8, zorder=2)
axB.set_xticks(x)
axB.set_xticklabels(["raw", "| spacing", "raw", "| wind"], fontsize=7)
axB.set_ylim(-0.70, 0.14)
axB.set_ylabel("Correlation with wake loss (r)")

axB.text(0.5, -0.685, "wind resource (WPD)\ncollapses to ~0", ha="center",
         va="bottom", fontsize=6.6, color=PAL["neutral"], linespacing=1.15)
axB.text(3.1, -0.685, "spacing\nsurvives", ha="center", va="bottom",
         fontsize=6.6, color=PAL["baseline"], fontweight="bold", linespacing=1.15)
axB.text(0.03, 0.965, "n = %d farms (with WPD)" % len(sub),
         transform=axB.transAxes, ha="left", va="top", fontsize=6.2,
         color=PAL["neutral"])
axB.grid(axis="x", visible=False)
panel_label(axB, "b")

# ======================================================================
# Panel (c) -- CF ~ ws_mean: wind sets the generation ceiling (exogenous)
# ======================================================================
common = farm_cf.index.intersection(farm_ws.index)
cf = farm_cf.reindex(common).to_numpy()
wsm = farm_ws.reindex(common).to_numpy()

# region colours from farms_master centroid longitude
lon = fm.set_index("farm_id")["centroid_lon"].reindex(common).to_numpy()
def region_of(l):
    if l < -30: return "us_east"
    if l < 45:  return "europe"
    return "east_asia"
reg = np.array([region_of(l) for l in lon])
for rname, rlab in [("europe", "Europe"), ("east_asia", "East Asia"),
                    ("us_east", "US East")]:
    m = reg == rname
    axC.scatter(wsm[m], cf[m], s=13, color=REGION_COL[rname], alpha=0.65,
                edgecolor="none", zorder=3, label=rlab)

coef = np.polyfit(wsm, cf, 1)
xx = np.linspace(wsm.min(), wsm.max(), 100)
axC.plot(xx, np.polyval(coef, xx), color=PAL["ink"], lw=1.4, zorder=4)
r_cf = pear(wsm, cf)

axC.text(0.03, 0.965, "r = %.2f  (n = %d)\nslope = +%.3f CF per m/s"
         % (r_cf, len(common), coef[0]),
         transform=axC.transAxes, ha="left", va="top", fontsize=6.8,
         color=PAL["ink"], linespacing=1.25)
axC.legend(loc="lower right", fontsize=6.4, handlelength=1.0, labelspacing=0.3,
           borderaxespad=0.2, markerscale=1.2)
axC.set_xlabel("Mean hub-height wind speed, 2014-2024 (m/s)")
axC.set_ylabel("Capacity factor")
panel_label(axC, "c")

# ======================================================================
# Panel (d) -- wind-year noise vs the two controllable levers
# ======================================================================
# CF interannual CV per farm (>=5 years), median
cv = []
for fid, gdf in t2.groupby("farm_id"):
    v = gdf["CF"].to_numpy()
    if len(v) >= 5 and v.mean() > 0:
        cv.append(v.std() / v.mean() * 100)
cv_med = float(np.median(cv))

xd = np.arange(3)
vals_d = [cv_med, 0.92, 6.2]
cols_d = [PAL["neutral"], PAL["neutral"], PAL["highlight"]]
bars_d = axD.bar(xd, vals_d, 0.58, color=cols_d, edgecolor="white",
                 linewidth=0.5, zorder=3)
bars_d[1].set_alpha(0.55)

for xi, v in zip(xd, vals_d):
    axD.text(xi, v + 0.14, "%.1f%%" % v, ha="center", va="bottom",
             fontsize=7.6, fontweight="bold", color=PAL["ink"])

axD.text(0, cv_med / 2, "exogenous\nuncontrollable", ha="center", va="center",
         fontsize=6.2, color="white", linespacing=1.15)
axD.text(2, 3.1, "controllable\ndominant", ha="center", va="center",
         fontsize=6.2, color="white", fontweight="bold", linespacing=1.15)

axD.set_xticks(xd)
axD.set_xticklabels(["Wind-year noise\n(CF interannual CV,\nmedian)",
                     "Orientation\n(rigid rotation)",
                     "Geometry\n(lattice POC, median)"], fontsize=6.6)
axD.set_ylabel("Magnitude (% of AEP / CF)")
axD.set_ylim(0, 7.3)
axD.grid(axis="x", visible=False)
axD.text(0.5, 0.955, "only geometry rises above the wind-year noise floor",
         transform=axD.transAxes, ha="center", va="top", fontsize=6.6,
         color=PAL["ink"])
panel_label(axD, "d")

# ----------------------------------------------------------------------
savefig(fig, OUT)
print("panel-b: wind raw %.3f -> %.3f | spacing raw %.3f -> %.3f" %
      (r_w_raw, r_w_par, r_s_raw, r_s_par))
print("panel-c: r=%.3f slope=%.4f | panel-d: cv_med=%.2f" % (r_cf, coef[0], cv_med))
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
