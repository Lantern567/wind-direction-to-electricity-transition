import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆"
REPO = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition"
DD = os.path.join(BASE, "data_derived")
OUT = os.path.join(BASE, "figures", "Fig8_orientation_tail.png")

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
og = pd.read_csv(os.path.join(DD, "orientation_gain.csv"))
mp = pd.read_csv(os.path.join(DD, "farm_wakeloss_map.csv")).set_index("farm_id")
t2 = pd.read_csv(os.path.join(REPO, "offshore-task2", "output", "task2_annual_floris.csv"),
                 encoding="utf-8-sig")
t2 = t2[t2.wake_model == "gauss"]
wm = pd.read_csv(os.path.join(REPO, "task1_output", "task1_wind_metrics.csv"),
                 encoding="utf-8-sig")

fg = og.groupby("farm_id")["gain_pct"].mean()                       # 11-yr farm mean
n_final = t2.groupby("farm_id")["n_turb"].max()                     # final build-out
wci = wm.groupby("farm_id")["WCI_yearly"].mean()                    # same id scheme (verified)

df = pd.DataFrame({"gain": fg}).join(mp[["wl_mean_pct", "centroid_lon",
                                         "centroid_lat", "country"]])
df["n_final"] = n_final
df["wci"] = wci
df = df.dropna(subset=["gain", "wl_mean_pct"])

r_pool = float(np.corrcoef(df.dropna(subset=["wl_mean_pct"])["wl_mean_pct"],
                           df.dropna(subset=["wl_mean_pct"])["gain"])[0, 1])
sub = df.dropna(subset=["wci"])
r_wci = float(np.corrcoef(sub["wci"], sub["gain"])[0, 1])
hiW = sub["wci"] > sub["wci"].quantile(0.75)
hiL = sub["wl_mean_pct"] > sub["wl_mean_pct"].quantile(0.75)
inter_gain = sub.loc[hiW & hiL, "gain"].mean()
print("r(pool)=%.2f r(WCI)=%.2f interaction(n=%d)=%.2f%%"
      % (r_pool, r_wci, (hiW & hiL).sum(), inter_gain))

TOP7 = [57, 155, 66, 157, 91, 126, 159]

# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.15])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
axD = fig.add_subplot(gs[1, 1])

# ======================================================================
# (a) the long tail: 171 farms ranked by mean orientation gain
# ======================================================================
vals = np.sort(df["gain"].to_numpy())[::-1]
x = np.arange(1, len(vals) + 1)
cols = [PAL["highlight"] if v > 5 else (PAL["accent"] if v > 2 else PAL["light"])
        for v in vals]
axA.bar(x, np.maximum(vals, 0), width=1.0, color=cols, linewidth=0)
neg = vals < 0
axA.bar(x[neg], vals[neg], width=1.0, color=PAL["neutral"], linewidth=0)

axA.axhline(np.median(vals), color=PAL["ink"], ls=":", lw=0.9)
axA.text(168, np.median(vals) + 0.35, "median +0.34%", ha="right", fontsize=6.2,
         color=PAL["ink"])
axA.axhline(5, color=PAL["highlight"], ls="--", lw=0.8, alpha=0.7)
axA.text(168, 5.3, "5%", ha="right", fontsize=6.2, color=PAL["highlight"])

for rank, (fid, lab) in enumerate([(57, "F57 Vietnam +18.2%"),
                                   (155, "F155 Italy +9.6%"),
                                   (66, "F66 China +7.8%")]):
    v = df.loc[fid, "gain"]
    r = int(np.where(vals == v)[0][0]) + 1
    axA.annotate(lab, xy=(r, v), xytext=(r + 22, v - 0.4), fontsize=6.2,
                 color=PAL["ink"], va="center",
                 arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))

axA.set_xlim(0, 172)
axA.set_ylim(-2, 19.5)
axA.set_xlabel("Farm rank")
axA.set_ylabel("Mean orientation gain (%)")
axA.text(0.98, 0.55, "7 farms > 5%  (4.1%)\n26 farms > 2%  (15%)\n145 farms < 2%",
         transform=axA.transAxes, ha="right", va="center", fontsize=6.4,
         color=PAL["ink"], linespacing=1.4,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PAL["light"], lw=0.6))
panel_label(axA, "a")

# ======================================================================
# (b) mechanism: gain vs wake pool, colored by wind-direction concentration
# ======================================================================
sc = axB.scatter(sub["wl_mean_pct"], sub["gain"], c=sub["wci"], cmap="Blues",
                 vmin=0.08, vmax=0.42, s=22, edgecolor=PAL["neutral"],
                 linewidth=0.3, alpha=0.9, zorder=3)
cb = fig.colorbar(sc, ax=axB, shrink=0.85, pad=0.02, aspect=22)
cb.set_label("Wind-direction concentration (WCI)", fontsize=6.5)
cb.ax.tick_params(labelsize=6, width=0.5)
cb.outline.set_linewidth(0.4)

for fid, dx, dy in [(57, -6, 0.6), (155, 1.2, 0.7), (66, 1.2, 0.4), (91, 1.2, 0.3)]:
    r = df.loc[fid]
    axB.annotate("F%d" % fid, xy=(r.wl_mean_pct, r.gain),
                 xytext=(r.wl_mean_pct + dx, r.gain + dy), fontsize=6.0,
                 color=PAL["ink"])

axB.text(0.03, 0.965,
         "r(gain, wake pool) = +0.36\nr(gain, WCI) = +0.04\nnarrow rose × large pool:\nmean gain 2.4% (2.5× sample)",
         transform=axB.transAxes, ha="left", va="top", fontsize=6.2,
         color=PAL["ink"], linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PAL["light"], lw=0.6))
axB.set_xlabel("Wake-loss pool (%)")
axB.set_ylabel("Mean orientation gain (%)")
axB.set_ylim(-2, 19.5)
panel_label(axB, "b")

# ======================================================================
# (c) geography of the tail: East Asia monsoon coast
# ======================================================================
axC.set_extent([103.5, 124.5, 7.0, 34.5], crs=ccrs.PlateCarree())
axC.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F1F3F5", edgecolor="none")
axC.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#FFFFFF", edgecolor="none")
axC.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3,
                edgecolor=PAL["neutral"])
axC.spines["geo"].set_linewidth(0.5)

ea = df[df.centroid_lon > 45]
for _, r in ea.iterrows():
    if r.gain > 5:
        c, s, z = PAL["highlight"], 46, 6
    elif r.gain > 2:
        c, s, z = PAL["accent"], 26, 5
    else:
        c, s, z = PAL["light"], 9, 4
    axC.scatter(r.centroid_lon, r.centroid_lat, s=s, color=c, edgecolor="white",
                linewidth=0.3, transform=ccrs.PlateCarree(), zorder=z)

axC.annotate("Mekong Delta cluster\nF57 +18.2 / F126 +5.2 / F159 +5.2\nwinter NE-monsoon corridor",
             xy=(106.2, 9.4), xytext=(104.5, 15.5), transform=ccrs.PlateCarree(),
             fontsize=5.9, color=PAL["ink"], linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axC.annotate("Hangzhou Bay F66 +7.8\n(bay-mouth rectification)",
             xy=(121.5, 30.5), xytext=(111.5, 30.8), transform=ccrs.PlateCarree(),
             fontsize=5.9, color=PAL["ink"], linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axC.annotate("Pearl River mouth F91 +6.6\n(WCI 0.38, highest of top-7)",
             xy=(113.4, 21.9), xytext=(112.6, 25.6), transform=ccrs.PlateCarree(),
             fontsize=5.9, color=PAL["ink"], linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axC.text(0.975, 0.03,
         "Europe (not shown): F155 Taranto +9.6,\nF157 Danish strait +6.9; 9 farms > 2%",
         transform=axC.transAxes, ha="right", va="bottom", fontsize=5.8,
         color=PAL["neutral"], linespacing=1.25,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

handles = [Line2D([], [], marker="o", ls="", color=PAL["highlight"], ms=6,
                  label="gain > 5%"),
           Line2D([], [], marker="o", ls="", color=PAL["accent"], ms=5,
                  label="2–5%"),
           Line2D([], [], marker="o", ls="", color=PAL["light"], ms=3.5,
                  label="< 2%")]
axC.legend(handles=handles, loc="upper left", fontsize=6.0, handlelength=0.8,
           labelspacing=0.35, borderaxespad=0.3, framealpha=0.9)
panel_label(axC, "c")

# ======================================================================
# (d) stability across years and build-out
# ======================================================================
farm_style = [(57,  PAL["highlight"], "F57 Vietnam (48→80 turbines)"),
              (155, PAL["accent"],    "F155 Italy (2→9)"),
              (66,  PAL["baseline"],  "F66 Hangzhou Bay (3→74)"),
              (157, PAL["neutral"],   "F157 Denmark (7, unchanged)")]
n_by = t2.set_index(["farm_id", "year"])["n_turb"]
for fid, c, lab in farm_style:
    s = og[og.farm_id == fid].sort_values("year")
    axD.plot(s.year, s.gain_pct, "o-", color=c, lw=1.3, ms=3.2, label=lab, zorder=4)

axD.annotate("config jump n=2→9", xy=(2021.1, 1.9), xytext=(2016.4, 3.6),
             fontsize=5.9, color=PAL["accent"],
             arrowprops=dict(arrowstyle="-", color=PAL["accent"], lw=0.6))
axD.axhline(0.92, color=PAL["ink"], ls=":", lw=0.8)
axD.text(2014.1, 1.35, "global mean +0.9%", fontsize=6.0, color=PAL["ink"])

axD.set_xlabel("Year")
axD.set_ylabel("Orientation gain (%)")
axD.set_xlim(2013.5, 2024.5)
axD.set_ylim(-0.8, 23.5)
axD.legend(loc="upper right", fontsize=5.9, handlelength=1.4, labelspacing=0.35,
           borderaxespad=0.3)
axD.text(0.03, 0.55, "F57: 16–22% every year,\nrobust across build-out",
         transform=axD.transAxes, fontsize=6.2, color=PAL["highlight"],
         va="top", linespacing=1.25)
panel_label(axD, "d")

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
