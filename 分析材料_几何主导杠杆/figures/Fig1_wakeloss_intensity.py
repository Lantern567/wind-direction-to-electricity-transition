import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE = r"d:/onedrive/01_科研与论文/08_风向建设"
MAP_CSV = BASE + r"/分析材料_几何主导杠杆/data_derived/farm_wakeloss_map.csv"
T2_CSV = BASE + r"/wind-direction-to-electricity-transition/offshore-task2/output/task2_annual_floris.csv"
OUT = BASE + r"/分析材料_几何主导杠杆/figures/Fig1_wakeloss_intensity.png"

# ================================================================ load / stats
mp = pd.read_csv(MAP_CSV).dropna(subset=["wl_mean_pct", "centroid_lon", "centroid_lat"])
t2 = pd.read_csv(T2_CSV)
g = t2[t2["wake_model"] == "gauss"].copy()
g["WL_pct"] = g["WakeLoss"] * 100.0
g["CF_pct"] = g["CF"] * 100.0

wl = g["WL_pct"].values
wl_mean = wl.mean()
wl_median = np.median(wl)
wl_max = wl.max()
frac_over20 = (wl > 20).mean() * 100.0

reg_order = ["europe", "east_asia", "us_east"]
reg_lbl = {"europe": "Europe", "east_asia": "East Asia", "us_east": "US East"}
reg_stat = {r: (g.loc[g.region == r, "CF_pct"].mean(),
                g.loc[g.region == r, "WL_pct"].mean(),
                int((g.region == r).sum())) for r in reg_order}

mdl_order = ["gauss", "jensen", "cc"]
mdl_lbl = {"gauss": "Gauss", "jensen": "Jensen", "cc": "CC"}
mdl_wl = {wm: (t2.loc[t2.wake_model == wm, "WakeLoss"] * 100).mean() for wm in mdl_order}
mdl_cf = {wm: (t2.loc[t2.wake_model == wm, "CF"] * 100).mean() for wm in mdl_order}

# ================================================================ figure layout
fig = plt.figure(figsize=(7.2, 5.9), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.18, 1.0])
axa = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])
axd = fig.add_subplot(gs[1, 2])

# ---------------------------------------------------------------- (a) global map
axa.set_extent([-100, 145, 10, 62], crs=ccrs.PlateCarree())
axa.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F1F3F5", edgecolor="none")
axa.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#FFFFFF", edgecolor="none")
axa.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.35, edgecolor=PAL["neutral"])
axa.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25,
                edgecolor=PAL["light"], alpha=0.7)

gl = axa.gridlines(draw_labels=True, linewidth=0.4, color=PAL["light"],
                   alpha=0.5, linestyle=":", x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 6.5, "color": PAL["ink"]}
gl.ylabel_style = {"size": 6.5, "color": PAL["ink"]}

nt = mp["n_turb"].values.astype(float)
smin, smax = 12.0, 300.0
sizes = smin + (nt - nt.min()) / (nt.max() - nt.min()) * (smax - smin)
norm = Normalize(vmin=0, vmax=np.ceil(mp["wl_mean_pct"].max() / 5) * 5)
sc = axa.scatter(mp["centroid_lon"], mp["centroid_lat"], s=sizes,
                 c=mp["wl_mean_pct"], cmap="YlOrRd", norm=norm,
                 transform=ccrs.PlateCarree(), edgecolor="#3A3A3A",
                 linewidth=0.35, alpha=0.9, zorder=5)

cb = fig.colorbar(sc, ax=axa, orientation="vertical", shrink=0.80,
                  pad=0.008, aspect=24)
cb.set_label("Mean wake loss (%)", fontsize=7.5)
cb.ax.tick_params(labelsize=6.8, width=0.6)
cb.outline.set_linewidth(0.5)

leg_n = [10, 100, 240]
leg_handles = [Line2D([], [], marker="o", linestyle="none",
                      markersize=np.sqrt(smin + (v - nt.min()) / (nt.max() - nt.min()) * (smax - smin)),
                      markerfacecolor="#DCDCDC", markeredgecolor="#3A3A3A",
                      markeredgewidth=0.35, label=str(v)) for v in leg_n]
legn = axa.legend(handles=leg_handles, title="Turbines", loc="lower left",
                  labelspacing=1.2, handletextpad=1.0, borderpad=0.7,
                  fontsize=6.8, title_fontsize=7.0, framealpha=0.9,
                  frameon=True, edgecolor=PAL["light"])
legn.get_frame().set_linewidth(0.4)
panel_label(axa, "a", dx=-0.025, dy=1.11)

# ---------------------------------------------------------------- (b) histogram
xmax_b = np.ceil(wl_max / 5) * 5
bins = np.arange(0, xmax_b + 2.5, 2.5)
axb.hist(wl, bins=bins, color=PAL["baseline"], alpha=0.72,
         edgecolor="white", linewidth=0.4, zorder=2)
ymax = axb.get_ylim()[1]
axb.axvline(wl_mean, color=PAL["highlight"], linestyle="--", linewidth=1.2, zorder=4)
axb.axvline(wl_median, color=PAL["ink"], linestyle=":", linewidth=1.2, zorder=4)
axb.annotate("mean %.1f%%" % wl_mean, xy=(wl_mean, ymax * 0.98),
             xytext=(wl_mean + 4.0, ymax * 0.98), fontsize=6.9,
             color=PAL["highlight"], va="top", ha="left", fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=PAL["highlight"], lw=0.7))
axb.annotate("median %.1f%%" % wl_median, xy=(wl_median, ymax * 0.66),
             xytext=(wl_median + 4.0, ymax * 0.72), fontsize=6.9,
             color=PAL["ink"], va="top", ha="left",
             arrowprops=dict(arrowstyle="-", color=PAL["ink"], lw=0.7))
axb.text(0.965, 0.55,
         "%.1f%% of farm-years\n> 20%%;  max %.1f%%" % (frac_over20, wl_max),
         transform=axb.transAxes, fontsize=6.6, va="top", ha="right",
         color=PAL["ink"],
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                   edgecolor=PAL["light"], linewidth=0.5, alpha=0.92))
axb.text(0.965, 0.86, "Gauss, n = %d" % len(g), transform=axb.transAxes,
         fontsize=6.7, va="top", ha="right", color=PAL["neutral"])
axb.set_xlabel("Wake loss (%)")
axb.set_ylabel("Farm-years (count)")
axb.set_xlim(0, xmax_b)
panel_label(axb, "b")

# ---------------------------------------------------------------- (c) by region
x = np.arange(len(reg_order))
bw = 0.38
cf_vals = [reg_stat[r][0] for r in reg_order]
wl_vals = [reg_stat[r][1] for r in reg_order]
b1 = axc.bar(x - bw / 2, cf_vals, bw, color=PAL["positive"], alpha=0.92,
             label="Capacity factor", edgecolor="white", linewidth=0.5, zorder=2)
b2 = axc.bar(x + bw / 2, wl_vals, bw, color=PAL["highlight"], alpha=0.92,
             label="Wake loss", edgecolor="white", linewidth=0.5, zorder=2)
for rect in list(b1) + list(b2):
    h = rect.get_height()
    axc.text(rect.get_x() + rect.get_width() / 2, h + 0.7, "%.1f" % h,
             ha="center", va="bottom", fontsize=6.5, color=PAL["ink"])
axc.set_xticks(x)
axc.set_xticklabels(["%s\n(n = %d)" % (reg_lbl[r], reg_stat[r][2]) for r in reg_order])
axc.set_ylabel("%")
axc.set_ylim(0, max(cf_vals) * 1.24)
axc.legend(loc="upper right", handlelength=1.1, fontsize=6.7, borderpad=0.4)
panel_label(axc, "c")

# ---------------------------------------------------------------- (d) wake models
xm = np.arange(len(mdl_order))
mwl = [mdl_wl[m] for m in mdl_order]
grey_shades = ["#94A3B8", "#64748B", "#334155"]  # slate light -> dark
bd = axd.bar(xm, mwl, 0.58, color=grey_shades, edgecolor="white",
             linewidth=0.5, zorder=2)
for rect, m in zip(bd, mdl_order):
    h = rect.get_height()
    axd.text(rect.get_x() + rect.get_width() / 2, h + 0.35, "%.1f" % h,
             ha="center", va="bottom", fontsize=7.0, color=PAL["ink"],
             fontweight="bold")
axd.set_xticks(xm)
axd.set_xticklabels([mdl_lbl[m] for m in mdl_order])
axd.set_ylabel("Mean wake loss (%)")
axd.set_ylim(0, max(mwl) * 1.28)
axd.set_xlabel("Wake model")
axd.text(0.5, 0.95, "ordering stable\nacross wake models", transform=axd.transAxes,
         fontsize=6.6, va="top", ha="center", color=PAL["neutral"], style="italic")
panel_label(axd, "d")

savefig(fig, OUT)
print("bytes:", os.path.getsize(OUT))
