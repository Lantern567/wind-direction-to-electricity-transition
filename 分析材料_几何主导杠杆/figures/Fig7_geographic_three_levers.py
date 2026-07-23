import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆"
DD = os.path.join(BASE, "data_derived")
OUT = os.path.join(BASE, "figures", "Fig7_geographic_three_levers.png")

# ----------------------------------------------------------------------
# Data: one signed "effect on generation" (pp) per farm, per factor
#   wind    = (ws_mean - fleet mean) x slope(CF~ws)      [vs fleet average]
#   orient  = - mean rigid-rotation gain                  [vs own optimum]
#   lattice = - mean wake loss                            [vs no-wake ideal]
# ----------------------------------------------------------------------
mp = pd.read_csv(os.path.join(DD, "farm_wakeloss_map.csv")).dropna(
    subset=["wl_mean_pct", "centroid_lon", "centroid_lat"])
og = pd.read_csv(os.path.join(DD, "orientation_gain.csv"))
ws = pd.read_csv(os.path.join(DD, "farm_ws_stats.csv"))

farm_gain = og.groupby("farm_id")["gain_pct"].mean()
farm_ws = ws.groupby("farm_id")["ws_mean"].mean()

df = mp.set_index("farm_id")
df["ws_mean"] = farm_ws
df["orient_pp"] = -farm_gain
df["lattice_pp"] = -df["wl_mean_pct"]

# slope of CF (pp) per m/s across farms -> wind endowment vs fleet mean
ok = df.dropna(subset=["ws_mean", "cf_mean_pct"])
slope = np.polyfit(ok["ws_mean"], ok["cf_mean_pct"], 1)[0]
ws_fleet = ok["ws_mean"].mean()
df["wind_pp"] = (df["ws_mean"] - ws_fleet) * slope
df = df.dropna(subset=["wind_pp", "orient_pp", "lattice_pp"])

print("n farms plotted:", len(df), "| slope=%.2f pp/(m/s), fleet ws=%.2f" % (slope, ws_fleet))
for c in ["wind_pp", "orient_pp", "lattice_pp"]:
    print("  %-10s min=%+.1f  median=%+.1f  max=%+.1f"
          % (c, df[c].min(), df[c].median(), df[c].max()))
us = df[df.region == "us_east"]
print("US East (not mapped):")
for fid, r in us.iterrows():
    print("  farm %s: wind %+.1f, orient %+.1f, lattice %+.1f"
          % (fid, r.wind_pp, r.orient_pp, r.lattice_pp))

# ----------------------------------------------------------------------
# Figure: rows = Europe / East Asia, cols = three factors, one shared ruler
# ----------------------------------------------------------------------
VLIM = 20.0
norm = Normalize(vmin=-VLIM, vmax=VLIM)
CMAP = "RdBu"          # negative = red (loss), positive = blue (gain), gray-white at 0

REGIONS = [
    ("europe",    "Europe",    [-12.0, 17.0, 48.5, 58.5]),
    ("east_asia", "East Asia", [102.5, 125.5, 7.0, 41.5]),
]
FACTORS = [
    ("wind_pp",    "Wind-speed endowment\n(vs fleet-average wind)"),
    ("orient_pp",  "Orientation penalty\n(vs own optimal heading)"),
    ("lattice_pp", "Lattice / wake penalty\n(vs no-wake ideal)"),
]

nt = df["n_turb"].astype(float)
sizes_all = 7.0 + (nt - nt.min()) / (nt.max() - nt.min()) * 60.0

fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[0.36, 1.52])

sc = None
for i, (rkey, rlab, ext) in enumerate(REGIONS):
    sub = df[df.region == rkey]
    ssz = sizes_all.loc[sub.index]
    for j, (col, ctitle) in enumerate(FACTORS):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F1F3F5",
                       edgecolor="none")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#FFFFFF",
                       edgecolor="none")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3,
                       edgecolor=PAL["neutral"])
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.2,
                       edgecolor=PAL["light"], alpha=0.6)
        ax.spines["geo"].set_linewidth(0.5)

        sc = ax.scatter(sub["centroid_lon"], sub["centroid_lat"], s=ssz,
                        c=np.clip(sub[col], -VLIM, VLIM), cmap=CMAP, norm=norm,
                        transform=ccrs.PlateCarree(), edgecolor="white",
                        linewidth=0.35, alpha=0.92, zorder=5)

        if i == 0:
            ax.set_title(ctitle, fontsize=7.3, linespacing=1.15, pad=4)
            panel_label(ax, chr(ord("a") + j), dx=-0.06, dy=1.34)
        if j == 0:
            ax.text(-0.045, 0.5, rlab, transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=8, fontweight="bold",
                    color=PAL["ink"])

        # per-panel signed range annotation
        lo, hi = sub[col].min(), sub[col].max()
        ax.text(0.985, 0.03, "%+.1f … %+.1f pp" % (lo, hi),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
                color=PAL["ink"],
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                          alpha=0.8))

        # targeted callouts (East Asia row)
        if i == 1 and col == "orient_pp":
            ax.annotate("Vietnam: only cluster with a\nreal orientation penalty\n(down to −18 pp)",
                        xy=(106.3, 9.6), xytext=(104.5, 20.0),
                        transform=ccrs.PlateCarree(), fontsize=5.8,
                        color=PAL["ink"], linespacing=1.15,
                        arrowprops=dict(arrowstyle="-", color=PAL["neutral"],
                                        lw=0.6),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=PAL["light"], lw=0.5, alpha=0.9))
        if i == 1 and col == "lattice_pp":
            worst = sub.loc[sub[col].idxmin()]
            ax.annotate("densest layouts:\nup to %.0f pp" % worst[col],
                        xy=(worst.centroid_lon + 0.4, worst.centroid_lat + 0.3),
                        xytext=(111.5, 14.5), transform=ccrs.PlateCarree(),
                        fontsize=5.8, color=PAL["ink"], linespacing=1.15,
                        arrowprops=dict(arrowstyle="-", color=PAL["neutral"],
                                        lw=0.6),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=PAL["light"], lw=0.5, alpha=0.9))
        if i == 1 and col == "wind_pp":
            ax.text(0.03, 0.5,
                    "monsoon SE-Asia:\nwind-poor (red)\nvs windy N-China\n(blue)",
                    transform=ax.transAxes, fontsize=5.8, color=PAL["ink"],
                    va="center", linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=PAL["light"], lw=0.5, alpha=0.9))

# shared ruler
cb = fig.colorbar(sc, ax=fig.axes, orientation="horizontal", shrink=0.55,
                  pad=0.015, aspect=38, extend="both")
cb.set_label("Signed effect on generation (percentage points)   —   loss ← 0 → gain",
             fontsize=7.5)
cb.ax.tick_params(labelsize=6.8, width=0.6)
cb.outline.set_linewidth(0.5)
cb.ax.text(0.5, -3.6,
           "One shared ruler (±%.0f pp, clipped; true ranges in panel corners). Marker size ∝ turbines. US East (n=%d) omitted for space.\n"
           "Baselines: wind = fleet-average wind; orientation = own optimum; lattice = no-wake ideal."
           % (VLIM, len(us)), transform=cb.ax.transAxes, ha="center", va="top",
           fontsize=5.6, color=PAL["neutral"], linespacing=1.3)

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
