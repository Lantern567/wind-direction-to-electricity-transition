import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_style import PAL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DD = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig2_corridors.png")

og = pd.read_csv(os.path.join(DD, "orientation_gain.csv"))
mp = pd.read_csv(os.path.join(DD, "farm_wakeloss_map.csv")).set_index("farm_id")
fg = og.groupby("farm_id")["gain_pct"].mean()
df = pd.DataFrame({"gain": fg}).join(mp[["centroid_lon", "centroid_lat", "country"]]).dropna()
NOISE = 5.17

# farm-year pooled country means (consistent with the text numbers)
oc = og.merge(mp[["country"]], left_on="farm_id", right_index=True)
cm = oc.groupby("country")["gain_pct"].agg(["mean", "count"])
show = ["Italy", "Vietnam", "Ireland", "Taiwan", "China", "United Kingdom",
        "Denmark", "Netherlands", "Germany", "France", "Japan"]
cm = cm.reindex([c for c in show if c in cm.index])
print(cm.round(2))

# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.35, 1.0])
axA = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
axB = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])


def basemap(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F1F3F5", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#FFFFFF", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3,
                   edgecolor=PAL["neutral"])
    ax.spines["geo"].set_linewidth(0.5)


def plot_farms(ax, sub):
    for _, r in sub.iterrows():
        if r.gain > NOISE:
            c, s, z = PAL["highlight"], 46, 6
        elif r.gain > 2:
            c, s, z = PAL["accent"], 24, 5
        else:
            c, s, z = PAL["light"], 8, 4
        ax.scatter(r.centroid_lon, r.centroid_lat, s=s, color=c, edgecolor="white",
                   linewidth=0.3, transform=ccrs.PlateCarree(), zorder=z)

# ======================================================================
# (a) East Asia monsoon coast
# ======================================================================
basemap(axA, [102.5, 124.5, 7.0, 34.5])
plot_farms(axA, df[df.centroid_lon > 45])
axA.annotate("Mekong Delta corridor\nF57 +18.2 / F126 +5.2 / F159 +5.2",
             xy=(106.2, 9.4), xytext=(104.3, 15.6), transform=ccrs.PlateCarree(),
             fontsize=6.0, color=PAL["ink"], linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axA.annotate("Hangzhou Bay F66 +7.8",
             xy=(121.5, 30.5), xytext=(112.2, 31.4), transform=ccrs.PlateCarree(),
             fontsize=6.0, color=PAL["ink"],
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axA.annotate("Pearl River mouth F91 +6.6",
             xy=(113.4, 21.9), xytext=(112.8, 25.8), transform=ccrs.PlateCarree(),
             fontsize=6.0, color=PAL["ink"],
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
handles = [Line2D([], [], marker="o", ls="", color=PAL["highlight"], ms=6,
                  label="above noise floor (5.2%)"),
           Line2D([], [], marker="o", ls="", color=PAL["accent"], ms=5,
                  label="2% – 5.2%"),
           Line2D([], [], marker="o", ls="", color=PAL["light"], ms=3.5,
                  label="below 2%")]
axA.legend(handles=handles, loc="lower right", fontsize=5.8, handlelength=0.8,
           labelspacing=0.3, borderaxespad=0.3, framealpha=0.9)
axA.set_title("East Asia monsoon coast", fontsize=7.5, pad=3)
panel_label(axA, "a")

# ======================================================================
# (b) Europe strait channels
# ======================================================================
basemap(axB, [-11.5, 20.5, 35.5, 58.5])
plot_farms(axB, df[(df.centroid_lon > -30) & (df.centroid_lon < 45)])
axB.annotate("Gulf of Taranto F155 +9.6",
             xy=(17.1, 40.5), xytext=(2.5, 38.2), transform=ccrs.PlateCarree(),
             fontsize=6.0, color=PAL["ink"],
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axB.annotate("Danish strait F157 +6.9",
             xy=(10.8, 55.1), xytext=(-9.8, 47.0), transform=ccrs.PlateCarree(),
             fontsize=6.0, color=PAL["ink"],
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6),
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PAL["light"],
                       lw=0.5, alpha=0.92))
axB.set_title("Europe", fontsize=7.5, pad=3)
panel_label(axB, "b")

# ======================================================================
# (c) year-by-year stability of the top corridor farms
# ======================================================================
farm_style = [(57,  PAL["highlight"], "F57 Vietnam (48 to 80 turbines)"),
              (155, PAL["accent"],    "F155 Italy (2 to 9)"),
              (66,  PAL["baseline"],  "F66 Hangzhou Bay (3 to 74)"),
              (157, PAL["neutral"],   "F157 Denmark (7, unchanged)")]
for fid, c, lab in farm_style:
    s = og[og.farm_id == fid].sort_values("year")
    axC.plot(s.year, s.gain_pct, "o-", color=c, lw=1.3, ms=3.0, label=lab, zorder=4)
axC.axhline(0.92, color=PAL["ink"], ls=":", lw=0.8)
axC.text(2014.1, 1.3, "global mean +0.9%", fontsize=6.0, color=PAL["ink"])
axC.annotate("configuration jump,\n2 to 9 turbines", xy=(2021.15, 1.9),
             xytext=(2015.6, 4.6), fontsize=5.9, color=PAL["accent"],
             linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=PAL["accent"], lw=0.6))
axC.set_xlabel("Year")
axC.set_ylabel("Orientation gain (%)")
axC.set_xlim(2013.5, 2024.5)
axC.set_ylim(-0.8, 23.5)
axC.legend(loc="upper right", fontsize=5.8, handlelength=1.3, labelspacing=0.3,
           borderaxespad=0.3)
panel_label(axC, "c")

# ======================================================================
# (d) farm-year pooled mean gain by country
# ======================================================================
y = np.arange(len(cm))[::-1]
cols = [PAL["highlight"] if v > 2 else PAL["baseline"] for v in cm["mean"]]
axD.barh(y, cm["mean"], height=0.62, color=cols, edgecolor="white", linewidth=0.4,
         zorder=3)
for yi, (v, nfy) in zip(y, zip(cm["mean"], cm["count"])):
    vv = 0.0 if abs(v) < 0.05 else v
    axD.text(max(v, 0) + 0.12, yi, ("%+.1f%%" % vv) if vv else "0.0%",
             va="center", fontsize=6.2, color=PAL["ink"])
axD.set_yticks(y)
axD.set_yticklabels(["%s (n=%d)" % (c, int(cm.loc[c, "count"])) for c in cm.index],
                    fontsize=6.4)
axD.axvline(0, color=PAL["ink"], lw=0.8)
axD.set_xlabel("Mean orientation gain, farm-years pooled (%)")
axD.set_xlim(-0.8, 11.4)
axD.grid(axis="y", visible=False)
panel_label(axD, "d")

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
