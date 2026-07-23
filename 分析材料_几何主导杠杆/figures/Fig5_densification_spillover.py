import sys, os
sys.path.insert(0, r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures")
from nc_style import PAL, REGION_COL, PARADIGM_COL, apply_style, panel_label, savefig
apply_style()

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived/farm_pairs_all.csv"
T4 = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived/task4_cross_farm_wake.csv"
T5 = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived/task5_expansion_results.csv"
OUT = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/figures/Fig5_densification_spillover.png"

# --- Load pairs and classify region (real data only) -----------------------
with open(DATA, newline="", encoding="utf-8-sig") as _f:
    rows = list(csv.DictReader(_f))
NORTH_SEA = {"United Kingdom", "Netherlands", "Germany", "Denmark", "Belgium"}


def region(r):
    ci, cj = r["country_i"], r["country_j"]
    if "China" in (ci, cj):
        return "China"
    if ci in NORTH_SEA or cj in NORTH_SEA:
        return "NorthSea"
    return "Other"


def counts(pred):
    sel = [r for r in rows if pred(r)]
    out = {"China": 0, "NorthSea": 0, "Other": 0}
    for r in sel:
        out[region(r)] += 1
    return out, len(sel)


c_edge, n_edge = counts(lambda r: float(r["edge_gap_km"]) < 10)
c_c20, n_c20 = counts(lambda r: float(r["center_km"]) < 20)
c_c30, n_c30 = counts(lambda r: float(r["center_km"]) < 30)

thresholds = ["edge gap\n< 10 km", "centre dist.\n< 20 km", "centre dist.\n< 30 km"]
totals = [n_edge, n_c20, n_c30]
china = [c_edge["China"], c_c20["China"], c_c30["China"]]
nsea = [c_edge["NorthSea"], c_c20["NorthSea"], c_c30["NorthSea"]]
other = [c_edge["Other"], c_c20["Other"], c_c30["Other"]]

# --- Figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
axa, axb, axc = axes

# ===== Panel (a): proximity-pair counts, stacked by region =================
x = np.arange(3)
w = 0.62
reg_col = {"China": PAL["highlight"], "NorthSea": PAL["baseline"], "Other": PAL["neutral"]}
b1 = axa.bar(x, china, w, color=reg_col["China"], label="China coast", zorder=3)
b2 = axa.bar(x, nsea, w, bottom=china, color=reg_col["NorthSea"],
             label="North Sea (UK)", zorder=3)
b3 = axa.bar(x, other, w, bottom=np.array(china) + np.array(nsea),
             color=reg_col["Other"], label="Other", zorder=3)

for xi, t in zip(x, totals):
    axa.text(xi, t + 2.2, str(t), ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=PAL["ink"])

axa.set_ylim(0, 88)
axa.set_xticks(x)
axa.set_xticklabels(thresholds)
axa.set_ylabel("Number of farm pairs")
axa.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.1,
           labelspacing=0.3, borderaxespad=0.2)
axa.text(0.02, 0.70,
         "52 / 171 farms (30%)\nhave a neighbour\nwithin 10 km edge gap",
         transform=axa.transAxes, ha="left", va="top", fontsize=7.2,
         color=PAL["ink"],
         bbox=dict(boxstyle="round,pad=0.35", fc="white",
                   ec=PAL["light"], lw=0.8))
panel_label(axa, "a")

# ===== Panel (b): cross-farm wake -- merged-domain FLORIS (task 4) ==========
t4 = list(csv.DictReader(open(T4, encoding="utf-8-sig")))
by = {}
for r in t4:
    by.setdefault(r["pair"], {})[r["scenario"]] = r

pair_info = []  # (label, cross_loss_pct, trustworthy)
for pair, sc in by.items():
    alone = [k for k in sc if "alone" in k]
    merged = [k for k in sc if "merged" in k][0]
    aA = float(sc[alone[0]]["AEP_kWh"]); aB = float(sc[alone[1]]["AEP_kWh"])
    aM = float(sc[merged]["AEP_kWh"])
    cwl = (aA + aB - aM) / (aA + aB) * 100.0
    pair_info.append((pair, cwl))

# order: East-Asia (35_72, trustworthy) first, Europe (31_113, confounded) second
order = {"35_72": 0, "31_113": 1}
pair_info.sort(key=lambda p: order.get(p[0], 9))
lab_map = {"35_72": "East Asia\nF35+F72\n(12.9 km)",
           "31_113": "Europe\nF31+F113\n(13.0 km)"}
xb = np.arange(len(pair_info))
vals = [p[1] for p in pair_info]
cols = [PAL["highlight"], PAL["light"]]
hatch = [None, "////"]
bars = axb.bar(xb, vals, 0.55, color=cols, edgecolor=PAL["neutral"],
               linewidth=0.7, zorder=3)
for bar, h in zip(bars, hatch):
    if h:
        bar.set_hatch(h)

axb.axhline(0, color=PAL["ink"], lw=0.8, zorder=2)
axb.text(0, vals[0] + 0.18, "+0.47%", ha="center", va="bottom",
         fontsize=8, fontweight="bold", color=PAL["highlight"], zorder=4)
axb.text(1, vals[1] - 0.25, "confounded\n(single-point\nERA5, excluded)",
         ha="center", va="top", fontsize=6.0, color=PAL["neutral"],
         linespacing=1.05, zorder=4)

axb.set_xticks(xb)
axb.set_xticklabels([lab_map.get(p[0], p[0]) for p in pair_info], fontsize=6.6)
axb.set_ylabel("Cross-farm wake loss\n(% of combined AEP)")
axb.set_ylim(-3.2, 1.5)
axb.text(0.02, 0.05,
         "conservative lower bound:\nFLORIS underestimates\nwake beyond 10 km",
         transform=axb.transAxes, ha="left", va="bottom", fontsize=5.9,
         color=PAL["neutral"], linespacing=1.15)
panel_label(axb, "b")

# ===== Panel (c): technology axis -- turbine upgrade raises wake (task 5) ===
t5 = pd.read_csv(T5)
tt_order = ["iea_10MW", "iea_15MW", "iea_22MW"]
tt_lab = ["IEA\n10 MW", "IEA\n15 MW", "IEA\n22 MW"]
wl = [t5.loc[t5.turbine_type == tt, "WakeLoss"].mean() * 100 for tt in tt_order]
cf = [t5.loc[t5.turbine_type == tt, "CF"].mean() * 100 for tt in tt_order]

xc = np.arange(3)
bwl = axc.bar(xc, wl, 0.58, color=PAL["baseline"], edgecolor="white",
              linewidth=0.5, zorder=3, label="Mean wake loss")
for xi, v in zip(xc, wl):
    axc.text(xi, v + 0.25, "%.1f%%" % v, ha="center", va="bottom",
             fontsize=7.2, color=PAL["ink"], zorder=4)
axc.annotate("", xy=(2, wl[2] + 0.9), xytext=(0, wl[0] + 0.9),
             arrowprops=dict(arrowstyle="->", color=PAL["highlight"], lw=1.1))
axc.text(1, max(wl) + 1.9, "+4.3 pp wake / -1.8 pp CF", ha="center",
         va="bottom", fontsize=6.6, color=PAL["highlight"], fontweight="bold")

axc.set_xticks(xc)
axc.set_xticklabels(tt_lab)
axc.set_ylabel("Mean wake loss (%)")
axc.set_ylim(0, max(wl) + 4)

# CF on a twin axis
axc2 = axc.twinx()
axc2.plot(xc, cf, "o-", color=PAL["highlight"], lw=1.4, ms=4.5, zorder=5,
          label="Capacity factor")
axc2.set_ylabel("Capacity factor (%)", color=PAL["highlight"])
axc2.tick_params(axis="y", labelcolor=PAL["highlight"])
axc2.set_ylim(min(cf) - 1.2, max(cf) + 1.2)
axc2.grid(False)
axc2.spines["top"].set_visible(False)
panel_label(axc, "c")

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
