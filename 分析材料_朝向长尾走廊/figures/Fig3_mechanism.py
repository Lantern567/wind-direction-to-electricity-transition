import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_style import PAL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DD = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived"
REPO = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig3_mechanism.png")

og = pd.read_csv(os.path.join(DD, "orientation_gain.csv"))
mp = pd.read_csv(os.path.join(DD, "farm_wakeloss_map.csv")).set_index("farm_id")
wm = pd.read_csv(os.path.join(REPO, "task1_output", "task1_wind_metrics.csv"),
                 encoding="utf-8-sig")
t2 = pd.read_csv(os.path.join(REPO, "offshore-task2", "output", "task2_annual_floris.csv"),
                 encoding="utf-8-sig")
t2 = t2[t2.wake_model == "gauss"]

fg = og.groupby("farm_id")["gain_pct"].mean()
wci = wm.groupby("farm_id")["WCI_yearly"].mean()
n_final = t2.groupby("farm_id")["n_turb"].max()

df = pd.DataFrame({"gain": fg}).join(mp[["wl_mean_pct"]])
df["wci"] = wci
df["n_final"] = n_final
df = df.dropna(subset=["gain", "wl_mean_pct", "wci"])

r_pool = float(np.corrcoef(df.wl_mean_pct, df.gain)[0, 1])
r_wci = float(np.corrcoef(df.wci, df.gain)[0, 1])
hiW = df.wci > df.wci.quantile(0.75)
hiL = df.wl_mean_pct > df.wl_mean_pct.quantile(0.75)
g_all = df.gain.mean()
g_w = df.loc[hiW & ~hiL, "gain"].mean()
g_l = df.loc[hiL & ~hiW, "gain"].mean()
g_both = df.loc[hiW & hiL, "gain"].mean()
n_both = int((hiW & hiL).sum())
small = df.n_final <= 10
TAIL7 = [57, 155, 66, 157, 91, 126, 159]
corridor = df.index.isin(TAIL7)
m_all, r_all = df.loc[small, "gain"].mean(), df.loc[~small, "gain"].mean()
m_nc = df.loc[small & ~corridor, "gain"].mean()
r_nc = df.loc[~small & ~corridor, "gain"].mean()
print("r_pool=%.2f r_wci=%.2f | all=%.2f wciOnly=%.2f poolOnly=%.2f both=%.2f (n=%d)"
      % (r_pool, r_wci, g_all, g_w, g_l, g_both, n_both))
print("micro(final n<=10, n=%d): %.2f vs rest %.2f | excl corridor: micro %.2f (n=%d) vs rest %.2f"
      % (small.sum(), m_all, r_all, m_nc, (small & ~corridor).sum(), r_nc))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.9), layout="constrained",
                                    gridspec_kw={"width_ratios": [1.25, 0.95, 0.85]})

# ======================================================================
# (a) gain vs wake pool, coloured by wind-direction concentration
# ======================================================================
xq = df.wl_mean_pct.quantile(0.75)
yq = df.wci.quantile(0.75)
sc = axA.scatter(df.wl_mean_pct, df.gain, c=df.wci, cmap="Blues", vmin=0.08,
                 vmax=0.42, s=22, edgecolor=PAL["neutral"], linewidth=0.3,
                 alpha=0.9, zorder=3)
cb = fig.colorbar(sc, ax=axA, shrink=0.85, pad=0.02, aspect=22)
cb.set_label("Wind-direction concentration (WCI)", fontsize=6.4)
cb.ax.tick_params(labelsize=6, width=0.5)
cb.outline.set_linewidth(0.4)

axA.axvspan(xq, 50, color=PAL["highlight"], alpha=0.05, zorder=1)
axA.axvline(xq, color=PAL["neutral"], ls=":", lw=0.8)
axA.text(xq + 0.7, -1.7, "top-quartile wake pool", fontsize=5.8,
         color=PAL["neutral"])

for fid, dx, dy in [(57, -7.5, 0.5), (155, 1.0, 0.7), (66, 1.0, 0.4), (91, 1.0, 0.2)]:
    r = df.loc[fid]
    axA.annotate("F%d" % fid, xy=(r.wl_mean_pct, r.gain),
                 xytext=(r.wl_mean_pct + dx, r.gain + dy), fontsize=6.0,
                 color=PAL["ink"])

axA.text(0.03, 0.965, "r(gain, wake pool) = %+.2f\nr(gain, WCI) = %+.2f"
         % (r_pool, r_wci),
         transform=axA.transAxes, ha="left", va="top", fontsize=6.4,
         color=PAL["ink"], linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["light"], lw=0.6))
axA.set_xlim(0, 50)
axA.set_ylim(-2.2, 19.5)
axA.set_xlabel("Wake-loss pool (%)")
axA.set_ylabel("Mean orientation gain (%)")
panel_label(axA, "a")

# ======================================================================
# (b) the product condition: both factors together, or nothing
# ======================================================================
labels = ["All\nfarms", "Rose\nonly", "Pool\nonly", "Both"]
vals = [g_all, g_w, g_l, g_both]
cols = [PAL["light"], PAL["neutral"], PAL["neutral"], PAL["highlight"]]
x = np.arange(4)
axB.bar(x, vals, 0.62, color=cols, edgecolor="white", linewidth=0.5, zorder=3)
for xi, v in zip(x, vals):
    axB.text(xi, v + 0.06, "%.2f%%" % v, ha="center", va="bottom", fontsize=6.6,
             fontweight="bold", color=PAL["ink"])
axB.text(3, g_both / 2, "n = %d" % n_both, ha="center", va="center",
         fontsize=5.6, color="white")
axB.set_xticks(x)
axB.set_xticklabels(labels, fontsize=6.0)
axB.set_ylabel("Mean orientation gain (%)")
axB.set_ylim(0, 3.35)
axB.text(0.04, 0.965, "the two factors\nmultiply, not add",
         transform=axB.transAxes, ha="left", va="top", fontsize=6.4,
         color=PAL["ink"], linespacing=1.25)
axB.grid(axis="x", visible=False)
panel_label(axB, "b")

# ======================================================================
# (c) rejected alternative: apparent micro-farm effect is corridor membership
# ======================================================================
rng = np.random.RandomState(7)
for i, mask in enumerate([small, ~small]):
    v = df.loc[mask, "gain"]
    corr_mask = mask & corridor
    plain_mask = mask & ~corridor
    jitter = dict(zip(df.index[mask],
                      rng.uniform(-0.16, 0.16, int(mask.sum()))))
    axC.scatter([i + jitter[f] for f in df.index[plain_mask]],
                df.loc[plain_mask, "gain"], s=9, color=PAL["neutral"], alpha=0.5,
                edgecolor="none", zorder=3)
    axC.scatter([i + jitter[f] for f in df.index[corr_mask]],
                df.loc[corr_mask, "gain"], s=22, color=PAL["highlight"],
                edgecolor="white", linewidth=0.4, zorder=5)
    axC.hlines(v.mean(), i - 0.26, i + 0.26, color=PAL["ink"], lw=1.6, zorder=6)

# means excluding corridor farms, as short dashed ticks
for i, mask in enumerate([small & ~corridor, ~small & ~corridor]):
    axC.hlines(df.loc[mask, "gain"].mean(), i - 0.26, i + 0.26,
               color=PAL["baseline"], lw=1.2, ls=(0, (3, 2)), zorder=6)

axC.text(0, m_all + 0.6, "%.1f%%" % m_all, ha="center", fontsize=6.4,
         fontweight="bold", color=PAL["ink"])
axC.text(1, r_all + 0.6, "%.1f%%" % r_all, ha="center", fontsize=6.4,
         fontweight="bold", color=PAL["ink"])
axC.text(0.035, 0.975,
         "apparent micro-farm edge\nvanishes once corridor\nfarms (red) are removed;\ndashed means %.1f%% vs %.1f%%"
         % (m_nc, r_nc),
         transform=axC.transAxes, ha="left", va="top", fontsize=6.0,
         color=PAL["ink"], linespacing=1.3)

axC.set_xticks([0, 1])
axC.set_xticklabels(["10 turbines\nor fewer\n(n=%d)" % small.sum(),
                     "more than 10\nturbines\n(n=%d)" % (~small).sum()], fontsize=6.2)
axC.set_ylabel("Mean orientation gain (%)")
axC.set_ylim(-2.2, 19.5)
axC.grid(axis="x", visible=False)
panel_label(axC, "c")

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
