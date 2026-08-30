# -*- coding: utf-8 -*-
"""
图 7｜朝向收益 vs 风年运气：逐场信噪平面
复跑：python _build_v6_derived.py && python Fig2_gain_vs_noise.py
数据：四情景 v6（廷显 2026-08-10 交付的 four_scenario_floris_aep_v6.csv），
      派生层由本目录 _build_v6_derived.py 重算——不要用 output/ 下那几个
      four_scenario_{effects,farm_summary}*.csv，它们仍是 v5。
口径：分母用 S_shapley 的逐年标准差（真实年际波动），不用 RMS——
      RMS 混入了相对 1981-2010 基准的持续偏置，且该偏置在季风场最大。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
HI, MID, LO, LINE = "#b91c1c", "#ea580c", "#9ca3af", "#1f2937"

summ = pd.read_csv(os.path.join(HERE, "v6_farm_summary.csv"), encoding="utf-8-sig")
f = summ.set_index("farm_id").rename(
    columns={"n_years": "n", "M_S_std": "S_std", "M_S_mean": "S_mean",
             "M_S_rms": "S_rms", "R_std": "R"})
f = f[f.n >= 3].copy()

NAMES = {57: "F57 越南\n南海季风", 155: "F155 意大利\n塔兰托湾",
         66: "F66 杭州湾口", 157: "F157 丹麦\n海峡风道",
         91: "F91 珠江口", 126: "F126 越南", 159: "F159 越南"}
TOP = list(NAMES)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3),
                         gridspec_kw={"width_ratios": [1.15, 1]})

# ---- (a) 信噪平面 -------------------------------------------------------
ax = axes[0]
lim = 22
ax.fill_between([0, lim], [0, lim], [lim, lim], color=HI, alpha=.055, lw=0)
ax.plot([0, lim], [0, lim], color=LINE, ls="--", lw=1.1, zorder=2)
ax.text(10.4, 11.35, "$G = M_S^{std}$", fontsize=8.2, color=LINE,
        rotation=32, rotation_mode="anchor", va="bottom")
ax.text(1.1, 19.6, "收益 > 风年运气\n（朝向可辨识）", fontsize=8.8,
        color=HI, va="top", linespacing=1.5)

oth = f[~f.index.isin(TOP)]
ax.scatter(oth.S_std, oth.G_mean, s=17, c=LO, alpha=.55, lw=0, zorder=3,
           label=f"其余场（n={len(oth)}）")
hl = f.loc[[i for i in TOP if i in f.index]]
ax.scatter(hl.S_std, hl.G_mean, s=62, c=[HI if r > 1 else MID for r in hl.R],
           edgecolor="white", lw=1.1, zorder=5, label="七个走廊场")

# 逐场手工排版，避开互相重叠
OFF = {57: (.45, .1, "left", "center"),
       155: (-.42, 1.35, "right", "bottom"),
       66: (-.42, .1, "right", "center"),
       157: (-.42, -1.15, "right", "top"),
       91: (.42, .55, "left", "bottom"),
       126: (.45, -.25, "left", "top"),
       159: (-.40, -1.05, "right", "top")}
for fid, row in hl.iterrows():
    dx, dy, ha, va = OFF[fid]
    ax.annotate(f"{NAMES[fid]}  R={row.R:.2f}".replace("\n", " "),
                (row.S_std + dx, row.G_mean + dy), ha=ha, va=va, fontsize=7.5,
                color=HI if row.R > 1 else MID, linespacing=1.3, zorder=6)
ax.set_xlim(0, 12.5); ax.set_ylim(-0.6, lim)
ax.set_xlabel("自身风年波动 $M_S^{std}$：逐年风速效应的标准差 (%)")
ax.set_ylabel("朝向收益 $G$ (%)")
n5 = f[f.n >= 5]
ax.set_title(f"(a) 逐场信噪平面：只有 {(n5.R>1).sum()}/{len(n5)} 场越过自己的风年运气",
             fontsize=9.8, loc="left")
ax.legend(fontsize=8, loc="lower right", frameon=False)

# ---- (b) 两种分母之别 ---------------------------------------------------
ax = axes[1]
b = f.loc[[i for i in TOP if i in f.index]].copy()
b["R_rms_"] = b.G_mean / b.S_rms
b = b.sort_values("R")
y = np.arange(len(b))
ax.barh(y + .19, b.R, height=.36, color=HI, alpha=.88, label="$G/M_S^{std}$（年际波动，本文口径）")
ax.barh(y - .19, b.R_rms_, height=.36, color="#94a3b8", label="$G/M_S^{rms}$（含基准偏置）")
ax.axvline(1, color=LINE, ls="--", lw=1.1)
ax.set_yticks(y)
ax.set_yticklabels([NAMES[i].replace("\n", " ") for i in b.index], fontsize=8.2)
ax.set_xlabel("朝向收益 / 风速效应")
ax.set_title("(b) 分母选错会把中国湾口两个走廊场判成「不越线」", fontsize=9.8, loc="left")
ax.legend(fontsize=7.8, loc="lower right", frameon=False)
i66 = list(b.index).index(66)
ax.annotate("F66 的 $M_S^{rms}$ 里\n9.3 pp 是基准偏置", (b.loc[66, "R_rms_"], i66 - .19),
            (1.30, i66 - 1.55), fontsize=7.6, color="#475569",
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=.9), linespacing=1.4)

fig.suptitle("图 7｜朝向收益与风年运气的逐场对照", fontsize=11.5, x=.02, ha="left", y=1.02)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"Fig2_gain_vs_noise.{ext}"))
print("saved -> Fig2_gain_vs_noise.png/pdf")

print("\n可复现要点（四情景 v6）：")
print(f"  M_S_std 中位 {n5.S_std.median():.2f}%  (论文 CV 地板 5.2%)")
print(f"  R>1 场数 (n>=5): {(n5.R>1).sum()}/{len(n5)}  -> {sorted(n5[n5.R>1].index)}")
print(f"  R_rms>1 (n>=5): {(n5.G_mean/n5.S_rms>1).sum()}  -> {sorted(n5[n5.G_mean/n5.S_rms>1].index)}")
print(b[["country", "n", "G_mean", "S_std", "S_rms", "S_mean", "R", "R_rms_"]].round(2).to_string())
