# -*- coding: utf-8 -*-
"""
图 6｜新增稳健性计算与设计层级
复跑：python FigS1_robustness_design_limits.py
数据：补算/jensen_rotation_56farms.csv（75 场 × 18 角度，均匀风向）
      补算/ow_sensitivity_3farms.csv（3 场 × 3 机型 × 18 角度，均匀风向）
      补算/joint_optimization_F57.csv（F57 无面积约束联合试验）
      分析材料_朝向长尾走廊/../任务三朝向回测（Gauss+ERA5 长尾分组）
注意：(a)(b) 两面板均为均匀风向口径，是几何角度离散诊断，
      不是对 Gauss + ERA5 朝向收益的同口径复现。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..",
                                     "wind-direction-to-electricity-transition"))
BU = os.path.join(REPO, "补算")
FOUR = os.path.join(REPO, "四场景风速风向分解贡献")

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
HI, MID, LO, LINE = "#b91c1c", "#ea580c", "#9ca3af", "#1f2937"
TCOL = {"iea_10MW": "#1d4ed8", "ow_6MW": "#0891b2", "ow_8MW": "#7c3aed"}
TLAB = {"iea_10MW": "IEA 10 MW", "ow_6MW": "OW 6 MW", "ow_8MW": "OW 8 MW"}


def ang_range(df, by):
    """每组：角度间 AEP 极差占均值的百分比 + 最优角。"""
    out = []
    for k, g in df.groupby(by):
        a = g.set_index("angle_deg").AEP_kWh
        out.append({by if isinstance(by, str) else "key": k,
                    "rng_pct": (a.max() - a.min()) / a.mean() * 100,
                    "best": int(a.idxmax())})
    return pd.DataFrame(out)


fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0),
                         gridspec_kw={"width_ratios": [1, 1.05, 1.05]})

# ---- (a) 75 场 Jensen 均匀风向重跑 --------------------------------------
ax = axes[0]
jr = pd.read_csv(os.path.join(BU, "jensen_rotation_56farms.csv"))
r = ang_range(jr, "farm_id")
# 长尾分组：用四情景 farm_summary 的 G_mean（= Gauss+ERA5 朝向收益）
gm = pd.read_csv(os.path.join(FOUR, "four_scenario_farm_summary.csv"))
r = r.merge(gm[["farm_id", "G_mean"]], on="farm_id", how="left")
r["grp"] = np.where(r.G_mean > 2, "原 Gauss 回测 >2%", "对照组 ≤2%")

groups = ["原 Gauss 回测 >2%", "对照组 ≤2%"]
data = [r.loc[r.grp == g, "rng_pct"].dropna().values for g in groups]
bp = ax.boxplot(data, positions=[0, 1], widths=.5, showfliers=False,
                patch_artist=True, medianprops=dict(color=LINE, lw=1.6))
for patch, c in zip(bp["boxes"], [HI, LO]):
    patch.set(facecolor=c, alpha=.22, edgecolor=c, lw=1.2)
ticklabs = []
for xi, (g, c) in enumerate(zip(groups, [HI, LO])):
    v = r.loc[r.grp == g, "rng_pct"].dropna().values
    ax.scatter(np.random.RandomState(7 + xi).normal(xi, .075, len(v)), v,
               s=15, c=c, alpha=.6, lw=0, zorder=3)
    ticklabs.append(f"{g}\nn={len(v)}，中位 {np.median(v):.2f}%")
ax.set_xticks([0, 1])
ax.set_xticklabels(ticklabs, fontsize=8.2, linespacing=1.6)
ax.set_ylabel("角度间 AEP 极差 / 均值 (%)")
ax.set_ylim(-0.02, max(r.rng_pct.max() * 1.12, 1.5))
ax.set_title("(a) 均匀风向下两组无差别\n——该面板是几何诊断，非 ERA5 复现",
             fontsize=9.4, loc="left", linespacing=1.5)

# ---- (b) 三机型 × 三场 --------------------------------------------------
ax = axes[1]
ow = pd.read_csv(os.path.join(BU, "ow_sensitivity_3farms.csv"))
farms = [57, 66, 91]
types = ["iea_10MW", "ow_6MW", "ow_8MW"]
w = .24
for ti, t in enumerate(types):
    xs, ys, bs = [], [], []
    for fi, fid in enumerate(farms):
        g = ow[(ow.farm_id == fid) & (ow.turbine_type == t)]
        a = g.set_index("angle_deg").AEP_kWh
        xs.append(fi + (ti - 1) * w)
        ys.append((a.max() - a.min()) / a.mean() * 100)
        bs.append(int(a.idxmax()))
    ax.bar(xs, ys, width=w * .9, color=TCOL[t], alpha=.88, label=TLAB[t])
    for x, y, b in zip(xs, ys, bs):
        ax.text(x, y + .035, f"{b}°", ha="center", fontsize=7.2, color=TCOL[t])
ax.set_xticks(range(len(farms)))
ax.set_xticklabels([f"F{f}\n{'越南' if f==57 else '珠江口'}" for f in farms],
                   fontsize=8.4, linespacing=1.4)
ax.set_ylabel("角度间 AEP 极差 / 均值 (%)")
ax.set_ylim(0, 2.15)
ax.legend(fontsize=7.8, frameon=False, loc="upper right")
ax.set_title("(b) 最优角随机型改变（柱顶为最优角）\n——仍为均匀风向，不可与 +18.2% 比较",
             fontsize=9.4, loc="left", linespacing=1.5)

# ---- (c) F57 联合优化瀑布 -----------------------------------------------
ax = axes[2]
jo = pd.read_csv(os.path.join(BU, "joint_optimization_F57.csv"))
base = jo.loc[jo.scenario == "real", "AEP_kWh"].iat[0] / 1e9
geom = jo.loc[jo.scenario == "built_opt_geom", "AEP_kWh"].iat[0] / 1e9
joint = jo.loc[jo.scenario == "rotated_opt_joint", "AEP_kWh"].iat[0] / 1e9
steps = [("建成基线\n间距 ~1.9D", 0, base, LO),
         ("+ 间距优化\n8D × 12D", base, geom - base, HI),
         ("+ 朝向旋转\nθ_opt = 40°", geom, joint - geom, MID)]
for i, (lab, bot, h, c) in enumerate(steps):
    ax.bar(i, h, bottom=bot, width=.56, color=c, alpha=.88)
    if i == 0:
        ax.text(i, h / 2, f"{base:.3f}", ha="center", va="center",
                fontsize=8.6, color="white", weight="bold")
    elif h > joint * .08:                      # 柱体够高，标在柱内
        ax.text(i, bot + h / 2, f"+{h:.3f}", ha="center", va="center",
                fontsize=8.4, color="white", weight="bold")
    else:                                      # 柱体太薄，引到柱下方空白处
        ax.annotate(f"+{h:.3f}\n（朝向边际 +3.7 pp）",
                    (i, bot + h / 2), (i, bot * .62), ha="center", va="center",
                    fontsize=8.2, color=MID, weight="bold", linespacing=1.5,
                    arrowprops=dict(arrowstyle="->", color=MID, lw=1.0))
    ax.text(i, bot + h + .045, f"{bot+h:.3f}", ha="center", fontsize=8.2,
            color=LINE)
    if i < 2:
        ax.plot([i + .28, i + 1 - .28], [bot + h] * 2, color=LO, lw=.9, ls="--")
ax.set_xticks(range(3))
ax.set_xticklabels([s[0] for s in steps], fontsize=8.2, linespacing=1.4)
ax.set_ylabel("F57 年发电量 (TWh yr⁻¹)")
ax.set_ylim(0, joint * 1.20)
ax.text(.5, joint * 1.10, "几何 +63.0 pp", ha="center", fontsize=8.8, color=HI)
ax.set_title("(c) 无面积约束上限试验：几何 ≫ 朝向边际\n——未固定租区，非可实施净收益",
             fontsize=9.4, loc="left", linespacing=1.5)

fig.suptitle("图 6｜新增稳健性计算与设计层级", fontsize=11.5, x=.02, ha="left", y=1.03)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"FigS1_robustness_design_limits.{ext}"))
print("saved -> FigS1_robustness_design_limits.png/pdf")
print(f"(a) 长尾组中位 {np.median(data[0]):.3f}%  对照组中位 {np.median(data[1]):.3f}%  n={len(r)}")
print(f"(c) {base:.4f} -> {geom:.4f} (+{(geom/base-1)*100:.1f} pp) -> {joint:.4f} (+{(joint/base-1)*100:.1f} pp)")
