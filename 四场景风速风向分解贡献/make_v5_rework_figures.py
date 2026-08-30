# -*- coding: utf-8 -*-
"""
四情景 v5 返工意见配图（FigB1-B3）
复跑：python make_v5_rework_figures.py
输入：本目录 four_scenario_{aep,effects}_farmyear.csv
      ../offshore-task2/output/task2_counterfactual.csv
输出：figures_v5/FigB*.png|pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures_v5")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BAD, OK, WARN, GREY = "#dc2626", "#059669", "#d97706", "#6b7280"
SPLIT = 2017.5


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("  ->", name)


aep = pd.read_csv(os.path.join(BASE, "four_scenario_aep_farmyear.csv"))
eff = pd.read_csv(os.path.join(BASE, "four_scenario_effects_farmyear.csv"))
aep["era"] = np.where(aep.year <= 2017, "new", "copy")
eff["era"] = np.where(eff.year <= 2017, "new", "copy")


# ---------------------------------------------------------------- FigB1
def fig_b1():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    d = aep.assign(r10=aep.P10_kWh / aep.P11_kWh,
                   r01=aep.P01_kWh / aep.P00_kWh)

    for ax, col, title, sub in [
        (axes[0], "r10", "(a) $P_{10}/P_{11}$ —— 该量的算法在 2018 年换过",
         "2014–17 查表新算 → 2018–24 逐时复制"),
        (axes[1], "r01", "(b) $P_{01}/P_{00}$ —— 对照量，两段同为查表",
         "算法全程未变"),
    ]:
        by = d.groupby("year")[col]
        yrs = np.array(sorted(d.year.unique()))
        med = by.median().values
        q1, q3 = by.quantile(.25).values, by.quantile(.75).values
        cols = [BAD if y <= 2017 else OK for y in yrs]
        ax.vlines(yrs, q1, q3, color=cols, lw=5, alpha=.30)
        ax.plot(yrs, med, "-", color=GREY, lw=1, zorder=2)
        ax.scatter(yrs, med, c=cols, s=34, zorder=3)
        ax.axvline(SPLIT, color=GREY, ls="--", lw=1)
        ax.axhline(1.0, color="#9ca3af", lw=.7, zorder=0)
        ax.set_title(title + "\n" + sub, fontsize=9.5, loc="left")
        ax.set_xlabel("年份")
        ax.set_xticks(yrs[::2])

    # paired stats annotation
    piv = d.groupby(["farm_id", "era"]).r10.mean().unstack().dropna()
    dif = (piv["new"] - piv["copy"]).mean()
    pv = stats.ttest_rel(piv["new"], piv["copy"])[1]
    axes[0].text(.03, .06, f"同场配对(n={len(piv)})：断点 {dif*100:+.1f} pp,  p={pv:.0e}",
                 transform=axes[0].transAxes, fontsize=8.4, color=BAD)
    p2 = d.groupby(["farm_id", "era"]).r01.mean().unstack().dropna()
    d2 = (p2["new"] - p2["copy"]).mean()
    axes[1].text(.03, .06, f"同场配对(n={len(p2)})：断点 {d2*100:+.1f} pp（小一个量级）",
                 transform=axes[1].transAxes, fontsize=8.4, color=OK)
    axes[0].set_ylabel("比值（点为逐年中位数，竖条为 IQR）")

    fig.suptitle("图 B1　$P_{10}$ 双口径在 2017/2018 之间制造了一个人为断点",
                 fontsize=11, y=1.06, x=.02, ha="left")
    save(fig, "FigB1_p10_break")


# ---------------------------------------------------------------- FigB2
def fig_b2():
    terms = [("D_pct", "$D$ = $(P_{01}-P_{00})/P_{11}$", "查表 − 查表", OK),
             ("total_pct", "总效应 = $(P_{11}-P_{00})/P_{11}$", "逐时 − 查表（两段同）", OK),
             ("S_pct", "$S$ = $(P_{10}-P_{00})/P_{11}$", "口径随年份改变", BAD),
             ("I_pct", "$I$ = 总 − $S$ − $D$", "承接 $S$ 的断点（反号）", BAD)]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ys, labs, vals, cs, ps = [], [], [], [], []
    for i, (c, lab, note, col) in enumerate(terms):
        p = eff.groupby(["farm_id", "era"])[c].mean().unstack().dropna()
        v = (p["new"] - p["copy"]).mean()
        pv = stats.ttest_rel(p["new"], p["copy"])[1]
        ys.append(i); labs.append(lab + "\n" + note); vals.append(v); cs.append(col); ps.append(pv)
    ax.barh(ys, vals, color=cs, alpha=.85, height=.6)
    for y, v, pv in zip(ys, vals, ps):
        ax.text(v + (.15 if v >= 0 else -.15), y, f"{v:+.2f} pp  (p={pv:.0e})",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.3)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=8.6)
    ax.invert_yaxis()
    ax.axvline(0, color="#374151", lw=.8)
    ax.set_xlim(-6.2, 6.8)
    ax.set_xlabel("2014–17 段 减 2018–24 段（同场配对均值，百分点）")
    ax.set_title("图 B2　断点只出现在含 $P_{10}$ 的项上——方法性，不是气候性",
                 fontsize=11, loc="left", pad=10)
    save(fig, "FigB2_term_breaks")


# ---------------------------------------------------------------- FigB3
def fig_b3():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    e18 = eff[eff.year >= 2018]

    by = e18.groupby("year").S_shapley
    yrs = np.array(sorted(e18.year.unique()))
    axes[0].vlines(yrs, by.quantile(.25).values, by.quantile(.75).values,
                   color=WARN, lw=6, alpha=.32)
    axes[0].plot(yrs, by.median().values, "o-", color=WARN, lw=1.2, ms=6)
    axes[0].axhline(0, color=BAD, ls="--", lw=1.1)
    axes[0].text(yrs[0], .4, "真实年际波动应在 0 附近摆动", color=BAD, fontsize=8.3, va="bottom")
    axes[0].set_xlabel("年份"); axes[0].set_ylabel("$S_{shapley}$ (%)")
    axes[0].set_title("(a) 逐年场均：7 年全部为正", fontsize=9.5, loc="left")
    axes[0].set_xticks(yrs)

    s = e18.groupby(["farm_id", "country"]).S_shapley.mean().reset_index()
    r = s.groupby("country").S_shapley.agg(n="size", med="median")
    r = r[r.n >= 3].sort_values("med")
    axes[1].barh(range(len(r)), r.med.values, color=WARN, alpha=.85, height=.62)
    axes[1].set_yticks(range(len(r)))
    axes[1].set_yticklabels([f"{i}  (n={n})" for i, n in zip(r.index, r.n)], fontsize=8.2)
    axes[1].axvline(0, color="#374151", lw=.8)
    axes[1].set_xlabel("$S_{shapley}$ 场均值中位数 (%)")
    axes[1].set_title("(b) 国家间跨度 0.6→12.5 pp，气候信号不应如此", fontsize=9.5, loc="left")

    fig.suptitle("图 B3　$M_S$ 里有一个持续为正的偏置——它不是「年际波动」",
                 fontsize=11, y=1.05, x=.02, ha="left")
    save(fig, "FigB3_baseline_offset")


if __name__ == "__main__":
    print("生成 v5 返工配图 ->", FIG)
    fig_b1(); fig_b2(); fig_b3()
    print("完成。")
