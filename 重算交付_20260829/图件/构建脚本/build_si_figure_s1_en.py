# -*- coding: utf-8 -*-
"""Figure S1（英文版重建）：六套标准化建设范式的 64 机位排布示意图
====================================================================
学长要求：图内文字全部英文；所有图件字体统一（与主稿图 1–4 一致）。
本版改用 nc_style_nat.apply_style()（Arial，字号 8，NC 体例），
删除原微软雅黑/中文标注版（figures-new/FigS1_paradigm_layouts.png）。

数据：补算/output/wp2c_paradigm_layouts.csv（WP2c 生成，6 范式 × 64 台，未改动）
输出：figures-nat/FigS1_paradigm_layouts.png / .pdf
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from nc_style_nat import apply_style, panel_label, save_fig, DEEP_BLUE, RED, INK

CSV = os.path.join(REPO, '补算', 'output', 'wp2c_paradigm_layouts.csv')
OUT = os.path.join(HERE, 'FigS1_paradigm_layouts.png')

# 英文面板标题：范式名 + 生成规则（与 SI 表 S6 一致）
TITLES = {
    'S_A': 'S_A  Crosswind-aligned\n(row axis ⊥ energy-weighted wind)',
    'S_B0': 'S_B0  Constraint-first, N axis\n(fixed geographic axis 0°)',
    'S_B45': 'S_B45  Constraint-first, 45° axis\n(fixed geographic axis 45°)',
    'S_C': 'S_C  Phased expansion\n(core zone + crosswind extension)',
    'S_D': 'S_D  Wind-resource gradient\n(WPD-gradient siting)',
    'S_E': 'S_E  Wide spacing\n(S_A spacing × 1.25)',
}

df = pd.read_csv(CSV, encoding='utf-8-sig')
scen = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
assert sorted(df.paradigm.unique()) == sorted(scen)

apply_style()
# 字号统一（2026-08-30 用户标准）：SI 图 15cm 页宽/5310px，页面刻度量级≈4.8pt
# → 基底 15.4、面板字母 22.2；标题受面板宽度限制取 12.5、'Wind' 11
plt.rcParams.update({'font.size': 15.4, 'axes.labelsize': 15.4,
                     'xtick.labelsize': 13.6, 'ytick.labelsize': 13.6,
                     'legend.fontsize': 13.6})
fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.6))
for ax, s, letter in zip(axes.ravel(), scen, 'abcdef'):
    d = df[df.paradigm == s]
    assert len(d) == 64, (s, len(d))
    x, y = d.x_m.values / 1000.0, d.y_m.values / 1000.0
    ax.scatter(x, y, s=26, c=DEEP_BLUE, edgecolors='none', zorder=3)
    # 风向箭头（风沿 +x）：锚定面板左缘（axes 分数坐标），六面板统一位置。
    # 原数据坐标偏移随排布跨度放大，S_A/S_E 宽排布（跨 23.5/29.4 km）的箭头
    # 被推出面板落到图左缘空白带（2026-08-28 用户反馈 a 子图排版不一致）。
    # 不用 FancyArrowPatch：mpl 3.10.9 下其与 transAxes + 450dpi savefig 组合
    # 渲染异常（尖端位置不可控、头部坐标系畸变），改为 ax.plot 手绘杆+V 形头，
    # 逐像素可控。头部尺寸 0.045×0.022 轴比 ≈ 原 mutation_scale=16 的 50×30 px。
    ax.plot([-0.100, -0.025], [0.50, 0.50], color=RED, lw=1.6, zorder=4,
            transform=ax.transAxes, clip_on=False)
    ax.plot([-0.025, 0.020, -0.025], [0.489, 0.50, 0.511], color=RED, lw=1.6,
            zorder=4, transform=ax.transAxes, clip_on=False)
    ax.text(-0.115, 0.50, 'Wind', transform=ax.transAxes,
            ha='right', va='center', fontsize=11, color=RED, clip_on=False)
    ax.set_title(TITLES[s], fontsize=12.5, pad=8, color='#111111')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('top', 'right', 'bottom', 'left'):
        ax.spines[sp].set_visible(False)
    rng = max(np.ptp(x), np.ptp(y)) * 0.06
    ax.set_xlim(x.min() - rng, x.max() + rng)
    ax.set_ylim(y.min() - rng, y.max() + rng)
    ax.text(0.012, 0.985, letter, transform=ax.transAxes, fontsize=22.2,
            fontweight='bold', va='top', ha='left', color='#111111', zorder=40)

fig.subplots_adjust(left=0.075, right=0.99, top=0.97, bottom=0.03,
                    wspace=0.18, hspace=0.36)
save_fig(fig, OUT)
