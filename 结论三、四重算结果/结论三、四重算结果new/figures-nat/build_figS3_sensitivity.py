# -*- coding: utf-8 -*-
"""Figure S3：敏感性分析可视化（汇总四个敏感性维度）
====================================================================
学长要求：补充信息增加可视化，呈现敏感性分析；图内文字全英文；字体与
主稿图 1–4 统一（nc_style_nat.apply_style，Arial）。

面板：
  a. 记录长度敏感性：G 的累积分布（n≥5 主判据 n=108 vs 3≤n<5 附加 n=38），
     红星标出 5 个越线场 G 位置；主判据 4/108（3.7%）、放宽后 5/146。
     （v6 权威口径：four_scenario_farm_summary_AUTHORITATIVE.csv；F160 比值
     1.03→0.96 掉线，见重算回执 P0-1。）
  b. 机型口径敏感性（SI 表 S8）：走廊/其他 A 倍比随机型的变化（区间为
     六范式范围），基准 IEA 10 MW 为 5.5–7.4×。
  c. 受控气候反事实 C0–C3（SI 表 S4）：风向均匀化后 A 严格为 0，风速—
     风向解耦贡献约三成。
  d. 风速窗口机制诊断（5.8 节，补算 wp9g）：5 个越线场最优相位相对平均
     风向的可恢复能量按风速窗口分解，全部以 7–10 m/s 部分负荷窗口为主。

数据口径：a 与主稿图 1 同口径（G = orientation_gain.csv 场级多年平均；
SD(MS) = 四场景分解 M_S_std）；b/c 为 SI 冻结表数值；d 来自
补算/output/wp9g_windwindow_diagnosis.csv（口径 B＝平均风向）。

输出：figures-nat/FigS3_sensitivity.png / .pdf
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, '补算', 'output')
sys.path.insert(0, HERE)
from nc_style_nat import (apply_style, panel_label, halo, save_fig, RED,
                          ORANGE, GREY, DEEP_BLUE, MID_BLUE, LIGHT_BLUE,
                          PALE_GREEN, LIGHT_GREY, INK, BOX_EC)

# ── 数据 a/d ──
og = pd.read_csv(os.path.join(OUT, 'orientation_gain.csv'))
og.columns = [c.strip() for c in og.columns]
G = og.groupby('farm_id')['gain_pct'].mean()
NY = og.groupby('farm_id')['year'].count()
gv = G.values.astype(float)
ny = NY[G.index].values.astype(int)
fs = pd.read_csv(os.path.join(REPO, '四场景风速风向分解贡献', 'output',
                              'four_scenario_farm_summary_AUTHORITATIVE.csv')).set_index('farm_id')
MS = fs.loc[G.index, 'M_S_std'].values.astype(float)
ratio = gv / MS
cross5 = (ny >= 5) & (ratio > 1)
cross3 = (ny >= 3) & (ratio > 1)
c5 = G.index[cross5].tolist(); c3 = G.index[cross3].tolist()
assert sorted(c5) == [57, 66, 91, 157], sorted(c5)
assert sorted(c3) == [57, 66, 91, 155, 157], sorted(c3)
assert (ny >= 5).sum() == 108 and (ny >= 3).sum() == 146
assert len(G) == 171

wd = pd.read_csv(os.path.join(OUT, 'wp9g_windwindow_diagnosis.csv'))
assert sorted(wd.farm_id.tolist()) == [57, 66, 91, 155, 157, 160]
wd = wd[wd.farm_id.isin([57, 66, 91, 155, 157])].reset_index(drop=True)  # v6 口径 5 越线场
W_B = wd[['farm_id', 'shareB_3_6', 'shareB_7_10', 'shareB_11_14',
          'shareB_15_25', 'recover_B_pct']].copy()
assert np.allclose(W_B.iloc[:, 1:5].sum(axis=1), 100, atol=0.5), '窗口份额未闭合'
assert wd.E_opt_over_max.min() >= 0.996

# ── SI 表 S8 机型敏感性（冻结常数）──
TURBINE = [('NREL 5 MW', 5.4, 6.7), ('IEA 10 MW\n(base)', 5.5, 7.4),
           ('IEA 15 MW', 5.6, 7.4), ('Era-matched', 5.1, 6.8)]
# ── SI 表 S4 受控气候反事实（冻结常数）──
C0C3 = [('C0\nobserved\njoint dist.', 1.09, 0, ''),
        ('C1\ndecoupled\nmarginals', 0.75, -0.34, '−30.9%'),
        ('C2\n+uniform\nspeed marg.', 0.72, -0.37, ''),
        ('C3\nuniform\ndirection', 0.0, -1.09, 'A = 0')]

apply_style()
# 字号统一（2026-08-30 用户标准）：SI 图 15cm 页宽/5580px → 基底 16.2、刻度 14.2、
# 面板字母 23.3；面板内注释盒 13.5 级
plt.rcParams.update({'font.size': 16.2, 'axes.labelsize': 16.2,
                     'xtick.labelsize': 14.2, 'ytick.labelsize': 14.2,
                     'legend.fontsize': 14.2})
fig = plt.figure(figsize=(12.4, 11.0))

# ============ a. 记录长度敏感性 ECDF ============
axa = fig.add_axes([0.055, 0.575, 0.42, 0.40])
xs = np.linspace(gv.min() - 0.2, gv.max() + 0.2, 600)
for sel, col, ls, lbl in ((ny >= 5, DEEP_BLUE, '-', 'n ≥ 5 yr (main, n = 108)'),
                          ((ny >= 3) & (ny < 5), GREY, '--', '3 ≤ n < 5 yr (n = 38)')):
    s = np.sort(gv[sel])
    y = np.arange(1, len(s) + 1) / len(s)
    axa.step(np.concatenate([[xs[0]], s]), np.concatenate([[0], y]),
             where='post', color=col, ls=ls, lw=1.3)
axa.set_xlabel('Multi-year mean reorientation gain, G (%)')
axa.set_ylabel('Cumulative fraction')
# 5 个 F 标签在 6.6–18.2% 密集段分层错开（10.5pt 宽≈1.75u）
LV = {57: 1.06, 155: 1.06, 157: 1.06, 66: 1.17, 91: 1.28}
for f in c3:
    axa.scatter(gv[G.index.get_loc(f)], 1.02, marker='*', s=110, c=RED,
                edgecolors='white', linewidths=0.6, zorder=6, clip_on=False)
    axa.annotate('F%d' % f, (gv[G.index.get_loc(f)], LV[f]),
                 ha='center', va='bottom', fontsize=10.5, color=RED,
                 annotation_clip=False, zorder=6)
axa.set_ylim(0, 1.34)
axa.legend(loc='upper left', bbox_to_anchor=(0.0, -0.18), ncol=2,
           fontsize=11.5, handlelength=1.8, framealpha=0.95)
axa.text(0.03, 0.955,
         'G > SD(MS): 5/108 = 4.6% (≥5 yr)\nG > SD(MS): 6/146 = 4.1% (≥3 yr)',
         transform=axa.transAxes, va='top', fontsize=13.5, color='#333333',
         bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.6, alpha=0.9, pad=2.5))
for _sp in ('top', 'right'):
    axa.spines[_sp].set_visible(False)
panel_label(axa, 'a', fs=23.3)

# ============ b. 机型口径敏感性（表 S8）============
axb = fig.add_axes([0.545, 0.575, 0.42, 0.40])
names = [t[0] for t in TURBINE]
lo = np.array([t[1] for t in TURBINE])
hi = np.array([t[2] for t in TURBINE])
x = np.arange(len(TURBINE))
axb.bar(x, hi - lo, bottom=lo, width=0.55, color='#c9d2da',
        edgecolor=GREY, linewidth=0.6, zorder=2)
axb.scatter(x, (lo + hi) / 2, s=26, c=DEEP_BLUE, edgecolors='white',
            linewidths=0.5, zorder=4)
axb.hlines(5.5, -0.5, 0.0, color=RED, lw=1.2, zorder=3)
axb.hlines(7.4, -0.5, 0.0, color=RED, lw=1.2, zorder=3)
axb.set_xticks(x); axb.set_xticklabels(names, fontsize=13)
axb.set_ylabel('Corridor / other A ratio (six-paradigm range)')
axb.set_ylim(4.6, 7.9)
for xi, (l, h) in enumerate(zip(lo, hi)):
    axb.text(xi, h + 0.07, '%.1f–%.1f' % (l, h), ha='center', va='bottom',
             fontsize=12.5, color='#333333')
axb.text(0.02, 0.955, 'IEA 10 MW (base): 5.5–7.4×\n(all p < 1×10$^{-10}$)',
         transform=axb.transAxes, va='top', fontsize=13.5, color='#333333',
         bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.6, alpha=0.9, pad=2.5))
for _sp in ('top', 'right'):
    axb.spines[_sp].set_visible(False)
panel_label(axb, 'b', fs=23.3)

# ============ c. 受控气候反事实 C0–C3（表 S4）============
axc = fig.add_axes([0.055, 0.035, 0.42, 0.40])
A = np.array([c[1] for c in C0C3])
xc = np.arange(len(C0C3))
cols = [DEEP_BLUE, MID_BLUE, LIGHT_BLUE, '#c9d2da']
axc.bar(xc, A, width=0.58, color=cols, edgecolor=INK, linewidth=0.6, zorder=2)
axc.set_xticks(xc)
axc.set_xticklabels([c[0] for c in C0C3], fontsize=12)
axc.set_ylabel('Response amplitude, A (%)')
axc.set_ylim(0, 1.3)
for xi, (_, a, d, note) in enumerate(C0C3):
    if d or note:
        lbl = '%+.2f pp' % d
        if note:
            lbl += ' (%s)' % note
        axc.text(xi, a + 0.06, lbl, ha='center', va='bottom',
                 fontsize=12.5, color=RED)
        axc.annotate('', xy=(xi, a + 0.045), xytext=(xi, a + 0.005),
                     arrowprops=dict(arrowstyle='-', color=RED, lw=0.8))
axc.text(0.03, 0.955, 'Uniform direction ⇒ A = 0\n(direction structure is necessary)',
         transform=axc.transAxes, va='top', fontsize=13.5, color='#333333',
         bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.6, alpha=0.9, pad=2.5))
for _sp in ('top', 'right'):
    axc.spines[_sp].set_visible(False)
panel_label(axc, 'c', fs=23.3)

# ============ d. 风速窗口分解（wp9g，口径 B）============
axd = fig.add_axes([0.545, 0.035, 0.42, 0.40])
WINS = [('3–6 m/s', LIGHT_GREY), ('7–10 m/s', MID_BLUE),
        ('11–14 m/s', DEEP_BLUE), ('15–25 m/s', ORANGE)]
farms = W_B.farm_id.values
ypos = np.arange(len(farms))[::-1]
left = np.zeros(len(farms))
for (name, col), key in zip(WINS, ['shareB_3_6', 'shareB_7_10',
                                   'shareB_11_14', 'shareB_15_25']):
    v = W_B[key].values
    axd.barh(ypos, v, left=left, height=0.58, color=col, edgecolor='white',
             linewidth=0.5, label=name, zorder=2)
    left += v
for yi, f, rec in zip(ypos, farms, W_B.recover_B_pct.values):
    axd.text(102, yi, 'F%d\n+%.2f%%' % (f, rec), va='center', ha='left',
             fontsize=12.5, color='#333333')
axd.set_yticks(ypos)
axd.set_yticklabels(['F%d' % f for f in farms], fontsize=13)
axd.set_xlabel('Share of recoverable energy by wind-speed window (%)')
axd.set_xlim(0, 128)
axd.legend(loc='lower left', bbox_to_anchor=(0.0, 1.06), ncol=4,
           fontsize=11.5, handlelength=1.1, framealpha=0.95)
axd.text(0.03, 0.955, 'Optimal vs mean-direction phase,\n'
         'Weibull × direction-frequency weights',
         transform=axd.transAxes, va='top', fontsize=13.5, color='#333333',
         bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.6, alpha=0.9, pad=2.5))
for _sp in ('top', 'right'):
    axd.spines[_sp].set_visible(False)
panel_label(axd, 'd', fs=23.3)

save_fig(fig, os.path.join(HERE, 'FigS3_sensitivity.png'))
print('对账: 越线 n>=5: %s (n=%d)  n>=3: %s (n=%d)  %d/108=3.7%%  %d/146=3.4%%'
      % (sorted(c5), len(c5), sorted(c3), len(c3), len(c5), len(c3)))
print('对账: 窗口份额闭合（口径B），E(th_star)/E(max) min=%.4f'
      % wd.E_opt_over_max.min())
