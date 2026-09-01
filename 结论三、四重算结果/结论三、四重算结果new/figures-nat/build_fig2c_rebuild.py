# -*- coding: utf-8 -*-
"""图 2 (c) 面板重建（2026-08-30，用户反馈：文字重叠/被线遮挡）
====================================================================
学长原图 2 无构建脚本；本脚本只重建 (c) 面板的内容区（斜率图），
粘贴回 image2.png 的 (290,1350)-(2475,3100) 区域——原图面板边框线保留不动。

数据（实算，与冻结口径核对）：
  wp5c_farm_cross.csv 的 A_C0..A_C3（171 场 × 6 范式）：
  总平均 C0 1.0897 / C1 0.7531 / C2 0.7233 / C3 0.0 —— 复现 SI 表 S4 与正文
  （1.09% / 0.75% / 0.72% / 0）；灰线 = 六范式各自跨场均值。

字号（图 2 为 4196px @16.3cm 页宽，页面刻度量级≈4.8pt 标准）：
  刻度 9.7、基底 11.1、面板字母 16.1、注释 9.5-9.7。

文字防重叠：
  数值标签带白描边并避开灰线（灰线最高 1.68 在 C0，标签 1.09 置于其上；
  C1 灰线 1.12 与蓝线 0.75 之间标签置 0.81）；−30.9% 注释放 C0–C1 段上方
  空白（灰线群最高点之下、标注自身独占）；图例盒左上。

输出：figures-nat/_fig2c_panel.png（2185×1750 内容区）
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from nc_style_nat import apply_style, RED, INK, DEEP_BLUE, BOX_EC

# ── 数据 ──
d = pd.read_csv(os.path.join(REPO, '补算', 'output', 'wp5c_farm_cross.csv'))
SCEN = ['A_C0', 'A_C1', 'A_C2', 'A_C3']
para = d.groupby('paradigm')[SCEN].mean()          # 6 范式均值
means = d[SCEN].mean().values                        # 总平均
assert abs(means[0] - 1.0897) < 0.001 and abs(means[1] - 0.7531) < 0.001
assert abs(means[2] - 0.7233) < 0.001 and means[3] < 1e-6
assert abs(para.values.max() - 1.681) < 0.01
print('对账: 总平均 C0=%.4f C1=%.4f C2=%.4f C3=%.4f' % tuple(means))
print('对账: 六范式跨场均值范围 C0 %.3f..%.3f' % (para['A_C0'].min(), para['A_C0'].max()))

# ── 画布（内容区 2185×1750 px，450dpi） ──
apply_style()
W, H, DPI = 2197, 1769, 450
plt.rcParams.update({'font.size': 11.1, 'axes.labelsize': 11.1,
                     'xtick.labelsize': 9.7, 'ytick.labelsize': 9.7,
                     'legend.fontsize': 9.5})
fig = plt.figure(figsize=(W / DPI, H / DPI))
ax = fig.add_axes([0.075, 0.185, 0.88, 0.72])
# 四边框齐全（原图面板外框即坐标轴框），仅一套框线

# 六范式灰线
X = np.arange(4)
for p in para.index:
    ax.plot(X, para.loc[p].values, color='#9aa5ad', lw=1.1, zorder=2)
# 蓝线总平均
ax.plot(X, means, color=DEEP_BLUE, lw=2.2, zorder=4,
        marker='o', ms=3.6, mfc=DEEP_BLUE, mec='white', mew=0.6)

ax.set_xlim(-0.4, 3.4); ax.set_ylim(0, 1.9)
ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.set_ylabel('Response amplitude, A (%)')
ax.set_xticks(X)

# 情景名（两行，轴下方，与原图措辞一致）
LABS = ['Observed\njoint distribution', 'Decoupled\nmarginals',
        'Uniform\nwind speed', 'Uniform\ndirection']
ax.tick_params(axis='x', length=4, colors='#222222')
for x, s in zip(X, LABS):
    ax.text(x, -0.07, s, transform=ax.transAxes, ha='center', va='top',
            fontsize=9.7, color='#222222')

# 数值标签（白描边，避开灰线）
ST = dict(path_effects=[pe.withStroke(linewidth=2.6, foreground='white')], zorder=6)
ax.text(0.02, 1.12, '1.09', ha='center', va='bottom', fontsize=9.7,
        color='#333333', **ST)
ax.text(0.98, 0.81, '0.75', ha='center', va='bottom', fontsize=9.7,
        color='#333333', **ST)
ax.text(2.02, 0.78, '0.72', ha='center', va='bottom', fontsize=9.7,
        color='#333333', **ST)
ax.text(3.02, 0.07, 'A = 0', ha='left', va='bottom', fontsize=9.7,
        color='#333333', **ST)

# −30.9% 标注（C0–C1 段上方空白，灰线群最高点之上）
ax.text(0.5, 1.47, '−0.34 pp\n(−30.9%)', ha='center', va='bottom',
        fontsize=9.5, color=RED, **ST)

# 图例盒（左上）
import matplotlib.lines as mlines
ax.legend([mlines.Line2D([], [], color=DEEP_BLUE, lw=2.2),
           mlines.Line2D([], [], color='#9aa5ad', lw=1.1)],
          ['Mean across all farms', 'Each of six paradigms'],
          loc='upper right', fontsize=9.5, handlelength=1.4, frameon=True,
          framealpha=0.95, borderpad=0.45)

# 面板字母 c
ax.text(0.012, 0.99, 'c', transform=ax.transAxes, fontsize=16.1,
        fontweight='bold', va='top', ha='left', color=INK, zorder=10)

out = os.path.join(HERE, '_fig2c_panel.png')
fig.savefig(out, dpi=DPI, bbox_inches=None, pad_inches=0)
from PIL import Image
assert Image.open(out).size == (W, H), Image.open(out).size
print('saved:', out, Image.open(out).size)
