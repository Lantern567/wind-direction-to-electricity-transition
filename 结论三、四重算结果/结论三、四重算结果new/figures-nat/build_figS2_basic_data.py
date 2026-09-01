# -*- coding: utf-8 -*-
"""Figure S2：基础数据总览（171 场样本特征可视化）
====================================================================
学长要求：补充信息增加可视化，呈现基础数据；图内文字全英文；字体与主稿
图 1–4 统一（nc_style_nat.apply_style，Arial）。

面板：
  a. 171 场分布地图（East Asia / Europe / US East 三区着色，红星为 6 个
     越线场 G > SD(MS)）
  b–e. 装机容量、机组数、水深、场区面积分布（全球样本组成）
  f. 场级 Weibull 形状参数 k 与多年平均风速（2014–2024，ERA5 100 m）
     按区域着色——低风速近海场 k 更高（风况更稳定）

数据口径（全部实算）：
  场址：wp9c_farm_metrics.csv（质心）；farms_master.csv（容量/机组/面积/水深）
  风速边际与 Weibull k：wp3_climate_joint.npz 场级风速边际直方图矩法拟合
  越线集：G（orientation_gain.csv 场级多年平均）> SD(MS)（四场景分解
      farm_summary.csv M_S_std），与主稿图 1 同口径，n≥3 共 5 场（v6 权威口径，F160 比值 0.96 不越线）。

输出：figures-nat/FigS2_basic_data.png / .pdf
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, '补算', 'output')
sys.path.insert(0, HERE)
from nc_style_nat import (apply_style, panel_label, halo, north_arrow,
                          scale_bar, save_fig, RED, ORANGE, GREY, DEEP_BLUE,
                          INK, BOX_EC, LAND_COL, SEA_COL, COAST_COL,
                          GRIDLINE_COL, COAST_LW)

# ── 数据 ──
fm = pd.read_csv(os.path.join(REPO, 'data', 'task0', 'farms_master.csv'))
fm = fm.set_index('farm_id')
m = pd.read_csv(os.path.join(OUT, 'wp9c_farm_metrics.csv')).set_index('farm_id')
assert len(fm) == 171 and len(m) == 171

lon = m['cent_lon'].values.astype(float)
lat = m['cent_lat'].values.astype(float)
cap = fm['capacity_kW'].values / 1000.0              # MW
nt = fm['n_turb'].values
area = fm['area_km2'].values
dep = fm['avg_depth_m'].values

ea = lon >= 40
us = lon < -40
eu = (~ea) & (~us)
assert ea.sum() == 88 and eu.sum() == 78 and us.sum() == 5, \
    (ea.sum(), eu.sum(), us.sum())

# Weibull k 与平均风速（2014–2024 场级风速边际矩法）
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_fy, hours, ws = z3['p_fy'], z3['hours'], z3['ws']
hw = (hours.astype(float) / hours.sum(axis=1, keepdims=True))[:, :, None, None]
marg = np.sum(p_fy * hw, axis=(1, 3))                # (171,18) 场级风速边际
mu = np.sum(marg * ws[None, :], axis=1)
sig = np.sqrt(np.sum(marg * (ws[None, :] - mu[:, None]) ** 2, axis=1))
k = (sig / mu) ** (-1.086)
assert np.all((k > 1.3) & (k < 4.5)), 'Weibull k 超出合理范围'
assert np.all((mu > 5.5) & (mu < 11.5)), '平均风速超出合理范围'

# 越线集（与主稿图 1 同口径）
og = pd.read_csv(os.path.join(OUT, 'orientation_gain.csv'))
og.columns = [c.strip() for c in og.columns]
G = og.groupby('farm_id')['gain_pct'].mean()
NY = og.groupby('farm_id')['year'].count()
fs = pd.read_csv(os.path.join(REPO, '四场景风速风向分解贡献', 'output',
                              'four_scenario_farm_summary_AUTHORITATIVE.csv')).set_index('farm_id')
gv = G.reindex(fm.index).values.astype(float)
ny = NY.reindex(fm.index).values.astype(int)
MS = fs.loc[fm.index, 'M_S_std'].values.astype(float)
ratio = gv / MS
cross3 = (ny >= 3) & (ratio > 1)
c3 = fm.index[cross3].tolist()
assert sorted(c3) == [57, 66, 91, 155, 157], sorted(c3)
assert (ny >= 5).sum() == 108 and (ny >= 3).sum() == 146

# ── 样式与版面 ──
apply_style()
# 字号统一（2026-08-30 用户标准）：SI 图 15cm 页宽/5580px，页面刻度量级≈4.8pt
# → 基底 16.2、刻度 14.2、面板字母 23.3；地图内图例/罗盘/比例尺取 12 级
plt.rcParams.update({'font.size': 16.2, 'axes.labelsize': 16.2,
                     'xtick.labelsize': 14.2, 'ytick.labelsize': 14.2,
                     'legend.fontsize': 14.2})
fig = plt.figure(figsize=(12.4, 8.6))
REG_COL = {'East Asia': DEEP_BLUE, 'Europe': GREY, 'US East': ORANGE}


# ============ a. 三区分布地图 ============
# 框几何（2026-08-28 两轮修复）：
#   v1 原框 [0.055,0.53,0.42,0.44] 高度远超 cartopy 内容实际高度——GeoAxes 强制
#   等比，内容按宽填满仅 0.176 高并垂直居中于 0.44 框内，顶端飘到 y≈0.84，
#   高于旁边 b 面板顶 0.73 → 用户反馈 "a 跑到上面去"。
#   v2 宽 0.40 与 b 一致、顶 0.73 与 b 齐平、高按内容纵横比 3.4102（海色 bbox
#   实测；PlateCarree 等比 230°×cos34°/68°≈3.38）计算 = 0.1691 使内容填满
#   无内部留白。但整个版心未同步上移 → 上方留白 0.27 过大。
#   v3（本轮）整图版心重排：顶 0.86 / 底 0.14，三行各 0.20 高、行间 0.06
#   （容纳 xlabel+刻度标签，不再与下方面板重叠）。a 顶 0.86 与 b 顶齐平。
axa = fig.add_axes([0.055, 0.7109, 0.40, 0.1691], projection=ccrs.PlateCarree())
axa.set_extent([-80, 150, 0, 68], crs=ccrs.PlateCarree())
axa.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                facecolor=SEA_COL, edgecolor='none'), zorder=-1)
axa.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                facecolor=LAND_COL, edgecolor=COAST_COL, linewidth=COAST_LW), zorder=0)
axa.add_feature(cfeature.BORDERS, edgecolor='#c8c8c8', linewidth=0.3, zorder=1)
gl = axa.gridlines(linewidth=0.3, color=GRIDLINE_COL, linestyle='--',
                   draw_labels=True, dms=False,
                   xlocs=np.arange(-60, 151, 30), ylocs=np.arange(0, 69, 20))
gl.top_labels = False; gl.right_labels = False
gl.xlabel_style = {'size': 12.5}; gl.ylabel_style = {'size': 12.5}
for name, sel in (('East Asia', ea), ('Europe', eu), ('US East', us)):
    axa.scatter(lon[sel], lat[sel], s=16, c=REG_COL[name], alpha=0.9,
                edgecolors='white', linewidths=0.35, zorder=4,
                transform=ccrs.PlateCarree(), label=name)
for f in c3:
    i = fm.index.get_loc(f)
    axa.scatter(lon[i], lat[i], s=95, marker='*', c=RED, edgecolors='white',
                linewidths=0.6, zorder=6, transform=ccrs.PlateCarree())
axa.legend([mlines.Line2D([], [], marker='o', ls='', mfc=DEEP_BLUE, mec='white',
                          mew=0.4, ms=6),
            mlines.Line2D([], [], marker='o', ls='', mfc=GREY, mec='white',
                          mew=0.4, ms=6),
            mlines.Line2D([], [], marker='o', ls='', mfc=ORANGE, mec='white',
                          mew=0.4, ms=6),
            mlines.Line2D([], [], marker='*', ls='', mfc=RED, mec='white',
                          mew=0.5, ms=12)],
           ['East Asia (%d)' % ea.sum(), 'Europe (%d)' % eu.sum(),
            'US East (%d)' % us.sum(), 'G > SD(MS) (%d)' % len(c3)],
           loc='lower left', fontsize=12, ncol=2, handlelength=1.1, framealpha=0.95)
for _sp in ('top', 'bottom', 'left', 'right', 'geo'):
    try:
        axa.spines[_sp].set_visible(False)
    except KeyError:
        pass
north_arrow(axa, x=0.10, y0=0.80, y1=0.62, fs=12)
scale_bar(axa, 62, 4, km=1000, fs=11)
panel_label(axa, 'a', fs=23.3)

# ============ b–f. 分布面板 ============
def hist_panel(ax, data, bins, label, letter, log=False, unit=''):
    ax.hist(data, bins=bins, color='#b8c2cc', edgecolor='white', linewidth=0.4)
    for _sp in ('top', 'right'):
        ax.spines[_sp].set_visible(False)
    if log:
        ax.set_xscale('log')
    ax.set_xlabel(label + ((' (%s)' % unit) if unit else ''))
    ax.set_ylabel('Number of farms')
    ax.text(0.97, 0.95,
            'n = 171\nmedian = %s' % ('%.1f' % np.median(data) if np.median(data) >= 100
                                      else '%.2f' % np.median(data)),
            transform=ax.transAxes, ha='right', va='top', fontsize=12.5,
            color='#333333')
    panel_label(ax, letter, fs=23.3)


axb = fig.add_axes([0.545, 0.68, 0.40, 0.20])
hist_panel(axb, cap, np.logspace(np.log10(20), np.log10(10000), 18),
           'Installed capacity', 'b', log=True, unit='MW')
axb.set_xlim(20, 10000)

axc = fig.add_axes([0.545, 0.39, 0.40, 0.20])
hist_panel(axc, nt, np.logspace(np.log10(2), np.log10(1000), 18),
           'Turbine count', 'c', log=True)
axc.set_xlim(2, 1000)

axd = fig.add_axes([0.055, 0.39, 0.40, 0.20])
hist_panel(axd, dep, np.arange(0, 111, 10), 'Water depth', 'd', unit='m')

axe = fig.add_axes([0.545, 0.10, 0.40, 0.20])
hist_panel(axe, area, np.logspace(np.log10(0.5), np.log10(3000), 18),
           'Farm area', 'e', log=True, unit='km$^2$')
axe.set_xlim(0.5, 3000)

axf = fig.add_axes([0.055, 0.10, 0.40, 0.20])
for name, sel in (('East Asia', ea), ('Europe', eu), ('US East', us)):
    axf.scatter(mu[sel], k[sel], s=14, c=REG_COL[name], alpha=0.85,
                edgecolors='white', linewidths=0.35, zorder=3, label=name)
for _sp in ('top', 'right'):
    axf.spines[_sp].set_visible(False)
axf.set_xlabel('Mean wind speed, 2014–2024 (m s$^{-1}$)')
axf.set_ylabel('Weibull shape k (marginal fit)')
axf.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=3,
           fontsize=12.5, handlelength=1.1, framealpha=0.95)
panel_label(axf, 'f', fs=23.3)

save_fig(fig, os.path.join(HERE, 'FigS2_basic_data.png'))
print('对账: 区域 EA=%d EU=%d US=%d | Weibull k [%.2f, %.2f] | 风速 [%.2f, %.2f] m/s'
      % (ea.sum(), eu.sum(), us.sum(), k.min(), k.max(), mu.min(), mu.max()))
print('对账: 越线集 %s (n=%d) | n>=5=%d n>=3=%d'
      % (sorted(c3), len(c3), (ny >= 5).sum(), (ny >= 3).sum()))
