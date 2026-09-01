# -*- coding: utf-8 -*-
"""图 1 按学长原图五面板版式重建 v2（2026-08-30，v6 权威口径）
====================================================================
用户反馈修复：
  1. (d)(e) 不再拉长——按原图像素实测版面：标题 →(a)(b)→ 白空 →(c) 地图行
     → 白空带（地图图例）→(d)(e)。(d)(e) 高度=原图 26%（1376/5286 px）。
  2. 美国东海岸插图删除（F160 掉线后无内容），(c) 行改两幅宽图：东亚+欧洲。
  3. 地图体例与学长原图一致：白海、浅灰陆、深灰岸线（非 NC 青蓝海）。
  4. 消除坐标轴重叠：图例移出面板（地图图例放 (c)(d) 间白空带）、(d) 线端
     标签错开、统计框置 (a) 左上、色标内嵌 (a) 右侧。

数据口径（对账断言内置，全部实算）：
  G = 补算/output/orientation_gain.csv 场级多年平均；SD(MS) = AUTHORITATIVE
  表 M_S_std；越线 = G > SD(MS)（n>=5 与 n>=3 两级）；新增电量 = Σ_y
  (AEP_s1_opt - AEP_real)（与 G 同定义）。
  v6 口径：越线 n>=5=[57,66,91,157]、n>=3 加 155；F160=0.96 掉线。
  面板 e：学长原图注 5.6%/37% 不可复现（穷举口径），实算 1.5%/16.8%、
  10% 装机 →48.2%，按实算标注。

输出：figures-nat/Fig1_senior_nat.png（4342×5286，与原图同尺寸同比例）
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
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, '补算', 'output')
sys.path.insert(0, HERE)
from nc_style_nat import (apply_style, panel_label, halo, RED, ORANGE, INK,
                          BOX_EC, GREY, DEEP_BLUE, GRIDLINE_COL)

# ════════════════ 数据 ════════════════
og = pd.read_csv(os.path.join(OUT, 'orientation_gain.csv'))
og.columns = [c.strip() for c in og.columns]
G = og.groupby('farm_id')['gain_pct'].mean()
NY = og.groupby('farm_id')['year'].count()
POS = og.assign(pos=og.gain_pct > 0).groupby('farm_id')['pos'].mean() * 100
fs = pd.read_csv(os.path.join(REPO, '四场景风速风向分解贡献', 'output',
                              'four_scenario_farm_summary_AUTHORITATIVE.csv')).set_index('farm_id')
fm = pd.read_csv(os.path.join(OUT, 'wp9c_farm_metrics.csv')).set_index('farm_id')

farms = G.index
lon = fm.loc[farms, 'cent_lon'].values.astype(float)
lat = fm.loc[farms, 'cent_lat'].values.astype(float)
ny = NY[farms].values.astype(int)
gv = G.values.astype(float)
MS = fs.loc[farms, 'M_S_std'].values.astype(float)
corr = fm.loc[farms, 'is_corr'].fillna(False).values.astype(bool)

# ════════════════ 对账断言 ════════════════
assert len(farms) == 171 and (ny >= 5).sum() == 108 and (ny >= 3).sum() == 146
assert abs(np.median(gv) - 0.34) < 0.05 and abs(np.mean(gv) - 0.95) < 0.05
win = float((og['gain_pct'] > 0).mean() * 100); assert abs(win - 64.3) < 1.0
y57 = og[og.farm_id == 57].gain_pct.values
y157 = og[og.farm_id == 157].gain_pct.values
assert y57.min() >= 15.5 and y57.max() <= 22.5 and abs(y157.min() - 5.0) < 0.3
ratio = gv / MS
cross5 = (ny >= 5) & (ratio > 1)
cross3 = (ny >= 3) & (ratio > 1)
c5 = farms[cross5].tolist(); c3 = farms[cross3].tolist()
assert sorted(c5) == [57, 66, 91, 157] and sorted(c3) == [57, 66, 91, 155, 157]
assert abs(ratio[farms.get_loc(160)] - 0.96) < 0.03
print('对账: n=171  中位G=%.4f%%  均值G=%.4f%%  n>=5=%d  n>=3=%d  赢率=%.1f%%' %
      (np.median(gv), np.mean(gv), (ny >= 5).sum(), (ny >= 3).sum(), win))
print('v6 越线: n>=5: %s (%.1f%%)  n>=3: %s (%.1f%%)  比值 %.2f–%.2f  F160=%.3f' %
      (sorted(c5), 100 * len(c5) / 108, sorted(c3), 100 * len(c3) / 146,
       ratio[cross3].min(), ratio[cross3].max(), ratio[farms.get_loc(160)]))

# ════════════════ 面板 e 数据 ════════════════
s4 = pd.read_csv(os.path.join(REPO, 'task3', 'Task3-output', 's4_balanced_gauss.csv'))
real = s4[s4.layout_type == 'real'].set_index(['farm_id', 'year'])['AEP_kWh']
rows = []
for (fid, y), r in s4[s4.layout_type != 'real'].groupby(['farm_id', 'year']):
    th = int(r.theta_opt.iloc[0])
    sel = r[r.layout_type == 's1_opt_%ddeg' % th]
    rows.append((fid, y, sel.AEP_kWh.iloc[0] - real.get((fid, y), np.nan)))
df = pd.DataFrame(rows, columns=['farm_id', 'year', 'dAEP_kWh'])
assert len(df) == 1203 and df.farm_id.nunique() == 171
dE = df.groupby('farm_id')['dAEP_kWh'].sum() / 1e9
cap = fm.loc[dE.index, 'n_turb'].astype(float) * 10 / 1000
order_e = G.sort_values(ascending=False).index
csum_cap = cap[order_e].cumsum() / cap.sum() * 100
csum_e = dE[order_e].cumsum() / dE[order_e].sum() * 100
top5 = G.sort_values(ascending=False).head(5).index
top5_es = float(dE[top5].sum() / dE.sum() * 100)
top5_cs = float(cap[top5].sum() / cap.sum() * 100)
i10 = int(np.searchsorted(csum_cap.values, 10.0))
e_at_10 = float(csum_e.iloc[i10])
assert list(top5) == [57, 155, 66, 157, 91]
assert abs(top5_cs - 1.5) < 0.15 and abs(top5_es - 16.8) < 0.5
print('面板e: top5=%s  装机 %.1f%%  新增电量 %.1f%%  10%%装机→%.1f%%' %
      (list(top5), top5_cs, top5_es, e_at_10))

# ════════════════ 命名与配色 ════════════════
COAST = {57: 'Vietnam coast', 66: 'Hangzhou Bay', 91: 'Pearl River mouth',
         155: 'Gulf of Taranto', 157: 'Danish straits'}
LINE_COL = {57: RED, 66: DEEP_BLUE, 91: '#a577ad', 155: ORANGE, 157: '#73c79e'}
CROSS_COL, HIGH_COL, BASE_COL = RED, ORANGE, '#b8c2cc'
MAP_BASE = '#878175'  # 原图 (c) 灰点实测色
# 学长原图地图体例：白海、浅灰陆、深灰岸线
MAP_SEA, MAP_LAND, MAP_COAST = '#ffffff', '#e9ecef', '#8a8a8a'
apply_style()
# 字号统一：学长图 2/3/4 页面刻度数字≈4.8pt（44–50px @ 4196–4462px 宽、16.3cm 页宽），
# 本图 4342px 宽下需面板内字号 ×1.45（刻度 7→10.2、基底 8→11.6、面板字母 11.5→16.7）
FS = 1.45
plt.rcParams.update({'font.size': 11.6, 'axes.labelsize': 11.6,
                     'xtick.labelsize': 10.2, 'ytick.labelsize': 10.2,
                     'legend.fontsize': 10.2})

# ════════════════ 画布（与原图同像素 4342×5286） ════════════════
W, H, DPI = 4342, 5286, 450
fig = plt.figure(figsize=(W / DPI, H / DPI))
# 原图像素实测版面（分数坐标）；标题已按用户要求删除（论文图不用标题）

# ────────── a. 场级增益长尾排序散点 ──────────
axa = fig.add_axes([333 / W, (H - 1532) / H, (2120 - 333) / W, 1402 / H])
order = np.argsort(-gv)
rank = np.empty(len(gv)); rank[order] = np.arange(1, len(gv) + 1)
axa.set_xlim(0, 172); axa.set_yscale('symlog', linthresh=1.0)
axa.set_ylim(-2.0, 21)
axa.set_yticks([-1, 0, 1, 3, 10, 20])
axa.set_ylabel('Multi-year mean reorientation gain, G (%)')
axa.set_xlabel('Farms ranked by G (n = 171)')
cmap = LinearSegmentedColormap.from_list('posyrs', [DEEP_BLUE, '#c9d2da', RED])
sc = axa.scatter(rank, gv, c=POS[farms], cmap=cmap, vmin=0, vmax=100,
                 s=15, linewidths=0.2, edgecolors='white', zorder=2)
axa.axhline(np.median(gv), color=RED, ls='--', lw=1.0, zorder=1)
axa.axhline(np.mean(gv), color='#7a7a7a', ls=':', lw=1.2, zorder=1)
axa.text(0.035, 0.055, 'Median %.2f%%  |  Mean %.2f%%' % (np.median(gv), np.mean(gv)),
         transform=axa.transAxes, ha='left', va='bottom', fontsize=10.2, color='#333333',
         bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.6, alpha=0.95, pad=2.5),
         zorder=8)
LAB_A = {57: (3, 18.0), 155: (6, 10.8)}
for f in (57, 155):
    i = farms.get_loc(f)
    halo(axa, rank[i] + LAB_A[f][0], LAB_A[f][1],
         'F%d (%.1f%%)' % (f, gv[i]), color=CROSS_COL, fs=10.7, ha='left')
cax = fig.add_axes([1950 / W, (H - 1100) / H, 0.011, 700 / H])
cb = fig.colorbar(sc, cax=cax)
cb.set_ticks([0, 50, 100]); cb.ax.tick_params(labelsize=9.0)
cax.text(0.5, 1.10, 'Years with a\npositive gain (%)', transform=cax.transAxes,
         ha='center', va='bottom', fontsize=9.3, color='#333333')
for sp in ('top', 'right'):
    axa.spines[sp].set_visible(False)
panel_label(axa, 'a', fs=16.7)

# ────────── b. G vs SD(MS) 1:1 ──────────
axb = fig.add_axes([2484 / W, (H - 1506) / H, (4270 - 2484) / W, 1376 / H])
s146 = ny >= 3
xm = 11.0; ym = 21.0
axb.set_xlim(-0.5, xm); axb.set_ylim(-2.4, ym)
axb.plot([-0.5, 10.5], [0.0, 11.0], '--', color='#9aa5ad', lw=1.0, zorder=1)
below = s146 & ~cross3
axb.scatter(MS[below], gv[below], s=11, c=BASE_COL, linewidths=0, zorder=2)
axb.scatter(MS[cross5], gv[cross5], s=42, c=CROSS_COL, edgecolors='white',
            linewidths=0.5, zorder=5)
axb.scatter(MS[cross3 & ~cross5], gv[cross3 & ~cross5], s=46, facecolors='white',
            edgecolors=CROSS_COL, linewidths=1.5, zorder=5)
OFF_B = {57: (-4.2, -0.7), 66: (1.1, 0.6), 91: (1.3, -1.0), 155: (-2.8, 1.2),
         157: (1.1, 0.5)}
for f in (57, 66, 91, 155, 157):
    i = farms.get_loc(f)
    halo(axb, MS[i] + OFF_B[f][0], gv[i] + OFF_B[f][1], COAST[f],
         color='#222222', fs=10.4, weight='normal')
axb.set_xlabel('Own interannual wind-speed variability (%)')
axb.set_ylabel('Multi-year mean reorientation gain, G (%)')
axb.legend(
    [plt.Line2D([], [], marker='o', ls='', mfc=BASE_COL, mec='none', ms=4),
     plt.Line2D([], [], marker='o', ls='', mfc=CROSS_COL, mec='white', mew=0.4, ms=5.5),
     plt.Line2D([], [], marker='o', ls='', mfc='white', mec=CROSS_COL, mew=1.3, ms=5.5)],
    ['Below own variability (n = %d)' % int(below.sum()),
     'Crossing, ≥5 yr record (n = %d)' % int(cross5.sum()),
     'Crossing, 3–4 yr record (n = %d)' % int((cross3 & ~cross5).sum())],
    loc='lower right', fontsize=9.6, handlelength=1.1, columnspacing=0.9,
    handletextpad=0.35)
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
panel_label(axb, 'b', fs=16.7)

# ────────── c. 两幅区域地图（美国插图已删） ──────────
def draw_region(ax, sel, xlocs, ylocs):
    # 面板盒 1787×1376；cartopy 按原始经纬度比例（1°lon=1°lat）在盒内居中绘制，
    # 故 extent 的 dx/dy 取 1.2985（=盒纵横比）即内容满盒；风场范围保底 ±2.5°，
    # 经度跨幅上限 45°。
    la_min, la_max = lat[sel].min(), lat[sel].max()
    dy = max(la_max - la_min + 5.0, (lon[sel].max() - lon[sel].min() + 5.0) / 1.2985)
    dx = min(dy * 1.2985, 45.0)
    lc = (lon[sel].min() + lon[sel].max()) / 2
    lmid = (la_min + la_max) / 2
    ext = [lc - dx / 2, lc + dx / 2, lmid - dy / 2, lmid + dy / 2]
    print('地图范围: lon %.0f..%.0f lat %.0f..%.0f' % tuple(ext))
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
        facecolor=MAP_SEA, edgecolor='none'), zorder=-1)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
        facecolor=MAP_LAND, edgecolor=MAP_COAST, linewidth=0.6), zorder=0)
    ax.add_feature(cfeature.BORDERS, edgecolor='#c0c0c0', linewidth=0.3, zorder=1)
    gl = ax.gridlines(linewidth=0.3, color=GRIDLINE_COL, linestyle='--',
                      draw_labels=True, dms=False, xlocs=xlocs, ylocs=ylocs)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 9.0}; gl.ylabel_style = {'size': 9.0}
    size = np.clip(10 + gv * 9, 8, 160)
    others = sel & ~corr & ~cross3
    ax.scatter(lon[others], lat[others], s=size[others], c=MAP_BASE, alpha=0.9,
               edgecolors='white', linewidths=0.3, zorder=4,
               transform=ccrs.PlateCarree())
    corr_sel = sel & corr & ~cross3
    ax.scatter(lon[corr_sel], lat[corr_sel], s=size[corr_sel], c=HIGH_COL,
               alpha=0.9, edgecolors='white', linewidths=0.35, zorder=5,
               transform=ccrs.PlateCarree())
    for f in sorted(c3):
        if sel[farms.get_loc(f)]:
            i = farms.get_loc(f)
            ax.scatter(lon[i], lat[i], s=230, marker='*', c=CROSS_COL,
                       edgecolors='white', linewidths=0.6, zorder=6,
                       transform=ccrs.PlateCarree())


# (c) 两幅插图与 a/b/d/e 同尺寸（1787×1376，x 列对齐），版心：c 行 y 1900..3276
axc_ea = fig.add_axes([333 / W, (H - 3276) / H, (2120 - 333) / W, 1376 / H],
                      projection=ccrs.PlateCarree())
draw_region(axc_ea, (lon >= 100) & (lon <= 130) & (lat >= 5) & (lat <= 38),
            np.arange(95, 136, 10), np.arange(10, 39, 10))
axc_eu = fig.add_axes([2484 / W, (H - 3276) / H, (4271 - 2484) / W, 1376 / H],
                      projection=ccrs.PlateCarree())
draw_region(axc_eu, (lon >= -6) & (lon <= 24) & (lat >= 36) & (lat <= 62),
            np.arange(-10, 32, 10), np.arange(40, 61, 10))
for _ax in (axc_ea, axc_eu):
    for _sp in ('top', 'bottom', 'left', 'right', 'geo'):
        try:
            _ax.spines[_sp].set_visible(False)
        except KeyError:
            pass
panel_label(axc_ea, 'c', fs=16.7)

# 地图图例：放 (c)(d) 之间白空带（学长原图位置）
lg = fig.add_axes([0.0767, (H - 3460) / H, 0.62, 105 / H])
lg.set_axis_off()
lg.legend([mlines.Line2D([], [], marker='*', ls='', mfc=RED, mec='white', mew=0.5, ms=11),
           mlines.Line2D([], [], marker='o', ls='', mfc=HIGH_COL, mec='white', mew=0.4, ms=6),
           mlines.Line2D([], [], marker='o', ls='', mfc=MAP_BASE, mec='white', mew=0.4, ms=5)],
          ['G exceeds own variability', 'Other corridor farm', 'Other farm'],
          loc='center left', fontsize=10.4, ncol=3, columnspacing=2.0,
          handletextpad=0.4, frameon=False)

# ────────── d. 越线风场逐年增益 + 全部风场 IQR 带 ──────────
axd = fig.add_axes([333 / W, (H - 5049) / H, (2120 - 333) / W, 1376 / H])
gp = og.pivot_table(index='year', columns='farm_id', values='gain_pct')
axd.set_xlim(2014, 2026)
axd.set_yscale('symlog', linthresh=0.3)
axd.set_ylim(-0.3, 24)
axd.set_yticks([0, 0.5, 1, 3, 10, 20])
axd.set_ylabel('Gain (%)')
axd.set_xlabel('Year', labelpad=1)
p50 = gp.median(axis=1); p25 = gp.quantile(0.25, axis=1); p75 = gp.quantile(0.75, axis=1)
axd.fill_between(gp.index, p25, p75, color='#d6dbe0', alpha=0.85, zorder=1,
                 edgecolor='#b8c2cc', linewidth=0.5)
axd.plot(gp.index, p50, color='#8e99a4', lw=1.8, zorder=2)
# 5 条越线风场：5 种线型 + 5 种线端 marker（色盲友好）
LSTYLE = {57: '-', 155: '-', 66: '--', 91: ':', 157: '-.'}
LMARK = {57: 'o', 155: 's', 66: '^', 91: 'v', 157: 'D'}
for f in (57, 66, 91, 155, 157):
    yv = gp[f].dropna()
    axd.plot(yv.index, yv.values, color=LINE_COL[f], lw=1.6, ls=LSTYLE[f],
             zorder=4)
    axd.scatter([yv.index[-1]], [yv.values[-1]], s=24, marker=LMARK[f],
                c=LINE_COL[f], edgecolors='white', linewidths=0.6, zorder=5)
# 名称移到 (d) 面板正下方一行（标记+名称，与线色一一对应，无引线零重叠）。
# (d)(e) 间空隙实际可用仅 280px（2400 之后是 e 的 y 轴刻度标签），
# 最长名 'Pearl River mouth' 6pt≈330px 放不下，故置于面板下方留白带。
ITEMS = [(57, 'o'), (155, 's'), (66, '^'), (157, 'D'), (91, 'v')]
COL_X = 0.015 + 0.194 * np.arange(5)
for k, (f, mk) in enumerate(ITEMS):
    xf = COL_X[k]
    axd.plot([xf + 0.006], [-0.155], transform=axd.transAxes, marker=mk,
             ms=4.6, mfc=LINE_COL[f], mec='white', mew=0.5, clip_on=False,
             zorder=6)
    axd.text(xf + 0.020, -0.155, COAST[f], transform=axd.transAxes,
             color=LINE_COL[f], fontsize=6.0, va='center', ha='left',
             fontweight='bold', clip_on=False, zorder=6,
             path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
axd.legend([plt.matplotlib.patches.Patch(facecolor='#d6dbe0', edgecolor='#b8c2cc',
                                         linewidth=0.5),
            plt.Line2D([], [], color='#8e99a4', lw=1.8)],
           ['All farms, interquartile range', 'All farms, median'],
           loc='lower left', fontsize=9.0, handlelength=1.2, frameon=True,
           framealpha=0.95, borderpad=0.4)
for sp in ('top', 'right'):
    axd.spines[sp].set_visible(False)
panel_label(axd, 'd', fs=16.7)

# ────────── e. 集中度曲线 ──────────
axe = fig.add_axes([2457 / W, (H - 5049) / H, (4269 - 2457) / W, 1376 / H])
axe.set_xlim(-2, 102); axe.set_ylim(-6, 112)
axe.set_xlabel('Cumulative share of installed capacity (%)')
axe.set_ylabel('Cumulative share of added energy (%)')
axe.set_xticks(np.arange(0, 101, 20)); axe.set_yticks(np.arange(0, 101, 20))
axe.fill_between(csum_cap.values, csum_e.values, color='#2664eb', alpha=0.30, zorder=1)
axe.plot(csum_cap.values, csum_e.values, color='#5b6770', lw=1.4, zorder=3)
axe.scatter(csum_cap.values, csum_e.values, s=6, c=BASE_COL, linewidths=0, zorder=2)
for f in top5:
    axe.scatter(csum_cap.loc[f], csum_e.loc[f], s=36, c=ORANGE,
                edgecolors='white', linewidths=0.5, zorder=5)
x5, y5 = float(csum_cap.loc[top5[-1]]), float(csum_e.loc[top5[-1]])
# 两条注释放右侧中部空白区（x≥35 曲线已超 ylim；ha='right' 防盒超右缘），
# 直线箭头指向目标（数学验证全程在曲线下方，不穿线）。
# 对调位置：目标高的 (10,48.2) 在上、目标低的 (1.5,16.8) 在下，两箭头不交叉
x10 = float(csum_cap.iloc[i10]); y10 = float(csum_e.iloc[i10])
axe.scatter([10.0], [y10], s=42, marker='o', facecolors='none',
            edgecolors='#333333', linewidths=1.2, zorder=6)
axe.annotate('10%% of capacity carries %.0f%%\nof added energy' % e_at_10,
             xy=(10.0, y10), xytext=(97, 60), fontsize=9.9, color='#333333',
             ha='right', va='center',
             arrowprops=dict(arrowstyle='->', color='#333333', lw=0.7),
             bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.5, alpha=0.95, pad=2.2))
axe.annotate('%d highest-gain farms: %.1f%% of capacity,\n%.1f%% of added energy'
             % (len(top5), top5_cs, top5_es), xy=(x5, y5),
             xytext=(97, 40), fontsize=9.9, color='#333333', ha='right', va='center',
             arrowprops=dict(arrowstyle='->', color='#333333', lw=0.7),
             bbox=dict(facecolor='white', edgecolor=BOX_EC, lw=0.5, alpha=0.95, pad=2.2))
for sp in ('top', 'right'):
    axe.spines[sp].set_visible(False)
panel_label(axe, 'e', fs=16.7)

# ════════════════ 保存（与原图同像素尺寸） ════════════════
out_png = os.path.join(HERE, 'Fig1_senior_nat.png')
fig.savefig(out_png, dpi=DPI, bbox_inches=None, pad_inches=0)
from PIL import Image
assert Image.open(out_png).size == (W, H), Image.open(out_png).size
print('saved:', out_png, Image.open(out_png).size)
print('OK  Fig1_senior_nat  越线 n>=5: %s  n>=3: %s  F160=%.3f' %
      (sorted(c5), sorted(c3), ratio[farms.get_loc(160)]))
