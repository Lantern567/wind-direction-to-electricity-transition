"""
图 4 结论四：S1-S3 三级建设自由度（四面板，方案 §5.8）
====================================================
a. 候选走廊与纳入核算项目容量地图（点大小 ∝ 项目容量，背景为 R 稳健走廊）
b. S1/S2/S3 百分比增益及 5-95% 区间（G_plan，非 sparse 项目）
c. 各走廊新增 GWh 贡献（dE_GWh_S1/S2/S3 分组柱）
d. V1/V2/V3 增量价值分解（朝向校正/类型保持重排/规划前置重构 的份额）

口径（方案 §5.6 降级路径）：已建风场方法验证；G_plan 基线 = 建成朝向 E(0°)；
不提前写全球 TWh 总量。风格沿用 nc_style，图内英文。
输入：wp7b_scenario_table.csv / wp7b_corridor_summary.csv / wp5_rfd_grid.csv
"""
import os, io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BUSH), 'output')
REPO = os.path.dirname(os.path.dirname(BUSH))
sys.path.insert(0, os.path.join(REPO, '分析材料_几何主导杠杆', 'figures'))
from nc_style import PAL, apply_style, panel_label

tb = pd.read_csv(os.path.join(OUT, 'wp7b_scenario_table.csv'), encoding='utf-8-sig')
sumdf = pd.read_csv(os.path.join(OUT, 'wp7b_corridor_summary.csv'), encoding='utf-8-sig')
rfd = pd.read_csv(os.path.join(OUT, 'wp5_rfd_grid.csv'), encoding='utf-8-sig')
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')

CORR_COL = {'Vietnam': '#DC2626', 'China_strait': '#F59E0B',
            'Italy': '#8B5CF6', 'Denmark': '#059669', 'other': '#94A3B8'}
S_COL = {'S1': PAL['baseline'], 'S2': PAL['accent'], 'S3': PAL['highlight']}
V_COL = {'V1': PAL['baseline'], 'V2': PAL['accent'], 'V3': PAL['highlight']}

main = tb[tb.layout_type != 'sparse'].copy()

apply_style()
fig = plt.figure(figsize=(11.5, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.36, wspace=0.3)

# ─────────────────────────── a. 容量地图 ───────────────────────────
axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
axa.set_extent([-100, 160, 0, 70], crs=ccrs.PlateCarree())
import cartopy.feature as cfeature
axa.add_feature(cfeature.OCEAN, fc='white', ec='none', zorder=0)
axa.add_feature(cfeature.LAND, fc='#EEF2F6', ec='none', zorder=0)
axa.coastlines(lw=0.3, color='#475569')
# 背景 R 走廊（浅色）
v = np.full(len(rfd), np.nan)
zg = np.load(os.path.join(OUT, 'wp5_cross_grid.npz'))
v[zg['valid']] = rfd['R'].values[zg['valid']]
axa.scatter(rfd['lon'], rfd['lat'], c=v, s=7, cmap='Greys', alpha=0.5,
            vmin=np.nanpercentile(v, 5), vmax=np.nanpercentile(v, 98),
            transform=ccrs.PlateCarree(), zorder=2, linewidths=0)
for cname, g in main.groupby('corridor'):
    lons = [geo.loc[f, 'cent_lon'] for f in g.farm_id if f in geo.index]
    lats = [geo.loc[f, 'cent_lat'] for f in g.farm_id if f in geo.index]
    cap = g.capacity_MW.values
    axa.scatter(lons, lats, s=np.clip(cap / 3.5, 12, 220), color=CORR_COL[cname],
                alpha=0.85, edgecolors='white', linewidths=0.4, zorder=5,
                label=f'{cname} (n={len(g)})', transform=ccrs.PlateCarree())
axa.legend(loc='lower left', fontsize=6.5, markerscale=0.7, frameon=False)
panel_label(axa, 'a')

# ─────────────────────────── b. S1-S3 百分比增益 ───────────────────────────
axb = fig.add_subplot(gs[0, 1])
scen = ['S1', 'S2', 'S3']
xpos = np.arange(3)
for j, s in enumerate(scen):
    g = main[f'G_plan_{s}'].values
    g = g[np.isfinite(g)]
    med, p05, p95 = np.median(g), np.percentile(g, 5), np.percentile(g, 95)
    axb.plot([xpos[j] - 0.18, xpos[j] + 0.18], [med, med], color=S_COL[s], lw=2.2)
    axb.plot([xpos[j]] * 2, [p05, p95], color=S_COL[s], lw=1.2, alpha=0.9)
    axb.plot([xpos[j] - 0.06, xpos[j] + 0.06], [p05, p05], color=S_COL[s], lw=1.2)
    axb.plot([xpos[j] - 0.06, xpos[j] + 0.06], [p95, p95], color=S_COL[s], lw=1.2)
    axb.text(xpos[j], med, f'  {med:.1f}%', va='center', fontsize=8, color=S_COL[s])
axb.set_xticks(xpos)
axb.set_xticklabels(['S1\norientation fix', 'S2\ntype-preserving\nre-layout',
                     'S3\nfull re-design'])
axb.set_ylabel('G_plan gain vs built orientation (%)')
axb.set_title(f'Median & 5-95% range ({len(main)} non-sparse projects)', fontsize=8)
panel_label(axb, 'b')

# ─────────────────────────── c. 走廊 GWh 贡献 ───────────────────────────
axc = fig.add_subplot(gs[1, 0])
sumd = sumdf[sumdf.corridor != 'other'].sort_values('capacity_GW', ascending=False)
x = np.arange(len(sumd))
w = 0.26
for j, s in enumerate(scen):
    axc.bar(x + (j - 1) * w, sumd[f'dE_{s}_GWh'], width=w, color=S_COL[s], label=s)
axc.set_xticks(x)
axc.set_xticklabels([f'{r.corridor}\n({r.capacity_GW:.1f} GW)' for r in sumd.itertuples()], fontsize=7)
axc.set_ylabel('Added generation (GWh yr$^{-1}$)')
axc.legend(fontsize=7, loc='upper right')
panel_label(axc, 'c')

# ─────────────────────────── d. V1/V2/V3 增量价值分解 ───────────────────────────
axd = fig.add_subplot(gs[1, 1])
sumd2 = sumdf[sumdf.corridor != 'other'].sort_values('capacity_GW', ascending=False)
x = np.arange(len(sumd2) + 1)   # 末位为 all
allrow = pd.DataFrame([dict(
    corridor='all', V1_share=main.V1.sum() / main.V1.abs().sum(),
    V2_share=main.V2.sum() / main.V1.abs().sum(),
    V3_share=main.V3.sum() / main.V1.abs().sum())])
plotd = pd.concat([sumd2[['corridor', 'V1_share', 'V2_share', 'V3_share']], allrow], ignore_index=True)
w = 0.22   # 分组条形（V2 为负时不叠加）
for j, s in enumerate(['V1', 'V2', 'V3']):
    axd.bar(x + (j - 1) * w, plotd[f'{s}_share'], width=w,
            color=V_COL[s], label=f'{s} ' + {'V1': 'orientation fix',
                                             'V2': 'type-preserving re-layout',
                                             'V3': 'full re-design'}[s])
for xi, (_, r) in zip(x, plotd.iterrows()):
    axd.text(xi, r.V1_share + 0.03, f'{r.V1_share:.2f}', ha='center', fontsize=6, color=V_COL['V1'])
axd.set_xticks(x)
axd.set_xticklabels(list(plotd.corridor), fontsize=7)
axd.axhline(0, color=PAL['light'], lw=0.8)
axd.set_ylabel('Increment value V1/V2/V3\n(share of V1 magnitude; V2 may be negative)')
axd.legend(fontsize=7, loc='upper left')
panel_label(axd, 'd')

fig.savefig(os.path.join(BUSH, 'Fig4_conclusion4.png'))
print('输出: Fig4_conclusion4.png')
print(f'非 sparse 项目: {len(main)} | 覆盖容量 {main.capacity_MW.sum()/1000:.1f} GW')
print(f'G_plan 中位: S1 {main.G_plan_S1.median():.2f}% | S2 {main.G_plan_S2.median():.2f}% | '
      f'S3 {main.G_plan_S3.median():.2f}%')
print(f'ΔE 合计: S1 {main.dE_GWh_S1.sum():.1f} | S2 {main.dE_GWh_S2.sum():.1f} | '
      f'S3 {main.dE_GWh_S3.sum():.1f} GWh/yr')
