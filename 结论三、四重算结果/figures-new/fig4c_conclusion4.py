"""
图 4 结论四（建设范式版，2026-08-17 六面板重绘）：三级建设自由度 + 六情境大小对比
============================================================================
a. 候选走廊与纳入核算项目容量地图（点大小 ∝ 项目容量，背景为 R 稳健走廊）
b. S1/S2/S3 百分比增益及 5-95% 区间（G_plan，n≥10 项目）
c. 各走廊新增 GWh 贡献（dE_GWh_S1/S2/S3 分组柱，S2 可为负）
d. 六套建设范式情境各自全圆最优朝向的中位增益与 5-95%（正蓝负橙，零线）
e. 最小间距–增益单调性：3.33D→−11.7%、4D→−5.6%、5D→+1.0%、9.4D→+9.6%、11.8D→+10.8%
f. V1/V2/V3 增量价值分解（朝向校正/范式保持重排/规划前置重构 的份额）

子刊体例：Arial、Okabe-Ito 走廊色、S1⊆S2⊆S3 用 ColorBrewer Blues 顺序色、
灰陆地+经纬网格地图、细轴线浅网格、面板字母左上外侧。
口径（方案 §5.6 降级路径）：已建风场方法验证；G_plan 基线 = 建成朝向 E(0°)；
S2 = 场址主范式情境（task1 多标签取主标签）；S3 = 6 范式情境全集（全局最优 100% 落在 S_E）。
输入：wp7c_scenario_table.csv / wp7c_corridor_summary.csv / wp5c_rfd_grid.csv /
     wp7d_scenario_attribution.csv
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
import cartopy.feature as cfeature

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BUSH), 'output')
REPO = os.path.dirname(os.path.dirname(BUSH))
sys.path.insert(0, os.path.join(REPO, '分析材料_几何主导杠杆', 'figures'))
from nc_style import PAL, CORRIDOR_COL, SCENARIO_COL, apply_style, panel_label

tb = pd.read_csv(os.path.join(OUT, 'wp7c_scenario_table.csv'), encoding='utf-8-sig')
sumdf = pd.read_csv(os.path.join(OUT, 'wp7c_corridor_summary.csv'), encoding='utf-8-sig')
rfd = pd.read_csv(os.path.join(OUT, 'wp5c_rfd_grid.csv'), encoding='utf-8-sig')
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
w7d = pd.read_csv(os.path.join(OUT, 'wp7d_scenario_attribution.csv'), encoding='utf-8-sig')

CORR_COL = {k: CORRIDOR_COL[k] for k in
            ['Vietnam', 'China_strait', 'Italy', 'Denmark', 'other']}
S_COL = {s: SCENARIO_COL[s] for s in ['S1', 'S2', 'S3']}
V_COL = {v: SCENARIO_COL[v] for v in ['V1', 'V2', 'V3']}

main = tb[tb.n_turb >= 10].copy()

apply_style()
fig = plt.figure(figsize=(13.8, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.02, 1.0], hspace=0.46, wspace=0.34)

# ─────────────────────────── a. 容量地图 ───────────────────────────
axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
axa.set_extent([-100, 160, 0, 70], crs=ccrs.PlateCarree())
try:
    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '50m', facecolor='#E8E8E8', edgecolor='none')
except Exception:
    land = cfeature.LAND
axa.add_feature(land, zorder=0)
axa.add_feature(cfeature.OCEAN, fc='#FFFFFF', ec='none', zorder=0)
axa.coastlines(lw=0.5, color='#666666')
gl = axa.gridlines(draw_labels=True, lw=0.3, color='#B9B9B9', alpha=0.8,
                   xlocs=range(-180, 181, 30), ylocs=range(0, 91, 30))
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 6, 'color': '#666666'}
gl.ylabel_style = {'size': 6, 'color': '#666666'}
v = np.full(len(rfd), np.nan)
zg = np.load(os.path.join(OUT, 'wp5c_cross_grid.npz'))
v[zg['valid']] = rfd['R'].values[zg['valid']]
axa.scatter(rfd['lon'], rfd['lat'], c=v, s=6, cmap='Greys', alpha=0.5,
            vmin=np.nanpercentile(v, 5), vmax=np.nanpercentile(v, 98),
            transform=ccrs.PlateCarree(), zorder=2, linewidths=0)
for cname, g in main.groupby('corridor'):
    lons = [geo.loc[f, 'cent_lon'] for f in g.farm_id if f in geo.index]
    lats = [geo.loc[f, 'cent_lat'] for f in g.farm_id if f in geo.index]
    cap = g.capacity_MW.values
    axa.scatter(lons, lats, s=np.clip(cap / 3.0, 14, 260), color=CORR_COL[cname],
                alpha=0.85, edgecolors='white', linewidths=0.6, zorder=5,
                label=f'{cname} (n={len(g)})', transform=ccrs.PlateCarree())
axa.legend(loc='lower left', fontsize=6.5, markerscale=0.7, frameon=False,
           handletextpad=0.4, borderaxespad=0.4)
panel_label(axa, 'a')

# ─────────────────────────── b. S1-S3 百分比增益 ───────────────────────────
axb = fig.add_subplot(gs[0, 1])
scen = ['S1', 'S2', 'S3']
xpos = np.arange(3)
for j, s in enumerate(scen):
    g = main[f'G_plan_{s}'].values
    g = g[np.isfinite(g)]
    med, p05, p95 = np.median(g), np.percentile(g, 5), np.percentile(g, 95)
    axb.plot([xpos[j] - 0.18, xpos[j] + 0.18], [med, med], color=S_COL[s], lw=2.6,
             solid_capstyle='butt', zorder=4)
    axb.plot([xpos[j]] * 2, [p05, p95], color=S_COL[s], lw=1.0, alpha=0.9,
             zorder=3)
    axb.plot([xpos[j] - 0.07, xpos[j] + 0.07], [p05, p05], color=S_COL[s], lw=1.0)
    axb.plot([xpos[j] - 0.07, xpos[j] + 0.07], [p95, p95], color=S_COL[s], lw=1.0)
    axb.text(xpos[j], med, f' +{med:.1f}%', va='center', ha='left', fontsize=7.5,
             color=S_COL[s], fontweight='bold')
axb.set_xticks(xpos)
axb.set_xticklabels(['S1\norientation fix', 'S2\nparadigm-preserving\nre-layout',
                     'S3\nany-paradigm\nre-design'])
axb.set_ylabel('G$_{plan}$ gain vs built orientation (%)')
axb.set_title(f'Median & 5-95% range ({len(main)} projects, n≥10)', fontsize=8)
axb.set_xlim(-0.5, 2.5)
panel_label(axb, 'b')

# ─────────────────────────── c. 走廊 GWh 贡献 ───────────────────────────
axc = fig.add_subplot(gs[0, 2])
sumd = sumdf[sumdf.corridor != 'other'].sort_values('capacity_GW', ascending=False)
x = np.arange(len(sumd))
w = 0.26
for j, s in enumerate(scen):
    axc.bar(x + (j - 1) * w, sumd[f'dE_{s}_GWh'], width=w, color=S_COL[s],
            label=s, edgecolor='white', linewidth=0.3)
axc.axhline(0, color=PAL['ink'], lw=0.5)
axc.set_xticks(x)
axc.set_xticklabels([f'{r.corridor}\n({r.capacity_GW:.1f} GW)' for r in sumd.itertuples()],
                    fontsize=6.8)
axc.set_ylabel('Added generation (GWh yr$^{-1}$)')
axc.legend(fontsize=6.8, loc='upper right', ncol=3, frameon=False,
           columnspacing=0.8, handlelength=1.2)
axc.margins(x=0.02)
panel_label(axc, 'c')

# ─────────────────────────── d. 六套范式情境大小对比 ───────────────────────────
axd = fig.add_subplot(gs[1, 0])
LBL = {'S_A': 'S_A\n9.4D\naligned', 'S_B0': 'S_B0\n3.3D\ndense',
       'S_B45': 'S_B45\n3.3D\n45°', 'S_C': 'S_C\n5D',
       'S_D': 'S_D\n4D', 'S_E': 'S_E\n11.8D\nspacious'}
w7d = w7d.set_index('scenario')
x = np.arange(len(w7d))
for xi, (pid, r) in zip(x, w7d.iterrows()):
    col = PAL['baseline'] if r.med >= 0 else PAL['highlight']
    axd.bar(xi, r.med, width=0.58, color=col, alpha=0.85, edgecolor='white', lw=0.3)
    axd.plot([xi, xi], [r.p05, r.p95], color=col, lw=1.1, zorder=3)
    axd.plot([xi - 0.07, xi + 0.07], [r.p05, r.p05], color=col, lw=1.1)
    axd.plot([xi - 0.07, xi + 0.07], [r.p95, r.p95], color=col, lw=1.1)
    dy = 1.2 if r.med >= 0 else -1.2
    axd.text(xi, r.med + dy, f'{r.med:+.1f}%', ha='center', va='bottom' if r.med >= 0 else 'top',
             fontsize=7, fontweight='bold', color=col)
axd.axhline(0, color=PAL['ink'], lw=0.6)
axd.set_xticks(x)
axd.set_xticklabels([LBL[p] for p in w7d.index], fontsize=6.5)
axd.set_ylabel('Median G at scenario best\norientation vs built (%)')
axd.set_ylim(-26, 32)
axd.text(0.985, 0.94, 'blue: gain / orange: loss', transform=axd.transAxes,
         fontsize=6.5, ha='right', va='top', color=PAL['ink'])
panel_label(axd, 'd')

# ─────────────────────────── e. 间距–增益单调性 ───────────────────────────
axe = fig.add_subplot(gs[1, 1])
pts = w7d.loc[['S_B0', 'S_D', 'S_C', 'S_A', 'S_E']]
xs = pts.spacing_D.values
ys = pts.med.values
cols = [PAL['baseline'] if y >= 0 else PAL['highlight'] for y in ys]
axe.plot(xs, ys, '-', color=PAL['light'], lw=1.2, zorder=2)
axe.scatter(xs, ys, s=44, c=cols, edgecolors='white', lw=0.8, zorder=4)
off = {'S_B0': (0.0, 1.6), 'S_D': (0.0, -2.4), 'S_C': (0.0, 1.6),
       'S_A': (-0.35, 1.6), 'S_E': (-0.6, -2.6)}
for pid, r in pts.iterrows():
    axe.annotate(f'{pid}\n{r.spacing_D:.1f}D', (r.spacing_D, r.med),
                 xytext=(r.spacing_D + off[pid][0], r.med + off[pid][1]),
                 fontsize=6.5, ha='center', va='bottom' if off[pid][1] > 0 else 'top',
                 color=PAL['ink'], zorder=5)
axe.axhline(0, color=PAL['ink'], lw=0.6)
axe.set_xlabel('Minimum turbine spacing (D)')
axe.set_ylabel('Median G at best orientation (%)')
axe.set_xlim(2.6, 12.6)
axe.set_ylim(-26, 20)
panel_label(axe, 'e')

# ─────────────────────────── f. V1/V2/V3 增量价值分解 ───────────────────────────
axf = fig.add_subplot(gs[1, 2])
sumd2 = sumdf[sumdf.corridor != 'other'].sort_values('capacity_GW', ascending=False)
x = np.arange(len(sumd2) + 1)
allrow = pd.DataFrame([dict(
    corridor='all', V1_share=main.V1.sum() / main.V1.abs().sum(),
    V2_share=main.V2.sum() / main.V1.abs().sum(),
    V3_share=main.V3.sum() / main.V1.abs().sum())])
plotd = pd.concat([sumd2[['corridor', 'V1_share', 'V2_share', 'V3_share']], allrow],
                  ignore_index=True)
w = 0.22
for j, s in enumerate(['V1', 'V2', 'V3']):
    axf.bar(x + (j - 1) * w, plotd[f'{s}_share'], width=w,
            color=V_COL[s], edgecolor='white', linewidth=0.3,
            label=f'{s} ' + {'V1': 'orientation fix',
                             'V2': 'paradigm-preserving re-layout',
                             'V3': 'any-paradigm re-design'}[s])
for xi, (_, r) in zip(x, plotd.iterrows()):
    axf.text(xi, r.V1_share + 0.04, f'{r.V1_share:.2f}', ha='center', fontsize=6,
             color=V_COL['V1'])
axf.set_xticks(x)
axf.set_xticklabels(list(plotd.corridor), fontsize=6.8)
axf.axhline(0, color=PAL['ink'], lw=0.5)
axf.set_ylabel('Increment value V1/V2/V3\n(share of V1 magnitude; V2 may be negative)')
axf.legend(fontsize=6.8, loc='upper left', ncol=3, frameon=False,
           columnspacing=0.8, handlelength=1.2)
panel_label(axf, 'f')

fig.savefig(os.path.join(BUSH, 'Fig4c_conclusion4_paradigms.png'), dpi=450)
print('输出: Fig4c_conclusion4_paradigms.png')
print(f'项目: {len(main)}（n≥10）| 覆盖容量 {main.capacity_MW.sum()/1000:.1f} GW')
print(f'G_plan 中位: S1 {main.G_plan_S1.median():.2f}% | S2 {main.G_plan_S2.median():.2f}% | '
      f'S3 {main.G_plan_S3.median():.2f}%')
print(f'ΔE 合计: S1 {main.dE_GWh_S1.sum():.1f} | S2 {main.dE_GWh_S2.sum():.1f} | '
      f'S3 {main.dE_GWh_S3.sum():.1f} GWh/yr')
print('六情境: ' + ' | '.join(f'{p} {r.med:+.1f}%' for p, r in w7d.iterrows()))
