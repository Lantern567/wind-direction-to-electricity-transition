# -*- coding: utf-8 -*-
"""图 4 重绘 v9（2026-08-27，用户反馈：a 比例尺挡图例 / b 图拉大与 a 一样大）
====================================================================
v9 改动（用户 2026-08-27 第四轮）：
  1. (a) 比例尺从右下角 (137.5, 8.5) 移到左下角泰国湾海区 (106.3, 6.0)，
     不再被右下角图例遮挡（图例保持 lower-right 不动）。
  2. (b) 地图内容拉大：范围由 [-6,23,45,63] 收紧为 [-6,23,46,62.5]
     （两侧留 2° 以上边距保证 69 个欧洲项目点完整可见），欧洲内容
     放大约 8%；a/b 两幅地图面板尺寸本就完全一致（均为 0.28×0.425）。
v8 改动（用户 2026-08-27 第三轮）：
  1. (b) 删除美国东海岸镶边小地图（仍显突兀），改为 (b) 图内右下角注记
     'US East (n = 3) not shown'，如实交代 155 场中 3 场未显示。
  2. e/f/c 纵坐标标题改短（e 'Median difference at best orientation (%)'、
     f 'Median difference (%)'、c 'Δ Generation (GWh yr⁻¹)'），三列面板
     收窄至 0.28、列间距加大到 0.045，纵坐标文字与前一面板之间留足空白。
  3. (a)(b) 两幅地图面板尺寸完全一致（均为 0.28×0.425），删除小图后视觉对称。
  4. (f) 两个 3.33D 范式标签偏移 ±0.62→±0.72（列变窄后避免标签互碰）。
v7 改动：一行三图（三列两行）；面板字母全部移至面板上方空白带；
  (d) 换半小提琴表达。
v6 改动：ΔE 柱叠加逐场抖动点 + 总量注记；f 点标签两行 + 建成场间距注记；
  e 加 n=155 注记。
v4/v5 改动（学长反馈 2026-08-25）：(a)/(b) 欧亚大图拆为两幅独立地图；
  (c) 合并原 b+f；(f) 间距散点去折线。
体例 = NC 参考图：白底黑框面板、面板字母加粗黑、数值彩色加粗+白描边、
      地图青蓝海+白陆+深灰海岸线、黑框图例、N 箭头与比例尺。
数据：wp7c_scenario_table.csv / wp7d_scenario_attribution.csv / wp1_geometry_frozen.csv
统计数字全部实算并与冻结报告对账。
输出：figures-nat/Fig4_nat_v9.png / .pdf
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), '补算', 'output')
sys.path.insert(0, HERE)
from nc_style_nat import (apply_style, panel_label, halo, save_fig, RED,
                          ORANGE, DEEP_BLUE, MID_BLUE, LIGHT_BLUE, GREY, INK,
                          BOX_EC, SCOL, CORRIDOR_COL, PARADIGM_ORDER,
                          PARADIGM_NAME, LAND_COL, SEA_COL, COAST_COL,
                          COAST_LW, north_arrow, scale_bar)

tb = pd.read_csv(os.path.join(OUT, 'wp7c_scenario_table.csv'), encoding='utf-8-sig')
w7d = pd.read_csv(os.path.join(OUT, 'wp7d_scenario_attribution.csv'),
                  encoding='utf-8-sig')
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'),
                  encoding='utf-8-sig').set_index('farm_id')

main = tb[tb.n_turb >= 10].copy()          # 155 项目
corr20 = main[main.corridor != 'other']    # 20 项目（台湾海峡 8 + 越南 12）
oth135 = main[main.corridor == 'other']

# 建成场有效间距（v6 面板 f 上下文注记用，wp9c 实算）
m9c = pd.read_csv(os.path.join(OUT, 'wp9c_farm_metrics.csv')).set_index('farm_id')
S_built = m9c.loc[main.farm_id, 'spacing_D_med'].values.astype(float)
S_built = S_built[~np.isnan(S_built)]
medS, p5S, p95S = (float(np.median(S_built)), float(np.percentile(S_built, 5)),
                   float(np.percentile(S_built, 95)))
print('建成场间距: 中位 %.2f D, 5–95%%: %.2f–%.2f D (n=%d)' %
      (medS, p5S, p95S, len(S_built)))

# ── 对账断言（与 wp7c/wp9d 冻结数字；G_plan 单位 = %）──
v_s = [main[f'G_plan_{s}'].median() for s in ('S1', 'S2', 'S3')]
assert abs(v_s[0] - 0.47) < 0.05 and abs(v_s[1] - 1.60) < 0.10 \
    and abs(v_s[2] - 10.79) < 0.20, v_s
m_corr = [corr20[f'G_plan_{s}'].median() for s in ('S1', 'S2', 'S3')]
m_oth = [oth135[f'G_plan_{s}'].median() for s in ('S1', 'S2', 'S3')]
assert abs(m_corr[0] - 2.77) < 0.30 and abs(m_corr[1] + 1.36) < 0.30 \
    and abs(m_corr[2] - 10.37) < 0.30, m_corr
assert abs(m_oth[0] - 0.40) < 0.10 and abs(m_oth[1] - 2.28) < 0.30, m_oth
dE_med = [main[f'dE_GWh_{s}'].sum() for s in ('S1', 'S2', 'S3')]
assert abs(dE_med[0] - 3474) < 60 and abs(dE_med[1] - 15889) < 250 \
    and abs(dE_med[2] - 75966) < 600, dE_med
print('checks: project medians =', [round(x, 2) for x in v_s],
      '| dE =', [round(x) for x in dE_med])

# ── 项目经纬度与地图分组 ──
lon = np.array([geo.loc[f, 'cent_lon'] for f in main.farm_id])
lat = np.array([geo.loc[f, 'cent_lat'] for f in main.farm_id])
cap = main.capacity_MW.values / 1000.0      # GW
us_sel = lon < -40
ea_sel = (lon >= 40)
eu_sel = (~us_sel) & (~ea_sel)
print(f'地图分组: 东亚 {int(ea_sel.sum())} | 欧洲 {int(eu_sel.sum())} | '
      f'美国东海岸 {int(us_sel.sum())} | 合计 {int(len(main))}')
print(f'US 项目: lon {lon[us_sel].min():.1f}..{lon[us_sel].max():.1f}, '
      f'lat {lat[us_sel].min():.1f}..{lat[us_sel].max():.1f}')
print(f'EU 项目: lon {lon[eu_sel].min():.1f}..{lon[eu_sel].max():.1f}, '
      f'lat {lat[eu_sel].min():.1f}..{lat[eu_sel].max():.1f}')
print(f'EA 项目: lon {lon[ea_sel].min():.1f}..{lon[ea_sel].max():.1f}, '
      f'lat {lat[ea_sel].min():.1f}..{lat[ea_sel].max():.1f}')

apply_style()
fig = plt.figure(figsize=(12.9, 9.9))

MAP_KW = dict(facecolor='none', edgecolor='none')
def base_map(ax, extent, xlocs, ylocs, labels=True):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
        facecolor=SEA_COL, edgecolor='none'), zorder=-1)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
        facecolor=LAND_COL, edgecolor=COAST_COL, linewidth=COAST_LW), zorder=0)
    ax.add_feature(cfeature.BORDERS, edgecolor='#c8c8c8', linewidth=0.3, zorder=1)
    gl = ax.gridlines(linewidth=0.3, color='#cdd5da', linestyle='--',
                      draw_labels=labels, dms=False,
                      xlocs=xlocs, ylocs=ylocs)
    gl.top_labels = False; gl.right_labels = False
    ax.tick_params(labelsize=6.5)

def scatter_group(ax, sel, col, s=30.25):
    ax.scatter(lon[sel], lat[sel], s=s * cap[sel], c=col, alpha=0.9,
               edgecolors='white', linewidths=0.4, zorder=4,
               transform=ccrs.PlateCarree())

# ============ a. 东亚项目地图（v7：上排左；v8 与 b 同尺寸 0.28×0.425） ============
axa = fig.add_axes([0.045, 0.52, 0.28, 0.425], projection=ccrs.PlateCarree())
base_map(axa, [104, 142, 4, 46], np.arange(105, 145, 10), np.arange(10, 50, 10))
for g, col in (('China_strait', CORRIDOR_COL['China_strait']),
               ('Vietnam', CORRIDOR_COL['Vietnam'])):
    sel = ea_sel & (main.corridor == g).values
    scatter_group(axa, sel, col)
scatter_group(axa, ea_sel & (main.corridor == 'other').values, '#b8c2cc')
axa.legend([plt.Line2D([], [], marker='o', color='none', mfc=CORRIDOR_COL['China_strait'],
                       mec='white', mew=0.4, ms=6, ls=''),
            plt.Line2D([], [], marker='o', color='none', mfc=CORRIDOR_COL['Vietnam'],
                       mec='white', mew=0.4, ms=6, ls=''),
            plt.Line2D([], [], marker='o', color='none', mfc='#b8c2cc',
                       mec='white', mew=0.4, ms=6, ls='')],
           [f'Taiwan Strait (n = {int((ea_sel & (main.corridor == "China_strait").values).sum())})',
            f'Vietnam (n = {int((ea_sel & (main.corridor == "Vietnam").values).sum())})',
            f'Other (n = {int((ea_sel & (main.corridor == "other").values).sum())})'],
           loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=6.3,
           framealpha=0.95, handlelength=0.9, borderpad=0.4)
for gw, xl, yl in ((1, 106.5, 11.5), (5, 106.5, 16.5)):
    axa.plot([xl], [yl], 'o', ms=np.sqrt(30.25 * gw), mfc='none',
             mec=INK, mew=0.8, transform=ccrs.PlateCarree(), zorder=6)
axa.text(139.0, 40.5, 'Circle area scales with\ncapacity (1, 5 GW)',
         fontsize=6.0, ha='right', va='bottom', color='#555555',
         transform=ccrs.PlateCarree())
axa.text(0.015, 0.955, 'East Asia', transform=axa.transAxes, fontsize=7.5,
         va='top', color=INK,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.3))
north_arrow(axa, x=0.955, y0=0.93, y1=0.80, fs=8)
scale_bar(axa, 106.3, 6.0, km=300, fs=6.2)   # v9：移至左下角泰国湾，避开右下角图例
panel_label(axa, 'a', dy=1.06)

# ============ b. 欧洲项目地图（v8：上排中；US 小图已删除，改右下角注记） ============
axb = fig.add_axes([0.37, 0.52, 0.28, 0.425], projection=ccrs.PlateCarree())
base_map(axb, [-6, 23, 46, 62.5], [-5, 5, 15], [50, 55, 60])  # v9：收紧范围，欧洲内容放大
scatter_group(axb, eu_sel, '#b8c2cc')
for gw, xl, yl in ((1, -5.0, 46.2), (5, -5.0, 48.0)):
    axb.plot([xl], [yl], 'o', ms=np.sqrt(30.25 * gw), mfc='none',
             mec=INK, mew=0.8, transform=ccrs.PlateCarree(), zorder=6)
axb.text(1.0, 54.8, 'Capacity (GW): 1, 5', fontsize=6.0, ha='left',
         va='bottom', color='#555555', transform=ccrs.PlateCarree())
axb.text(0.015, 0.955, f'Europe (n = {int(eu_sel.sum())})',
         transform=axb.transAxes, fontsize=7.5, va='top', color=INK,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.3))
north_arrow(axb, x=0.955, y0=0.90, y1=0.66, fs=8)
scale_bar(axb, 1.0, 56.5, km=300, fs=6.2)
panel_label(axb, 'b', dy=1.06)

# 美国东海岸 3 项目（v8：小地图删除，改为图内右下角如实注记，不隐去事实）
axb.text(0.985, 0.03, 'US East (n = 3) not shown', transform=axb.transAxes,
         ha='right', va='bottom', fontsize=5.6, color='#666666',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5),
         zorder=40)

# ============ d. S1/S2/S3 × 走廊/其他 半小提琴（v7：换表达方式，区别于 f 的散点+须线） ============
axc = fig.add_axes([0.045, 0.065, 0.28, 0.395])
from scipy.stats import gaussian_kde
GROUPS_C = [('corr', corr20, RED, 'Corridor (n = 20)'),
            ('oth', oth135, DEEP_BLUE, 'Other (n = 135)')]
XOFF = {'corr': -0.16, 'oth': +0.16}
bots, tops = [], []
for k, s in enumerate(('S1', 'S2', 'S3')):
    for key, dfg, col, _ in GROUPS_C:
        g = dfg[f'G_plan_{s}'].values
        q005, q995 = np.percentile(g, [0.5, 99.5])
        pad = (q995 - q005) * 0.12
        gg = np.linspace(q005 - pad, q995 + pad, 200)
        dd = gaussian_kde(g)(gg)
        dd = dd / dd.max() * 0.30
        x0 = k + XOFF[key]
        side = 1 if key == 'oth' else -1
        axc.fill_betweenx(gg, x0, x0 + side * dd, color=col, alpha=0.85,
                          lw=0, zorder=3)
        axc.plot([x0, x0], [gg[0], gg[-1]], color=col, lw=0.7, zorder=4)
        axc.plot(x0, np.median(g), 'o', ms=2.6, mfc='white', mec=col,
                 mew=0.7, zorder=5)
        p05, p95 = np.percentile(g, [5, 95])
        axc.plot([x0, x0], [p05, p95], color=INK, lw=0.9, zorder=6)
        axc.plot([x0 - 0.04, x0 + 0.04], [p05, p05], color=INK, lw=0.9,
                 zorder=6)
        axc.plot([x0 - 0.04, x0 + 0.04], [p95, p95], color=INK, lw=0.9,
                 zorder=6)
        bots.append(q005 - pad); tops.append(q995 + pad)
        if key == 'corr':
            medv = np.median(g)
            halo(axc, x0, medv + (2.0 if medv >= 0 else -2.0),
                 f'{medv:+.1f}', color=col, fs=6.2, ha='center',
                 va='bottom' if medv >= 0 else 'top')
    med_all = np.median(main[f'G_plan_{s}'].values)
    axc.scatter(k, med_all, marker='D', s=24, c=GREY, zorder=7,
                edgecolors=INK, linewidths=0.6)
    halo(axc, k, med_all + 2.2, f'{med_all:+.1f}', color=INK, fs=6.4,
         va='bottom')
axc.axhline(0, color=INK, lw=0.8)
import matplotlib.patches as mpatches
axc.legend([mpatches.Patch(facecolor=RED, alpha=0.85, edgecolor='none'),
            mpatches.Patch(facecolor=DEEP_BLUE, alpha=0.85, edgecolor='none'),
            plt.Line2D([], [], marker='D', ls='', mfc=GREY, mec=INK, mew=0.6,
                       ms=5)],
           ['Corridor (n = 20)', 'Other (n = 135)', 'All (n = 155) median'],
           loc='upper left', fontsize=6.3, handlelength=1.2, borderpad=0.4)
axc.set_xticks(range(3)); axc.set_xticklabels(['S1', 'S2', 'S3'], fontsize=8)
axc.set_ylabel('Generation difference vs built layout (%)')
axc.set_xlim(-0.55, 2.55)
axc.set_ylim(min(0.0, min(bots)) - 2, max(tops) + 4)
axc.grid(axis='x', visible=False)
axc.text(0.5, -0.125, 'S1 orientation only · S2 matched-template re-layout · '
         'S3 best of six templates', transform=axc.transAxes, fontsize=5.2,
         ha='center', va='top', color='#555555')
axc.text(0.02, 0.025, 'Half-violins: kernel density of farm-level differences · '
         'white dot = median · whiskers 5–95%', transform=axc.transAxes,
         fontsize=5.6, color='#666666')
panel_label(axc, 'd', dy=1.06)

# ============ c. 走廊 ΔE 分组柱（v7：上排右） ============
axd = fig.add_axes([0.695, 0.52, 0.28, 0.425])
corr_g = corr20.groupby('corridor')
dE = {g: [corr_g.get_group(g)[f'dE_GWh_{s}'].sum() for s in ('S1', 'S2', 'S3')]
      for g in ('China_strait', 'Vietnam')}
labels = [f'Taiwan Strait\n({corr_g.get_group("China_strait").capacity_MW.sum() / 1000:.1f} GW)',
          f'Vietnam\n({corr_g.get_group("Vietnam").capacity_MW.sum() / 1000:.1f} GW)']
x = np.arange(2); W = 0.27
vals_all = []
for k, s in enumerate(('S1', 'S2', 'S3')):
    vals = [dE['China_strait'][k], dE['Vietnam'][k]]
    vals_all.append(vals)
    axd.bar(x + (k - 1) * W, vals, width=W * 0.92, color=SCOL[s],
            label=s, linewidth=0, zorder=2)
    for xi, v in zip(x + (k - 1) * W, vals):
        axd.text(xi, v + (350 if v > 0 else -350), f'{v:,.0f}', ha='center',
                 fontsize=6.3, va='bottom' if v > 0 else 'top', color=INK)
# v6：逐场 ΔE 抖动点（走廊 20 场逐场值，场景同色半透明，垫于柱面之上）
rng4 = np.random.default_rng(3)
for k, s in enumerate(('S1', 'S2', 'S3')):
    for gi, g in enumerate(('China_strait', 'Vietnam')):
        vals = corr_g.get_group(g)[f'dE_GWh_{s}'].values
        axd.scatter(x[gi] + (k - 1) * W + rng4.uniform(-0.11, 0.11, len(vals)),
                    vals, s=8, color=SCOL[s], alpha=0.45, lw=0, zorder=3)
axd.text(0.985, 0.96, 'Sample total: S1 3,474 · S2 15,889 · S3 75,966 GWh yr$^{-1}$',
         transform=axd.transAxes, ha='right', va='top', fontsize=5.8,
         color='#555555')
axd.axhline(0, color=INK, lw=0.8)
axd.set_xticks(x); axd.set_xticklabels(labels, fontsize=6.5)
axd.set_ylabel('Δ Generation (GWh yr$^{-1}$)')
axd.set_xlim(-0.55, 1.55)
vm = min(min(v) for v in vals_all); vx = max(max(v) for v in vals_all)
span = max(vx - vm, 1)
axd.set_ylim(vm - 0.16 * span, vx + 0.16 * span)
axd.grid(axis='x', visible=False)
axd.legend(loc='upper left', fontsize=6.5, ncol=3, handlelength=0.9,
           columnspacing=0.8)
panel_label(axd, 'c', dy=1.06)

# ============ e. 六范式中位 + 5–95%（v7：下排中） ============
axe = fig.add_axes([0.37, 0.065, 0.28, 0.395])
med = w7d.set_index('scenario').loc[PARADIGM_ORDER]
xs = np.arange(6)
axe.bar(xs, med['med'].values, width=0.52,
        color=[MID_BLUE if v > 0 else RED for v in med['med'].values],
        linewidth=0, zorder=2)
axe.errorbar(xs, med['med'].values,
             yerr=[med['med'].values - med['p05'].values,
                   med['p95'].values - med['med'].values],
             fmt='none', ecolor=INK, lw=1.0, capsize=2.0, capthick=1.0)
axe.axhline(0, color=INK, lw=0.8)
for xi, (m0, p0, p1) in enumerate(zip(med['med'].values, med['p05'].values,
                                      med['p95'].values)):
    halo(axe, xi, m0 + (1.4 if m0 > 0 else -1.4), f'{m0:+.1f}%',
         color=MID_BLUE if m0 > 0 else RED, fs=6.5,
         va='bottom' if m0 > 0 else 'top')
axe.set_xticks(xs)
axe.set_xticklabels([f'{PARADIGM_NAME[p]}\n{med.loc[p, "spacing_D"]:.1f}D'
                     for p in PARADIGM_ORDER], fontsize=6.3)
axe.set_ylabel('Median difference at best orientation (%)')
axe.set_xlim(-0.55, 5.55); axe.set_ylim(-34, 32)
axe.grid(axis='x', visible=False)
axe.text(0.02, 0.025, 'Blue: positive change  Red: negative change',
         transform=axe.transAxes, fontsize=5.8, color='#666666')
axe.text(0.02, 0.965, 'n = 155 projects per template',
         transform=axe.transAxes, va='top', fontsize=5.8, color='#555555')
panel_label(axe, 'e', dy=1.06)

# ============ f. 间距–中位变化散点（去折线，真实 x + dodge；v7：下排右） ============
axf = fig.add_axes([0.695, 0.065, 0.28, 0.395])
xd = med['spacing_D'].values; yd = med['med'].values
xd_j = xd + np.array([0.0, -0.13, 0.13, 0.0, 0.0, 0.0])
for xi, yi, p in zip(xd_j, yd, PARADIGM_ORDER):
    p05, p95 = med.loc[p, 'p05'], med.loc[p, 'p95']
    col = MID_BLUE if yi > 0 else RED
    axf.plot([xi, xi], [p05, p95], color=INK, lw=0.9, zorder=2)
    axf.scatter(xi, yi, s=32, c=col, zorder=3, edgecolors='white',
                linewidths=0.6)
axf.axhline(0, color=INK, lw=0.8)
offs = [(0.85, 2.4), (-0.72, -3.2), (0.72, -3.2), (0.85, 2.4),
        (-0.95, -3.0), (-0.95, 2.4)]
for p, xi, xi0, yi, (oxa, oyb) in zip(PARADIGM_ORDER, xd_j, xd, yd, offs):
    axf.text(xi + oxa, yi + oyb,
             f'{PARADIGM_NAME[p]} {xi0:.1f}D\n{yi:+.1f}%',
             fontsize=6.2, ha='center', va='center', color=INK)
axf.text(0.02, 0.965, f'Built farms: S = {medS:.1f} D '
         f'(5–95%: {p5S:.1f}–{p95S:.1f} D, n = {len(S_built)})',
         transform=axf.transAxes, va='top', fontsize=5.8, color='#555555')
axf.set_xlabel('Minimum turbine spacing (D)')
axf.set_ylabel('Median difference (%)')
axf.set_xlim(2.7, 12.7)
lo_f = min(0.0, med['p05'].min()) - 3
hi_f = med['p95'].max() + 4
axf.set_ylim(lo_f, hi_f)
axf.grid(axis='x', visible=False)
axf.text(0.02, 0.025, 'Blue: positive  Red: negative · whiskers 5–95%',
         transform=axf.transAxes, fontsize=5.8, color='#666666')
panel_label(axf, 'f', dy=1.06)

save_fig(fig, os.path.join(HERE, 'Fig4_nat_v9.png'))
print('OK  fig4v9  corridor medians =', [round(x, 2) for x in m_corr],
      ' other =', [round(x, 2) for x in m_oth],
      '| dE corridor =', {g: [round(v) for v in vals] for g, vals in dE.items()})
