# -*- coding: utf-8 -*-
"""图 3 重绘 v4（2026-08-25，严格按 figure-model 参考图体例）
==================================================
v4 改动（学长反馈：欧洲图"全是紫"、指标精度不足）：
  b 面板指标由「六范式中保持前四分位响应的情景数（0–6 计数）」
  →「六范式响应幅度均值 A（连续 %，与面板 a/c 的 A_para 同口径）」，
  配色 nat_heat + 对数色标 0.1–4%（欧洲数值集中在低端，线性色标会再次单色）。
  海陆掩膜沿用：仅海洋格点入图（650/1,446 陆地格点不画）。
排布 = 原图 3（学长版）不变：a 六范式A对比（上，宽扁） / b 东亚+欧洲两图并排
（近方形）+ 横色条 / c 散点（左下） / d ROC（右下）。
体例 = 参考图（NC groundwater 图1/图2）：面板细黑边框、框内左上加粗黑面板字母、
数值彩色加粗+白色描边、地图青蓝海+白陆+深灰海岸线、黑框图例、连续顺序蓝系色条。
数据：wp5c_cross_farms.npz（A 六情景）/ wp5c_cross_grid.npz（格点 A）/
      wp9c_farm_metrics.csv（A_real、is_corr、质心）/ wp6c_loo.npz（留出预测）/
      wp7d_scenario_attribution.csv（范式间距）
所有统计数字由数据实算 + 与冻结报告对账断言，不硬编码。
输出：figures-nat/Fig3_nat.png / .pdf
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import spearmanr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.ops as so
import shapely.vectorized as sv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), '补算', 'output')
sys.path.insert(0, HERE)
from nc_style_nat import (apply_style, panel_label, halo, save_fig, RED,
                          DEEP_BLUE, LIGHT_BLUE, PALE_GREEN, ORANGE, INK,
                          BOX_EC, GROUP2, GROUP2_SOFT, PARADIGM_ORDER,
                          PARADIGM_NAME, LAND_COL, SEA_COL, COAST_COL,
                          COAST_LW, GRIDLINE_COL)

# ── 数据 ──
cf = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'), allow_pickle=True)
grid = np.load(os.path.join(OUT, 'wp5c_cross_grid.npz'), allow_pickle=True)
loo = np.load(os.path.join(OUT, 'wp6c_loo.npz'), allow_pickle=True)
fm = pd.read_csv(os.path.join(OUT, 'wp9c_farm_metrics.csv'))
w7d = pd.read_csv(os.path.join(OUT, 'wp7d_scenario_attribution.csv'))

IDs = cf['farm_ids']; A = cf['A']                      # (171, 6)
fm = fm.set_index('farm_id').reindex(IDs)
corr = fm['is_corr'].fillna(False).values.astype(bool)
A_real = fm['A_real'].values.astype(float)
A_para = A.mean(axis=1)                                 # 六范式均值
spacing = w7d.set_index('scenario').loc[PARADIGM_ORDER, 'spacing_D'].values

# ── 对账断言（冻结报告 wp5c/wp9c/wp6c）──
med_c = [np.median(A[corr, j]) for j in range(6)]
med_o = [np.median(A[~corr, j]) for j in range(6)]
assert np.allclose(med_c, [1.20, 5.25, 5.25, 2.91, 4.59, 0.82], atol=0.01), med_c
assert np.allclose(med_o, [0.21, 0.71, 0.71, 0.51, 0.74, 0.15], atol=0.01), med_o
rho13 = spearmanr(A_para, A_real).statistic
assert abs(rho13 - 0.706) < 0.01, rho13

# ── 样式 ──
apply_style()
fig = plt.figure(figsize=(12.2, 13.0))

# ============ a. 六范式分组中位 + IQR（对数轴，宽扁条） ============
axa = fig.add_axes([0.085, 0.795, 0.83, 0.16])
axa.set_xlim(-0.45, 5.45); axa.set_ylim(0.06, 18)
axa.set_yscale('log')
axa.set_ylabel('Orientation-response amplitude, A (%)')
axa.set_yticks([0.1, 1, 10]); axa.set_yticklabels(['0.1', '1', '10'])
axa.set_xticks(range(6))
axa.set_xticklabels([f'{PARADIGM_NAME[p]}\n{spacing[j]:.1f}D'
                     for j, p in enumerate(PARADIGM_ORDER)], fontsize=7.5)
axa.grid(axis='x', visible=False)
for j in range(6):
    ac, ao = A[corr, j], A[~corr, j]
    q1c, m_c, q3c = np.percentile(ac, [25, 50, 75])
    q1o, m_o, q3o = np.percentile(ao, [25, 50, 75])
    axa.plot([j, j], [m_o, m_c], '-', color='#c9cfd6', lw=0.8, zorder=1)
    axa.errorbar(j, m_c, yerr=[[m_c - q1c], [q3c - m_c]], fmt='o', color=RED,
                 ms=4.6, lw=1.0, capsize=2.0, capthick=1.0, zorder=4,
                 label='Corridor farms (n=23)' if j == 0 else None)
    axa.errorbar(j, m_o, yerr=[[m_o - q1o], [q3o - m_o]], fmt='s', color=DEEP_BLUE,
                 ms=3.9, lw=1.0, capsize=2.0, capthick=1.0, zorder=3,
                 label='Other farms (n=148)' if j == 0 else None)
    halo(axa, j, 12.2, f'{m_c / m_o:.1f}$\\times$', color=RED, fs=7.6)
axa.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.22),
           handletextpad=0.35, columnspacing=1.2, fontsize=7.2)
axa.text(0.985, 0.05, 'Points show medians; whiskers show interquartile ranges',
         transform=axa.transAxes, ha='right', va='bottom', fontsize=6.8,
         color='#555555')
panel_label(axa, 'a')

# ============ b. 东亚 / 欧洲支持域（并排近方形 + 横色条） ============
glon, glat, gA, gvalid = grid['lon'], grid['lat'], grid['A'], grid['valid']
uq_lon = np.unique(np.round(glon, 2)); uq_lat = np.unique(np.round(glat, 2))
slon = np.median(np.diff(uq_lon)); slat = np.median(np.diff(uq_lat))
assert abs(slon - slat) < 0.05, (slon, slat, 'grid spacing mismatch')

# 支持域格点海陆掩膜（自然地球 50m 陆界）：只把海洋格点画进地图，
# 避免把陆上格点画成"陆上风机"。task1 1° 网格含陆地点（数据如实保留、
# 统计不动），此处仅地图显示按海陆掩膜过滤。
_land = so.unary_union(list(cfeature.NaturalEarthFeature('physical', 'land', '50m').geometries()))
_pts = np.column_stack([glon, glat])
at_sea = ~sv.contains(_land, _pts[:, 0], _pts[:, 1])
print(f'格点海陆掩膜: 海洋 {at_sea.sum()} / 陆地 {(~at_sea).sum()} (仅海洋格点入图)')

# 六范式响应幅度均值 A（连续 %，与面板 a/c 的 A_para = A.mean(axis=1) 同口径）
Amean = np.where(gvalid, np.nanmean(gA, axis=1), np.nan)

# 新指标分区对账（欧洲/东亚中位，供论文图注与正文引用）
_ea = (glon >= 100) & (glon <= 130) & (glat >= 2) & (glat <= 36) & gvalid & at_sea
_eu = (glon >= -10) & (glon <= 30) & (glat >= 36) & (glat <= 62) & gvalid & at_sea
_mea = np.nanmedian(Amean[_ea]); _meu = np.nanmedian(Amean[_eu])
print(f'图3b 新指标 A 中位: 东亚 {_mea:.2f}% (n={_ea.sum()}) / 欧洲 {_meu:.2f}% '
      f'(n={_eu.sum()})；全体 p99={np.nanpercentile(Amean, 99):.2f}%')

cmap = LinearSegmentedColormap.from_list('nat_heat',
                                         [LIGHT_BLUE, PALE_GREEN, ORANGE, RED])
norm = LogNorm(vmin=0.1, vmax=4.0)
# 1° 格点画成带白缝的细方阵（约 62% 边距，避免满铺成马赛克）
sq = max((slon * 7.14) ** 2, 20.0)

def draw_grid_map(ax, ext, title):
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
        facecolor=SEA_COL, edgecolor='none'), zorder=-1)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
        facecolor=LAND_COL, edgecolor=COAST_COL, linewidth=COAST_LW), zorder=0)
    ax.add_feature(cfeature.BORDERS, edgecolor='#c8c8c8', linewidth=0.3, zorder=1)
    gl = ax.gridlines(linewidth=0.3, color=GRIDLINE_COL, linestyle='--',
                      draw_labels=True, dms=False, x_inline=False,
                      y_inline=False,
                      xlocs=np.arange(-140, 140, 5), ylocs=np.arange(-60, 80, 5))
    gl.top_labels = False; gl.right_labels = False
    ax.tick_params(labelsize=7)
    ax.set_title(title, loc='left', x=0.10, fontsize=8.5, fontweight='bold',
                 pad=4, zorder=40)
    xm = (glon >= ext[0]) & (glon <= ext[1])
    ym = (glat >= ext[2]) & (glat <= ext[3])
    sel = xm & ym & gvalid & at_sea
    if sel.sum():
        ax.scatter(glon[sel], glat[sel], c=Amean[sel], s=sq, marker='s',
                   cmap=cmap, norm=norm, linewidths=0.25, edgecolors='white',
                   zorder=2)
    ccx = fm.loc[corr, 'cent_lon'].values; ccy = fm.loc[corr, 'cent_lat'].values
    ax.scatter(ccx, ccy, s=30, marker='*', c=RED, edgecolors='white',
               linewidths=0.5, zorder=5, transform=ccrs.PlateCarree())

axb1 = fig.add_axes([0.075, 0.395, 0.315, 0.38], projection=ccrs.PlateCarree())
draw_grid_map(axb1, [100, 130, 2, 36], 'East Asia')
axb2 = fig.add_axes([0.475, 0.395, 0.435, 0.38], projection=ccrs.PlateCarree())
draw_grid_map(axb2, [-10, 30, 36, 62], 'Europe')
panel_label(axb1, 'b')

cax = fig.add_axes([0.335, 0.35, 0.30, 0.016])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation='horizontal')
cb.set_ticks([0.1, 0.3, 1, 3])
cb.ax.tick_params(labelsize=7.2, length=2)
cb.outline.set_edgecolor(BOX_EC); cb.outline.set_linewidth(0.8)
cb.ax.xaxis.set_label_position('top')
cb.set_label('Mean response across six paradigms, A (%) — log colour scale',
             fontsize=7.4, labelpad=5)

# ============ c. 六范式均值 A vs 真实排布 A ============
axc = fig.add_axes([0.075, 0.05, 0.42, 0.29])
lo = min(A_para.min(), A_real.min()) * 0.85
hi = max(A_para.max(), A_real.max()) * 1.15
axc.set_xscale('log'); axc.set_yscale('log')
axc.set_xlim(lo, hi); axc.set_ylim(lo, hi)
axc.plot([lo, hi], [lo, hi], '--', color='#c9cfd6', lw=0.9, zorder=1)
axc.scatter(A_para[~corr], A_real[~corr], s=10, c=GROUP2_SOFT['other'],
            alpha=0.85, linewidths=0, zorder=2,
            label='Other farms (n=148)')
axc.scatter(A_para[corr], A_real[corr], s=16, c=RED, alpha=0.95,
            linewidths=0, zorder=3, label='Corridor farms (n=23)')
axc.set_xlabel('Mean response across six paradigms, A (%)')
axc.set_ylabel('Real-layout response, A (%)')
axc.legend(loc='lower right', bbox_to_anchor=(0.985, 0.02), fontsize=7,
           handlelength=1.0)
axc.text(0.035, 0.965, f'Spearman $\\rho$ = {rho13:.3f}\nn = 171 farms',
         transform=axc.transAxes, va='top', fontsize=8.2, color='#111111',
         fontweight='bold')
panel_label(axc, 'c')

# ============ d. 整走廊留出 ROC ============
from sklearn.metrics import roc_curve, roc_auc_score
axd = fig.add_axes([0.545, 0.05, 0.43, 0.29])
axd.set_xlim(-0.02, 1.02); axd.set_ylim(-0.02, 1.05)
axd.plot([0, 1], [0, 1], ':', color='#bbbbbb', lw=0.8)

def draw_roc(pred, lab, lab_thr, lbl, col, ls):
    y = lab >= lab_thr                       # 二分类：是否高于前四分位阈值
    fpr_c, tpr_c, thrs = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    m = pred >= np.quantile(pred, 0.75)      # 工作点 = 预测前 25% 直接阈值化
    tpr = (m & y).sum() / max(y.sum(), 1)
    fpr = (m & ~y).sum() / max((~y).sum(), 1)
    ba = (tpr + (1 - fpr)) / 2
    axd.plot(fpr_c, tpr_c, ls, color=col, lw=1.7,
             label=f'{lbl} (AUC = {auc:.3f})')
    axd.plot(fpr, tpr, 'o', mfc='white', mec=col, ms=6.5, mew=1.4, zorder=5)
    return auc, ba, tpr

auc_p, ba_p, rec_p = draw_roc(loo['pred_para'], loo['lab_para'], loo['thr_para'],
                              'Paradigm-mean target', DEEP_BLUE, '-')
auc_r, ba_r, rec_r = draw_roc(loo['pred_real'], loo['lab_real'], loo['thr_real'],
                              'Real-layout target', RED, '--')
assert abs(auc_p - 0.628) < 0.005 and abs(auc_r - 0.498) < 0.005, (auc_p, auc_r)
assert abs(ba_p - 0.571) < 0.005 and abs(rec_p - 0.357) < 0.005, (ba_p, rec_p)
assert abs(ba_r - 0.539) < 0.005 and abs(rec_r - 0.310) < 0.005, (ba_r, rec_r)

axd.set_xlabel('False-positive rate')
axd.set_ylabel('True-positive rate')
axd.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=7,
           framealpha=0.95, handlelength=1.8)
axd.text(0.985, 0.035,
         'Top-quartile screen\nBalanced accuracy / recall\n'
         f'Paradigm mean {ba_p * 100:.1f}% / {rec_p * 100:.1f}%\n'
         f'Real layout {ba_r * 100:.1f}% / {rec_r * 100:.1f}%',
         transform=axd.transAxes, ha='right', va='bottom', fontsize=6.8,
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=BOX_EC, lw=0.7))
panel_label(axd, 'd')

save_fig(fig, os.path.join(HERE, 'Fig3_nat.png'))
print(f'OK  a(med)={[round(float(x), 2) for x in med_c]}  rho={rho13:.3f}  '
      f'auc={auc_p:.3f}/{auc_r:.3f}  ba/rec={ba_p * 100:.1f}%/{rec_p * 100:.1f}% '
      f'{ba_r * 100:.1f}%/{rec_r * 100:.1f}%  b(A中位 EA/EU)='
      f'{_mea:.2f}%/{_meu:.2f}%')
