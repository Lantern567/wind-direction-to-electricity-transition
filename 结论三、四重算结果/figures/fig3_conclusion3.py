"""
图 3 结论三：受控交叉仿真与稳健走廊（五面板，方案 §4.8）
====================================================
a. A 响应矩阵：171 场址 × 36 模板（行按场址 A 均值降序，列按形态×间距分组）
b. 稳健走廊地图：跨排布中位分位 R（1,205 有效格点，观测支持域）
c. 支持率地图：跨模板前四分位保持比例 F75
d. 类型匹配散点：物理模板 A（36 模板均值）vs 真实排布 A，按留出走廊着色
e. 召回曲线：嵌套走廊留出预测，筛选比例 vs 观测高响应（A≥Q75）召回
   设计目标：前 25% 筛选捕获 ≥ 50%（图上标出）

风格：沿用 分析材料_几何主导杠杆/figures/nc_style.py（Nature Communications 体例，全英文）
地图：cartopy 海岸线，正文称"观测支持域"（方案 [73]：非完整全球海域）
输入：wp5_cross_farms.npz / wp5_rfd_grid.csv / wp5_cross_grid.npz / wp6a_loo.npz
     wp7a_real_curves.npz（如存在，真实 A 用自算口径；否则 task3 18 角度口径）
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
from scipy.stats import spearmanr

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BUSH), 'output')
REPO = os.path.dirname(os.path.dirname(BUSH))
FIGOUT = os.path.join(BUSH)
sys.path.insert(0, os.path.join(REPO, '分析材料_几何主导杠杆', 'figures'))
from nc_style import PAL, apply_style, panel_label

CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
CORR_COL = {'Vietnam': '#DC2626', 'China strait': '#F59E0B',
            'Italy': '#8B5CF6', 'Denmark': '#059669', 'other': '#94A3B8'}

# ═══════════════════════════════════════════════════════════════════════
z5 = np.load(os.path.join(OUT, 'wp5_cross_farms.npz'))
A_farm = z5['A']                       # (171,36)
farm_ids = z5['farm_ids'].astype(int)
summ = pd.read_csv(os.path.join(OUT, 'wp2_template_summary.csv'), encoding='utf-8-sig')
MORPH = summ['morphology'].values; SPAC = summ['spacing_D'].values
rfd = pd.read_csv(os.path.join(OUT, 'wp5_rfd_grid.csv'), encoding='utf-8-sig')
zg = np.load(os.path.join(OUT, 'wp5_cross_grid.npz'))
gvalid = zg['valid']
zloo = np.load(os.path.join(OUT, 'wp6a_loo.npz'))

# 真实 A：优先 wp7a 自算口径（36 档半圆与模板同口径），否则 task3
z7_path = os.path.join(OUT, 'wp7a_real_curves.npz')
if os.path.exists(z7_path):
    z7 = np.load(z7_path)
    A_real_s = pd.Series(z7['A'], index=z7['farm_ids'].astype(int))
    real_src = 'wp7a self-computed (36-bin semicircle)'
else:
    t3 = pd.read_csv(os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv'), encoding='utf-8-sig')
    g3 = t3.groupby('farm_id')['expected_AEP_kWh'].agg(['max', 'mean'])
    A_real_s = 100 * (g3['max'] - g3['mean']) / g3['mean']
    real_src = 'task3 (18-angle)'

apply_style()
fig = plt.figure(figsize=(11.5, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95],
                      width_ratios=[1.12, 1.0, 1.0], hspace=0.34, wspace=0.26)

# ─────────────────────────── a. A 响应矩阵 ───────────────────────────
axa = fig.add_subplot(gs[0, :2])
# 行排序：场址 A 均值降序；列排序：形态（belt/cluster/multi_cluster/rule_grid）×间距
row_ord = np.argsort(-A_farm.mean(axis=1))
morph_ord = {'belt': 0, 'cluster': 1, 'multi_cluster': 2, 'rule_grid': 3}
col_ord = sorted(range(36), key=lambda k: (morph_ord[MORPH[k]], SPAC[k], k))
Am = A_farm[np.ix_(row_ord, col_ord)]
im = axa.imshow(Am, aspect='auto', cmap='viridis', vmin=0, vmax=6)
axa.set_yticks([])
# 列标签：形态×间距边界
xs, labels = [], []
for k in range(36):
    c = col_ord[k]
    if k == 0 or (MORPH[col_ord[k - 1]], SPAC[col_ord[k - 1]]) != (MORPH[c], SPAC[c]):
        xs.append(k - 0.5)
        labels.append(f'{MORPH[c][:4]}{SPAC[c]:.0f}D')
xs.append(35.5)
axa.set_xticks([(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)])
axa.set_xticklabels(labels, rotation=0, fontsize=6.5)
for x in xs:
    axa.axvline(x, color='white', lw=0.6)
axa.set_xlabel('Standard template (morphology × spacing, 3 reps)')
axa.set_ylabel(f'{A_farm.shape[0]} farms (sorted by mean A)')
cb = fig.colorbar(im, ax=axa, shrink=0.8, pad=0.02)
cb.set_label('Response amplitude A (pp)')
panel_label(axa, 'a')

# ─────────────────────────── b. R 稳健走廊地图 ───────────────────────────
def map_panel(pos, letter, vals, cmap, label, cbar_fmt=None):
    ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
    ax.set_extent([-100, 160, 0, 70], crs=ccrs.PlateCarree())
    ax.coastlines(lw=0.3, color='#475569')
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.LAND, fc='#EEF2F6', ec='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, fc='white', ec='none', zorder=0)
    v = np.full(len(rfd), np.nan)
    v[gvalid] = vals
    sc = ax.scatter(rfd['lon'], rfd['lat'], c=v, s=9, cmap=cmap,
                    vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98),
                    transform=ccrs.PlateCarree(), zorder=3, linewidths=0)
    # 走廊成员农场位置（星标）
    geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
    for cname, mem in CORRIDORS.items():
        lons = [geo.loc[f, 'cent_lon'] for f in mem if f in geo.index]
        lats = [geo.loc[f, 'cent_lat'] for f in mem if f in geo.index]
        ax.scatter(lons, lats, marker='*', s=26, color=CORR_COL[cname],
                   edgecolor='white', linewidths=0.4, transform=ccrs.PlateCarree(),
                   zorder=5, label=cname)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label(label)
    if cbar_fmt:
        cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(cbar_fmt))
    panel_label(ax, letter)
    return ax

axb = map_panel(gs[0, 2], 'b', rfd['R'].values[gvalid], 'inferno',
                'Robust rank R (median percentile)')
axb.legend(loc='lower left', fontsize=6, markerscale=0.8, frameon=False)

# ─────────────────────────── c. F75 支持率地图（CSV 为 0-100 百分比刻度 → /100）──
axc = map_panel(gs[1, 0], 'c', rfd['F75'].values[gvalid] / 100.0, 'viridis',
                'Support F75 (share of templates in top quartile)',
                cbar_fmt=lambda x, _: f'{x:.0%}')
axc.legend(loc='lower left', fontsize=6, markerscale=0.8, frameon=False)

# ─────────────────────────── d. 类型匹配散点 ───────────────────────────
axd = fig.add_subplot(gs[1, 1])
common = [i for i, f in enumerate(farm_ids) if f in A_real_s.index]
x = A_farm[common].mean(axis=1)
y = A_real_s.loc[farm_ids[common]].values
fold_of = {}
for cname, mem in CORRIDORS.items():
    for f in mem:
        fold_of[f] = cname
cols = [CORR_COL.get(fold_of.get(int(f), 'other'), 'other') for f in farm_ids[common]]
for cname in ['other', 'Vietnam', 'China strait', 'Italy', 'Denmark']:
    m = np.array([c == (fold_of.get(int(f), 'other')) for f, c in
                  zip(farm_ids[common], cols)])
    axd.scatter(x[m], y[m], s=14, c=CORR_COL[cname], alpha=0.85,
                edgecolors='white', linewidths=0.3,
                label=f'{cname} (n={m.sum()})')
lim = [0, max(x.max(), y.max()) * 1.05]
axd.plot(lim, lim, color=PAL['light'], lw=1, ls='--')
rho = spearmanr(x, y)[0]
axd.set_xlim(lim); axd.set_ylim(lim)
axd.set_xlabel('Template-mean A (pp, 36 templates)')
axd.set_ylabel('Real-layout A (pp)')
axd.set_title(f'Spearman ρ = {rho:.2f}  [{real_src}]', fontsize=8)
axd.legend(fontsize=6.5, loc='upper left')
panel_label(axd, 'd')

# ─────────────────────────── e. 召回曲线 ───────────────────────────
axe = fig.add_subplot(gs[1, 2])
for lab, pred, thr, col, ls in [('Template A', zloo['pred_tpl'], zloo['thr_tpl'], PAL['baseline'], '-'),
                                ('Real A', zloo['pred_real'], zloo['thr_real'], PAL['highlight'], '--')]:
    y_hi = (zloo['lab_tpl'] if lab == 'Template A' else zloo['lab_real']) >= thr
    if y_hi.sum() == 0:
        continue
    xs_, ys_ = [], []
    for q in np.arange(1, 101):
        n_sel = int(np.ceil(q / 100 * len(pred)))
        sel = np.argsort(-pred)[:n_sel]
        recall = y_hi[sel].sum() / y_hi.sum()
        xs_.append(q); ys_.append(recall)
    axe.plot(xs_, ys_, color=col, ls=ls, lw=1.4, label=lab)
axe.plot([25, 25], [0, 1], color=PAL['light'], lw=0.8, ls=':')
axe.scatter([25], [0.5], marker='o', s=24, facecolors='none',
            edgecolors=PAL['highlight'], lw=1.2, zorder=6)
axe.text(26, 0.47, 'design target\n(top 25% → 50% recall)', fontsize=6.5, color=PAL['ink'])
axe.set_xlabel('Share of projects screened by predicted A (%)')
axe.set_ylabel('Recall of observed high-A (top quartile)')
axe.set_xlim(0, 100); axe.set_ylim(0, 1)
axe.legend(fontsize=6.5, loc='lower right')
panel_label(axe, 'e')

fig.savefig(os.path.join(FIGOUT, 'Fig3_conclusion3.png'))
print('输出: Fig3_conclusion3.png')
print(f'真实 A 口径: {real_src} | 散点 Spearman ρ = {rho:.3f} (n={len(x)})')
