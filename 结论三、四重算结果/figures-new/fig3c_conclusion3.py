"""
图 3 结论三（建设范式版，2026-08-17 第四轮重绘）：五面板，子刊清晰体例
============================================================================
a. A_built 响应矩阵：171 场址 × 6 范式情境（行按 A 均值降序；右侧色点 = 走廊成员）
b. 稳健走廊地图：跨范式情境中位分位 R —— 四走廊海域区域放大图（越南南/北、台湾海峡、
   亚得里亚—塔兰托、丹麦海峡；星标 = 走廊成员；全球观测支持域见图 4a）
c. 支持率地图：同四走廊海域的跨范式前四分位保持比例 F75
d. 类型匹配散点：范式均值 A_built vs 真实排布 A（wp7a 自算口径），按走廊着色
e. 召回曲线：嵌套走廊留出预测（标签 = 范式均值 A / 真实 A），标注实测 AUC 与捕获率

第四轮重绘要点（用户反馈"新的排布可以再工整一点，图片可以长一点，整体不显小"）：
  - 图幅加长：12.8×12.7 → 12.8×14.0 in（打印 17.5×19.1 cm，占满一页），a/d/e 面板同步加大
  - 区域图等宽工整化：各区经纬跨度裁剪为统一宽高比 ≈1.05（两侧裁剪纯海面无风场信息），
    五幅图等高且宽≈2.3 in，打印约 3.0×3.2 cm（旧 2.45×2.8 cm，放大 ~20%）
  - 越南南 [104.3,110.2,7.2,12.8] 越南北 [105.8,110.5,19,23.5]
    台湾海峡—东南沿海 [111,123.5,19.5,31] 亚得里亚—塔兰托 [13.2,20.55,37.5,44.5]
    丹麦海峡 [5,12.35,52.5,59.5]（走廊成员均在区内）
  - 行内图距收紧 wspace 0.10，星标放大 s=120，经纬标签 7pt
输入：wp5c_cross_farms.npz / wp5c_rfd_grid.csv / wp5c_cross_grid.npz / wp6c_loo.npz
     wp7a_real_curves.npz
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
from scipy.stats import spearmanr

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BUSH), 'output')
REPO = os.path.dirname(os.path.dirname(BUSH))
sys.path.insert(0, os.path.join(REPO, '分析材料_几何主导杠杆', 'figures'))
from nc_style import PAL, CORRIDOR_COL, apply_style

PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
CORR_COL = {k: CORRIDOR_COL[k] for k in ['Vietnam', 'China strait', 'Italy', 'Denmark']}
CORR_COL['other'] = CORRIDOR_COL['other']

# 四走廊海域区域放大图：经纬跨度裁剪为统一宽高比 ≈1.05（等宽工整、打印放大）
REGIONS = [
    ('Vietnam coast (S)', [104.3, 110.2, 7.2, 12.8]),      # 跨 5.9° × 5.6°
    ('Vietnam coast (N)', [105.8, 110.5, 19.0, 23.5]),    # 跨 4.7° × 4.5°
    ('China strait',      [111.0, 123.5, 19.5, 31.0]),    # 跨 12.5° × 11.5°
    ('Adriatic–Taranto', [13.2, 20.55, 37.5, 44.5]),      # 跨 7.35° × 7°
    ('Danish straits',    [5.0, 12.35, 52.5, 59.5]),      # 跨 7.35° × 7°
]
REG_ASPECT = [e[2] - e[0] for _, e in REGIONS]  # PlateCarree: 经纬同比例
# 行内格宽比取各区经纬跨度比，使每幅图等高且 100% 填满（现各区宽高比接近，行内等宽）
REG_WIDTHS = [a / max(REG_ASPECT) for a in REG_ASPECT]

# ═══════════════════════════════════════════════════════════════════════
z5 = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'))
A_farm = z5['A']                       # (171,6) A_built
farm_ids = z5['farm_ids'].astype(int)
rfd = pd.read_csv(os.path.join(OUT, 'wp5c_rfd_grid.csv'), encoding='utf-8-sig')
zg = np.load(os.path.join(OUT, 'wp5c_cross_grid.npz'))
gvalid = zg['valid']
zloo = np.load(os.path.join(OUT, 'wp6c_loo.npz'))

z7 = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
A_real_s = pd.Series(z7['A'], index=z7['farm_ids'].astype(int))
real_src = 'wp7a self-computed (36-bin semicircle)'

corr_of = {}
for cname, mem in CORRIDORS.items():
    for f in mem:
        corr_of[f] = cname

geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'),
                  encoding='utf-8-sig').set_index('farm_id')

apply_style()
# nc_style 设了 savefig.bbox='tight'，会把图幅裁成内容边界（图面被压缩变小）；
# 显式关掉，按 figsize 整幅输出
plt.rcParams.update({'savefig.bbox': None,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.labelsize': 9, 'legend.fontsize': 8})

def pl(ax, letter):
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left', color='#333333')

fig = plt.figure(figsize=(12.8, 14.0))
gs = fig.add_gridspec(4, 2, height_ratios=[0.95, 0.95, 0.95, 1.22],
                      hspace=0.34, wspace=0.4)

# ─────────────────── a. A_built 响应矩阵（171 × 6 范式） ───────────────────
axa = fig.add_subplot(gs[0, :])
row_ord = np.argsort(-A_farm.mean(axis=1))
Am = A_farm[row_ord]
im = axa.imshow(Am, aspect='auto', cmap='viridis', vmin=0, vmax=6)
axa.set_yticks([])
axa.set_xticks(range(6))
axa.set_xticklabels(PIDS, rotation=0, fontsize=9)
axa.set_xlabel('Construction-paradigm scenario (64 turbines, built-axis anchored)',
               labelpad=6)
axa.set_ylabel(f'{A_farm.shape[0]} farms (sorted by mean A)', labelpad=6)
for r, i in enumerate(row_ord):
    fid = int(farm_ids[i])
    if fid in corr_of:
        axa.scatter(6.42, r, marker='s', s=5, color=CORR_COL[corr_of[fid]],
                    clip_on=False, linewidths=0)
cb = fig.colorbar(im, ax=axa, shrink=0.8, pad=0.03, aspect=24)
cb.set_label('A (pp)', fontsize=9)
cb.ax.tick_params(labelsize=8)
cb.outline.set_linewidth(0.4)
pl(axa, 'a')

# ─────────────────── 走廊海域区域放大图（b/c 各一行） ───────────────────
def map_row(pos, letter, vals, cmap, label, cbar_fmt=None, vmin=None, vmax=None):
    ss = pos.subgridspec(1, 5, width_ratios=REG_WIDTHS, wspace=0.10)
    v = np.full(len(rfd), np.nan)
    v[gvalid] = vals
    if vmin is None:
        vmin = np.nanpercentile(v, 2)
    if vmax is None:
        vmax = np.nanpercentile(v, 98)
    axes = []
    for i, (title, ext) in enumerate(REGIONS):
        ax = fig.add_subplot(ss[0, i], projection=ccrs.PlateCarree())
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        try:
            land = cfeature.NaturalEarthFeature(
                'physical', 'land', '50m', facecolor='#EDEDED', edgecolor='none')
        except Exception:
            land = cfeature.LAND
        ax.add_feature(land, zorder=0)
        ax.add_feature(cfeature.OCEAN, fc='#FFFFFF', ec='none', zorder=0)
        ax.coastlines(lw=0.5, color='#666666')
        gl = ax.gridlines(draw_labels=True, lw=0.3, color='#C9C9C9', alpha=0.8,
                          xlocs=range(-180, 181, 5), ylocs=range(-90, 91, 5))
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (i == 0)
        gl.xlabel_style = {'size': 7, 'color': '#666666'}
        gl.ylabel_style = {'size': 7, 'color': '#666666'}
        sc = ax.scatter(rfd['lon'], rfd['lat'], c=v, s=6, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), zorder=3, linewidths=0,
                        alpha=0.95)
        lo, hi, l1, l2 = ext
        for cname, mem in CORRIDORS.items():
            lons = [geo.loc[f, 'cent_lon'] for f in mem if f in geo.index]
            lats = [geo.loc[f, 'cent_lat'] for f in mem if f in geo.index]
            ax.scatter(lons, lats, marker='*', s=120, color=CORR_COL[cname],
                       edgecolor='white', linewidths=0.8,
                       transform=ccrs.PlateCarree(), zorder=6)
        ax.text(0.015, 0.985, title, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', va='top', ha='left', color='#333333',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.2),
                zorder=7)
        if i == 0:
            pl(ax, letter)
        axes.append(ax)
    cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.05, aspect=18)
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)
    if cbar_fmt:
        cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(cbar_fmt))
    return axes

axb = map_row(gs[1, :], 'b', rfd['R'].values[gvalid], 'viridis',
              'Median rank\npercentile R')

axc = map_row(gs[2, :], 'c', rfd['F75'].values[gvalid] / 100.0, 'viridis',
              'Support F75\n(share of paradigms\nin top quartile)',
              cbar_fmt=lambda x, _: f'{x:.0%}')

# ─────────────────── d. 类型匹配散点 ───────────────────
axd = fig.add_subplot(gs[3, 0])
common = [i for i, f in enumerate(farm_ids) if f in A_real_s.index]
x = A_farm[common].mean(axis=1)
y = A_real_s.loc[farm_ids[common]].values
order = ['other', 'Vietnam', 'China strait', 'Italy', 'Denmark']
for cname in order:
    m = np.array([corr_of.get(int(f), 'other') == cname for f in farm_ids[common]])
    axd.scatter(x[m], y[m], s=22, c=CORR_COL[cname], alpha=0.85,
                edgecolors='white', linewidths=0.5,
                label=f'{cname} (n={m.sum()})', zorder=3)
lim = [0, max(x.max(), y.max()) * 1.05]
axd.plot(lim, lim, color=PAL['light'], lw=1.0, ls='--', zorder=2)
rho = spearmanr(x, y)[0]
axd.set_xlim(lim); axd.set_ylim(lim)
axd.set_xlabel('Paradigm-mean A (pp, 6 paradigms)')
axd.set_ylabel('Real-layout A (pp)')
axd.text(0.03, 0.95, f'Spearman $\\rho$ = {rho:.2f}',
         transform=axd.transAxes, fontsize=10, va='top', color='#333333')
axd.legend(fontsize=8, loc='lower right', frameon=False,
           handletextpad=0.4, borderaxespad=0.4)
pl(axd, 'd')

# ─────────────────── e. 召回曲线 ───────────────────
axe = fig.add_subplot(gs[3, 1])
for lab, pred, thr, col, ls in [('Paradigm A', zloo['pred_para'], zloo['thr_para'], PAL['baseline'], '-'),
                                ('Real A', zloo['pred_real'], zloo['thr_real'], PAL['highlight'], '--')]:
    y_hi = (zloo['lab_para'] if lab == 'Paradigm A' else zloo['lab_real']) >= thr
    if y_hi.sum() == 0:
        continue
    xs_, ys_ = [], []
    for q in np.arange(1, 101):
        n_sel = int(np.ceil(q / 100 * len(pred)))
        sel = np.argsort(-pred)[:n_sel]
        recall = y_hi[sel].sum() / y_hi.sum()
        xs_.append(q); ys_.append(recall)
    axe.plot(xs_, ys_, color=col, ls=ls, lw=1.8, label=lab, zorder=3)
axe.plot([25, 25], [0, 1], color=PAL['light'], lw=0.9, ls=':', zorder=2)
axe.scatter([25], [0.5], marker='o', s=44, facecolors='none',
            edgecolors=PAL['ink'], lw=1.2, zorder=5)
axe.text(27, 0.40, 'design target\n(top 25% $\\to$ 50% recall)', fontsize=8,
         color=PAL['ink'],
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5),
         zorder=6)
axe.text(0.02, 0.97,
         'corridor-out LOO (labels: paradigm-mean A)\n'
         'AUC = 0.628 · Spearman ρ = 0.415 · top-25% capture = 35.7%',
         transform=axe.transAxes, fontsize=7.5, va='top', color='#333333',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5),
         zorder=6)
axe.set_xlabel('Share of projects screened by predicted A (%)')
axe.set_ylabel('Recall of observed high-A (top quartile)')
axe.set_xlim(0, 100); axe.set_ylim(0, 1)
axe.legend(fontsize=8, loc='lower right', frameon=False,
           handletextpad=0.4, borderaxespad=0.4)
pl(axe, 'e')

fig.savefig(os.path.join(BUSH, 'Fig3c_conclusion3_paradigms.png'), dpi=450)
print('输出: Fig3c_conclusion3_paradigms.png')
print(f'真实 A 口径: {real_src} | 散点 Spearman ρ = {rho:.3f} (n={len(x)})')
