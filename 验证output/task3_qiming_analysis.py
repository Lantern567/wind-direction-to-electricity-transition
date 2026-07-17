"""
=============================================================================
琪明任务脚本 — 任务一(间距构型) + 任务三(内因方差分解)
=============================================================================
数据依赖:
  任务一: 仅需 task2 layout_morphology.csv (已有) → 独立完成
  任务三: 需 S3 对比表 (廷显已产出, 已有) + S1 最优朝向 (廷显已产出, 已有)
           + layout_morphology (已有) → 可以完成
  任务五(a): 需 S2 范式排布生成管线 + 真实风机 UTM 坐标 → 等廷显交付后做

生成时间: 2026-07-15
=============================================================================
"""
import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

TASK3_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TASK3_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

S1_PATH = os.path.join(TASK3_DIR, 'task3_s1_optimal_orientation.csv')
S3_PATH = os.path.join(TASK3_DIR, 'task3_s3_comparison.csv')
TASK2_LAYOUT = r'C:\Users\beyqm\Desktop\北大实习\data\task2_output\layout_morphology.csv'

print('=' * 60)
print('琪明任务分析 | 任务一(间距构型) + 任务三(内因分解)')
print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 60)

# ═══════════════════════════════════════════════════════════════════════════
# 0. 数据加载
# ═══════════════════════════════════════════════════════════════════════════
print('\n--- 0. 数据加载 ---')
lm = pd.read_csv(TASK2_LAYOUT)
lm['farm_id_int'] = lm['farm_id'].str.extract(r'farm_(\d+)').astype(int)
print(f'  layout_morphology: {len(lm)} farms')

s1 = pd.read_csv(S1_PATH)
idx_opt = s1.groupby('farm_id')['expected_AEP_kWh'].idxmax()
df_opt = s1.loc[idx_opt, ['farm_id', 'angle_deg']].copy()
df_opt.columns = ['farm_id', 'theta_opt']
print(f'  S1 optimal: {len(df_opt)} farms')

s3 = pd.read_csv(S3_PATH)
# Get country per farm
farm_country = s3.groupby('farm_id')['country'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
print(f'  S3 comparison: {len(s3)} records, {s3["farm_id"].nunique()} farms, {s3["year"].nunique()} years')

# Merge: layout + theta_opt + country (all on farm_id_int)
merged_farm = lm.merge(df_opt, left_on='farm_id_int', right_on='farm_id', how='left')
merged_farm = merged_farm.merge(farm_country, left_on='farm_id_int', right_on='farm_id', how='left',
                                suffixes=('_opt', '_country'))
# Fix: after merge, farm_id from lm is str, farm_id from df_opt is int
# Use farm_id_int as the canonical integer id
merged_farm['axis_deviation'] = merged_farm['theta_opt'] - merged_farm['axis_deg']
merged_farm['axis_deviation'] = ((merged_farm['axis_deviation'] + 90) % 180) - 90
merged_farm['axis_deviation_abs'] = merged_farm['axis_deviation'].abs()
print(f'  合并后: {len(merged_farm)} farms')

# ═══════════════════════════════════════════════════════════════════════════
# 任务一: 间距 vs 5D 基准 + 构型分析
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('任务一: 间距 vs 5D 基准 + 构型分析')
print('  [数据依赖: 仅 task2 layout_morphology.csv]')
print('=' * 60)

# --- 1a. 整体分布 ---
print('\n[1a] 整体 SpacingD 分布')
stats_spacing = {
    'mean': lm['spacing_D'].mean(), 'median': lm['spacing_D'].median(),
    'std': lm['spacing_D'].std(), 'min': lm['spacing_D'].min(), 'max': lm['spacing_D'].max(),
    'P5': lm['spacing_D'].quantile(0.05), 'P25': lm['spacing_D'].quantile(0.25),
    'P75': lm['spacing_D'].quantile(0.75), 'P95': lm['spacing_D'].quantile(0.95),
    'pct_below_3D': (lm['spacing_D'] < 3).mean() * 100,
    'pct_below_4D': (lm['spacing_D'] < 4).mean() * 100,
    'pct_below_5D': (lm['spacing_D'] < 5).mean() * 100,
    'n_total': len(lm),
}
for k, v in stats_spacing.items():
    print(f'  {k}: {v:.2f}' if isinstance(v, float) else f'  {k}: {v}')

# --- 1b. 按范式 ---
print('\n[1b] 按范式分组')
for p in ['A', 'B', 'C', 'D', 'E']:
    sub = lm[lm['paradigm'] == p]
    if len(sub) == 0: continue
    print(f'  范式{p}: n={len(sub)}, median={sub["spacing_D"].median():.1f}D, '
          f'mean={sub["spacing_D"].mean():.1f}D, <5D: {(sub["spacing_D"]<5).sum()}/{len(sub)}')

# --- 1c. 按国家 ---
print('\n[1c] 按国家分组 (top 10)')
top_countries = merged_farm.groupby('country').agg(
    n=('spacing_D', 'count'), median_SD=('spacing_D', 'median'),
    mean_dev=('axis_deviation_abs', 'mean')
).sort_values('n', ascending=False).head(10)
for c, r in top_countries.iterrows():
    print(f'  {c}: n={int(r["n"])}, median_SD={r["median_SD"]:.1f}D, mean|dev|={r["mean_dev"]:.0f}°')

# --- 1d. 按年份 ---
print('\n[1d] 按首台投运年份')
for yr in sorted(lm['first_year'].unique()):
    sub = lm[lm['first_year'] == yr]
    print(f'  {int(yr)}: n={len(sub)}, median={sub["spacing_D"].median():.1f}D')

# --- 1e. 构型分析 ---
print('\n[1e] 构型特征')
print(f'  pc1_share: mean={lm["pc1_share"].mean():.3f}, median={lm["pc1_share"].median():.3f}')
print(f'  强对齐(pc1>0.8): {(lm["pc1_share"]>0.8).sum()} ({(lm["pc1_share"]>0.8).mean()*100:.1f}%)')
print(f'  中等(0.6-0.8): {((lm["pc1_share"]>=0.6)&(lm["pc1_share"]<=0.8)).sum()} ({((lm["pc1_share"]>=0.6)&(lm["pc1_share"]<=0.8)).mean()*100:.1f}%)')
print(f'  弱对齐/错列(<0.6): {(lm["pc1_share"]<0.6).sum()} ({(lm["pc1_share"]<0.6).mean()*100:.1f}%)')
print(f'  aspect_ratio: mean={lm["aspect_ratio"].mean():.2f}, median={lm["aspect_ratio"].median():.2f}')
print(f'  近方形(<1.5): {(lm["aspect_ratio"]<1.5).sum()}, 带状(>3): {(lm["aspect_ratio"]>3).sum()}')

print('\n  布局类型分布:')
for lt in lm['layout_type'].value_counts().index:
    sub = lm[lm['layout_type'] == lt]
    print(f'    {lt}: {len(sub)} ({len(sub)/len(lm)*100:.1f}%), Clark-Evans R={sub["clark_evans_R"].mean():.3f}')

print('\n  范式 × 构型交叉表:')
ct = pd.crosstab(lm['paradigm'], lm['layout_type'])
for p in ct.index:
    parts = [f'{lt}={ct.loc[p, lt]}' for lt in ct.columns if ct.loc[p, lt] > 0]
    print(f'    范式{p}: {", ".join(parts)}')

# Save
pd.Series(stats_spacing).to_csv(os.path.join(OUTPUT_DIR, 'qiming_task1_spacing_stats.csv'))
print('\n  已保存: qiming_task1_spacing_stats.csv')

# ═══════════════════════════════════════════════════════════════════════════
# 任务三: 内因方差分解 (用实际 S3 FLORIS 数据, 不用 RF 代理)
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('任务三: 内因方差分解')
print('  [数据依赖: S3 对比表(廷显已产出) + S1 theta_opt(廷显已产出) + layout_morphology]')
print('=' * 60)

# 取 Gauss 模型 real 排布的数据 (实际 FLORIS 输出)
gauss_real = s3[(s3['wake_model'] == 'gauss') & (s3['layout_type'] == 'real')].copy()

# Merge with farm factors: spacing_D, aspect_ratio, n_turbines, axis_deviation
df_ana = gauss_real.merge(
    merged_farm[['farm_id_int', 'spacing_D', 'aspect_ratio', 'n_turbines', 'axis_deviation_abs', 'paradigm', 'country']],
    left_on='farm_id', right_on='farm_id_int', how='inner'
)
print(f'\n  合并后: {len(df_ana)} 条 ({df_ana["farm_id"].nunique()} farms × {df_ana["year"].nunique()} years)')

# 四因素分箱
df_ana['density_bin'] = pd.qcut(df_ana['spacing_D'], 3, labels=['密(<2.9D)', '中(2.9-4.3D)', '疏(>4.3D)'])
df_ana['aspect_bin']  = pd.qcut(df_ana['aspect_ratio'], 3, labels=['近方形', '中等', '狭长'])
df_ana['size_bin']    = pd.qcut(df_ana['n_turbines'], 3, labels=['小(<20)', '中(20-70)', '大(>70)'])
df_ana['dev_bin']     = pd.qcut(df_ana['axis_deviation_abs'], 3, labels=['小偏差', '中偏差', '大偏差'])

# --- 3a. 分层 ANOVA — 用真实的 FLORIS AEP/WakeLoss 值 ---
import statsmodels.api as sm
from statsmodels.formula.api import ols

print('\n[3a] 分层 ANOVA — WakeLoss (真实 FLORIS Gauss 输出)')
formula_wl = 'WakeLoss ~ C(density_bin) + C(aspect_bin) + C(size_bin) + C(dev_bin)'
model_wl = ols(formula_wl, data=df_ana).fit()
anova_wl = sm.stats.anova_lm(model_wl, typ=2)
print(anova_wl.to_string())
ss_wl = anova_wl['sum_sq']
for idx in anova_wl.index:
    if idx != 'Residual':
        eta2 = ss_wl[idx] / ss_wl.sum()
        print(f'  eta2({idx}) = {eta2:.4f} ({eta2*100:.1f}%)')

print('\n[3b] 分层 ANOVA — AEP (真实 FLORIS Gauss 输出)')
df_ana['AEP_GWh'] = df_ana['AEP_kWh'] / 1e9
formula_aep = 'AEP_GWh ~ C(density_bin) + C(aspect_bin) + C(size_bin) + C(dev_bin)'
model_aep = ols(formula_aep, data=df_ana).fit()
anova_aep = sm.stats.anova_lm(model_aep, typ=2)
print(anova_aep.to_string())
ss_aep = anova_aep['sum_sq']
for idx in anova_aep.index:
    if idx != 'Residual':
        eta2 = ss_aep[idx] / ss_aep.sum()
        print(f'  eta2({idx}) = {eta2:.4f} ({eta2*100:.1f}%)')

# --- 3b. 四因素对 WakeLoss 的组均值 ---
print('\n[3c] 各因素分组均值 (WakeLoss)')
for factor, col in [('密度(SpacingD)', 'density_bin'), ('构型(长宽比)', 'aspect_bin'),
                     ('规模(台数)', 'size_bin'), ('朝向偏差', 'dev_bin')]:
    grp = df_ana.groupby(col)['WakeLoss'].mean() * 100
    print(f'  {factor}:')
    for name, val in grp.items():
        print(f'    {name}: {val:.2f}%')

# --- 3c. 各因素对 AEP 的组均值 ---
print('\n[3d] 各因素分组均值 (AEP GWh)')
for factor, col in [('密度(SpacingD)', 'density_bin'), ('构型(长宽比)', 'aspect_bin'),
                     ('规模(台数)', 'size_bin'), ('朝向偏差', 'dev_bin')]:
    grp = df_ana.groupby(col)['AEP_GWh'].mean()
    print(f'  {factor}:')
    for name, val in grp.items():
        print(f'    {name}: {val:.2f} GWh')

# --- 3d. 偏 Eta 方汇总 ---
print('\n[3e] 四因素贡献汇总 (偏 eta2)')
summary = []
for idx in ['C(density_bin)', 'C(aspect_bin)', 'C(size_bin)', 'C(dev_bin)']:
    summary.append({
        'factor': idx.replace('C(', '').replace(')', ''),
        'eta2_WakeLoss': ss_wl[idx] / ss_wl.sum() if idx in ss_wl.index else 0,
        'eta2_AEP': ss_aep[idx] / ss_aep.sum() if idx in ss_aep.index else 0,
    })
df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'qiming_task3_factor_eta2.csv'), index=False)
print(df_summary.to_string(index=False))
print('\n  已保存: qiming_task3_factor_eta2.csv')

# Save decomposition data for reference
df_ana.to_csv(os.path.join(OUTPUT_DIR, 'qiming_task3_decomposition_data.csv'), index=False)
print('  已保存: qiming_task3_decomposition_data.csv')

# ═══════════════════════════════════════════════════════════════════════════
# 任务五(a): 扩建排布 — 等廷显交付 S2 管线 + 真实 UTM 坐标
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('任务五(a): 范式扩建排布生成')
print('  [状态: 等待廷显交付以下数据后开始]')
print('=' * 60)
print('')
print('  需要的数据:')
print('    1. S2 范式排布生成管线 (task3_paradigm_layouts.py 中的 generate_paradigm_layout 函数)')
print('    2. offshore-task0 turbine_coordinates.csv (真实风机 UTM 米制坐标)')
print('    3. offshore-task0 farms_master.csv (风场主表: 区域/国家/投运年份)')
print('')
print('  拿到后做:')
print('    a. 选取 6-10 个代表风场 (覆盖范式 A/B/C/E, 不同规模)')
print('    b. 按各自范式规则生成 2030/2035/2040 扩建排布 UTM 坐标')
print('    c. 保存为 paradigm_expansion_layouts.csv → 交给廷显跑 FLORIS')
print('')
print('  当前状态: 待廷显交付，暂不做。')

# ═══════════════════════════════════════════════════════════════════════════
# 制图 (仅已完成的任务)
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('生成图表')
print('=' * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.dpi': 150})

# --- 图1: SpacingD vs 5D 基准 ---
print('  [1/4] SpacingD vs 5D')
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.hist(lm['spacing_D'], bins=30, color='#3498db', alpha=0.7, edgecolor='white')
ax.axvline(x=5, color='red', linestyle='--', linewidth=2.5, label='W&P 5D 理想基准')
ax.axvline(x=lm['spacing_D'].median(), color='darkblue', linestyle='-', linewidth=2,
           label=f'真实中位数 = {lm["spacing_D"].median():.1f}D')
ax.set_xlabel('SpacingD (间距 / 叶轮直径)')
ax.set_ylabel('风场数量')
ax.set_title('171 场真实 SpacingD 分布 vs W&P 5D 基准')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

ax = axes[1]
p_order = ['A', 'B', 'C', 'E']
p_data = [lm[lm['paradigm']==p]['spacing_D'].dropna().values for p in p_order]
p_labels = [f'{p}\n(n={(lm["paradigm"]==p).sum()})' for p in p_order]
p_colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
bp = ax.boxplot(p_data, labels=p_labels, patch_artist=True,
                medianprops={'color': 'black', 'linewidth': 2})
for patch, c in zip(bp['boxes'], p_colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='W&P 5D 基准')
ax.set_ylabel('SpacingD'); ax.set_title('各建设范式的 SpacingD 分布')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

fig.suptitle('真实排布间距 vs W&P 5D 理想基准', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'qiming_fig1_spacing_vs_5D.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# --- 图2: 按国家 + 年份 ---
print('  [2/4] 按国家/年份')
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

top10 = merged_farm.groupby('country')['spacing_D'].median().sort_values().tail(10).index
cdata = [merged_farm[merged_farm['country']==c]['spacing_D'].dropna().values for c in top10]
axes[0].boxplot(cdata, labels=top10, patch_artist=True,
                medianprops={'color': 'black', 'linewidth': 1.5})
axes[0].axhline(y=5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].set_ylabel('SpacingD'); axes[0].set_title('按国家分组 (Top 10)')
axes[0].tick_params(axis='x', rotation=45); axes[0].grid(axis='y', alpha=0.3)

yr_med = lm.groupby('first_year')['spacing_D'].median()
yr_q25 = lm.groupby('first_year')['spacing_D'].quantile(0.25)
yr_q75 = lm.groupby('first_year')['spacing_D'].quantile(0.75)
axes[1].plot(yr_med.index, yr_med.values, 'o-', color='#e74c3c', linewidth=2.5, markersize=8)
axes[1].fill_between(yr_med.index, yr_q25, yr_q75, alpha=0.15, color='#e74c3c')
axes[1].axhline(y=5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='5D 基准')
axes[1].set_xlabel('首台投运年份'); axes[1].set_ylabel('SpacingD 中位数')
axes[1].set_title('SpacingD 年际趋势 (中位数 ± IQR)')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

fig.suptitle('SpacingD 按国家和投运年份分组', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'qiming_fig2_country_year.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# --- 图3: 构型特征 ---
print('  [3/4] 构型特征')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(lm['pc1_share'], bins=25, color='#9b59b6', alpha=0.7, edgecolor='white')
axes[0].axvline(x=0.8, color='green', linestyle='--', linewidth=1.5, label='强对齐 (>0.8)')
axes[0].axvline(x=0.6, color='orange', linestyle='--', linewidth=1.5, label='弱对齐 (<0.6)')
axes[0].set_xlabel('PC1 方差占比'); axes[0].set_ylabel('风场数')
axes[0].set_title('排布对齐度 (pc1_share)')
axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

ar = lm['aspect_ratio'].clip(0, 5)
axes[1].hist(ar, bins=30, color='#e67e22', alpha=0.7, edgecolor='white')
axes[1].axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, label='近方形 (<1.5)')
axes[1].set_xlabel('长宽比'); axes[1].set_ylabel('风场数')
axes[1].set_title('排布长宽比 (截断至 5)')
axes[1].legend(fontsize=8); axes[1].grid(axis='y', alpha=0.3)

lt_counts = lm['layout_type'].value_counts()
lt_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
axes[2].pie(lt_counts.values, labels=lt_counts.index, autopct='%1.1f%%',
            colors=lt_colors[:len(lt_counts)], explode=[0.03]*len(lt_counts))
axes[2].set_title('布局类型分布')

fig.suptitle('风场构型特征分析', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'qiming_fig3_configuration.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# --- 图4: 内因贡献条形图 (用真实 ANOVA eta2) ---
print('  [4/4] 内因贡献图')
fig, ax = plt.subplots(figsize=(9, 5.5))

factor_names = ['密度\n(SpacingD)', '构型\n(长宽比)', '规模\n(台数)', '朝向偏差\n(|dev|)']
wl_vals = [ss_wl.get(f'C(density_bin)', 0),
           ss_wl.get(f'C(aspect_bin)', 0),
           ss_wl.get(f'C(size_bin)', 0),
           ss_wl.get(f'C(dev_bin)', 0)]
wl_vals = np.array(wl_vals) / ss_wl.sum()

aep_vals = [ss_aep.get(f'C(density_bin)', 0),
            ss_aep.get(f'C(aspect_bin)', 0),
            ss_aep.get(f'C(size_bin)', 0),
            ss_aep.get(f'C(dev_bin)', 0)]
aep_vals = np.array(aep_vals) / ss_aep.sum()

x = np.arange(len(factor_names))
w = 0.35
bars1 = ax.bar(x - w/2, wl_vals, w, color='#e74c3c', edgecolor='white', label='对 WakeLoss 的贡献')
bars2 = ax.bar(x + w/2, aep_vals, w, color='#3498db', edgecolor='white', label='对 AEP 的贡献')

for bar, val in zip(bars1, wl_vals):
    if val > 0.001:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#c0392b')
for bar, val in zip(bars2, aep_vals):
    if val > 0.001:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#2471a3')

ax.set_xticks(x); ax.set_xticklabels(factor_names, fontsize=11)
ax.set_ylabel('方差贡献占比 (偏 η²)', fontsize=12)
ax.set_title('四因素对 WakeLoss 和 AEP 的方差贡献\n(分层 ANOVA, 真实 FLORIS Gauss 输出, n=1,203)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'qiming_fig4_factor_contribution.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('琪明任务完成 (任务一 + 任务三)')
print(f'任务五(a): 待廷显交付 S2 管线 + 真实 UTM 坐标后继续')
print(f'输出目录: {OUTPUT_DIR}/')
print(f'{"="*60}')

for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('qiming_'):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f'  {f} ({size/1024:.0f} KB)')
