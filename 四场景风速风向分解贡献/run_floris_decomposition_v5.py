"""
四情景风速—风向贡献分解 v5（最终版）
======================================
输入: 廷显 v5 FLORIS 直算 AEP（P11 从 S3 复制，P10 从 counterfactual 复制，
      P01/P00 为 FLORIS 库查找表 × 分位数映射）
输出: 任务书 §8 全部交付物
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats

REPO = r'd:\01学习资料\wind-direction-to-electricity-transition'
V5_CSV = os.path.join(REPO, '四场景风速风向分解贡献', 'four_scenario_floris_aep_v5.csv')
S3_CSV = os.path.join(REPO, 'task3', 'task3_s3_comparison.csv')
S1_CSV = os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv')
OUT = os.path.join(REPO, '补算', 'output')
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print(" 四情景分解 v5（最终版）")
print("=" * 60)

# ═══════════════════════════════════════════
# 0. 加载数据
# ═══════════════════════════════════════════
print('\n[1/5] 加载数据...')

# 廷显 v5 AEP
v5 = pd.read_csv(V5_CSV, encoding='utf-8-sig')
print(f'  v5 AEP: {len(v5)} rows, {v5.farm_id.nunique()} farms, {v5.year.min()}-{v5.year.max()}')

# S3 Gauss reference
s3 = pd.read_csv(S3_CSV, encoding='utf-8-sig')
s3_gauss = s3[(s3['wake_model'] == 'gauss') & (s3['layout_type'] == 'real')].copy()
print(f'  S3 Gauss: {len(s3_gauss)} rows')

# Merge
df = v5.merge(s3_gauss[['farm_id', 'year', 'AEP_kWh', 'AEP_noWake_kWh', 'CF', 'WakeLoss', 'country']],
              on=['farm_id', 'year'], how='left')

# P11 validation
df['P11_err_pct'] = abs(df['P11_kWh'] - df['AEP_kWh']) / df['AEP_kWh'] * 100

# Orientation gain
s1 = pd.read_csv(S1_CSV, encoding='utf-8-sig')
idx_opt = s1.groupby('farm_id')['expected_AEP_kWh'].idxmax()
theta_opt = s1.loc[idx_opt, ['farm_id', 'angle_deg']].copy()
theta_opt.columns = ['farm_id', 'theta_opt']

opt_all = s3[(s3['wake_model'] == 'gauss') & (s3['layout_type'].str.startswith('s1_opt'))].copy()
opt_all['angle'] = opt_all['layout_type'].str.extract(r's1_opt_(\d+)deg').astype(int)
opt_all = opt_all.merge(theta_opt, on='farm_id')
opt_best = opt_all[opt_all['angle'] == opt_all['theta_opt']][['farm_id', 'year', 'AEP_kWh']].copy()
opt_best.columns = ['farm_id', 'year', 'opt_AEP']
real_aep = s3_gauss[['farm_id', 'year', 'AEP_kWh']].copy()
real_aep.columns = ['farm_id', 'year', 'real_AEP']
gain_fy = real_aep.merge(opt_best, on=['farm_id', 'year'], how='inner')
gain_fy['gain_pct'] = (gain_fy['opt_AEP'] - gain_fy['real_AEP']) / gain_fy['real_AEP'] * 100
df = df.merge(gain_fy[['farm_id', 'year', 'gain_pct']], on=['farm_id', 'year'], how='left')

# ═══════════════════════════════════════════
# 1. S/D/I 分解
# ═══════════════════════════════════════════
print('\n[2/5] S/D/I 分解...')

P00, P10, P01, P11 = df['P00_kWh'].values, df['P10_kWh'].values, df['P01_kWh'].values, df['P11_kWh'].values

df['S_pct']   = (P10 - P00) / P11 * 100
df['D_pct']   = (P01 - P00) / P11 * 100
df['I_pct']   = (P11 - P10 - P01 + P00) / P11 * 100
df['total_pct'] = (P11 - P00) / P11 * 100
df['S_shapley'] = 50 * ((P10 - P00) + (P11 - P01)) / P11
df['D_shapley'] = 50 * ((P01 - P00) + (P11 - P10)) / P11

df['closure_err']    = df['S_pct'] + df['D_pct'] + df['I_pct'] - df['total_pct']
df['shapley_closure'] = df['S_shapley'] + df['D_shapley'] - df['total_pct']

print(f'  S+D+I closure: {df.closure_err.abs().mean():.2e} pct points')
print(f'  Shapley closure: {df.shapley_closure.abs().mean():.2e} pct points')

# ═══════════════════════════════════════════
# 2. 风场级汇总
# ═══════════════════════════════════════════
print('\n[3/5] 风场级汇总...')

farm_records = []
for fid, grp in df.groupby('farm_id'):
    n_yr = len(grp)
    G      = grp['gain_pct'].mean()
    G_ew   = (grp['gain_pct'] * grp['P11_kWh']).sum() / grp['P11_kWh'].sum()
    M_S_rms = np.sqrt((grp['S_shapley'] ** 2).mean())
    M_S_mae = grp['S_shapley'].abs().mean()
    M_S_std = grp['S_shapley'].std()
    M_S_mean = grp['S_shapley'].mean()
    M_D_rms = np.sqrt((grp['D_shapley'] ** 2).mean())
    M_D_mae = grp['D_shapley'].abs().mean()
    R_i = G / M_S_rms if M_S_rms > 0 else np.nan
    H_i = 1 if R_i > 1 else 0

    farm_records.append({
        'farm_id': fid, 'n_years': n_yr,
        'country': grp['country'].iloc[0] if 'country' in grp.columns else '',
        'G_mean': G, 'G_ew_mean': G_ew,
        'M_S_rms': M_S_rms, 'M_S_mae': M_S_mae, 'M_S_std': M_S_std, 'M_S_mean': M_S_mean,
        'M_D_rms': M_D_rms, 'M_D_mae': M_D_mae,
        'R_i': R_i, 'H_i': H_i,
        'P11_median_err': grp['P11_err_pct'].median(),
    })

df_farm = pd.DataFrame(farm_records)
df_main = df_farm[df_farm['n_years'] >= 5].copy()
M_S_global = df_main['M_S_rms'].median()

# ═══════════════════════════════════════════
# 3. 输出文件
# ═══════════════════════════════════════════
print('\n[4/5] 保存输出...')

# ① AEP 表
cols_aep = ['farm_id', 'year', 'country', 'P00_kWh', 'P10_kWh', 'P01_kWh', 'P11_kWh',
            'AEP_kWh', 'AEP_noWake_kWh', 'CF', 'WakeLoss', 'P11_err_pct']
df[cols_aep].to_csv(os.path.join(OUT, 'four_scenario_aep_farmyear.csv'), index=False, encoding='utf-8-sig')

# ② 效应表
cols_eff = ['farm_id', 'year', 'country', 'S_pct', 'D_pct', 'I_pct', 'total_pct',
            'S_shapley', 'D_shapley', 'gain_pct', 'P11_err_pct']
df[cols_eff].to_csv(os.path.join(OUT, 'four_scenario_effects_farmyear.csv'), index=False, encoding='utf-8-sig')

# ③ 农场汇总
df_farm.to_csv(os.path.join(OUT, 'four_scenario_farm_summary.csv'), index=False, encoding='utf-8-sig')

# ④ 阈值农场
th = df_farm[df_farm['n_years'] >= 5].copy()
th['exceeds_global_M_S'] = th['G_mean'] > M_S_global
th['exceeds_own_M_S'] = th['R_i'] > 1
th.to_csv(os.path.join(OUT, 'four_scenario_threshold_farms.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════
# 4. 结果摘要
# ═══════════════════════════════════════════
print(f'\n[5/5] 结果摘要')
print(f'{"=" * 60}')

print(f'\n--- 数据概况 ---')
print(f'  农场-年: {len(df)} ({df.farm_id.nunique()} farms, {df.year.min()}-{df.year.max()})')
print(f'  ≥5年进入主统计: {len(df_main)} farms')

print(f'\n--- P11 验收 ---')
print(f'  中位误差: {df.P11_err_pct.median():.3f}% (标准 ≤1.5%)')
print(f'  P95 误差: {df.P11_err_pct.quantile(0.95):.3f}% (标准 ≤3%)')
print(f'  样本内 ≤1.5%: {(df.P11_err_pct<=1.5).sum()}/{df.P11_err_pct.notna().sum()}')

print(f'\n--- 风速效应 M_S (Shapley RMS) ---')
print(f'  均值: {df_main.M_S_rms.mean():.2f}%')
print(f'  中位 (全球参考线): {M_S_global:.2f}%')
print(f'  P25-P75: [{df_main.M_S_rms.quantile(0.25):.2f}%, {df_main.M_S_rms.quantile(0.75):.2f}%]')
print(f'  最小-最大: [{df_main.M_S_rms.min():.2f}%, {df_main.M_S_rms.max():.2f}%]')

print(f'\n--- 风向效应 M_D (Shapley RMS) ---')
print(f'  均值: {df_main.M_D_rms.mean():.2f}%')
print(f'  中位: {df_main.M_D_rms.median():.2f}%')

print(f'\n--- 朝向收益 G ---')
print(f'  均值: {df_main.G_mean.mean():.2f}%')
print(f'  中位: {df_main.G_mean.median():.2f}%')
print(f'  P25-P75: [{df_main.G_mean.quantile(0.25):.2f}%, {df_main.G_mean.quantile(0.75):.2f}%]')

print(f'\n--- 交互项 |I| ---')
print(f'  中位: {df.I_pct.abs().median():.2f}%')
print(f'  P95: {df.I_pct.abs().quantile(0.95):.2f}%')

print(f'\n--- S 和 D 同号比例 ---')
print(f'  {(df.S_shapley * df.D_shapley > 0).mean() * 100:.1f}%')

print(f'\n--- R_i = G / M_S (orientation vs wind noise) ---')
exceed = df_main[df_main['R_i'] > 1].sort_values('R_i', ascending=False)
print(f'  R_i > 1: {len(exceed)}/{len(df_main)} farms')
for _, r in exceed.iterrows():
    print(f'    F{int(r.farm_id)} ({r.country}): G={r.G_mean:+.1f}%, M_S={r.M_S_rms:.1f}%, R={r.R_i:.2f}, n={int(r.n_years)}yr')

print(f'\n--- G > 全球 M_S ({M_S_global:.1f}%) ---')
exceed_g = df_main[df_main['G_mean'] > M_S_global].sort_values('G_mean', ascending=False)
print(f'  超过: {len(exceed_g)}/{len(df_main)} farms')
for _, r in exceed_g.iterrows():
    print(f'    F{int(r.farm_id)} ({r.country}): G={r.G_mean:+.1f}%, M_S={r.M_S_rms:.1f}%')

# S_shapley 分年分布
print(f'\n--- S_shapley 逐年分布 ---')
print(f'  均值: {df.S_shapley.mean():.2f}%')
print(f'  标准差: {df.S_shapley.std():.2f}%')
print(f'  范围: [{df.S_shapley.min():.2f}%, {df.S_shapley.max():.2f}%]')

print(f'\n所有文件已保存到 {OUT}/')
