"""
四情景风速—风向贡献分解 v6（派生表重算，对应学长重算清单 P0-1）
================================================================
与 run_floris_decomposition_v5.py 完全相同的分解逻辑（S/D/I 与 Shapley、
风场级汇总、越线判定），唯一区别：输入从廷显 v5 FLORIS AEP 换为
廷显 v6 FLORIS 逐时精确 AEP（P10 全部 FLORIS 逐时自算、P01/P00 逐时
精确功率插值，即 v5→v6 的修复 1/2 已进入）。

用途：裁决"逐场越线是 5 个还是 4 个"（P0-1）。
输出：output/four_scenario_farm_summary_v6.csv 及
      output/four_scenario_farm_summary_AUTHORITATIVE.csv（权威表落盘）。
不覆盖 v5 派生的四张旧表。
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)

V6_CSV = os.path.join(SCRIPT_DIR, 'output', 'four_scenario_floris_aep_v6.csv')
S3_CSV = os.path.join(REPO, 'task3', 'task3_s3_comparison.csv')
S1_CSV = os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv')
OUT = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT, exist_ok=True)

for _f, _label in [(V6_CSV, 'v6 AEP CSV'), (S3_CSV, 'S3 comparison'), (S1_CSV, 'S1 optimal orientation')]:
    if not os.path.exists(_f):
        raise FileNotFoundError(f'{_label} not found: {_f}')
    print(f'  ✓ {_label}: {os.path.basename(_f)}')

print('=' * 60)
print(' 四情景分解 v6 派生表重算（P0-1 裁决）')
print('=' * 60)

# ── 0. 加载 ──
v6 = pd.read_csv(V6_CSV, encoding='utf-8-sig')
print(f'  v6 AEP: {len(v6)} rows, {v6.farm_id.nunique()} farms, {v6.year.min()}-{v6.year.max()}')

s3 = pd.read_csv(S3_CSV, encoding='utf-8-sig')
s3_gauss = s3[(s3['wake_model'] == 'gauss') & (s3['layout_type'] == 'real')].copy()
print(f'  S3 Gauss: {len(s3_gauss)} rows')

df = v6.merge(s3_gauss[['farm_id', 'year', 'AEP_kWh', 'AEP_noWake_kWh', 'CF', 'WakeLoss', 'country']],
              on=['farm_id', 'year'], how='left')
df['P11_err_pct'] = abs(df['P11_kWh'] - df['AEP_kWh']) / df['AEP_kWh'] * 100

# 朝向增益（与 v5 逻辑一致）
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
print(f'  gain_pct 覆盖: {df.gain_pct.notna().sum()}/{len(df)}')

# ── 1. S/D/I 分解（与 v5 完全相同） ──
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

# ── 2. 风场级汇总（与 v5 完全相同） ──
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

# ── 3. 输出（v6 派生表 + 权威表） ──
cols_aep = ['farm_id', 'year', 'country', 'P00_kWh', 'P10_kWh', 'P01_kWh', 'P11_kWh',
            'AEP_kWh', 'AEP_noWake_kWh', 'CF', 'WakeLoss', 'P11_err_pct']
df[cols_aep].to_csv(os.path.join(OUT, 'four_scenario_aep_farmyear_v6.csv'), index=False, encoding='utf-8-sig')
cols_eff = ['farm_id', 'year', 'country', 'S_pct', 'D_pct', 'I_pct', 'total_pct',
            'S_shapley', 'D_shapley', 'gain_pct', 'P11_err_pct']
df[cols_eff].to_csv(os.path.join(OUT, 'four_scenario_effects_farmyear_v6.csv'), index=False, encoding='utf-8-sig')
df_farm.to_csv(os.path.join(OUT, 'four_scenario_farm_summary_v6.csv'), index=False, encoding='utf-8-sig')
df_farm.to_csv(os.path.join(OUT, 'four_scenario_farm_summary_AUTHORITATIVE.csv'), index=False, encoding='utf-8-sig')

# ── 4. 越线判定与打印 ──
print('\n[越线判定] 判据 ratio = G_mean(%) / M_S_std(%) >= 1')
for minyr, tag in [(5, '≥5 年'), (3, '≥3 年')]:
    sub = df_farm[df_farm.n_years >= minyr].copy()
    sub['ratio'] = sub['G_mean'] / sub['M_S_std']
    cross = sub[sub.ratio >= 1].sort_values('ratio')
    print(f'  {tag} 越线: {len(cross)} 个 → {list(cross.farm_id.astype(int))}')
    for r in cross.itertuples():
        print(f'    F{int(r.farm_id)} ({r.country}): n={int(r.n_years)}, G={r.G_mean:.3f}%, '
              f'M_S_std={r.M_S_std:.3f}%, ratio={r.ratio:.2f}')
    if len(cross):
        print(f'    比值区间: {cross.ratio.min():.2f}–{cross.ratio.max():.2f}')

print(f'\n  全局: G 中位 {df_main.G_mean.median():.3f}% / 均值 {df_main.G_mean.mean():.3f}%')
print(f'  越线场 n_years≥5 的 H_i 计数: {int(df_main.H_i.sum())}')

# ── 5. 与 v5 派生表对比 ──
v5_path = os.path.join(OUT, 'four_scenario_farm_summary.csv')
if os.path.exists(v5_path):
    v5f = pd.read_csv(v5_path, encoding='utf-8-sig')
    m = df_farm.merge(v5f[['farm_id', 'G_mean', 'M_S_std']], on='farm_id', suffixes=('_v6', '_v5'))
    print('\n[v5→v6 关键场对比]')
    for fid in [57, 66, 91, 155, 157, 160]:
        r = m[m.farm_id == fid].iloc[0]
        print(f"  F{fid}: G {r.G_mean_v5:+.3f}→{r.G_mean_v6:+.3f}%, "
              f"M_S_std {r.M_S_std_v5:.3f}→{r.M_S_std_v6:.3f}%, "
              f"ratio {r.G_mean_v5/r.M_S_std_v5:.2f}→{r.G_mean_v6/r.M_S_std_v6:.2f}")
    print(f'  M_S_std 中位变化(≥5年): {v5f[v5f.n_years>=5].M_S_std.median():.3f}→{df_main.M_S_std.median():.3f}')

print('\n已完成。权威表:', os.path.join(OUT, 'four_scenario_farm_summary_AUTHORITATIVE.csv'))
