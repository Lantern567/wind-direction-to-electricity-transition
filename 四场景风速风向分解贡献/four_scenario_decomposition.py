"""
四情景风速—风向贡献分解
=========================
Stage 1: 每场-年构造四情景概率表
Stage 2: AEP = Σ p(d,v) × p_noWake × we × n_hours
Stage 3: S/D/I 分解 + Shapley + 朝向收益比较
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

BUSH = r'd:\01学习资料\wind-direction-to-electricity-transition\补算'
TASK3 = r'd:\01学习资料\wind-direction-to-electricity-transition\task3'
DATA = r'd:\01学习资料\wind-direction-to-electricity-transition\data'
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

# ═════════════════════════════════════════════
# 0. 加载数据
# ═════════════════════════════════════════════
print('0. Loading data...')
lt = pd.read_csv(os.path.join(OUT, 'four_scenario_lookup_table.csv'))
cm = pd.read_csv(os.path.join(OUT, 'four_scenario_config_map.csv'))
s3 = pd.read_csv(os.path.join(TASK3, 'task3_s3_comparison.csv'))
lm = pd.read_csv(r'C:\Users\beyqm\Desktop\北大实习\data\task2_output\layout_morphology.csv')
lm['fid'] = lm['farm_id'].str.extract(r'farm_(\d+)').astype(int)

# ERA5 parquet
print('  Loading ERA5...')
era5_files = [os.path.join(DATA,'era5',f) for f in os.listdir(os.path.join(DATA,'era5')) if f.endswith('.parquet')]
era5 = pd.concat([pd.read_parquet(f) for f in era5_files], ignore_index=True)
era5 = era5.drop_duplicates(subset=['farm_id','year','month','day','hour'])
era5['V_ms'] = era5['V_ms'].astype(np.float32)
print(f'  ERA5: {len(era5):,} rows (deduped), {era5.farm_id.nunique()} farms')

# Bin definitions (matching lookup table)
WS_BINS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30]
WS_CENTERS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30]
WD_BINS = np.arange(0, 370, 10)
WD_CENTERS = list(range(0, 360, 10))
N_WS = len(WS_BINS)
N_WD = len(WD_CENTERS)

# Lookup table: build dict config_hash -> (pnw_mat[36,19], we_mat[36,19])
print('  Building lookup dict...')
lookup = {}
for ch, grp in lt.groupby('config_hash'):
    pnw = np.zeros((N_WD, N_WS))
    we = np.zeros((N_WD, N_WS))
    for _, r in grp.iterrows():
        wi = WS_CENTERS.index(int(r['ws_bin_m_s'])) if int(r['ws_bin_m_s']) in WS_CENTERS else -1
        di = WD_CENTERS.index(int(r['wd_sector_deg']))
        if wi >= 0:
            pnw[di, wi] = r['p_noWake_kW']
            we[di, wi] = r['wake_efficiency']
    n_turb = grp['n_turbines'].values[0]
    lookup[ch] = (pnw, we, n_turb)
print(f'  Lookup entries: {len(lookup)}')

# Country + region for validation
fc = s3.groupby('farm_id')['country'].first()
cm['country'] = cm['farm_id'].map(fc)
cm = cm.merge(lm[['fid', 'region']], left_on='farm_id', right_on='fid', how='left')

# ═════════════════════════════════════════════
# 1. 构建四情景概率表 + 计算 AEP
# ═════════════════════════════════════════════
print('\n1. Computing four-scenario AEP...')
# For historical baseline: pool all years per farm from ERA5 parquet
# Bin ERA5 data
print('  Binning ERA5 data...')
ws_edges = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30, 100]
wd_edges = np.arange(0, 370, 10)
era5['ws_i'] = pd.cut(era5['V_ms'], bins=ws_edges, labels=False, right=False)
era5['wd_i'] = pd.cut(era5['theta_deg'], bins=wd_edges, labels=False, right=False)
era5 = era5.dropna(subset=['ws_i', 'wd_i'])
era5['ws_i'] = era5['ws_i'].astype(int)
era5['wd_i'] = era5['wd_i'].astype(int)

# Pre-compute historical (all-year pooled) & per-year joint distributions
farms = sorted(cm['farm_id'].unique())
years_by_farm = cm.groupby('farm_id')['year'].apply(list).to_dict()

# Joint freq: p_hist(d,v) = all-year average
# We need p(d) × p(v|d), not p(d,v) directly for four-scenario mixing

# Pre-build historical distributions
print('  Building historical distributions...')
hist_dist = {}  # farm_id -> {p_d[36], p_v_given_d[36,19]}
for fid in sorted(farms):
    farm_data = era5[era5['farm_id'] == fid]
    if len(farm_data) < 1000:
        continue
    # p(d): marginal direction frequency
    d_counts = farm_data.groupby('wd_i').size()
    p_d = np.zeros(36)
    for di, c in d_counts.items():
        if 0 <= di < 36:
            p_d[di] = c
    p_d = p_d / p_d.sum()

    # p(v|d): conditional wind speed given direction
    p_v_given_d = np.zeros((36, 19))
    for di in range(36):
        di_data = farm_data[farm_data['wd_i'] == di]
        if len(di_data) > 30:
            v_counts = di_data.groupby('ws_i').size()
            for wi, c in v_counts.items():
                if 0 <= wi < 19:
                    p_v_given_d[di, wi] = c
            p_v_given_d[di] = p_v_given_d[di] / p_v_given_d[di].sum()
        else:
            # Sparse: use marginal
            v_counts = farm_data.groupby('ws_i').size()
            p_v_marg = np.zeros(19)
            for wi, c in v_counts.items():
                if 0 <= wi < 19:
                    p_v_marg[wi] = c
            p_v_marg = p_v_marg / p_v_marg.sum()
            p_v_given_d[di] = p_v_marg.copy()

    hist_dist[fid] = {'p_d': p_d, 'p_v_given_d': p_v_given_d}

# Per-year distributions
print('  Building per-year distributions...')
year_dist = {}  # (farm_id, year) -> {p_d[36], p_v_given_d[36,19]}
for fid in sorted(farms):
    for yr in years_by_farm.get(fid, []):
        farm_yr = era5[(era5['farm_id'] == fid) & (era5['year'] == yr)]
        if len(farm_yr) < 100:
            continue

        d_counts = farm_yr.groupby('wd_i').size()
        p_d = np.zeros(36)
        for di, c in d_counts.items():
            if 0 <= di < 36:
                p_d[di] = c
        p_d = p_d / p_d.sum()

        p_v_given_d = np.zeros((36, 19))
        for di in range(36):
            di_data = farm_yr[farm_yr['wd_i'] == di]
            N_di = len(di_data)
            if N_di > 30:
                v_counts = di_data.groupby('ws_i').size()
                for wi, c in v_counts.items():
                    if 0 <= wi < 19:
                        p_v_given_d[di, wi] = c
                p_v_given_d[di] = p_v_given_d[di] / p_v_given_d[di].sum()
            else:
                # Shrinkage toward marginal
                v_marg = np.zeros(19)
                vc = farm_yr.groupby('ws_i').size()
                for wi, c in vc.items():
                    if 0 <= wi < 19:
                        v_marg[wi] = c
                v_marg = v_marg / v_marg.sum()
                lam = N_di / (N_di + 30)
                if p_v_given_d[di].sum() > 0:
                    p_v_given_d[di] = lam * p_v_given_d[di] / p_v_given_d[di].sum() + (1 - lam) * v_marg
                else:
                    p_v_given_d[di] = v_marg.copy()

        year_dist[(fid, yr)] = {'p_d': p_d, 'p_v_given_d': p_v_given_d}

print(f'  Historical: {len(hist_dist)} farms, Per-year: {len(year_dist)} farm-years')

# ═════════════════════════════════════════════
# 2. 计算四情景 AEP
# ═════════════════════════════════════════════
print('\n2. Computing scenario AEP...')

records = []
processed = 0
N_HOURS = 8760  # standardized

for _, row in cm.iterrows():
    fid = row['farm_id']
    yr = row['year']
    ch = row['config_hash']
    if ch not in lookup:
        continue
    if fid not in hist_dist or (fid, yr) not in year_dist:
        continue

    pnw, we, n_turb = lookup[ch]
    h = hist_dist[fid]
    y = year_dist[(fid, yr)]

    # Total hours for this farm-year
    farm_yr_era5 = era5[(era5['farm_id'] == fid) & (era5['year'] == yr)]
    n_h = len(farm_yr_era5)
    if n_h < 100:
        continue

    # Four scenarios: P_ab = n_h × Σ_d Σ_v p_ab(d,v) × pnw(d,v) × we(d,v)
    # where p_ab(d,v) = p_ab(d) × p_ab(v|d)
    def compute_AEP(p_d_scenario, p_v_given_d_scenario):
        total = 0.0
        for di in range(36):
            for wi in range(19):
                prob = p_d_scenario[di] * p_v_given_d_scenario[di, wi]
                total += prob * pnw[di, wi] * we[di, wi]
        return total * n_h

    def compute_noWake(p_d_scenario, p_v_given_d_scenario):
        total = 0.0
        for di in range(36):
            for wi in range(19):
                prob = p_d_scenario[di] * p_v_given_d_scenario[di, wi]
                total += prob * pnw[di, wi]
        return total * n_h

    # P00: hist WS, hist WD
    P00 = compute_AEP(h['p_d'], h['p_v_given_d'])
    P00_nw = compute_noWake(h['p_d'], h['p_v_given_d'])

    # P10: actual WS, hist WD
    P10 = compute_AEP(h['p_d'], y['p_v_given_d'])
    P10_nw = compute_noWake(h['p_d'], y['p_v_given_d'])

    # P01: hist WS, actual WD
    P01 = compute_AEP(y['p_d'], h['p_v_given_d'])
    P01_nw = compute_noWake(y['p_d'], h['p_v_given_d'])

    # P11: actual WS, actual WD
    P11 = compute_AEP(y['p_d'], y['p_v_given_d'])
    P11_nw = compute_noWake(y['p_d'], y['p_v_given_d'])

    records.append({
        'farm_id': fid, 'year': yr, 'config_hash': ch, 'n_turb': n_turb,
        'n_hours': n_h, 'country': row.get('country', ''), 'region': row.get('region', ''),
        'P00_kWh': P00, 'P10_kWh': P10, 'P01_kWh': P01, 'P11_kWh': P11,
        'P00_noWake': P00_nw, 'P11_noWake': P11_nw,
    })

    processed += 1
    if processed % 100 == 0:
        print(f'  Processed {processed} farm-years...')

df_aep = pd.DataFrame(records)
print(f'  Total: {len(df_aep)} farm-years')

# ═════════════════════════════════════════════
# 3. 效应分解
# ═════════════════════════════════════════════
print('\n3. Decomposition...')

# S = (P10 - P00) / P11 * 100  (wind speed effect, %)
# D = (P01 - P00) / P11 * 100  (wind direction effect, %)
# I = (P11 - P10 - P01 + P00) / P11 * 100  (interaction, %)
# Total = (P11 - P00) / P11 * 100 = S + D + I

df_aep['S_pct'] = (df_aep['P10_kWh'] - df_aep['P00_kWh']) / df_aep['P11_kWh'] * 100
df_aep['D_pct'] = (df_aep['P01_kWh'] - df_aep['P00_kWh']) / df_aep['P11_kWh'] * 100
df_aep['I_pct'] = (df_aep['P11_kWh'] - df_aep['P10_kWh'] - df_aep['P01_kWh'] + df_aep['P00_kWh']) / df_aep['P11_kWh'] * 100
df_aep['total_pct'] = (df_aep['P11_kWh'] - df_aep['P00_kWh']) / df_aep['P11_kWh'] * 100

# Shapley
df_aep['S_shapley'] = 50 * ((df_aep['P10_kWh'] - df_aep['P00_kWh']) + (df_aep['P11_kWh'] - df_aep['P01_kWh'])) / df_aep['P11_kWh']
df_aep['D_shapley'] = 50 * ((df_aep['P01_kWh'] - df_aep['P00_kWh']) + (df_aep['P11_kWh'] - df_aep['P10_kWh'])) / df_aep['P11_kWh']

# Verify closure
df_aep['closure_err'] = df_aep['S_pct'] + df_aep['D_pct'] + df_aep['I_pct'] - df_aep['total_pct']
df_aep['shapley_closure'] = df_aep['S_shapley'] + df_aep['D_shapley'] - df_aep['total_pct']
print(f'  Closure err (S+D+I vs total): mean={df_aep.closure_err.abs().mean():.2e} pct')
print(f'  Shapley closure err: mean={df_aep.shapley_closure.abs().mean():.2e} pct')

# ═════════════════════════════════════════════
# 4. 风场级汇总 + 朝向收益比较
# ═════════════════════════════════════════════
print('\n4. Farm-level summary...')

# Per-farm orientation gain (from S3)
s3r = s3[(s3['wake_model']=='gauss') & (s3['layout_type']=='real')]
real = s3r[['farm_id','year','AEP_kWh']].copy(); real.columns = ['farm_id','year','real_AEP']
opt_all = s3[(s3['wake_model']=='gauss') & (s3['layout_type'].str.startswith('s1_opt'))].copy()
opt_all['angle'] = opt_all['layout_type'].str.extract(r's1_opt_(\d+)deg').astype(int)
s1 = pd.read_csv(os.path.join(TASK3, 'task3_s1_optimal_orientation.csv'))
idx_opt = s1.groupby('farm_id')['expected_AEP_kWh'].idxmax()
theta_opt = s1.loc[idx_opt, ['farm_id','angle_deg']]; theta_opt.columns = ['farm_id','theta_opt']
opt_all = opt_all.merge(theta_opt, on='farm_id')
opt_best = opt_all[opt_all['angle'] == opt_all['theta_opt']][['farm_id','year','AEP_kWh']].copy()
opt_best.columns = ['farm_id','year','opt_AEP']
gain_fy = real.merge(opt_best, on=['farm_id','year'], how='inner')
gain_fy['gain_pct'] = (gain_fy['opt_AEP'] - gain_fy['real_AEP']) / gain_fy['real_AEP'] * 100

# Merge with decomposition
df_aep = df_aep.merge(gain_fy[['farm_id','year','gain_pct']], on=['farm_id','year'], how='left')

# Farm-level: RMS wind speed impact vs mean orientation gain
farm_summary = []
for fid, grp in df_aep.groupby('farm_id'):
    n_yr = len(grp)
    G = grp['gain_pct'].mean()  # orientation gain (mean)
    M_S = np.sqrt((grp['S_shapley']**2).mean())  # RMS wind speed impact
    M_D = np.sqrt((grp['D_shapley']**2).mean())  # RMS wind direction impact
    R_i = G / M_S if M_S > 0 else np.nan
    farm_summary.append({
        'farm_id': fid, 'n_years': n_yr,
        'G_mean': G, 'M_S_rms': M_S, 'M_D_rms': M_D,
        'R_i': R_i, 'H_i': 1 if R_i > 1 else 0,
        'country': grp['country'].iloc[0], 'region': grp['region'].iloc[0],
    })

df_farm = pd.DataFrame(farm_summary)
# Filter: at least 5 years
df_farm_main = df_farm[df_farm['n_years'] >= 5].copy()

# Global reference: median RMS wind impact
M_S_global = df_farm_main['M_S_rms'].median()
print(f'  Global median M_S (RMS wind impact): {M_S_global:.2f}%')
n_ri = (df_farm_main['R_i'] > 1).sum()
n_ms = (df_farm_main['G_mean'] > M_S_global).sum()
print(f'  Farms with R_i > 1 (orientation > wind): {n_ri}/{len(df_farm_main)}')
print(f'  Farms exceeding global M_S: {n_ms}/{len(df_farm_main)}')

# ================================================
# 5. Save outputs
# ================================================
print('\n5. Saving...')

# 5a. Farm-year table
df_aep.to_csv(os.path.join(OUT, 'four_scenario_aep_farmyear.csv'), index=False, encoding='utf-8-sig')

# 5b. Effects per farm-year
df_effects = df_aep[['farm_id','year','country','region','S_pct','D_pct','I_pct','total_pct','S_shapley','D_shapley','gain_pct','n_hours']]
df_effects.to_csv(os.path.join(OUT, 'four_scenario_effects_farmyear.csv'), index=False, encoding='utf-8-sig')

# 5c. Farm summary
df_farm.to_csv(os.path.join(OUT, 'four_scenario_farm_summary.csv'), index=False, encoding='utf-8-sig')

# 5d. Threshold farms
threshold_farms = df_farm[df_farm['n_years'] >= 5].copy()
threshold_farms['exceeds_global_M_S'] = threshold_farms['G_mean'] > M_S_global
threshold_farms['exceeds_own_M_S'] = threshold_farms['R_i'] > 1
threshold_farms.to_csv(os.path.join(OUT, 'four_scenario_threshold_farms.csv'), index=False, encoding='utf-8-sig')

# ================================================
# 6. Summary printout
# ================================================
print(f'\n{"="*60}')
print(f'RESULTS SUMMARY')
print(f'{"="*60}')
print(f'Farm-years computed: {len(df_aep)}')
print(f'Farms with >=5 years: {len(df_farm_main)}')
print(f'\n--- Wind speed effect (Shapley, RMS) ---')
print(f'  Mean: {df_farm_main.M_S_rms.mean():.2f}%')
print(f'  Median (global ref): {M_S_global:.2f}%')
print(f'  P25-P75: [{df_farm_main.M_S_rms.quantile(0.25):.2f}%, {df_farm_main.M_S_rms.quantile(0.75):.2f}%]')
print(f'\n--- Wind direction effect (Shapley, RMS) ---')
print(f'  Mean: {df_farm_main.M_D_rms.mean():.2f}%')
print(f'  Median: {df_farm_main.M_D_rms.median():.2f}%')
print(f'\n--- Orientation gain G ---')
print(f'  Mean: {df_farm_main.G_mean.mean():.2f}%')
print(f'  Median: {df_farm_main.G_mean.median():.2f}%')
print(f'\n--- R_i = G / M_S ---')
exceed = df_farm_main[df_farm_main['R_i']>1]
print(f'  Farms where G > self M_S: {len(exceed)}/{len(df_farm_main)}')
if len(exceed) > 0:
    print(f'  Top R_i farms:')
    for _,r in exceed.nlargest(5,'R_i').iterrows():
        print(f'    F{int(r.farm_id)} ({r.country}): G={r.G_mean:+.1f}%, M_S={r.M_S_rms:.1f}%, R={r.R_i:.1f}')
print(f'\nFiles saved to {OUT}/')
