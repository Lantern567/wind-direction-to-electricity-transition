"""
任务4: 乘积回归+置换检验(升级n=13四分位为正规统计)
任务8: 全球赌注核算(走廊内装机GW × 朝向对错TWh/年)
===============================================================
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm

BUSH = r'd:\01学习资料\wind-direction-to-electricity-transition\补算'
TASK3 = r'd:\01学习资料\wind-direction-to-electricity-transition\task3'
DATA = r'd:\01学习资料\wind-direction-to-electricity-transition\data'
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════
# 0. 加载数据
# ══════════════════════════════════════
print('0. 加载数据...')
s1 = pd.read_csv(os.path.join(TASK3, 'task3_s1_optimal_orientation.csv'))
s3 = pd.read_csv(os.path.join(TASK3, 'task3_s3_comparison.csv'))
wm = pd.read_csv(os.path.join(DATA, 'task1_output', 'task1_wind_metrics.csv'))
lm = pd.read_csv(r'C:\Users\beyqm\Desktop\北大实习\data\task2_output\layout_morphology.csv')
fm = pd.read_csv(os.path.join(DATA, 'task0', 'farms_master.csv'))
gain = pd.read_csv(os.path.join(TASK3, 'output', 'orientation_gain_all_farms.csv'))

# A = (max-mean)/mean
fa = s1.groupby('farm_id')['expected_AEP_kWh'].agg(['max','mean']).reset_index()
fa['A'] = (fa['max']-fa['mean'])/fa['mean']*100

# wake_pool
wp = s3[(s3['wake_model']=='gauss')&(s3['layout_type']=='real')].groupby('farm_id')['WakeLoss'].mean().reset_index()
wp.columns = ['farm_id','wake_pool']

# WCI
wci_f = wm.groupby('farm_id')['WCI_yearly'].mean().reset_index(); wci_f.columns=['farm_id','WCI']

# Geometry
lm['fid'] = lm['farm_id'].str.extract(r'farm_(\d+)').astype(int)
geo = lm[['fid','spacing_D','n_turbines','paradigm']].copy()
geo.columns = ['farm_id','spacing_D','n_turb','paradigm']

# Merge
df = fa.merge(wp,on='farm_id').merge(wci_f,on='farm_id').merge(geo,on='farm_id')
df = df.merge(gain[['farm_id','mean_gain']],on='farm_id').merge(fm[['farm_id','country','centroid_lat','centroid_lon','n_turb']],on='farm_id')
df = df.rename(columns={'centroid_lat':'lat','centroid_lon':'lon','n_turb_x':'n_turb'})
if 'n_turb_x' in df.columns: df['n_turb'] = df['n_turb_x']
df['WCI_x_pool'] = df['WCI'] * df['wake_pool']
df = df.dropna(subset=['WCI','wake_pool','mean_gain','A'])
print(f'  有效农场: {len(df)}')

# ══════════════════════════════════════
# 任务4: 乘积回归 + 置换检验
# ══════════════════════════════════════
print('\n' + '='*60)
print('任务4: 乘积回归 + 置换检验')
print('='*60)

# 4a. 描述统计: WCI×wake_pool 四分位
print('\n4a. 四分位验证...')
q_wci = df['WCI'].quantile(0.75)
q_pool = df['wake_pool'].quantile(0.75)
hi_both = df[(df['WCI']>q_wci)&(df['wake_pool']>q_pool)]
print(f'  WCI top 25%: >{q_wci:.3f}, wake_pool top 25%: >{q_pool:.3f}')
print(f'  双高组: n={len(hi_both)}, mean_gain={hi_both["mean_gain"].mean():.2f}%')
print(f'  其余: n={len(df)-len(hi_both)}, mean_gain={df[~df.index.isin(hi_both.index)]["mean_gain"].mean():.2f}%')
print(f'  倍数: {hi_both["mean_gain"].mean()/df[~df.index.isin(hi_both.index)]["mean_gain"].mean():.1f}x')

# 4b. 乘积回归: gain ~ WCI + wake_pool + WCI×wake_pool
print('\n4b. 乘积回归...')
X = sm.add_constant(df[['WCI','wake_pool','WCI_x_pool']])
for ycol, ylabel in [('mean_gain','gain_pct'), ('A','A')]:
    y = df[ycol]
    m = sm.OLS(y, X).fit()
    interaction_p = m.pvalues['WCI_x_pool']
    print(f'\n  {ylabel} ~ WCI + pool + WCI×pool:')
    print(f'    WCI:          coef={m.params["WCI"]:+.3f}, p={m.pvalues["WCI"]:.3f}')
    print(f'    wake_pool:    coef={m.params["wake_pool"]:+.3f}, p={m.pvalues["wake_pool"]:.3f}')
    print(f'    WCI×pool:     coef={m.params["WCI_x_pool"]:+.3f}, p={interaction_p:.3f}')
    print(f'    R²={m.rsquared:.3f}')

# 4c. 置换检验 (10,000 permutations of WCI_x_pool)
print('\n4c. 置换检验 (B=10,000)...')
B = 10000
y_g = df['mean_gain'].values
X_base = sm.add_constant(df[['WCI','wake_pool']].values)
obs_interaction = m.params['WCI_x_pool']  # from gain regression above

perm_coefs = np.zeros(B)
interaction_col = df['WCI_x_pool'].values
np.random.seed(42)
for b in range(B):
    X_perm = np.column_stack([X_base, np.random.permutation(interaction_col)])
    perm_coefs[b] = sm.OLS(y_g, X_perm).fit().params[3]

perm_p = (np.abs(perm_coefs) >= np.abs(obs_interaction)).mean()
print(f'  Observed WCI×pool coef: {obs_interaction:+.3f}')
print(f'  Permutation p (two-sided): {perm_p:.4f}')
print(f'  置换分布: mean={perm_coefs.mean():.3f}, std={perm_coefs.std():.3f}, '
      f'CI95=[{np.percentile(perm_coefs,2.5):.3f},{np.percentile(perm_coefs,97.5):.3f}]')

# 4d. Leave-one-corridor-out
print('\n4d. Leave-one-corridor-out...')
corridors = {
    'Vietnam': [56,57,64,86,107,112,115,126,130,133,141,143,159],
    'China_strait': [66,91,12,92,97,85,103,105],
    'Italy': [155], 'Denmark': [157],
}
farm_cor = {}
for cn, fids in corridors.items():
    for fid in fids: farm_cor[fid] = cn
df['corridor'] = df['farm_id'].map(farm_cor).fillna('other')

for cn in corridors:
    tr = df[df['corridor']!=cn]; te = df[df['corridor']==cn]
    if len(te)==0 or len(tr)<20: continue
    X_tr = sm.add_constant(tr[['WCI','wake_pool','WCI_x_pool']])
    m_tr = sm.OLS(tr['A'], X_tr).fit()
    X_te = sm.add_constant(te[['WCI','wake_pool','WCI_x_pool']], has_constant='add')
    pred = m_tr.predict(X_te)
    r_c = np.corrcoef(te['A'], pred)[0,1] if len(te)>=3 else np.nan
    print(f'  {cn}: n_test={len(te)}, corr(A,pred)={r_c:.3f}, '
          f'mean_A={te["A"].mean():.1f}%, pred_mean={pred.mean():.1f}%')

# ══════════════════════════════════════
# 任务8: 全球赌注核算
# ══════════════════════════════════════
print('\n' + '='*60)
print('任务8: 全球赌注核算')
print('='*60)

# Simplified: use known offshore wind capacity by country (GWEC 2024 estimates)
# and overlay with corridor classification
# Data: country-level planned offshore wind capacity by 2030/2035
planned_cap = {
    'China': 80, 'United Kingdom': 45, 'Germany': 30, 'Netherlands': 22,
    'Denmark': 13, 'Vietnam': 10, 'South Korea': 14, 'Japan': 8, 'Taiwan': 15,
    'United States of America': 25, 'France': 5, 'Ireland': 7, 'Sweden': 3,
    'Italy': 3, 'Portugal': 2, 'Belgium': 6, 'Finland': 2,
}
# Average A per country (from training data)
country_A = df.groupby('country')['A'].mean()

# Classify countries as high/mid/low corridor
high_corridor = ['Vietnam', 'China', 'Taiwan', 'Italy']  # from Fig 4 analysis
mid_corridor = ['United Kingdom', 'Denmark', 'Germany', 'Ireland', 'South Korea', 'Japan']

print('\n  走廊国家 × 计划装机:')
total_GW = 0; at_risk_GWh = 0
records = []
for ctry, gw in planned_cap.items():
    A_val = country_A.get(ctry, country_A.median())
    if ctry in high_corridor: tier = 'high'
    elif ctry in mid_corridor: tier = 'mid'
    else: tier = 'low'

    # TWh at risk = GW × 8760h × avg_CF × A
    # avg_CF from data
    cf_val = df[df['country']==ctry]['mean_gain'].mean() * 0 + 0.4  # approximation
    if len(df[df['country']==ctry]) > 0:
        # Use actual mean gain as rough CF proxy * A/100
        pass
    annual_GWh = gw * 8760 * 0.40 * A_val / 100
    records.append({'country':ctry,'planned_GW':gw,'mean_A_pct':round(A_val,1),
                    'corridor_tier':tier,'est_at_risk_GWh':round(annual_GWh,0)})
    if tier in ('high','mid'):
        total_GW += gw
        at_risk_GWh += annual_GWh

rk = pd.DataFrame(records).sort_values('planned_GW',ascending=False)
print(rk.to_string(index=False))

print(f'\n  高+中走廊计划装机: {total_GW:.0f} GW')
print(f'  朝向选择差错年损失: {at_risk_GWh/1000:.1f} TWh/年')
print(f'  以$50/MWh计, 年经济损失: ${at_risk_GWh*50/1e6:.2f}B')

# 8b. Vietnam-specific
vn_gw = planned_cap.get('Vietnam', 0)
vn_A_val = country_A.get('Vietnam', 15)
vn_GWh = vn_gw * 8760 * 0.40 * vn_A_val / 100
print(f'\n  Vietnam corridor: {vn_gw} GW planned, A={vn_A_val:.1f}%')
print(f'  年损失: {vn_GWh/1000:.1f} TWh (若朝向全部选错)')
print(f'  Vietnam A={vn_A_val:.1f}%, 年损失={vn_GWh/1000:.1f} TWh (若全错)')

# Save
rk.to_csv(os.path.join(OUT, 'task8_global_bet.csv'), index=False, encoding='utf-8-sig')
print('\n产出: task8_global_bet.csv')
print('任务4+8 完成.')
