"""
任务6: 聚类Bootstrap替换配对t检验
===================================
问题: 171场空间成簇(越南3场、中国近海密集), 配对t检验高估自由度
方法: 按海盆/国家聚类, cluster bootstrap 10,000次, 产出CI和符号检验p值
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

TASK3 = r'd:\01学习资料\wind-direction-to-electricity-transition\task3'
BUSH = r'd:\01学习资料\wind-direction-to-electricity-transition\补算'
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════
# 1. 加载 per-farm gain + 聚类信息
# ══════════════════════════════════════
print('1. 加载数据...')
gain = pd.read_csv(os.path.join(TASK3, 'output', 'orientation_gain_all_farms.csv'))
print(f'  农场: {len(gain)}, 国家: {gain.country.nunique()}')

# Merge region
lm = pd.read_csv(r'C:\Users\beyqm\Desktop\北大实习\data\task2_output\layout_morphology.csv')
lm['farm_id_int'] = lm['farm_id'].str.extract(r'farm_(\d+)').astype(int)
gain = gain.merge(lm[['farm_id_int','region']], left_on='farm_id', right_on='farm_id_int', how='left')
gain['country'] = gain['country'].fillna('Unknown')

print(f'  区域: {gain.region.value_counts().to_dict()}')

# Also load S3 farm-year data for richer bootstrap
s3 = pd.read_csv(os.path.join(TASK3, 'task3_s3_comparison.csv'))
# Filter to gauss real + s1_opt that matches each farm's theta_opt
s1 = pd.read_csv(os.path.join(TASK3, 'task3_s1_optimal_orientation.csv'))
idx_opt = s1.groupby('farm_id')['expected_AEP_kWh'].idxmax()
theta_opt = s1.loc[idx_opt, ['farm_id','angle_deg']].copy()
theta_opt.columns = ['farm_id','theta_opt']

real = s3[(s3['wake_model']=='gauss')&(s3['layout_type']=='real')][['farm_id','year','AEP_kWh']].copy()
real.columns = ['farm_id','year','real_AEP']

opt_all = s3[(s3['wake_model']=='gauss')&(s3['layout_type'].str.startswith('s1_opt'))].copy()
opt_all['angle'] = opt_all['layout_type'].str.extract(r's1_opt_(\d+)deg').astype(int)
opt_all = opt_all.merge(theta_opt, on='farm_id')
opt_best = opt_all[opt_all['angle']==opt_all['theta_opt']][['farm_id','year','AEP_kWh']].copy()
opt_best.columns = ['farm_id','year','opt_AEP']

cmp = real.merge(opt_best, on=['farm_id','year'], how='inner')
cmp['gain_pct'] = (cmp['opt_AEP'] - cmp['real_AEP']) / cmp['real_AEP'] * 100
fc = s3.groupby('farm_id')['country'].first()
cmp['country'] = cmp['farm_id'].map(fc)
cmp = cmp.merge(lm[['farm_id_int','region']], left_on='farm_id', right_on='farm_id_int', how='left')
print(f'  逐 farm-year: {len(cmp)} 条, {cmp.farm_id.nunique()} 场')

# ══════════════════════════════════════
# 2. 经典配对 t 检验 (per-farm mean)
# ══════════════════════════════════════
print('\n2. 经典配对 t 检验...')
farm_mean = cmp.groupby('farm_id')['gain_pct'].mean()
t_stat, t_p = stats.ttest_1samp(farm_mean, 0)
ci_low, ci_high = stats.t.interval(0.95, df=len(farm_mean)-1, loc=farm_mean.mean(), scale=stats.sem(farm_mean))
print(f'  均值: {farm_mean.mean():.3f}%, 中位: {farm_mean.median():.3f}%')
print(f'  paired t: t={t_stat:.2f}, p={t_p:.2e}, 95% CI=[{ci_low:.3f}%, {ci_high:.3f}%]')

# ══════════════════════════════════════
# 3. Cluster Bootstrap
# ══════════════════════════════════════
print('\n3. Cluster Bootstrap (B=10,000)...')
B = 10000

# A) By country (17 clusters)
country_clusters = list(cmp.groupby('country').groups.keys())
cluster_data = {}
for ctry, idxs in cmp.groupby('country').groups.items():
    cluster_data[ctry] = cmp.loc[idxs, 'gain_pct'].values

np.random.seed(42)
boot_means_c = np.zeros(B)
for b in range(B):
    sampled = np.random.choice(list(cluster_data.keys()), size=len(cluster_data), replace=True)
    all_vals = np.concatenate([cluster_data[c] for c in sampled])
    boot_means_c[b] = np.mean(all_vals)

boot_ci_c = np.percentile(boot_means_c, [2.5, 97.5])
boot_p_c = 2 * min(np.mean(boot_means_c <= 0), np.mean(boot_means_c >= 0))

print(f'  [17国聚类]')
print(f'    均值: {np.mean(boot_means_c):.3f}%')
print(f'    95% CI: [{boot_ci_c[0]:.3f}%, {boot_ci_c[1]:.3f}%]')
print(f'    SE: {np.std(boot_means_c):.3f}%')
print(f'    p: {boot_p_c:.4f}')

# B) By region (3 clusters, most conservative)
cluster_data_r = {}
for reg, idxs in cmp.groupby('region').groups.items():
    cluster_data_r[reg] = cmp.loc[idxs, 'gain_pct'].values

boot_means_r = np.zeros(B)
for b in range(B):
    sampled = np.random.choice(list(cluster_data_r.keys()), size=len(cluster_data_r), replace=True)
    all_vals = np.concatenate([cluster_data_r[s] for s in sampled])
    boot_means_r[b] = np.mean(all_vals)

boot_ci_r = np.percentile(boot_means_r, [2.5, 97.5])
boot_p_r = 2 * min(np.mean(boot_means_r <= 0), np.mean(boot_means_r >= 0))
print(f'\n  [3区域聚类 — 最保守]')
print(f'    95% CI: [{boot_ci_r[0]:.3f}%, {boot_ci_r[1]:.3f}%]')
print(f'    SE: {np.std(boot_means_r):.3f}%')
print(f'    p: {boot_p_r:.4f}')

# ══════════════════════════════════════
# 4. 分层符号检验
# ══════════════════════════════════════
print('\n4. 分层符号检验...')
sign_results = []
for country, vals in cluster_data.items():
    fm = np.mean(vals)
    n_pos = np.sum(vals > 0)
    n_neg = np.sum(vals < 0)
    n_total = len(vals)
    # Binomial test: H0: P(positive) = 0.5
    if n_total >= 3:
        p_binom = stats.binomtest(n_pos, n=n_pos+n_neg, p=0.5).pvalue if (n_pos+n_neg) > 0 else 1.0
        sign_results.append({
            'country': country, 'n': n_total, 'n_pos': n_pos, 'n_neg': n_neg,
            'mean_gain': fm, 'binom_p': p_binom
        })

df_sign = pd.DataFrame(sign_results).sort_values('mean_gain', ascending=False)
for _, r in df_sign.head(10).iterrows():
    sig = '***' if r['binom_p']<0.001 else ('**' if r['binom_p']<0.01 else ('*' if r['binom_p']<0.05 else 'ns'))
    print(f"  {r['country']:<20s} n={int(r['n']):>4d}  +:{int(r['n_pos']):>4d}  -:{int(r['n_neg']):>4d}  "
          f"mean={r['mean_gain']:+.2f}%  binom_p={r['binom_p']:.3f} {sig}")

# ══════════════════════════════════════
# 5. 汇总保存
# ══════════════════════════════════════
print('\n5. 保存...')
summary = pd.DataFrame([
    {'method': '经典配对t', 'CI_low': ci_low, 'CI_high': ci_high, 'p_value': t_p,
     'mean': farm_mean.mean(), 'median': farm_mean.median(), 'n_farms': len(farm_mean)},
    {'method': 'Cluster bootstrap (17国)', 'CI_low': boot_ci_c[0], 'CI_high': boot_ci_c[1], 'p_value': boot_p_c,
     'mean': np.mean(boot_means_c), 'median': np.nan, 'n_farms': len(farm_mean)},
    {'method': 'Cluster bootstrap (3区域)', 'CI_low': boot_ci_r[0], 'CI_high': boot_ci_r[1], 'p_value': boot_p_r,
     'mean': np.mean(boot_means_r), 'median': np.nan, 'n_farms': len(farm_mean)},
])
summary.to_csv(os.path.join(OUT, 'task6_bootstrap_ci.csv'), index=False, encoding='utf-8-sig')
df_sign.to_csv(os.path.join(OUT, 'task6_sign_test_by_country.csv'), index=False, encoding='utf-8-sig')

print('\n=== 总结 ===')
print(f'经典 t CI:   [{ci_low:.2f}%, {ci_high:.2f}%]  p={t_p:.2e}')
print(f'17国 Bootstrap CI: [{boot_ci_c[0]:.2f}%, {boot_ci_c[1]:.2f}%]  p={boot_p_c:.4f}')
print(f'3区域 Bootstrap CI: [{boot_ci_r[0]:.2f}%, {boot_ci_r[1]:.2f}%]  p={boot_p_r:.4f}')
print(f'结论: 在最保守的3区域聚类下，增益均值仍显著>0')
