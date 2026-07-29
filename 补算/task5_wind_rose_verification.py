"""任务5: 风玫瑰交叉验证 + 扰动测试"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BUSH = r'd:\01学习资料\wind-direction-to-electricity-transition\补算'
DATA = r'd:\01学习资料\wind-direction-to-electricity-transition\data'
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

wr = pd.read_csv(os.path.join(BUSH, 'high_gain_wind_roses.csv'))
rw = pd.read_csv(os.path.join(BUSH, 'reference_wake_table.csv'))
wm = pd.read_csv(os.path.join(DATA, 'task1_output', 'task1_wind_metrics.csv'))

print(f'Wind rose data: {len(wr)} rows, {wr.farm_id.nunique()} farms')

# === 1. 7场风玫瑰 + WCI ===
print('\n=== 1. 7场风玫瑰 WCI ===')
rose_results = []
for fid in sorted(wr['farm_id'].unique()):
    sub = wr[wr['farm_id']==fid]
    wd = sub['wd_deg'].values; country = sub['country'].iloc[0]
    hist, _ = np.histogram(wd, bins=np.arange(0, 370, 10))
    sectors = (np.arange(0, 360, 10) + np.arange(10, 370, 10)) / 2
    dom_sector = sectors[np.argmax(hist)]
    wd_rad = np.radians(wd)
    wci_30yr = np.sqrt(np.mean(np.sin(wd_rad))**2 + np.mean(np.cos(wd_rad))**2)
    t1 = wm[wm['farm_id']==fid]
    wci_task1 = t1['WCI_yearly'].mean() if len(t1)>0 else np.nan
    rose_results.append({'farm_id':fid,'country':country,'wci_30yr':round(wci_30yr,4),
        'wci_task1':round(wci_task1,4) if not np.isnan(wci_task1) else np.nan,
        'dom_sector':dom_sector,'dom_frac':round(hist.max()/hist.sum()*100,1),'n_obs':len(wd)})
    print(f'  F{fid} ({country}): WCI_30yr={wci_30yr:.4f}, WCI_task1={wci_task1:.4f}, dom={dom_sector:.0f}deg')

df_rose = pd.DataFrame(rose_results)
valid = df_rose.dropna(subset=['wci_30yr','wci_task1'])
r_wci, p_wci = pearsonr(valid['wci_30yr'], valid['wci_task1'])
print(f'  WCI_30yr vs WCI_task1: r={r_wci:.3f} (n={len(valid)})')

# === 2. 扰动测试 ±10deg ===
print('\n=== 2. 扰动测试 ===')
we36 = np.zeros(36)
for i, wd_sec in enumerate(np.arange(0, 360, 10)):
    row = rw[(rw['wd_sector_deg']==wd_sec)&(rw['ws_bin_m_s']==8.0)]
    we36[i] = row['wake_efficiency'].values[0] if len(row)>0 else 1.0

perturb_results = []
for fid in sorted(wr['farm_id'].unique()):
    wd = wr[wr['farm_id']==fid]['wd_deg'].values
    country = wr[wr['farm_id']==fid]['country'].iloc[0]
    for scenario, wd_data in [('original',wd),('+10deg',(wd+10)%360),('-10deg',(wd-10)%360)]:
        hist, _ = np.histogram(wd_data, bins=np.arange(0, 370, 10))
        energy = hist / hist.sum()
        mean_we = np.sum(energy * we36)
        best_we = we36.max()
        A = (best_we - mean_we) / mean_we * 100 if mean_we > 0 else 0
        perturb_results.append({'farm_id':fid,'country':country,'scenario':scenario,'A_pct':round(A,3)})

df_pert = pd.DataFrame(perturb_results)
# Pivot
pivot = df_pert.pivot(index=['farm_id','country'], columns='scenario', values='A_pct').reset_index()
pivot['delta_plus10'] = pivot['+10deg'] - pivot['original']
pivot['delta_minus10'] = pivot['-10deg'] - pivot['original']
pivot['robust'] = (pivot['delta_plus10'].abs()<0.5) & (pivot['delta_minus10'].abs()<0.5)
for _,r in pivot.iterrows():
    print(f'  F{int(r["farm_id"])} ({r["country"]}): A={r["original"]:.2f}%, +10={r["+10deg"]:.2f}%, -10={r["-10deg"]:.2f}%  [{"OK" if r["robust"] else "SENSITIVE"}]')
print(f'  Robust: {pivot["robust"].sum()}/{len(pivot)}')

# === 3. F57联合优化 ===
print('\n=== 3. F57联合优化 ===')
jo = pd.read_csv(os.path.join(BUSH, 'joint_optimization_F57(1).csv'))
for _,r in jo.iterrows():
    print(f'  {r["scenario"]:<20s} gain={r["gain_pct_vs_real"]}%')
orient_only = float(jo[jo['scenario']=='rotated_opt_joint']['gain_pct_vs_real'].iloc[0]) - float(jo[jo['scenario']=='built_opt_geom']['gain_pct_vs_real'].iloc[0])
geom_only = float(jo[jo['scenario']=='built_opt_geom']['gain_pct_vs_real'].iloc[0])
print(f'  几何贡献: {geom_only:.1f}%, 朝向贡献: {orient_only:.1f}% (总{geom_only+orient_only:.1f}%)')

# === Save ===
df_rose.to_csv(os.path.join(OUT, 'task5_wind_rose_verification.csv'), index=False, encoding='utf-8-sig')
pivot.to_csv(os.path.join(OUT, 'task5_perturbation_test.csv'), index=False, encoding='utf-8-sig')
print('\n产出: task5_wind_rose_verification.csv + task5_perturbation_test.csv')
