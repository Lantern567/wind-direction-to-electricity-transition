"""
走廊图谱 v3: 19特征RF模型 + 全球近海网格插值 → Fig 4
=====================================================
前两版失败原因: 只用WCI×wake_pool两个特征
v3: 加入风速分布特征(ws_std,ws_mean,weibull等) → RF R²=0.48
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import spearmanr

BUSH = r'd:\01学习资料\wind-direction-to-electricity-transition\补算'
TASK3 = r'd:\01学习资料\wind-direction-to-electricity-transition\task3'
DATA = r'd:\01学习资料\wind-direction-to-electricity-transition\data'
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════
print('1. 加载 + 构建 19 特征训练集')
# ══════════════════════════════════════
s1 = pd.read_csv(os.path.join(TASK3, 'task3_s1_optimal_orientation.csv'))
s3 = pd.read_csv(os.path.join(TASK3, 'task3_s3_comparison.csv'))
wm = pd.read_csv(os.path.join(DATA, 'task1_output', 'task1_wind_metrics.csv'))
lm = pd.read_csv(r'C:\Users\beyqm\Desktop\北大实习\data\task2_output\layout_morphology.csv')
fm = pd.read_csv(os.path.join(DATA, 'task0', 'farms_master.csv'))
fs = pd.read_csv(os.path.join(TASK3, '__pycache__', 'farm_ws_stats.csv'))
proxy = pd.read_csv(os.path.join(BUSH, 'output', 'task1_corridor_proxy_171farms.csv'))

# A = (max-mean)/mean
fa = s1.groupby('farm_id')['expected_AEP_kWh'].agg(['max','mean']).reset_index()
fa['A'] = (fa['max']-fa['mean'])/fa['mean']*100

# wake_pool
wp = s3[(s3['wake_model']=='gauss')&(s3['layout_type']=='real')].groupby('farm_id')['WakeLoss'].mean()
wp = wp.reset_index(); wp.columns = ['farm_id','wake_pool']

# WCI, geometry, wind stats
wci = wm.groupby('farm_id')['WCI_yearly'].mean().reset_index(); wci.columns=['farm_id','WCI']
lm['fid'] = lm['farm_id'].str.extract(r'farm_(\d+)').astype(int)
geo = lm[['fid','spacing_D','aspect_ratio','n_turbines','pc1_share','axis_deg']].copy()
geo.columns = ['farm_id','spacing_D','aspect_ratio','n_turb','pc1_share','axis_deg']
ws = fs.groupby('farm_id')[['ws_mean','ws_std','weibull_A','weibull_k','frac_below_rated']].mean().reset_index()
pf = proxy[['farm_id','orient_sensitivity','gain_proxy_raw','wd_entropy_norm','exp_wake_loss']]

df = fa.merge(wp,on='farm_id').merge(wci,on='farm_id').merge(geo,on='farm_id').merge(ws,on='farm_id')
df = df.merge(pf,on='farm_id').merge(fm[['farm_id','centroid_lat','centroid_lon','country','n_turb']],on='farm_id')
df = df.rename(columns={'centroid_lat':'lat','centroid_lon':'lon'})
# After fm+geo merge, n_turb from fm becomes n_turb_x (geo also has n_turb)
if 'n_turb_x' in df.columns:
    df['n_turb'] = df['n_turb_x']
elif 'n_turb' not in df.columns:
    df['n_turb'] = df['n_turb_y']

# Engineered
df['WCI_x_pool'] = df['WCI'] * df['wake_pool']
df['log_n'] = np.log10(df['n_turb'])
df['WCI_density'] = df['WCI'] / df['spacing_D']
df['ws_x_WCI'] = df['ws_mean'] * df['WCI']
df['pool_density'] = df['wake_pool'] / df['spacing_D']

feats = ['WCI','wake_pool','WCI_x_pool','spacing_D','aspect_ratio','n_turb','log_n','pc1_share',
         'ws_mean','ws_std','weibull_A','weibull_k','frac_below_rated',
         'orient_sensitivity','gain_proxy_raw','wd_entropy_norm','exp_wake_loss',
         'WCI_density','pool_density','ws_x_WCI']

df = df.dropna(subset=feats)
X = df[feats].values
y = df['A'].values
print(f'  训练集: {len(df)} farms × {len(feats)} features')

# ══════════════════════════════════════
print('2. RF 模型训练 + 验证')
# ══════════════════════════════════════
rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, min_samples_leaf=8)
rf.fit(X, y)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
print(f'  Train R²: {rf.score(X,y):.3f}')
print(f'  5-CV R²:  {scores.mean():.3f} +/- {scores.std():.3f}')

# Feature importance
imp = pd.DataFrame({'feat':feats,'imp':rf.feature_importances_}).sort_values('imp',ascending=False)
print('  Top 8 features:')
for _,r in imp.head(8).iterrows():
    print(f'    {r["feat"]:<20s} FI={r["imp"]:.3f}')

# LOO by country
countries = df['country'].values
preds = []
for ctry in np.unique(countries):
    tr = countries!=ctry; te = countries==ctry
    if te.sum()==0 or tr.sum()<10: continue
    rf_c = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_c.fit(X[tr], y[tr])
    p = rf_c.predict(X[te])
    for i, idx in enumerate(np.where(te)[0]):
        preds.append({'farm_id':df.iloc[idx]['farm_id'],'country':ctry,
                       'actual_A':y[idx],'pred_A':p[i]})
pl = pd.DataFrame(preds)
r_loo = np.corrcoef(pl['actual_A'],pl['pred_A'])[0,1]
pl['a_hi'] = pl['actual_A'] > np.percentile(pl['actual_A'],75)
pl['p_hi'] = pl['pred_A'] > np.percentile(pl['pred_A'],75)
tp = ((pl['a_hi'])&(pl['p_hi'])).sum()
fn = ((pl['a_hi'])&(~pl['p_hi'])).sum()
print(f'  LOO r: {r_loo:.3f}, recall(top25%): {tp/(tp+fn):.2f}')

# Check: Vietnam/China strait corridors
for corr_name, fids in [('Vietnam',[57,126,159]),('China_strait',[66,91])]:
    sub = pl[pl['farm_id'].isin(fids)]
    print(f'  {corr_name}: pred_A={sub["pred_A"].mean():.2f}%, actual_A={sub["actual_A"].mean():.2f}%')

# ══════════════════════════════════════
print('3. 全球近海网格预测')
# ══════════════════════════════════════
# Coarse grid (1°)
lons = np.arange(-80, 145, 1.0)
lats = np.arange(5, 62, 1.0)
grid_lon, grid_lat = np.meshgrid(lons, lats)
grid_flat = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
n_grid = len(grid_flat)

# IDW interpolate all features from 171 farms to grid
train_pts = df[['lon','lat']].values
mid_lat = np.mean(train_pts[:,1])
sx = 6371*np.cos(np.radians(mid_lat))*np.pi/180
sy = 6371*np.pi/180
train_xy = train_pts * np.array([sx, sy])
grid_xy = grid_flat * np.array([sx, sy])
tree = cKDTree(train_xy)
distances, indices = tree.query(grid_xy, k=5)
weights = 1.0/(distances**2 + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)

# Interpolate features
X_grid = np.zeros((n_grid, len(feats)))
for k, feat in enumerate(feats):
    X_grid[:,k] = np.sum(df[feat].values[indices] * weights, axis=1)
# Restore lat/lon (not interpolated — use grid values for latitude-sensitive features)
# ws_mean, weibull_A etc are already interpolated from nearby farms

# Predict A on grid
A_grid = rf.predict(X_grid).reshape(grid_lon.shape)

# Mask: only within ~500km of known farms
max_dist = np.min(distances, axis=1)
A_grid[max_dist.reshape(grid_lon.shape) > 500] = np.nan  # 500 km

valid_mask = ~np.isnan(A_grid)
print(f'  Grid: {lons.min()}~{lons.max()}°E × {lats.min()}~{lats.max()}°N')
print(f'  Valid points: {valid_mask.sum()}/{n_grid}')
print(f'  A range: [{np.nanmin(A_grid):.1f}%, {np.nanmax(A_grid):.1f}%]')

# Identify corridor candidates: A > 4% (top quartile threshold)
threshold = np.nanpercentile(A_grid, 75)
corridor_mask = A_grid > threshold
print(f'  Corridor candidates (A > {threshold:.1f}%): {corridor_mask.sum()} grid cells')

# ══════════════════════════════════════
print('4. 保存')
# ══════════════════════════════════════
df.to_csv(os.path.join(OUT, 'task1_v3_training_data.csv'), index=False, encoding='utf-8-sig')
pl.to_csv(os.path.join(OUT, 'task1_v3_loo_predictions.csv'), index=False, encoding='utf-8-sig')
grid_out = pd.DataFrame({
    'lon': grid_lon.ravel(), 'lat': grid_lat.ravel(),
    'A_pred_pct': A_grid.ravel()
}).dropna()
grid_out.to_csv(os.path.join(OUT, 'task1_v3_corridor_grid.csv'), index=False, encoding='utf-8-sig')

print('\n=== 走廊候选区域摘要 ===')
for lat_band in [(5,20),(20,35),(35,50),(50,62)]:
    sub = grid_out[(grid_out['lat']>=lat_band[0])&(grid_out['lat']<lat_band[1])]
    if len(sub)>0:
        high = sub[sub['A_pred_pct']>threshold]
        print(f'  {lat_band[0]:.0f}-{lat_band[1]:.0f}°N: {len(high)}/{len(sub)} cells >{threshold:.1f}%, max={sub["A_pred_pct"].max():.1f}%')

print('\n产出: task1_v3_training_data.csv + task1_v3_loo_predictions.csv + task1_v3_corridor_grid.csv')
