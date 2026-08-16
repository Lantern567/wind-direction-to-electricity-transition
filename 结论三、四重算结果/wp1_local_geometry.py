"""
WP1 几何冻结：逐场局部等距方位投影重算
==========================================
对应《结论三与结论四补充计算方案》§3 执行顺序第 1 步。

修复旧口径两个问题：
  1. 旧口径用全样本平均纬度平面换算（cos(mid_lat) 缩放），跨纬度比较有扭曲；
  2. 旧口径 Clark-Evans 面积用包围盒近似，主轴用平均纬度平面 PCA。

新口径（本脚本冻结）：
  - 逐场 pyproj 局部等距方位投影（aeqd，中心=场心）；
  - PCA 主轴（数学角，度，[0,180)）、pc1_share、长宽比 = σ1/σ2；
  - 最近邻间距：cKDTree 于投影坐标，均值与中位数（m 与 D，统一参考 D=198 m）；
  - Clark-Evans R：凸包面积版（优于包围盒）；
  - 形态判定（规则树，冻结；阈值语义与旧口径 74/48/24/21/4 对齐）：
      稀疏 sparse        : n_turb < 10
      多簇 multi_cluster : 连通分量 ≥ 2（分量间最小距离 > 3×场内中位NN间距）
      带状 belt          : pc1_share ≥ 0.85（单一主轴、细长排布）
      规则网格 rule_grid : ce_r > 1.8（过度分散=规则性）
      单簇 cluster       : 其余（ce_r ∈ [1.2,1.8] 且非细长）
    判定顺序：sparse → multi_cluster → belt → rule_grid → cluster。
  - 方位角换算：axis_bearing = (90° − pc1_angle) mod 180°（数学角→北向方位角）。

输出：补算/output/wp1_geometry_frozen.csv
      补算/output/wp1_type_transition.csv（旧-新形态转移矩阵）
      补算/output/wp1_geometry_report.txt
"""
import os, io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull
from pyproj import Transformer, CRS

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
DATA = os.path.join(REPO, 'data')
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

D_REF = 198.0   # 统一参考转子直径（IEA 10MW RWT, m）
EARTH = 6371.0  # km

# ═══════════════════════════════════════════════════════════════════════
# 1. 数据
# ═══════════════════════════════════════════════════════════════════════
tc = pd.read_csv(os.path.join(DATA, 'task0', 'turbine_coordinates.csv'))
fm = pd.read_csv(os.path.join(DATA, 'task0', 'farms_master.csv'))
old = pd.read_csv(os.path.join(DATA, 'task0', 'layout_geometry.csv'))

# 终期机位（每年份取最后一年；farm-year 形态由终期机位冻结）
tc = tc.sort_values(['farm_id', 'year'])
final_pos = tc.groupby('farm_id').last().reset_index()
final_pos = final_pos[['farm_id', 'year', 'turbine_id']].merge(
    tc[['farm_id', 'year', 'turbine_id', 'lon', 'lat']],
    on=['farm_id', 'year', 'turbine_id'], how='left')
# 上面 merge 只取到终年一行/台；更稳妥：直接按终年筛
last_year = tc.groupby('farm_id')['year'].max().reset_index().rename(columns={'year': 'last_year'})
tc_last = tc.merge(last_year, on='farm_id')
tc_last = tc_last[tc_last['year'] == tc_last['last_year']]
print(f'终期机位: {len(tc_last)} 台 / {tc_last.farm_id.nunique()} 场')

old_t = old[old['year'] == old.groupby('farm_id')['year'].transform('max')].copy()
old_t = old_t[['farm_id', 'spacing_d', 'pc1_angle', 'pc1_share', 'aspect_ratio', 'layout_type', 'ce_r']]

# ═══════════════════════════════════════════════════════════════════════
# 2. 逐场几何重算
# ═══════════════════════════════════════════════════════════════════════
rows = []
for fid, g in tc_last.groupby('farm_id'):
    lon = g['lon'].values.astype(float)
    lat = g['lat'].values.astype(float)
    n = len(g)
    clon, clat = float(np.median(lon)), float(np.median(lat))

    # 局部等距方位投影（场心为原点）
    tf = Transformer.from_crs(CRS.from_epsg(4326),
                              CRS.from_proj4(f'+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m'),
                              always_xy=True)
    x, y = tf.transform(lon, lat)
    x = np.asarray(x); y = np.asarray(y)
    xy = np.column_stack([x, y])

    # PCA 主轴
    xy_c = xy - xy.mean(axis=0)
    cov = xy_c.T @ xy_c / (n - 1)
    w, V = np.linalg.eigh(cov)
    i1, i2 = np.argsort(w)[::-1][:2]
    s1, s2 = np.sqrt(w[i1]), np.sqrt(w[i2])
    pc1 = V[:, i1]
    ang = float(np.degrees(np.arctan2(pc1[1], pc1[0])))
    if ang < 0:
        ang += 180.0
    pc1_share = float(w[i1] / (w[i1] + w[i2]))
    aspect = float(s1 / s2) if s2 > 1e-9 else np.nan
    bearing = (90.0 - ang) % 180.0   # 数学角 → 北向方位角

    # 最近邻间距
    tree = cKDTree(xy)
    d2 = tree.query(xy, k=min(2, n))[0][:, 1] if n > 1 else np.full(n, np.nan)
    nn_mean_m = float(np.nanmean(d2))
    nn_med_m = float(np.nanmedian(d2))
    spacing_D_mean = nn_mean_m / D_REF
    spacing_D_med = nn_med_m / D_REF

    # Clark-Evans R（凸包面积版）
    ce_r = np.nan
    if n >= 3:
        try:
            hull = ConvexHull(xy)
            area = hull.volume  # 2D 凸包面积 m²
            rho = n / area
            r_obs = nn_mean_m
            r_exp = 1.0 / (2.0 * np.sqrt(rho))
            ce_r = float(r_obs / r_exp) if r_exp > 0 else np.nan
        except Exception:
            ce_r = np.nan

    # 连通分量（分量间最小距离 > 3×场内中位NN）
    n_comp = 1
    if n > 1:
        from scipy.sparse import csgraph
        thr = 3.0 * max(nn_med_m, 1.0)
        sm = tree.sparse_distance_matrix(tree, thr, output_type='coo_matrix')
        G = (sm > 0).astype(int)
        n_comp = int(csgraph.connected_components(G, directed=False)[0])

    # 形态判定（冻结规则树：sparse → multi_cluster → belt → rule_grid → cluster）
    if n < 10:
        ltype = 'sparse'
    elif n_comp >= 2:
        ltype = 'multi_cluster'
    elif pc1_share >= 0.85:
        ltype = 'belt'
    elif ce_r > 1.8:
        ltype = 'rule_grid'
    else:
        ltype = 'cluster'

    rows.append(dict(farm_id=int(fid), n_turb=n, cent_lon=clon, cent_lat=clat,
                     pc1_angle=round(ang, 2), axis_bearing=round(bearing, 2),
                     pc1_share=round(pc1_share, 4), aspect_ratio=round(aspect, 3),
                     nn_mean_m=round(nn_mean_m, 1), nn_med_m=round(nn_med_m, 1),
                     spacing_D_mean=round(spacing_D_mean, 3), spacing_D_med=round(spacing_D_med, 3),
                     ce_r=round(ce_r, 3), n_comp=n_comp, layout_type=ltype,
                     last_year=int(g['last_year'].iloc[0])))

geo = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════
# 3. 旧-新形态转移矩阵
# ═══════════════════════════════════════════════════════════════════════
old_map = {'规则网格': 'rule_grid', '带状': 'belt', '团簇': 'cluster',
           '多组团': 'multi_cluster', '稀疏': 'sparse'}
old_t['layout_type_en'] = old_t['layout_type'].map(old_map).fillna(old_t['layout_type'])

merged = geo.merge(old_t[['farm_id', 'layout_type_en', 'spacing_d', 'pc1_angle', 'pc1_share', 'aspect_ratio']],
                   on='farm_id', suffixes=('_new', '_old'))
trans = pd.crosstab(merged['layout_type_en'], merged['layout_type'])
trans.to_csv(os.path.join(OUT, 'wp1_type_transition.csv'), encoding='utf-8-sig')

# 新旧间距相关性（方案 §3：局部投影后 spacing 与 A 的 Spearman ≈ −0.746）
from scipy.stats import spearmanr
s1 = pd.read_csv(os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv'))
fa = s1.groupby('farm_id')['expected_AEP_kWh'].agg(['max', 'mean']).reset_index()
fa['A'] = (fa['max'] - fa['mean']) / fa['mean'] * 100
chk = merged.merge(fa[['farm_id', 'A']], on='farm_id').dropna(subset=['spacing_D_med', 'A'])
rho_new, p_new = spearmanr(chk['spacing_D_med'], chk['A'])
rho_old, p_old = spearmanr(chk['spacing_d'], chk['A'])
print(f'\n间距-A Spearman: 新局部投影 rho={rho_new:.3f} (p={p_new:.2e}) | 旧口径 rho={rho_old:.3f} (p={p_old:.2e})')

# ═══════════════════════════════════════════════════════════════════════
# 4. 输出
# ═══════════════════════════════════════════════════════════════════════
geo = geo.merge(fm[['farm_id', 'country']], on='farm_id', how='left')
geo.to_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), index=False, encoding='utf-8-sig')

with open(os.path.join(OUT, 'wp1_geometry_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP1 几何冻结报告（局部等距方位投影重算）\n')
    f.write('=' * 60 + '\n')
    f.write(f'农场数: {len(geo)} | 参考转子直径: {D_REF} m\n\n')
    f.write('形态分布（新口径）:\n')
    for t, c in geo['layout_type'].value_counts().items():
        f.write(f'  {t:<15s} {c}\n')
    f.write('\n形态分布（旧口径）:\n')
    for t, c in old_t['layout_type'].value_counts().items():
        f.write(f'  {t:<15s} {c}\n')
    f.write('\n新旧形态转移矩阵:\n')
    f.write(trans.to_string())
    f.write(f'\n\n间距-A Spearman: 新={rho_new:.3f} (p={p_new:.2e}) | 旧={rho_old:.3f} (p={p_old:.2e})\n')
    f.write('\n几何摘要:\n')
    f.write(geo[['nn_mean_m', 'spacing_D_med', 'aspect_ratio', 'pc1_share', 'ce_r']].describe().to_string())

print('\n形态分布（新口径）:')
print(geo['layout_type'].value_counts())
print('\n转移矩阵:')
print(trans)
print(f'\n输出: {os.path.join(OUT, "wp1_geometry_frozen.csv")}')
