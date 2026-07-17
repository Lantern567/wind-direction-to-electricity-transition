"""
任务五(a): 扩建排布生成 — 为廷显生成 2030/2035/2040 扩建 UTM 坐标
复用 S2 范式管线 (task3_paradigm_layouts.py 的生成函数)
输出: output/for_tingxian_expansion_layouts.csv
"""
import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# Add task3 directory to path for importing paradigm generators
TASK3_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK3_DIR)

from task3_paradigm_layouts import (
    load_data, get_farm_boundary, generate_grid_in_boundary,
    generate_paradigm_A, generate_paradigm_B, generate_paradigm_C,
    generate_paradigm_E, D_M
)

OUTPUT_DIR = os.path.join(TASK3_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target years and expansion multipliers
EXPANSION_PLAN = {
    2030: 1.20,
    2035: 1.50,
    2040: 2.00,
}

# Select representative farms from POC 12, covering all paradigms
# Use the same farms Tingxian already ran POC on
REP_FARMS = [
    # (farm_id, description)
    (162, 'Small-dense-A'),
    (107, 'Small-dense-A-Europe'),
    (4,   'Small-sparse-E'),
    (14,  'Mid-standard-A'),
    (129, 'Mid-standard-A-Europe'),
    (112, 'Mid-constrained-B'),
    (91,  'Mid-expansion-C'),
    (83,  'Mid-expansion-C-dense'),
    (21,  'Large-A'),
    (151, 'Large-A-Europe'),
    (73,  'Large-A-big'),
    (153, 'Large-A-100'),
]

print('=' * 60)
print('任务五(a): 扩建排布生成')
print(f'代表风场: {len(REP_FARMS)} 个')
print(f'目标年份: {list(EXPANSION_PLAN.keys())}')
print('=' * 60)

# Load data (reuse S2 pipeline for coordinates + generator internals, use TASK2 for paradigms)
print('\n加载数据...')
data = load_data()
paradigm_df = data['paradigm']  # Keep for generator internal use (country fallback)
theta_2024 = data['theta_2024']
wind_2024 = data['wind_2024']
tcoords_2024 = data['tcoords_2024']
tcoords_all = data['tcoords']

# Use TASK2 paradigm (single-label, consistent with POC and all prior stats)
TASK2_LAYOUT = r'C:\Users\beyqm\Desktop\北大实习\data\task2_output\layout_morphology.csv'
task2_paradigm = pd.read_csv(TASK2_LAYOUT)
task2_paradigm['farm_id_int'] = task2_paradigm['farm_id'].str.extract(r'farm_(\d+)').astype(int)

# Pre-group
tcoords_by_farm = {fid: grp for fid, grp in tcoords_all.groupby('farm_id')}
tcoords_2024_by_farm = {fid: grp[['x_m', 'y_m']].values
                        for fid, grp in tcoords_2024.groupby('farm_id')}

generator_map = {
    'A': generate_paradigm_A,
    'B': generate_paradigm_B,
    'C': generate_paradigm_C,
    'E': generate_paradigm_E,
}

records = []
stats = []

for farm_id, desc in REP_FARMS:
    # Get paradigm from TASK2 (consistent with POC)
    p_row = task2_paradigm[task2_paradigm['farm_id_int'] == farm_id]
    if len(p_row) == 0:
        print(f'  farm_{farm_id}: NOT FOUND in task2 layout, skipping')
        continue

    primary = p_row['paradigm'].values[0]
    n_current = int(p_row['n_turbines'].values[0])

    if primary not in generator_map:
        print(f'  farm_{farm_id}: paradigm={primary}, skipping')
        continue

    # Get real layout geometry
    real_coords = tcoords_2024_by_farm.get(farm_id, np.array([]).reshape(0, 2))
    if len(real_coords) < 3:
        print(f'  farm_{farm_id}: only {len(real_coords)} turbines, skipping')
        continue

    centroid, hull_verts, path, extent = get_farm_boundary(real_coords, buffer_m=500)

    # Get EPSG
    farm_epsg = 32650
    farm_utm = tcoords_2024[tcoords_2024['farm_id'] == farm_id]
    if len(farm_utm) > 0 and 'utm_epsg' in farm_utm.columns:
        epsg_vals = farm_utm['utm_epsg'].dropna()
        if len(epsg_vals) > 0:
            epsg_str = str(epsg_vals.values[0])
            farm_epsg = int(epsg_str.split(':')[1]) if ':' in epsg_str else int(epsg_str)

    print(f'\n  farm_{farm_id} ({desc}): {n_current}t, paradigm={primary}')

    for target_year, mult in EXPANSION_PLAN.items():
        n_target = max(n_current + 2, int(n_current * mult))

        # Cap expansion to avoid cramming too many turbines in small boundary
        if n_target > n_current * 3:
            n_target = n_current * 3

        generator = generator_map[primary]

        try:
            if primary == 'C':
                farm_tcoords_hist = tcoords_by_farm.get(farm_id)
                points, meta = generator(
                    farm_id, n_target, centroid, path, extent, real_coords,
                    farm_tcoords_hist, wind_2024, paradigm_df
                )
            elif primary in ('A', 'E'):
                points, meta = generator(
                    farm_id, n_target, centroid, path, extent, real_coords,
                    theta_2024, wind_2024, paradigm_df
                )
            else:  # B
                points, meta = generator(
                    farm_id, n_target, centroid, path, extent, real_coords,
                    theta_2024, paradigm_df
                )

            # Cap at target to avoid overflow (e.g., Paradigm C may generate too many)
            if len(points) > n_target:
                points = points[:n_target]
            n_actual = len(points)

            # Write turbine records
            for ti in range(n_actual):
                px, py = points[ti]
                records.append({
                    'farm_id': farm_id,
                    'paradigm': primary,
                    'target_year': target_year,
                    'n_target': n_target,
                    'n_actual': n_actual,
                    'turbine_id': f'{farm_id}_{target_year}_{ti}',
                    'x_m': round(float(px), 1),
                    'y_m': round(float(py), 1),
                    'utm_epsg': farm_epsg,
                })

            stats.append({
                'farm_id': farm_id, 'desc': desc, 'paradigm': primary,
                'target_year': target_year, 'n_current': n_current,
                'n_target': n_target, 'n_actual': n_actual,
            })
            print(f'    {target_year}: {n_current} -> {n_target} target, {n_actual} actual')

        except Exception as e:
            print(f'    {target_year}: ERROR - {str(e)[:100]}')

# Save results
df = pd.DataFrame(records)
out_path = os.path.join(OUTPUT_DIR, 'for_tingxian_expansion_layouts.csv')
df.to_csv(out_path, index=False)
print(f'\n保存: {out_path}')
print(f'  总记录: {len(df):,} 行')
print(f'  农场: {df["farm_id"].nunique()} 个')
print(f'  年份: {sorted(df["target_year"].unique())}')
print(f'  范式: {sorted(df["paradigm"].unique())}')

# Save stats summary
df_stats = pd.DataFrame(stats)
stats_path = os.path.join(OUTPUT_DIR, 'for_tingxian_expansion_stats.csv')
df_stats.to_csv(stats_path, index=False)
print(f'  统计: {stats_path}')

# Summary table
print(f'\n===== 扩建排布摘要 =====')
print(f'{"farm_id":>8s} {"paradigm":>5s} {"n_current":>9s} {"2030":>8s} {"2035":>8s} {"2040":>8s}')
for _, r in df_stats.iterrows():
    pass
# Print per-farm summary
for fid in sorted(df_stats['farm_id'].unique()):
    sub = df_stats[df_stats['farm_id'] == fid]
    nc = sub['n_current'].iloc[0]
    p = sub['paradigm'].iloc[0]
    n30 = sub[sub['target_year']==2030]['n_actual'].values
    n35 = sub[sub['target_year']==2035]['n_actual'].values
    n40 = sub[sub['target_year']==2040]['n_actual'].values
    s30 = str(int(n30[0])) if len(n30)>0 else '-'
    s35 = str(int(n35[0])) if len(n35)>0 else '-'
    s40 = str(int(n40[0])) if len(n40)>0 else '-'
    print(f'{fid:>8d} {p:>5s} {nc:>9d} {s30:>8s} {s35:>8s} {s40:>8s}')

print('\nDone! 交给廷显: for_tingxian_expansion_layouts.csv')
