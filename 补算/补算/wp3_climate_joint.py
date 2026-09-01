"""
WP3 逐场址联合风况 p(u,d)：171 场址 × 11 年（2014-2024）× 逐时
==================================================================
对应《结论三与结论四补充计算方案》§3 执行顺序第 3 步。

口径（与论文主分析 offshore-task2/task2_floris.py 完全一致）：
  - ERA5 u100/v100，取农场质心（farms_master）最近格点；
  - 风向：气象惯例（来向），wd = (270° − atan2(v,u)) mod 360；
  - 风速高度换算到轮毂高度 119 m：V_hub = V_100 × (119/100)^0.11；
  - 区域划分：get_region_for_farm（east_asia / europe / us_east / japan）。

为什么全量重提取：
  仓库内 farm_wind_east_asia/europe/us_east.parquet 与 NC 在质心处的
  逐时序列相关系数 ≈ 0.0（旧版农场编号/位置映射已失效），仅
  farm_wind_missing_82.parquet 正确（V corr = 1.0000 已验证）。
  故 171 个场址统一直接从 data/111/era5_{region}_{year}.nc 重提取，
  并用 missing_82 做交叉验证。

分箱：风速 18 档（WS_BINS 剔除 0 档，边=相邻档中点）；风向 72 个 5° 扇区
（中心 0,5,...,355，边 ∓2.5°）——与 WP4 尾流查表网格逐格对齐，
下游 E(θ) 只需按索引平移。

输出：补算/output/wp3_climate_joint.npz
  p_fy  [171, 11, 18, 72] float32  逐场-年联合概率（Σ=1）
  hours [171, 11] int32            逐场-年有效小时数
  farm_ids [171], years [11], ws [18], wd [72]
  补算/output/wp3_climate_summary.csv（逐场：格点、区域、年均风速、主风向）
  补算/output/wp3_climate_report.txt
"""
import os, io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import netCDF4

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)
NC_DIR = os.path.join(REPO, 'data', '111')
os.chdir(NC_DIR)   # netCDF4 的 C 层打不开含中文的绝对路径 → 切到 NC 目录用相对路径
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import WS_BINS, get_turbine_params

H_REF = 100.0
H_TURB = get_turbine_params('iea_10MW')['H']   # 119 m
ALPHA = 0.11
YEARS = list(range(2014, 2025))

# ---- 分箱边（与 WP4 查表网格对齐）----
WS = np.array([w for w in WS_BINS if w >= 3.0])          # 18 档
ws_mid = (WS[:-1] + WS[1:]) / 2.0
WS_EDGES = np.concatenate([[WS[0] - (WS[1] - WS[0]) / 2], ws_mid, [np.inf]])
WD = np.arange(0.0, 360.0, 5.0)                          # 72 扇区
WD_EDGES = np.arange(-2.5, 362.5, 5.0)

def region_of(lat, lon):
    if 8 <= lat <= 44 and 104 <= lon <= 143: return 'east_asia'
    if 39 <= lat <= 63 and -12 <= lon <= 32: return 'europe'
    if 36 <= lat <= 42 and -78 <= lon <= -68: return 'us_east'
    if 30 <= lat <= 46 and 128 <= lon <= 146: return 'japan'
    return 'east_asia'

# ═══════════════════════════════════════════════════════════════════════
# 1. 场址表
# ═══════════════════════════════════════════════════════════════════════
fm = pd.read_csv(os.path.join(REPO, 'data', 'task0', 'farms_master.csv'))
fm['region'] = fm.apply(lambda r: region_of(r.centroid_lat, r.centroid_lon), axis=1)
fm = fm.sort_values('farm_id').reset_index(drop=True)
print('场址区域分布:', fm.groupby('region').size().to_dict())

F = len(fm)
p_fy = np.zeros((F, len(YEARS), len(WS), len(WD)), dtype=np.float32)
hours = np.zeros((F, len(YEARS)), dtype=np.int32)
grid_info = []

# ═══════════════════════════════════════════════════════════════════════
# 2. 逐 (region, year) 打开 NC，提取该区域全部场址
# ═══════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
for region, regf in fm.groupby('region'):
    for yi, yr in enumerate(YEARS):
        nc_path = f'era5_{region}_{yr}.nc'   # cwd 已切到 NC_DIR（C 层中文路径问题）
        if not os.path.exists(nc_path):
            print(f'  !! 缺文件 {os.path.abspath(nc_path)} — {region} {yr} 跳过')
            continue
        ds = netCDF4.Dataset(nc_path, 'r')
        lat_arr = np.array(ds['latitude'][:], float)
        lon_arr = np.array(ds['longitude'][:], float)
        for _, row in regf.iterrows():
            fid = int(row.farm_id)
            ilat = int(np.argmin(np.abs(lat_arr - row.centroid_lat)))
            ilon = int(np.argmin(np.abs(lon_arr - row.centroid_lon)))
            u = np.array(ds['u100'][:, ilat, ilon], float)
            v = np.array(ds['v100'][:, ilat, ilon], float)
            ws_100 = np.sqrt(u**2 + v**2)
            wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
            ws_hub = ws_100 * (H_TURB / H_REF) ** ALPHA
            H, _, _ = np.histogram2d(wd, ws_hub, bins=[WD_EDGES, WS_EDGES])
            # histogram2d 返回 (wd, ws) → 转置为 (ws, wd)
            H = H.T
            fi = fm.index[fm.farm_id == fid][0]
            hours[fi, yi] = int(H.sum())
            if H.sum() > 0:
                p_fy[fi, yi] = (H / H.sum()).astype(np.float32)
            if yr == YEARS[-1]:
                grid_info.append(dict(farm_id=fid, grid_lat=round(float(lat_arr[ilat]), 3),
                                      grid_lon=round(float(lon_arr[ilon]), 3)))
        ds.close()
    print(f'{region}: 完成 ({time.perf_counter()-t0:.0f}s)', flush=True)

# ═══════════════════════════════════════════════════════════════════════
# 3. 交叉验证：与 canonical missing_82 parquet 比对
# ═══════════════════════════════════════════════════════════════════════
mi = pd.read_parquet(os.path.join(REPO, 'data', 'era5', 'farm_wind_missing_82.parquet'))
val = []
rng = np.random.default_rng(0)
for fid in rng.choice(sorted(mi.farm_id.unique()), size=min(6, mi.farm_id.nunique()), replace=False):
    f = fm[fm.farm_id == fid].iloc[0]
    yr = 2018
    nc_path = f'era5_{f.region}_{yr}.nc'   # cwd 已在 NC_DIR
    ds = netCDF4.Dataset(nc_path, 'r')
    lat_arr = np.array(ds['latitude'][:], float); lon_arr = np.array(ds['longitude'][:], float)
    ilat = int(np.argmin(np.abs(lat_arr - f.centroid_lat))); ilon = int(np.argmin(np.abs(lon_arr - f.centroid_lon)))
    u = np.array(ds['u100'][:, ilat, ilon], float); v = np.array(ds['v100'][:, ilat, ilon], float)
    ds.close()
    ws_100 = np.sqrt(u**2 + v**2)
    m = mi.query('farm_id==@fid and year==@yr')
    corr = float(np.corrcoef(ws_100, m.V_ms)[0, 1])
    val.append((int(fid), corr))
print('missing_82 交叉验证 (V corr):', [(f, round(c, 4)) for f, c in val])

# ═══════════════════════════════════════════════════════════════════════
# 4. 摘要与输出
# ═══════════════════════════════════════════════════════════════════════
gi = pd.DataFrame(grid_info)
summary = fm[['farm_id', 'region', 'centroid_lon', 'centroid_lat']].merge(gi, on='farm_id')
summary['n_years'] = (hours > 0).sum(axis=1)
p_f = p_fy.sum(axis=1) / np.maximum(p_fy.sum(axis=1).sum(axis=-1).sum(axis=-1), 1e-9)[:, None, None]
ws_marg = (p_f.sum(axis=2) * WS[None, :]).sum(axis=1)   # 逐场平均轮毂风速
summary['mean_ws_hub'] = ws_marg.round(3)
# 主风向（方向边际的向量平均）
e_theta = p_f.sum(axis=1)                               # 逐场方向边际 (171, 72)
main_wd = (np.degrees(np.angle((e_theta * np.exp(1j * np.radians(WD))[None, :]).sum(axis=1))) % 360)
summary['main_wd_deg'] = main_wd.round(1)
summary.to_csv(os.path.join(OUT, 'wp3_climate_summary.csv'), index=False, encoding='utf-8-sig')

np.savez_compressed(os.path.join(OUT, 'wp3_climate_joint.npz'),
                    p_fy=p_fy, hours=hours, farm_ids=fm.farm_id.values,
                    years=np.array(YEARS), ws=WS, wd=WD)

with open(os.path.join(OUT, 'wp3_climate_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP3 逐场址联合风况报告（ERA5 全量重提取）\n')
    f.write('=' * 60 + '\n')
    f.write(f'场址数: {F} | 年份: {YEARS[0]}-{YEARS[-1]} | 轮毂高度: {H_TURB} m\n')
    f.write(f'风速档: {len(WS)} ({WS[0]}-{WS[-1]} m/s) | 风向扇区: {len(WD)} × 5°\n\n')
    f.write('区域分布:\n' + fm.groupby('region').size().to_string() + '\n\n')
    f.write('年份覆盖（每场址有效年数）:\n')
    f.write(summary['n_years'].value_counts().sort_index().to_string() + '\n\n')
    f.write('与 canonical missing_82 交叉验证 (V corr):\n')
    f.write('\n'.join(f'  farm {ff}: {cc:.4f}' for ff, cc in val) + '\n\n')
    f.write('逐场摘要:\n')
    f.write(summary.to_string(index=False))

print(f'\n年份覆盖: {summary.n_years.value_counts().sort_index().to_dict()}')
print(f'总耗时 {time.perf_counter()-t0:.0f}s | 输出: wp3_climate_joint.npz')
