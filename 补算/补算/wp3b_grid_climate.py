"""
WP3b 观测支持域格点联合风况：1,446 个 1° 海洋格点 × 11 年（2014-2024）
=========================================================================
对应《结论三与结论四补充计算方案》表 1 的"修复后的 1,446 个观测支持域格点"。

格点来自 task1_corridor_grid.csv（上一阶段走廊图谱：1° 海洋网格，
距已观测风场 ≤ 500 km）。本脚本对每个格点用与 WP3 完全相同的口径
（u100/v100 → 来向风向 → 119 m 轮毂高度幂律修正 → 18×72 联合分箱）
从 data/111/era5_{region}_{year}.nc 提取逐时气候。

区域判定：格点必须落在某个区域 NC 的实际网格范围内（按经纬度边界
逐个区域检查），否则标记为无数据（NaN）——避免"默认 east_asia"
把超出范围的格点静默映射到错误位置。

输出：补算/output/wp3b_grid_climate.npz
  p_fy  [1446, 11, 18, 72] float32（无数据格点全为 NaN）
  hours [1446, 11] int32
  valid [1446] bool（是否有至少 1 年数据）
  lon/lat [1446], years [11], ws [18], wd [72]
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
os.chdir(NC_DIR)   # netCDF4 C 层打不开含中文绝对路径 → 相对路径
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import WS_BINS, get_turbine_params

H_REF = 100.0
H_TURB = get_turbine_params('iea_10MW')['H']   # 119 m
ALPHA = 0.11
YEARS = list(range(2014, 2025))

WS = np.array([w for w in WS_BINS if w >= 3.0])
ws_mid = (WS[:-1] + WS[1:]) / 2.0
WS_EDGES = np.concatenate([[WS[0] - (WS[1] - WS[0]) / 2], ws_mid, [np.inf]])
WD = np.arange(0.0, 360.0, 5.0)
WD_EDGES = np.arange(-2.5, 362.5, 5.0)

REGIONS = ['east_asia', 'europe', 'us_east', 'japan']

# ═══════════════════════════════════════════════════════════════════════
# 1. 格点表 + 区域归属（按 NC 实际网格范围）
# ═══════════════════════════════════════════════════════════════════════
g = pd.read_csv(os.path.join(OUT, 'task1_corridor_grid.csv'))
glon = g['lon'].values.astype(float)
glat = g['lat'].values.astype(float)
G = len(g)
print(f'格点数: {G}')

# 每个区域 NC 的网格范围（用 2018 年文件探测）
region_bbox = {}
for reg in REGIONS:
    p = f'era5_{reg}_2018.nc'
    if not os.path.exists(p):
        continue
    ds = netCDF4.Dataset(p, 'r')
    la = np.array(ds['latitude'][:], float); lo = np.array(ds['longitude'][:], float)
    ds.close()
    region_bbox[reg] = (la.min(), la.max(), lo.min(), lo.max())

grid_region = np.full(G, '', dtype=object)
for i in range(G):
    for reg, (la0, la1, lo0, lo1) in region_bbox.items():
        if la0 <= glat[i] <= la1 and lo0 <= glon[i] <= lo1:
            grid_region[i] = reg
            break
valid = grid_region != ''
print('区域归属:', pd.Series(grid_region[valid]).value_counts().to_dict())
print(f'无数据格点: {int((~valid).sum())}')

# ═══════════════════════════════════════════════════════════════════════
# 2. 逐 (region, year) 提取
# ═══════════════════════════════════════════════════════════════════════
p_fy = np.full((G, len(YEARS), len(WS), len(WD)), np.nan, dtype=np.float32)
hours = np.zeros((G, len(YEARS)), dtype=np.int32)
t0 = time.perf_counter()
for reg in REGIONS:
    if reg not in region_bbox:
        continue
    idx = np.where(grid_region == reg)[0]
    if len(idx) == 0:
        continue
    for yi, yr in enumerate(YEARS):
        p = f'era5_{reg}_{yr}.nc'
        if not os.path.exists(p):
            continue
        ds = netCDF4.Dataset(p, 'r')
        lat_arr = np.array(ds['latitude'][:], float)
        lon_arr = np.array(ds['longitude'][:], float)
        for i in idx:
            ilat = int(np.argmin(np.abs(lat_arr - glat[i])))
            ilon = int(np.argmin(np.abs(lon_arr - glon[i])))
            u = np.array(ds['u100'][:, ilat, ilon], float)
            v = np.array(ds['v100'][:, ilat, ilon], float)
            ws_100 = np.sqrt(u**2 + v**2)
            wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
            ws_hub = ws_100 * (H_TURB / H_REF) ** ALPHA
            H, _, _ = np.histogram2d(wd, ws_hub, bins=[WD_EDGES, WS_EDGES])
            H = H.T
            hours[i, yi] = int(H.sum())
            if H.sum() > 0:
                p_fy[i, yi] = (H / H.sum()).astype(np.float32)
        ds.close()
    print(f'{reg}: 完成 ({time.perf_counter()-t0:.0f}s)', flush=True)

# ═══════════════════════════════════════════════════════════════════════
# 3. 输出
# ═══════════════════════════════════════════════════════════════════════
np.savez_compressed(os.path.join(OUT, 'wp3b_grid_climate.npz'),
                    p_fy=p_fy, hours=hours, valid=valid,
                    lon=glon, lat=glat, years=np.array(YEARS), ws=WS, wd=WD)
print(f'有效格点: {valid.sum()}/{G} | 总耗时 {time.perf_counter()-t0:.0f}s')
