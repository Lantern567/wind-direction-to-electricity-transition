"""
WP7a 真实排布尾流查表（结论四 S1/E_base 基础）
===============================================
背景：task3_s1_optimal_orientation.csv 的 expected_AEP 与 WP3 独立口径
（missing_82 交叉验证 corr=1.0）系统性偏差 ~20-28%，其风数据源存疑。
故真实排布 E(θ) 用自己的 FLORIS 查表 + WP3 联合风况重算，与 WP4/WP5
完全同口径（同一 FLORIS 配置、同一 18 风速档 × 72 风向网格）。

阶段 1（本脚本）：171 终期排布 × 逐场 TI × 18 风速 × 72 风向 → eta_real
  断点续跑，按 n_turb 升序（大场放最后，结果尽早可用）
阶段 2（wp7a_stage2.py）：FFT 卷积 wp3 p → E(θ) 曲线、A_real、θ*、
  S1* = max E(θ)、E_base = E(0°)（原坐标=建成朝向）、逐年曲线

输出：补算/output/wp7a_real_eta.npz
  eta_real [171, 18, 72] float32 | P0 [18] | ws [18] | wd [72]
  farm_ids [171] | n_turb [171] | xy 列表（joblib）| done 索引
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import create_floris_model, WS_BINS, get_ti_for_farm

WS = np.array([w for w in WS_BINS if w >= 3.0])     # 18 档
WD = np.arange(0.0, 360.0, 5.0)                     # 72 风向
NPZ = os.path.join(OUT, 'wp7a_real_eta.npz')

# ---- 终期机位（与 wp1 相同口径：每场取最后一年）----
DATA = os.path.join(REPO, 'data', 'task0')
tc = pd.read_csv(os.path.join(DATA, 'turbine_coordinates.csv'), encoding='utf-8-sig')
tc = tc.sort_values(['farm_id', 'year'])
last_year = tc.groupby('farm_id')['year'].max().reset_index()
tc_last = tc.merge(last_year, on=['farm_id', 'year'])
print(f'终期机位: {len(tc_last)} 台 / {tc_last.farm_id.nunique()} 场')

geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')

# ---- 逐场局部坐标（aeqd，场心原点，与 wp1 一致）----
farms = []
for fid, g in tc_last.groupby('farm_id'):
    lon = g['lon'].values.astype(float)
    lat = g['lat'].values.astype(float)
    clon, clat = float(np.median(lon)), float(np.median(lat))
    tf = Transformer.from_crs(CRS.from_epsg(4326),
                              CRS.from_proj4(f'+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m'),
                              always_xy=True)
    x, y = tf.transform(lon, lat)
    farms.append((int(fid), np.asarray(x), np.asarray(y), len(g)))
farms.sort(key=lambda t: t[3])                     # n_turb 升序
FIDS = [t[0] for t in farms]
N = [t[3] for t in farms]
XY = np.array([np.column_stack([t[1], t[2]]) for t in farms], dtype=object)

# ---- 单机功率 P0（与 wp4 相同）----
fm0, tmp0 = create_floris_model([0.0], [0.0], turbine_type='iea_10MW', ti=0.07,
                                alpha=0.11, wake_model_name='gauss')
fm0.set(wind_speeds=WS.copy(), wind_directions=np.full(len(WS), 270.0),
        turbulence_intensities=np.full(len(WS), 0.07))
fm0.run()
P0 = fm0.get_farm_power().flatten().astype(np.float64)
os.unlink(tmp0)

# ---- 断点续跑 ----
nF = len(FIDS)
eta = np.full((nF, len(WS), len(WD)), np.nan, dtype=np.float32)
done = set()
if os.path.exists(NPZ):
    z = np.load(NPZ, allow_pickle=True)
    eta[:z['eta'].shape[0]] = z['eta']
    done = set(map(int, z['done'])) if 'done' in z.files else set()
    print(f'断点续跑: 已完成 {len(done)}/{nF}')

def save():
    np.savez_compressed(NPZ, eta=eta, P0=P0, ws=WS, wd=WD,
                        farm_ids=np.array(FIDS), n_turb=np.array(N),
                        done=np.array(sorted(done), dtype=int))

t_start = time.perf_counter()
for i, fid in enumerate(FIDS):
    if i in done:
        continue
    x = XY[i][:, 0]; y = XY[i][:, 1]
    ti = get_ti_for_farm(geo.loc[fid, 'cent_lat'], geo.loc[fid, 'cent_lon'])
    fm, tmp = create_floris_model(x, y, turbine_type='iea_10MW', ti=ti,
                                  alpha=0.11, wake_model_name='gauss')
    P = np.zeros((len(WS), len(WD)))
    for j, w in enumerate(WS):
        fm.set(wind_speeds=np.full(len(WD), float(w)),
               wind_directions=WD.astype(float),
               turbulence_intensities=np.full(len(WD), ti))
        fm.run()
        P[j] = fm.get_farm_power().flatten()
    os.unlink(tmp)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = P / (N[i] * P0[:, None])
    e[~np.isfinite(e)] = 1.0
    e[P0 < 1e-3, :] = 1.0
    eta[i] = e
    done.add(i)
    el = time.perf_counter() - t_start
    print(f'[{len(done)}/{nF}] farm {fid:3d} (n={N[i]:4d}) TI={ti:.2f} | '
          f'已用时 {el/60:.1f} min', flush=True)
    if i % 5 == 4 or i == nF - 1:
        save()

save()
print(f'完成 | 总耗时 {(time.perf_counter()-t_start)/60:.1f} min | 输出 {NPZ}')
