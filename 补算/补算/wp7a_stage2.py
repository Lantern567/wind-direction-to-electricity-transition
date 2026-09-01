"""
WP7a 阶段 2：真实排布 E(θ) 曲线（FFT 卷积 wp3 联合风况）
========================================================
输入：wp7a_real_eta.npz（阶段 1 查表）+ wp3_climate_joint.npz
输出：wp7a_real_curves.npz
  E_pool [171,72]  多年平均 E(θ)（kWh/台·年，72 风向 5°）
  E_y    [171,11,72] 逐年
  A      [171] 半圆 36 档响应幅度（与 wp5 模板 A 同口径）
  A_full [171] 全圆
  th_star[171] 最优相位（度，半圆）
  E_base [171] = E(0°)（原坐标=建成朝向）| S1 [171] = max E(θ)
  G_so [171,5] 历史期选角→评价期增益 | pos_frac | regret
  drift [171] 逐年最优相位漂移（圆周标准差，度）
  area_km2 [171] 机位凸包面积（单位海域口径用）
"""
import os, io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')

NHALF = 36
HOURS = 8760.0
HIST_YEARS = list(range(2014, 2020))
EVAL_YEARS = list(range(2020, 2025))

z = np.load(os.path.join(OUT, 'wp7a_real_eta.npz'), allow_pickle=True)
ETA_R = z['eta'].astype(np.float64)          # (171,18,72)
P0 = z['P0'].astype(np.float64)
farm_ids = z['farm_ids'].astype(int)
n_turb = z['n_turb'].astype(int)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
# wp3 按 farm_id 升序，wp7a 按 n_turb 升序 → 显式重排
# 目标：p_fy 第 i 行 = farm_ids[i] 的风况（与 ETA_R 第 i 行同一场址）
# 2026-08-15 修正：原实现 idx=[pos[z3_farm]] 再取 p[idx] 实际把 p 映射回了
# z3 原序（恒等重排），导致第 i 行 η(场址 B_i) 配 p(场址 A_i) 错位
# ——54 场 E(0°)>E_nowake、typematch 0.307 皆源于此。
pos3 = {int(f): j for j, f in enumerate(z3['farm_ids'].astype(int))}
idx = np.array([pos3[int(f)] for f in farm_ids])
p_fy = z3['p_fy'].astype(np.float64)[idx]    # (171,11,18,72) 与 farm_ids 同序

nF = len(farm_ids)
E_y = np.zeros((nF, 11, 72))
t0 = time.perf_counter()
for i in range(nF):
    FFTp = np.fft.fft(p_fy[i], axis=2)                       # (11,18,72)c
    FFT_ETA = np.fft.fft(ETA_R[i], axis=1)                  # (18,72)c
    # E(θ)=Σ p(d)·η((d+θ) mod 72)（θ=排布逆时针，与 wp5 新约定一致）
    G = np.fft.ifft(np.conj(FFTp) * FFT_ETA[None, :, :], axis=2).real   # (11,18,72)
    E_y[i] = np.einsum('u,yud->yd', P0, G) * HOURS / 1000.0
print(f'E(θ) 曲线完成 ({time.perf_counter()-t0:.0f}s)')

E_pool = E_y.mean(axis=1)
Eh = E_pool[:, :NHALF]
A = 100 * (Eh.max(axis=1) - Eh.mean(axis=1)) / np.maximum(Eh.mean(axis=1), 1e-9)
Afull = 100 * (E_pool.max(axis=1) - E_pool.mean(axis=1)) / np.maximum(E_pool.mean(axis=1), 1e-9)
th_star = np.argmax(Eh, axis=1) * 5
E_base = E_pool[:, 0]                       # 原坐标 = 建成朝向
S1 = E_pool.max(axis=1)

# 样本外：历史期选角 → 评价期
E_hist = E_y[:, [HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=1)
th_hist = np.argmax(E_hist[:, :NHALF], axis=1) * 5
E_eval = E_y[:, [EVAL_YEARS.index(y) for y in EVAL_YEARS]]         # (171,5,72)
E_sel = E_eval[np.arange(nF), :, th_hist // 5]
G_so = 100 * (E_sel - E_eval[:, :, :NHALF].mean(axis=2)) / np.maximum(E_eval[:, :, :NHALF].mean(axis=2), 1e-9)
pos_frac = (G_so > 0).mean(axis=1)
# 后悔值：逐年最优 vs 历史选角（评价期，kWh/台·年）
E_perf = E_eval[:, :, :NHALF].max(axis=2)
regret = (E_perf - E_sel).mean(axis=1)
# 相位漂移
circ = np.deg2rad(np.argmax(E_y[:, :, :NHALF], axis=2).astype(float) * 5)
Csum = np.abs(np.exp(1j * circ).mean(axis=1))
drift = np.sqrt(-2 * np.log(np.maximum(Csum, 1e-12))) * 57.2958 / np.pi

# 单位海域口径：机位凸包面积
area = np.full(nF, np.nan)
tc = pd.read_csv(os.path.join(REPO, 'data', 'task0', 'turbine_coordinates.csv'), encoding='utf-8-sig')
tc = tc.sort_values(['farm_id', 'year'])
ly = tc.groupby('farm_id')['year'].max().reset_index()
tcl = tc.merge(ly, on=['farm_id', 'year'])
from pyproj import Transformer, CRS
for i, fid in enumerate(farm_ids):
    g = tcl[tcl.farm_id == fid]
    lon = g['lon'].values.astype(float); lat = g['lat'].values.astype(float)
    if len(lon) < 3:
        continue
    clon, clat = float(np.median(lon)), float(np.median(lat))
    tf = Transformer.from_crs(CRS.from_epsg(4326),
                              CRS.from_proj4(f'+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m'),
                              always_xy=True)
    x, y = tf.transform(lon, lat)
    try:
        area[i] = ConvexHull(np.column_stack([x, y])).volume / 1e6   # km²
    except Exception:
        pass

np.savez_compressed(os.path.join(OUT, 'wp7a_real_curves.npz'),
                    E_pool=E_pool, E_y=E_y, A=A, A_full=Afull, th_star=th_star,
                    E_base=E_base, S1=S1, G_so=G_so, pos_frac=pos_frac,
                    regret=regret, drift=drift, area_km2=area,
                    farm_ids=farm_ids, n_turb=n_turb)
print(f'真实排布 A: 中位 {np.median(A):.2f}% | 范围 [{A.min():.2f}, {A.max():.2f}]')
print(f'G_plan(S1, 相对建成朝向): 中位 {np.median(100*(S1-E_base)/E_base):.2f}%')
print(f'样本外正增益年份比例: 中位 {np.median(pos_frac):.2f} | 后悔值中位 {np.median(regret):.1f} kWh/台·年')
print('输出: wp7a_real_curves.npz')
