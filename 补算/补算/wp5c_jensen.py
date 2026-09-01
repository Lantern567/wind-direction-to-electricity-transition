# -*- coding: utf-8 -*-
"""WP5c-J 尾流模型敏感性：用 jensen 效率表重算农场交叉仿真
====================================================================
与 wp5c 完全同配置（气候、情境、窗口），仅把尾流效率表换成
wp4c_jensen.npz（jensen 速度模型）。输出关键指标对比：
  - A_built / A_fixed / A_C0-C3（171 场 × 6 范式）
  - 走廊 vs 非走廊 A 中位与倍数
  - 与 gauss 基准的差异
输出：补算/output/wp5c_jensen.csv + wp5c_jensen_report.txt
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BUSH, 'output')
sys.path.insert(0, os.path.join(BUSH, '..', 'offshore-task2'))
from floris_config import get_ti_for_farm

PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
NWD = 72; NHALF = 36; HOURS = 8760.0

# ── 载入 jensen 效率表与气候 ──
z4 = np.load(os.path.join(OUT, 'wp4c_jensen.npz'))
ETA = z4['eta'].astype(np.float64)
P0 = z4['P0'].astype(np.float64)
TI_VALS = z4['ti']
n_para = len(PIDS)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
farm_ids = z3['farm_ids'].astype(int)
p_farm = z3['p_fy'].astype(np.float64)       # (171,11,18,72)
WS = z3['ws']
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
farm_lat = np.array([geo.loc[f, 'cent_lat'] for f in farm_ids])
farm_lon = np.array([geo.loc[f, 'cent_lon'] for f in farm_ids])
z3b = np.load(os.path.join(OUT, 'wp3b_grid_climate.npz'))
p_grid = z3b['p_fy'].astype(np.float64); grid_valid = z3b['valid']

def energy_dir_circmean(p_pool):
    E_d = P0 @ p_pool
    a = np.deg2rad(np.arange(0, 360, 5.0))
    ang = np.arctan2((E_d * np.sin(a)).sum(), (E_d * np.cos(a)).sum())
    return float(np.rad2deg(ang) % 360.0)

p_pool = p_farm.mean(axis=1)
te = np.array([energy_dir_circmean(p_pool[i]) for i in range(len(farm_ids))])
def theta_to_tb(theta):
    return np.round(((-theta) % 180.0) / 5.0).astype(int) % NHALF
WIND_INFORMED = np.array([True, False, False, True, False, True])
TB_FIXED = {'S_B45': 9}
TB = np.zeros((len(farm_ids), n_para), dtype=np.int16)
for k, pid in enumerate(PIDS):
    if WIND_INFORMED[k]:
        TB[:, k] = theta_to_tb(te)
    elif pid in TB_FIXED:
        TB[:, k] = TB_FIXED[pid]

FFT_ETA = np.fft.fft(ETA, axis=3)
def e_curve_fft(FFTp, ti):
    nc = FFTp.shape[0]
    out = np.zeros((nc, n_para, NWD))
    for k in range(n_para):
        G = np.fft.ifft(np.conj(FFTp) * FFT_ETA[k, ti][None, :, :], axis=2).real
        out[:, k] = np.einsum('u,cud->cd', P0, G)
    return out * HOURS / 1000.0

def a_window(E, tb):
    idx = (np.arange(NHALF)[None, :] + np.asarray(tb)[..., None]) % NWD
    w = np.take_along_axis(E, idx, axis=-1)
    return 100.0 * (w.max(axis=-1) - w.mean(axis=-1)) / np.maximum(w.mean(axis=-1), 1e-9)

def ti_idx(lat, lon):
    return int(np.argmin(np.abs(np.array(TI_VALS) - get_ti_for_farm(lat, lon))))
ti_farm = np.array([ti_idx(a, b) for a, b in zip(farm_lat, farm_lon)])

def counterfactuals(p):
    # p: (18,72)；参考统一风速边际取格点气候（同 wp5c）
    pu = p.sum(axis=1); pd = p.sum(axis=0)
    C0 = p
    C1 = pu[:, None] * pd[None, :]
    p_gpool = np.nanmean(p_grid, axis=1)                     # (1446,18,72)
    g_fin = grid_valid & np.isfinite(p_gpool).all(axis=(1, 2))
    pu_grid = p_gpool[g_fin].sum(axis=2)                     # (N,18)
    P_REF = pu_grid.mean(axis=0); P_REF = P_REF / P_REF.sum()
    C2 = P_REF[:, None] * pd[None, :]
    C3 = pu[:, None] * np.full((1, NWD), 1.0 / NWD)
    return np.stack([C0, C1, C2, C3])

nF = len(farm_ids)
A_b = np.zeros((nF, n_para)); A_fixed = np.zeros((nF, n_para))
A_c = np.zeros((nF, n_para, 4))
t0 = time.perf_counter()
for i in range(nF):
    ti = ti_farm[i]
    ppool = p_farm[i].mean(axis=0)
    Ecf = e_curve_fft(np.fft.fft(counterfactuals(ppool), axis=2), ti)   # (4,6,72)
    A_b[i] = a_window(Ecf[0], TB[i])
    A_fixed[i] = a_window(Ecf[0], 0)
    A_c[i] = np.stack([a_window(Ecf[c], TB[i]) for c in range(4)], axis=1)
print(f'jensen 交叉仿真完成 ({time.perf_counter()-t0:.0f}s)')

# ── 保存 ──
df = pd.DataFrame({'farm_id': farm_ids})
for k, pid in enumerate(PIDS):
    df[f'A_built_{pid}'] = A_b[:, k]
    df[f'A_fixed_{pid}'] = A_fixed[:, k]
    df[f'A_C0_{pid}'] = A_c[:, k, 0]
    df[f'A_C1_{pid}'] = A_c[:, k, 1]
    df[f'A_C2_{pid}'] = A_c[:, k, 2]
    df[f'A_C3_{pid}'] = A_c[:, k, 3]
df.to_csv(os.path.join(OUT, 'wp5c_jensen.csv'), index=False)

# ── 报告（与 gauss 基准对比）──
base = pd.read_csv(os.path.join(OUT, 'wp5c_farm_cross.csv'))
w9 = pd.read_csv(os.path.join(OUT, 'wp9c_farm_metrics.csv'))[['farm_id', 'is_corr']]
lines = []
lines.append('WP5c-J 尾流模型敏感性（jensen vs gauss 基准）')
lines.append('=' * 60)
for k, pid in enumerate(PIDS):
    bA = base[base.paradigm == pid].set_index('farm_id')['A_built']
    jA = df.set_index('farm_id')[f'A_built_{pid}']
    jA.index = jA.index.astype(int)
    bA.index = bA.index.astype(int)
    diff = jA - bA
    lines.append(f'{pid}: A_built 中位 jensen={jA.median():.3f} vs gauss={bA.median():.3f} (Δ中位 {diff.median():+.3f}pp, |Δ|中位 {diff.abs().median():.3f})')
# 走廊 vs 非走廊（A_built 六范式平均）
m = df.set_index('farm_id')
corr = w9[w9.is_corr == True].farm_id
corr = corr[corr.isin(m.index)]
base_m = base[base.paradigm == 'S_A'].set_index('farm_id')['A_built']
for pid in PIDS:
    jv = m.loc[corr, f'A_built_{pid}']; bv = base_m.loc[corr] if False else base[base.paradigm==pid].set_index('farm_id')['A_built'].loc[corr]
    jo = m.loc[~m.index.isin(corr), f'A_built_{pid}']; bo = base[base.paradigm==pid].set_index('farm_id')['A_built'].loc[~m.index.isin(corr)]
    lines.append(f'{pid}: 走廊中位 jensen={jv.median():.3f} vs gauss={bv.median():.3f}；倍数 jensen={jv.median()/jo.median():.1f}x vs gauss={bv.median()/bo.median():.1f}x')
with open(os.path.join(OUT, 'wp5c_jensen_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('saved: wp5c_jensen.csv + wp5c_jensen_report.txt')
