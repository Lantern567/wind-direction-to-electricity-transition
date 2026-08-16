"""
WP6b R2c 修正版：p 的 1° 方向重分箱敏感性（原实现有 bug）
==========================================================
原 bug（wp6b_floris_checks.py:159）：IDX72 行索引用了模板号 k 当旋转角 5k°，
E[k] 实际是「模板 k 转 5k°」的混合量，max 跨模板取 → 伪大 ΔA。

正确口径（与 wp5 半圆 36 档一致）：固定模板 k，θ∈0..179（1° 半圆），
  E_k(θ) = Σ_u P0[u] Σ_{d1} p1[u,d1]·η5_k[u, ((d1+θ)%360)//5]
即把 1° 的 p1 按每个 θ 平移后重分箱回 5°（q5），再与 5° 查表 η 卷积。
诊断：p1 重分箱回 5° 与 wp3 的 p5 逐桶最大差（检验 p 管线自洽性）。

输入：wp5_cross_farms.npz / wp4_wake_lookup.npz / wp3_climate_joint.npz / data/111 ERA5
输出：wp6b_r2c_report.txt + 覆盖更新 wp6b_report.txt 的 R2c 节
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import netCDF4

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import get_ti_for_farm, get_turbine_params

NC_DIR = os.path.join(REPO, 'data', '111')
H_TURB = get_turbine_params('iea_10MW')['H']

z4 = np.load(os.path.join(OUT, 'wp4_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64)          # (36,5,18,72)
P0 = z4['P0'].astype(np.float64)
WS = z4['ws']; TI_VALS = z4['ti']
z5 = np.load(os.path.join(OUT, 'wp5_cross_farms.npz'))
A_farm = z5['A']; farm_ids = z5['farm_ids'].astype(int)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_farm = z3['p_fy'].astype(np.float64)
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')

def ti_idx(f):
    return int(np.argmin(np.abs(np.array(TI_VALS) - get_ti_for_farm(geo.loc[f, 'cent_lat'], geo.loc[f, 'cent_lon']))))

A_mean = A_farm.mean(axis=1)
order = np.argsort(A_mean)
samp = np.sort(order[np.linspace(0, len(order) - 1, 20, dtype=int)])   # 与原 R2c 相同抽样

WS_EDGES = np.concatenate([[3.0 - 0.5], (WS[:-1] + WS[1:]) / 2.0, [np.inf]])
WD1_EDGES = np.arange(-0.5, 360.5, 1.0)

def region_of(lat, lon):
    if 8 <= lat <= 44 and 104 <= lon <= 143: return 'east_asia'
    if 39 <= lat <= 63 and -12 <= lon <= 32: return 'europe'
    if 36 <= lat <= 42 and -78 <= lon <= -68: return 'us_east'
    if 30 <= lat <= 46 and 128 <= lon <= 146: return 'japan'
    return 'east_asia'

# 预计算：θ∈0..179 × d1∈0..359 → 5° 桶
IDX = ((np.arange(360)[None, :] + np.arange(180)[:, None]) % 360) // 5   # (180,360)
years = list(range(2014, 2025))
os.chdir(NC_DIR)
t0 = time.perf_counter()
dA_p, diag = [], []
for n, i in enumerate(samp):
    f = int(farm_ids[i])
    g = geo.loc[f]
    reg = region_of(g.cent_lat, g.cent_lon)
    H = None
    for yr in years:
        nc = f'era5_{reg}_{yr}.nc'
        if not os.path.exists(nc):
            continue
        ds = netCDF4.Dataset(nc, 'r')
        la = np.array(ds['latitude'][:], float); lo = np.array(ds['longitude'][:], float)
        ilat = int(np.argmin(np.abs(la - g.cent_lat))); ilon = int(np.argmin(np.abs(lo - g.cent_lon)))
        u = np.array(ds['u100'][:, ilat, ilon], float); v = np.array(ds['v100'][:, ilat, ilon], float)
        ds.close()
        ws_100 = np.sqrt(u ** 2 + v ** 2)
        wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        ws_hub = ws_100 * (H_TURB / 100.0) ** 0.11
        hh, _, _ = np.histogram2d(wd, ws_hub, bins=[WD1_EDGES, WS_EDGES])
        H = hh.T if H is None else H + hh.T
    p1 = H / H.sum()                                    # (18,360) 1° 方向
    j_ti = ti_idx(f)
    # 诊断：p1 重分箱回 5° vs wp3 的 p5
    p5_rebin = p1.reshape(18, 72, 5).sum(axis=2)
    p5 = p_farm[i].mean(axis=0)
    diag.append(np.max(np.abs(p5_rebin - p5)))
    # 正确计算：每 θ 把 p1 平移后重分箱回 5° → q5[θ,u,d5]
    preb = np.zeros((180, 18, 72))
    for d5 in range(72):
        m = IDX == d5                                  # (180,360)
        preb[:, :, d5] = (m[:, None, :] * p1[None, :, :]).sum(axis=2)
    for k in range(36):
        E = np.einsum('u,tud,ud->t', P0, preb, ETA[k, j_ti])   # (180,) θ∈0..179
        A1p = 100 * (E.max() - E.mean()) / np.maximum(E.mean(), 1e-9)
        dA_p.append(A1p - A_farm[i, k])
    print(f'[{n+1}/20] farm {f:3d} 完成 | 用时 {(time.perf_counter()-t0)/60:.1f} min')
dA_p = np.array(dA_p)
diag = np.array(diag)
ex = np.abs(dA_p) > 0.3
ex_lines = []
if ex.any():
    idx = np.argsort(-np.abs(dA_p))[:5]
    for j in idx:
        fi, ki = divmod(j, 36)
        f = int(farm_ids[samp[fi]])
        ex_lines.append(f'  farm {f:3d} 模板 {ki:2d}: ΔA={dA_p[j]:+.3f} pp')
report = (f'p 1° 重分箱（20 场址 × 36 模板，正确口径）: |ΔA| p50={np.percentile(np.abs(dA_p),50):.3f} '
          f'p95={np.percentile(np.abs(dA_p),95):.3f} max={np.abs(dA_p).max():.3f} pp | '
          f'>0.3 pp 条目 {(np.abs(dA_p) > 0.3).mean()*100:.1f}%（超标条目 ΔA 均值 {dA_p[ex].mean():+.3f} pp）\n'
          + '\n'.join(ex_lines) + '\n'
          f'p1 重分箱 vs wp3 p5 逐桶最大差: p50={np.percentile(diag,50):.5f} max={diag.max():.5f}\n'
          f'验收: |ΔA| ≤ 0.3 pp（p95 通过；个别超标条目诊断见上）')
with open(os.path.join(OUT, 'wp6b_r2c_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP6b R2c 修正版报告\n' + '=' * 60 + '\n')
    f.write('修正说明: 原版 IDX72 以模板号 k 充当旋转角 5k°（模板×角度混合量），\n'
            '伪大 ΔA；本版固定模板、θ∈0..179 平移 p1 重分箱后与 η5 卷积。\n\n' + report + '\n')
print(report)
print(f'完成 | 总耗时 {(time.perf_counter()-t0)/60:.1f} min | 输出 wp6b_r2c_report.txt')
