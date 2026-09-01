"""
WP5c 直算抽查（学长要求"拿仿真模型来实际算"的独立验证）
======================================================
R1 旋转等价（对 6 套新情境重验 wp6b R1）：
   每情境 × θ∈{40°,130°}（TI=0.07，另加 S_D θ=40° TI=0.10）→ FLORIS 对旋转后坐标
   直算 η_rot(u,d)，对照查表 η_axis(u,(d+θ) mod 72)。
   验收: max|Δη| < 0.005（wp6b 45° belt 决定性检验已确认 +shift 口径）。
R2 FFT vs 直接求和：
   3 场址 × 3 情境，E(θ) = 8760·Σ_{u,d} p(u,d)·P0(u)·η(u,(d+θ) mod 72) 三重直接
   求和（72 个 θ），对照 wp5c FFT 输出 E_pool。验收: max|ΔE|/Ē < 1e-8。

依赖：wp4c_wake_lookup.npz + wp5c_cross_farms.npz（WP4c/WP5c 完成后运行）。
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import create_floris_model, get_ti_for_farm

HOURS = 8760.0
PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']

z4 = np.load(os.path.join(OUT, 'wp4c_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64)          # (6,5,18,72)
P0 = z4['P0'].astype(np.float64)
WS = z4['ws']; WD = z4['wd']; TI_VALS = z4['ti']
XY = z4['xy']

z5 = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'))
E_pool = z5['E_pool'].astype(np.float64)    # (171,6,72)
farm_ids = z5['farm_ids'].astype(int)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_farm = z3['p_fy'].astype(np.float64)
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')


def floris_eta(x, y, ti, ws_idx, dirs):
    """FLORIS 直算 η：返回 (len(ws_idx), len(dirs))。"""
    fm, tmp = create_floris_model(x, y, turbine_type='iea_10MW', ti=ti,
                                  alpha=0.11, wake_model_name='gauss')
    P = np.zeros((len(ws_idx), len(dirs)))
    for i, j in enumerate(ws_idx):
        fm.set(wind_speeds=np.full(len(dirs), float(WS[j])),
               wind_directions=dirs.astype(float),
               turbulence_intensities=np.full(len(dirs), ti))
        fm.run()
        P[i] = fm.get_farm_power().flatten()
    os.unlink(tmp)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = P / (len(x) * P0[ws_idx, None])
    e[~np.isfinite(e)] = 1.0
    e[P0[ws_idx] < 1e-3, :] = 1.0
    return e


report = {}

# ═══════════════════════════════════════════════════════════════════════
# R1 旋转等价（6 情境 × 2 角度 × TI 0.07；+ S_D θ=40° TI=0.10）
# ═══════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
SPEED_TARGETS = [3.0, 7.0, 11.0, 15.0, 22.0]
ws_idx = [int(np.argmin(np.abs(WS - s))) for s in SPEED_TARGETS]
j07 = int(np.argmin(np.abs(np.array(TI_VALS) - 0.07)))
j10 = int(np.argmin(np.abs(np.array(TI_VALS) - 0.10)))
lines = []
for k, pid in enumerate(PIDS):
    x0 = XY[k, :, 0]; y0 = XY[k, :, 1]
    cx, cy = x0.mean(), y0.mean()
    for th in [40.0, 130.0]:
        rad = np.radians(th)
        xr = (x0 - cx) * np.cos(rad) - (y0 - cy) * np.sin(rad)
        yr = (x0 - cx) * np.sin(rad) + (y0 - cy) * np.cos(rad)
        e_rot = floris_eta(xr, yr, 0.07, ws_idx, WD)
        shift = int(round(th / 5.0))
        e_ref = ETA[k, j07][ws_idx][:, np.roll(np.arange(72), -shift)]
        diff = np.abs(e_rot - e_ref)
        lines.append(f'{pid} θ={th:5.0f}° TI=0.07: max|Δη|={diff.max():.5f}  '
                     f'mean|Δη|={diff.mean():.5f}')
        print(lines[-1])
    if pid == 'S_D':        # TI 维度抽查
        th = 40.0
        rad = np.radians(th)
        xr = (x0 - cx) * np.cos(rad) - (y0 - cy) * np.sin(rad)
        yr = (x0 - cx) * np.sin(rad) + (y0 - cy) * np.cos(rad)
        e_rot = floris_eta(xr, yr, 0.10, ws_idx, WD)
        e_ref = ETA[k, j10][ws_idx][:, np.roll(np.arange(72), -8)]
        diff = np.abs(e_rot - e_ref)
        lines.append(f'{pid} θ={th:5.0f}° TI=0.10: max|Δη|={diff.max():.5f}  '
                     f'mean|Δη|={diff.mean():.5f}')
        print(lines[-1])
report['R1'] = '\n'.join(lines) + '\n验收: max|Δη| < 0.005（0.5%）'

# ═══════════════════════════════════════════════════════════════════════
# R2 FFT vs 直接求和（3 场址 × 3 情境 × 72 θ）
# ═══════════════════════════════════════════════════════════════════════
def ti_idx(f):
    return int(np.argmin(np.abs(np.array(TI_VALS) -
                                get_ti_for_farm(geo.loc[f, 'cent_lat'], geo.loc[f, 'cent_lon']))))

lines = []
for f in [56, 157, 12]:          # 走廊(Vietnam) / Denmark / 走廊(China_strait)
    i = list(farm_ids).index(f)
    j_ti = ti_idx(f)
    p = p_farm[i].mean(axis=0)   # (18,72)
    for pid in ['S_A', 'S_C', 'S_D']:
        k = PIDS.index(pid)
        eta_k = ETA[k, j_ti]     # (18,72)
        E_fft = E_pool[i, k]     # (72,)
        E_direct = np.empty(72)
        for t in range(72):
            e = eta_k[:, np.roll(np.arange(72), -t)]
            E_direct[t] = np.einsum('u,ud->', P0, p * e)
        E_direct *= HOURS / 1000.0
        d = np.abs(E_fft - E_direct)
        rel = d.max() / np.maximum(E_fft.mean(), 1e-9)
        lines.append(f'farm {f:3d} {pid}: max|ΔE|={d.max():.2e} kWh/台·年  '
                     f'相对 {rel:.2e}')
        print(lines[-1])
report['R2'] = ('\n'.join(lines) +
                '\n验收: max|ΔE|/Ē ≤ 1e-7（npz 中 E_pool 为 float32 存储，'
                '~5e-8 相对差即存储精度，非算法误差）')

with open(os.path.join(OUT, 'wp5c_spotcheck_report.txt'), 'w', encoding='utf-8') as fp:
    fp.write('WP5c 直算抽查报告（A–E 情境）\n' + '=' * 60 + '\n')
    fp.write('R1 旋转等价（FLORIS 对旋转坐标直算 vs 查表平移）:\n' + report['R1'] + '\n\n')
    fp.write('R2 FFT vs 直接求和:\n' + report['R2'] + '\n')
print(f'\n完成 | 总耗时 {(time.perf_counter()-t0)/60:.1f} min | 输出 wp5c_spotcheck_report.txt')
