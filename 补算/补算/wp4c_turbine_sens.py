"""
WP4c-S 机型敏感性查表：nrel_5MW 与 iea_15MW（对应建设年代敏感性，学长要求）
==============================================================================
与 wp4c 完全同配置（gauss / crespo_hernandez / sosfs / ka=0.38 kb=0.004 /
alpha=0.11 / 5 TI × 18 风速档 × 72 风向），仅换机型；
六套范式情境布局按机型叶轮直径等比缩放（情境定义是 D 的倍数，D 变则布局变）。

输出：补算/output/wp4c_sens_{turbine}.npz（结构与 wp4c_wake_lookup.npz 相同）
断点续跑：已完成的 情境-TI 组合跳过。
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

from floris_config import create_floris_model, WS_BINS, REGION_TI, get_turbine_params

WS = np.array([w for w in WS_BINS if w >= 3.0])     # 18 档
WD = np.arange(0.0, 360.0, 5.0)                     # 72 风向
TI_VALS = sorted(set(REGION_TI.values()))           # 5 个 TI
N_TURB = 64
D_REF = 198.0                                       # iea_10MW 叶轮直径

TURBINES = ['nrel_5MW', 'iea_15MW']

# ---- 情境布局（基准 iea_10MW 口径，米）----
ly = pd.read_csv(os.path.join(OUT, 'wp2c_paradigm_layouts.csv'))
PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
XY0 = np.zeros((len(PIDS), N_TURB, 2), dtype=np.float64)
for k, pid in enumerate(PIDS):
    g = ly[ly.paradigm == pid].sort_values('turbine_i')
    assert len(g) == N_TURB
    XY0[k, :, 0] = g['x_m'].values
    XY0[k, :, 1] = g['y_m'].values

for turb in TURBINES:
    tp = get_turbine_params(turb)
    D = float(tp['D'])
    scale = D / D_REF
    XY = XY0 * scale
    NPZ = os.path.join(OUT, f'wp4c_sens_{turb}.npz')
    t0 = time.perf_counter()

    # ---- 单机功率基准 P0 ----
    fm0, tmp0 = create_floris_model([0.0], [0.0], turbine_type=turb,
                                    ti=0.07, alpha=0.11, wake_model_name='gauss')
    fm0.set(wind_speeds=WS.copy(), wind_directions=np.full(len(WS), 270.0),
            turbulence_intensities=np.full(len(WS), 0.07))
    fm0.run()
    P0 = fm0.get_farm_power().flatten().astype(np.float64)
    os.unlink(tmp0)
    print(f'[{turb}] D={D:.0f}m scale={scale:.3f} | P0@3m/s={P0[0]:.3e} W', flush=True)

    # ---- 断点续跑 ----
    eta = np.full((len(PIDS), len(TI_VALS), len(WS), len(WD)), np.nan, dtype=np.float32)
    done = set()
    if os.path.exists(NPZ):
        z = np.load(NPZ)
        eta = z['eta'].astype(np.float64)
        done = set(map(int, z['done'])) if 'done' in z.files else set()
        print(f'  断点续跑: 已完成 {len(done)} 个情境-TI 组合', flush=True)

    def save():
        np.savez_compressed(NPZ, eta=eta.astype(np.float32), P0=P0, ws=WS, wd=WD,
                            ti=np.array(TI_VALS), pid=np.array(PIDS), xy=XY,
                            done=np.array(sorted(done), dtype=int))

    t_start = time.perf_counter()
    n_combos = len(PIDS) * len(TI_VALS)
    for k, pid in enumerate(PIDS):
        x = XY[k, :, 0]; y = XY[k, :, 1]
        for j, ti in enumerate(TI_VALS):
            if k * len(TI_VALS) + j in done:
                continue
            fm, tmp = create_floris_model(x, y, turbine_type=turb, ti=ti,
                                          alpha=0.11, wake_model_name='gauss')
            P = np.zeros((len(WS), len(WD)))
            for i, w in enumerate(WS):
                fm.set(wind_speeds=np.full(len(WD), float(w)),
                       wind_directions=WD.astype(float),
                       turbulence_intensities=np.full(len(WD), ti))
                fm.run()
                P[i] = fm.get_farm_power().flatten()
            os.unlink(tmp)
            with np.errstate(divide='ignore', invalid='ignore'):
                e = P / (N_TURB * P0[:, None])
            e[~np.isfinite(e)] = 1.0
            e[P0 < 1e-3, :] = 1.0
            eta[k, j] = e
            done.add(k * len(TI_VALS) + j)
            el = time.perf_counter() - t_start
            done_n = len(done)
            print(f'  [{turb} {done_n}/{n_combos}] {pid} TI={ti:.2f} | '
                  f'已用 {el/60:.1f} min | 预计剩余 {(n_combos-done_n)*(el/done_n)/60:.1f} min',
                  flush=True)
        save()

    save()
    print(f'[{turb}] 完成: {n_combos} 组合 | 总耗时 {(time.perf_counter()-t0)/60:.1f} min',
          flush=True)
print('全部机型查表完成')
