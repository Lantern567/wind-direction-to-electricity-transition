"""
WP4 尾流效率查表：36 模板 × 5 TI × 72 风向 × 18 风速档
========================================================
对应《结论三与结论四补充计算方案》§3 执行顺序第 4 步。

与论文主分析同一配置（offshore-task2/floris_config.py）：
  gauss 速度模型 / crespo_hernandez 湍流 / sosfs 组合 /
  ka=0.38 kb=0.004（Niayifar k=0.38·TI+0.004）/ iea_10MW / alpha=0.11
TI 取 REGION_TI 的全部 5 个不同取值 {0.06,0.07,0.08,0.09,0.10}，
下游按场址区域 TI 选取对应层（论文口径逐场 TI）。

FLORIS v4.6.6 校验器要求 wind_speeds/wind_directions 一维等长，
故按"每风速档一次 run、72 风向批量"组织（18 run / 模板-TI，~22.5 s）。

输出：补算/output/wp4_wake_lookup.npz
  eta  [36, 5, 18, 72]  = P_farm(u,θ) / (64·P0(u))   （float32）
  P0   [18]            = 单机功率（W）
  ws   [18], wd [72], ti [5], tid [36], 模板坐标 xy[36][64][2]
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))

from floris_config import create_floris_model, WS_BINS, REGION_TI

WS = np.array([w for w in WS_BINS if w >= 3.0])     # 18 档（0 档无功率，跳过）
WD = np.arange(0.0, 360.0, 5.0)                     # 72 个风向（5°）
TI_VALS = sorted(set(REGION_TI.values()))           # 5 个取值
N_TURB = 64
NPZ = os.path.join(OUT, 'wp4_wake_lookup.npz')

# ---- 模板 ----
tpl = pd.read_csv(os.path.join(OUT, 'wp2_standard_templates.csv'))
summ = pd.read_csv(os.path.join(OUT, 'wp2_template_summary.csv'))
TIDS = summ['template_id'].tolist()

# ---- 单机功率基准 P0（1 台机，无尾流；TI 不影响 P0）----
fm0, tmp0 = create_floris_model([0.0], [0.0], turbine_type='iea_10MW',
                                ti=0.07, alpha=0.11, wake_model_name='gauss')
fm0.set(wind_speeds=WS.copy(), wind_directions=np.full(len(WS), 270.0),
        turbulence_intensities=np.full(len(WS), 0.07))
fm0.run()
P0 = fm0.get_farm_power().flatten().astype(np.float64)   # (18,) W，单机
os.unlink(tmp0)
print(f'P0 ({len(WS)} 档): {P0[0]:.3e} W @3m/s ... {P0[-1]:.3e} W @30m/s')

# ---- 断点续跑 ----
eta = np.full((len(TIDS), len(TI_VALS), len(WS), len(WD)), np.nan, dtype=np.float32)
done = set()
if os.path.exists(NPZ):
    z = np.load(NPZ)
    eta = z['eta'].astype(np.float64)
    done = set(map(int, z['done'])) if 'done' in z.files else set()
    print(f'断点续跑: 已完成 {len(done)} 个模板-TI 组合')

XY = np.zeros((len(TIDS), N_TURB, 2), dtype=np.float64)
for k, tid in enumerate(TIDS):
    g = tpl[tpl.template_id == tid].sort_values('turbine_i')
    XY[k, :, 0] = g['x_m'].values
    XY[k, :, 1] = g['y_m'].values

def save():
    np.savez_compressed(NPZ, eta=eta.astype(np.float32), P0=P0, ws=WS, wd=WD,
                        ti=np.array(TI_VALS), tid=np.array(TIDS), xy=XY,
                        done=np.array(sorted(done), dtype=int))

t_start = time.perf_counter()
n_combos = len(TIDS) * len(TI_VALS)
for k, tid in enumerate(TIDS):
    x = XY[k, :, 0]; y = XY[k, :, 1]
    for j, ti in enumerate(TI_VALS):
        if k * len(TI_VALS) + j in done:
            continue
        fm, tmp = create_floris_model(x, y, turbine_type='iea_10MW', ti=ti,
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
        e[~np.isfinite(e)] = 1.0          # 极低风速无有效尾流 → η=1
        e[P0 < 1e-3, :] = 1.0             # 功率≈0 的风速档（如 30 m/s 切出）无意义 → η=1
        eta[k, j] = e
        done.add(k * len(TI_VALS) + j)
        el = time.perf_counter() - t_start
        done_n = len(done)
        print(f'[{done_n}/{n_combos}] {tid} TI={ti:.2f} | 已用时 {el/60:.1f} min | '
              f'预计剩余 {(n_combos-done_n)*(el/done_n)/60:.1f} min', flush=True)
    save()   # 每模板保存一次（压缩耗时 ~秒级，避免逐组合 I/O）

save()
print(f'完成: {n_combos} 组合 | 总耗时 {(time.perf_counter()-t_start)/60:.1f} min')
print(f'输出: {NPZ}')
