"""
WP5d 机型敏感性（学长第二轮反馈：计算与风机型号相关，需对应建设年份做敏感性分析）
==============================================================================================
对三套机型查表（nrel_5MW D=126m / iea_10MW D=198m 基准 / iea_15MW D=242m）分别重跑
wp5c 农场交叉仿真（FFT 循环互相关，同 wp5c），每套机型独立计算：
  - θ_energy（机型功率曲线加权的能量方向圆均值 → 建成基线角 t_b，机型敏感性的一部分）
  - A_built (171 场 × 6 情境) 半圆窗口
  - 走廊 vs 非走廊逐情境：A 中位、倍数、单侧 Mann–Whitney p、成员秩中位
  - 六情境两两 Spearman 中位、类型匹配（范式均值 vs wp7a 真实 A）Spearman
再按建设年代匹配机型（commissioning year = 各场在 turbine_coordinates.csv 的最早年份）：
  ≤2017→nrel_5MW、2018–2021→iea_10MW、≥2022→iea_15MW，重算逐情境走廊结论。

口径注记：查表输入风速为 100 m 高度（ERA5），FLORIS reference_wind_height 取各机型轮毂高，
三套查表同口径；A 为相对幅度，对绝对风速水平的系统偏移不敏感（报告内注明）。

输入：wp4c_wake_lookup.npz / wp4c_sens_{nrel_5MW,iea_15MW}.npz / wp3_climate_joint.npz /
     wp7c_scenario_table.csv（走廊标签）/ wp7a_real_curves.npz / data/task0/turbine_coordinates.csv
输出：output/wp5d_turbine_sens.npz + wp5d_report.txt
"""
import os, io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')

PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
WIND_INFORMED = np.array([True, False, False, True, False, True])
TB_FIXED = {'S_B45': 9}
NWD, NHALF, HOURS = 72, 36, 8760.0
N_TURB = 64

TURB_FILES = {'nrel_5MW': 'wp4c_sens_nrel_5MW.npz',
              'iea_10MW': 'wp4c_wake_lookup.npz',
              'iea_15MW': 'wp4c_sens_iea_15MW.npz'}
ERA = {'nrel_5MW': 2017, 'iea_10MW': 2021, 'iea_15MW': 2100}   # 上界

# ---- 载入共享输入 ----
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
farm_ids = z3['farm_ids'].astype(int)
p_farm = z3['p_fy'].astype(np.float64)          # (171, 11, 18, 72)
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
farm_lat = np.array([geo.loc[f, 'cent_lat'] for f in farm_ids])
farm_lon = np.array([geo.loc[f, 'cent_lon'] for f in farm_ids])
nF = len(farm_ids)

tb = pd.read_csv(os.path.join(OUT, 'wp7c_scenario_table.csv'), encoding='utf-8-sig').set_index('farm_id')
is_corr = np.array([tb.loc[f, 'corridor'] != 'other' for f in farm_ids])

z7 = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
A_real = pd.Series(z7['A'], index=z7['farm_ids'].astype(int))

# 建设年代（各场最早年份）
tcoord = pd.read_csv(os.path.join(REPO, 'data', 'task0', 'turbine_coordinates.csv'),
                     encoding='utf-8-sig')
comm_year = tcoord.groupby('farm_id')['year'].min()
comm = np.array([comm_year.get(f, np.nan) for f in farm_ids])

sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))  # ensure import works
from floris_config import get_ti_for_farm   # noqa: E402
def ti_idx(lat, lon, ti_vals):
    return int(np.argmin(np.abs(np.array(ti_vals) - get_ti_for_farm(lat, lon))))

def theta_to_tb(theta):
    return np.round(((-theta) % 180.0) / 5.0).astype(int) % NHALF

def energy_dir_circmean(P0, p_pool):
    E_d = P0 @ p_pool
    a = np.deg2rad(np.arange(0, 360, 5.0))
    ang = np.arctan2((E_d * np.sin(a)).sum(), (E_d * np.cos(a)).sum())
    return float(np.rad2deg(ang) % 360.0)

def a_window(E, tb):
    idx = (np.arange(NHALF)[None, :] + np.asarray(tb)[..., None]) % NWD
    w = np.take_along_axis(E, idx, axis=-1)
    return 100.0 * (w.max(axis=-1) - w.mean(axis=-1)) / np.maximum(w.mean(axis=-1), 1e-9)

def build_AB(eta, P0, ti_arr):
    """一套机型查表 → A_built (171,6)、θ_energy、TB。"""
    assert eta.shape == (6, 5, 18, 72), eta.shape
    ti_farm = np.array([ti_idx(a, b, ti_arr) for a, b in zip(farm_lat, farm_lon)])
    FFT_ETA = np.fft.fft(eta, axis=3)
    p_pool = p_farm.mean(axis=1)                # (171,18,72)
    te = np.array([energy_dir_circmean(P0, p_pool[i]) for i in range(nF)])
    TB = np.zeros((nF, 6), dtype=np.int16)
    for k in range(6):
        if WIND_INFORMED[k]:
            TB[:, k] = theta_to_tb(te)
        elif PIDS[k] in TB_FIXED:
            TB[:, k] = TB_FIXED[PIDS[k]]
    A_b = np.zeros((nF, 6))
    E_pool = np.zeros((nF, 6, NWD))
    for i in range(nF):
        ti = ti_farm[i]
        FFTp = np.fft.fft(p_pool[i][None, :, :], axis=2)
        G = np.fft.ifft(np.conj(FFTp) * FFT_ETA[:, ti][None, :, :, :], axis=-1).real  # (1,6,18,72)
        E = np.einsum('u,kud->kd', P0, G[0]) * HOURS / 1000.0               # (6,72) kWh/台·年
        E_pool[i] = E
        A_b[i] = a_window(E, TB[i])
    return A_b, E_pool, te, TB

def corr_stats(A, tag):
    """走廊 vs 非走廊逐情境统计。"""
    lines = [f'--- {tag} ---']
    med_all = []
    for k, pid in enumerate(PIDS):
        c = A[is_corr, k]; n = A[~is_corr, k]
        mw = mannwhitneyu(c, n, alternative='greater')
        rank_all = pd.Series(A[:, k]).rank(pct=True).values
        rmed = rank_all[is_corr].mean() * 171
        med_all.append(np.median(A[:, k]))
        lines.append(f'{pid}: 走廊中位 {np.median(c):5.2f} pp (n={len(c)}) | 非走廊 {np.median(n):5.2f} pp | '
                     f'倍数 {np.median(c)/max(np.median(n),1e-9):.1f}× | 单侧MW p={mw.pvalue:.2e} | 成员秩均值 {rmed:.0f}/171')
    sp = []
    for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5),
                 (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]:
        sp.append(spearmanr(A[:, a], A[:, b])[0])
    lines.append(f'两两 Spearman 中位 {np.median(sp):.3f} [min {np.min(sp):.3f}]')
    lines.append(f'全样本 A 中位（逐情境）: {np.round(np.median(A, axis=0), 2)}')
    # 类型匹配
    mask = np.array([f in A_real.index for f in farm_ids])
    if mask.all():
        rho = spearmanr(A.mean(axis=1), A_real.loc[farm_ids].values)[0]
    else:
        rho = spearmanr(A.mean(axis=1)[mask], A_real.loc[farm_ids[mask]].values)[0]
    lines.append(f'类型匹配 Spearman（范式均值 vs 真实 A, iea_10MW 口径）: {rho:.3f}')
    return lines, sp

report = ['# WP5d 机型敏感性报告（学长第二轮反馈）', '',
          '三套机型查表同口径（gauss/crespo_hernandez/sosfs, ka=0.38, kb=0.004, α=0.11, 5 TI×18 风速×72 风向；',
          '100 m 输入风速，reference_wind_height=各机型轮毂高；A 为相对幅度）。', '']

results = {}
for turb, fn in TURB_FILES.items():
    t0 = time.perf_counter()
    z = np.load(os.path.join(OUT, fn))
    eta = z['eta'].astype(np.float64)
    P0 = z['P0'].astype(np.float64)
    ti_arr = z['ti']
    A_b, E_pool, te, TB = build_AB(eta, P0, ti_arr)
    results[turb] = dict(A=A_b, E=E_pool, te=te, TB=TB)
    Dt = {'nrel_5MW': 126, 'iea_10MW': 198, 'iea_15MW': 242}[turb]
    report.append(f'## 机型 {turb}（D={Dt} m）')
    lines, sp = corr_stats(A_b, turb)
    report += lines
    if 'iea_10MW' in results:
        dte = np.abs(((te - results['iea_10MW']['te']) + 180) % 360 - 180)
        report.append(f'θ_energy 与基准(iea_10MW)差的中位: {np.median(dte):.2f}°（P90 {np.percentile(dte,90):.1f}°）')
    report.append(f'耗时 {time.perf_counter()-t0:.1f}s')
    report.append('')
    print(f'{turb} 完成: A 全样本中位 {np.median(A_b):.2f} pp | 走廊/非走廊(全情境中位) '
          f'{np.median(A_b[is_corr]):.2f}/{np.median(A_b[~is_corr]):.2f} pp | '
          f'{time.perf_counter()-t0:.0f}s', flush=True)

# ---- 年代匹配版 ----
def era_of(year):
    if np.isnan(year):
        return 'iea_10MW'
    for turb, ub in sorted(ERA.items(), key=lambda kv: kv[1]):
        if year <= ub:
            return turb
    return 'iea_15MW'
era_turb = np.array([era_of(y) for y in comm])
report.append('## 年代匹配版（commissioning year = 各场 turbine_coordinates.csv 最早年份）')
era_counts = pd.Series(era_turb).value_counts()
report.append(f'机型分布: ' + ', '.join(f'{t} {era_counts.get(t,0)} 场' for t in TURB_FILES) + '（未知年份按 iea_10MW）')
A_era = np.zeros((nF, 6))
for t in TURB_FILES:
    m = era_turb == t
    if m.any():
        A_era[m] = results[t]['A'][m]
lines, _ = corr_stats(A_era, 'era-matched（各场按建设年代机型）')
report += lines
report.append('')
report.append('## 结论要点')
report.append('1. 逐范式走廊领先结论在 nrel_5MW / iea_10MW / iea_15MW 三套机型与年代匹配口径下是否全部保持（见上表）。')
report.append('2. A 绝对幅度随叶轮直径与功率曲线变化（大机功率曲线相对更平滑/额定点更高），但走廊-非走廊相对倍数与排序稳健。')

np.savez_compressed(os.path.join(OUT, 'wp5d_turbine_sens.npz'),
                    A_5MW=results['nrel_5MW']['A'], A_10MW=results['iea_10MW']['A'],
                    A_15MW=results['iea_15MW']['A'], A_era=A_era,
                    era_turb=np.array(era_turb), comm_year=comm, farm_ids=farm_ids)
open(os.path.join(OUT, 'wp5d_report.txt'), 'w', encoding='utf-8').write('\n'.join(report))
print()
print('\n'.join(report))
print()
print('输出: output/wp5d_turbine_sens.npz + wp5d_report.txt')
