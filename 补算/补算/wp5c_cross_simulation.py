"""
WP5c 建设范式交叉仿真：场址气候 × 6 套范式情境（S_A/S_B0/S_B45/S_C/S_D/S_E）
=============================================================================
学长批准口径（建设范式情境设计 v3 §3.3）：A 以"建成基线角"锚定半圆窗口——
  S_A/S_C/S_E（风况知情）：建成朝向 = 行轴 ⊥ 能量风向 θ_energy
      → t_b = (−θ_energy mod 180)/5°；A_built = 半圆窗口 [t_b, t_b+36) 的
      (max−mean)/mean——"在范式建设基准之上，主轴朝向的边际影响"
  S_B0/S_B45（约束优先）：t_b = 0/9（固定地理约束轴）——直接衡量朝向自由度
  S_D（风资源梯度）：t_b = 0（WPD 梯度轴地理锚定，生成时沿 +x）
θ_energy：wp3 联合气候能量加权方向圆均值（与仿真气候同源，农场+格点统一口径；
task1 theta_energy_hist 仅作对照注记——同定义、不同风数据提取口径）。

同一仿真链：wp4c FLORIS 查表 η(u,d)（6 情境 × 5 TI × 18 风速 × 72 风向）→
E(θ) = 8760·Σ p(u,d)·P0(u)·η(u,(d+θ) mod 72)（FFT 循环互相关，wp6b R1 已验证
"排布逆时针 θ ≡ 风况平移 +θ"；对 6 套新情境的直算抽查见 wp5c_spotcheck.py）。

稳健性证据链（结果三新叙事）：
  A>0.05pp 普遍性（每范式 + 全 6 范式同时）、最差范式 A（走廊 vs 非走廊）、
  两两范式排序 Spearman、走廊成员跨范式保持前四分位、C0–C3 反事实、
  历史期选角→评价期样本外收益份额、范式 ANOVA（气候/范式/交互 η²）。

输出：wp5c_cross_farms.npz / wp5c_cross_grid.npz / wp5c_farm_cross.csv /
      wp5c_rfd_grid.csv / wp5c_anova.txt / wp5c_report.txt
"""
import os, io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import get_ti_for_farm

PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
WIND_INFORMED = np.array([True, False, False, True, False, True])   # 建成轴是否由 θ_energy 决定
TB_FIXED = {'S_B45': 9}                                             # 固定轴范式（其余固定轴 = 0）
NWD = 72
NHALF = 36                      # 半圆 180°
HIST_YEARS = list(range(2014, 2020))
EVAL_YEARS = list(range(2020, 2025))
HOURS = 8760.0

# ═══════════════════════════════════════════════════════════════════════
# 0. 载入
# ═══════════════════════════════════════════════════════════════════════
z4 = np.load(os.path.join(OUT, 'wp4c_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64)          # (6, 5, 18, 72)
P0 = z4['P0'].astype(np.float64)            # (18,)
TI_VALS = z4['ti']
assert list(z4['pid']) == PIDS, 'wp4c 情境顺序不符'
n_para = len(PIDS)

z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
farm_ids = z3['farm_ids'].astype(int)
p_farm = z3['p_fy'].astype(np.float64)      # (171, 11, 18, 72)
WS = z3['ws']
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
farm_lat = np.array([geo.loc[f, 'cent_lat'] for f in farm_ids])
farm_lon = np.array([geo.loc[f, 'cent_lon'] for f in farm_ids])

z3b = np.load(os.path.join(OUT, 'wp3b_grid_climate.npz'))
p_grid = z3b['p_fy'].astype(np.float64)     # (1446, 11, 18, 72)
grid_valid = z3b['valid']
grid_lon = z3b['lon']; grid_lat = z3b['lat']

cls = pd.read_csv(os.path.join(BUSH, 'input_task1', 'task1_paradigm_classification.csv'),
                  encoding='utf-8-sig').set_index('farm_id')
te_task1 = pd.Series(cls['theta_energy_hist'], dtype=float)   # 72/171 场有值
print(f'数据: 农场 {len(farm_ids)} | 格点 {len(grid_lat)} (有效 {grid_valid.sum()}) | 情境 {n_para}')

# ═══════════════════════════════════════════════════════════════════════
# 1. θ_energy：wp3 联合气候能量加权方向圆均值（与仿真气候同源，171 场 + 格点统一口径）
#    task1 theta_energy_hist 仅作对照注记（同定义、不同风数据提取口径）
# ═══════════════════════════════════════════════════════════════════════
def energy_dir_circmean(p_pool):
    """p_pool: (18,72)。能量加权方向圆均值（0-360，风向 FROM 约定）。"""
    E_d = P0 @ p_pool                           # (72,)
    a = np.deg2rad(np.arange(0, 360, 5.0))
    ang = np.arctan2((E_d * np.sin(a)).sum(), (E_d * np.cos(a)).sum())
    return float(np.rad2deg(ang) % 360.0)

p_farm_pool = p_farm.mean(axis=1)              # (171,18,72)
te = np.array([energy_dir_circmean(p_farm_pool[i]) for i in range(len(farm_ids))])
# 对照注记：与 task1 theta_energy_hist（72/171 场有值）的一致性
have_task1 = np.array([f in te_task1.index and pd.notna(te_task1[f]) for f in farm_ids])
d_te = np.abs(((te_task1[farm_ids[have_task1]].values - te[have_task1]) + 180) % 360 - 180)
print(f'θ_energy: wp3 联合气候能量圆均值（全 {len(farm_ids)} 场统一口径）；'
      f'task1 口径对照（{have_task1.sum()} 场）中位差 {np.median(d_te):.1f}° '
      f'(P90 {np.percentile(d_te, 90):.1f}°，风数据提取口径差异，不影响稳健性结论)')

def theta_to_tb(theta):
    """θ_energy(风向FROM, 0-360) → 建成基线角 bin t_b ∈ [0,36)（行轴 ⊥ 风向）。"""
    return np.round(((-theta) % 180.0) / 5.0).astype(int) % NHALF

TB = np.zeros((len(farm_ids), n_para), dtype=np.int16)     # 每(场址,情境)基线 bin
for k, pid in enumerate(PIDS):
    if WIND_INFORMED[k]:
        TB[:, k] = theta_to_tb(te)
    elif pid in TB_FIXED:
        TB[:, k] = TB_FIXED[pid]
    # else: 0（S_B0/S_D 固定轴）

# ═══════════════════════════════════════════════════════════════════════
# 2. FFT 交叉仿真（同 wp5：IFFT(conj(FFT(p))·FFT(η))，wp5c_spotcheck 直算抽查）
# ═══════════════════════════════════════════════════════════════════════
FFT_ETA = np.fft.fft(ETA, axis=3)             # (6,5,18,72)

def e_curve_fft(FFTp, ti):
    """FFTp: (ncase, 18, 72)c → E[t]: (ncase, 6, 72) kWh/台·年。"""
    nc = FFTp.shape[0]
    out = np.zeros((nc, n_para, NWD))
    for k in range(n_para):
        G = np.fft.ifft(np.conj(FFTp) * FFT_ETA[k, ti][None, :, :], axis=2).real
        out[:, k] = np.einsum('u,cud->cd', P0, G)
    return out * HOURS / 1000.0

def a_window(E, tb):
    """A_built：半圆窗口 [tb, tb+36) mod 72 的 (max−mean)/mean ×100 pp。E:(...,72), tb:(...)。"""
    idx = (np.arange(NHALF)[None, :] + np.asarray(tb)[..., None]) % NWD
    w = np.take_along_axis(E, idx, axis=-1)
    return 100.0 * (w.max(axis=-1) - w.mean(axis=-1)) / np.maximum(w.mean(axis=-1), 1e-9)

def theta_star_window(E, tb):
    """窗口内最优相位（相对基线偏移，0-175°）。"""
    idx = (np.arange(NHALF)[None, :] + np.asarray(tb)[..., None]) % NWD
    w = np.take_along_axis(E, idx, axis=-1)
    return np.argmax(w, axis=-1) * 5

def ti_idx(lat, lon):
    return int(np.argmin(np.abs(np.array(TI_VALS) - get_ti_for_farm(lat, lon))))

ti_farm = np.array([ti_idx(a, b) for a, b in zip(farm_lat, farm_lon)])
ti_grid = np.array([ti_idx(a, b) for a, b in zip(grid_lat, grid_lon)])

def counterfactuals(p):
    pu = p.sum(axis=1); pd = p.sum(axis=0)
    C0 = p
    C1 = pu[:, None] * pd[None, :]
    C2 = P_REF[:, None] * pd[None, :]
    C3 = pu[:, None] * np.full((1, NWD), 1.0 / NWD)
    return np.stack([C0, C1, C2, C3])

p_gpool = np.nanmean(p_grid, axis=1)
g_fin = grid_valid & np.isfinite(p_gpool).all(axis=(1, 2))
pu_grid = p_gpool[g_fin].sum(axis=2)
P_REF = pu_grid.mean(axis=0); P_REF = P_REF / P_REF.sum()
print(f'p_ref 统一风速边际: 均值风速 {np.sum(P_REF*WS):.2f} m/s')

# ═══════════════════════════════════════════════════════════════════════
# 3. 农场交叉仿真
# ═══════════════════════════════════════════════════════════════════════
nF = len(farm_ids)
E_pool = np.zeros((nF, n_para, NWD), dtype=np.float32)     # 多年平均 E(θ)
E_fy = np.zeros((nF, 11, n_para, NWD), dtype=np.float32)
A_b = np.zeros((nF, n_para), dtype=np.float32)             # A_built
A_fixed = np.zeros((nF, n_para), dtype=np.float32)         # A 固定窗口（v1 口径对照）
A_b_c = np.zeros((nF, n_para, 4), dtype=np.float32)        # C0-C3 A_built
A_b_y = np.zeros((nF, 11, n_para), dtype=np.float32)       # 逐年 A_built
th_off = np.zeros((nF, n_para), dtype=np.int16)            # 窗口内最优相位偏移
pos_frac = np.zeros((nF, n_para), dtype=np.float32)
E_nw_pool = HOURS / 1000.0 * np.einsum('u,fyu->fy', P0, p_farm.sum(axis=3)).mean(axis=1)  # (171,)

t0 = time.perf_counter()
for i in range(nF):
    ti = ti_farm[i]
    pfy = p_farm[i]
    FFTp_y = np.fft.fft(pfy, axis=2)
    Ey = e_curve_fft(FFTp_y, ti)                          # (11,6,72)
    E_fy[i] = Ey
    A_b_y[i] = a_window(Ey, TB[i][None, :])
    ppool = pfy.mean(axis=0)
    Ecf = e_curve_fft(np.fft.fft(counterfactuals(ppool), axis=2), ti)   # (4,6,72)
    E_pool[i] = Ecf[0]
    A_b[i] = a_window(Ecf[0], TB[i])
    A_fixed[i] = a_window(Ecf[0], 0)
    A_b_c[i] = np.stack([a_window(Ecf[c], TB[i]) for c in range(4)], axis=1)
    th_off[i] = theta_star_window(Ecf[0], TB[i])
    # 样本外：历史期窗口内选角 → 评价期收益份额
    E_hist = Ey[[HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)   # (6,72)
    th_sel = np.argmax(np.take_along_axis(E_hist, (np.arange(NHALF)[None, :] + TB[i][:, None]) % NWD, axis=-1), axis=-1)  # (6,)
    E_eval = Ey[[EVAL_YEARS.index(y) for y in EVAL_YEARS]]                # (5,6,72)
    idx_sel = (th_sel[None, :] + TB[i][None, :]) % NWD
    E_sel = np.take_along_axis(E_eval, idx_sel[:, :, None], axis=2)[:, :, 0]        # (5,6)
    E_mean_ev = np.take_along_axis(E_eval, (np.arange(NHALF)[None, None, :] + TB[i][None, :, None]) % NWD, axis=2).mean(axis=2)  # (5,6)
    pos_frac[i] = (E_sel > E_mean_ev).mean(axis=0)
print(f'农场交叉仿真完成 ({time.perf_counter()-t0:.0f}s)')

wake_loss_built = 1 - np.stack([E_pool[i, k, TB[i, k]] for i in range(nF) for k in range(n_para)]
                               , axis=0).reshape(nF, n_para) / np.maximum(E_nw_pool[:, None], 1e-9)
wake_loss_rot = 1 - E_pool.mean(axis=-1) / np.maximum(E_nw_pool[:, None], 1e-9)
aep_built = E_pool[np.arange(nF)[:, None], np.arange(n_para)[None, :], TB] / 1e6   # MWh/台·年

np.savez_compressed(os.path.join(OUT, 'wp5c_cross_farms.npz'),
                    E_pool=E_pool, E_fy=E_fy, A=A_b, A_fixed=A_fixed, A_c=A_b_c,
                    A_y=A_b_y, th_off=th_off, pos_frac=pos_frac, TB=TB,
                    wake_loss_built=wake_loss_built, wake_loss_rot=wake_loss_rot,
                    aep_built=aep_built, theta_energy=te, farm_ids=farm_ids)

# ═══════════════════════════════════════════════════════════════════════
# 4. 格点交叉仿真（图 4 稳健地图 + 前四分位参照分布）
# ═══════════════════════════════════════════════════════════════════════
nG = len(grid_lat)
A_grid = np.full((nG, n_para), np.nan)
A_grid_c = np.full((nG, n_para, 4), np.nan)
TB_grid = np.zeros((nG, n_para), dtype=np.int16)
t0 = time.perf_counter()
for i in range(nG):
    if not grid_valid[i]:
        continue
    ti = ti_grid[i]
    pfy = p_grid[i]
    ok = np.isfinite(pfy).all(axis=(1, 2))
    if ok.sum() == 0:
        continue
    ppool = pfy[ok].mean(axis=0)
    te_g = energy_dir_circmean(ppool)
    for k, pid in enumerate(PIDS):
        if WIND_INFORMED[k]:
            TB_grid[i, k] = theta_to_tb(te_g)
        elif pid in TB_FIXED:
            TB_grid[i, k] = TB_FIXED[pid]
    Ecf = e_curve_fft(np.fft.fft(counterfactuals(ppool), axis=2), ti)
    A_grid[i] = a_window(Ecf[0], TB_grid[i])
    A_grid_c[i] = np.stack([a_window(Ecf[c], TB_grid[i]) for c in range(4)], axis=1)
print(f'格点交叉仿真完成 ({time.perf_counter()-t0:.0f}s)')

np.savez_compressed(os.path.join(OUT, 'wp5c_cross_grid.npz'),
                    A=A_grid, A_c=A_grid_c, TB=TB_grid, valid=grid_valid,
                    lon=grid_lon, lat=grid_lat)

# ═══════════════════════════════════════════════════════════════════════
# 5. R_g / F_g / D_g（6 情境空间分位）+ 农场分位 + 走廊保持率
# ═══════════════════════════════════════════════════════════════════════
v = grid_valid & np.isfinite(A_grid).all(axis=1)
A_v = A_grid[v]
Q = np.full((nG, n_para), np.nan)
Q[v] = (np.argsort(np.argsort(A_v, axis=0), axis=0) + 1) / v.sum() * 100
R = np.nanmedian(Q, axis=1)
F75 = np.nanmean(Q >= 75, axis=1) * 100
D = np.subtract(*np.nanpercentile(Q, [75, 25], axis=1))
pd.DataFrame(dict(lon=grid_lon, lat=grid_lat, R=R.round(2), F75=F75.round(2), D=D.round(2))
             ).to_csv(os.path.join(OUT, 'wp5c_rfd_grid.csv'), index=False, encoding='utf-8-sig')

Q_farm = np.array([(A_v < A_b[i]).mean(axis=0) * 100 for i in range(nF)])   # (171,6)
np.savez_compressed(os.path.join(OUT, 'wp5c_farm_q.npz'), Q_farm=Q_farm, farm_ids=farm_ids)

CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China_strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
corr_members = {}
for cname, members in CORRIDORS.items():
    mm = [i for i, f in enumerate(farm_ids) if int(f) in members]
    corr_members[cname] = mm

# ═══════════════════════════════════════════════════════════════════════
# 6. 跨范式稳健性（结果三新叙事核心证据）
# ═══════════════════════════════════════════════════════════════════════
# 6.1 A>0.05pp 普遍性
uni_per = (A_b > 0.05).mean(axis=0) * 100
uni_all = (A_b > 0.05).all(axis=1).sum()
# 6.2 最差范式 A（走廊 vs 非走廊）
A_worst = A_b.min(axis=1)
A_med = np.median(A_b, axis=1)
A_span = A_b.max(axis=1) - A_b.min(axis=1)
corr_idx = sorted(set().union(*[set(mm) for mm in corr_members.values()]))
nonc_idx = [i for i in range(nF) if i not in corr_idx]
# 6.3 两两范式排序 Spearman
from scipy.stats import spearmanr
spear = np.zeros((n_para, n_para))
for a in range(n_para):
    for b in range(n_para):
        spear[a, b] = spearmanr(A_b[:, a], A_b[:, b]).statistic
ij = np.triu_indices(n_para, 1)
# 6.4 走廊成员跨范式保持前四分位（Q≥75）
corr_support = {}
for cname, mm in corr_members.items():
    sup = (Q_farm[mm] >= 75).mean(axis=0) * 100
    corr_support[cname] = (len(mm), float(sup.mean()), float(sup.min()), float(sup.max()))
# 6.5 C3 负对照
c3_max_farm = np.nanmax(A_b_c[:, :, 3])
c3_max_grid = np.nanmax(A_grid_c[:, :, 3])

# ═══════════════════════════════════════════════════════════════════════
# 7. 范式 ANOVA（A_built：气候/范式/交互 η²，平衡 171×6）
# ═══════════════════════════════════════════════════════════════════════
grand = A_b.mean()
SST = ((A_b - grand) ** 2).sum()
site_mean = A_b.mean(axis=1)
para_mean = A_b.mean(axis=0)
SS_site = n_para * ((site_mean - grand) ** 2).sum()
SS_para = nF * ((para_mean - grand) ** 2).sum()
SS_int = ((A_b - site_mean[:, None] - para_mean[None, :] + grand) ** 2).sum()
SSs = [SS_site, SS_para, SS_int]
names = ['气候(场址)', '建设范式', '场址×范式']
eta2 = [s / SST * 100 for s in SSs]
anova_lines = ['方差分解（A_built 的 η²，171 场 × 6 情境）', '=' * 50]
for nm, s, e2 in zip(names, SSs, eta2):
    anova_lines.append(f'{nm:<12s} SS={s:.4f}  η²={e2:.2f}%')

# ═══════════════════════════════════════════════════════════════════════
# 8. 农场级输出表 + 报告
# ═══════════════════════════════════════════════════════════════════════
rows = []
for i, fid in enumerate(farm_ids):
    for k, pid in enumerate(PIDS):
        rows.append(dict(farm_id=int(fid), paradigm=pid,
                         A_built=round(float(A_b[i, k]), 4),
                         A_fixed=round(float(A_fixed[i, k]), 4),
                         A_C0=round(float(A_b_c[i, k, 0]), 4), A_C1=round(float(A_b_c[i, k, 1]), 4),
                         A_C2=round(float(A_b_c[i, k, 2]), 4), A_C3=round(float(A_b_c[i, k, 3]), 4),
                         t_built_bin=int(TB[i, k]), theta_energy_deg=round(float(te[i]), 1),
                         theta_star_off_deg=int(th_off[i, k]),
                         pos_frac=round(float(pos_frac[i, k]), 3),
                         wake_loss_built=round(float(wake_loss_built[i, k]), 4),
                         wake_loss_rot=round(float(wake_loss_rot[i, k]), 4),
                         aep_built_MWh=round(float(aep_built[i, k]), 1)))
fc = pd.DataFrame(rows)
fc.to_csv(os.path.join(OUT, 'wp5c_farm_cross.csv'), index=False, encoding='utf-8-sig')

def seg(sel, name):
    m = float(np.median(A_b[sel, :]))
    w = float(np.median(A_worst[sel]))
    return f'  {name:<16s} n={len(sel):3d} A_built 全情境中位 {m:5.2f} pp | 最差范式中位 {w:5.2f} pp | 2A 极差中位 {np.median(A_span[sel]):.2f} pp'

with open(os.path.join(OUT, 'wp5c_anova.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(anova_lines) + '\n')
with open(os.path.join(OUT, 'wp5c_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP5c 建设范式交叉仿真报告（A–E 五范式，学长批准版）\n' + '=' * 60 + '\n')
    f.write(f'场址: {nF} 农场 + {int(v.sum())} 有效格点 | 情境: 6（S_A/S_B0/S_B45/S_C/S_D/S_E）\n')
    f.write(f'A_built 口径: 半圆窗口锚定建成基线角（风况知情范式 t_b=(−θ_energy mod 180)/5°，S_B45 t_b=9，其余 0）\n')
    f.write(f'θ_energy: wp3 联合气候能量圆均值（全 {nF} 场统一口径）；'
            f'task1 口径对照 {have_task1.sum()} 场中位差 {np.median(d_te):.1f}°'
            f'（P90 {np.percentile(d_te, 90):.1f}°，风数据提取口径差异）\n\n')
    f.write('各情境 A_built (农场, %):\n')
    f.write(pd.DataFrame(dict(paradigm=PIDS, mean=A_b.mean(axis=0).round(3),
                              median=np.median(A_b, axis=0).round(3),
                              p25=np.percentile(A_b, 25, axis=0).round(3),
                              p75=np.percentile(A_b, 75, axis=0).round(3))).to_string(index=False) + '\n\n')
    f.write('稳健性证据:\n')
    f.write(f'  A>0.05pp 普遍性: 全 6 情境同时满足 {uni_all}/171 场 | 分情境: ' +
            ', '.join(f'{p} {u:.0f}%' for p, u in zip(PIDS, uni_per)) + '\n')
    f.write(seg(corr_idx, '走廊成员(23场)') + '\n')
    f.write(seg(nonc_idx, '非走廊(148场)') + '\n')
    f.write(f'  两两范式排序 Spearman: 中位 {np.median(spear[ij]):.3f} 范围 [{spear[ij].min():.3f}, {spear[ij].max():.3f}]\n\n')
    f.write('走廊成员跨情境保持前四分位 (Q≥75) 比例 (%):\n')
    for cname, (nm, supm, supmin, supmax) in corr_support.items():
        f.write(f'  {cname:<14s} n={nm:3d} 均值 {supm:5.1f} | 最差 {supmin:5.1f} | 最好 {supmax:5.1f}\n')
    f.write(f'\nC3 均匀方向负对照: 农场 max A = {c3_max_farm:.6f} pp | 格点 max A = {c3_max_grid:.6f} pp | 验收线 0.1 pp → {"通过" if max(c3_max_farm, c3_max_grid) <= 0.1 else "失败"}\n\n')
    f.write('\n'.join(anova_lines) + '\n\n')
    f.write('建成基线处尾流损失（范式设计态）vs 旋转平均尾流损失 (%):\n')
    f.write(pd.DataFrame(dict(paradigm=PIDS,
                              wake_loss_built=wake_loss_built.mean(axis=0).round(2),
                              wake_loss_rot=wake_loss_rot.mean(axis=0).round(2))).to_string(index=False) + '\n')

print('各情境 A_built 农场中位:', np.round(np.median(A_b, axis=0), 2))
print(f'A>0.05pp 全 6 情境: {uni_all}/171')
print(seg(corr_idx, '走廊成员'), '\n', seg(nonc_idx, '非走廊'))
print(f'两两 Spearman: 中位 {np.median(spear[ij]):.3f} [{spear[ij].min():.3f},{spear[ij].max():.3f}]')
print(f'C3 负对照: 农场 {c3_max_farm:.6f} | 格点 {c3_max_grid:.6f} → {"通过" if max(c3_max_farm, c3_max_grid) <= 0.1 else "失败"}')
print('\n'.join(anova_lines))
print(f'输出完成 | 总耗时 {time.perf_counter()-t0:.0f}s')
