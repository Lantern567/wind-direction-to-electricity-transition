"""
WP5 受控交叉仿真：场址气候 × 排布形态 × 间距 × 模板重复 × 年份 × 朝向
=========================================================================
对应《结论三与结论四补充计算方案》§4.2-4.5。

数据流：
  wp3  (171 场址) + wp3b (1,446 格点) 联合风况 p(u,d)
  wp4  (36 模板 × 5 TI) 尾流效率查表 η(u,d) + 单机功率 P0(u)
  ↓
  E(θ) = 8760·Σ_{u,d} p(u,d)·P0(u)·η(u,(d+θ) mod 72)   （θ=+90° 为排布逆时针旋转，
  FLORIS 直算验证见 wp6b R1：排布逆时针 θ ≡ 风况平移 +θ）
  （72 扇区上为循环互相关，用 FFT 实现：E_t = 8760·Σ_u P0[u]·IFFT(conj(FFT(p))·FFT(η))[u,t]）
  ↓
  A（半圆 θ∈{0..175°，36 档}，论文口径）+ 全圆敏感性
  C0-C3 气候反事实（C3 均匀方向负对照，A 必须 ≈ 0）
  逐年 A_y、θ*_y → 相位漂移；历史期 2014-2019 选角 → 2020-2024 样本外
  R_g / F_g / D_g 跨排布稳健性地图（1,446 格点内空间分位）
  方差分解（气候/形态/间距主效应与交互的 η²）

输出：
  wp5_cross_farms.npz  wp5_cross_grid.npz  wp5_rfd_grid.csv
  wp5_farm_cross.csv   wp5_anova.txt       wp5_report.txt
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

WS_BINS_N = 18
NWD = 72
NHALF = 36                      # θ ∈ {0,5,...,175}
HIST_YEARS = list(range(2014, 2020))    # 历史期选角
EVAL_YEARS = list(range(2020, 2025))    # 独立评价期
HOURS = 8760.0

# ═══════════════════════════════════════════════════════════════════════
# 0. 载入
# ═══════════════════════════════════════════════════════════════════════
z4 = np.load(os.path.join(OUT, 'wp4_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64)          # (36, 5, 18, 72)
P0 = z4['P0'].astype(np.float64)            # (18,) W
TI_VALS = z4['ti']
TIDS = list(z4['tid'])
n_tpl = len(TIDS)
summ = pd.read_csv(os.path.join(OUT, 'wp2_template_summary.csv'), encoding='utf-8-sig')
MORPH = summ['morphology'].values           # (36,)
SPAC = summ['spacing_D'].values             # (36,)

z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
farm_ids = z3['farm_ids'].astype(int)
p_farm = z3['p_fy'].astype(np.float64)      # (171, 11, 18, 72)
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
farm_lat = np.array([geo.loc[f, 'cent_lat'] for f in farm_ids])
farm_lon = np.array([geo.loc[f, 'cent_lon'] for f in farm_ids])

z3b = np.load(os.path.join(OUT, 'wp3b_grid_climate.npz'))
p_grid = z3b['p_fy'].astype(np.float64)     # (1446, 11, 18, 72)
grid_valid = z3b['valid']
grid_lon = z3b['lon']; grid_lat = z3b['lat']
print(f'数据: 农场 {len(farm_ids)} | 格点 {len(grid_lat)} (有效 {grid_valid.sum()}) | 模板 {n_tpl}')

# 站点 TI 索引
def ti_idx(lat, lon):
    return int(np.argmin(np.abs(np.array(TI_VALS) - get_ti_for_farm(lat, lon))))
ti_farm = np.array([ti_idx(a, b) for a, b in zip(farm_lat, farm_lon)])
ti_grid = np.array([ti_idx(a, b) for a, b in zip(grid_lat, grid_lon)])

# 预计算 FFT(η)：FFT_ETA[k, ti, u, :]
# 旋转约定（FLORIS 直算已验证，wp6b R1）：排布逆时针旋转 θ ≡ 风况平移 +θ
# → E(θ) = Σ_d p(d)·η((d+θ) mod 72) = IFFT(conj(FFT(p))·FFT(η))[θ]
FFT_ETA = np.fft.fft(ETA, axis=3)                 # (36,5,18,72)

# ═══════════════════════════════════════════════════════════════════════
# 1. FFT 数值验证：直接 roll 求和 vs FFT（随机场址×模板）
# ═══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(7)
p_test = rng.random((18, 72)); p_test /= p_test.sum()
eta_test = ETA[0, 2]
# 逐 u 验证：IFFT(conj(FFT(p))·FFT(η))[u,t] = Σ_d p[u,d]·η[u,(d+t) mod 72]
t_ref = np.stack([np.sum(p_test * np.roll(eta_test, -t, axis=1), axis=1) for t in range(72)], axis=1)
t_fft = np.fft.ifft(np.conj(np.fft.fft(p_test, axis=1)) * np.fft.fft(eta_test, axis=1), axis=1).real
assert np.max(np.abs(t_ref - t_fft)) < 1e-9, 'FFT 循环互相关验证失败'
print('FFT 循环互相关验证通过 (max diff < 1e-9)')

# ═══════════════════════════════════════════════════════════════════════
# 2. 核心函数
# ═══════════════════════════════════════════════════════════════════════
def e_curve_fft(FFTp, ti):
    """FFTp: (ncase, 18, 72) 复数。返回 E[t]: (ncase, n_tpl, 72)，单位 kWh/台·年。"""
    nc = FFTp.shape[0]
    out = np.zeros((nc, n_tpl, NWD))
    for k in range(n_tpl):
        G = np.fft.ifft(np.conj(FFTp) * FFT_ETA[k, ti][None, :, :], axis=2).real  # (nc,18,72)
        out[:, k] = np.einsum('u,cud->cd', P0, G)   # Σ_u P0[u]·G
    return out * HOURS / 1000.0                    # W→kWh/台·年

def a_from_e(E):
    """A（半圆 36 档）+ 全圆 72 档，单位 pp（百分点）。E: (..., 72)。"""
    Eh = E[..., :NHALF]
    Efull = E
    Ah = 100.0 * (Eh.max(axis=-1) - Eh.mean(axis=-1)) / np.maximum(Eh.mean(axis=-1), 1e-9)
    Af = 100.0 * (Efull.max(axis=-1) - Efull.mean(axis=-1)) / np.maximum(Efull.mean(axis=-1), 1e-9)
    return Ah, Af

def counterfactuals(p):
    """p: (18,72)。返回 C0/C1/C2/C3 四种气候 p。p_ref 由外部提供（统一风速边际）。"""
    pu = p.sum(axis=1); pd = p.sum(axis=0)
    C0 = p
    C1 = pu[:, None] * pd[None, :]
    C2 = P_REF[:, None] * pd[None, :]
    C3 = pu[:, None] * np.full((1, NWD), 1.0 / NWD)
    return np.stack([C0, C1, C2, C3])     # (4, 18, 72)

# 统一风速边际 p_ref：有效格点 p_u 的等权平均
p_gpool = np.nanmean(p_grid, axis=1)                       # (1446,18,72)
g_fin = grid_valid & np.isfinite(p_gpool).all(axis=(1, 2))
pu_grid = p_gpool[g_fin].sum(axis=2)                       # (nvalid,18)
P_REF = pu_grid.mean(axis=0)
P_REF = P_REF / P_REF.sum()
print(f'p_ref 统一风速边际: 均值风速 {np.sum(P_REF*np.array(z3["ws"])):.2f} m/s')

# ═══════════════════════════════════════════════════════════════════════
# 3. 农场交叉仿真（171 场址，逐年 + 反事实全存）
# ═══════════════════════════════════════════════════════════════════════
nF = len(farm_ids)
E_fy = np.zeros((nF, 11, n_tpl, NWD), dtype=np.float32)   # 逐年 E(θ)
E_cf = np.zeros((nF, 4, n_tpl, NWD), dtype=np.float32)    # C0-C3 多年平均
A_y = np.zeros((nF, 11, n_tpl), dtype=np.float32)
th_y = np.zeros((nF, 11, n_tpl), dtype=np.int8)
t0 = time.perf_counter()
for i in range(nF):
    ti = ti_farm[i]
    pfy = p_farm[i]                                        # (11,18,72)
    FFTp_y = np.fft.fft(pfy, axis=2)                       # (11,18,72)c
    Ey = e_curve_fft(FFTp_y, ti)                           # (11,36,72)
    E_fy[i] = Ey
    A_y[i], _ = a_from_e(Ey)
    th_y[i] = np.argmax(Ey[..., :NHALF], axis=-1)          # 0..35 → 5°×idx
    # 多年平均 + 反事实
    ppool = pfy.mean(axis=0)
    Ecf = e_curve_fft(np.fft.fft(counterfactuals(ppool), axis=2), ti)   # (4,36,72)
    E_cf[i] = Ecf
print(f'农场交叉仿真完成 ({time.perf_counter()-t0:.0f}s)')

A_farm, A_farm_full = a_from_e(E_cf[:, 0])                 # (171,36)
A_farm_c = np.stack([a_from_e(E_cf[:, c])[0] for c in range(4)], axis=2)  # (171,36,4)
th_star = np.argmax(E_cf[:, 0, :, :NHALF], axis=-1) * 5    # 最优相位（度）
# 样本外：历史期选角 → 评价期
E_hist = E_fy[:, [HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=1)   # (171,36,72)
th_hist = np.argmax(E_hist[..., :NHALF], axis=-1) * 5
E_eval = E_fy[:, [EVAL_YEARS.index(y) for y in EVAL_YEARS]]                # (171,5,36,72)
ii = np.arange(nF)[:, None]; kk = np.arange(n_tpl)[None, :]
E_sel = E_eval[ii, :, kk, th_hist // 5]                    # 评价期按历史选角取得的 E (171,36,5)
E_mean_eval = E_eval[ii, :, kk, :NHALF].mean(axis=3)       # 同轴序 (171,36,5)
G_so = 100 * (E_sel - E_mean_eval) / np.maximum(E_mean_eval, 1e-9)         # (171,36,5)
pos_frac = (G_so > 0).mean(axis=2)                                         # (171,36)
# 相位漂移：逐年 θ* 的圆周标准差
circ = np.deg2rad(th_y.astype(float) * 5)
Csum = np.abs(np.exp(1j * circ).mean(axis=1))
drift = np.sqrt(-2 * np.log(np.maximum(Csum, 1e-12))) * 57.2958 / np.pi    # 度 (171,36)
# 尾流损失与 AEP
E_nw = HOURS / 1000.0 * np.einsum('u,fyu->fy', P0, p_farm.sum(axis=3))     # 无尾流 kWh/台·年 (171,11)
E_nw_pool = E_nw.mean(axis=1)
wake_loss = 1 - E_cf.mean(axis=-1) / np.maximum(E_nw_pool[:, None, None], 1e-9)  # (171,4,36)
aep_turb = E_cf[ii, 0, kk, th_star // 5] / 1e6                           # MWh/台·年 (171,36)

np.savez_compressed(os.path.join(OUT, 'wp5_cross_farms.npz'),
                    E_fy=E_fy, E_cf=E_cf, A_y=A_y, th_y=th_y,
                    A=A_farm, A_full=A_farm_full, A_c=A_farm_c, th_star=th_star,
                    G_so=G_so, pos_frac=pos_frac, drift=drift,
                    wake_loss=wake_loss, aep_turb=aep_turb, E_nw_pool=E_nw_pool,
                    farm_ids=farm_ids)

# ═══════════════════════════════════════════════════════════════════════
# 4. 格点交叉仿真（1,446 格点，反事实 + 逐年 A/θ*，省全曲线）
# ═══════════════════════════════════════════════════════════════════════
nG = len(grid_lat)
A_grid = np.full((nG, n_tpl), np.nan)
A_grid_c = np.full((nG, n_tpl, 4), np.nan)
th_grid = np.zeros((nG, n_tpl), dtype=np.int16)
A_g_y = np.zeros((nG, 11, n_tpl), dtype=np.float32)
th_g_y = np.zeros((nG, 11, n_tpl), dtype=np.int8)
E_gpool = np.zeros((nG, n_tpl, NWD), dtype=np.float32)
t0 = time.perf_counter()
for i in range(nG):
    if not grid_valid[i]:
        continue
    ti = ti_grid[i]
    pfy = p_grid[i]
    ok = np.isfinite(pfy).all(axis=(1, 2))
    if ok.sum() == 0:
        continue
    FFTp_y = np.fft.fft(np.nan_to_num(pfy), axis=2)
    Ey = e_curve_fft(FFTp_y, ti)
    A_g_y[i], _ = a_from_e(Ey)
    th_g_y[i] = np.argmax(Ey[..., :NHALF], axis=-1)
    ppool = pfy[ok].mean(axis=0)
    Ecf = e_curve_fft(np.fft.fft(counterfactuals(ppool), axis=2), ti)
    E_gpool[i] = Ecf[0]
    A_grid[i], _ = a_from_e(Ecf[0])
    A_grid_c[i] = np.stack([a_from_e(Ecf[c])[0] for c in range(4)], axis=1)
    th_grid[i] = np.argmax(Ecf[0, :, :NHALF], axis=-1) * 5
print(f'格点交叉仿真完成 ({time.perf_counter()-t0:.0f}s)')

np.savez_compressed(os.path.join(OUT, 'wp5_cross_grid.npz'),
                    A=A_grid, A_c=A_grid_c, th=th_grid, A_y=A_g_y, th_y=th_g_y,
                    E_pool=E_gpool, valid=grid_valid, lon=grid_lon, lat=grid_lat)

# ═══════════════════════════════════════════════════════════════════════
# 5. R_g / F_g / D_g（空间分位在 1,446 格点内）
# ═══════════════════════════════════════════════════════════════════════
v = grid_valid & np.isfinite(A_grid).all(axis=1)
A_v = A_grid[v]
Q = np.full((nG, n_tpl), np.nan)
Q[v] = (np.argsort(np.argsort(A_v, axis=0), axis=0) + 1) / v.sum() * 100   # 百分位
R = np.nanmedian(Q, axis=1)
F75 = np.nanmean(Q >= 75, axis=1) * 100
D = np.subtract(*np.nanpercentile(Q, [75, 25], axis=1))
rfd = pd.DataFrame(dict(lon=grid_lon, lat=grid_lat, R=R.round(2), F75=F75.round(2), D=D.round(2)))
rfd.to_csv(os.path.join(OUT, 'wp5_rfd_grid.csv'), index=False, encoding='utf-8-sig')

# 农场在格点分布中的空间分位（走廊成员"保持前四分位"的依据）
Q_farm = np.array([(A_v < A_farm[i]).mean(axis=0) * 100 for i in range(nF)])   # (171,36)

# 稳健地图 vs 统计模型地图的空间秩相关
from scipy.stats import spearmanr
stat = pd.read_csv(os.path.join(OUT, 'task1_corridor_grid.csv'), encoding='utf-8-sig')
r_spear, p_spear = spearmanr(R[v], stat['A_pred_pct'].values[v])
print(f'R 地图 vs 统计模型地图 Spearman: {r_spear:.3f} (p={p_spear:.2e})')

# 不同排布高值前四分位 Jaccard（格点）
topq = Q[v] >= 75
m = topq.shape[1]
jac = np.zeros((m, m))
for a in range(m):
    for b in range(m):
        inter = (topq[:, a] & topq[:, b]).sum()
        union = (topq[:, a] | topq[:, b]).sum()
        jac[a, b] = inter / max(union, 1)
ij = np.triu_indices(m, 1)
jaccard_mean = jac[ij].mean()
print(f'36 模板高值前四分位 Jaccard: 均值 {jaccard_mean:.3f}')

# ═══════════════════════════════════════════════════════════════════════
# 6. 方差分解（ANOVA：气候/形态/间距主效应与交互，η²）
# ═══════════════════════════════════════════════════════════════════════
# 平衡设计：A[站点 n, 形态 4, 间距 3, 重复 3]（36 模板 → 4×3×3 张量）
A_t = A_grid[v]                                   # (nvalid, 36)
grand = A_t.mean()
SST = ((A_t - grand) ** 2).sum()
n = A_t.shape[0]
A_cube = np.zeros((n, 4, 3, 3))
for k in range(36):
    mi = {'rule_grid': 0, 'belt': 1, 'cluster': 2, 'multi_cluster': 3}[MORPH[k]]
    si = {3.0: 0, 5.0: 1, 7.0: 2}[SPAC[k]]
    A_cube[:, mi, si, int(summ['rep'].values[k]) - 1] = A_t[:, k]
site_mean = A_cube.mean(axis=(1, 2, 3))           # (n,) 每站点 36 模板均值
morph_mean = A_cube.mean(axis=(0, 2, 3))          # (4,)
spac_mean = A_cube.mean(axis=(0, 1, 3))           # (3,)
SS_site = 36 * ((site_mean - grand) ** 2).sum()
SS_morph = n * 9 * ((morph_mean - grand) ** 2).sum()
SS_spac = n * 12 * ((spac_mean - grand) ** 2).sum()
ms = A_cube.mean(axis=0).mean(axis=-1)            # 形态×间距 (4,3)
SS_ms = n * 3 * ((ms - morph_mean[:, None] - spac_mean[None, :] + grand) ** 2).sum()
sm = A_cube.mean(axis=2).mean(axis=-1)            # 站点×形态 (n,4)
SS_sm = 9 * ((sm - site_mean[:, None] - morph_mean[None, :] + grand) ** 2).sum()
ssp = A_cube.mean(axis=1).mean(axis=-1)           # 站点×间距 (n,3)
SS_ssp = 12 * ((ssp - site_mean[:, None] - spac_mean[None, :] + grand) ** 2).sum()
SS_res = SST - SS_site - SS_morph - SS_spac - SS_ms - SS_sm - SS_ssp
names = ['气候(场址)', '形态', '间距', '形态×间距', '场址×形态', '场址×间距', '残差']
SSs = [SS_site, SS_morph, SS_spac, SS_ms, SS_sm, SS_ssp, max(SS_res, 0)]
eta2 = [s / SST * 100 for s in SSs]
anova_lines = ['方差分解（A 的 η²，格点 × 36 模板）', '=' * 50]
for nm, s, e2 in zip(names, SSs, eta2):
    anova_lines.append(f'{nm:<12s} SS={s:.4f}  η²={e2:.2f}%')
print('\n'.join(anova_lines))

# C3 负对照验收：A^C3 必须 ≈ 0（≤0.1 pp）
c3_max = np.nanmax(A_grid_c[:, :, 3])
c3_farm_max = np.nanmax(A_farm_c[:, :, 3])
print(f'\nC3 均匀方向负对照: 格点最大 A = {c3_max:.6f} pp | 农场最大 A = {c3_farm_max:.6f} pp '
      f'| 验收线 0.1 pp → {"通过" if c3_max <= 0.1 else "失败"}')

# ═══════════════════════════════════════════════════════════════════════
# 7. 农场级输出表 + 报告
# ═══════════════════════════════════════════════════════════════════════
morph4 = np.array(['rule_grid', 'belt', 'cluster', 'multi_cluster'])
rows = []
for i, fid in enumerate(farm_ids):
    for k in range(36):
        rows.append(dict(farm_id=int(fid), template_id=TIDS[k], morphology=MORPH[k],
                         spacing_D=SPAC[k], rep=int(summ['rep'].values[k]),
                         A_pct=round(float(A_farm[i, k]), 4), A_full_pct=round(float(A_farm_full[i, k]), 4),
                         A_C0=round(float(A_farm_c[i, k, 0]), 4), A_C1=round(float(A_farm_c[i, k, 1]), 4),
                         A_C2=round(float(A_farm_c[i, k, 2]), 4), A_C3=round(float(A_farm_c[i, k, 3]), 4),
                         theta_star_deg=int(th_star[i, k]), pos_frac=round(float(pos_frac[i, k]), 3),
                         drift_deg=round(float(drift[i, k]), 2), wake_loss=round(float(wake_loss[i, 0, k]), 4),
                         aep_MWh_turb=round(float(aep_turb[i, k]), 1)))
fc = pd.DataFrame(rows)
fc.to_csv(os.path.join(OUT, 'wp5_farm_cross.csv'), index=False, encoding='utf-8-sig')

# 已知走廊成员在多少种标准排布下保持前四分位（Q≥75）
CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China_strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
corr_support = {}
for cname, members in CORRIDORS.items():
    mm = [i for i, f in enumerate(farm_ids) if int(f) in members]
    if not mm:
        continue
    sup = (Q_farm[mm] >= 75).mean(axis=0) * 100          # 每模板：走廊成员中 Q≥75 的比例
    corr_support[cname] = (len(mm), float(sup.mean()), float(sup.min()), float(sup.max()))
np.savez_compressed(os.path.join(OUT, 'wp5_farm_q.npz'), Q_farm=Q_farm, farm_ids=farm_ids)

with open(os.path.join(OUT, 'wp5_anova.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(anova_lines) + '\n')
with open(os.path.join(OUT, 'wp5_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP5 交叉仿真报告\n' + '=' * 60 + '\n')
    f.write(f'场址: {nF} 农场 + {int(v.sum())} 有效格点 | 模板: {n_tpl} | 风向: 72×5°\n')
    f.write(f'A 口径: 半圆 36 档（论文口径，θ∈0-175°）；反事实 C0-C3；p_ref 均值风速 {np.sum(P_REF*np.array(z3["ws"])):.2f} m/s\n\n')
    f.write('形态×间距平均 A (C0, 农场, %):\n')
    f.write(fc.groupby(['morphology', 'spacing_D'])['A_pct'].mean().round(3).to_string() + '\n\n')
    f.write('形态×间距平均 A (C0, 格点, %):\n')
    gs = pd.DataFrame(dict(template=[f'{MORPH[k]}_{SPAC[k]:.0f}' for k in range(36)],
                           A_grid_mean=A_v.mean(axis=0).round(3)))
    f.write(gs.to_string(index=False) + '\n\n')
    f.write('\n'.join(anova_lines) + '\n\n')
    f.write(f'C3 负对照: 格点 max A = {c3_max:.6f} pp | 农场 max A = {c3_farm_max:.6f} pp\n')
    f.write(f'R vs 统计模型地图 Spearman: {r_spear:.3f} (p={p_spear:.2e})\n')
    f.write(f'36 模板高值前四分位 Jaccard 均值: {jaccard_mean:.3f}\n\n')
    f.write('已知走廊成员跨模板保持前四分位 (Q≥75) 比例 (%):\n')
    for cname, (nm, supm, supmin, supmax) in corr_support.items():
        f.write(f'  {cname:<14s} n={nm:3d} 均值 {supm:5.1f} | 最差 {supmin:5.1f} | 最好 {supmax:5.1f}\n')
print(f'\n输出完成 | 总耗时 {time.perf_counter()-t0:.0f}s（不含此前模块）')
