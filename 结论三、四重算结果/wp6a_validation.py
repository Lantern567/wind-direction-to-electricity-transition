"""
WP6a 验证（无 FLORIS 部分）：验收红线 + 类型匹配 + 嵌套整走廊留出
==================================================================
对应《结论三与结论四补充计算方案》§4.6 与验收红线。

1. C3 均匀方向负对照验收：格点/农场 max A_C3 ≤ 0.1 pp
2. 5°→1° 收敛（η 傅里叶插值到 1° 的全量敏感性）：|ΔA| 分布 ≤ 0.3 pp 目标
   （p 保持 5° 直方图口径；p 重分箱与 FLORIS 1° 直接验证见 wp6b）
3. 类型匹配：WP5 物理模板 A vs task3 真实排布 A（Spearman 目标 ≥ 0.5）
4. 统计管线（特征 → A）嵌套整走廊留出：
   标签 = 物理模板 A（均值）与真实 A 两个版本对照
   指标：留出走廊成员前四分位捕获率（目标 ≥ 50%）、AUC（目标 ≥ 0.75）、
         预测-实际 Spearman

依赖：wp5_cross_farms.npz / wp5_cross_grid.npz / wp4_wake_lookup.npz
     task3/task3_s1_optimal_orientation.csv（真实 A）
     补算/output/task1_training_data.csv（特征）
"""
import os, io, sys, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')

CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China_strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
FEATURES = ['spacing_D', 'aspect_ratio', 'pc1_share', 'ws_mean', 'ws_std',
            'weibull_A', 'weibull_k', 'frac_below_rated', 'orient_sensitivity',
            'wd_entropy_norm', 'exp_wake_loss', 'WCI', 'log_n']

# ═══════════════════════════════════════════════════════════════════════
# 0. 载入
# ═══════════════════════════════════════════════════════════════════════
z5 = np.load(os.path.join(OUT, 'wp5_cross_farms.npz'))
A_farm = z5['A']                 # (171,36) C0 半圆
A_farm_c = z5['A_c']             # (171,36,4)
farm_ids = z5['farm_ids'].astype(int)
Q_farm = np.load(os.path.join(OUT, 'wp5_farm_q.npz'))['Q_farm']
z5g = np.load(os.path.join(OUT, 'wp5_cross_grid.npz'))
A_grid = z5g['A']; A_grid_c = z5g['A_c']; gv = z5g['valid']
z4 = np.load(os.path.join(OUT, 'wp4_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64); P0 = z4['P0'].astype(np.float64)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_farm = z3['p_fy'].astype(np.float64)
summ = pd.read_csv(os.path.join(OUT, 'wp2_template_summary.csv'), encoding='utf-8-sig')
MORPH = summ['morphology'].values; SPAC = summ['spacing_D'].values
nF, n_tpl = A_farm.shape

rep = {}
# ═══════════════════════════════════════════════════════════════════════
# 1. C3 负对照验收
# ═══════════════════════════════════════════════════════════════════════
c3_farm = np.nanmax(np.abs(A_farm_c[:, :, 3]))
c3_grid = np.nanmax(np.abs(A_grid_c[:, :, 3]))
rep['C3'] = f'C3 均匀方向负对照: 农场 max|A|={c3_farm:.6f} pp, 格点 max|A|={c3_grid:.6f} pp '
rep['C3'] += '→ 通过 (≤0.1 pp)' if max(c3_farm, c3_grid) <= 0.1 else '→ 失败'
print(rep['C3'])

# ═══════════════════════════════════════════════════════════════════════
# 2. 5°→1° 收敛（η 傅里叶插值，全量农场+格点）
# ═══════════════════════════════════════════════════════════════════════
NWD1 = 360
# η → 1° 网格：复 FFT 正/负频率平移补零（采样点精确重建，无 rfft Nyquist 权重问题）
F = np.fft.fft(ETA, axis=3)                        # (36,5,18,72)
Fp = np.zeros((36, 5, 18, NWD1), dtype=complex)
Fp[:, :, :, :36] = F[:, :, :, :36]                 # 正频率 0..35
Fp[:, :, :, NWD1 - 36:] = F[:, :, :, 36:]          # 负频率 −36..−1 → 324..359
ETA1 = np.fft.ifft(Fp, axis=3).real * (NWD1 / 72.0)
assert np.max(np.abs(ETA1[:, :, :, 5 * np.arange(72)] - ETA)) < 1e-9, 'η 1° 插值缩放错误'
ETA1 = np.maximum(ETA1, 0.0)                       # 插值振铃截断
# 方向索引 (θ, d): η360[u, (5d+θ) mod 360]（θ=排布逆时针旋转，与 wp5 新约定一致）
d5 = np.arange(72) * 5
IDX = (d5[None, :] + np.arange(NWD1)[:, None]) % NWD1   # (360,72)

def a1deg(p_arr, ti_arr):
    """p_arr: (N,18,72) 或 (18,72)；ti_arr: 逐场址 TI 索引 (N,)。返回 A_1deg 半圆(0-179)/全圆。"""
    p = p_arr if p_arr.ndim == 3 else p_arr[None]
    N = p.shape[0]
    out_h = np.zeros((N, n_tpl)); out_f = np.zeros((N, n_tpl))
    for k in range(n_tpl):
        for tj in range(5):
            idx = np.where(ti_arr == tj)[0]
            if len(idx) == 0:
                continue
            Hp = ETA1[k, tj][:, IDX] * P0[:, None, None]      # (18,360,72)
            E = np.tensordot(Hp, p[idx], axes=([0, 2], [1, 2]))  # (360, nj)
            Eh = E[:180]; Ef = E
            out_h[idx, k] = 100 * (Eh.max(axis=0) - Eh.mean(axis=0)) / np.maximum(Eh.mean(axis=0), 1e-9)
            out_f[idx, k] = 100 * (Ef.max(axis=0) - Ef.mean(axis=0)) / np.maximum(Ef.mean(axis=0), 1e-9)
    return out_h.squeeze(), out_f.squeeze()

# 农场 TI（与 wp5 相同口径：逐场址区域 TI → 5 档取最近）
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import get_ti_for_farm
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
ti_farm = np.array([int(np.argmin(np.abs(np.array(z4['ti']) - get_ti_for_farm(geo.loc[f, 'cent_lat'], geo.loc[f, 'cent_lon'])))) for f in farm_ids])

A1h_farm, A1f_farm = a1deg(p_farm.mean(axis=1), ti_farm)          # (171,36) 用多年平均 p
dA_h = A1h_farm - A_farm                                         # 半圆 A 变化
dA_f = A1f_farm - z5['A_full']
rep['1deg_farm'] = (f'5°→1°(η 插值, 农场): 半圆 |ΔA| p50={np.percentile(np.abs(dA_h),50):.3f} '
                    f'p95={np.percentile(np.abs(dA_h),95):.3f} max={np.abs(dA_h).max():.3f} pp')
print(rep['1deg_farm'])
# 格点（全量，按 TI 分组批量）
zg = np.load(os.path.join(OUT, 'wp3b_grid_climate.npz'))
p_g = zg['p_fy'].astype(np.float64)
gvalid = zg['valid'] & np.isfinite(p_g).all(axis=(1, 2, 3))
lat_g = zg['lat']; lon_g = zg['lon']
ti_grid = np.array([int(np.argmin(np.abs(np.array(z4['ti']) - get_ti_for_farm(la, lo))))
                    for la, lo in zip(lat_g, lon_g)])
A1h_g = np.full((len(lat_g), n_tpl), np.nan)
A1h_g[gvalid], _ = a1deg(p_g[gvalid].mean(axis=1), ti_grid[gvalid])
dA_g = A1h_g[gvalid] - A_grid[gvalid]
rep['1deg_grid'] = (f'5°→1°(η 插值, 格点): |ΔA| p50={np.nanpercentile(np.abs(dA_g),50):.3f} '
                    f'p95={np.nanpercentile(np.abs(dA_g),95):.3f} max={np.nanmax(np.abs(dA_g)):.3f} pp')
print(rep['1deg_grid'])

# ═══════════════════════════════════════════════════════════════════════
# 3. 类型匹配：物理模板 A vs 真实排布 A（优先 wp7a 自算口径；task3 作敏感性）
# ═══════════════════════════════════════════════════════════════════════
z7_path = os.path.join(OUT, 'wp7a_real_curves.npz')
A_real_src = 'task3(18角度)'
if os.path.exists(z7_path):
    z7 = np.load(z7_path)
    A_real = pd.Series(z7['A'], index=z7['farm_ids'].astype(int))
    A_real_src = 'wp7a 自算(36档半圆, 与模板同口径)'
else:
    t3 = pd.read_csv(os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv'), encoding='utf-8-sig')
    g3 = t3.groupby('farm_id')['expected_AEP_kWh'].agg(['max', 'mean'])
    A_real = 100 * (g3['max'] - g3['mean']) / g3['mean']               # 18 角度口径
common = [i for i, f in enumerate(farm_ids) if f in A_real.index]
A_real_c = A_real.loc[farm_ids[common]].values
A_tpl_c = A_farm[common]
r_per_tpl = np.array([spearmanr(A_tpl_c[:, k], A_real_c)[0] for k in range(n_tpl)])
A_tpl_mean = A_tpl_c.mean(axis=1)
r_mean = spearmanr(A_tpl_mean, A_real_c)[0]
rep['typematch'] = (f'类型匹配 Spearman(模板A, 真实A[{A_real_src}]): 逐模板均值 {r_per_tpl.mean():.3f} '
                    f'[{r_per_tpl.min():.3f}, {r_per_tpl.max():.3f}] | 36 模板平均A vs 真实A: {r_mean:.3f}')
print(rep['typematch'])
# task3 敏感性对照（wp7a 存在时）
rep['typematch_t3'] = ''
if os.path.exists(z7_path):
    try:
        t3 = pd.read_csv(os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv'), encoding='utf-8-sig')
        g3 = t3.groupby('farm_id')['expected_AEP_kWh'].agg(['max', 'mean'])
        A_t3 = 100 * (g3['max'] - g3['mean']) / g3['mean']
        cm = [i for i, f in enumerate(farm_ids) if f in A_t3.index]
        r_t3m = spearmanr(A_farm[cm].mean(axis=1), A_t3.loc[farm_ids[cm]].values)[0]
        rep['typematch_t3'] = f'敏感性: 模板A均值 vs task3 A Spearman = {r_t3m:.3f}'
        print(rep['typematch_t3'])
    except Exception as e:
        rep['typematch_t3'] = f'task3 对照跳过: {e}'
# 按形态×间距单元
r_cell = {}
for k in range(n_tpl):
    r_cell.setdefault((MORPH[k], SPAC[k]), []).append(r_per_tpl[k])
rep['typematch_cell'] = '\n'.join(f'  {m:>12s} {s:.0f}D: Spearman {np.mean(v):.3f}'
                                  for (m, s), v in sorted(r_cell.items()))

# ═══════════════════════════════════════════════════════════════════════
# 4. 统计管线嵌套整走廊留出（标签：物理模板A 均值 / 真实A 两版对照）
# ═══════════════════════════════════════════════════════════════════════
td = pd.read_csv(os.path.join(OUT, 'task1_training_data.csv'), encoding='utf-8-sig')
feats = [c for c in FEATURES if c in td.columns]
X = td[feats].values
ids = td['farm_id'].values.astype(int)
# 标签对齐（167 场 → 171 场集合内）
lab_tpl = np.array([A_tpl_mean[common.index(i)] if i in farm_ids[common] else np.nan for i in ids])
lab_real = np.array([A_real_c[common.index(i)] if i in farm_ids[common] else np.nan for i in ids])
ok = np.isfinite(lab_tpl) & np.isfinite(X).all(axis=1)
X, ids, lab_tpl, lab_real = X[ok], ids[ok], lab_tpl[ok], lab_real[ok]

def nested_loo(y):
    pred = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for ci, (cname, members) in enumerate(CORRIDORS.items()):
        te = np.isin(ids, members)
        fold_id[te] = ci
    for ci in np.unique(fold_id):
        te = fold_id == ci
        if te.sum() == 0:
            continue
        tr = ~te
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    # 前四分位阈值来自训练分布（各折训练集 y 的 Q75 的按成员加权均值近似：直接用全量 y Q75）
    thr = np.quantile(y, 0.75)
    y_hi = (y >= thr).astype(int)
    auc = roc_auc_score(y_hi, pred)
    rho = spearmanr(pred, y)[0]
    cap = (pred >= np.quantile(pred, 0.75)).astype(int) & y_hi
    capture = cap.sum() / max(y_hi.sum(), 1)
    return dict(auc=auc, spearman=rho, capture=capture, pred=pred, y=y, fold_id=fold_id)

res_tpl = nested_loo(lab_tpl)
res_real = nested_loo(lab_real)
np.savez_compressed(os.path.join(OUT, 'wp6a_loo.npz'),
                    ids=ids, lab_tpl=lab_tpl, lab_real=lab_real,
                    pred_tpl=res_tpl['pred'], pred_real=res_real['pred'],
                    fold_id=res_tpl['fold_id'],
                    thr_tpl=np.quantile(lab_tpl, 0.75), thr_real=np.quantile(lab_real, 0.75))
rep['loo_tpl'] = (f'嵌套走廊留出 [标签=物理模板A均值]: AUC={res_tpl["auc"]:.3f} '
                  f'Spearman={res_tpl["spearman"]:.3f} 前四分位捕获={res_tpl["capture"]:.1%}')
rep['loo_real'] = (f'嵌套走廊留出 [标签=真实A]:          AUC={res_real["auc"]:.3f} '
                   f'Spearman={res_real["spearman"]:.3f} 前四分位捕获={res_real["capture"]:.1%}')
print(rep['loo_tpl'])
print(rep['loo_real'])
# 物理 A 与统计管线预测的一致性（用 task1 已有 LOO 预测）
try:
    loo = pd.read_csv(os.path.join(OUT, 'task1_loo_predictions.csv'), encoding='utf-8-sig')
    m2 = loo.merge(pd.DataFrame(dict(farm_id=ids, A_tpl=lab_tpl, A_real=lab_real)), on='farm_id')
    r_tpl_pred = spearmanr(m2['A_tpl'], m2['pred_A'])[0]
    r_real_pred = spearmanr(m2['A_real'], m2['pred_A'])[0]
    rep['stat_vs_phys'] = (f'统计模型 pred_A vs 物理模板A Spearman={r_tpl_pred:.3f} | '
                           f'vs 真实A Spearman={r_real_pred:.3f}')
    print(rep['stat_vs_phys'])
except Exception as e:
    rep['stat_vs_phys'] = f'task1 LOO 预测对比跳过: {e}'

# ═══════════════════════════════════════════════════════════════════════
# 5. 报告
# ═══════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT, 'wp6a_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP6a 验证报告\n' + '=' * 60 + '\n')
    f.write(rep['C3'] + '\n' + rep['1deg_farm'] + '\n' + rep['1deg_grid'] + '\n\n')
    f.write(rep['typematch'] + '\n' + rep['typematch_t3'] + '\n' + rep['typematch_cell'] + '\n\n')
    f.write(rep['loo_tpl'] + '\n' + rep['loo_real'] + '\n' + rep['stat_vs_phys'] + '\n\n')
    f.write('管线设计目标: 类型匹配 Spearman≥0.5 | AUC≥0.75 | 前25%捕获≥50%\n')
    f.write('验收红线: C3 A≤0.1 pp | 5°vs1° |ΔA|≤0.3 pp（p 重分箱与 FLORIS 直算见 wp6b）\n')
print('\n完成 → 补算/output/wp6a_report.txt')
