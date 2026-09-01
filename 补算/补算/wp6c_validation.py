"""
WP6c 验证（建设范式版）：C3 负对照 + 5°→1° 收敛 + 类型匹配 + 嵌套走廊留出
==========================================================================
wp6a 的范式情境替换版（学长批准口径）：
1. C3 均匀方向负对照验收：农场/格点 max A_C3 ≤ 0.1 pp
2. 5°→1° 收敛（η 傅里叶插值到 1° 全量敏感性）：A_built 口径 |ΔA| 分布
3. 类型匹配：6 范式情境均值 A_built vs wp7a 真实 A（Spearman ≥ 0.5 目标）；
   逐范式、按 task1 主范式标签分组对照
4. 统计管线嵌套整走廊留出：标签 = 范式均值 A_built / 真实 A 两版对照
   （指标：留出走廊成员前四分位捕获率 ≥ 50%、AUC ≥ 0.75、预测-实际 Spearman）

依赖：wp4c_wake_lookup.npz / wp5c_cross_farms.npz / wp5c_cross_grid.npz /
      wp5c_farm_q.npz / wp7a_real_curves.npz / task1_training_data.csv /
      task1_paradigm_classification.csv
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

PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
n_para = len(PIDS)
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
z5 = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'))
A_built = z5['A']                    # (171,6)
A_c = z5['A_c']                      # (171,6,4)
farm_ids = z5['farm_ids'].astype(int)
TB = z5['TB']                        # (171,6)
Q_farm = np.load(os.path.join(OUT, 'wp5c_farm_q.npz'))['Q_farm']
z5g = np.load(os.path.join(OUT, 'wp5c_cross_grid.npz'))
A_grid = z5g['A']; A_grid_c = z5g['A_c']; gv = z5g['valid']; TB_grid = z5g['TB']
z4 = np.load(os.path.join(OUT, 'wp4c_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64); P0 = z4['P0'].astype(np.float64)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_farm = z3['p_fy'].astype(np.float64)
nF = A_built.shape[0]

rep = {}

# ═══════════════════════════════════════════════════════════════════════
# 1. C3 负对照验收
# ═══════════════════════════════════════════════════════════════════════
c3_farm = np.nanmax(np.abs(A_c[:, :, 3]))
c3_grid = np.nanmax(np.abs(A_grid_c[:, :, 3]))
rep['C3'] = f'C3 均匀方向负对照: 农场 max|A|={c3_farm:.6f} pp, 格点 max|A|={c3_grid:.6f} pp '
rep['C3'] += '→ 通过 (≤0.1 pp)' if max(c3_farm, c3_grid) <= 0.1 else '→ 失败'
print(rep['C3'])

# ═══════════════════════════════════════════════════════════════════════
# 2. 5°→1° 收敛（η 傅里叶插值；A_built 半圆窗口在 1° 网格上重算）
# ═══════════════════════════════════════════════════════════════════════
NWD1 = 360
F = np.fft.fft(ETA, axis=3)                        # (6,5,18,72)
Fp = np.zeros((6, 5, 18, NWD1), dtype=complex)
Fp[:, :, :, :36] = F[:, :, :, :36]
Fp[:, :, :, NWD1 - 36:] = F[:, :, :, 36:]
ETA1 = np.fft.ifft(Fp, axis=3).real * (NWD1 / 72.0)
assert np.max(np.abs(ETA1[:, :, :, 5 * np.arange(72)] - ETA)) < 1e-9, 'η 1° 插值缩放错误'
ETA1 = np.maximum(ETA1, 0.0)
d5 = np.arange(72) * 5
IDX = (d5[None, :] + np.arange(NWD1)[:, None]) % NWD1   # (360,72) θ=排布逆时针

sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import get_ti_for_farm
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
ti_farm = np.array([int(np.argmin(np.abs(np.array(z4['ti']) -
                    get_ti_for_farm(geo.loc[f, 'cent_lat'], geo.loc[f, 'cent_lon'])))) for f in farm_ids])

def a1deg(p_arr, ti_arr, TB_arr):
    """1° 网格上 A_built。p_arr:(N,18,72) ti_arr:(N,) TB_arr:(N,6)。返回 (N,6)。"""
    N = p_arr.shape[0]
    out = np.zeros((N, n_para))
    for k in range(n_para):
        for tj in range(5):
            idx = np.where(ti_arr == tj)[0]
            if len(idx) == 0:
                continue
            Hp = ETA1[k, tj][:, IDX] * P0[:, None, None]      # (18,360,72)
            E = np.tensordot(Hp, p_arr[idx], axes=([0, 2], [1, 2]))  # (360, nj)
            tb = TB_arr[idx, k] * 5                          # 建成基线角（度）
            w = np.stack([E[(t + np.arange(180)) % NWD1, j] for j, t in enumerate(tb)], axis=1)  # (180,nj)
            out[idx, k] = 100 * (w.max(axis=0) - w.mean(axis=0)) / np.maximum(w.mean(axis=0), 1e-9)
    return out

A1_farm = a1deg(p_farm.mean(axis=1), ti_farm, TB)
dA_f = A1_farm - A_built
rep['1deg_farm'] = (f'5°→1°(η 插值, 农场, A_built): |ΔA| p50={np.percentile(np.abs(dA_f),50):.3f} '
                    f'p95={np.percentile(np.abs(dA_f),95):.3f} max={np.abs(dA_f).max():.3f} pp')
print(rep['1deg_farm'])
zg = np.load(os.path.join(OUT, 'wp3b_grid_climate.npz'))
p_g = zg['p_fy'].astype(np.float64)
gvalid = zg['valid'] & np.isfinite(p_g).all(axis=(1, 2, 3))
lat_g = zg['lat']; lon_g = zg['lon']
ti_grid = np.array([int(np.argmin(np.abs(np.array(z4['ti']) - get_ti_for_farm(la, lo))))
                    for la, lo in zip(lat_g, lon_g)])
A1_g = np.full((len(lat_g), n_para), np.nan)
A1_g[gvalid] = a1deg(p_g[gvalid].mean(axis=1), ti_grid[gvalid], TB_grid[gvalid])
dA_g = A1_g[gvalid] - A_grid[gvalid]
rep['1deg_grid'] = (f'5°→1°(η 插值, 格点, A_built): |ΔA| p50={np.nanpercentile(np.abs(dA_g),50):.3f} '
                    f'p95={np.nanpercentile(np.abs(dA_g),95):.3f} max={np.nanmax(np.abs(dA_g)):.3f} pp')
print(rep['1deg_grid'])

# ═══════════════════════════════════════════════════════════════════════
# 3. 类型匹配：范式情境 A_built vs wp7a 真实 A
# ═══════════════════════════════════════════════════════════════════════
z7 = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
A_real = pd.Series(z7['A'], index=z7['farm_ids'].astype(int))
common = [i for i, f in enumerate(farm_ids) if f in A_real.index]
A_real_c = A_real.loc[farm_ids[common]].values
A_para_c = A_built[common]
r_per_para = np.array([spearmanr(A_para_c[:, k], A_real_c)[0] for k in range(n_para)])
A_para_mean = A_para_c.mean(axis=1)
r_mean = spearmanr(A_para_mean, A_real_c)[0]
rep['typematch'] = (f'类型匹配 Spearman(范式A, 真实A[wp7a 自算]): 逐范式均值 {r_per_para.mean():.3f} '
                    f'[{r_per_para.min():.3f}, {r_per_para.max():.3f}] | 6 范式平均A vs 真实A: {r_mean:.3f}')
print(rep['typematch'])
rep['typematch_para'] = '\n'.join(f'  {p:>6s}: Spearman {r:.3f}'
                                  for p, r in zip(PIDS, r_per_para))
# 按 task1 主范式标签分组
cls = pd.read_csv(os.path.join(BUSH, 'input_task1', 'task1_paradigm_classification.csv'),
                  encoding='utf-8-sig').set_index('farm_id')
main_label = cls['paradigm_labels'].fillna('').apply(lambda s: str(s).split('+')[0])
g_lines = []
for lab in ['A', 'B', 'C', 'D']:
    gi = [i for i, f in enumerate(farm_ids[common]) if main_label.get(f, '') == lab]
    if len(gi) < 5:
        continue
    r_g = spearmanr(A_para_mean[gi], A_real_c[gi])[0]
    g_lines.append(f'  主标签 {lab} (n={len(gi):3d}): 范式均值A vs 真实A Spearman {r_g:.3f}')
rep['typematch_group'] = '\n'.join(g_lines)
print(rep['typematch_group'])
# task3 敏感性对照
try:
    t3 = pd.read_csv(os.path.join(REPO, 'task3', 'task3_s1_optimal_orientation.csv'), encoding='utf-8-sig')
    g3 = t3.groupby('farm_id')['expected_AEP_kWh'].agg(['max', 'mean'])
    A_t3 = 100 * (g3['max'] - g3['mean']) / g3['mean']
    cm = [i for i, f in enumerate(farm_ids) if f in A_t3.index]
    r_t3m = spearmanr(A_built[cm].mean(axis=1), A_t3.loc[farm_ids[cm]].values)[0]
    rep['typematch_t3'] = f'敏感性: 范式A均值 vs task3 A Spearman = {r_t3m:.3f}'
    print(rep['typematch_t3'])
except Exception as e:
    rep['typematch_t3'] = f'task3 对照跳过: {e}'

# ═══════════════════════════════════════════════════════════════════════
# 4. 统计管线嵌套整走廊留出（标签：范式均值 A_built / 真实 A）
# ═══════════════════════════════════════════════════════════════════════
td = pd.read_csv(os.path.join(OUT, 'task1_training_data.csv'), encoding='utf-8-sig')
feats = [c for c in FEATURES if c in td.columns]
X = td[feats].values
ids = td['farm_id'].values.astype(int)
lab_para = np.array([A_para_mean[common.index(i)] if i in farm_ids[common] else np.nan for i in ids])
lab_real = np.array([A_real_c[common.index(i)] if i in farm_ids[common] else np.nan for i in ids])
ok = np.isfinite(lab_para) & np.isfinite(X).all(axis=1)
X, ids, lab_para, lab_real = X[ok], ids[ok], lab_para[ok], lab_real[ok]

def nested_loo(y):
    pred = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for ci, (cname, members) in enumerate(CORRIDORS.items()):
        fold_id[np.isin(ids, members)] = ci
    for ci in np.unique(fold_id):
        te = fold_id == ci
        if te.sum() == 0:
            continue
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=0)
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    thr = np.quantile(y, 0.75)
    y_hi = (y >= thr).astype(int)
    auc = roc_auc_score(y_hi, pred)
    rho = spearmanr(pred, y)[0]
    cap = (pred >= np.quantile(pred, 0.75)).astype(int) & y_hi
    capture = cap.sum() / max(y_hi.sum(), 1)
    return dict(auc=auc, spearman=rho, capture=capture, pred=pred)

res_para = nested_loo(lab_para)
res_real = nested_loo(lab_real)
np.savez_compressed(os.path.join(OUT, 'wp6c_loo.npz'),
                    ids=ids, lab_para=lab_para, lab_real=lab_real,
                    pred_para=res_para['pred'], pred_real=res_real['pred'],
                    thr_para=np.quantile(lab_para, 0.75), thr_real=np.quantile(lab_real, 0.75))
rep['loo_para'] = (f'嵌套走廊留出 [标签=范式均值A]: AUC={res_para["auc"]:.3f} '
                   f'Spearman={res_para["spearman"]:.3f} 前四分位捕获={res_para["capture"]:.1%}')
rep['loo_real'] = (f'嵌套走廊留出 [标签=真实A]:       AUC={res_real["auc"]:.3f} '
                   f'Spearman={res_real["spearman"]:.3f} 前四分位捕获={res_real["capture"]:.1%}')
print(rep['loo_para'])
print(rep['loo_real'])
try:
    loo = pd.read_csv(os.path.join(OUT, 'task1_loo_predictions.csv'), encoding='utf-8-sig')
    m2 = loo.merge(pd.DataFrame(dict(farm_id=ids, A_para=lab_para, A_real=lab_real)), on='farm_id')
    r_para_pred = spearmanr(m2['A_para'], m2['pred_A'])[0]
    r_real_pred = spearmanr(m2['A_real'], m2['pred_A'])[0]
    rep['stat_vs_phys'] = (f'统计模型 pred_A vs 范式均值A Spearman={r_para_pred:.3f} | '
                           f'vs 真实A Spearman={r_real_pred:.3f}')
    print(rep['stat_vs_phys'])
except Exception as e:
    rep['stat_vs_phys'] = f'task1 LOO 预测对比跳过: {e}'

# ═══════════════════════════════════════════════════════════════════════
# 5. 报告
# ═══════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT, 'wp6c_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP6c 验证报告（建设范式情境版）\n' + '=' * 60 + '\n')
    f.write(rep['C3'] + '\n' + rep['1deg_farm'] + '\n' + rep['1deg_grid'] + '\n\n')
    f.write(rep['typematch'] + '\n' + rep['typematch_para'] + '\n' +
            rep['typematch_group'] + '\n' + rep['typematch_t3'] + '\n\n')
    f.write(rep['loo_para'] + '\n' + rep['loo_real'] + '\n' + rep['stat_vs_phys'] + '\n\n')
    f.write('管线设计目标: 类型匹配 Spearman≥0.5 | AUC≥0.75 | 前25%捕获≥50%\n')
    f.write('验收红线: C3 A≤0.1 pp | 5°vs1° |ΔA|≤0.3 pp（FLORIS 直算抽查见 wp5c_spotcheck）\n')
print('\n完成 → 补算/output/wp6c_report.txt')
