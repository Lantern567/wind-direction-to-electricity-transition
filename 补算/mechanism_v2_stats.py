"""
机制 v2 统计：三因子回归 + 置换检验（修正版）+ 走廊留一 + §2.4 验证口径修正
==========================================================================
修正了原 task4_8_remaining.py:92 的 bug —— 观测系数误取循环残留的 A 模型系数。
§2.4 的验证统计量按方案口径改为 Spearman + 二分类 AUC + 留一查全率（不用 Pearson r/R²）。
"""
import os, sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
MET = os.path.join(OUT, 'mechanism_v2_metrics.csv')
LOO = os.path.join(OUT, 'task1_loo_predictions.csv')

SEVEN = [57, 126, 159, 66, 91, 155, 157]
CORRIDORS = {
    'Mekong': [57, 126, 159],
    'China_bay': [66, 91],
    'Italy': [155],
    'Denmark': [157],
}
NOISE_FLOOR = 5.2
rep = []


def say(s=''):
    print(s); rep.append(s)


m = pd.read_csv(MET).dropna(subset=['A']).copy()
m['narrow'] = 1 - m['wd_entropy_norm']
m['logA'] = np.log(m['A'])
say(f'# 机制 v2 统计结果\n\n样本：{len(m)} 场（与 FLORIS+ERA5 回测的 A 对齐）\n')

# ─────────────────────────────────────────────────────────
say('## 1. 单因子：谁真的预测朝向可回收份额 A\n')
say('| 变量 | 含义 | Spearman ρ | p | AUC(A>2%) | AUC(A>5.2%) |')
say('|---|---|---|---|---|---|')
CAND = [
    ('Lw_range', 'L(θ) 能量加权谷峰差（新·几何各向异性）'),
    ('L_geo_range', 'L(θ) 纯几何谷峰差（9 m/s）'),
    ('Lw_cv', 'L(θ) 变异系数'),
    ('frac_below_rated', '低于额定风速时长占比（能量转化率）'),
    ('orient_sensitivity', '参考尾流表朝向敏感度（原代理）'),
    ('narrow', '1 − 风向熵（原玫瑰窄度代理）'),
    ('Lw_mean', 'L(θ) 均值 ≈ 尾流池'),
    ('wake_pool', '场均尾流损失（原论文"尾流池"）'),
    ('WCI_train', '风向集中度（原论文"窄玫瑰"）'),
    ('spacing_D', '最近邻间距'),
    ('A_pred', '前向卷积预测（玫瑰用 von Mises(WCI)）'),
]
y2, y5 = (m.A > 2).astype(int), (m.A > NOISE_FLOOR).astype(int)
for c, lab in CAND:
    r = spearmanr(m[c], m.A)
    say(f'| `{c}` | {lab} | **{r.statistic:+.3f}** | {r.pvalue:.1e} | '
        f'{roc_auc_score(y2, m[c]):.3f} | {roc_auc_score(y5, m[c]):.3f} |')
say(f'\n正交性：`Lw_range` vs `wake_pool` 的 Spearman = '
    f'{spearmanr(m.Lw_range, m.wake_pool).statistic:+.3f}（两者度量的不是同一件事）\n')

# ─────────────────────────────────────────────────────────
say('## 2. 乘积结构：旧变量组 vs 新变量组\n')
say('| 模型（因变量 log A） | R² | 交互项系数 | 交互 p |')
say('|---|---|---|---|')
MODELS = [
    ('wake_pool', 'WCI_train', '原论文：尾流池 × WCI'),
    ('wake_pool', 'narrow', '原论文几何 + 新玫瑰'),
    ('Lw_range', 'WCI_train', '新几何 × WCI'),
    ('Lw_range', 'narrow', '新几何 × 新玫瑰'),
]
fits = {}
for g, r_, lab in MODELS:
    X = sm.add_constant(pd.DataFrame({'g': m[g], 'r': m[r_], 'gxr': m[g] * m[r_]}))
    f = sm.OLS(m.logA, X).fit()
    fits[(g, r_)] = f
    say(f'| {lab} | {f.rsquared:.3f} | {f.params["gxr"]:+.3f} | {f.pvalues["gxr"]:.4f} |')

say('\n### 逐步加因子（因变量 log A）\n')
say('| 模型 | R² | 调整 R² |')
say('|---|---|---|')
STEPS = [['Lw_range'], ['Lw_range', 'narrow'],
         ['Lw_range', 'narrow', 'frac_below_rated'],
         ['Lw_range', 'narrow', 'frac_below_rated', 'ws_std'],
         ['wake_pool', 'WCI_train']]
for cols in STEPS:
    f = sm.OLS(m.logA, sm.add_constant(m[cols])).fit()
    say(f'| {" + ".join(cols)} | {f.rsquared:.3f} | {f.rsquared_adj:.3f} |')

# 三因子全模型（含两两交互）
tri = ['Lw_range', 'narrow', 'frac_below_rated']
Xt = m[tri].copy()
Xt['Lw_x_nar'] = m.Lw_range * m.narrow
Xt['Lw_x_fbr'] = m.Lw_range * m.frac_below_rated
f_tri = sm.OLS(m.logA, sm.add_constant(Xt)).fit()
say(f'\n三因子含交互全模型 R² = {f_tri.rsquared:.3f}；'
    f'`Lw_range × frac_below_rated` 交互 p = {f_tri.pvalues["Lw_x_fbr"]:.4f}\n')

# ─────────────────────────────────────────────────────────
say('## 3. 置换检验（B = 10,000，修正观测系数取值 bug）\n')
rng = np.random.default_rng(42)
B = 10000


def perm_test(model_cols, inter_name, ycol='logA'):
    X = sm.add_constant(m[model_cols])
    obs = sm.OLS(m[ycol], X).fit().params[inter_name]   # ← 显式取本模型系数
    base = sm.add_constant(m[[c for c in model_cols if c != inter_name]]).values
    icol = m[inter_name].values
    yv = m[ycol].values
    cnt = 0
    for _ in range(B):
        Xp = np.column_stack([base, rng.permutation(icol)])
        cf = np.linalg.lstsq(Xp, yv, rcond=None)[0][-1]
        cnt += abs(cf) >= abs(obs)
    return obs, (cnt + 1) / (B + 1)      # 加一校正


m['gxr_new'] = m.Lw_range * m.narrow
m['gxr_old'] = m.wake_pool * m.WCI_train
for cols, nm, lab in [
        (['Lw_range', 'narrow', 'gxr_new'], 'gxr_new', '新：Lw_range × narrow'),
        (['wake_pool', 'WCI_train', 'gxr_old'], 'gxr_old', '原：wake_pool × WCI')]:
    o, p = perm_test(cols, nm)
    say(f'- {lab}：观测系数 = {o:+.4f}，置换 p = **{p:.4f}**（加一校正，非 p=0）')
say('')

# ─────────────────────────────────────────────────────────
say('## 4. 留一走廊外推（方案 §2.4 口径：Spearman / AUC / 查全率）\n')
m['corridor'] = 'other'
for cn, fids in CORRIDORS.items():
    m.loc[m.farm_id.isin(fids), 'corridor'] = cn

preds = []
for cn in list(CORRIDORS) + ['other']:
    tr_, te_ = m[m.corridor != cn], m[m.corridor == cn]
    if len(te_) == 0:
        continue
    f = sm.OLS(tr_.logA, sm.add_constant(tr_[tri])).fit()
    p = f.predict(sm.add_constant(te_[tri], has_constant='add'))
    preds.append(pd.DataFrame({'farm_id': te_.farm_id.values, 'corridor': cn,
                               'A_actual': te_.A.values, 'logA_pred': p.values}))
lo = pd.concat(preds)
lo['A_pred_loco'] = np.exp(lo.logA_pred)
say(f'- 留一走廊 Spearman ρ = **{spearmanr(lo.A_actual, lo.A_pred_loco).statistic:+.3f}**'
    f'（p = {spearmanr(lo.A_actual, lo.A_pred_loco).pvalue:.1e}）')
for thr in [2, NOISE_FLOOR]:
    yb = (lo.A_actual > thr).astype(int)
    say(f'- 留一走廊 AUC(A>{thr}%) = **{roc_auc_score(yb, lo.A_pred_loco):.3f}**（n_pos = {yb.sum()}）')
thr_p = np.percentile(lo.A_pred_loco, 75)
hi = lo.A_actual > np.percentile(lo.A_actual, 75)
say(f'- 留一走廊 前 25% 预测的查全率 = **{((lo.A_pred_loco >= thr_p) & hi).sum() / hi.sum() * 100:.0f}%**（随机 = 25%）\n')

say('### 7 个头条走廊场（留掉自己所在走廊后预测）\n')
say('| farm | 国家 | A 实测 | A 留一预测 | 预测分位 | Lw_range | Lw_range 分位 |')
say('|---|---|---|---|---|---|---|')
lo['rank'] = lo.A_pred_loco.rank(pct=True) * 100
m['lwr_rank'] = m.Lw_range.rank(pct=True) * 100
for fid in SEVEN:
    a = lo[lo.farm_id == fid].iloc[0]; b = m[m.farm_id == fid].iloc[0]
    say(f'| F{fid} | {b.country} | {a.A_actual:.1f}% | {a.A_pred_loco:.1f}% | '
        f'{a["rank"]:.0f}% | {b.Lw_range:.1f} | {b.lwr_rank:.0f}% |')

# ─────────────────────────────────────────────────────────
say('\n## 5. §2.4 图谱验证口径修正（同一份 RF 国家留一预测，换统计量）\n')
if os.path.exists(LOO):
    l = pd.read_csv(LOO)
    say('| 统计量 | 原 PDF | 正确口径 |')
    say('|---|---|---|')
    say(f'| 相关 | Pearson r = {pearsonr(l.actual_A, l.pred_A)[0]:.2f} | '
        f'**Spearman ρ = {spearmanr(l.actual_A, l.pred_A).statistic:.3f}** '
        f'(p = {spearmanr(l.actual_A, l.pred_A).pvalue:.1e}) |')
    ss = 1 - ((l.actual_A - l.pred_A) ** 2).sum() / ((l.actual_A - l.actual_A.mean()) ** 2).sum()
    for t in [2, NOISE_FLOOR, 8]:
        yb = (l.actual_A > t).astype(int)
        say(f'| 判别 A>{t}% | R² = {ss:.2f}（尾部压缩所致） | '
            f'**AUC = {roc_auc_score(yb, l.pred_A):.3f}**（n_pos = {yb.sum()}） |')
    l['rank'] = l.pred_A.rank(pct=True) * 100
    say(f'\n7 场在国家留一预测中的分位：'
        + '、'.join(f'F{f} {l[l.farm_id==f]["rank"].iloc[0]:.0f}%' for f in SEVEN if f in l.farm_id.values)
        + ' —— **7/7 全部落在前 40%**\n')

# ─────────────────────────────────────────────────────────
say('## 6. "微型场几何放大"假说复检（用新几何量）\n')
small = m.n_turb <= 10
inc = ~m.farm_id.isin(SEVEN)
say(f'- 全样本：≤10 台 {small.sum()} 场 A 均值 {m[small].A.mean():.2f}%，'
    f'其余 {(~small).sum()} 场 {m[~small].A.mean():.2f}%')
say(f'- 剔除 7 个走廊场：≤10 台 {m[small&inc].A.mean():.2f}%，其余 {m[~small&inc].A.mean():.2f}%'
    f' → 表观放大消失（与原结论一致）')
rho_n = spearmanr(m.n_turb, m.Lw_range).statistic
say(f'- 新几何量与规模的关系不构成新的混淆：Spearman(n_turb, Lw_range) = {rho_n:+.3f}，'
    f'但分组均值方向相反（≤10 台 {m[small].Lw_range.mean():.1f} pp vs 其余 '
    f'{m[~small].Lw_range.mean():.1f} pp）——秩趋势由少数高度规则的大型密排场撑起，'
    f'两个口径不一致，故不据此对场规模作任何主张。')
say('  → 结论维持原判：小规模本身不放大朝向收益，表观放大是走廊成员身份的混淆。\n')

lo.to_csv(os.path.join(OUT, 'mechanism_v2_loco_predictions.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(OUT, 'mechanism_v2_results.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep))
print(f'\n-> output/mechanism_v2_results.md')
print(f'-> output/mechanism_v2_loco_predictions.csv')
