"""
结论二复算（2026-08-18，学长反馈"窄风玫瑰 vs 小间距到底哪个是决定因素，重新 check"）
====================================================================================
用 wp3 全样本扇区级风玫瑰（wp3_climate_joint.npz，171 场 × 11 年 × 18 风速档 × 72 风向扇区）
把一阶集中度 R1（"窄风玫瑰"）与二阶轴向集中度 R2（"沿固定轴双向往返"）扩到全 171 场，
与有效最近邻间距（wp1 局部投影重算）、平均风速、响应幅度 A（wp7a 真实排布 / wp5c 范式均值）
做联合归因：
  ① 复现 n=7 走廊场的 R1/R2 符号（R1 −0.536 / R2 +0.857 是否在全样本成立）
  ② 全样本 Spearman + 偏相关：间距 vs 轴向风，谁决定 A
  ③ 分层表：小间距是筛子（必要条件），轴向风是幅度调制（充分性）
  ④ 走廊 vs 非走廊的间距/R2 对比 + 区域（东亚/欧洲）对比 → 地理汇聚解释
  ⑤ 机型口径已在 wp5d 完成（5.1–7.4×, p<2e-10），不重复
输出：补算/output/wp9c_conclusion2_recheck.txt + wp9c_farm_metrics.csv
"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
corr_of = {}
for cn, mem in CORRIDORS.items():
    for f in mem:
        corr_of[f] = cn
SEVEN = [57, 126, 159, 155, 66, 91, 157]   # 正文 2.2 已用的 7 个扇区级风玫瑰走廊场

# ─────────────── 1. 全样本 R1/R2（频率与 V³ 能量两种权重） ───────────────
z = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_fy = z['p_fy'].astype(float)          # (171, 11, 18, 72)
hours = z['hours']; ws = z['ws']; wd = z['wd']; farm_ids = z['farm_ids'].astype(int)
p_fs = p_fy.sum(axis=1)                 # (171, 18, 72) 场级小时数
th = np.deg2rad(wd)                     # 72 扇区中心角
e1 = np.exp(1j * th); e2 = np.exp(2j * th)
rows = []
for i, fid in enumerate(farm_ids):
    pf = p_fs[i]
    wf = pf.sum(axis=0)                      # 频率权重方向分布
    we = (pf * ws[:, None] ** 3).sum(axis=0) # V³ 能量权重
    wf = wf / wf.sum(); we = we / we.sum()
    r1f = abs((wf * e1).sum()); r2f = abs((wf * e2).sum())
    r1e = abs((we * e1).sum()); r2e = abs((we * e2).sum())
    ws_mean = (pf * ws[:, None]).sum() / pf.sum()
    rows.append((fid, r1f, r2f, r1e, r2e, ws_mean))
clim = pd.DataFrame(rows, columns=['farm_id', 'R1_f', 'R2_f', 'R1_e', 'R2_e', 'ws_mean'])
clim['disp_f'] = 1 - clim['R1_f']            # 风向分散度（一阶）

# ─────────────── 2. 几何（wp1 局部投影重算） ───────────────
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'),
                  encoding='utf-8-sig').set_index('farm_id')
geo['region'] = np.where(geo['country'].isin(['China', 'Vietnam', 'Taiwan', 'Japan', 'South Korea']),
                         'east_asia', 'europe')

# ─────────────── 3. 响应幅度 A（两口径） ───────────────
z5 = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'))
A_para = pd.Series(z5['A'].mean(axis=1), index=z5['farm_ids'].astype(int))  # 六范式均值
z7 = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
A_real = pd.Series(z7['A'], index=z7['farm_ids'].astype(int))               # 真实排布

df = clim.merge(geo.reset_index(), on='farm_id')
df['A_para'] = df['farm_id'].map(A_para)
df['A_real'] = df['farm_id'].map(A_real)
df['corr'] = df['farm_id'].map(lambda f: corr_of.get(f, 'other'))
df['is_corr'] = df['corr'] != 'other'

S = df['spacing_D_med']
def rho(a, b):
    m = a.notna() & b.notna()
    return spearmanr(a[m], b[m])

def line(k, v):
    out.append(f'{k}: {v}')

out = []
out.append('═' * 78)
out.append('结论二复算报告（2026-08-18）')
out.append('口径：R1/R2 由 wp3 ERA5 全样本 171 场 × 11 年 × 72 扇区直算；')
out.append('      间距 = wp1 局部投影有效最近邻间距中位（D）；A = 响应幅度')
out.append('═' * 78)

# ① 复现 n=7
out.append('\n① n=7 走廊场复现（对照正文 2.2：R1 与 A −0.536，R2 与 A +0.857）')
s7 = df[df.farm_id.isin(SEVEN)].sort_values('A_real')
out.append(s7[['farm_id', 'R1_f', 'R2_f', 'R1_e', 'R2_e', 'A_real']].round(3).to_string(index=False))
r1v = spearmanr(s7['R1_f'], s7['A_real'])[0]
r2v = spearmanr(s7['R2_f'], s7['A_real'])[0]
r2e = spearmanr(s7['R2_e'], s7['A_real'])[0]
out.append(f'  n=7: corr(A_real, R1_f) = {r1v:.3f} | corr(A_real, R2_f) = {r2v:.3f} | corr(A_real, R2_e) = {r2e:.3f}')

# ② 全样本相关
out.append('\n② 全样本 Spearman（171 场，A 两口径）')
out.append('  变量            A_real          A_para(6范式均值)')
for name, v in [('spacing_D_med (S)', S), ('R1_f (一阶集中)', df.R1_f), ('R2_f (二阶轴向)', df.R2_f),
                ('R1_e (能量一阶)', df.R1_e), ('R2_e (能量二阶)', df.R2_e),
                ('ws_mean (平均风速)', df.ws_mean), ('disp_f (分散度)', df.disp_f)]:
    r1_, p1_ = rho(df.A_real, v); r2_, p2_ = rho(df.A_para, v)
    out.append(f'  {name:18s} {r1_:+.3f} (p={p1_:.1e})   {r2_:+.3f} (p={p2_:.1e})')

# ③ 偏相关（秩回归残差）
def partial(a, b, c):
    m = a.notna() & b.notna() & c.notna()
    a_, b_, c_ = a[m].rank(), b[m].rank(), c[m].rank()
    rb = b_ - np.polyval(np.polyfit(c_, b_, 1), c_)
    ra = a_ - np.polyval(np.polyfit(c_, a_, 1), c_)
    return spearmanr(ra, rb)
out.append('\n③ 秩偏相关（控制第三方）')
r1_, p1_ = partial(df.A_real, df.R2_e, S); out.append(f'  corr(A, R2_e | S)   = {r1_:+.3f} (p={p1_:.1e})   ← 控制间距后轴向风仍在')
r2_, p2_ = partial(df.A_real, S, df.R2_e); out.append(f'  corr(A, S | R2_e)   = {r2_:+.3f} (p={p2_:.1e})   ← 控制轴向风后间距仍在')
r3_, p3_ = partial(df.A_real, df.R1_e, S); out.append(f'  corr(A, R1_e | S)   = {r3_:+.3f} (p={p3_:.1e})   ← 一阶集中度控制间距后')
r4_, p4_ = partial(df.A_real, S, df.ws_mean); out.append(f'  corr(A, S | ws)     = {r4_:+.3f} (p={p4_:.1e})   ← 控制风速后间距仍在')

# ④ 分层表
out.append('\n④ 分层表（A_real > 5.2% 为高响应；密排 = S < 4D；高轴向 = R2_e > 中位）')
hi = df.A_real > 5.2
dense = S < 4
axhi = df.R2_e > df.R2_e.median()
for nm, m in [('全部', np.ones(len(df), bool)), ('密排(S<4D)', dense),
              ('密排×高轴向', dense & axhi), ('密排×低轴向', dense & ~axhi),
              ('疏排(S≥4D)', ~dense), ('疏排×高轴向', ~dense & axhi), ('疏排×低轴向', ~dense & ~axhi)]:
    n = int(m.sum()); nh = int((m & hi).sum())
    out.append(f'  {nm:14s} n={n:3d}  高响应 {nh:2d} 场  占比 {nh/n*100:4.1f}%')

# ⑤ 走廊 vs 非走廊 + 区域
out.append('\n⑤ 走廊 vs 非走廊（间距与轴向集中度，地理汇聚证据）')
for nm, m in [('走廊成员(23)', df.is_corr), ('非走廊(148)', ~df.is_corr)]:
    s_med = df.loc[m, 'spacing_D_med'].median()
    r_med = df.loc[m, 'R2_e'].median()
    a_med = df.loc[m, 'A_real'].median()
    out.append(f'  {nm:12s} S 中位 {s_med:.2f} D | R2_e 中位 {r_med:.3f} | A_real 中位 {a_med:.2f} pp')
u1 = mannwhitneyu(df.loc[df.is_corr, 'spacing_D_med'], df.loc[~df.is_corr, 'spacing_D_med'], alternative='less')
u2 = mannwhitneyu(df.loc[df.is_corr, 'R2_e'], df.loc[~df.is_corr, 'R2_e'], alternative='greater')
out.append(f'  间距：走廊显著更密 p={u1.pvalue:.1e} | R2_e：走廊显著更高 p={u2.pvalue:.1e}')
out.append('  区域对比：')
for reg, sub in df.groupby('region'):
    out.append(f'    {reg}: S 中位 {sub.spacing_D_med.median():.2f} D | R2_e 中位 {sub.R2_e.median():.3f} | n={len(sub)}')

# ⑥ 联合模型 R²
out.append('\n⑥ 对数 A 联合回归（OLS R²，仅作叙事性比较）')
X = df[['R2_e', 'R1_e', 'ws_mean']].join(df['spacing_D_med'].rename('S')).apply(lambda c: c.rank())
y = np.log(df.A_real).rank()
import numpy.linalg as la
def r2_ols(cols):
    m = y.notna() & X[cols].notna().all(axis=1)
    Xm = np.column_stack([np.ones(m.sum())] + [X.loc[m, c] for c in cols])
    b = la.lstsq(Xm, y[m], rcond=None)[0]
    r = y[m] - Xm @ b
    return 1 - (r @ r) / ((y[m] - y[m].mean()) @ (y[m] - y[m].mean()))
out.append(f'  仅 S          R² = {r2_ols(["S"]):.3f}')
out.append(f'  仅 R2_e       R² = {r2_ols(["R2_e"]):.3f}')
out.append(f'  S + R2_e      R² = {r2_ols(["S", "R2_e"]):.3f}')
out.append(f'  S + R2_e + ws R² = {r2_ols(["S", "R2_e", "ws_mean"]):.3f}')

# ⑦ 高响应场口径：正文 29 的出处与 27 的关系
t1 = pd.read_csv(os.path.join(OUT, 'task1_training_data.csv'), encoding='utf-8-sig').set_index('farm_id')
com = [f for f in t1.index if f in A_real.index]
a1, a7 = t1.loc[com, 'A'], A_real.loc[com]
out_t1 = set(t1.index) - set(A_real.index)
out_w7 = set(A_real.index) - set(t1.index)
d1, d7 = a1 > 5.2, a7 > 5.2
flips = sorted((a1[d1 & ~d7]).index.tolist())
_rho = spearmanr(a1, a7)[0]
out.append(f'  两代 A 在同 167 场上的 Spearman = {_rho:.3f}；task1 29 场 vs wp7a 27 场；')
out.append(f'  翻转场：{flips}（task1 A = {[round(a1[f], 2) for f in flips]} → '
           f'wp7a A = {[round(a7[f], 2) for f in flips]}）；')
out.append(f'  wp7a 多出 4 场 {sorted(out_w7)}（task1 缺）：A = '
           f'{[round(A_real[f], 2) for f in sorted(out_w7)]}，均不越线')


out.append('\n⑦ 高响应场口径：29 与 27 都成立，来自两代管线')
out.append('  正文 29 出处：task1_training_data.csv（167 场，task1 旧管线：nrel_5MW 统一机型）；')
out.append('  当前链 27 出处：wp7a 冻结版（171 场，iea_10MW + 逐场 TI）。')
out.append('  两代 A 在共同 167 场上的 Spearman = 0.983——管线升级未改变任何结论。')
out.append('  差异只发生在阈值边缘的 2 个场：F90 5.56→4.81、F145 6.75→4.83（5.2% 线两侧）；')
out.append('  wp7a 多出的 4 场（F1/F93/F160/F161，A 0.28–1.72）均不越线。')
out.append('  ⇒ 27/171 与 29/167 是同一事实的两种口径；正文随建设范式链（iea_10MW）统一用 27。')
hi = df.A_real > 5.2
dense = S < 4
out.append(f'  27 场高响应：100% 在密排内（S<4D），R2_e 中位 {df.loc[hi, "R2_e"].median():.3f}，'
           f'走廊成员 {int(df.loc[hi, "is_corr"].sum())}/27')
out.append('  密排内部判别（123 场）：')
out.append(f'    corr(A, disp_f|密排) = {spearmanr(df.A_real[dense], df.disp_f[dense])[0]:+.3f}  '
           f'corr(A, R2_e|密排) = {spearmanr(df.A_real[dense], df.R2_e[dense])[0]:+.3f}  '
           f'corr(disp_f, R2_e) = {spearmanr(df.disp_f[dense], df.R2_e[dense])[0]:+.3f}')
out.append(f'  走廊 23 场内: S 范围 {df.spacing_D_med[df.is_corr].min():.2f}–{df.spacing_D_med[df.is_corr].max():.2f} D，'
           f'R2_e 范围 {df.R2_e[df.is_corr].min():.2f}–{df.R2_e[df.is_corr].max():.2f}，'
           f'corr(S, R2_e) = {spearmanr(df.spacing_D_med[df.is_corr], df.R2_e[df.is_corr])[0]:+.3f}'
           f'（更密与更轴向同现）')

# ⑧ 裁定
out.append('\n⑧ 故事裁定（窄风玫瑰 vs 小间距）')
out.append('  ① 窄风玫瑰（一阶集中度 R1）不是可靠解释：本文 2014–2024 逐时口径 ρ=+0.06 (p=0.46)、'
           'ρ(A,R1_e)=−0.05 (p=0.51)——区分力≈0；廷显 1981–2010 逐日口径 ρ=+0.26 (p=7e-4) '
           '但高/低响应组中位几乎相同（0.268 vs 0.263），控制 R2 后残差仅 +0.18（R2 残差 +0.60，'
           '3.3 倍强）。两套风数据下 R1 自身跨管线相关仅 0.545（R2 为 0.881）。')
out.append('     旧 n=7 的 −0.536 是小样本选择伪影：7 场中 R1 与 R2 负相关，高响应场恰为双峰轴向风玫瑰。')
out.append('     旧联合模型"风向分散度系数为正"不再引用（分散度在全样本与密排内均与 A 无关）。')
out.append('  ② 小间距是必要条件（门槛）：疏排(S≥4D) 48 场 0 场高响应；27 场高响应 100% 密排；'
           'ρ(A,S)=−0.75 (p=4.8e-32)。')
out.append('  ③ 轴向风气候（二阶轴向集中度 R2，物理上对应 L(θ)≈L(θ+180°) 双峰往返）是幅度调制：'
           '密排×高R2 32.8% vs 密排×低R2 8.9%；偏相关 ρ(A,R2_e|S)=+0.59 (p=4.7e-17)。')
out.append('  ④ 两者独立贡献且间距占优：偏相关 |S| 0.73 > |R2| 0.59；单因子 R² S 0.561 > R2 0.341；'
           '联合 R² 0.693（+风速 0.775）。')
out.append('  ⑤ 地理汇聚：走廊 S 中位 2.01D vs 3.43D、R2_e 0.779 vs 0.303（p 均 <2e-6）；'
           '走廊内 S 与 R2_e 更同现（−0.66）——海峡/湾口受约束海域密排，海峡管束与季风形成轴向风。')
out.append('  ⇒ 结论二新故事：小间距是门槛（必要条件），轴向风气候在门槛内塑造幅度（3.7 倍判别），'
           '建成相位决定兑现；窄风玫瑰不作为自变量出现。')

# ⑨ 廷显独立复查裁定（2026-08-19）
out.append('\n⑨ 廷显独立复查裁定（2026-08-19，ERA5 1981–2010 逐日 12:00 UTC 就近格点 + task3 18 角度 A）')
out.append('  其结论（对照本文口径）：')
out.append('    间距 −0.748 (p=6.8e-32)  ← 与本文 −0.749 逐位一致，小间距主导铁证')
out.append('    R2 +0.629 / R2E +0.622      ← 与本文 +0.610 / +0.584 一致，两套风数据下稳定')
out.append('    密排内高 R2E 37.5% vs 低 9.1%、疏排 0% ← 与本文 32.8% / 8.9% / 0% 同结构')
out.append('    R1 +0.256 (p=7e-4) 弱正    ← 与本文 +0.06 (p=0.46) 不一致 → 裁定如下')
tx = pd.read_csv(os.path.join(os.path.dirname(OUT), '..', '结论三、四重算结果', '_tx_recheck',
                              'conclusion2_all171_per_farm.csv'), encoding='utf-8-sig')
def pcorr(a, b, c):
    m = a.notna() & b.notna() & c.notna()
    a_, b_, c_ = a[m].rank(), b[m].rank(), c[m].rank()
    return spearmanr(b_ - np.polyval(np.polyfit(c_, b_, 1), c_),
                     a_ - np.polyval(np.polyfit(c_, a_, 1), c_))
_r = pcorr(tx.A_pct, tx.R1_freq, tx.R2_freq)
out.append('  R1 裁定：')
out.append('    ① 跨风数据不稳定：廷显 R1 与本文 R1 的 Spearman 仅 0.545（R2 为 0.881）——'
           'R1 本身对数据源敏感；')
out.append('    ② 区分力≈0（廷显自报）：高响应组 R1 中位 0.268 vs 低响应组 0.263；')
out.append(f'    ③ 控制 R2 后 R1 残差仅 {_r[0]:+.3f} (p={_r[1]:.1e})，'
           f'R2 残差 +0.596——R2 解释力为 R1 的 3.3 倍；')
out.append('    ④ 控制间距后 R1 在廷显数据上 +0.37，但在本文数据上 −0.06——残差信号同样不稳健。')
out.append('  ⇒ 措辞修正：不写"R1 与 A 无关"，改为"窄风玫瑰不是高响应的可靠解释：'
           '区分力≈0、信号跨风数据不稳定；可靠的气候侧变量是二阶轴向集中度 R2"。')
out.append('  29/27：廷显 A 出自 task3 18 角度 S1 扫描（与 wp7a 72 角度 A 的 Spearman 0.982、'
           '与 wp1 间距 0.978），29 场是阈值边缘口径差，故事不受影响。')

txt = '\n'.join(out)
print(txt)
with open(os.path.join(OUT, 'wp9c_conclusion2_recheck.txt'), 'w', encoding='utf-8') as f:
    f.write(txt + '\n')
df.to_csv(os.path.join(OUT, 'wp9c_farm_metrics.csv'), index=False, encoding='utf-8-sig')
print('\n已写: wp9c_conclusion2_recheck.txt + wp9c_farm_metrics.csv')
