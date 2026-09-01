# -*- coding: utf-8 -*-
"""WP9G 风速窗口机制诊断（5.8 节补算：6 个逐场越线风场，含 F160）
====================================================================
背景：v3.10 正文越线集改为 [57,66,91,155,157,160]（6 场），5.8 节"对5个
逐场越线风场"需补算 F160 才能如实改为 6。原 5.8 计算脚本不在仓库内
（与冻结比值 1.14–3.19 同属学长上游流水线），故按正文方法描述重建，
6 场统一用同一口径重算（本脚本为唯一权威实现）。

方法（按 5.8 文字逐项落实）：
  - 长期真实风向频率 f(d)：wp3_climate_joint.npz 的 p_fy（171 场 × 11 年
    2014–2024 × 18 风速档 × 72 风向档，ERA5 逐时聚成，按年时数加权合并；
    仓库内唯一完整的场级真实长期记录，1981–2010 场级数据不全）。
  - 场级 Weibull 风速边际 w(u)：风速边际直方图矩法拟合 (k, A)，按 CDF 求
    各档权重（档界：3–14 ±0.5；16:[15,17]、18:[17,19]、20:[19,21]、
    22:[21,23.5]、25:[23.5,27.5]；30 m/s 档 P0=0 剔除），档内归一。
  - 方向—风速权重 = f(d)·w(u) 独立组合（与 5.8"不保留逐时风速—风向相关"一致）。
  - 相位响应 E(θ_k)：建成排布方向—风速效率面 η(u,d)（wp7a_real_eta）刚性
    旋转 E_k = Σ_d f(d) Σ_u w(u) P0(u) η(u, (d+k) mod 72)，k=0..71（5°步长）。
    （方向约定与 wp7a 冻结口径一致：η 的 d 索引下"旋转 k 档"对应 (d+k)；
     经 E_pool 验证，本式 argmax 落在 th_star 或 th_star+180°（2 瓣响应），
     与冻结 th_star 一致。）
  - 最优相位 θ_opt = th_star（1981–2010 历史选角，wp7a 冻结值）。
  - 平均相位（三种口径全算，主口径 B＝平均风向，供学长裁决）：
      A. 响应相位均值：E_k 能量加权的圆周平均；
      B. 平均风向：f(d) 加权的圆周平均（"平均相位"的自然读法）；
      C. 正交对照：θ_opt+90°（2 瓣响应谷值，解释最干净）。
  - 可恢复能量分解：R(bin) = Σ_d f(d) Σ_{u∈bin} w(u) P0(u)
      [η(u,(d+k_opt)) − η(u,(d+k_ref))]，窗口 3–6 / 7–10 / 11–14 / 15–25 m/s。
诊断口径不参与逐时联合气象的 G 数值比较（5.11 节口径声明一致）。

输出：补算/output/wp9g_windwindow_diagnosis.csv
"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import weibull_min

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')

CROSS = [57, 66, 91, 155, 157, 160]        # 6 个逐场越线风场（v3.10 口径）

# ── 数据 ──
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_fy, hours = z3['p_fy'], z3['hours']       # (171,11,18,72) / (171,11)
ids3, ws, wd = z3['farm_ids'], z3['ws'], z3['wd']
eta_z = np.load(os.path.join(OUT, 'wp7a_real_eta.npz'))
eta = eta_z['eta'].astype(float)            # (171,18,72)
P0 = eta_z['P0'].astype(float)              # (18,)
eta_ids = eta_z['farm_ids']
cur = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
th_star, A_full, cur_ids = cur['th_star'], cur['A_full'], cur['farm_ids']

# 三个 npz 的 farm_ids 排列不同（wp3 顺序 0..170，wp7a 为并行任务序），逐场查索引
assert np.array_equal(ws, eta_z['ws']) and np.array_equal(wd, eta_z['wd'])
assert np.array_equal(wd, np.arange(0, 360, 5))
assert np.array_equal(np.sort(ids3), np.sort(eta_ids)) and \
       np.array_equal(np.sort(ids3), np.sort(cur_ids)), 'id 集合不一致'

# 窗口档映射（ws 中心值；30 档功率为 0，剔除）
BINS = {'3-6 m/s': [3, 4, 5, 6], '7-10 m/s': [7, 8, 9, 10],
        '11-14 m/s': [11, 12, 13, 14], '15-25 m/s': [16, 18, 20, 22, 25]}
U_IDX = {float(u): j for j, u in enumerate(ws) if u != 30.0}
assert all(float(u) in U_IDX for us in BINS.values() for u in us)
U_LIST = [float(u) for u in ws if u != 30.0]        # 17 档
U_J = [j for j, u in enumerate(ws) if u != 30.0]
P0U = np.array([P0[j] for j in U_J])                # (17,)
# 档界（CDF 积分用）
EDGES = {}
for u in U_LIST:
    if u <= 14:
        EDGES[u] = (u - 0.5, u + 0.5)
    elif u == 16:
        EDGES[u] = (15.0, 17.0)
    elif u == 18:
        EDGES[u] = (17.0, 19.0)
    elif u == 20:
        EDGES[u] = (19.0, 21.0)
    elif u == 22:
        EDGES[u] = (21.0, 23.5)
    elif u == 25:
        EDGES[u] = (23.5, 27.5)

rows = []
for f in CROSS:
    i = int(np.where(ids3 == f)[0][0])
    iE = int(np.where(eta_ids == f)[0][0])
    iC = int(np.where(cur_ids == f)[0][0])
    # 方向频率（按年时数加权合并 11 年真实逐时）
    pf = p_fy[i]                                   # (11,18,72)
    hw = hours[i].astype(float)
    assert hw.sum() > 0
    joint = np.tensordot(hw / hw.sum(), pf, axes=(0, 0))    # (18,72)
    f_dir = joint[U_J, :].sum(axis=0)              # 边际到方向 (72,)
    f_dir = f_dir / f_dir.sum()
    # Weibull 风速边际（矩法，边际直方图）
    marg = joint.sum(axis=1)                       # (18,)
    mu = float(np.sum(marg * ws))
    sig = float(np.sqrt(np.sum(marg * (ws - mu) ** 2)))
    kw = (sig / mu) ** (-1.086)                    # 矩法近似形状参数
    A_weib = mu / gamma(1 + 1 / kw)
    w_u = np.array([weibull_min.cdf(EDGES[u][1], kw, loc=0, scale=A_weib)
                    - weibull_min.cdf(EDGES[u][0], kw, loc=0, scale=A_weib)
                    for u in U_LIST])
    w_u = w_u / w_u.sum()
    # 相位响应 E_k（72 档 5°，旋转 = 索引平移 (d+k)，与 wp7a 冻结口径一致）
    etaU = eta[iE][U_J, :]                         # (17,72)
    wp = (w_u[:, None] * P0U[:, None] * etaU)      # (17,72) 每方向每档加权响应
    E = np.array([np.sum(f_dir * wp[:, (np.arange(72) + k) % 72].sum(axis=0))
                  for k in range(72)])
    # 相位量
    k_opt = int(th_star[iC] // 5)                  # 冻结历史最优相位
    k_arg = int(np.argmax(E))
    th = np.deg2rad(np.arange(72) * 5)
    C = np.sum(E * np.cos(th)); S = np.sum(E * np.sin(th))
    th_meanA = float(np.rad2deg(np.arctan2(S, C)) % 360)   # 口径 A
    k_meanA = int(round(th_meanA / 5)) % 72
    Cw = np.sum(f_dir * np.cos(th)); Sw = np.sum(f_dir * np.sin(th))
    th_meanB = float(np.rad2deg(np.arctan2(Sw, Cw)) % 360)  # 口径 B＝平均风向
    k_meanB = int(round(th_meanB / 5)) % 72
    k_orth = (k_opt + 18) % 72                     # 口径 C
    # 可恢复能量分解
    def decomp(k_ref):
        tot = 0.0; per = {}
        for name, us in BINS.items():
            js = [U_IDX[float(u)] for u in us]
            s = 0.0
            for j in js:
                s += w_u[j] * P0U[j] * np.sum(
                    f_dir * (etaU[j, (np.arange(72) + k_opt) % 72]
                             - etaU[j, (np.arange(72) + k_ref) % 72]))
            per[name] = s
            tot += s
        return tot, per
    RA, perA = decomp(k_meanA)
    RB, perB = decomp(k_meanB)
    RO, perO = decomp(k_orth)
    Eopt = E[k_opt]
    # 自检：冻结 th_star 应落在 f(d) 加权响应的峰位（2 瓣响应下 argmax 可为对瓣；
    # 允许 1% 容差，超出即说明历史选角与 2014–2024 加权响应跨期漂移，须上报）
    assert Eopt >= 0.99 * E.max(), \
        f'F{f}: E(th_star)/E(max) = {Eopt/E.max():.3f} < 0.99'
    row = {
        'farm_id': f, 'Weibull_k': round(kw, 3), 'Weibull_A': round(A_weib, 2),
        'mean_ws_marg': round(mu, 2),
        'th_opt_deg': float(th_star[iC]), 'th_argmax_deg': int(k_arg * 5),
        'E_opt_over_max': round(float(Eopt / E.max()), 4),
        'th_meanA_deg': round(th_meanA, 1), 'th_meanB_deg': round(th_meanB, 1),
        'recover_A_pct': round(100 * (Eopt / E[k_meanA] - 1), 2),
        'recover_B_pct': round(100 * (Eopt / E[k_meanB] - 1), 2),
        'recover_orth_pct': round(100 * (Eopt / E[k_orth] - 1), 2),
        'A_full_pct': round(float(A_full[iC]), 2),
    }
    for lbl, R, per in (('A', RA, perA), ('B', RB, perB), ('O', RO, perO)):
        for name in BINS:
            row[f'share{lbl}_{name.replace(" m/s", "").replace("-", "_")}'] = \
                round(100 * per[name] / R, 1) if R != 0 else float('nan')
    rows.append(row)
    print(f'F{f}: Weibull k={kw:.2f} A={A_weib:.1f}  θ_opt={th_star[iC]}° '
          f'argmaxE={k_arg*5}°  θ_meanA={th_meanA:.0f}° θ_meanB={th_meanB:.0f}° '
          f'recover A={100*(Eopt/E[k_meanA]-1):+.2f}% '
          f'B={100*(Eopt/E[k_meanB]-1):+.2f}% '
          f'orth={100*(Eopt/E[k_orth]-1):+.2f}% (A_full={A_full[iC]:.2f}%)')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'wp9g_windwindow_diagnosis.csv'),
          index=False, encoding='utf-8-sig')
print('\n=== 风速窗口份额（主口径 B：相对平均风向）===')
cols = [c for c in df.columns if c.startswith('shareB_')]
print(df[['farm_id'] + cols].to_string(index=False))
print('\n=== 风速窗口份额（正交对照：相对 θ_opt+90°）===')
cols = [c for c in df.columns if c.startswith('shareO_')]
print(df[['farm_id'] + cols].to_string(index=False))
print('\nsaved: wp9g_windwindow_diagnosis.csv')
