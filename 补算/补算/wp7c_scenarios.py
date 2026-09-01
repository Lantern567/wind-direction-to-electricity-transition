"""
WP7c 结论四：S1-S3 建设情境（建设范式版，替换 wp7b 形态模板口径）
====================================================================
方案 §5.6/§5.9 + 建设范式情境设计 v3 §五：只报告已建风场方法验证的
项目级百分比、MWh MW⁻¹ yr⁻¹、已覆盖项目 GWh（全球 TWh 汇总待管线数据）。

可行集（统一机型 iea_10MW、统一项目容量 C_i = n_turb_i × 10 MW）：
  S1：建成排布刚性旋转（真实机位 × 72 朝向）              → wp7a E_pool
  S2：场址主范式情境（task1 多标签取主标签 → S_A/S_B0/S_C/S_D/S_E
      场景，64 台 × 72 朝向）                             → wp5c E_pool
  S3：5 范式情境全集（6 套 × 72 朝向）                    → wp5c E_pool
基线：G_plan 相对建成方案 E(0°)；G_info 相对可行集均匀先验
  （S2 先验 = 本范式建成窗口 36 档；S3 先验 = 6 范式 × 各自建成窗口 216 档）
分解：V1 = E_S1*−E_base；V2 = E_S2*−E_S1*；V3 = E_S3*−E_S2*
样本外：2014—2019 选角（S2/S3 在本范式建成窗口内）→ 2020—2024 逐年评价
口径：MWh MW⁻¹ yr⁻¹ = E_turb/10e3；GWh yr⁻¹ = C_i·Δe/1000；
      单位海域 = E/S（S1 机位凸包；S2/S3 用范式情境面积 wp2c 摘要）
单调性验收：E_S1* ≤ E_S2* ≤ E_S3*（违反时如实报告——n_turb≠64 的场址
  情境尾流损失差异属预期；主分析不含 sparse（n<10））

输入：wp5c_cross_farms.npz / wp7a_real_curves.npz / wp2c_paradigm_summary.csv /
      task1_paradigm_classification.csv
输出：wp7c_scenario_table.csv / wp7c_corridor_summary.csv / wp7c_report.txt
"""
import os, io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BUSH, 'output')
HOURS = 8760.0
HIST_YEARS = list(range(2014, 2020))
EVAL_YEARS = list(range(2020, 2025))
CORRIDORS = {
    'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
    'China_strait': [66, 91, 12, 92, 97, 85, 103, 105],
    'Italy': [155], 'Denmark': [157],
}
PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
LABEL2SCEN = {'A': 'S_A', 'B': 'S_B0', 'C': 'S_C', 'D': 'S_D', 'E': 'S_E'}

z5 = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'))
E_pool = z5['E_pool'].astype(np.float64)        # (171,6,72) C0 多年平均
E_fy = z5['E_fy'].astype(np.float64)            # (171,11,6,72)
farm_ids = z5['farm_ids'].astype(int)
TB = z5['TB'].astype(int)                       # (171,6) 建成基线 bin
z7 = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
pos = {int(f): i for i, f in enumerate(z7['farm_ids'].astype(int))}
idx = np.array([pos[int(f)] for f in farm_ids])
E_real = z7['E_pool'][idx]                      # (171,72)
E_real_y = z7['E_y'][idx]                       # (171,11,72)
n_turb = z7['n_turb'][idx]
area_real = z7['area_km2'][idx]

cls = pd.read_csv(os.path.join(BUSH, 'input_task1', 'task1_paradigm_classification.csv'),
                  encoding='utf-8-sig').set_index('farm_id')
main_label = {int(f): str(cls.loc[f, 'paradigm_labels']).split('+')[0] for f in farm_ids}
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
summ = pd.read_csv(os.path.join(OUT, 'wp2c_paradigm_summary.csv'), encoding='utf-8-sig').set_index('paradigm')
nF = len(farm_ids)
nH = 36

# ═══════════════════════════════════════════════════════════════════════
# 1. 逐项目三情境最优发电量
# ═══════════════════════════════════════════════════════════════════════
E_S1 = E_real.max(axis=1)                       # 刚性旋转最优（全圆）
E_base = E_real[:, 0]                           # 建成朝向
rows = []
for i, fid in enumerate(farm_ids):
    lab = main_label[fid]
    k2 = PIDS.index(LABEL2SCEN[lab])            # 主范式情境
    E_S2 = E_pool[i, k2].max()                  # 本范式全圆最优
    E_S3 = E_pool[i].max()                      # 6 范式 × 72 朝向最优
    # 样本外：S1 历史期半圆绝对窗口选角（真实排布建成轴 = 0）
    E_hist1 = E_real_y[i, [HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)
    th_hist1 = np.argmax(E_hist1[:nH]) * 5
    E_ev1 = E_real_y[i, [EVAL_YEARS.index(y) for y in EVAL_YEARS]]
    g_so1 = 100 * (E_ev1[:, th_hist1 // 5] - E_ev1[:, :nH].mean(axis=1)) / np.maximum(E_ev1[:, :nH].mean(axis=1), 1e-9)
    # S2：本范式建成窗口内选角
    Ey2 = E_fy[i][:, k2]                        # (11,72)
    Eh2 = Ey2[[HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)
    w2 = (np.arange(nH) + TB[i, k2]) % 72
    ti2 = int(np.argmax(Eh2[w2]))
    Ey2_ev = Ey2[[EVAL_YEARS.index(y) for y in EVAL_YEARS]]
    g_so2 = 100 * (Ey2_ev[:, w2[ti2]] - Ey2_ev[:, w2].mean(axis=1)) / \
        np.maximum(Ey2_ev[:, w2].mean(axis=1), 1e-9)
    # S3：6 范式 × 各自建成窗口选 (k,t)
    Ey3 = E_fy[i]                               # (11,6,72)
    Eh3 = Ey3[[HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)   # (6,72)
    wins = [(np.arange(nH) + TB[i, k]) % 72 for k in range(6)]
    best = max(((k, int(np.argmax(Eh3[k, w])), float(Eh3[k, w].max())) for k, w in enumerate(wins)),
               key=lambda x: x[2])
    k3, ti3, _ = best
    w3 = wins[k3]
    E_ev3 = Ey3[[EVAL_YEARS.index(y) for y in EVAL_YEARS], k3]
    g_so3 = 100 * (E_ev3[:, w3[ti3 // 5]] - E_ev3[:, w3].mean(axis=1)) / \
        np.maximum(E_ev3[:, w3].mean(axis=1), 1e-9)
    rows.append(dict(
        farm_id=int(fid), n_turb=int(n_turb[i]), main_label=lab, scenario_S2=PIDS[k2],
        capacity_MW=int(n_turb[i]) * 10,
        E_base=float(E_base[i]), E_S1=float(E_S1[i]),
        E_S2=float(E_S2), E_S3=float(E_S3),
        G_plan_S1=100 * (E_S1[i] - E_base[i]) / E_base[i],
        G_plan_S2=100 * (E_S2 - E_base[i]) / E_base[i],
        G_plan_S3=100 * (E_S3 - E_base[i]) / E_base[i],
        G_info_S1=100 * (E_S1[i] - E_real[i, :nH].mean()) / E_real[i, :nH].mean(),
        G_info_S2=100 * (E_S2 - E_pool[i, k2, w2].mean()) / E_pool[i, k2, w2].mean(),
        G_info_S3=100 * (E_S3 - np.mean([E_pool[i, k, w].mean() for k, w in enumerate(wins)])) /
                   np.mean([E_pool[i, k, w].mean() for k, w in enumerate(wins)]),
        V1=float(E_S1[i] - E_base[i]), V2=float(E_S2 - E_S1[i]), V3=float(E_S3 - E_S2),
        pos_frac_S1=float((g_so1 > 0).mean()), pos_frac_S2=float((g_so2 > 0).mean()),
        pos_frac_S3=float((g_so3 > 0).mean()),
        g_so_S1_p05=np.percentile(g_so1, 5), g_so_S1_p95=np.percentile(g_so1, 95),
        g_so_S2_p05=np.percentile(g_so2, 5), g_so_S2_p95=np.percentile(g_so2, 95),
        g_so_S3_p05=np.percentile(g_so3, 5), g_so_S3_p95=np.percentile(g_so3, 95),
        monotonic=bool(E_S1[i] <= E_S2 and E_S2 <= E_S3),
    ))
tb = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════
# 2. GWh 与单位口径
# ═══════════════════════════════════════════════════════════════════════
C_MW = tb['capacity_MW'].values
for s, name in [('S1', 'E_S1'), ('S2', 'E_S2'), ('S3', 'E_S3')]:
    e = tb[name].values / 10e3
    tb[f'dE_GWh_{s}'] = C_MW * (e - tb['E_base'].values / 10e3) / 1000
    tb[f'e_MWhperMW_{s}'] = e
tb['e_base_MWhperMW'] = tb['E_base'] / 10e3
# 单位海域（S1 机位凸包；S2/S3 范式情境面积）
tb['area_real_km2'] = area_real
tb['Y_area_S1'] = tb['E_S1'] / np.maximum(area_real, 1e-9)
para_area = {pid: float(summ.loc[pid, 'area_km2']) for pid in PIDS}
tb['Y_area_S2'] = [tb['E_S2'].values[i] / np.maximum(para_area[tb['scenario_S2'].values[i]], 1e-9)
                   for i in range(len(tb))]
k3_all = np.unravel_index(E_pool.reshape(nF, -1).argmax(axis=1), (6, 72))[0]
tb['Y_area_S3'] = [tb['E_S3'].values[i] / np.maximum(para_area[PIDS[k3_all[i]]], 1e-9)
                   for i in range(len(tb))]

def corr_of(fid):
    for c, ms in CORRIDORS.items():
        if fid in ms:
            return c
    return 'other'
tb['corridor'] = tb['farm_id'].apply(corr_of)
tb.to_csv(os.path.join(OUT, 'wp7c_scenario_table.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════
# 3. 走廊汇总（主结果表，方案 §5.8）
# ═══════════════════════════════════════════════════════════════════════
main = tb[tb.n_turb >= 10]
sum_rows = []
for c, g in main.groupby('corridor'):
    cap_gw = g['capacity_MW'].sum() / 1000
    sum_rows.append(dict(
        corridor=c, n_projects=len(g), capacity_GW=round(cap_gw, 2),
        G_plan_S1_med=round(g['G_plan_S1'].median(), 2),
        G_plan_S2_med=round(g['G_plan_S2'].median(), 2),
        G_plan_S3_med=round(g['G_plan_S3'].median(), 2),
        dE_S1_GWh=round(g['dE_GWh_S1'].sum(), 2),
        dE_S2_GWh=round(g['dE_GWh_S2'].sum(), 2),
        dE_S3_GWh=round(g['dE_GWh_S3'].sum(), 2),
        V1_share=round(g['V1'].sum() / g['V1'].abs().sum(), 3),
        V2_share=round(g['V2'].sum() / g['V1'].abs().sum(), 3),
        V3_share=round(g['V3'].sum() / g['V1'].abs().sum(), 3),
        pos_frac_S1_med=round(g['pos_frac_S1'].median(), 2),
        pos_frac_S3_med=round(g['pos_frac_S3'].median(), 2),
        monotonic_frac=round(g['monotonic'].mean(), 3),
    ))
sum_df = pd.DataFrame(sum_rows)
sum_df.to_csv(os.path.join(OUT, 'wp7c_corridor_summary.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════
# 4. 报告
# ═══════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT, 'wp7c_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP7c 结论四：S1-S3 建设情境（建设范式版）\n' + '=' * 60 + '\n')
    f.write(f'项目数: {len(main)}（n≥10）/ {len(tb)}（全部）\n')
    f.write('S2 = 场址主范式情境（task1 多标签取主标签 → S_A/S_B0/S_C/S_D/S_E，64 台）\n')
    f.write('S3 = 5 范式情境全集（6 套 × 72 朝向）；G_plan 基线 = 建成朝向 E(0°)；\n')
    f.write('G_info 基线 = 可行集均匀先验（S2 本范式建成窗口 / S3 全范式建成窗口）\n')
    f.write('样本外: 2014-2019 选角 → 2020-2024 逐年评价\n\n')
    f.write('全组合（n≥10）:\n')
    f.write(f'  G_plan: S1 中位 {main.G_plan_S1.median():.2f}% | S2 {main.G_plan_S2.median():.2f}% | '
            f'S3 {main.G_plan_S3.median():.2f}%\n')
    f.write(f'  ΔE:     S1 {main.dE_GWh_S1.sum():.1f} GWh/yr | S2 {main.dE_GWh_S2.sum():.1f} | '
            f'S3 {main.dE_GWh_S3.sum():.1f}（覆盖 {main.capacity_MW.sum()/1000:.1f} GW）\n')
    f.write(f'  分解:   V1/V2/V3 = {main.V1.sum()/main.V1.abs().sum():.2f} / '
            f'{main.V2.sum()/main.V1.abs().sum():.2f} / {main.V3.sum()/main.V1.abs().sum():.2f}\n')
    f.write(f'  单调性 E_S1≤E_S2≤E_S3: {main.monotonic.mean():.1%} 项目通过（n_turb≠64 场址差异属预期）\n')
    f.write(f'  样本外正增益年份比例（中位）: S1 {main.pos_frac_S1.median():.2f} | '
            f'S3 {main.pos_frac_S3.median():.2f}\n\n')
    f.write('走廊汇总:\n')
    f.write(sum_df.to_string(index=False) + '\n\n')
    f.write('注: 全球 TWh 汇总待项目管线/租区多边形数据（方案 §5.6）；本表仅代表已建风场方法验证。\n')
    f.write('sparse（n<10）项目单独报告：\n')
    sp = tb[tb.n_turb < 10]
    if len(sp):
        f.write(sp[['farm_id', 'G_plan_S1', 'G_plan_S2', 'G_plan_S3', 'dE_GWh_S1']].to_string(index=False))
print('输出: wp7c_scenario_table.csv / wp7c_corridor_summary.csv / wp7c_report.txt')
print(sum_df.to_string(index=False))
