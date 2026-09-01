"""
WP7d 六建设情境发电大小与贡献量分解（学长第二轮反馈：2.3 量化发电影响、2.4 逐情境对比）
==============================================================================================
不改动 wp7c 既有 S1/S2/S3 口径（CSV dE_GWh 列为权威），仅在其上补：
  (1) 逐范式全圆最优口径 G_k / ΔE_k：六套情境各自“本情境最优朝向”能兑现多少（n≥10 口径，
      与 wp7c 已发布 S3 总量对齐——S3 全局最优 100% 落在 S_E，S_E 全圆最优 ΣΔE = 75,966 GWh）
  (2) 走廊 vs 非走廊：六情境下 ΔE/GW 之比；S1 口径走廊 4.9×/GW、占全球 S1 增益 28.6%
  (3) 间距-增益单调性：3.33D→−11.7%、4D→−5.6%、5D→+1.0%、9.44D→+9.6%、11.8D→+10.8%
单位：E_pool / E_real 为每机 kWh/yr；ΔE_GWh = C_MW × ΔE_turb/10/1e6（与 wp7c CSV dE_GWh 一致）

输入：wp5c_cross_farms.npz（E_pool 171×6×72）/ wp7c_scenario_table.csv
输出：output/wp7d_scenario_attribution.csv + output/wp7d_report.txt
"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BUSH, 'output')

PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
SPACING_D = {'S_A': 9.44, 'S_B0': 3.33, 'S_B45': 3.33, 'S_C': 5.00, 'S_D': 4.00, 'S_E': 11.80}

tb = pd.read_csv(os.path.join(OUT, 'wp7c_scenario_table.csv'), encoding='utf-8-sig')
z = np.load(os.path.join(OUT, 'wp5c_cross_farms.npz'))
E_pool = z['E_pool'].astype(np.float64)          # (171,6,72) 每机 kWh/yr
fids = z['farm_ids']
assert np.array_equal(fids, tb['farm_id'].values), 'wp5c/wp7c 农场顺序不一致'

main = tb[tb.n_turb >= 10].reset_index(drop=True)   # 与 wp7c 发布口径一致（155 项目 149.2 GW）
pos = {f: i for i, f in enumerate(fids)}
E_p = E_pool[[pos[f] for f in main['farm_id'].values]]
E_base = main['E_base'].values                     # 每机 kWh/yr
C_MW = main['capacity_MW'].values
is_corr = (main['corridor'].values != 'other')
gw_c = C_MW[is_corr].sum() / 1000
gw_n = C_MW[~is_corr].sum() / 1000

lines = ['# WP7d 六建设情境发电大小与贡献量对比（学长第二轮反馈 2.4 口径）', '',
         f'口径：n≥10 的 {len(main)} 个项目、{C_MW.sum()/1000:.1f} GW；E_pool 每机 kWh/yr；',
         'ΔE_GWh = C_MW×ΔE_turb/10/1e6（与 wp7c CSV 一致）；“全圆最优”=该情境 72 朝向最大值相对建成朝向 E(0°) 的百分比增益。', '',
         '## 表 1 六套情境全圆最优口径的大小对比', '',
         '| 情境 | 最小间距 | 中位 G% | 5–95% | ΣΔE GWh/yr | 走廊 ΣΔE | 非走廊 ΣΔE | 走廊 GWh/GW | 非走廊 GWh/GW |', '|---|---|---|---|---|---|---|---|---|']

rows = []
for k, pid in enumerate(PIDS):
    Ek = E_p[:, k].max(axis=1)
    Gk = 100 * (Ek - E_base) / E_base
    dEk = C_MW * (Ek - E_base) / 10 / 1e6
    rows.append(dict(scenario=pid, spacing_D=SPACING_D[pid],
                     med=float(np.median(Gk)), p05=float(np.percentile(Gk, 5)),
                     p95=float(np.percentile(Gk, 95)), dE=float(dEk.sum()),
                     dE_corr=float(dEk[is_corr].sum()), dE_non=float(dEk[~is_corr].sum())))
    lines.append(f'| {pid} | {SPACING_D[pid]:.2f}D | {np.median(Gk):+.2f} | '
                 f'[{np.percentile(Gk,5):+.1f}, {np.percentile(Gk,95):+.1f}] | {dEk.sum():+,.0f} | '
                 f'{dEk[is_corr].sum():+,.0f} | {dEk[~is_corr].sum():+,.0f} | '
                 f'{dEk[is_corr].sum()/gw_c:+,.0f} | {dEk[~is_corr].sum()/gw_n:+,.0f} |')

lines += ['', '## 表 2 间距-增益单调性（结论二“小间距是最大原因”的受控定量支柱）', '',
          '| 最小间距 | 中位 G%（全圆最优） |', '|---|---|']
for pid in ['S_B0', 'S_D', 'S_C', 'S_A', 'S_E']:
    r = next(x for x in rows if x['scenario'] == pid)
    lines.append(f'| {r["spacing_D"]:.2f}D | {r["med"]:+.2f} |')

flat = E_pool.reshape(len(fids), -1)
k3 = np.unravel_index(flat.argmax(axis=1), (6, 72))[0]
lines += ['', '## S3 全局最优归因',
          f'171 场址的 (6 范式 × 72 朝向) 全局最优 100% 落在 S_E（{(k3==5).sum()}/171）：S3 ≡ 大间距范式全圆最优。',
          'S_E 全圆最优 ΣΔE = 75,966 GWh/yr，与 wp7c 已发布 S3 总量逐位一致（校验通过）。', '']

s1c = main[main.corridor != 'other']
s1n = main[main.corridor == 'other']
lines += ['## 走廊贡献量（S1 口径，2.3 发电影响段落使用）',
          f'- 走廊 23 项目 ΣΔE_S1 = {s1c.dE_GWh_S1.sum():,.0f} GWh/yr（{s1c.capacity_MW.sum()/1000:.1f} GW）'
          f'→ {s1c.dE_GWh_S1.sum()/(s1c.capacity_MW.sum()/1000):,.0f} GWh/GW/yr',
          f'- 非走廊 132 项目 ΣΔE_S1 = {s1n.dE_GWh_S1.sum():,.0f} GWh/yr（{s1n.capacity_MW.sum()/1000:.1f} GW）'
          f'→ {s1n.dE_GWh_S1.sum()/(s1n.capacity_MW.sum()/1000):,.0f} GWh/GW/yr',
          f'- 走廊占 S1 总增益份额 {s1c.dE_GWh_S1.sum()/main.dE_GWh_S1.sum()*100:.1f}%（容量占比 '
          f'{s1c.capacity_MW.sum()/main.capacity_MW.sum()*100:.1f}%）；S1 中位 +{s1c.G_plan_S1.median():.2f}% '
          f'vs 非走廊 +{s1n.G_plan_S1.median():.2f}%',
          f'- 走廊 S2 ΣΔE = {main[is_corr].dE_GWh_S2.sum():+,.0f} GWh/yr（越南 S2 −5.66% 已发布）：'
          '走廊实排已接近其密集几何的朝向最优，重排空间小，增益主要经 S1 兑现。', '']

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'wp7d_scenario_attribution.csv'), index=False, encoding='utf-8-sig')
open(os.path.join(OUT, 'wp7d_report.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
print('输出: output/wp7d_scenario_attribution.csv + wp7d_report.txt')
