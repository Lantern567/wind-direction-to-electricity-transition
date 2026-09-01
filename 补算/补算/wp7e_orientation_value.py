"""
P2-1（wp7e）：模板替换下的项目级朝向价值（155 项目 × 六模板）
================================================================
口径（与清单 P2-1 一致，全部实算）：
  - E_pool[f, k, θ]（wp5c_cross_farms.npz）：kWh/台·年，场址 f 的 2014–2024 气候期
    多年平均年发电量（64 机位标准模板 k，72 个 5° 刚性旋转角）；
  - dE_kWh_per_turb = max_θ E_pool[f,k,:] − mean_θ E_pool[f,k,:]（θ 遍历 72 个方向，
    等价于 A 的半圆窗口口径：E(θ)=E(θ+180°)）；
  - 项目级：dE_GWh_orientation = n_turb[f] × dE_kWh_per_turb / 1e9；
  - pct_orientation = 100 × dE / E_mean（应与 wp5c A[:,k] 逐格一致——A 即
    100×(max−mean)/mean 窗口口径）；
  - 项目级口径 = n_turb ≥ 10 的 155 个项目（与 wp7c_scenario_table 一致）。
输出：结论三、四重算结果/output-new/wp7e_orientation_value_by_template.csv
      列 farm_id, template, dE_GWh_orientation, pct_orientation, A_pct, E_rot_mean_kWh_per_turb
"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT_NEW = os.path.join(REPO, '结论三、四重算结果', 'output-new')
PARADIGM_ORDER = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']

z5 = np.load(os.path.join(OUT_NEW, 'wp5c_cross_farms.npz'), allow_pickle=True)
E_pool = z5['E_pool'].astype(float)          # (171,6,72) kWh/台·年
A5 = z5['A'].astype(float)                   # (171,6) %
fids = z5['farm_ids'].astype(int)

tc = pd.read_csv(os.path.join(REPO, 'data', 'task0', 'turbine_coordinates.csv'), encoding='utf-8-sig')
nt = tc.groupby('farm_id')['turbine_id'].nunique()
proj = sorted(nt[nt >= 10].index)
print(f'项目级（n_turb ≥ 10）: {len(proj)} 个项目')

rows = []
for f in proj:
    i = int(np.where(fids == f)[0][0])
    for k, pn in enumerate(PARADIGM_ORDER):
        E = E_pool[i, k, :]
        Emax, Emean = E.max(), E.mean()
        dE_kwh = Emax - Emean
        dE_GWh = nt[f] * dE_kwh / 1e9
        pct = 100 * dE_kwh / Emean
        rows.append(dict(farm_id=f, template=pn,
                         dE_GWh_orientation=round(dE_GWh, 6),
                         pct_orientation=round(pct, 3),
                         A_pct=round(float(A5[i, k]), 3),
                         E_rot_mean_kWh_per_turb=round(float(Emean), 1)))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT_NEW, 'wp7e_orientation_value_by_template.csv'),
           index=False, encoding='utf-8-sig')
print(f'wp7e_orientation_value_by_template.csv 落盘: {len(tab)} 行（155 × 6）')

# 交叉检验：pct_orientation == A
gap = (tab.pct_orientation - tab.A_pct).abs().max()
print(f'交叉检验: pct_orientation 与 wp5c A 逐格最大差 = {gap:.3f} pp（应≈0）')

# ── 汇总（六模板 × 155 项目） ──
print('\n[155 项目 × 六模板：最优朝向相对旋转平均的年发电量差]')
summ = tab.groupby('template').agg(
    dE_med_GWh=('dE_GWh_orientation', 'median'), dE_sum_GWh=('dE_GWh_orientation', 'sum'),
    pct_med=('pct_orientation', 'median'), n=('farm_id', 'size'))
print(summ.round(3).to_string())

# 走廊项目（项目级 20 个）拆分（S_E 模板）
CORR = {'Vietnam': [56, 57, 64, 86, 107, 112, 115, 126, 130, 133, 141, 143, 159],
        'TaiwanStrait': [12, 66, 85, 91, 92, 97, 103, 105], 'Adriatic': [155], 'Danish': [157]}
corr_set = {f for lst in CORR.values() for f in lst}
corr_proj = [f for f in proj if f in corr_set]
se = tab[tab.template == 'S_E'].set_index('farm_id')
print(f'\n[S_E (11.8D) 模板：155 项目中走廊 {len(corr_proj)} 个 vs 其他 {155-len(corr_proj)} 个]')
for nm, lst in [('走廊', corr_proj), ('其他', [f for f in proj if f not in corr_set])]:
    sub = se.loc[lst]
    print(f'  {nm}: dE 中位 {sub.dE_GWh_orientation.median():.3f} GWh/yr, '
          f'合计 {sub.dE_GWh_orientation.sum():.1f} GWh/yr, pct 中位 {sub.pct_orientation.median():.2f}%')
print(f'  S_E 全部 155 项目: 合计 {se.dE_GWh_orientation.sum():.1f} GWh/yr, '
      f'pct 中位 {se.pct_orientation.median():.2f}%')
