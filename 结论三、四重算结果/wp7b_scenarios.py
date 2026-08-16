"""
WP7b 结论四：三类嵌套建设情境（已建风场方法验证版）
=================================================
方案 §5.6/§5.9 明确：全球 TWh 层缺项目管线/租区数据 → 先完成已建风场的
S1—S3 方法验证，只报告项目级百分比、MWh MW⁻¹ yr⁻¹、已覆盖项目 GWh。
（"旧版 225 GW/23.5 TWh 只保留为审计对照，不进入中心结果"）

可行集（统一机型 iea_10MW、统一项目容量 C_i = n_turb_i × 10 MW）：
  S1：建成排布刚性旋转（真实机位 × 72 朝向）           → wp7a E_pool
  S2：保持建成形态类型 + 匹配间距档（3 重复 × 72 朝向）→ wp5 E_cf 子集
  S3：任意形态 × 任意间距（36 模板 × 72 朝向）         → wp5 E_cf 全集
基线：G_plan 相对建成方案 E(0°)；G_info（敏感性）相对可行集均匀先验
分解：V1 = E_S1*−E_base；V2 = E_S2*−E_S1*；V3 = E_S3*−E_S2*
样本外：2014—2019 选角 → 2020—2024 逐年评价（正增益年份比例、后悔值）
口径：MWh MW⁻¹ yr⁻¹ = E_turb/10e3；GWh yr⁻¹ = C_i·Δe/1000；
      单位海域 = E/S（S1 用机位凸包，S2/S3 用模板面积）
单调性验收：E_S1* ≤ E_S2* ≤ E_S3*（违反时如实报告——n_turb<64 的场址
  模板尾流损失更高属预期；主分析不含 sparse（n<10，方案 §3 主实验四类）

输入：wp5_cross_farms.npz / wp7a_real_curves.npz / wp2_template_summary.csv
输出：wp7b_scenario_table.csv / wp7b_corridor_summary.csv / wp7b_report.txt
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

z5 = np.load(os.path.join(OUT, 'wp5_cross_farms.npz'))
E_cf = z5['E_cf']                       # (171,4,36,72) kWh/台·年
E_fy = z5['E_fy']                       # (171,11,36,72)
farm_ids = z5['farm_ids'].astype(int)
z7 = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
# wp7a 按 n_turb 升序保存 → 重排到 z5 的 farm_id 升序
pos = {int(f): i for i, f in enumerate(z7['farm_ids'].astype(int))}
idx = np.array([pos[int(f)] for f in farm_ids])
E_real = z7['E_pool'][idx]              # (171,72)
E_real_y = z7['E_y'][idx]               # (171,11,72)
n_turb = z7['n_turb'][idx]
area_real = z7['area_km2'][idx]

summ = pd.read_csv(os.path.join(OUT, 'wp2_template_summary.csv'), encoding='utf-8-sig')
MORPH = summ['morphology'].values; SPAC = summ['spacing_D'].values
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')
nF = len(farm_ids)
nH = 36

# ═══════════════════════════════════════════════════════════════════════
# 1. 逐项目三情境最优发电量
# ═══════════════════════════════════════════════════════════════════════
E_S1 = E_real.max(axis=1)                      # 刚性旋转最优
E_base = E_real[:, 0]                          # 建成朝向
rows = []
for i, fid in enumerate(farm_ids):
    lt = geo.loc[fid, 'layout_type']
    sp_med = geo.loc[fid, 'spacing_D_med']
    sp_match = 3.0 if sp_med <= 4.0 else (5.0 if sp_med <= 6.0 else 7.0)
    s2 = [k for k in range(36) if MORPH[k] == lt and SPAC[k] == sp_match]
    s3 = list(range(36))
    E_S2 = E_cf[i, 0, s2].max() if s2 else np.nan
    E_S3 = E_cf[i, 0, s3].max()
    # 样本外（2014-2019 选角 → 2020-2024 评价）
    # S1：真实排布
    E_hist = E_real_y[i, [HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)
    th_hist = np.argmax(E_hist[:nH]) * 5
    E_ev = E_real_y[i, [EVAL_YEARS.index(y) for y in EVAL_YEARS]]        # (5,72)
    g_so1 = 100 * (E_ev[:, th_hist // 5] - E_ev[:, :nH].mean(axis=1)) / np.maximum(E_ev[:, :nH].mean(axis=1), 1e-9)
    # S2/S3：模板（历史期选模板+相位）
    if s2:
        Ey2 = E_fy[i][:, s2]                       # (11,ns2,72)
        Eh2 = Ey2[[HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)  # (ns2,72)
        k2, t2 = np.unravel_index(np.argmax(Eh2[:, :nH]), Eh2[:, :nH].shape)
        E_ev2 = Ey2[[EVAL_YEARS.index(y) for y in EVAL_YEARS], k2]       # (5,72)
        g_so2 = 100 * (E_ev2[:, t2] - E_ev2[:, :nH].mean(axis=1)) / np.maximum(E_ev2[:, :nH].mean(axis=1), 1e-9)
    else:
        g_so2 = np.full(5, np.nan)
    Ey3 = E_fy[i]
    Eh3 = Ey3[[HIST_YEARS.index(y) for y in HIST_YEARS]].mean(axis=0)
    k3, t3 = np.unravel_index(np.argmax(Eh3[:, :nH]), Eh3[:, :nH].shape)
    E_ev3 = Ey3[[EVAL_YEARS.index(y) for y in EVAL_YEARS], k3]
    g_so3 = 100 * (E_ev3[:, t3] - E_ev3[:, :nH].mean(axis=1)) / np.maximum(E_ev3[:, :nH].mean(axis=1), 1e-9)
    rows.append(dict(
        farm_id=int(fid), n_turb=int(n_turb[i]), layout_type=lt,
        spacing_D_med=float(sp_med), spacing_match=sp_match,
        capacity_MW=int(n_turb[i]) * 10,
        E_base=float(E_base[i]), E_S1=float(E_S1[i]),
        E_S2=float(E_S2) if s2 else np.nan, E_S3=float(E_S3),
        G_plan_S1=100 * (E_S1[i] - E_base[i]) / E_base[i],
        G_plan_S2=100 * (E_S2 - E_base[i]) / E_base[i] if s2 else np.nan,
        G_plan_S3=100 * (E_S3 - E_base[i]) / E_base[i],
        G_info_S1=100 * (E_S1[i] - E_real[i, :nH].mean()) / E_real[i, :nH].mean(),
        G_info_S2=100 * (E_S2 - E_cf[i, 0, s2, :nH].mean()) / E_cf[i, 0, s2, :nH].mean() if s2 else np.nan,
        G_info_S3=100 * (E_S3 - E_cf[i, 0, :, :nH].mean()) / E_cf[i, 0, :, :nH].mean(),
        V1=float(E_S1[i] - E_base[i]), V2=float(E_S2 - E_S1[i]) if s2 else np.nan,
        V3=float(E_S3 - (E_S2 if s2 else E_S1[i])),
        pos_frac_S1=float((g_so1 > 0).mean()), pos_frac_S2=float((g_so2 > 0).mean()),
        pos_frac_S3=float((g_so3 > 0).mean()),
        g_so_S1_p05=np.percentile(g_so1, 5), g_so_S1_p95=np.percentile(g_so1, 95),
        g_so_S2_p05=np.percentile(g_so2, 5), g_so_S2_p95=np.percentile(g_so2, 95),
        g_so_S3_p05=np.percentile(g_so3, 5), g_so_S3_p95=np.percentile(g_so3, 95),
        monotonic=bool((E_S1[i] <= (E_S2 if s2 else np.inf)) and
                       ((E_S2 if s2 else -np.inf) <= E_S3) and np.isfinite(E_S2)),
    ))
tb = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════
# 2. GWh 与单位口径
# ═══════════════════════════════════════════════════════════════════════
C_MW = tb['capacity_MW'].values
for s, name in [('S1', 'E_S1'), ('S2', 'E_S2'), ('S3', 'E_S3')]:
    e = tb[name].values / 10e3                      # MWh/MW/yr
    e_base = tb['E_base'].values / 10e3
    tb[f'dE_GWh_{s}'] = C_MW * (e - e_base) / 1000  # GWh/yr
    tb[f'e_MWhperMW_{s}'] = e
tb['e_base_MWhperMW'] = tb['E_base'] / 10e3
# 单位海域（S1 用机位凸包；S2/S3 用所选模板面积）
tpl_area = {k: float(summ['area_km2'].values[k]) for k in range(36)}
tb['area_real_km2'] = area_real
tb['Y_area_S1'] = tb['E_S1'] / np.maximum(area_real, 1e-9)
Y_area_S2, Y_area_S3 = [], []
for i, fid in enumerate(farm_ids):
    s2 = [k for k in range(36) if MORPH[k] == geo.loc[fid, 'layout_type'] and
          SPAC[k] == (3.0 if geo.loc[fid, 'spacing_D_med'] <= 4 else (5.0 if geo.loc[fid, 'spacing_D_med'] <= 6 else 7.0))]
    Y_area_S2.append(float(E_cf[i, 0, s2].max(axis=0).max() / np.maximum(np.mean([tpl_area[k] for k in s2]), 1e-9)) if s2 else np.nan)
    k3 = int(np.unravel_index(np.argmax(E_cf[i, 0, :, :]), (36, 72))[0])   # 与 E_S3 同口径：全圆选模板
    Y_area_S3.append(float(E_cf[i, 0, k3].max() / np.maximum(tpl_area[k3], 1e-9)))
tb['Y_area_S2'] = Y_area_S2; tb['Y_area_S3'] = Y_area_S3

# 走廊标注（fig4/汇总共同使用 → 入表）
def corr_of(fid):
    for c, ms in CORRIDORS.items():
        if fid in ms:
            return c
    return 'other'
tb['corridor'] = tb['farm_id'].apply(corr_of)

tb.to_csv(os.path.join(OUT, 'wp7b_scenario_table.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════
# 3. 走廊汇总（主结果表，方案 §5.8）
# ═══════════════════════════════════════════════════════════════════════
main = tb[tb.layout_type != 'sparse']
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
sum_df.to_csv(os.path.join(OUT, 'wp7b_corridor_summary.csv'), index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════════
# 4. 报告
# ═══════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT, 'wp7b_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP7b 结论四：S1-S3 建设情境（已建风场方法验证）\n' + '=' * 60 + '\n')
    f.write(f'项目数: {len(main)}（非 sparse）/ {len(tb)}（全部）\n')
    f.write('容量口径: C_i = n_turb × 10 MW | G_plan 基线 = 建成朝向 E(0°) | '
            'G_info 基线 = 可行集均匀先验\n')
    f.write('样本外: 2014-2019 选角 → 2020-2024 逐年评价\n\n')
    f.write('全组合（非 sparse）:\n')
    allm = main
    f.write(f'  G_plan: S1 中位 {allm.G_plan_S1.median():.2f}% | S2 {allm.G_plan_S2.median():.2f}% | '
            f'S3 {allm.G_plan_S3.median():.2f}%\n')
    f.write(f'  ΔE:     S1 {allm.dE_GWh_S1.sum():.1f} GWh/yr | S2 {allm.dE_GWh_S2.sum():.1f} | '
            f'S3 {allm.dE_GWh_S3.sum():.1f}（覆盖 {allm.capacity_MW.sum()/1000:.1f} GW）\n')
    f.write(f'  分解:   V1/V2/V3 = {allm.V1.sum()/allm.V1.abs().sum():.2f} / '
            f'{allm.V2.sum()/allm.V1.abs().sum():.2f} / {allm.V3.sum()/allm.V1.abs().sum():.2f}\n')
    f.write(f'  单调性 E_S1≤E_S2≤E_S3: {allm.monotonic.mean():.1%} 项目通过（n_turb<64 的场址'
            f'模板尾流更高属预期，见逐项目表）\n')
    f.write(f'  样本外正增益年份比例（中位）: S1 {allm.pos_frac_S1.median():.2f} | '
            f'S3 {allm.pos_frac_S3.median():.2f}\n\n')
    f.write('走廊汇总:\n')
    f.write(sum_df.to_string(index=False) + '\n\n')
    f.write('注: 全球 TWh 汇总待项目管线/租区多边形数据（方案 §5.6）；'
            '本表仅代表已建风场方法验证。sparse（n<10）项目单独报告：\n')
    sp = tb[tb.layout_type == 'sparse']
    if len(sp):
        f.write(sp[['farm_id', 'G_plan_S1', 'G_plan_S3', 'dE_GWh_S1', 'dE_GWh_S3']].to_string(index=False))
print('输出: wp7b_scenario_table.csv / wp7b_corridor_summary.csv / wp7b_report.txt')
print(sum_df.to_string(index=False))
