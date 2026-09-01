"""
wp9d：按学长论点与论据框架补算 R4.1–R4.3 走廊级"可避免发电损失"数字
========================================================================
框架占位（XX）：
  R4.2（S2 范式保持重排）：走廊中位损失%、其他海域中位%；走廊 5–95% 区间、
  其他 5–95%；走廊贡献 S2 总可避免电量%。
  R4.3（S3 任意重构）：走廊中位、其他中位；损失>5% 项目占比（走廊/其他）、
  损失>10% 占比；走廊贡献总可避免电量 GWh 及 %。
数据源：output/wp7c_scenario_table.csv（场级 S1/S2/S3 增益与电量，corridor 标记）。
输出：output/wp9d_r4_corridor_losses.txt
"""
import io, sys, os
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
REPO = 'd:/01学习资料/wind-direction-to-electricity-transition'
TBL = REPO + '/结论三、四重算结果/output/wp7c_scenario_table.csv'
OUT = REPO + '/补算/output/wp9d_r4_corridor_losses.txt'

df = pd.read_csv(TBL, encoding='utf-8-sig')
print(f'行数 {len(df)}；corridor 取值:', sorted(df.corridor.unique()))
print(f'n_turb 分布: min {df.n_turb.min()} / 各分位 {np.percentile(df.n_turb, [10, 25, 50])}')
print(f'容量合计 {df.capacity_MW.sum()/1000:.1f} GW')

# ── 与既有口径校验（全文用 155 项目 n≥10？先看全表） ──
for s in ('S1', 'S2', 'S3'):
    tot = df[f'dE_GWh_{s}'].sum()
    med = df[f'G_plan_{s}'].median()
    print(f'全表 {s}: 合计 {tot:,.0f} GWh/yr  中位 {med:+.2f}%')
# 记忆口径: 3,474 / 15,889 / 75,966；中位 +0.47/+1.60/+10.79

# ── n≥10 子集（若与记忆口径一致则采用） ──
sub = df[df.n_turb >= 10]
for s in ('S1', 'S2', 'S3'):
    tot = sub[f'dE_GWh_{s}'].sum()
    med = sub[f'G_plan_{s}'].median()
    print(f'n≥10 子集({len(sub)}项目, {sub.capacity_MW.sum()/1000:.1f} GW) {s}: '
          f'合计 {tot:,.0f} GWh/yr  中位 {med:+.2f}%')

# ═══════════════════════════════════════════════════════════════════════
# 走廊 vs 非走廊 统计
# ═══════════════════════════════════════════════════════════════════════
lines = []
def w(t):
    lines.append(t)

cor = sub[sub.corridor != 'other']
oth = sub[sub.corridor == 'other']
w(f'走廊成员 {len(cor)} 项目, {cor.capacity_MW.sum()/1000:.2f} GW；'
  f'非走廊 {len(oth)} 项目, {oth.capacity_MW.sum()/1000:.2f} GW')
w(f'走廊装机占比 {cor.capacity_MW.sum()/sub.capacity_MW.sum()*100:.1f}%')

for s in ('S1', 'S2', 'S3'):
    cmed = cor[f'G_plan_{s}'].median()
    omed = oth[f'G_plan_{s}'].median()
    c05, c95 = np.percentile(cor[f'G_plan_{s}'], 5), np.percentile(cor[f'G_plan_{s}'], 95)
    o05, o95 = np.percentile(oth[f'G_plan_{s}'], 5), np.percentile(oth[f'G_plan_{s}'], 95)
    cE = cor[f'dE_GWh_{s}'].sum()
    oE = oth[f'dE_GWh_{s}'].sum()
    totE = sub[f'dE_GWh_{s}'].sum()
    w(f'\n{s}: 走廊中位 {cmed:+.2f}%  其他 {omed:+.2f}%  |  走廊 5–95% [{c05:+.2f}, {c95:+.2f}]  '
      f'其他 [{o05:+.2f}, {o95:+.2f}]')
    w(f'{s}: 走廊电量 {cE:,.0f} GWh/yr, 其他 {oE:,.0f}, 合计 {totE:,.0f}; '
      f'走廊占比 {cE/totE*100:.1f}%; 单位装机 走廊 {cE/(cor.capacity_MW.sum()/1000):,.0f} '
      f'vs 其他 {oE/(oth.capacity_MW.sum()/1000):,.0f} GWh/GW/yr')

# ── R4.3 损失>5% / >10% 项目占比 ──
for th in (5, 10):
    cp = (cor[f'G_plan_S3'] > th).mean() * 100
    op = (oth[f'G_plan_S3'] > th).mean() * 100
    w(f'\nS3 损失>{th}% 项目占比: 走廊 {cp:.1f}% ({int((cor.G_plan_S3>th).sum())}/{len(cor)})  '
      f'其他 {op:.1f}% ({int((oth.G_plan_S3>th).sum())}/{len(oth)})')

# ── R4.1 交叉校验（框架已知数字：2.77/0.40、88/18、7.6%/28.6%/995） ──
s = 'S1'
cmed = cor[f'G_plan_{s}'].median(); omed = oth[f'G_plan_{s}'].median()
cE = cor[f'dE_GWh_{s}'].sum(); totE = sub[f'dE_GWh_{s}'].sum()
w(f'\nR4.1 交叉校验: 走廊中位 {cmed:+.2f}% vs 其他 {omed:+.2f}%; '
  f'走廊 {cE:,.0f}/{totE:,.0f} GWh = {cE/totE*100:.1f}%; '
  f'单位装机 走廊 {cE/(cor.capacity_MW.sum()/1000):,.0f} vs 其他 '
  f'{oth.dE_GWh_S1.sum()/(oth.capacity_MW.sum()/1000):,.0f} GWh/GW/yr')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('\n报告已写入', OUT)
