from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

from plot_config import DATA_ROOT, OUTPUT_ROOT, save_600dpi
HERE = OUTPUT_ROOT
ROOT = DATA_ROOT
DATA = ROOT / 'analysis_data'
from figure_common import style, axis, plt, Line2D, BLUE, ORANGE, TEAL, LIGHT, GREY, INK, GRID

p = pd.read_csv(DATA / 'planning_projects.csv').set_index('farm_id')
summary = pd.read_csv(DATA / 'model_summary.csv')
base = summary.query("scenario == 'baseline'").set_index('group')
annual = pd.read_csv(DATA / 'planning_group_years.csv')
assert p.index.is_unique and len(p) == 155
assert p.coastal_group.sum() == 20
assert np.isfinite(p[['S1_gain_pct', 'delta_GWh', 'baseline_GWh', 'pooled_gain_pct']]).all().all()
assert np.isclose(p.delta_GWh.sum(), base.loc['all155', 'annual_delta_GWh'])
assert np.isclose(100 * p.delta_GWh.sum() / p.baseline_GWh.sum(), base.loc['all155', 'weighted_gain_pct'])
p['capacity_GW'] = p.n_turb * 10 / 1000
p['unit_added'] = p.delta_GWh / p.capacity_GW
sel = p.coastal_group.to_numpy(bool)
groups = ['all155', 'coastal20', 'other135']
colors = [BLUE, ORANGE, LIGHT]
labels = ['All projects  (n = 155)', 'Coastal group  (n = 20)', 'Other projects  (n = 135)']
masks = [np.ones(len(p), bool), sel, ~sel]
style()
fig = plt.figure(figsize=(7.9, 9.62))
rng = np.random.default_rng(20260920)

# 沿用早期版本的横向分布、右侧汇总值和底部跨情景区间。
a = axis(fig, [.205, .755, .565, .204], grid=False)
fig.text(.028, .974, 'a', fontsize=12, fontweight='bold')
for j, (g, mask, color) in enumerate(zip(groups, masks, colors)):
    v = p.loc[mask, 'S1_gain_pct'].to_numpy()
    a.scatter(v, j + rng.uniform(-.12, .12, len(v)), s=12, c=color, alpha=.55, lw=0, zorder=2)
    q25, med, q75 = np.quantile(v, [.25, .5, .75])
    a.plot([q25, q75], [j, j], color=color, lw=4, zorder=3)
    a.scatter(med, j, s=38, marker='D', c=color, edgecolors='white', lw=.6, zorder=4)
a.axvline(0, c=GREY, ls='--', lw=.8)
a.set(xlim=(-1.3, 17.5), ylim=(2.65, -.65), xlabel='Project generation gain in 2020–2024 (%)')
a.set_xticks([0, 5, 10, 15])
a.set_yticks(range(3), ['All projects\n(n = 155)', 'Coastal group\n(n = 20)', 'Other projects\n(n = 135)'])
a.tick_params(axis='y', length=0, labelsize=8, pad=10)
a.spines['left'].set_visible(False)
at = fig.add_axes([.811, .755, .16, .204])
at.set(xlim=(0, 1), ylim=(2.65, -.65))
at.axis('off')
at.text(.5, -.5, 'Total generation\nincrease (%)', ha='center', va='bottom', fontsize=8)
for j, g in enumerate(groups):
    r = base.loc[g]
    at.text(.5, j, f'{r.weighted_gain_pct:+.2f}', ha='center', va='center', fontsize=11, color=BLUE,
            bbox=dict(boxstyle='round,pad=.43,rounding_size=.15', fc='#EAF0F3', ec='none'))
    at.text(.5, j+.34, f'[{r.weather_year_bootstrap_lower_pct:.2f}, {r.weather_year_bootstrap_upper_pct:.2f}]',
            fontsize=7, color=GREY, ha='center', va='center')
fig.legend(handles=[Line2D([], [], marker='o', ms=5, mfc=ORANGE, mec='white', lw=0, label='Coastal group (n = 20)'),
                    Line2D([], [], marker='o', ms=5, mfc=LIGHT, mec='white', lw=0, label='Other projects (n = 135)')],
           loc='center', bbox_to_anchor=(.54, .700), ncol=2, fontsize=8, handletextpad=.2, columnspacing=1.3)

b = axis(fig, [.130, .488, .325, .165], grid=False)
fig.text(.028, .663, 'b', fontsize=12, fontweight='bold')
unit_totals = {}
for j, (mask, color, key) in enumerate([(sel, ORANGE, 'coastal20'), (~sel, LIGHT, 'other135')]):
    g = p.loc[mask]
    total = g.delta_GWh.sum() / g.capacity_GW.sum()
    unit_totals[key] = float(total)
    b.scatter(g.unit_added, j+rng.uniform(-.16, .16, len(g)), s=15, c=color, alpha=.65, lw=0)
    b.scatter(total, j, s=56, marker='D', c=ORANGE if j == 0 else BLUE, edgecolors='white', lw=.8, zorder=4)
    b.text(total+5, j-.32, f'{total:.1f}', color=ORANGE if j == 0 else BLUE, fontsize=8.6)
b.set_yticks([0, 1], ['Coastal\n20 projects', 'Other\n135 projects'])
b.tick_params(axis='y', length=0, labelsize=7.5)
b.spines['left'].set_visible(False)
b.axvline(0, c=GREY, lw=.6, ls='--')
b.set(xlim=(min(-5, p.unit_added.min()-5), p.unit_added.max()*1.08), ylim=(1.6, -.65),
      xlabel='Added energy per reference capacity\n(GWh GW$^{-1}$ yr$^{-1}$)')
b.set_xticks([0, 100, 200])
b.text(.98, .98, f'{unit_totals["coastal20"]/unit_totals["other135"]:.1f}×', transform=b.transAxes,
       color=ORANGE, fontsize=9, ha='right', va='top')

c = axis(fig, [.605, .488, .320, .165])
fig.text(.514, .663, 'c', fontsize=12, fontweight='bold')
ranked = p.sort_values('delta_GWh', ascending=False)
x = np.r_[0, ranked.capacity_GW.cumsum()/p.capacity_GW.sum()*100]
y = np.r_[0, ranked.delta_GWh.cumsum()/p.delta_GWh.sum()*100]
assert np.allclose([x[-1], y[-1]], [100, 100])
c.fill_between(x, x, y, color=LIGHT, alpha=.22)
c.plot(x, y, c=BLUE, lw=1.5)
c.plot([0, 100], [0, 100], c=GREY, ls='--', lw=.7)
cap_share = p.loc[sel, 'capacity_GW'].sum()/p.capacity_GW.sum()*100
energy_share = base.loc['coastal20', 'net_added_energy_share_pct']
c.text(.98, .065, f'20 coastal projects\n{cap_share:.1f}% of capacity\n{energy_share:.1f}% of added energy',
       transform=c.transAxes, ha='right', va='bottom', fontsize=7.7, color=ORANGE)
c.set(xlim=(0, 100), ylim=(0, max(103, y.max()+2)), xlabel='Cumulative reference capacity (%)',
      ylabel='Cumulative net added energy (%)')
c.set_xticks([0, 50, 100])
c.set_yticks([0, 50, 100])

d = axis(fig, [.130, .282, .325, .137])
fig.text(.028, .431, 'd', fontsize=12, fontweight='bold')
for group, color, label in zip(groups, [BLUE, ORANGE, TEAL], ['All projects', 'Coastal group', 'Other projects']):
    r = annual[annual.group.eq(group)].sort_values('year')
    assert len(r) == 5 and r.gain_pct.min() > 0
    d.plot(r.year, r.gain_pct, c=color, marker='o', ms=3.5, lw=1.3, label=label)
d.set(xlim=(2019.7, 2024.3), ylim=(0, 4.25), ylabel='Total generation gain (%)', xlabel='Evaluation year')
d.set_xticks(range(2020, 2025))
d.set_yticks([0, 2, 4])
d.tick_params(axis='x', labelsize=7)
d.legend(loc='center left', bbox_to_anchor=(0, .45), fontsize=6.7, labelspacing=.25, handlelength=1.6)

e = axis(fig, [.605, .282, .320, .137])
fig.text(.514, .431, 'e', fontsize=12, fontweight='bold')
for mask, color in [(~sel, LIGHT), (sel, ORANGE)]:
    e.scatter(p.loc[mask, 'pooled_gain_pct'], p.loc[mask, 'S1_gain_pct'], s=17, c=color, edgecolors='white', lw=.35, zorder=3)
e.plot([-1, 17], [-1, 17], c=GREY, ls='--', lw=.75)
e.axhline(0, c=GREY, lw=.4)
e.set(xlim=(-1, 17), ylim=(-1, 17), xlabel='Gain selected and evaluated\nin 2014–2024 (%)', ylabel='Gain in 2020–2024 (%)')
e.set_xticks([0, 5, 10, 15])
e.set_yticks([0, 5, 10, 15])

f = axis(fig, [.100, .055, .825, .133], 'f')
all_rows = summary.query("group == 'all155'").set_index('scenario')
# 由存档情景名称显式连接，缺失时立即报错。
actual_order = ['nrel5_fixed_coordinates', 'baseline', 'iea15_fixed_coordinates', 'gauss_ti_minus002', 'gauss_ti_plus002', 'jensen']
name_map = {'nrel5_fixed_coordinates': 'NREL 5 MW', 'baseline': 'IEA 10 MW\nBaseline', 'iea15_fixed_coordinates': 'IEA 15 MW',
            'gauss_ti_minus002': 'TI − 0.02', 'gauss_ti_plus002': 'TI + 0.02', 'jensen': 'Jensen'}
assert set(actual_order) == set(all_rows.index), list(all_rows.index)
for j, key in enumerate(actual_order):
    r = all_rows.loc[key]
    color = ORANGE if key == 'baseline' else BLUE
    f.errorbar(j, r.weighted_gain_pct,
               yerr=[[r.weighted_gain_pct-r.weather_year_bootstrap_lower_pct], [r.weather_year_bootstrap_upper_pct-r.weighted_gain_pct]],
               fmt='o', color=color, ms=5.5, mec='white', mew=.6, elinewidth=1.6, capsize=4)
    f.text(j, r.weather_year_bootstrap_upper_pct+.045, f'{r.weighted_gain_pct:+.2f}%', ha='center', fontsize=8, color=color)
f.set(xlim=(-.5, 5.5), ylim=(0, 1.10), ylabel='Total generation gain (%)')
f.set_xticks(range(6), [name_map[key] for key in actual_order])
f.set_yticks([0, .5, 1.0])
fig.text(.100, .211, 'Fixed historical orientations; 2020–2024 evaluation; 95% year-resampling intervals', fontsize=7.4, color=GREY)

save_600dpi(fig, 'Figure_4_600dpi')
fig.savefig(HERE / 'figure4_preview.png', dpi=180, facecolor='white')
plt.close(fig)
pd.DataFrame({'capacity_pct': x, 'net_added_energy_pct': y}).to_csv(HERE / 'figure4_cumulative.csv', index=False)
p[['n_turb', 'S1_gain_pct', 'delta_GWh', 'capacity_GW', 'unit_added', 'coastal_group']].to_csv(HERE / 'figure4_project_data.csv')
meta = {'projects': len(p), 'coastal_projects': int(sel.sum()), 'capacity_share_pct': float(cap_share),
        'coastal_net_added_share_pct': float(energy_share), 'unit_energy': unit_totals,
        'cumulative_peak_pct': float(y.max()), 'negative_projects_retained': int((p.delta_GWh < -1e-7).sum()),
        'style_reference': 'manuscript/2026-09-13_APEN_review/figure4_preview.png',
        'inputs': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['planning_projects.csv', 'model_summary.csv', 'planning_group_years.csv']}}
(HERE / 'figure4_validation.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False))
