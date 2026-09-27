from pathlib import Path
import json
import os
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2]))
HERE = ROOT / 'analysis_data'
from scipy.stats import spearmanr

OLD = ROOT / 'server/outputs'
DATA = HERE
DATA.mkdir(exist_ok=True)
with np.load(OLD / 'weather_joint/wrapped_all_hours.npz') as z:
    ids, probability, years = z['farm_ids'], z['p_fy'], z['years']
geometry = pd.read_csv(OLD / 'geometry_audit.csv').set_index('farm_id')
groups = pd.read_csv(ROOT / 'server/inputs/legacy_scenarios.csv').set_index('farm_id')
old_selection = pd.read_csv(ROOT / 'provenance/previous_selection_metrics.csv').query("scenario == 'baseline' and climate == 'wrapped_all_hours'").set_index('farm_id')
scenarios = ['baseline', 'nrel5_fixed_coordinates', 'iea15_fixed_coordinates',
             'gauss_ti_minus002', 'gauss_ti_plus002', 'jensen']


def energy(power, prob):
    result = np.fft.ifft(np.conj(np.fft.fft(prob, axis=-1)) * np.fft.fft(power, axis=-1)[None, :, :], axis=-1).real.sum(axis=1)*8760/1e9
    # 核对卷积符号，正角度对应来风方向索引增加。
    for angle in [0, 7, 35, 71]:
        direct = np.sum(prob * np.roll(power, -angle, axis=-1)[None, :, :], axis=(1, 2))*8760/1e9
        np.testing.assert_allclose(result[:, angle], direct, rtol=1e-12, atol=1e-10)
    return result


selected, site_rows, year_rows, curves = {}, [], [], []
for scenario in scenarios:
    for fi, fid in enumerate(ids):
        if scenario == 'jensen' and geometry.loc[fid, 'n_turb'] < 10:
            continue
        with np.load(OLD / f'models/{scenario}/F{fid:03d}_5deg.npz') as z:
            e = energy(z['power_W'], probability[fi])
            flag = z['negative_rotor_points'] > 0
            min_velocity = float(z['minimum_rotor_velocity_ms'].min())
        if scenario == 'baseline':
            selected[int(fid)] = int(e[:6].mean(axis=0).argmax())
            curves.append(e)
        angle = selected[int(fid)]
        pool = e.mean(axis=0)
        a_half = 100*(pool[:36].max()/pool[:36].mean()-1)
        a_full = 100*(pool.max()/pool.mean()-1)
        gain = 100*(e[6:, angle]/e[6:, 0]-1)
        de = e[6:, angle]-e[6:, 0]
        pooled_angle = int(pool.argmax())
        flag_base = np.sum(probability[fi, 6:].mean(axis=0)*flag)
        flag_selected = np.sum(probability[fi, 6:].mean(axis=0)*np.roll(flag, -angle, axis=1))
        row = {'farm_id': int(fid), 'scenario': scenario, 'n_turb': int(geometry.loc[fid, 'n_turb']),
               'selected_angle_deg': angle*5, 'previous_angle_deg': int(old_selection.loc[fid, 'selected_angle_deg']),
               'selection_source': '2014-2019 corrected all-hours baseline; shared by all models',
               'A_half_pct': float(a_half), 'A_full_pct': float(a_full), 'S1_gain_pct': float(gain.mean()),
               'baseline_GWh': float(e[6:, 0].mean()), 'delta_GWh': float(de.mean()),
               'energy_ratio_gain_pct': float(100*de.sum()/e[6:, 0].sum()),
               'nn_median_m': float(geometry.loc[fid, 'nn_median_m']),
               'min_distance_m': float(geometry.loc[fid, 'min_distance_m']),
               'negative_rotor_free': not bool(flag.any()), 'minimum_rotor_velocity_ms': min_velocity,
               'original_flagged_weather_pct': float(100*flag_base),
               'selected_flagged_weather_pct': float(100*flag_selected),
               'coastal_group': bool(groups.loc[fid, 'corridor'] != 'other'),
               'pooled_gain_pct': float((100*(e[:, pooled_angle]/e[:, 0]-1)).mean()),
               'pooled_angle_deg': pooled_angle*5, 'pooled_baseline_GWh': float(pool[0]),
               'pooled_delta_GWh': float(pool[pooled_angle]-pool[0])}
        site_rows.append(row)
        if row['n_turb'] >= 10:
            for yi in range(6, 11):
                year_rows.append({'farm_id': int(fid), 'scenario': scenario, 'year': int(years[yi]),
                                  'coastal_group': row['coastal_group'], 'n_turb': row['n_turb'],
                                  'baseline_GWh': float(e[yi, 0]), 'selected_GWh': float(e[yi, angle]),
                                  'delta_GWh': float(e[yi, angle]-e[yi, 0]), 'gain_pct': float(gain[yi-6]),
                                  'negative_rotor_free': row['negative_rotor_free']})
sites, year_data = pd.DataFrame(site_rows), pd.DataFrame(year_rows)
base = sites[sites.scenario == 'baseline'].copy()
projects = base[base.n_turb >= 10].copy()
np.savez_compressed(DATA / 'baseline_current_curves.npz', farm_ids=ids, E_year_GWh=np.array(curves), years=years)
sites.to_csv(DATA / 'all_site_metrics.csv', index=False)
base.to_csv(DATA / 'baseline_sites.csv', index=False)
projects.to_csv(DATA / 'planning_projects.csv', index=False)
year_data.to_csv(DATA / 'all_model_years.csv', index=False)
year_data[year_data.scenario == 'baseline'].to_csv(DATA / 'planning_years.csv', index=False)
pd.DataFrame([{'farm_id': k, 'angle_deg': v*5} for k, v in selected.items()]).to_csv(DATA / 'frozen_angles.csv', index=False)

draws = np.random.default_rng(20260919).integers(0, 5, size=(10000, 5))
summary, group_years = [], []
for scenario in scenarios:
    s = sites.query('scenario == @scenario and n_turb >= 10')
    y = year_data.query('scenario == @scenario')
    for screen in ['all', 'unflagged']:
        ss = s if screen == 'all' else s[s.negative_rotor_free]
        yy = y[y.farm_id.isin(ss.farm_id)]
        for label, mask in [('all155', np.ones(len(ss), bool)), ('coastal20', ss.coastal_group), ('other135', ~ss.coastal_group)]:
            sp = ss.loc[mask]
            gy = yy[yy.farm_id.isin(sp.farm_id)].groupby('year')[['baseline_GWh', 'delta_GWh']].sum().reindex(range(2020, 2025))
            boot = 100*gy.delta_GWh.to_numpy()[draws].sum(axis=1)/gy.baseline_GWh.to_numpy()[draws].sum(axis=1)
            net = float(sp.delta_GWh.sum())
            summary.append({'scenario': scenario, 'screen': screen, 'group': label, 'n_projects': len(sp),
                'project_median_pct': float(sp.S1_gain_pct.median()),
                'weighted_gain_pct': float(100*net/sp.baseline_GWh.sum()),
                'annual_delta_GWh': net, 'annual_baseline_GWh': float(sp.baseline_GWh.sum()),
                'positive': int((sp.delta_GWh*5 > 1e-7).sum()), 'zero': int((sp.delta_GWh.abs()*5 <= 1e-7).sum()),
                'negative': int((sp.delta_GWh*5 < -1e-7).sum()),
                'weather_year_bootstrap_lower_pct': float(np.quantile(boot, .025)),
                'weather_year_bootstrap_upper_pct': float(np.quantile(boot, .975)),
                'reference_capacity_GW': float(sp.n_turb.sum()*.01),
                'coastal_net_delta_share_pct': float(100*ss.loc[ss.coastal_group, 'delta_GWh'].sum()/ss.delta_GWh.sum()),
                'net_added_energy_share_pct': float(100*net/ss.delta_GWh.sum())})
            if scenario == 'baseline' and screen == 'all':
                group_years.extend([{'group': label, 'year': int(yr), 'gain_pct': float(100*r.delta_GWh/r.baseline_GWh)} for yr, r in gy.iterrows()])
summary = pd.DataFrame(summary)
summary.to_csv(DATA / 'model_summary_screened.csv', index=False)
summary[summary.screen == 'all'].to_csv(DATA / 'model_summary.csv', index=False)
pd.DataFrame(group_years).to_csv(DATA / 'planning_group_years.csv', index=False)
clean = base[base.negative_rotor_free]
high = projects.A_half_pct > 5.2
findings = {'changed_angles': int((base.selected_angle_deg != base.previous_angle_deg).sum()),
    'A_spacing_rho': float(spearmanr(base.nn_median_m, base.A_half_pct).statistic),
    'A_spacing_unflagged_rho': float(spearmanr(clean.nn_median_m, clean.A_half_pct).statistic),
    'unflagged_sites': len(clean), 'A_S1_rho': float(spearmanr(projects.A_half_pct, projects.S1_gain_pct).statistic),
    'high_A_projects': int(high.sum()), 'high_A_under_one_pct': int((high & (projects.S1_gain_pct < 1)).sum()),
    'planning': summary.query("scenario == 'baseline'").to_dict('records'),
    'source_climate_sha256': hashlib.sha256((OLD / 'weather_joint/wrapped_all_hours.npz').read_bytes()).hexdigest()}
(HERE / 'planning_evidence.json').write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(findings, ensure_ascii=False, indent=2))
