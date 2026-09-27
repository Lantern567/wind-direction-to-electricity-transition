import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t

HERE = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2])) / 'server/final_consistency'
BASE = HERE.parent
OUT = HERE / 'outputs'
PIDS = ['S_A', 'S_B0', 'S_B45', 'S_C', 'S_D', 'S_E']
UNIQUE = [0, 1, 3, 4, 5]


def curve(p, power):
    return np.fft.ifft(np.conj(np.fft.fft(p, axis=-1))*np.fft.fft(power, axis=-1), axis=-1).real.sum(axis=-2)*8760/1e9


def amplitude(e, first=0):
    window = np.take(e, (np.arange(36)+first) % 72, axis=-1)
    return 100*(window.max(axis=-1)/window.mean(axis=-1)-1)


def standard():
    with np.load(OUT / 'weather_evaluation.npz') as z:
        ids, p = z['farm_ids'], z['p_fy']
    geo = pd.read_csv(BASE / 'outputs/geometry_audit.csv').set_index('farm_id')
    groups = pd.read_csv(BASE / 'inputs/legacy_scenarios.csv').set_index('farm_id').loc[ids]
    p0 = np.r_[0, np.load(BASE / 'inputs/legacy_eta.npz')['P0']]
    pool = p.mean(axis=1)
    pref = pool.sum(axis=-1).mean(axis=0)
    phi = np.deg2rad(np.arange(0, 360, 5))
    theta_energy = np.rad2deg(np.angle((pool*p0[None, :, None]).sum(axis=1) @ np.exp(1j*phi))) % 360
    tb = np.zeros((len(ids), 6), dtype=int)
    tb[:, [0, 3, 5]] = (np.round((-theta_energy % 180)/5).astype(int) % 36)[:, None]
    tb[:, 2] = 9
    variant_rows, region_rows, planning_rows = [], [], []
    for variant in ['baseline', 'jensen', 'nrel5', 'iea15']:
        ep = np.empty((len(ids), 6, 72))
        ey = np.empty((len(ids), 11, 6, 72))
        a = np.empty((len(ids), 6))
        ac = np.empty((len(ids), 6, 4))
        full = np.empty_like(a)
        for fi, fid in enumerate(ids):
            pu, pdirection = pool[fi].sum(axis=1), pool[fi].sum(axis=0)
            cf = np.array([pool[fi], pu[:, None]*pdirection[None, :],
                           pref[:, None]*pdirection[None, :], pu[:, None]*np.ones((1, 72))/72])
            np.testing.assert_allclose(cf.sum(axis=(1, 2)), 1., atol=1e-12)
            for pi in range(6):
                path = OUT / 'standard_models' / f'{variant}_P{pi}_TI{geo.loc[fid, "ti"]:.2f}.npz'
                with np.load(path) as z:
                    power = np.vstack([np.zeros((1, 72)), z['power_W']])
                ey[fi, :, pi] = curve(p[fi], power)
                ep[fi, pi] = ey[fi, :, pi].mean(axis=0)
                c = curve(cf, power)
                np.testing.assert_allclose(c[0], ep[fi, pi], atol=1e-10)
                a[fi, pi] = amplitude(ep[fi, pi], tb[fi, pi])
                ac[fi, pi] = amplitude(c, tb[fi, pi])
                full[fi, pi] = 100*(ep[fi, pi].max()/ep[fi, pi].mean()-1)
        selected = groups.corridor.ne('other').to_numpy()
        ratios = np.median(a[selected][:, UNIQUE], axis=0)/np.median(a[~selected][:, UNIQUE], axis=0)
        ranks = pd.DataFrame(a[:, UNIQUE]).rank().corr().to_numpy()
        variant_rows.append({'variant': variant, 'ratio_min': float(ratios.min()), 'ratio_max': float(ratios.max()),
                             'rank_correlation_median': float(np.median(ranks[np.triu_indices(5, 1)])),
                             'B0_B45_full_A_max_difference_pp': float(np.abs(full[:, 1]-full[:, 2]).max())})
        for region, mask in [('Southern Vietnam', groups.corridor.eq('Vietnam').to_numpy() & (ids != 56)),
                              ('China coast', groups.corridor.eq('China_strait').to_numpy()),
                              ('Comparison group', selected), ('Other sites', ~selected)]:
            region_rows.append({'variant': variant, 'region': region, 'n': int(mask.sum()),
                                'median_five_A': float(np.median(a[mask][:, UNIQUE].mean(axis=1))),
                                'median_five_full_A': float(np.median(full[mask][:, UNIQUE].mean(axis=1))),
                                'all_five_top_quartile_n': int(((a[:, UNIQUE] >= np.quantile(a[:, UNIQUE], .75, axis=0)).all(axis=1) & mask).sum())})
        np.savez_compressed(OUT / f'standard_{variant}.npz', farm_ids=ids, E_pool_GWh=ep,
                            E_fy_GWh=ey, A=a, A_c=ac, A_full=full, TB=tb, theta_energy_deg=theta_energy)
        if variant == 'baseline':
            cmeans = ac[:, UNIQUE].mean(axis=(0, 1))
            pd.DataFrame([{'layout': PIDS[pi], 'C0_mean': ac[:, pi, 0].mean(), 'C1_mean': ac[:, pi, 1].mean(),
                           'delta_median_pp': np.median(ac[:, pi, 1]-ac[:, pi, 0]),
                           'delta_q25_pp': np.quantile(ac[:, pi, 1]-ac[:, pi, 0], .25),
                           'delta_q75_pp': np.quantile(ac[:, pi, 1]-ac[:, pi, 0], .75),
                           'n_decreased': int((ac[:, pi, 1] < ac[:, pi, 0]).sum())} for pi in UNIQUE]).to_csv(OUT / 'climate_by_geometry.csv', index=False)
            findings = {'counterfactual_means': cmeans.tolist(),
                        'dependency_reduction_pct': float(100*(1-cmeans[1]/cmeans[0])),
                        'n_sites_decreased': int((ac[:, UNIQUE, 1].mean(axis=1) < ac[:, UNIQUE, 0].mean(axis=1)).sum()),
                        'C3_max_A': float(np.abs(ac[:, :, 3]).max()),
                        'standard_rank_matrix': ranks.tolist(), 'group_ratios': ratios.tolist()}
            for fi, fid in enumerate(ids):
                if groups.loc[fid, 'n_turb'] < 10:
                    continue
                with np.load(BASE / f'outputs/models/baseline/F{fid:03d}_5deg.npz') as z:
                    actual = curve(p[fi, :, 1:, :], z['power_W'])
                scaled = ey[fi] * groups.loc[fid, 'n_turb']/64
                train = scaled[:6].mean(axis=0)
                # 旋转等价的B45可保留作辅助一致性比较，联合择优仅使用五种不同几何。
                matched = PIDS.index(groups.loc[fid, 'scenario_S2'])
                best_local = np.argmax(train[UNIQUE])
                best_pi, best_angle = UNIQUE[best_local // 72], best_local % 72
                for label, pi, angle in [('S2', matched, int(train[matched].argmax())), ('S3', best_pi, best_angle)]:
                    e0, e1 = actual[6:, 0], scaled[6:, pi, angle]
                    planning_rows.append({'farm_id': int(fid), 'scheme': label, 'layout': PIDS[pi], 'angle_deg': angle*5,
                                          'coastal_group': bool(selected[fi]), 'n_turb': int(groups.loc[fid, 'n_turb']),
                                          'baseline_GWh': float(e0.mean()), 'delta_GWh': float((e1-e0).mean()),
                                          'mean_gain_pct': float((100*(e1/e0-1)).mean())})
    pd.DataFrame(variant_rows).to_csv(OUT / 'standard_sensitivity.csv', index=False)
    pd.DataFrame(region_rows).to_csv(OUT / 'standard_regional.csv', index=False)
    pd.DataFrame(planning_rows).to_csv(OUT / 'auxiliary_planning.csv', index=False)
    findings['variants'] = variant_rows
    findings['regions'] = region_rows
    (OUT / 'standard_evidence.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2))
    print(json.dumps(findings, ensure_ascii=False), flush=True)


def historical():
    with np.load(OUT / 'weather_complete.npz') as z:
        ids, prob, ph, years = z['farm_ids'], z['p_fy'], z['p_history'], z['years']
    manifest = pd.read_csv(OUT / 'annual_geometry_manifest.csv').query('n_turb >= 2')
    manifest.to_csv(OUT/'historical_geometry_mapping.csv', index=False)
    rows, fixed_rows = [], []
    for fi, fid in enumerate(ids):
        with np.load(BASE / f'outputs/models/baseline/F{fid:03d}_5deg.npz') as z:
            terminal = z['power_W']
            nt = int(z['n_turb'])
        trained = curve(ph[fi, 1:], terminal)
        angle = int(trained[:36:2].argmax())*2
        efixed = curve(prob[fi, :, 1:], terminal)
        fixed_rows.append({'farm_id': int(fid), 'historical_angle_deg': angle*5, 'n_turb': nt,
                           'fixed_geometry_annual_delta_GWh': float((efixed[:, angle]-efixed[:, 0]).mean()),
                           'fixed_geometry_gain_pct': float((100*(efixed[:, angle]/efixed[:, 0]-1)).mean()),
                           'reference_capacity_GW': nt*.01})
        for r in manifest[manifest.farm_id == fid].itertuples():
            if r.terminal_equivalent:
                power = terminal
            else:
                with np.load(OUT / 'annual_models' / f'{r.key}.npz') as z:
                    power = z['power_W']
            e = curve(prob[fi, r.year-2014, 1:], power)
            rows.append({'farm_id': int(fid), 'year': int(r.year), 'n_turb': int(r.n_turb),
                         'angle_deg': angle*5, 'baseline_GWh': float(e[0]), 'selected_GWh': float(e[angle]),
                         'gain_pct': float(100*(e[angle]/e[0]-1))})
    annual = pd.DataFrame(rows)
    assert len(annual) == 1203 and not annual.duplicated(['farm_id', 'year']).any()
    annual.to_csv(OUT / 'historical_annual.csv', index=False)
    fixed = pd.DataFrame(fixed_rows).set_index('farm_id')
    farm = annual.groupby('farm_id').agg(G=('gain_pct', 'mean'), years=('year', 'count'),
                                       positive_years=('gain_pct', lambda x: int((x > 1e-8).sum())),
                                       min_gain=('gain_pct', 'min'), max_gain=('gain_pct', 'max'))
    farm = farm.join(fixed, validate='one_to_one')
    rng = np.random.default_rng(20260920)
    intervals = []
    for fid, g in annual.groupby('farm_id'):
        values = g.gain_pct.to_numpy()
        if len(values) < 5:
            intervals.append({'farm_id': int(fid), 'G_ci_low': None, 'G_ci_high': None})
        else:
            boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
            intervals.append({'farm_id': int(fid), 'G_ci_low': float(np.quantile(boot, .025)), 'G_ci_high': float(np.quantile(boot, .975))})
    farm = farm.join(pd.DataFrame(intervals).set_index('farm_id'))
    farm.to_csv(OUT / 'historical_sites.csv')
    top5 = farm.G.nlargest(5).index
    comparisons = []
    for first in [2014, 2016, 2017]:
        part = annual[annual.year >= first]
        gs = part.groupby('farm_id').gain_pct.mean()
        comparisons.append({'first_year': first, 'records': len(part), 'sites': len(gs),
                             'mean_pct': float(gs.mean()), 'median_pct': float(gs.median()),
                             'rank_rho': float(spearmanr(gs, farm.loc[gs.index, 'G']).statistic),
                             'top5': list(map(int, gs.nlargest(5).index))})
    positive_energy = fixed.fixed_geometry_annual_delta_GWh.clip(lower=0).sum()
    findings = {'sites': len(farm), 'records': len(annual), 'mean_pct': float(farm.G.mean()),
                'median_pct': float(farm.G.median()), 'positive_years_pct': float(100*(annual.gain_pct > 1e-8).mean()),
                'top5': list(map(int, top5)), 'without_top5_mean_pct': float(farm.drop(top5).G.mean()),
                'top5_mean_reduction_pct': float(100*(1-farm.drop(top5).G.mean()/farm.G.mean())),
                'top5_capacity_pct': float(100*fixed.loc[top5, 'n_turb'].sum()/fixed.n_turb.sum()),
                'top5_positive_energy_pct': float(100*fixed.loc[top5, 'fixed_geometry_annual_delta_GWh'].clip(lower=0).sum()/positive_energy),
                'negative_to_positive_energy_pct': float(-100*fixed.fixed_geometry_annual_delta_GWh.clip(upper=0).sum()/positive_energy),
                'year_exclusions': comparisons,
                'cases': farm.loc[top5].reset_index().astype(object).where(pd.notna(farm.loc[top5].reset_index()), None).to_dict('records')}
    (OUT / 'historical_evidence.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(findings, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['standard', 'historical'])
    options = parser.parse_args()
    {'standard': standard, 'historical': historical}[options.mode]()
