from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2]))
HERE = ROOT
from scipy.stats import spearmanr, t

sites = pd.read_csv(HERE/'analysis_data/baseline_sites.csv').set_index('farm_id').sort_index()
wind = pd.read_csv(HERE/'server/final_consistency/outputs/wind_statistics.csv').set_index('farm_id').loc[sites.index]
meta = pd.read_csv(ROOT/'analysis_data/farm_metadata.csv').set_index('farm_id').loc[sites.index]
frame = sites.join(wind, validate='one_to_one').join(meta[['country', 'cent_lon', 'cent_lat', 'corr']], validate='one_to_one')
frame['spacing_D'] = frame.nn_median_m/198
frame['high_A'] = frame.A_half_pct > 5.2
with np.load(HERE/'analysis_data/baseline_current_curves.npz') as z:
    np.testing.assert_array_equal(z['farm_ids'], sites.index)
    ey = z['E_year_GWh']
ep = ey.mean(axis=1)
theta = ep[:, :36].argmax(axis=1)*5
frame['phase_distance_deg'] = np.minimum(theta % 180, 180-theta % 180)
frame['pooled_half_peak_deg'] = theta
frame['Gdiag_pct'] = 100*((ey[np.arange(len(frame)), :, theta//5]-ey[:, :, 0])/ey[:, :, 0]).mean(axis=1)
frame['Gdiag_over_A_pct'] = 100*frame.Gdiag_pct/frame.A_half_pct
features = ['R1_frequency', 'R2_frequency', 'R1_v3', 'R2_v3']
correlations = []
for feature in features:
    rho, pvalue = spearmanr(frame[feature], frame.A_half_pct)
    rank = frame[['A_half_pct', 'spacing_D', feature]].rank()
    x = np.column_stack([np.ones(len(frame)), rank.spacing_D])
    a = rank.A_half_pct-x@np.linalg.lstsq(x, rank.A_half_pct, rcond=None)[0]
    b = rank[feature]-x@np.linalg.lstsq(x, rank[feature], rcond=None)[0]
    partial = float(np.corrcoef(a, b)[0, 1])
    pv = float(2*t.sf(abs(partial)*np.sqrt((len(frame)-3)/(1-partial**2)), len(frame)-3))
    correlations.append({'feature': feature, 'spearman': float(rho), 'pvalue': float(pvalue),
                          'partial_spearman': partial, 'partial_pvalue': pv})
pd.DataFrame(correlations).to_csv(HERE/'analysis_data/concentration_correlations.csv', index=False)
threshold_rows = []
for threshold in [3, 3.5, 4, 4.5, 5]:
    dense = frame.spacing_D < threshold
    for label, mask in [('dense', dense), ('other', ~dense)]:
        threshold_rows.append({'threshold_D': threshold, 'group': label, 'n': int(mask.sum()),
                               'median_A': float(frame.loc[mask, 'A_half_pct'].median()),
                               'high_A_count': int(frame.loc[mask, 'high_A'].sum()),
                               'high_A_pct': float(100*frame.loc[mask, 'high_A'].mean())})
pd.DataFrame(threshold_rows).to_csv(HERE/'analysis_data/spacing_thresholds.csv', index=False)
country_rows = []
for country in sorted(frame.country.unique()):
    subset = frame[frame.country != country]
    country_rows.append({'excluded_country': country, 'n': len(subset),
                         'spacing_A_rho': float(spearmanr(subset.spacing_D, subset.A_half_pct).statistic)})
pd.DataFrame(country_rows).to_csv(HERE/'analysis_data/leave_country_out.csv', index=False)
block_ids = list(zip(np.floor(frame.cent_lon/5), np.floor(frame.cent_lat/5)))
block_members = {}
for i, block in enumerate(block_ids):
    block_members.setdefault(block, []).append(i)
blocks = list(block_members.values())
rng = np.random.default_rng(20260920)
rho_draws = []
for _ in range(2000):
    index = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), size=len(blocks))])
    rho_draws.append(float(spearmanr(frame.spacing_D.iloc[index], frame.A_half_pct.iloc[index]).statistic))
np.savez_compressed(HERE/'analysis_data/spacing_block_bootstrap.npz', rho=np.array(rho_draws), block_size_degrees=5)
dense = frame.spacing_D < 4
median_r2 = float(frame.loc[dense, 'R2_v3'].median())
high_concentration = dense & (frame.R2_v3 >= median_r2)
low_concentration = dense & (frame.R2_v3 < median_r2)
frame.to_csv(HERE/'analysis_data/mechanism_sites.csv')
findings = {'spacing_rho': float(spearmanr(frame.spacing_D, frame.A_half_pct).statistic),
            'n_high_A': int(frame.high_A.sum()), 'n_dense': int(dense.sum()),
            'all_high_A_dense': bool((frame.loc[frame.high_A, 'spacing_D'] < 4).all()),
            'country_exclusion_rho_min': min(r['spacing_A_rho'] for r in country_rows),
            'country_exclusion_rho_max': max(r['spacing_A_rho'] for r in country_rows),
            'spatial_blocks': len(blocks), 'block_rho_ci': np.quantile(rho_draws, [.025, .975]).tolist(),
            'dense_R2_high_n': int(high_concentration.sum()), 'dense_R2_low_n': int(low_concentration.sum()),
            'dense_R2_high_highA_pct': float(100*frame.loc[high_concentration, 'high_A'].mean()),
            'dense_R2_low_highA_pct': float(100*frame.loc[low_concentration, 'high_A'].mean()),
            'dense_R2_threshold': median_r2,
            'phase_rho': float(spearmanr(frame.loc[frame.high_A, 'phase_distance_deg'], frame.loc[frame.high_A, 'Gdiag_over_A_pct']).statistic),
            'correlations': correlations}
(HERE/'analysis_data/mechanism_evidence.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(findings, ensure_ascii=False, indent=2))
