from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import BytesIO, TextIOWrapper
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math

ORIGINAL_SHA256 = '0b55a3cf8b8c16084d747fd63828c5010dd4fa59160c4abb3cdc3ecb1157eebc'
DEFAULT_ARCHIVE = 'APEN_Supplementary_Data_v5.33_20260920.zip'
SCENARIOS = ['baseline', 'nrel5_fixed_coordinates', 'iea15_fixed_coordinates',
             'gauss_ti_minus002', 'gauss_ti_plus002', 'jensen']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(archive, name):
    with archive.open(name) as stream:
        return list(csv.DictReader(TextIOWrapper(stream, encoding='utf-8-sig')))


def boolean(value):
    require(value in ('True', 'False'), 'Invalid Boolean field')
    return value == 'True'


def verify_integrity(path, expected_sha256):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(digest == expected_sha256, 'Archive SHA-256 mismatch')
    with ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        require(len(names) == len(set(names)), 'Duplicate ZIP member names')
        for name in names:
            member = PurePosixPath(name)
            require(not member.is_absolute() and '..' not in member.parts and '\\' not in name, 'Unsafe ZIP member name')
        manifest = json.loads(archive.read('manifest.json'))
        listed = [record['path'] for record in manifest['files']]
        require(len(listed) == len(set(listed)), 'Duplicate manifest paths')
        require(set(names) - set(listed) == {'README.md', 'manifest.json'}, 'Unexpected unlisted files')
        for record in manifest['files']:
            content = archive.read(record['path'])
            require(len(content) == record['bytes'], 'Manifest size mismatch: ' + record['path'])
            require(hashlib.sha256(content).hexdigest() == record['sha256'], 'Manifest checksum mismatch: ' + record['path'])
        if digest == ORIGINAL_SHA256:
            require((len(names), len(listed)) == (2822, 2820), 'Unexpected original archive inventory')
        rows = read_csv(archive, 'analysis_data/all_site_metrics.csv')
        baseline = [r for r in rows if r['scenario'] == 'baseline']
        planning = [r for r in baseline if int(r['n_turb']) >= 10]
        clean = [r for r in planning if boolean(r['negative_rotor_free'])]
        coastal = [r for r in clean if boolean(r['coastal_group'])]
        delta = sum(float(r['delta_GWh']) for r in clean)
        energy = sum(float(r['baseline_GWh']) for r in clean)
        gain = 100 * delta / energy
        share = 100 * sum(float(r['delta_GWh']) for r in coastal) / delta
        require((len(baseline), len(planning), len(clean), len(coastal)) == (171, 155, 139, 11), 'Screened sample mismatch')
        require(math.isclose(gain, 0.4967388663098729, rel_tol=1e-10), 'Screened gain mismatch')
        require(math.isclose(share, 27.231113765544883, rel_tol=1e-10), 'Screened coastal share mismatch')
    return {'archive_name': path.name, 'archive_sha256': digest, 'files': len(names), 'verified_manifest_files': len(listed),
            'baseline_sites': len(baseline), 'baseline_flagged_sites': sum(not boolean(r['negative_rotor_free']) for r in baseline),
            'planning_sites': len(planning), 'screened_planning_sites': len(clean), 'screened_coastal_sites': len(coastal),
            'screened_gain_pct': gain, 'screened_annual_delta_GWh': delta, 'screened_coastal_share_pct': share}


def recompute_from_arrays(archive):
    import numpy as np

    # 全部计算直接读取科学数组；不导入或执行归档中的原始脚本。
    def load_npz(name):
        with np.load(BytesIO(archive.read(name)), allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}

    def close(actual, expected, label):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8, err_msg=label)

    def energy(power, probability):
        result = np.fft.ifft(np.conj(np.fft.fft(probability, axis=-1)) * np.fft.fft(power, axis=-1)[None, :, :], axis=-1).real.sum(axis=1) * 8760 / 1e9
        for angle in (0, 7, 35, 71):
            direct = (probability * np.roll(power, -angle, axis=-1)[None, :, :]).sum(axis=(1, 2)) * 8760 / 1e9
            close(result[:, angle], direct, 'FFT/direct energy agreement')
        return result

    weather = load_npz('server/outputs/weather_joint/wrapped_all_hours.npz')
    ids, years, probabilities = weather['farm_ids'], weather['years'], weather['p_fy']
    require(years.tolist() == list(range(2014, 2025)), 'Weather years mismatch')
    require(probabilities.shape == (171, 11, 18, 72), 'Weather shape mismatch')
    require(bool(np.isfinite(probabilities).all() and (probabilities >= 0).all()), 'Invalid probabilities')
    require(bool((probabilities.sum(axis=(2, 3)) <= 1 + 1e-12).all()), 'Probability denominator mismatch')
    expected_sites = {(r['scenario'], int(r['farm_id'])): r for r in read_csv(archive, 'analysis_data/all_site_metrics.csv')}
    expected_years = {(r['scenario'], int(r['farm_id']), int(r['year'])): r for r in read_csv(archive, 'analysis_data/all_model_years.csv')}
    groups = {int(r['farm_id']): r['corridor'] != 'other' for r in read_csv(archive, 'server/inputs/legacy_scenarios.csv')}
    selected, rebuilt_sites, rebuilt_years, baseline_curves = {}, [], [], []
    max_energy_error = 0.0
    for scenario in SCENARIOS:
        for index, farm_id in enumerate(ids.tolist()):
            if (scenario, farm_id) not in expected_sites:
                require(scenario == 'jensen', 'Missing site/scenario reference')
                continue
            reference = expected_sites[(scenario, farm_id)]
            path = f'server/outputs/models/{scenario}/F{farm_id:03d}_5deg.npz'
            arrays = load_npz(path)
            require(path.replace('.npz', '_config.json') in archive.namelist(), 'Missing response configuration')
            require(int(arrays['farm_id']) == farm_id and int(arrays['n_turb']) == int(reference['n_turb']), 'Site geometry identity mismatch')
            power = arrays['power_W']
            require(bool(np.isfinite(power).all() and (power >= 0).all()), 'Invalid farm power array')
            curves = energy(power, probabilities[index])
            if scenario == 'baseline':
                selected[farm_id] = int(curves[:6].mean(axis=0).argmax())
                baseline_curves.append(curves)
            angle = selected[farm_id]
            flags = arrays['negative_rotor_points'] > 0
            pool = curves.mean(axis=0)
            pool_angle = int(pool.argmax())
            annual_gains = 100 * (curves[6:, angle] / curves[6:, 0] - 1)
            delta = curves[6:, angle] - curves[6:, 0]
            actual = {
                'scenario': scenario, 'farm_id': farm_id, 'n_turb': int(arrays['n_turb']),
                'coastal_group': groups[farm_id], 'negative_rotor_free': not bool(flags.any()),
                'selected_angle_deg': angle * 5,
                'A_half_pct': float(100 * (pool[:36].max() / pool[:36].mean() - 1)),
                'A_full_pct': float(100 * (pool.max() / pool.mean() - 1)),
                'S1_gain_pct': float(annual_gains.mean()), 'baseline_GWh': float(curves[6:, 0].mean()),
                'delta_GWh': float(delta.mean()), 'energy_ratio_gain_pct': float(100 * delta.sum() / curves[6:, 0].sum()),
                'minimum_rotor_velocity_ms': float(arrays['minimum_rotor_velocity_ms'].min()),
                'original_flagged_weather_pct': float(100 * (probabilities[index, 6:].mean(axis=0) * flags).sum()),
                'selected_flagged_weather_pct': float(100 * (probabilities[index, 6:].mean(axis=0) * np.roll(flags, -angle, axis=1)).sum()),
                'pooled_angle_deg': pool_angle * 5,
                'pooled_gain_pct': float((100 * (curves[:, pool_angle] / curves[:, 0] - 1)).mean()),
                'pooled_baseline_GWh': float(pool[0]), 'pooled_delta_GWh': float(pool[pool_angle] - pool[0]),
            }
            for key, value in actual.items():
                if key == 'scenario':
                    require(value == reference[key], 'Scenario mismatch')
                elif isinstance(value, bool):
                    require(value == boolean(reference[key]), f'Flag/group mismatch: {path}, {key}')
                else:
                    close(value, float(reference[key]), f'{path}: {key}')
            max_energy_error = max(max_energy_error, abs(actual['baseline_GWh'] - float(reference['baseline_GWh'])), abs(actual['delta_GWh'] - float(reference['delta_GWh'])))
            rebuilt_sites.append(actual)
            if actual['n_turb'] >= 10:
                for yi in range(6, 11):
                    year = int(years[yi])
                    row = {'scenario': scenario, 'farm_id': farm_id, 'year': year,
                           'baseline_GWh': float(curves[yi, 0]), 'selected_GWh': float(curves[yi, angle]),
                           'delta_GWh': float(curves[yi, angle] - curves[yi, 0]), 'gain_pct': float(annual_gains[yi - 6])}
                    reference_year = expected_years[(scenario, farm_id, year)]
                    for key in ('baseline_GWh', 'selected_GWh', 'delta_GWh', 'gain_pct'):
                        close(row[key], float(reference_year[key]), f'{path}, {year}: {key}')
                    rebuilt_years.append(row)
    require(len(rebuilt_sites) == len(expected_sites) == 1010, 'Model/site count mismatch')
    require(len(rebuilt_years) == len(expected_years) == 4650, 'Planning site/year count mismatch')
    archived_curves = load_npz('analysis_data/baseline_current_curves.npz')
    require(archived_curves['farm_ids'].tolist() == ids.tolist() and archived_curves['years'].tolist() == years.tolist(), 'Baseline curve index mismatch')
    close(np.array(baseline_curves), archived_curves['E_year_GWh'], 'Baseline annual/orientation curve archive')
    for row in read_csv(archive, 'analysis_data/frozen_angles.csv'):
        require(selected[int(row['farm_id'])] * 5 == int(row['angle_deg']), 'Frozen angle mismatch')

    expected_summaries = read_csv(archive, 'analysis_data/model_summary_screened.csv')
    draws = np.random.default_rng(20260919).integers(0, 5, size=(10000, 5))
    summaries = []
    for reference in expected_summaries:
        scenario, screen, group = reference['scenario'], reference['screen'], reference['group']
        retained = [r for r in rebuilt_sites if r['scenario'] == scenario and r['n_turb'] >= 10 and (screen == 'all' or r['negative_rotor_free'])]
        subset = [r for r in retained if group == 'all155' or (group == 'coastal20' and r['coastal_group']) or (group == 'other135' and not r['coastal_group'])]
        subset_ids = {r['farm_id'] for r in subset}
        annual = [r for r in rebuilt_years if r['scenario'] == scenario and r['farm_id'] in subset_ids]
        by_year = np.array([[sum(r[key] for r in annual if r['year'] == year) for key in ('baseline_GWh', 'delta_GWh')] for year in range(2020, 2025)])
        bootstrap = 100 * by_year[:, 1][draws].sum(axis=1) / by_year[:, 0][draws].sum(axis=1)
        net, base = sum(r['delta_GWh'] for r in subset), sum(r['baseline_GWh'] for r in subset)
        total_delta = sum(r['delta_GWh'] for r in retained)
        summary = {'scenario': scenario, 'screen': screen, 'group': group, 'n_projects': len(subset),
                   'project_median_pct': float(np.median([r['S1_gain_pct'] for r in subset])),
                   'weighted_gain_pct': 100 * net / base, 'annual_delta_GWh': net, 'annual_baseline_GWh': base,
                   'positive': sum(r['delta_GWh'] * 5 > 1e-7 for r in subset),
                   'zero': sum(abs(r['delta_GWh']) * 5 <= 1e-7 for r in subset),
                   'negative': sum(r['delta_GWh'] * 5 < -1e-7 for r in subset),
                   'weather_year_bootstrap_lower_pct': float(np.quantile(bootstrap, .025)),
                   'weather_year_bootstrap_upper_pct': float(np.quantile(bootstrap, .975)),
                   'reference_capacity_GW': sum(r['n_turb'] for r in subset) * .01,
                   'coastal_net_delta_share_pct': 100 * sum(r['delta_GWh'] for r in retained if r['coastal_group']) / total_delta,
                   'net_added_energy_share_pct': 100 * net / total_delta}
        for key, value in summary.items():
            if key not in ('scenario', 'screen', 'group'):
                close(value, float(reference[key]), f'Summary {scenario}/{screen}/{group}: {key}')
        summaries.append(summary)
    require(len(summaries) == 36, 'Summary row count mismatch')

    historical = read_csv(archive, 'server/final_consistency/outputs/historical_annual.csv')
    historical_sites = read_csv(archive, 'server/final_consistency/outputs/historical_sites.csv')
    complete_weather = load_npz('server/final_consistency/outputs/weather_complete.npz')
    require(complete_weather['farm_ids'].tolist() == ids.tolist(), 'Historical weather index mismatch')
    mapping = {(int(r['farm_id']), int(r['year'])): r for r in read_csv(archive, 'server/final_consistency/outputs/historical_geometry_mapping.csv')}
    id_positions = {int(farm_id): i for i, farm_id in enumerate(ids)}
    historical_angles, historical_power = {}, {}
    for farm_id, index in id_positions.items():
        terminal = load_npz(f'server/outputs/models/baseline/F{farm_id:03d}_5deg.npz')['power_W']
        historical_power[farm_id] = terminal
        training_curve = energy(terminal, complete_weather['p_history'][index, 1:][None, :, :])[0]
        historical_angles[farm_id] = int(training_curve[:36:2].argmax()) * 2
    by_site = defaultdict(list)
    for row in historical:
        farm_id, year = int(row['farm_id']), int(row['year'])
        geometry = mapping[(farm_id, year)]
        power = historical_power[farm_id] if boolean(geometry['terminal_equivalent']) else load_npz('server/final_consistency/outputs/annual_models/' + geometry['key'] + '.npz')['power_W']
        annual_curve = energy(power, complete_weather['p_fy'][id_positions[farm_id], year - 2014, 1:][None, :, :])[0]
        angle = historical_angles[farm_id]
        require(angle * 5 == int(row['angle_deg']), 'Historical selection angle mismatch')
        close(annual_curve[0], float(row['baseline_GWh']), 'Historical baseline energy from annual geometry')
        close(annual_curve[angle], float(row['selected_GWh']), 'Historical selected energy from annual geometry')
        gain = 100 * (float(row['selected_GWh']) / float(row['baseline_GWh']) - 1)
        close(gain, float(row['gain_pct']), 'Historical annual energy/gain identity')
        by_site[int(row['farm_id'])].append(gain)
    for row in historical_sites:
        gains = by_site[int(row['farm_id'])]
        require(len(gains) == int(row['years']), 'Historical site year count mismatch')
        close(np.mean(gains), float(row['G']), 'Historical site mean gain')
    require((len(historical), len(historical_sites)) == (1203, 171), 'Historical record count mismatch')
    hist_gains = np.array([float(row['G']) for row in historical_sites])
    return {'numpy_version': np.__version__, 'model_site_rows_recomputed': len(rebuilt_sites),
            'planning_model_year_rows_recomputed': len(rebuilt_years), 'model_configurations_checked': len(rebuilt_sites),
            'frozen_angles_reselected': len(selected), 'baseline_curve_shape': list(archived_curves['E_year_GWh'].shape),
            'maximum_site_energy_difference_GWh': max_energy_error, 'summary_rows_recomputed': len(summaries),
            'summaries': summaries,
            'historical_array_checks': {'annual_rows': len(historical), 'site_rows': len(historical_sites), 'historical_angles_reselected': len(historical_angles),
                                      'mean_site_gain_pct': float(hist_gains.mean()), 'median_site_gain_pct': float(np.median(hist_gains)),
                                      'positive_annual_fraction_pct': 100 * sum(float(r['gain_pct']) > 1e-8 for r in historical) / len(historical)},
            'scope': 'Archived power responses and weather probabilities; no wake-model rerun or raw-weather reconstruction.'}


def main():
    parser = argparse.ArgumentParser(description='Verify the APEN numerical archive and recompute planning summaries without extracting or executing archived scripts.')
    parser.add_argument('archive', type=Path, nargs='?', default=Path(DEFAULT_ARCHIVE))
    parser.add_argument('--expected-sha256', default=ORIGINAL_SHA256, help='Expected ZIP SHA-256; default is the original v5.33 scientific archive.')
    parser.add_argument('--integrity-only', action='store_true', help='Use only the Python 3.11+ standard library for checksums and screened CSV summaries.')
    parser.add_argument('--output', type=Path, help='Write the complete verification report as JSON.')
    args = parser.parse_args()
    report = {'integrity': verify_integrity(args.archive, args.expected_sha256)}
    if not args.integrity_only:
        with ZipFile(args.archive) as archive:
            report['recomputation'] = recompute_from_arrays(archive)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
