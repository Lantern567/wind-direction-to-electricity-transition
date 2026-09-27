import argparse
import hashlib
import importlib.metadata
import json
import os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial.distance import pdist, squareform
from floris import FlorisModel
import floris

ROOT = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2])) / 'server'
sys.path.insert(0, str(ROOT / 'inputs'))
from legacy_floris_config import build_base_config, get_ti_for_farm

SCENARIOS = {
    'baseline': ('iea_10MW', 'gauss', 0.0),
    'nrel5_fixed_coordinates': ('nrel_5MW', 'gauss', 0.0),
    'iea15_fixed_coordinates': ('iea_15MW', 'gauss', 0.0),
    'gauss_ti_minus002': ('iea_10MW', 'gauss', -0.02),
    'gauss_ti_plus002': ('iea_10MW', 'gauss', 0.02),
    'jensen': ('iea_10MW', 'jensen', 0.0),
}


@lru_cache(maxsize=1)
def geometry():
    tc = pd.read_csv(ROOT / 'inputs/turbine_coordinates.csv').sort_values(['farm_id', 'year'])
    tc = tc.merge(tc.groupby('farm_id')['year'].max().reset_index(), on=['farm_id', 'year'], validate='many_to_one')
    geo = pd.read_csv(ROOT / 'inputs/geometry.csv').set_index('farm_id')
    assert geo.index.is_unique
    coordinates = {}
    rows = []
    for fid, group in tc.groupby('farm_id', sort=True):
        assert not group.duplicated(['lon', 'lat']).any(), fid
        lon, lat = group.lon.to_numpy(), group.lat.to_numpy()
        transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(
            f'+proj=aeqd +lat_0={np.median(lat)} +lon_0={np.median(lon)} +datum=WGS84 +units=m'), always_xy=True)
        x, y = transformer.transform(lon, lat)
        assert len(x) == geo.loc[fid, 'n_turb'], fid
        coordinates[int(fid)] = (x, y)
        distances = squareform(pdist(np.column_stack([x, y])))
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        rows.append({'farm_id': int(fid), 'n_turb': len(x), 'terminal_year': int(group.year.max()),
                     'min_distance_m': nearest.min(), 'nn_median_m': np.median(nearest),
                     'ti': get_ti_for_farm(geo.loc[fid, 'cent_lat'], geo.loc[fid, 'cent_lon']),
                     'centroid_lon': geo.loc[fid, 'cent_lon'], 'centroid_lat': geo.loc[fid, 'cent_lat']})
    assert len(coordinates) == 171
    return coordinates, pd.DataFrame(rows).set_index('farm_id')


def job(fid, scenario, step):
    start = time.perf_counter()
    coords, audit = geometry()
    output = ROOT / 'outputs/models' / scenario
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'F{fid:03d}_{step}deg.npz'
    if target.exists():
        z = np.load(target)
        assert int(z['farm_id']) == fid and np.isfinite(z['power_W']).all()
        if 'negative_rotor_points' in z.files:
            return {'farm_id': fid, 'scenario': scenario, 'status': 'existing'}
    x, y = coords[fid]
    turbine, model, perturbation = SCENARIOS[scenario]
    ti = float(audit.loc[fid, 'ti']) + perturbation
    assert ti > 0
    cfg = build_base_config(x, y, turbine_type=turbine, ti=ti, alpha=0.11, wake_model_name=model)
    # 各情景使用相同轮毂处风速输入；原生轮毂高度仅用于模型几何及转子风切变。
    # 因而本组对照没有同时引入ERA5风速高度换算的资源变化。
    config_path = output / f'F{fid:03d}_{step}deg_config.json'
    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    fm = FlorisModel(cfg)
    ws = np.load(ROOT / 'inputs/legacy_eta.npz')['ws']
    wd = np.arange(0.0, 360.0, float(step))
    power = np.empty((len(ws), len(wd)))
    negative_points = np.zeros((len(ws), len(wd)), dtype=np.int32)
    minimum_velocity = np.empty((len(ws), len(wd)))
    batch = 24 if len(x) > 400 else 72
    for ui, speed in enumerate(ws):
        for first in range(0, len(wd), batch):
            directions = wd[first:first + batch]
            fm.set(wind_speeds=np.full(len(directions), speed), wind_directions=directions,
                   turbulence_intensities=np.full(len(directions), ti))
            fm.run()
            power[ui, first:first + batch] = fm.get_farm_power().ravel()
            velocity = fm.core.flow_field.u.reshape(len(directions), -1)
            negative_points[ui, first:first + batch] = (velocity < 0).sum(axis=1)
            minimum_velocity[ui, first:first + batch] = velocity.min(axis=1)
    assert np.isfinite(power).all() and (power >= 0).all(), (fid, scenario)
    temporary = target.with_suffix('.partial.npz')
    np.savez_compressed(temporary, farm_id=fid, n_turb=len(x), power_W=power, ws=ws, wd=wd,
                        turbulence_intensity=ti, scenario=scenario,
                        negative_rotor_points=negative_points, minimum_rotor_velocity_ms=minimum_velocity,
                        reference_wind_height=cfg['flow_field']['reference_wind_height'])
    temporary.replace(target)
    return {'farm_id': fid, 'scenario': scenario, 'seconds': round(time.perf_counter() - start, 2), 'status': 'computed'}


def benchmark():
    coords, audit = geometry()
    (ROOT / 'outputs').mkdir(exist_ok=True)
    audit.to_csv(ROOT / 'outputs/geometry_audit.csv')
    legacy = np.load(ROOT / 'inputs/legacy_eta.npz')
    rows = []
    for fid in [155, 57, 91]:
        result = job(fid, 'baseline', 5)
        fresh = np.load(ROOT / f'outputs/models/baseline/F{fid:03d}_5deg.npz')
        i = np.flatnonzero(legacy['farm_ids'] == fid).item()
        expected = legacy['eta'][i].astype(float) * legacy['P0'][:, None] * legacy['n_turb'][i]
        error = np.max(np.abs(fresh['power_W'] - expected))
        relative = error / max(np.max(expected), 1)
        result.update(max_absolute_W=float(error), max_relative_to_peak=float(relative))
        rows.append(result)
        print(json.dumps(result), flush=True)
    cfg = build_base_config([0.0], [0.0], turbine_type='iea_10MW', ti=0.07)
    fm = FlorisModel(cfg)
    fm.set(wind_speeds=legacy['ws'], wind_directions=np.full(18, 270.0), turbulence_intensities=np.full(18, 0.07))
    fm.run()
    np.testing.assert_allclose(fm.get_farm_power(), legacy['P0'], rtol=1e-12, atol=1e-7)
    replicated = all(row['max_relative_to_peak'] < 2e-6 for row in rows)
    report = {'status': 'exact_replication' if replicated else 'full_baseline_recalculation_required',
              'floris': importlib.metadata.version('floris'), 'checks': rows,
              'single_turbine_curve': 'matched at rtol 1e-12, atol 1e-7 W',
              'definition': 'maximum absolute power difference divided by peak farm power; replication tolerance 2e-6',
              'interpretation': 'Cross-host farm-power difference remains unresolved. Recompute baseline and all controls in one environment. Do not call the legacy arrays an exact reproduction.'}
    (ROOT / 'outputs/baseline_model_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    versions = {name: importlib.metadata.version(name) for name in ['floris', 'numpy', 'pandas', 'scipy', 'pyproj']}
    (ROOT / 'outputs/environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    library = Path(floris.__file__).parent / 'turbine_library'
    records = []
    (ROOT / 'outputs/turbine_curves').mkdir(exist_ok=True)
    for name in ['iea_10MW', 'nrel_5MW', 'iea_15MW']:
        content = (library / f'{name}.yaml').read_bytes()
        (ROOT / 'outputs/turbine_curves' / f'{name}.yaml').write_bytes(content)
        records.append({'turbine': name, 'sha256': hashlib.sha256(content).hexdigest()})
    (ROOT / 'outputs/turbine_curve_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['benchmark', 'run'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--scenarios', nargs='+', choices=list(SCENARIOS), default=list(SCENARIOS)[1:])
    parser.add_argument('--farms', nargs='+', type=int)
    parser.add_argument('--step', type=int, choices=[1, 5], default=5)
    args = parser.parse_args()
    if args.mode == 'benchmark':
        benchmark()
        return
    check = json.loads((ROOT / 'outputs/baseline_model_check.json').read_text())
    if check['status'] != 'exact_replication':
        assert check['status'] == 'full_baseline_recalculation_required'
        assert 'baseline' in args.scenarios, 'The comparison requires a fresh baseline in this environment.'
    coords, audit = geometry()
    farms = args.farms or list(audit.sort_values('n_turb', ascending=False).index)
    jobs = [(int(fid), scenario, args.step) for scenario in args.scenarios for fid in farms
            if scenario != 'jensen' or audit.loc[fid, 'n_turb'] >= 10]
    manifest = {'scenarios': args.scenarios, 'step_deg': args.step, 'farms': [int(f) for f in farms],
                'workers': args.workers, 'jobs': len(jobs), 'wind_input': 'same hub-level speed nodes in every scenario',
                'coordinates': 'fixed metres, no rotor-based rescaling', 'selected_angles': 'inputs/frozen_angles.csv'}
    identity = hashlib.sha256(json.dumps(jobs).encode()).hexdigest()[:12]
    (ROOT / f'outputs/run_manifest_{args.step}deg_{identity}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(job, *parameters) for parameters in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps({'completed': completed, 'total': len(jobs), **result}), flush=True)


if __name__ == '__main__':
    main()
