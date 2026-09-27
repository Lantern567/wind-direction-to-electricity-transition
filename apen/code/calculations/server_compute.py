import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
import shutil
from pathlib import Path
import sys
import time

import aiohttp
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import xarray as xr
import yaml
from floris import FlorisModel

HERE = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2])) / 'server/final_consistency'
BASE = HERE.parent
sys.path.insert(0, str(BASE / 'inputs'))
from legacy_floris_config import build_base_config, get_turbine_params

OUT = HERE / 'outputs'
URL = 'https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr'


@lru_cache(maxsize=1)
def geometry():
    tc = pd.read_csv(BASE / 'inputs/turbine_coordinates.csv')
    audit = pd.read_csv(BASE / 'outputs/geometry_audit.csv').set_index('farm_id')
    geometries, records = {}, []
    for fid, group in tc.groupby('farm_id'):
        terminal = group[group.year == group.year.max()]
        lon, lat = terminal.lon.to_numpy(), terminal.lat.to_numpy()
        tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(
            f'+proj=aeqd +lat_0={np.median(lat)} +lon_0={np.median(lon)} +datum=WGS84 +units=m'), always_xy=True)
        terminal_ids = tuple(sorted(terminal.turbine_id.tolist()))
        for year, annual in group.groupby('year'):
            if len(annual) < 2:
                continue
            ids = tuple(sorted(annual.turbine_id.tolist()))
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
            key = f'F{fid:03d}_{digest}'
            x, y = tr.transform(annual.lon.to_numpy(), annual.lat.to_numpy())
            geometries[key] = (x, y, float(audit.loc[fid, 'ti']))
            records.append({'farm_id': int(fid), 'year': int(year), 'key': key,
                            'n_turb': len(x), 'terminal_equivalent': ids == terminal_ids})
    return geometries, pd.DataFrame(records)


def response(x, y, ti, turbine='iea_10MW', model='gauss'):
    cfg = build_base_config(x, y, turbine_type=turbine, ti=ti, alpha=0.11, wake_model_name=model)
    fm = FlorisModel(cfg)
    ws = np.load(BASE / 'inputs/legacy_eta.npz')['ws']
    wd = np.arange(0., 360., 5.)
    power = np.empty((len(ws), len(wd)))
    flags = np.zeros_like(power, dtype=np.int32)
    vmin = np.empty_like(power)
    batch = 24 if len(x) > 400 else 72
    for ui, speed in enumerate(ws):
        for first in range(0, len(wd), batch):
            directions = wd[first:first + batch]
            fm.set(wind_speeds=np.full(len(directions), speed), wind_directions=directions,
                   turbulence_intensities=np.full(len(directions), ti))
            fm.run()
            power[ui, first:first+batch] = fm.get_farm_power().ravel()
            velocity = fm.core.flow_field.u.reshape(len(directions), -1)
            flags[ui, first:first+batch] = (velocity < 0).sum(axis=1)
            vmin[ui, first:first+batch] = velocity.min(axis=1)
    assert np.isfinite(power).all() and (power >= 0).all()
    return dict(power_W=power, negative_rotor_points=flags, minimum_rotor_velocity_ms=vmin,
                ws=ws, wd=wd, n_turb=len(x), ti=ti), cfg


def annual_job(key):
    started = time.perf_counter()
    target = OUT / 'annual_models' / f'{key}.npz'
    if target.exists():
        return {'key': key, 'status': 'existing'}
    coords, _ = geometry()
    x, y, ti = coords[key]
    data, cfg = response(x, y, ti)
    np.savez_compressed(target.with_suffix('.partial.npz'), **data)
    target.with_suffix('.partial.npz').replace(target)
    target.with_suffix('.json').write_text(json.dumps(cfg, indent=2))
    return {'key': key, 'seconds': round(time.perf_counter()-started, 2)}


def annual_additional_job(key):
    started = time.perf_counter()
    destination = OUT/'annual_models'/f'{key}.npz'
    if destination.exists():
        return {'key': key, 'status': 'existing'}
    x, y, ti = geometry()[0][key]
    data, cfg = response(x, y, ti)
    # 额外工作进程使用独立临时文件，避免同一模型的写入相互覆盖。
    target = OUT/'annual_additional'/f'{key}.npz'
    np.savez_compressed(target, **data)
    config = target.with_suffix('.json')
    config.write_text(json.dumps(cfg, indent=2))
    if not destination.exists():
        temporary = destination.with_suffix('.extra_ready.npz')
        shutil.copy2(target, temporary)
        temporary.replace(destination)
        shutil.copy2(config, destination.with_suffix('.json'))
    return {'key': key, 'seconds': round(time.perf_counter()-started, 2)}


def standard_job(pi, ti, variant):
    started = time.perf_counter()
    target = OUT / 'standard_models' / f'{variant}_P{pi}_TI{ti:.2f}.npz'
    if target.exists():
        return {'key': target.stem, 'status': 'existing'}
    xy = np.load(HERE / 'standard_geometry.npz')['xy'][pi].copy()
    turbine = {'baseline': 'iea_10MW', 'jensen': 'iea_10MW', 'nrel5': 'nrel_5MW', 'iea15': 'iea_15MW'}[variant]
    # 标准排布的机型对照保持以D计的间距，真实场址对照保持米制机位。
    xy *= get_turbine_params(turbine)['D'] / get_turbine_params('iea_10MW')['D']
    data, cfg = response(xy[:, 0], xy[:, 1], ti, turbine, 'jensen' if variant == 'jensen' else 'gauss')
    np.savez_compressed(target.with_suffix('.partial.npz'), **data, xy=xy)
    target.with_suffix('.partial.npz').replace(target)
    target.with_suffix('.json').write_text(json.dumps(cfg, indent=2))
    return {'key': target.stem, 'seconds': round(time.perf_counter()-started, 2)}


def history_job(key, lat, lon):
    target = HERE / 'weather_history' / f'{key}.nc'
    if target.exists():
        return {'key': key, 'status': 'existing'}
    token = os.environ['ECMWF_ARCO_TOKEN']
    ds = xr.open_zarr(URL, consolidated=True, chunks=None,
                      storage_options={'headers': {'Authorization': f'Bearer {token}'},
                                       'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=120, sock_read=40)}})
    point = ds[['u100', 'v100']].sel(latitude=lat, longitude=lon, method='nearest').sel(
        time=slice('1981-01-01T12:00', '2010-12-31T12:00')).isel(time=slice(None, None, 24)).load()
    expected = pd.date_range('1981-01-01T12:00', '2010-12-31T12:00', freq='D')
    assert pd.DatetimeIndex(point.time.values).equals(expected)
    assert float(point.latitude) == lat and float(point.longitude) == lon
    assert np.isfinite(point.u100).all() and np.isfinite(point.v100).all()
    point.attrs['retrieval_source'] = URL
    temporary = target.with_suffix(f'.partial_{os.getpid()}.nc')
    point.to_netcdf(temporary)
    temporary.replace(target)
    return {'key': key, 'status': 'downloaded', 'samples': len(expected)}


def weather_statistics(include_history=True):
    farms = pd.read_csv(BASE / 'inputs/weather_grid_mapping.csv').sort_values('farm_id')
    ws = np.load(BASE / 'inputs/legacy_eta.npz')['ws']
    speed_edges = np.r_[0, 2.5, (ws[1:]+ws[:-1])/2, np.inf]
    direction_edges = np.arange(-2.5, 362.5, 5.)
    probability, history, stats = [], [], []
    for row in farms.itertuples():
        periods = {}
        periods_to_read = [('evaluation', Path(os.environ.get('APEN_WEATHER_DIR', BASE / 'weather_arco')))]
        if include_history:
            periods_to_read.append(('historical', HERE / 'weather_history'))
        for label, folder in periods_to_read:
            with xr.open_dataset(folder / f'{row.grid_key}.nc') as ds:
                u, v = ds.u100.values.astype(float), ds.v100.values.astype(float)
                time_name = 'time' if 'time' in ds.dims else 'valid_time'
                timestamps = pd.DatetimeIndex(ds[time_name].values)
            speed = np.hypot(u, v) * 1.19**.11
            direction = (270-np.rad2deg(np.arctan2(v, u))) % 360
            wrapped = (direction+2.5) % 360-2.5
            if label == 'evaluation':
                annual = [np.histogram2d(speed[timestamps.year == y], wrapped[timestamps.year == y],
                            bins=[speed_edges, direction_edges])[0]/(timestamps.year == y).sum()
                          for y in range(2014, 2025)]
                probability.append(annual)
                phi = np.deg2rad(((direction+2.5)//5 % 72)*5)
                energy_weight = speed**3
                for k in [1, 2]:
                    periods[f'R{k}_frequency'] = abs(np.exp(1j*k*phi).mean())
                    periods[f'R{k}_v3'] = abs(np.sum(energy_weight*np.exp(1j*k*phi))/energy_weight.sum())
                periods['mean_speed_ms'] = speed.mean()
            else:
                history.append(np.histogram2d(speed, wrapped, bins=[speed_edges, direction_edges])[0]/len(speed))
        stats.append({'farm_id': row.farm_id, **periods})
    p, ph = np.array(probability), np.array(history)
    np.testing.assert_allclose(p.sum(axis=(-1, -2)), 1.)
    if include_history:
        np.testing.assert_allclose(ph.sum(axis=(-1, -2)), 1.)
    with np.load(BASE / 'outputs/weather_joint/wrapped_all_hours.npz') as old:
        np.testing.assert_array_equal(farms.farm_id, old['farm_ids'])
        np.testing.assert_allclose(p[:, :, 1:, :], old['p_fy'], rtol=0, atol=1e-15)
    filename = 'weather_complete.npz' if include_history else 'weather_evaluation.npz'
    np.savez_compressed(OUT / filename, farm_ids=farms.farm_id.to_numpy(),
                        p_fy=p, p_history=ph, years=np.arange(2014, 2025), ws=np.r_[0, ws], wd=np.arange(0, 360, 5))
    pd.DataFrame(stats).to_csv(OUT / 'wind_statistics.csv', index=False)
    print(json.dumps({'status': 'complete', 'sites': len(farms), 'joint_probability': 'all hours, circular direction, explicit zero-power class'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['annual', 'annual_additional', 'standard', 'history', 'weather', 'weather_evaluation'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args()
    for folder in [OUT, OUT / 'annual_models', OUT/'annual_additional', OUT / 'standard_models', HERE / 'weather_history']:
        folder.mkdir(exist_ok=True, parents=True)
    if args.mode in ['weather', 'weather_evaluation']:
        weather_statistics(include_history=args.mode == 'weather')
        return
    if args.mode in ['annual', 'annual_additional']:
        coords, manifest = geometry()
        if args.mode == 'annual':
            manifest.to_csv(OUT / 'annual_geometry_manifest.csv', index=False)
        unique = manifest[~manifest.terminal_equivalent].drop_duplicates('key').sort_values('n_turb', ascending=False)
        function = annual_job if args.mode == 'annual' else annual_additional_job
        jobs = [(key,) for key in unique.key.iloc[args.start_index:]]
    elif args.mode == 'standard':
        function = standard_job
        jobs = [(pi, float(ti), variant) for variant in ['baseline', 'jensen', 'nrel5', 'iea15']
                for ti in np.load(HERE / 'standard_geometry.npz')['ti'] for pi in range(6)]
    else:
        farms = pd.read_csv(BASE / 'inputs/weather_grid_mapping.csv').drop_duplicates('grid_key')
        function, jobs = history_job, [(r.grid_key, float(r.grid_lat), float(r.grid_lon)) for r in farms.itertuples()]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(function, *parameters) for parameters in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps({'completed': done, 'total': len(jobs), **future.result()}), flush=True)
    (OUT / f'{args.mode}_complete.json').write_text(json.dumps({'completed': len(jobs)}))


if __name__ == '__main__':
    main()
