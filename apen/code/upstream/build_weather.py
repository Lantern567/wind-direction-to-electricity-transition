import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2])) / 'server'
YEARS = np.arange(2014, 2025)


def calculate(farms, folder):
    ws = np.load(ROOT / 'inputs/legacy_eta.npz')['ws']
    edges = np.r_[2.5, (ws[1:] + ws[:-1]) / 2, np.inf]
    direction_edges = np.arange(-2.5, 362.5, 5.0)
    outputs = {mode: np.zeros((len(farms), 11, 18, 72), dtype=np.float64)
               for mode in ['legacy', 'wrapped_conditional', 'wrapped_all_hours']}
    audit = []
    for fi, row in enumerate(farms.itertuples()):
        source = folder / f'{row.grid_key}.nc'
        with xr.open_dataset(source) as ds:
            time_name = 'time' if 'time' in ds.dims else 'valid_time'
            timestamp = pd.DatetimeIndex(ds[time_name].values)
            expected = pd.date_range('2014-01-01', '2024-12-31T23:00', freq='h')
            assert timestamp.equals(expected), (row.farm_id, source)
            assert float(ds.latitude) == row.grid_lat and float(ds.longitude) == row.grid_lon
            u, v = ds.u100.values.astype(float), ds.v100.values.astype(float)
        assert np.isfinite(u).all() and np.isfinite(v).all()
        speed = np.hypot(u, v) * 1.19 ** 0.11
        direction = (270 - np.rad2deg(np.arctan2(v, u))) % 360
        wrapped = (direction + 2.5) % 360 - 2.5
        for yi, year in enumerate(YEARS):
            mask = timestamp.year == year
            h = np.histogram2d(speed[mask], direction[mask], bins=[edges, direction_edges])[0]
            h_wrap = np.histogram2d(speed[mask], wrapped[mask], bins=[edges, direction_edges])[0]
            total = int(mask.sum())
            outputs['legacy'][fi, yi] = (h/h.sum()).astype(np.float32)
            outputs['wrapped_conditional'][fi, yi] = h_wrap/h_wrap.sum()
            # 低于2.5m/s作为零功率时间保留在总小时数分母中。
            outputs['wrapped_all_hours'][fi, yi] = h_wrap/total
            audit.append({'farm_id': row.farm_id, 'year': int(year), 'total_hours': total,
                          'legacy_retained_hours': int(h.sum()), 'wrapped_retained_hours': int(h_wrap.sum()),
                          'omitted_low_speed_hours': int(total-h_wrap.sum()),
                          'omitted_north_sector_hours': int(h_wrap.sum()-h.sum())})
    return outputs, pd.DataFrame(audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--source', choices=['weather', 'weather_arco'], default='weather_arco')
    parser.add_argument('--check-legacy', action='store_true', help='Also compare against the separately supplied legacy_curves.npz.')
    args = parser.parse_args()
    farms = pd.read_csv(ROOT / 'inputs/weather_grid_mapping.csv').sort_values('farm_id')
    if args.pilot:
        farms = farms[farms.farm_id == 57]
    output, audit = calculate(farms, ROOT / args.source)
    rows = []
    if args.check_legacy:
        legacy_curves = np.load(ROOT / 'inputs/legacy_curves.npz')
        legacy_eta = np.load(ROOT / 'inputs/legacy_eta.npz')
        for i, fid in enumerate(farms.farm_id):
            old_i = np.flatnonzero(legacy_curves['farm_ids'] == fid).item()
            eta_i = np.flatnonzero(legacy_eta['farm_ids'] == fid).item()
            eta = legacy_eta['eta'][eta_i].astype(float)
            power_per_turbine = eta * legacy_eta['P0'][:, None]
            p = output['legacy'][i]
            energy = np.fft.ifft(np.conj(np.fft.fft(p, axis=-1)) * np.fft.fft(power_per_turbine, axis=-1)[None, :, :], axis=-1).real.sum(axis=1)*8.76
            old = legacy_curves['E_y'][old_i]
            rows.append({'farm_id': int(fid), 'max_energy_difference_pct': float(np.max(np.abs(energy-old)/old)*100),
                         'mean_energy_difference_pct': float(np.mean((energy-old)/old)*100)})
    destination = ROOT / 'outputs/weather_pilot' if args.pilot else ROOT / 'outputs/weather_joint'
    destination.mkdir(parents=True, exist_ok=True)
    for name, p in output.items():
        np.savez_compressed(destination / f'{name}.npz', p_fy=p, farm_ids=farms.farm_id.to_numpy(), years=YEARS)
    audit.to_csv(destination / 'hours_audit.csv', index=False)
    if args.check_legacy:
        pd.DataFrame(rows).to_csv(destination / 'legacy_energy_comparison.csv', index=False)
    print(json.dumps({'farms': len(farms), 'source': args.source, 'energy_comparison': rows[:5],
                      'omitted_hours': audit.groupby('farm_id')[['omitted_low_speed_hours', 'omitted_north_sector_hours']].mean().head().to_dict()}), flush=True)
