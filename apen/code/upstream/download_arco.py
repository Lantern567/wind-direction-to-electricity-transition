import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(os.environ.get('APEN_DATA_ROOT', Path(__file__).resolve().parents[2])) / 'server'
URL = 'https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr'


def download(key, lat, lon):
    target = ROOT / 'weather_arco' / f'{key}.nc'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        return {'key': key, 'status': 'existing'}
    token = os.environ['ECMWF_ARCO_TOKEN']
    ds = xr.open_zarr(URL, consolidated=True, chunks=None,
                      storage_options={'headers': {'Authorization': f'Bearer {token}'}})
    point = ds[['u100', 'v100']].sel(latitude=lat, longitude=lon, method='nearest').sel(time=slice('2014-01-01', '2024-12-31')).load()
    assert float(point.latitude) == lat and float(point.longitude) == lon
    assert point.sizes['time'] == 96432
    assert np.isfinite(point.u100).all() and np.isfinite(point.v100).all()
    point.attrs['retrieval_source'] = URL
    point.to_netcdf(target.with_suffix('.partial.nc'))
    target.with_suffix('.partial.nc').rename(target)
    pilot = ROOT / 'weather' / f'{key}.nc'
    result = {'key': key, 'status': 'downloaded', 'bytes': target.stat().st_size}
    if pilot.exists():
        with xr.open_dataset(pilot) as comparison:
            result['max_difference_CDS_u100'] = float(np.abs(point.u100.to_numpy()-comparison.u100.to_numpy()).max())
            result['max_difference_CDS_v100'] = float(np.abs(point.v100.to_numpy()-comparison.v100.to_numpy()).max())
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    farms = pd.read_csv(ROOT / 'inputs/weather_grid_mapping.csv')
    if args.pilot:
        farms = farms[farms.farm_id == 57]
    grid = farms.drop_duplicates('grid_key')
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row.grid_key, float(row.grid_lat), float(row.grid_lon)) for row in grid.itertuples()]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps({'completed': i, 'total': len(grid), **result}), flush=True)
