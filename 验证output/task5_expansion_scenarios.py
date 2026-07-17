"""
任务五(b)：扩建情景多机型 FLORIS 仿真
输入: 琪明扩建排布 CSV + 3档机型 (IEA 10/15/22MW)
输出: task5_expansion_results.csv (farm, paradigm, target_year, turbine_type, AEP/CF/WL)
"""
import os, sys, csv, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    get_ti_for_farm, get_turbine_params,
    load_task0_coordinates, load_farms_master,
)
from task2_floris import (
    precompute_wake_table, replay_hourly_from_bins,
    get_era5_nc_path, get_region_for_farm, extract_wind_series,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# 3 turbine scenarios
TURBINES = ['iea_10MW', 'iea_15MW', 'iea_22MW']

def main():
    coords = load_task0_coordinates(); farms = load_farms_master()
    ws = np.array(WS_BINS); wd = np.array(WD_SECTORS)
    YR = 2024

    # Load expansion layouts
    layouts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'for_tingxian_expansion_layouts.csv')
    layout_rows = list(csv.DictReader(open(layouts_path, 'r', encoding='utf-8-sig')))

    # Group by (farm_id, paradigm, target_year)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in layout_rows:
        key = (int(r['farm_id']), r['paradigm'], r['target_year'])
        groups[key].append((float(r['x_m']), float(r['y_m'])))

    combos = sorted(groups.items(), key=lambda kv: len(kv[1]))
    total = len(combos) * len(TURBINES)

    csv_out = os.path.join(OUT_DIR, 'task5_expansion_results.csv')
    with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(['farm_id', 'paradigm', 'target_year', 'turbine_type', 'n_turb',
                               'AEP_kWh', 'AEP_noWake_kWh', 'CF', 'WakeLoss', 'precompute_time_s'])

    print(f'{len(combos)} combos x {len(TURBINES)} turbines = {total} runs')
    t0 = time.time(); done = 0; failed = 0

    for (fid, paradigm, target_year), pts in combos:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        n = len(pts)
        info = farms.get(fid, {})
        lat = info.get('centroid_lat', 0); lon = info.get('centroid_lon', 0)
        ti = get_ti_for_farm(lat, lon)
        region = get_region_for_farm(lat, lon)
        nc = get_era5_nc_path(region, YR)
        if nc is None:
            failed += len(TURBINES)
            continue
        ws100, wdd = extract_wind_series(nc, lat, lon)

        for tt in TURBINES:
            tp = get_turbine_params(tt)
            ws_hub = ws100 * (tp['H'] / 100) ** ALPHA_DEFAULT
            rated = float(max(tp['power_table']['power']))  # max power in kW
            cap = n * rated

            try:
                we, et, _ = precompute_wake_table(xs, ys, tt, ti, ALPHA_DEFAULT, ws, wd, 'gauss')
                rr = replay_hourly_from_bins(ws_hub, wdd, we, list(ws), list(wd), tp, tt, n, cap, fid, YR)

                with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([fid, paradigm, target_year, tt, n,
                        rr['AEP_kWh'], rr['AEP_noWake_kWh'], rr['CF'], rr['WakeLoss'], et])
                done += 1
                elapsed = (time.time() - t0) / 60
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                cf_val = rr['CF']; wl_val = rr['WakeLoss']
                print(f'[{done}/{total}] {fid}/{paradigm}/{target_year}/{tt}: CF={cf_val:.3f} WL={wl_val:.3f} | {elapsed:.0f}min ETA{eta:.0f}min')
            except Exception as e:
                failed += 1
                print(f'[{done}/{total}] F{fid}/{paradigm}/{target_year}/{tt}: ERR {str(e)[:80]}')

    print(f'\nDone: {done}/{total}, {failed} failed, in {(time.time()-t0)/60:.0f}min')
    print(f'Output: {csv_out}')

if __name__ == '__main__':
    main()
