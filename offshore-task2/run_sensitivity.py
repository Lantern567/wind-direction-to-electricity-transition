"""机型敏感性分析: 5场 x 5机型 x 2模型(Gauss+Jensen) = 50 runs"""
import sys, time, csv, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floris_config import (
    OUT_DIR, get_ti_for_farm, get_turbine_params,
    load_task0_coordinates, load_farms_master,
    WS_BINS, WD_SECTORS, WS_BINS_JENSEN, WD_SECTORS_JENSEN,
)
from task2_floris import precompute_wake_table, replay_hourly_from_bins
from task2_floris import get_era5_nc_path, get_region_for_farm, extract_wind_series

coords = load_task0_coordinates(); farms = load_farms_master()

FARMS = [
    (152, 2024, 'Denmark', 'Small (10t)'),
    (50,  2024, 'China',   'Medium (100t)'),
    (42,  2024, 'Taiwan',  'Medium (111t)'),
    (22,  2024, 'Germany', 'Large (161t)'),
    (3,   2024, 'China',   'Mega (441t)'),
]
TURBINES = ['ow_6MW', 'ow_8MW', 'ow_10MW', 'ow_12MW', 'ow_15MW']
MODELS = ['gauss', 'jensen']

total = len(FARMS) * len(TURBINES) * len(MODELS)
csv_out = os.path.join(OUT_DIR, 'turbine_sensitivity.csv')

with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerow([
        'farm_id','year','country','farm_label','n_turb','turbine_type','wake_model',
        'AEP_kWh','AEP_noWake_kWh','CF','WakeLoss','precompute_time_s'
    ])

print(f' 5 farms x 5 turbines x 2 models = {total} runs')
print(f' Farms: F152(10t DK), F50(100t CN), F42(111t TW), F22(161t DE), F3(441t CN)')
print(f' Models: gauss + jensen\n')

ws_g = np.array(WS_BINS); wd_g = np.array(WD_SECTORS)
ws_j = np.array(WS_BINS_JENSEN); wd_j = np.array(WD_SECTORS_JENSEN)
t0 = time.time(); idx = 0

for fid, yr, country, label in FARMS:
    turbs = coords[fid].get(yr, [])
    n = len(turbs)
    info = farms[fid]
    lat, lon = info['centroid_lat'], info['centroid_lon']
    ti = get_ti_for_farm(lat, lon)
    xs = [float(t['x_m']) for t in turbs]; ys = [float(t['y_m']) for t in turbs]

    region = get_region_for_farm(lat, lon)
    nc = get_era5_nc_path(region, yr)
    ws100, wdd = extract_wind_series(nc, lat, lon)

    for ttype in TURBINES:
        tp = get_turbine_params(ttype)
        ws_hub = ws100 * (tp['H'] / 100.0) ** 0.11
        rated = tp['power_table'].get('controller_dependent_turbine_parameters', {}).get('rated_power', 10000) or 10000
        cap = n * rated

        for wm in MODELS:
            if wm == 'jensen':
                ws_b, wd_b = ws_j, wd_j
            else:
                ws_b, wd_b = ws_g, wd_g

            we, et, _ = precompute_wake_table(xs, ys, ttype, ti, 0.11, ws_b, wd_b, wm)
            r = replay_hourly_from_bins(ws_hub, wdd, we, list(ws_b), list(wd_b), tp, ttype, n, cap, fid, yr)

            with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([
                    fid, yr, country, label, n, ttype, wm,
                    r['AEP_kWh'], r['AEP_noWake_kWh'], r['CF'], r['WakeLoss'], et
                ])

            idx += 1
            et_total = time.time() - t0
            print(f'[{idx}/{total}] F{fid} {country}({n}t) {ttype}/{wm}: CF={r["CF"]:.3f} WL={r["WakeLoss"]:.3f} t={et:.0f}s | ETA {(total-idx)*et_total/idx/60:.0f}min')

print(f'\nDone! {idx} runs in {(time.time()-t0)/60:.1f}min')
print(f'Result: {csv_out}')
