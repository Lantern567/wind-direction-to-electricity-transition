"""
任务四：邻近场簇跨场尾流仿真
挑 2 对邻近风场（东亚 1 对 + 欧洲 1 对），比较单独跑 vs 合并跑
若合并后 AEP < 单独总和 → 跨场尾流损失
"""
import os, sys, csv, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    get_ti_for_farm, get_turbine_params, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master,
)
from task2_floris import (
    precompute_wake_table, replay_hourly_from_bins,
    get_era5_nc_path, get_region_for_farm, extract_wind_series,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# 两对邻近场
PAIRS = [
    (35, 72, 'East Asia (China)', 12.9),
    (31, 113, 'Europe (Netherlands)', 13.0),
]

def run_kW(xs, ys, farms, fid, yr, ws_bins, wd_bins):
    n = len(xs); tp = get_turbine_params(DEFAULT_TURBINE)
    rated = tp['power_table'].get('controller_dependent_turbine_parameters',{}).get('rated_power',10000) or 10000
    cap = n * rated
    info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']; ti = get_ti_for_farm(lat, lon)
    region = get_region_for_farm(lat, lon); nc = get_era5_nc_path(region, yr)
    if nc is None: return None
    ws100, wdd = extract_wind_series(nc, lat, lon)
    ws_hub = ws100 * (tp['H']/100) ** ALPHA_DEFAULT
    we, et, _ = precompute_wake_table([float(v) for v in xs], [float(v) for v in ys], DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws_bins, wd_bins, 'gauss')
    rr = replay_hourly_from_bins(ws_hub, wdd, we, list(ws_bins), list(wd_bins), tp, DEFAULT_TURBINE, n, cap, fid, yr)
    return rr, et

def main():
    coords = load_task0_coordinates(); farms = load_farms_master()
    ws = np.array(WS_BINS); wd = np.array(WD_SECTORS)
    YR = 2024

    csv_out = os.path.join(OUT_DIR, 'task4_cross_farm_wake.csv')
    with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(['pair', 'farm', 'scenario', 'n_turb', 'AEP_kWh', 'CF', 'WakeLoss', 'precompute_s'])

    for pair_name, (fid_a, fid_b, label, dist_km) in enumerate(PAIRS):
        na = farms[fid_a]['n_turb']; nb = farms[fid_b]['n_turb']
        print(f'=== Pair {pair_name+1}: F{fid_a}({na}t) + F{fid_b}({nb}t) {label} {dist_km:.1f}km ===')

        turbs_a = coords[fid_a].get(YR, []); turbs_b = coords[fid_b].get(YR, [])
        xs_a = np.array([t['x_m'] for t in turbs_a], dtype=np.float64)
        ys_a = np.array([t['y_m'] for t in turbs_a], dtype=np.float64)
        xs_b = np.array([t['x_m'] for t in turbs_b], dtype=np.float64)
        ys_b = np.array([t['y_m'] for t in turbs_b], dtype=np.float64)

        # 1) A alone
        print(f'  Running F{fid_a} alone ({len(xs_a)} turbines)...')
        rA, etA = run_kW(xs_a, ys_a, farms, fid_a, YR, ws, wd)
        aep_A = float(rA['AEP_kWh']); cf_A = float(rA['CF']); wl_A = float(rA['WakeLoss'])

        # 2) B alone
        print(f'  Running F{fid_b} alone ({len(xs_b)} turbines)...')
        rB, etB = run_kW(xs_b, ys_b, farms, fid_b, YR, ws, wd)
        aep_B = float(rB['AEP_kWh']); cf_B = float(rB['CF']); wl_B = float(rB['WakeLoss'])

        # 3) Merged
        xs_merged = list(xs_a) + list(xs_b)
        ys_merged = list(ys_a) + list(ys_b)
        n_merged = len(xs_merged)
        print(f'  Running merged ({n_merged} turbines)...')
        # Use the larger farm's ERA5 point
        fid_main = fid_a if len(xs_a) >= len(xs_b) else fid_b
        rM, etM = run_kW(xs_merged, ys_merged, farms, fid_main, YR, ws, wd)
        aep_M = float(rM['AEP_kWh']); cf_M = float(rM['CF']); wl_M = float(rM['WakeLoss'])

        cross_wake_loss = (aep_A + aep_B - aep_M) / (aep_A + aep_B) * 100 if (aep_A + aep_B) > 0 else 0

        print(f'\n  A alone: {aep_A / 1e9:.3f} GWh, CF={cf_A:.3f}, WL={wl_A:.3f}')
        print(f'  B alone: {aep_B / 1e6:.1f} MWh, CF={cf_B:.3f}, WL={wl_B:.3f}')
        print(f'  A+B sum: {(aep_A + aep_B) / 1e9:.3f} GWh')
        print(f'  Merged:  {aep_M / 1e9:.3f} GWh, CF={cf_M:.3f}, WL={wl_M:.3f}')
        print(f'  ===> Cross-farm wake loss: {cross_wake_loss:+.2f}%')

        # Save
        with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            for label_str, aep_val, cf_val, wl_val, et_val, n_val in [
                (f'F{fid_a}_alone', aep_A, cf_A, wl_A, etA, len(xs_a)),
                (f'F{fid_b}_alone', aep_B, cf_B, wl_B, etB, len(xs_b)),
                (f'F{fid_a}+F{fid_b}_merged', aep_M, cf_M, wl_M, etM, n_merged)
            ]:
                w.writerow([f'{fid_a}_{fid_b}', label_str, label_str, n_val, aep_val, cf_val, wl_val, et_val])

    print(f'\nDone. Results: {csv_out}')

if __name__ == '__main__':
    main()
