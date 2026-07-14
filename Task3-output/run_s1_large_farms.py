"""
S1 大场专用: 逐个处理 >200台的13个场
从小(F12 244t)到大(F0 928t)，每个跑完18角度自动保存
"""
import sys, os, time, csv, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')

from floris_config import get_ti_for_farm, load_task0_coordinates, load_farms_master
from task3_s1_optimal_orientation import (
    SCAN_ANGLES, load_historical_wind_distribution, compute_historical_aep_for_rotation
)

coords = load_task0_coordinates(); farms = load_farms_master()

LARGE_FARMS = [
    (12, 244), (10, 246), (11, 246), (9, 267), (8, 278),
    (7, 284), (6, 293), (5, 339), (4, 364), (3, 443),
    (2, 572), (1, 589), (0, 928),
]

csv_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'task3_s1_optimal_orientation.csv')

t_total = time.time()
for idx, (fid, expected_n) in enumerate(LARGE_FARMS):
    turbs = coords[fid].get(2024, [])
    n = len(turbs)
    info = farms[fid]
    lat = info['centroid_lat']; lon = info['centroid_lon']
    ti = get_ti_for_farm(lat, lon)
    country = info.get('country','?')

    print(f'\n[{idx+1}/13] F{fid}: {n}turb, {country}, TI={ti}')
    print(f'  加载历史风分布...')

    hist_dist = load_historical_wind_distribution(lat, lon)
    xs = np.array([t['x_m'] for t in turbs], dtype=np.float64)
    ys = np.array([t['y_m'] for t in turbs], dtype=np.float64)

    results = []
    t_farm = time.time()
    crashed = False

    for ang_idx, ang in enumerate(SCAN_ANGLES):
        try:
            r = compute_historical_aep_for_rotation(
                xs, ys, ang, 'iea_10MW', ti, 0.11, 'gauss', hist_dist
            )
            r['angle'] = ang
            results.append(r)
            elapsed = time.time() - t_farm
            per_angle = elapsed / len(results)
            remaining = (len(SCAN_ANGLES) - len(results)) * per_angle
            print(f'  [{len(results)}/18] {ang}deg: CF={r["expected_CF"]:.3f} AEP={r["expected_AEP_kWh"]/1e9:.2f}GWh ETA={remaining/60:.0f}min')
        except Exception as e:
            print(f'  [{len(results)}/18] {ang}deg: CRASH {str(e)[:100]}')
            crashed = True
            break

    if crashed:
        print(f'  F{fid} SKIPPED (FLORIS crash at angle {ang}deg)')
        continue

    if results:
        best = max(results, key=lambda r: r['expected_AEP_kWh'])
        worst = min(results, key=lambda r: r['expected_AEP_kWh'])
        spread = ((best['expected_AEP_kWh'] - worst['expected_AEP_kWh']) / best['expected_AEP_kWh'] * 100)
        print(f'  F{fid} DONE: theta_opt={best["angle"]}deg CF={best["expected_CF"]:.3f} spread={spread:.1f}% ({len(results)} angles, {(time.time()-t_farm)/60:.0f}min)')

        with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            for r in results:
                w.writerow([fid, r['angle'], n, country, ti,
                           r['expected_AEP_kWh'], r['expected_AEP_noWake_kWh'],
                           r['expected_CF'], r['expected_WakeLoss'], r['precompute_time_s']])

elapsed_total = time.time() - t_total
done = sum(1 for fid,_ in LARGE_FARMS if any(
    int(r['farm_id'])==fid for r in csv.DictReader(open(csv_out,'r',encoding='utf-8-sig')) if int(r.get('farm_id',-1))==fid
))
print(f'\nDone: {done}/13 large farms in {elapsed_total/60:.0f}min total')
