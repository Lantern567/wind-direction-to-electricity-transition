"""
补算2: Jensen旋转重跑 — 26个长尾场 + 30个分层样场
只用Jensen加速分箱(252组合), 18角度, 每个角度单独FLORIS run
"""
import os, sys, csv, time, numpy as np
sys.path.insert(0, r'D:/1风力发电实习/offshore-task2')
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS_JENSEN as WS_BINS, WD_SECTORS_JENSEN as WD_SECTORS,
    ALPHA_DEFAULT, get_ti_for_farm, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master,
)
from task2_floris import precompute_wake_table

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jensen_rotation_56farms.csv')
coords = load_task0_coordinates(); farms = load_farms_master()
ws = np.array(WS_BINS); wd = np.array(WD_SECTORS)
YR = 2024; ANGLES = list(range(0, 180, 10))

# Select farms
og_path = r'D:/1风力发电实习/wind-direction-to-electricity-transition-main/分析材料_几何主导杠杆/data_derived/orientation_gain.csv'
og = {}
for r in csv.DictReader(open(og_path,'r',encoding='utf-8-sig')):
    fid = int(r['farm_id']); g = float(r['gain_pct'])
    if fid not in og or g > og[fid]: og[fid] = g

long_tail = sorted([fid for fid, g in og.items() if g > 2])
sizes = [(fid, int(farms[fid]['n_turb'])) for fid in range(171) if int(farms[fid]['n_turb'])>1]
small = [f for f,n in sizes if n<=20 and f not in long_tail]
med   = [f for f,n in sizes if 20<n<=80 and f not in long_tail]
large = [f for f,n in sizes if n>80 and f not in long_tail]
np.random.seed(42)
stratified = list(np.random.choice(small, min(10,len(small)),replace=False)) + \
             list(np.random.choice(med,   min(10,len(med)),  replace=False)) + \
             list(np.random.choice(large, min(10,len(large)),replace=False))
all_farms = sorted(set(long_tail + stratified))
print(f'Long tail: {len(long_tail)}, Stratified: {len(stratified)}, Total: {len(all_farms)}')

done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT,'r',encoding='utf-8-sig')):
        done.add((int(r['farm_id']), int(r['angle_deg'])))

with open(OUT, 'w' if not os.path.exists(OUT) else 'a', newline='', encoding='utf-8-sig') as f:
    if not os.path.exists(OUT) or os.path.getsize(OUT)==0:
        csv.writer(f).writerow(['farm_id','angle_deg','n_turb','country','AEP_kWh','AEP_noWake_kWh','CF','WakeLoss','precompute_s'])

t0 = time.time(); cnt = 0; total = sum(1 for fid in all_farms for _ in ANGLES)

for fid in all_farms:
    turbs = coords[fid].get(YR, []); n = len(turbs)
    if n < 2: continue
    info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
    ti = get_ti_for_farm(lat, lon)
    country = info.get('country', '?')
    xs_orig = np.array([float(t['x_m']) for t in turbs], dtype=np.float64)
    ys_orig = np.array([float(t['y_m']) for t in turbs], dtype=np.float64)
    cx, cy = np.mean(xs_orig), np.mean(ys_orig)

    for ang in ANGLES:
        if (fid, ang) in done: continue
        theta = np.radians(ang); ct, st = np.cos(theta), np.sin(theta)
        xr = (xs_orig - cx) * ct - (ys_orig - cy) * st + cx
        yr = (xs_orig - cx) * st + (ys_orig - cy) * ct + cy
        we, et, nt = precompute_wake_table([float(v) for v in xr], [float(v) for v in yr],
                                           DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws, wd, 'jensen')
        from floris_config import get_turbine_params
        tp = get_turbine_params(DEFAULT_TURBINE)
        ws_pc = np.array(tp['power_table']['wind_speed'])
        pw_pc = np.array(tp['power_table']['power'])
        per_turb_power = np.array([np.interp(w, ws_pc, pw_pc, left=0, right=0) for w in ws])
        pn = per_turb_power * n
        uniform_weight = 1.0 / float(len(ws))
        aep = sum(float(pn[iws]) * float(np.mean(we[:, iws])) * uniform_weight for iws in range(len(ws))) * 8760.0
        aep_nw = sum(float(pn[iws]) for iws in range(len(ws))) * uniform_weight * 8760.0
        rated_per_turb = float(max(tp['power_table']['power']))
        cap = n * rated_per_turb
        cf = aep / (cap * 8760.0) if cap > 0 else 0.0
        wl = (aep_nw - aep) / aep_nw if aep_nw > 0 else 0.0

        with open(OUT, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow([fid, ang, n, country, aep, aep_nw, cf, wl, et])

        cnt += 1
        if cnt % 20 == 0 or cnt == total:
            e = (time.time() - t0) / 60
            r = cnt / e if e > 0 else 0
            eta = (total - cnt) / r if r > 0 else 0
            cf_val = cf; wl_val = wl
            print(f'[{cnt}/{total}] F{fid}/{ang}deg: CF={cf_val:.3f} WL={wl_val:.3f} | {e:.0f}min ETA{eta:.0f}min')

print(f'Done: {cnt}/{total} in {(time.time()-t0)/60:.0f}min')
