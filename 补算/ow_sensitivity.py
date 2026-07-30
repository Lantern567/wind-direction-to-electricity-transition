"""
补算3: F57/F66/F91 真实机型敏感性
对三个头条场用 ow_6MW / ow_8MW 重跑旋转扫描, 与 IEA 10MW 对比增益幅度
"""
import os, sys, csv, time, numpy as np
sys.path.insert(0, r'D:/1风力发电实习/offshore-task2')
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS_JENSEN as WS_BINS, WD_SECTORS_JENSEN as WD_SECTORS,
    ALPHA_DEFAULT, get_ti_for_farm, load_task0_coordinates, load_farms_master,
)
from task2_floris import precompute_wake_table

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ow_sensitivity_3farms.csv')
coords = load_task0_coordinates(); farms = load_farms_master()
ws = np.array(WS_BINS); wd = np.array(WD_SECTORS)
YR = 2024; ANGLES = list(range(0, 180, 10))
FARMS = [57, 66, 91]  # Vietnam, Hangzhou Bay, Pearl River Delta
TURBINES = ['iea_10MW', 'ow_6MW', 'ow_8MW']

done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT,'r',encoding='utf-8-sig')):
        done.add((int(r['farm_id']), int(r['angle_deg']), r['turbine_type']))

with open(OUT, 'w' if not os.path.exists(OUT) else 'a', newline='', encoding='utf-8-sig') as f:
    if not os.path.exists(OUT) or os.path.getsize(OUT)==0:
        csv.writer(f).writerow(['farm_id','angle_deg','turbine_type','n_turb','country',
                               'AEP_kWh','AEP_noWake_kWh','CF','WakeLoss','precompute_s'])

total = len(FARMS) * len(ANGLES) * len(TURBINES)
t0 = time.time(); cnt = 0

for fid in FARMS:
    turbs = coords[fid].get(YR, []); n = len(turbs)
    if n < 2: continue
    info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
    ti = get_ti_for_farm(lat, lon); country = info.get('country', '?')
    xs_orig = np.array([float(t['x_m']) for t in turbs], dtype=np.float64)
    ys_orig = np.array([float(t['y_m']) for t in turbs], dtype=np.float64)
    cx, cy = np.mean(xs_orig), np.mean(ys_orig)

    for ang in ANGLES:
        theta = np.radians(ang); ct, st = np.cos(theta), np.sin(theta)
        xr = (xs_orig - cx) * ct - (ys_orig - cy) * st + cx
        yr = (xs_orig - cx) * st + (ys_orig - cy) * ct + cy

        for tt in TURBINES:
            if (fid, ang, tt) in done: continue
            from floris_config import get_turbine_params
            tp = get_turbine_params(tt)
            we, et, nt = precompute_wake_table([float(v) for v in xr], [float(v) for v in yr],
                                               tt, ti, ALPHA_DEFAULT, ws, wd, 'gauss')
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
                csv.writer(f).writerow([fid, ang, tt, n, country, aep, aep_nw, cf, wl, et])

            cnt += 1
            if cnt % 10 == 0 or cnt == total:
                e = (time.time() - t0) / 60
                r = cnt / e if e > 0 else 0
                eta = (total - cnt) / r if r > 0 else 0
                print(f'[{cnt}/{total}] F{fid}/{ang}deg/{tt}: CF={cf:.3f} WL={wl:.3f} | {e:.0f}min ETA{eta:.0f}min')

print(f'Done: {cnt}/{total} in {(time.time()-t0)/60:.0f}min')
