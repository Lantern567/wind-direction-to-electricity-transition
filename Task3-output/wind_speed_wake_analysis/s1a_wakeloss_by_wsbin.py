"""
S1a: wakeloss_by_wsbin.csv — 每场每 ws_bin 尾流效率, 方向加权(Methods A+B combined)
也用历史风频得到 farm_ws_stats.csv (S1b)
"""
import os, sys, csv, time, tempfile, shutil, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')
import netCDF4

from floris_config import (
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    get_ti_for_farm, get_turbine_params, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master,
)
from task2_floris import precompute_wake_table, get_era5_nc_path, get_region_for_farm, extract_wind_series

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

def load_baseline_daily(lat, lon):
    """Load 1981-2010 baseline daily ERA5 and compute WSxWD histogram, plus farm stats"""
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2', 'data')
    BASELINE = {
        'east_asia': ['era5_baseline_daily_east_asia_b1981_1990.nc','era5_baseline_daily_east_asia_b1991_2000.nc','era5_baseline_daily_east_asia_b2001_2010.nc'],
        'europe': ['era5_baseline_daily_europe_b1981_1990.nc','era5_baseline_daily_europe_b1991_2000.nc','era5_baseline_daily_europe_b2001_2010.nc'],
        'us_east': ['era5_baseline_daily_us_east_b1981_1990.nc','era5_baseline_daily_us_east_b1991_2000.nc','era5_baseline_daily_us_east_b2001_2010.nc'],
    }
    region = get_region_for_farm(lat, lon)
    files = [os.path.join(DATA_DIR, f) for f in BASELINE.get(region, BASELINE['east_asia'])]
    ws_vals, wd_vals = [], []
    for fp in files:
        if not os.path.exists(fp): continue
        tmpdir = tempfile.mkdtemp(); tmpnc = os.path.join(tmpdir,'d.nc')
        shutil.copy2(fp, tmpnc)
        ds = netCDF4.Dataset(tmpnc,'r')
        ilat = int(np.argmin(np.abs(ds['latitude'][:]-lat)))
        ilon = int(np.argmin(np.abs(ds['longitude'][:]-lon)))
        u = np.array(ds['u100'][: ,ilat, ilon]); v = np.array(ds['v100'][: ,ilat, ilon])
        ws_vals.extend(np.sqrt(u**2+v**2).tolist())
        wd_vals.extend(((270-np.degrees(np.arctan2(v,u)))%360).tolist())
        ds.close(); shutil.rmtree(tmpdir)
    ws_arr = np.array(ws_vals); wd_arr = np.array(wd_vals)
    # WSxWD histogram
    WD_EDGES = np.arange(0,361,10); WS_EDGES = np.array([0,3,5,7,9,11,14,18,25,50])
    hist, _, _ = np.histogram2d(wd_arr, ws_arr, bins=[WD_EDGES, WS_EDGES])
    hist = hist / hist.sum()
    return hist, ws_arr

def compute_ws_stats_for_farm(fid, yr, lat, lon, farms):
    """S1b: farm_ws_stats from ERA5 hourly = mean, median, p10, p90, std, weibull, frac_below_rated"""
    tp = get_turbine_params(DEFAULT_TURBINE)
    H_t = tp['H']
    # Find rated wind speed
    rated_ws = None
    pw = np.array(tp['power_table']['power']); ws_pc = np.array(tp['power_table']['wind_speed'])
    for i in range(len(pw)):
        if pw[i] >= max(pw)*0.98:
            rated_ws = ws_pc[i]; break
    if rated_ws is None: rated_ws = 11.0
    region = get_region_for_farm(lat, lon)
    nc = get_era5_nc_path(region, yr)
    if nc is None: return None
    ws100, _ = extract_wind_series(nc, lat, lon)
    ws_hub = ws100 * (H_t/100)**ALPHA_DEFAULT
    from scipy.stats import weibull_min
    shape, _, scale = weibull_min.fit(ws_hub, floc=0)
    return {
        'farm_id': fid, 'year': yr,
        'ws_mean': float(np.mean(ws_hub)), 'ws_median': float(np.median(ws_hub)),
        'ws_p10': float(np.percentile(ws_hub,10)), 'ws_p90': float(np.percentile(ws_hub,90)),
        'ws_std': float(np.std(ws_hub)),
        'weibull_A': float(scale), 'weibull_k': float(shape),
        'frac_below_rated': float(np.mean(ws_hub < rated_ws)),
    }

def main():
    coords = load_task0_coordinates(); farms = load_farms_master()
    ws = np.array(WS_BINS); wd = np.array(WD_SECTORS)
    YR = 2024

    # === S1a: wakeloss_by_wsbin ===
    csv_a = os.path.join(OUT_DIR, 'wakeloss_by_wsbin.csv')
    done_a = set()
    if os.path.exists(csv_a):
        for r in csv.DictReader(open(csv_a,'r',encoding='utf-8-sig')):
            done_a.add(int(r['farm_id']))

    # === S1b: farm_ws_stats ===
    csv_b = os.path.join(OUT_DIR, 'farm_ws_stats.csv')
    done_b = set()
    if os.path.exists(csv_b):
        for r in csv.DictReader(open(csv_b,'r',encoding='utf-8-sig')):
            done_b.add((int(r['farm_id']), int(r['year'])))

    # Build task list
    tasks = []
    for fid in sorted(farms.keys()):
        turbs = coords[fid].get(YR, [])
        n = len(turbs)
        if n >= 2:
            tasks.append((fid, n))
    tasks.sort(key=lambda x: x[1])

    n_a = len([t for t in tasks if t[0] not in done_a])
    n_b = sum(1 for fid,_ in tasks for yr in range(2014,2025) if len(coords[fid].get(yr,[]))>=2 and (fid,yr) not in done_b)
    print(f'S1a: {n_a} farms to process ({n_b} farm-years for S1b)')

    # Headers
    if not os.path.exists(csv_a):
        with open(csv_a,'w',newline='',encoding='utf-8-sig') as f:
            w1 = csv.writer(f)
            w1.writerow(['farm_id','ws_bin_m_s','wd_weighted_efficiency','energy_share','wakeloss_pct'])
    if not os.path.exists(csv_b):
        with open(csv_b,'w',newline='',encoding='utf-8-sig') as f:
            csv.writer(f).writerow(['farm_id','year','ws_mean','ws_median','ws_p10','ws_p90','ws_std','weibull_A','weibull_k','frac_below_rated'])

    t0_s1a = time.time(); done_count_a = 0
    for fid, n in tasks:
        if fid in done_a:
            done_count_a += 1; continue
        info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
        ti = get_ti_for_farm(lat, lon)
        turbs = coords[fid].get(YR, [])
        xs = [float(t['x_m']) for t in turbs]; ys = [float(t['y_m']) for t in turbs]

        try:
            we, et, nt = precompute_wake_table(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws, wd, 'gauss')
            # Load historical wind distribution for this farm
            hist, ws_all = load_baseline_daily(lat, lon)
            # hist shape = (36, 9) — 36 wd sectors x 9 ws bins
            # Map baseline ws bin edges to FLORIS ws bins
            baseline_ws_edges = np.array([0,3,5,7,9,11,14,18,25,50])
            baseline_ws_centers = 0.5*(baseline_ws_edges[:-1]+baseline_ws_edges[1:])

            with open(csv_a,'a',newline='',encoding='utf-8-sig') as f:
                w1 = csv.writer(f)
                for iws, ws_val in enumerate(ws):
                    total_w = 0
                    weighted_eff = 0
                    for iwd in range(36):
                        # Find nearest baseline ws bin
                        baseline_iws = np.argmin(np.abs(baseline_ws_centers - ws_val))
                        if baseline_iws < hist.shape[1]:
                            w = hist[iwd, baseline_iws]
                            total_w += w
                            weighted_eff += w * we[iwd, iws]
                    if total_w > 0:
                        avg_eff = weighted_eff / total_w
                        wakeloss = (1.0 - avg_eff) * 100
                        energy_share = total_w  # fraction of time wind is in this ws_bin (direction-weighted)
                    else:
                        avg_eff = 1.0; wakeloss = 0.0; energy_share = 0.0
                    w1.writerow([fid, ws_val, round(float(avg_eff),6), round(float(energy_share),6), round(float(wakeloss),4)])

            done_count_a += 1
            rem_a = n_a - done_count_a
            elapsed = (time.time()-t0_s1a)/60
            rate = done_count_a/elapsed if elapsed>0 else 0
            eta = rem_a/rate if rate>0 else 0
            print(f'S1a [{done_count_a}/{n_a}] F{fid}({n}t) t={et:.0f}s ETA={eta:.0f}min')

        except Exception as e:
            print(f'S1a F{fid}: ERR {str(e)[:80]}')

    # === S1b: farm_ws_stats (run after S1a is done, single-threaded for simplicity) ===
    print(f'\nS1b: farm_ws_stats for 2014-2024...')
    t0_b = time.time(); done_b = 0
    total_b = sum(1 for fid,_ in tasks for yr in range(2014,2025) if len(coords[fid].get(yr,[]))>=2)
    for fid,_ in tasks:
        info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
        for yr in range(2014,2025):
            if (fid, yr) in done_b: continue
            if len(coords[fid].get(yr,[])) < 2: continue
            try:
                r = compute_ws_stats_for_farm(fid, yr, lat, lon, farms)
                if r:
                    with open(csv_b, 'a', newline='', encoding='utf-8-sig') as f:
                        csv.writer(f).writerow([r['farm_id'], r['year'], r['ws_mean'], r['ws_median'],
                            r['ws_p10'], r['ws_p90'], r['ws_std'], r['weibull_A'], r['weibull_k'], r['frac_below_rated']])
                    done_b += 1
                    elapsed = (time.time()-t0_b)/60
                    rate = done_b/elapsed if elapsed>0 else 0
                    eta = (total_b-done_b)/rate if rate>0 else 0
                    if done_b % 20 == 0:
                        print(f'S1b [{done_b}/{total_b}] F{fid}/{yr}: ws_mean={r["ws_mean"]:.1f} ETA={eta:.0f}min')
            except:
                pass

    print(f'\nDone: S1a={n_a} farms, S1b={done_b} farm-years')

if __name__ == '__main__':
    main()
