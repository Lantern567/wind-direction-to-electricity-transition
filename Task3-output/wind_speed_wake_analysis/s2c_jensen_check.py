"""
S2c: Jensen robustness check — 10 farms, Gauss vs Jensen wake-loss by ws_bin
"""
import sys, os, csv, time, numpy as np
sys.path.insert(0, r'D:/1风力发电实习/offshore-task2')
np.seterr(divide='ignore', invalid='ignore')
import tempfile, shutil, netCDF4

from floris_config import (
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, get_ti_for_farm,
    DEFAULT_TURBINE, load_task0_coordinates, load_farms_master,
)
from task2_floris import precompute_wake_table, get_region_for_farm

DATA_DIR = os.path.join(r'D:/1风力发电实习/offshore-task2', 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

def load_baseline_daily(lat, lon):
    BASELINE = {
        'east_asia': ['era5_baseline_daily_east_asia_b1981_1990.nc','era5_baseline_daily_east_asia_b1991_2000.nc','era5_baseline_daily_east_asia_b2001_2010.nc'],
        'europe': ['era5_baseline_daily_europe_b1981_1990.nc','era5_baseline_daily_europe_b1991_2000.nc','era5_baseline_daily_europe_b2001_2010.nc'],
        'us_east': ['era5_baseline_daily_us_east_b1981_1990.nc','era5_baseline_daily_us_east_b1991_2000.nc','era5_baseline_daily_us_east_b2001_2010.nc'],
    }
    region = get_region_for_farm(lat, lon)
    files = [os.path.join(DATA_DIR,f) for f in BASELINE.get(region, BASELINE['east_asia'])]
    ws_vals, wd_vals = [], []
    for fp in files:
        if not os.path.exists(fp): continue
        tmpdir = tempfile.mkdtemp(); tmpnc = os.path.join(tmpdir,'d.nc')
        shutil.copy2(fp, tmpnc)
        ds = netCDF4.Dataset(tmpnc,'r')
        ilat = int(np.argmin(np.abs(ds['latitude'][:]-lat)))
        ilon = int(np.argmin(np.abs(ds['longitude'][:]-lon)))
        u = np.array(ds['u100'][:,ilat,ilon]); v = np.array(ds['v100'][:,ilat,ilon])
        ws_vals.extend(np.sqrt(u**2+v**2).tolist()); wd_vals.extend(((270-np.degrees(np.arctan2(v,u)))%360).tolist())
        ds.close(); shutil.rmtree(tmpdir)
    hist, _, _ = np.histogram2d(np.array(wd_vals), np.array(ws_vals),
                                bins=[np.arange(0,361,10), np.array([0,3,5,7,9,11,14,18,25,50])])
    return hist / hist.sum()

def main():
    coords = load_task0_coordinates(); farms = load_farms_master()
    ws = np.array(WS_BINS); wd = np.array(WD_SECTORS); YR = 2024
    JENSEN_FARMS = [162,153,151,112,107,91,83,73,21,4]

    csv_out = os.path.join(OUT_DIR, 's2c_jensen_check.csv')
    with open(csv_out,'w',newline='',encoding='utf-8-sig') as f:
        csv.writer(f).writerow(['farm_id','ws_bin_m_s','wakeloss_pct','model'])

    baseline_ws_centers = np.array([1.5,4,6,8,10,12.5,16,21.5,37.5])
    t0 = time.time()

    for i, fid in enumerate(JENSEN_FARMS):
        turbs = coords[fid].get(YR,[]); n = len(turbs)
        if n < 2: continue
        info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
        ti = get_ti_for_farm(lat, lon)
        xs = [float(t['x_m']) for t in turbs]; ys = [float(t['y_m']) for t in turbs]
        hist = load_baseline_daily(lat, lon)

        for model in ['gauss','jensen']:
            we, et, nt = precompute_wake_table(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws, wd, model)
            with open(csv_out,'a',newline='',encoding='utf-8-sig') as f:
                wtr = csv.writer(f)
                for iws, ws_val in enumerate(ws):
                    tw = 0; weff = 0.0
                    for iwd in range(36):
                        biws = np.argmin(np.abs(baseline_ws_centers - ws_val))
                        if biws < hist.shape[1]:
                            wv = hist[iwd, biws]; tw += wv; weff += wv*we[iwd,iws]
                    avg_eff = weff/tw if tw>0 else 1.0
                    wtr.writerow([fid, ws_val, round(float((1.0-avg_eff)*100),4), model])

        # Correlation check
        gauss_wl = [float(r['wakeloss_pct']) for r in csv.DictReader(open(csv_out,'r',encoding='utf-8-sig'))
                     if int(r['farm_id'])==fid and r['model']=='gauss' and float(r['ws_bin_m_s'])>=5]
        jensen_wl = [float(r['wakeloss_pct']) for r in csv.DictReader(open(csv_out,'r',encoding='utf-8-sig'))
                      if int(r['farm_id'])==fid and r['model']=='jensen' and float(r['ws_bin_m_s'])>=5]
        if gauss_wl and jensen_wl:
            corr = np.corrcoef(gauss_wl, jensen_wl)[0,1]
            print(f'F{fid}: Gauss-Jensen r={corr:.3f} ({n}t)')

    print(f'Done: {csv_out}')

if __name__ == '__main__':
    main()
