"""走廊TWh计算 — 全量1,446格点 FLORIS Gauss (标准8x8 5D方阵)
优化: we矩阵只算一次 (所有格点用同一标准布局), 每格点仅读ERA5+算分箱加权"""
import os, sys, csv, time, numpy as np, tempfile, shutil, netCDF4
sys.path.insert(0, r'D:/1风力发电实习/offshore-task2')
np.seterr(divide='ignore', invalid='ignore')

from floris_config import get_turbine_params, DEFAULT_TURBINE, ALPHA_DEFAULT
from task2_floris import precompute_wake_table

GRID_PATH = r'D:/1风力发电实习/wind-direction-to-electricity-transition-main/补算/output/task1_corridor_grid.csv'
ERA5_DIR = r'D:/1风力发电实习/offshore-task2/data'

# Standard 8x8 5D grid (same for all points)
ROWS, COLS = 8, 8; D_ref = 198.0; SP = 5 * D_ref
XS = [float(c) * SP for r in range(ROWS) for c in range(COLS)]
YS = [float(r) * SP for r in range(ROWS) for c in range(COLS)]
N_TURB = len(XS)

WS = np.array([0.0,3.0]+[float(x) for x in range(5,26,2)]+[30.0])  # 14 bins
WD = np.array(list(range(0,360,20)))  # 18 sectors
TP = get_turbine_params(DEFAULT_TURBINE)
H_TURB = TP['H']; RATED = float(max(TP['power_table']['power']))
CAP = N_TURB * RATED

# ===== STEP 1: Precompute wake table ONCE =====
print('Precomputing FLORIS wake table (8x8 5D standard grid, Gauss)...')
t0 = time.time()
WE, et_we, _ = precompute_wake_table(XS, YS, DEFAULT_TURBINE, 0.07, ALPHA_DEFAULT, WS, WD, 'gauss')
print(f'  Done in {et_we:.0f}s. WE shape={WE.shape} ({len(WD)}WD x {len(WS)}WS)')

# ===== STEP 2: Load grid and process =====
grid_rows = list(csv.DictReader(open(GRID_PATH, 'r', encoding='utf-8-sig')))
TOTAL = len(grid_rows)
print(f'Processing {TOTAL} grid points...')

# Power curve for no-wake bin power
ws_pc = np.array(TP['power_table']['wind_speed'])
pw_pc = np.array(TP['power_table']['power'])
pn_bin = np.array([float(np.interp(w, ws_pc, pw_pc, left=0, right=0)) * N_TURB for w in WS])

def era5_hist(lat, lon):
    """Build WSxWD histogram for a grid point"""
    if 8 <= lat <= 44 and 104 <= lon <= 143: nc_name = 'era5_east_asia_2024.nc'
    elif 39 <= lat <= 63 and -12 <= lon <= 32: nc_name = 'era5_europe_2024.nc'
    elif 36 <= lat <= 42 and -78 <= lon <= -68: nc_name = 'era5_us_east_2024.nc'
    else: nc_name = 'era5_east_asia_2024.nc'
    fp = os.path.join(ERA5_DIR, nc_name)
    if not os.path.exists(fp): return None
    tmpdir = tempfile.mkdtemp(); tmpnc = os.path.join(tmpdir, 'd.nc')
    shutil.copy2(fp, tmpnc)
    ds = netCDF4.Dataset(tmpnc, 'r')
    ilat = int(np.argmin(np.abs(ds['latitude'][:] - lat)))
    ilon = int(np.argmin(np.abs(ds['longitude'][:] - lon)))
    u = np.array(ds['u100'][:, ilat, ilon]); v = np.array(ds['v100'][:, ilat, ilon])
    ws100 = np.sqrt(u**2+v**2); wd_deg = (270-np.degrees(np.arctan2(v,u)))%360
    ds.close(); shutil.rmtree(tmpdir)
    ws_hub = ws100 * (H_TURB/100.0)**ALPHA_DEFAULT
    hist = np.zeros((len(WD), len(WS)))
    for h in range(len(ws_hub)):
        iws = np.argmin(np.abs(WS-ws_hub[h]))
        iwd = np.argmin(np.abs(WD-wd_deg[h]))
        hist[iwd, iws] += 1.0
    hist /= hist.sum()
    return hist

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'corridor_aep_validation.csv')
with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerow(['lon', 'lat', 'A_pred_pct', 'AEP_kWh', 'CF', 'WakeLoss'])

t_start = time.time(); failed = 0
for i, r in enumerate(grid_rows):
    lon, lat = float(r['lon']), float(r['lat']); a_pred = float(r['A_pred_pct'])
    hist = era5_hist(lat, lon)
    if hist is None: failed += 1; continue

    # Weight AEP using same WE x histogram
    aep = sum(float(pn_bin[iws]) * float(WE[iwd,iws]) * float(hist[iwd,iws]) for iwd in range(len(WD)) for iws in range(len(WS))) * 8760.0
    aep_nw = sum(float(pn_bin[iws]) * float(hist[iwd,iws]) for iwd in range(len(WD)) for iws in range(len(WS))) * 8760.0
    cf = aep/(CAP*8760.0) if CAP>0 else 0.0
    wl = (aep_nw-aep)/aep_nw if aep_nw>0 else 0.0

    with open(OUT, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow([lon, lat, a_pred, aep, cf, wl])

    if (i+1) % 200 == 0:
        e = (time.time()-t_start)/60; r = (i+1)/e if e>0 else 0
        print(f'  [{i+1}/{TOTAL}] lat={lat:.0f} lon={lon:.0f} CF={cf:.3f} | {e:.0f}min ETA{(TOTAL-i-1)/r:.0f}min')

e = (time.time()-t_start)/60
print(f'Done: {TOTAL-failed}/{TOTAL} ({failed} failed) in {e:.0f}min')
print(f'Output: {OUT}')
