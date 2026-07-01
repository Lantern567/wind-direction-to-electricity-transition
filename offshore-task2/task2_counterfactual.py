"""
=============================================================================
 任务二 §2.8 风向变化反事实分析 — v2.0
 基准期: 1981-2010 ERA5 逐日12:00UTC → 构建逐月16扇区风向频率分布
 情景A: 真实风速 + 真实风向 → AEP_real, WakeLoss_real
 情景B: 真实风速 + 从基准期分布采样的风向 → AEP_baseWD, WakeLoss_baseWD
 边算边写, 可续传, 按年分批打开NC
=============================================================================
"""
import os, math, time, csv, tempfile, shutil
import numpy as np
from collections import defaultdict
from datetime import datetime
from numba import njit
import netCDF4
from netCDF4 import num2date

TASK0_DIR = r"D:\1风力发电实习\offshore-task0\output\task0"
DATA_DIR  = r"D:\1风力发电实习\offshore-task2\data"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

ANALYSIS_YRS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
H_REF = 100.0; ALPHA = 0.11; K_WAKE = 0.05; WAKE_LIMIT = 90.0
ELECTRICAL_LOSS = 0.95 * 0.97
N_SECTORS = 16

# ============================================================
# 1. LOAD TASK0 + MODEL
# ============================================================
def load_task0_coords():
    coords = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(TASK0_DIR, "turbine_coordinates.csv"), 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id']); yr = int(r['year']); tid = int(r['turbine_id'])
            coords[fid][yr].append({'x_m': float(r['x_m']), 'y_m': float(r['y_m']),
                                     'lon': float(r['lon']), 'lat': float(r['lat'])})
    return coords

def load_farms_master():
    farms = {}
    with open(os.path.join(TASK0_DIR, "farms_master.csv"), 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id'])
            farms[fid] = {'centroid_lat': float(r['centroid_lat']), 'centroid_lon': float(r['centroid_lon'])}
    return farms

def load_turbine_model():
    import yaml
    with open(os.path.join(DATA_DIR, "iea_10MW.yaml"), 'r', encoding='utf-8') as f:
        c = yaml.safe_load(f)
    pt = c['power_thrust_table']
    return {'D': c['rotor_diameter'], 'H': c['hub_height'],
            'Pr': pt['controller_dependent_turbine_parameters']['rated_power'],
            'ws': np.array(pt['wind_speed'], dtype=np.float64),
            'power': np.array(pt['power'], dtype=np.float64),
            'ct': np.array(pt['thrust_coefficient'], dtype=np.float64)}

# ============================================================
# 2. NUMBA WAKE
# ============================================================
@njit
def wake_gauss(pos, wdr, ws, D, wsn, ctn, k=0.05, al=90.0):
    n = len(pos)
    if n <= 1: return ws
    wx, wy = math.cos(wdr), math.sin(wdr); proj = pos[:,0]*wx + pos[:,1]*wy
    order = np.argsort(proj); ve = ws.copy()
    for ii in range(1, n):
        i = order[ii]; losses = 0.0
        for jj in range(ii):
            j = order[jj]; dx = (proj[i] - proj[j]) * 1000.0
            if dx <= 0: continue
            dy = abs((pos[i,0]-pos[j,0])*wy - (pos[i,1]-pos[j,1])*wx) * 1000.0
            if math.degrees(math.atan2(dy, dx)) > al: continue
            ct = np.interp(ve[j], wsn, ctn)
            if ct <= 0 or ct >= 1: continue
            sigma = k*dx + 0.1414213562373095*D
            deficit = (1.0 - math.sqrt(1.0 - ct))*math.exp(-0.5*(dy/sigma)**2)*(D/(D + 2.0*k*dx))**2
            if deficit > 0: losses += deficit*deficit
        if losses > 0: ve[i] = ws[i] * max(0.0, 1.0 - math.sqrt(losses))
    return ve

@njit
def power_nb(ws_arr, wsn, pwn):
    out = np.zeros_like(ws_arr)
    for i in range(len(ws_arr)):
        v = ws_arr[i]
        if v < wsn[1] or v >= wsn[-2]: out[i] = 0.0
        else: out[i] = np.interp(v, wsn, pwn)
    return out

# ============================================================
# 3. BUILD BASELINE WD DISTRIBUTION FROM 1981-2010 DAILY DATA
# ============================================================
class ERA5Reader:
    def __init__(self, path): self.path = path
    def __enter__(self):
        self.tmp = tempfile.mkdtemp(); self.tmpnc = os.path.join(self.tmp, "data.nc")
        shutil.copy2(self.path, self.tmpnc); self.ds = netCDF4.Dataset(self.tmpnc, 'r')
        return self
    def __exit__(self, *a):
        if hasattr(self,'ds'): self.ds.close()
        if hasattr(self,'tmp'): shutil.rmtree(self.tmp)
    def times(self):
        vt = self.ds['valid_time'] if 'valid_time' in self.ds.variables else self.ds['time']
        return num2date(vt[:], units=vt.units)
    def wind_at(self, lat, lon):
        la = self.ds['latitude'][:]; lo = self.ds['longitude'][:]
        ila = int(np.argmin(np.abs(la-lat))); ilo = int(np.argmin(np.abs(lo-lon)))
        return (np.array(self.ds['u100'][:,ila,ilo]), np.array(self.ds['v100'][:,ila,ilo]))

def build_baseline_wd_dist(farms):
    """For each farm, build monthly 16-sector WD frequency from daily baseline 1981-2010"""
    baseline_nc = {}
    for bname in ['b1981_1990', 'b1991_2000', 'b2001_2010']:
        for rkey in ['east_asia', 'europe', 'us_east']:
            path = os.path.join(DATA_DIR, f"era5_baseline_daily_{rkey}_{bname}.nc")
            if os.path.exists(path):
                baseline_nc[(rkey, bname)] = path

    # Map farm→region
    farm_region = {}
    for fid, info in farms.items():
        clat, clon = info['centroid_lat'], info['centroid_lon']
        if 8 <= clat <= 44 and 104 <= clon <= 143: farm_region[fid] = 'east_asia'
        elif 39 <= clat <= 63 and -12 <= clon <= 32: farm_region[fid] = 'europe'
        else: farm_region[fid] = 'us_east'

    # Pre-extract sectors per farm
    se = np.linspace(0, 360, N_SECTORS+1)
    farm_monthly_sectors = {}

    for rkey in ['east_asia', 'europe', 'us_east']:
        region_farms = [fid for fid, r in farm_region.items() if r == rkey]
        if not region_farms: continue

        # Merge all 3 decades for this region
        for (rk, bname), path in sorted(baseline_nc.items()):
            if rk != rkey: continue
            with ERA5Reader(path) as era5:
                vt = era5.ds['valid_time']
                times = num2date(vt[:], units=vt.units)
                months = np.array([t.month for t in times])
                for fid in region_farms:
                    clat, clon = farms[fid]['centroid_lat'], farms[fid]['centroid_lon']
                    u, v = era5.wind_at(clat, clon)
                    ws = np.sqrt(u*u + v*v)

                    if fid not in farm_monthly_sectors:
                        farm_monthly_sectors[fid] = {m: np.zeros(N_SECTORS) for m in range(1,13)}

                    for m in range(1, 13):
                        idx = (months == m) & (ws >= 3.0)
                        if idx.sum() == 0: continue
                        wd = (np.degrees(np.arctan2(u[idx], v[idx])) + 180) % 360
                        s = np.clip((wd / (360/N_SECTORS)).astype(np.int32), 0, N_SECTORS-1)
                        for si in s: farm_monthly_sectors[fid][m][si] += 1

    # Normalize to probabilities
    for fid in farm_monthly_sectors:
        for m in range(1, 13):
            total = farm_monthly_sectors[fid][m].sum()
            if total > 0: farm_monthly_sectors[fid][m] /= total
            else: farm_monthly_sectors[fid][m] = np.ones(N_SECTORS) / N_SECTORS

    print(f"  基准风向覆盖: {len(farm_monthly_sectors)}/{len(farms)} farms")
    return farm_monthly_sectors, se

def sample_baseline_wd(month, wd_dist, se, seed_base):
    """Sample a wind direction from the baseline monthly distribution"""
    np.random.seed(seed_base + month * 1000)  # deterministic per month
    probs = wd_dist[month]
    si = np.random.choice(N_SECTORS, p=probs)
    return (se[si] + se[si+1]) / 2 + np.random.uniform(-5, 5)

# ============================================================
# 4. MAIN
# ============================================================
def main():
    t0 = datetime.now()
    print("任务二 §2.8 反事实分析 v2.0 — 真实逐日基准期风向分布")
    print(f"基准期: 1981-2010 (ERA5 daily 12:00UTC), 检验期: {ANALYSIS_YRS}")
    print("="*60)

    print("[1/3] 加载底座 + 构建基准风向分布...")
    coords = load_task0_coords()
    farms = load_farms_master()
    model = load_turbine_model()
    wd_dist, se = build_baseline_wd_dist(farms)

    # Warm Numba
    tp = np.random.rand(10,2)*1000; tw = np.full(10, 8.0)
    _ = wake_gauss(tp, math.radians(180), tw, model['D'], model['ws'], model['ct'])
    print("  Numba OK\n")

    # Build farm→region map for ERA5 main files
    farm_region = {}
    for fid, info in farms.items():
        clat, clon = info['centroid_lat'], info['centroid_lon']
        if 8 <= clat <= 44 and 104 <= clon <= 143: farm_region[fid] = 'east_asia'
        elif 39 <= clat <= 63 and -12 <= clon <= 32: farm_region[fid] = 'europe'
        else: farm_region[fid] = 'us_east'

    CSV = os.path.join(OUT_DIR, "task2_counterfactual.csv")
    completed = set()
    if os.path.exists(CSV):
        with open(CSV, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed.add((int(r['farm_id']), int(r['year'])))

    # Build task list per region per year (same batch strategy as task2_core)
    tasks_by_ry = defaultdict(lambda: defaultdict(list))
    total_pending = 0
    for fid in sorted(farms.keys()):
        if fid not in wd_dist: continue
        for yr in ANALYSIS_YRS:
            if (fid, yr) in completed: continue
            yr_turbs = coords.get(fid, {}).get(yr, [])
            if len(yr_turbs) >= 2:
                rkey = farm_region[fid]
                tasks_by_ry[rkey][yr].append((fid, yr, yr_turbs))
                total_pending += 1

    print(f"[2/3] 反事实: {len(completed)} 已完成, {total_pending} 待算")

    if total_pending == 0:
        print("全部已完成!")
        return

    fout = open(CSV, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(fout)
    if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
        w.writerow(['farm_id','year','AEP_real_kWh','AEP_baseWD_kWh','Delta_AEP_WD_kWh',
                    'WakeLoss_real','WakeLoss_baseWD','Delta_WakeLoss_WD'])

    done = 0

    for rkey in ['east_asia', 'europe', 'us_east']:
        for yr in ANALYSIS_YRS:
            task_list = tasks_by_ry.get(rkey, {}).get(yr, [])
            if not task_list: continue

            nc_path = os.path.join(DATA_DIR, f"era5_{rkey}_{yr}.nc")
            jp_path = os.path.join(DATA_DIR, f"era5_japan_{yr}.nc")
            if not os.path.exists(nc_path): continue

            # Open NC once for this region-year
            with ERA5Reader(nc_path) as era5:
                all_times = era5.times()

                # Pre-extract wind for all farms in this batch
                wind_cache = {}
                for fid, yr_f, turbs in task_list:
                    clat, clon = farms[fid]['centroid_lat'], farms[fid]['centroid_lon']
                    cache_key = (clat, clon)
                    if cache_key not in wind_cache:
                        u100, v100 = era5.wind_at(clat, clon)
                        ws_raw = np.sqrt(u100**2 + v100**2)
                        wd_raw = (np.degrees(np.arctan2(u100, v100)) + 180) % 360
                        mask = ws_raw >= 3.0
                        idx = np.where(mask)[0]
                        if len(idx) == 0:
                            wind_cache[cache_key] = None; continue
                        months_arr = np.array([all_times[i].month for i in idx])
                        wind_cache[cache_key] = (
                            ws_raw[idx] * (model['H'] / H_REF) ** ALPHA,
                            wd_raw[idx],
                            months_arr)

                for fid, yr_f, turbs in task_list:
                    clat, clon = farms[fid]['centroid_lat'], farms[fid]['centroid_lon']
                    cache_key = (clat, clon)
                    wind_data = wind_cache.get(cache_key)
                    if wind_data is None: continue
                    V_arr, th_real, months_arr = wind_data

                    n_turb = len(turbs); H, D, Pr = model['H'], model['D'], model['Pr']
                    wsn, pwn, ctn = model['ws'], model['power'], model['ct']
                    pos = np.array([[t['x_m']/1000.0, t['y_m']/1000.0] for t in turbs], dtype=np.float64)
                    nh = len(V_arr)

                    # Sample baseline direction for each hour
                    th_base = np.zeros(nh)
                    for s in range(nh):
                        m = months_arr[s]
                        th_base[s] = sample_baseline_wd(m, wd_dist[fid], se, fid*10000 + yr*100 + s)

                    pn_sum = 0.0; pr_sum = 0.0; pb_sum = 0.0
                    for s in range(nh):
                        ws = np.full(n_turb, V_arr[s])
                        pn_hour = float(power_nb(ws, wsn, pwn).sum()); pn_sum += pn_hour
                        ver = wake_gauss(pos, math.radians(270.0-th_real[s]), ws, D, wsn, ctn, K_WAKE, WAKE_LIMIT)
                        pr_sum += float(power_nb(ver, wsn, pwn).sum())
                        veb = wake_gauss(pos, math.radians(270.0-th_base[s]), ws, D, wsn, ctn, K_WAKE, WAKE_LIMIT)
                        pb_sum += float(power_nb(veb, wsn, pwn).sum())

                    wl_real = (pn_sum-pr_sum)/pn_sum if pn_sum>0 else 0
                    wl_base = (pn_sum-pb_sum)/pn_sum if pn_sum>0 else 0
                    w.writerow([fid, yr,
                               f"{pr_sum*ELECTRICAL_LOSS:.1f}", f"{pb_sum*ELECTRICAL_LOSS:.1f}",
                               f"{(pr_sum-pb_sum)*ELECTRICAL_LOSS:.1f}",
                               f"{wl_real:.4f}", f"{wl_base:.4f}", f"{wl_real-wl_base:.4f}"])
                    done += 1

            fout.flush()
            if done > 0:
                pct = done*100//total_pending
                print(f"  [{datetime.now().strftime('%H:%M')}] {done}/{total_pending} ({pct}%)")

    fout.close()
    elapsed = (datetime.now()-t0).total_seconds()/60
    print(f"\n耗时: {elapsed:.1f} min | 输出: {CSV}")
    print("完成!")

if __name__ == "__main__":
    main()
