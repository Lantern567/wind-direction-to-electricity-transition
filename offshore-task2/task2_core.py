"""
=============================================================================
 任务二 逐时尾流出力核算 — v1.0
 严格使用任务零统一底座 (farms_master + turbine_coordinates + turbine_params)
 Numba JIT Gaussian wake + 逐台真实坐标 + 边算边写 + 可续传
=============================================================================
"""
import os, math, time, csv, tempfile, shutil
import numpy as np
from collections import defaultdict
from datetime import datetime
from numba import njit
import netCDF4
from netCDF4 import num2date

# ============================================================
# PATH CONFIG — 全部引用任务零
# ============================================================
TASK0_DIR = r"D:\1风力发电实习\offshore-task0\output\task0"
TASK1_DIR = r"D:\1风力发电实习\task1_output"
DATA_DIR  = r"D:\1风力发电实习\offshore-task2\data"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 物理参数
H_REF = 100.0; ALPHA = 0.11; K_WAKE = 0.05; WAKE_LIMIT = 90.0

# 电气损耗系数 (任务书 §2.11 / 修改意见 §4.2)
AVAILABILITY = 0.95
COLLECTION_EFF = 0.97
ELECTRICAL_LOSS = AVAILABILITY * COLLECTION_EFF  # ≈ 0.92

ALL_YRS = list(range(2014, 2025))
BASELINE_YRS = [2014, 2015, 2016, 2017]  # 临时基准期，待月度气候态补
ANALYSIS_YRS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ============================================================
# 1. LOAD TASK0 BASE — 不自行聚类，不自行定义 farm_id
# ============================================================
def load_task0_base():
    """Import turbine coordinates, farm master, and turbine params from task0"""
    # turbine_coordinates: farm_id, year, turbine_id, lon, lat, x_m, y_m, utm_epsg
    coords = defaultdict(lambda: defaultdict(list))
    coord_path = os.path.join(TASK0_DIR, "turbine_coordinates.csv")
    with open(coord_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id']); yr = int(r['year']); tid = int(r['turbine_id'])
            coords[fid][yr].append({
                'turbine_id': tid,
                'x_m': float(r['x_m']), 'y_m': float(r['y_m']),
                'lon': float(r['lon']), 'lat': float(r['lat']),
                'utm_epsg': r['utm_epsg']
            })

    # farms_master: farm_id, n_turb, centroid_lon, centroid_lat, area_km2, capacity_kW, country
    farms = {}
    fm_path = os.path.join(TASK0_DIR, "farms_master.csv")
    with open(fm_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id'])
            farms[fid] = {
                'n_turb': int(r['n_turb']), 'country': r['country'],
                'centroid_lon': float(r['centroid_lon']), 'centroid_lat': float(r['centroid_lat']),
                'capacity_kW': int(r['capacity_kW']), 'area_km2': float(r['area_km2']),
            }

    # turbine_params: turbine_id, farm_id, turbine_model (all iea_10MW)
    tp_path = os.path.join(TASK0_DIR, "turbine_params.csv")
    # All turbines are iea_10MW — load once
    return coords, farms

# ============================================================
# 2. NUMBA WAKE + POWER (same validated engine as task1 v3)
# ============================================================
@njit
def wake_gauss(pos, wdr, ws, D, wsn, ctn, k=0.05, al=90.0):
    n = len(pos)
    if n <= 1: return ws
    wx, wy = math.cos(wdr), math.sin(wdr)
    proj = pos[:, 0]*wx + pos[:, 1]*wy
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
            deficit = (1.0 - math.sqrt(1.0 - ct)) * math.exp(-0.5*(dy/sigma)**2)
            deficit *= (D/(D + 2.0*k*dx))**2
            if deficit > 0: losses += deficit*deficit
        if losses > 0: ve[i] = ws[i] * max(0.0, 1.0 - math.sqrt(losses))
    return ve

@njit
def wake_jensen(pos, wdr, ws, D, wsn, ctn, k=0.075, al=90.0):
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
            r_wake = k*dx + D/2
            if abs(dy) >= r_wake: continue
            if math.degrees(math.atan2(dy, dx)) > al: continue
            ct = np.interp(ve[j], wsn, ctn)
            if ct <= 0: continue
            deficit = (1.0 - math.sqrt(1.0 - ct)) * (D/(D + 2.0*k*dx))**2
            if deficit > 0: losses += deficit*deficit
        if losses > 0: ve[i] = ws[i] * max(0.0, 1.0 - math.sqrt(losses))
    return ve

@njit
def wake_curl(pos, wdr, ws, D, wsn, ctn, k=0.05, al=90.0):
    n = len(pos)
    if n <= 1: return ws
    wx, wy = math.cos(wdr), math.sin(wdr); proj = pos[:,0]*wx + pos[:,1]*wy
    order = np.argsort(proj); ve = ws.copy()
    for ii in range(1, n):
        i = order[ii]; losses = 0.0
        for jj in range(ii):
            j = order[jj]; dx = (proj[i] - proj[j]) * 1000.0
            if dx <= 0: continue
            dy_raw = abs((pos[i,0]-pos[j,0])*wy - (pos[i,1]-pos[j,1])*wx) * 1000.0
            dy = abs(dy_raw - 0.02*dx)
            if math.degrees(math.atan2(dy, dx)) > al: continue
            ct = np.interp(ve[j], wsn, ctn)
            if ct <= 0 or ct >= 1: continue
            sigma = k*dx + 0.1767766952966369*D
            deficit = (1.0 - math.sqrt(1.0 - ct)) * math.exp(-0.5*(dy/sigma)**2)*(D/(D + 2.5*k*dx))**2
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
# 3. TURBINE MODEL (unified IEA 10MW from task0 caliber_config)
# ============================================================
def load_turbine_model():
    import yaml
    with open(os.path.join(DATA_DIR, "iea_10MW.yaml"), 'r', encoding='utf-8') as f:
        c = yaml.safe_load(f)
    pt = c['power_thrust_table']
    return {
        'D': c['rotor_diameter'], 'H': c['hub_height'],
        'Pr': pt['controller_dependent_turbine_parameters']['rated_power'],
        'ws': np.array(pt['wind_speed'], dtype=np.float64),
        'power': np.array(pt['power'], dtype=np.float64),
        'ct': np.array(pt['thrust_coefficient'], dtype=np.float64),
    }

# ============================================================
# 4. ERA5 READER — per-farm-year wind extraction
# ============================================================
class ERA5Reader:
    """读 NC 文件（临时目录方案，解决 Windows 中文路径问题）"""
    def __init__(self, path): self.path = path
    def __enter__(self):
        self.tmp = tempfile.mkdtemp(); self.tmpnc = os.path.join(self.tmp, "data.nc")
        shutil.copy2(self.path, self.tmpnc); self.ds = netCDF4.Dataset(self.tmpnc, 'r')
        return self
    def __exit__(self, *a):
        if hasattr(self, 'ds'): self.ds.close()
        if hasattr(self, 'tmp'): shutil.rmtree(self.tmp)
    def times(self): return num2date(self.ds['valid_time'][:], units=self.ds['valid_time'].units)
    def wind_at(self, lat, lon):
        la = self.ds['latitude'][:]; lo = self.ds['longitude'][:]
        ila = int(np.argmin(np.abs(la - lat))); ilo = int(np.argmin(np.abs(lo - lon)))
        return (np.array(self.ds['u100'][:, ila, ilo]),
                np.array(self.ds['v100'][:, ila, ilo]))

def get_nc_path(rkey, yr):
    """Map region+year to ERA5 NC file path"""
    # Attempt east_asia, europe, us_east, japan
    paths = [
        os.path.join(DATA_DIR, f"era5_{rkey}_{yr}.nc"),
    ]
    # Japan files use same filename pattern
    for p in paths:
        if os.path.exists(p): return p
    return None

def get_region_for_farm(centroid_lat, centroid_lon):
    """Determine which ERA5 region a farm belongs to"""
    if 8 <= centroid_lat <= 44 and 104 <= centroid_lon <= 143: return "east_asia"
    if 39 <= centroid_lat <= 63 and -12 <= centroid_lon <= 32: return "europe"
    if 36 <= centroid_lat <= 42 and -78 <= centroid_lon <= -68: return "us_east"
    return "east_asia"  # fallback

# ============================================================
# 5. MAIN COMPUTATION LOOP
# ============================================================
def compute_all_models(fid, yr, turbs_list, model, V_arr, th_arr, output_hourly=False):
    """Compute one farm-year with ALL THREE wake models. Returns dict of dicts keyed by model name.
    If output_hourly=True, also returns hourly arrays for each model."""
    n_turb = len(turbs_list); cap = n_turb * model['Pr']
    H, D = model['H'], model['D']
    wsn, pwn, ctn = model['ws'], model['power'], model['ct']
    pos = np.array([[t['x_m']/1000.0, t['y_m']/1000.0] for t in turbs_list], dtype=np.float64)
    nh = len(V_arr)
    if nh == 0: return None

    wakes = [('gaussian', wake_gauss, K_WAKE),
             ('jensen', wake_jensen, 0.075),
             ('curl', wake_curl, K_WAKE)]

    pn_sum = 0.0
    pw_sums = {wn: 0.0 for wn, _, _ in wakes}
    pw_hourly = {wn: [] for wn, _, _ in wakes}

    for s in range(nh):
        ws = np.full(n_turb, V_arr[s])
        pn_sum += float(power_nb(ws, wsn, pwn).sum())
        wdr = math.radians(270.0 - th_arr[s])
        for wname, wfunc, wk in wakes:
            ve = wfunc(pos, wdr, ws, D, wsn, ctn, wk, WAKE_LIMIT)
            pw = float(power_nb(ve, wsn, pwn).sum())
            pw_sums[wname] += pw
            pw_hourly[wname].append(pw)  # always collect for volatility

    results = {}
    for wname, _, _ in wakes:
        pw_arr = np.array(pw_hourly[wname])
        pw_sum = pw_sums[wname]
        AEP = pw_sum * ELECTRICAL_LOSS
        WL = (pn_sum - pw_sum)/pn_sum if pn_sum > 0 else 0
        CF = AEP/(cap*nh) if cap > 0 else 0
        results[wname] = {
            'farm_id': fid, 'year': yr, 'n_turb': n_turb, 'capacity_kW': cap, 'n_hours': nh,
            'AEP_kWh': AEP, 'CF': CF, 'WakeLoss': WL,
            'Volatility_kW': float(np.std(pw_arr*ELECTRICAL_LOSS)),
            'CV': float(np.std(pw_arr)/float(np.mean(pw_arr))) if float(np.mean(pw_arr))>0 else 0,
            'P5_kW': float(np.percentile(pw_arr,5))*ELECTRICAL_LOSS,
            'P95_kW': float(np.percentile(pw_arr,95))*ELECTRICAL_LOSS,
            'RampFreq': float(np.mean(np.abs(np.diff(pw_arr))>0.2*cap)) if len(pw_arr)>1 else 0,
            'low_hours': float(np.mean(pw_arr<0.1*cap)),
            'high_hours': float(np.mean(pw_arr>0.9*cap)),
        }
    if output_hourly:
        return results, pn_sum, pw_hourly, pos
    return results

# ============================================================
# 5. MAIN — batch by year: open each ERA5 NC once per year
# ============================================================
def main():
    t0 = datetime.now()
    print("任务二 v1.0 — 逐时尾流出力核算 (按年分批, 每NC打开一次)")
    print(f"使用任务零底座: turbine_coordinates (UTM 真实坐标)")
    print(f"机型: 统一 IEA 10MW | 电气损耗: {ELECTRICAL_LOSS:.3f}")
    print("=" * 60)

    # Load base
    print("[1/4] 加载任务零底座...")
    coords, farms = load_task0_base()
    model = load_turbine_model()
    print(f"  {len(farms)} 风场, {sum(len(coords[fid].get(2024,[])) for fid in farms):,} 台风机 (2024)")

    # Warm Numba
    tp = np.random.rand(10,2)*1000; tw = np.full(10, 8.0)
    _ = wake_gauss(tp, math.radians(180), tw, model['D'], model['ws'], model['ct'])
    print("  Numba 预热完成\n")

    # Build farm-to-region mapping (using task0 centroid)
    print("[1/4] 构建区域映射...")
    farm_region = {}
    for fid, info in farms.items():
        clat, clon = info['centroid_lat'], info['centroid_lon']
        if 8 <= clat <= 44 and 104 <= clon <= 143: farm_region[fid] = 'east_asia'
        elif 39 <= clat <= 63 and -12 <= clon <= 32: farm_region[fid] = 'europe'
        elif 36 <= clat <= 42 and -78 <= clon <= -68: farm_region[fid] = 'us_east'
        else: farm_region[fid] = 'east_asia'

    # Region → NC file list
    region_nc = {}
    for rkey in ['east_asia', 'europe', 'us_east']:
        region_nc[rkey] = {}
        for yr in ALL_YRS:
            path = os.path.join(DATA_DIR, f"era5_{rkey}_{yr}.nc")
            if os.path.exists(path): region_nc[rkey][yr] = path

    # CSV — v4 with 3 wake models + wake_model column
    CSV = os.path.join(OUT_DIR, "task2_summary_v4.csv")
    completed = set()
    if os.path.exists(CSV):
        with open(CSV, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed.add((int(r['farm_id']), int(r['year']), r['wake_model']))

    # Representative farms for HOURLY output (large farms across 3 regions)
    REP_FARMS = {0: 'East Asia mega-farm (928 turbines)',
                 2: 'Europe Belgium cluster (572 turbines)',
                 5: 'Europe UK belt (339 turbines)'}

    # Task list
    tasks_by_region_year = defaultdict(lambda: defaultdict(list))
    total_remaining = 0
    for fid in sorted(farms.keys()):
        rkey = farm_region.get(fid, 'east_asia')
        for yr in ALL_YRS:
            for wm in ['gaussian','jensen','curl']:
                if (fid, yr, wm) not in completed:
                    yr_turbs = coords.get(fid, {}).get(yr, [])
                    if len(yr_turbs) >= 2:
                        if wm == 'gaussian':  # only count once per farm-year
                            tasks_by_region_year[rkey][yr].append((fid, yr, yr_turbs))
                            total_remaining += 1

    total_records = total_remaining * 3  # 3 wake models per farm-year
    print(f"[2/4] 农场年: {total_remaining} 待算, 共 {total_records} 条记录 (3模型)")
    print(f"  策略: 每NC一次, 3种尾流并行计算, 5个代表场输出逐时表\n")

    if total_remaining == 0:
        print("全部已完成!")
        return

    # Open summary CSV
    fout = open(CSV, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(fout)
    if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
        w.writerow(['farm_id','year','n_turb','capacity_kW','n_hours','wake_model',
                   'AEP_kWh','CF','WakeLoss','Volatility_kW','CV',
                   'P5_kW','P95_kW','RampFreq','low_hours','high_hours'])

    # Open hourly CSVs for representative farms (one per region)
    hourly_files = {}
    for fid, desc in REP_FARMS.items():
        hp = os.path.join(OUT_DIR, f"task2_hourly_F{fid}.csv")
        hf = open(hp, 'a', newline='', encoding='utf-8-sig')
        hw = csv.writer(hf)
        if os.path.getsize(hp) == 0:
            hw.writerow(['farm_id','year','month','day','hour','V_free_ms','theta_deg',
                        'P_noWake_kW','P_wake_Gaussian_kW','P_wake_Jensen_kW','P_wake_Curl_kW','n_turb'])
        hourly_files[fid] = (hf, hw)

    t_start = datetime.now(); done = 0
    ga_warm = wake_gauss(tp, math.radians(180), tw, model['D'], model['ws'], model['ct'])
    _ = wake_jensen(tp, math.radians(180), tw, model['D'], model['ws'], model['ct'])
    _ = wake_curl(tp, math.radians(180), tw, model['D'], model['ws'], model['ct'])
    print("  Jensen+Curl 预热完成")

    for rkey in ['east_asia', 'europe', 'us_east']:
        for yr in ALL_YRS:
            task_list = tasks_by_region_year.get(rkey, {}).get(yr, [])
            if not task_list: continue
            nc_path = region_nc.get(rkey, {}).get(yr)
            if nc_path is None: continue
            japan_path = os.path.join(DATA_DIR, f"era5_japan_{yr}.nc")
            has_japan = os.path.exists(japan_path)

            print(f"[{rkey} {yr}] {len(task_list)} farms", flush=True)

            with ERA5Reader(nc_path) as era5:
                all_times = era5.times()
                farm_wind_cache = {}
                for fid_yr_tuple in task_list:
                    fid = fid_yr_tuple[0]
                    info = farms[fid]
                    clat, clon = info['centroid_lat'], info['centroid_lon']
                    if rkey == 'east_asia' and clat > 41 and has_japan: continue
                    cache_key = (clat, clon)
                    if cache_key not in farm_wind_cache:
                        u100, v100 = era5.wind_at(clat, clon)
                        ws_raw = np.sqrt(u100**2 + v100**2)
                        wd_raw = (np.degrees(np.arctan2(u100, v100)) + 180) % 360
                        mask = ws_raw >= 3.0
                        idx = np.where(mask)[0]
                        V_arr = ws_raw[idx] * (model['H'] / H_REF) ** ALPHA
                        th_arr = wd_raw[idx]
                        farm_wind_cache[cache_key] = (V_arr, th_arr, all_times, idx)

                for fid, yr_f, turbs in task_list:
                    info = farms[fid]
                    clat, clon = info['centroid_lat'], info['centroid_lon']
                    cache_key = (clat, clon)
                    if cache_key not in farm_wind_cache: continue

                    V_arr, th_arr, all_times, idx_t = farm_wind_cache[cache_key]
                    is_rep = (fid in REP_FARMS)
                    if is_rep:
                        results, pn_sum, pw_hourly, pos = compute_all_models(
                            fid, yr, turbs, model, V_arr, th_arr, output_hourly=True)
                        # Write hourly rows
                        hf, hw = hourly_files[fid]
                        nh = len(V_arr)
                        D_model = model['D']
                        for s in range(nh):
                            dt = all_times[idx_t[s]]
                            ws_arr = np.full(len(turbs), V_arr[s])
                            # Re-compute no-wake and wake for this hour from stored pw_hourly
                            pn_h = float(power_nb(ws_arr, model['ws'], model['power']).sum())
                            pw_g = float(pw_hourly['gaussian'][s])
                            pw_j = float(pw_hourly['jensen'][s])
                            pw_c = float(pw_hourly['curl'][s])
                            hw.writerow([fid, yr, dt.month, dt.day, dt.hour,
                                        f"{V_arr[s]:.2f}", f"{th_arr[s]:.1f}",
                                        f"{pn_h:.1f}", f"{pw_g:.1f}", f"{pw_j:.1f}", f"{pw_c:.1f}",
                                        len(turbs)])
                    else:
                        results = compute_all_models(fid, yr, turbs, model, V_arr, th_arr)

                    if results is None: continue
                    for wm in ['gaussian','jensen','curl']:
                        r = results[wm]
                        w.writerow([r['farm_id'], r['year'], r['n_turb'], r['capacity_kW'],
                                   r['n_hours'], wm,
                                   f"{r['AEP_kWh']:.1f}", f"{r['CF']:.4f}",
                                   f"{r['WakeLoss']:.4f}", f"{r['Volatility_kW']:.1f}",
                                   f"{r['CV']:.4f}", f"{r['P5_kW']:.1f}", f"{r['P95_kW']:.1f}",
                                   f"{r['RampFreq']:.4f}", f"{r['low_hours']:.4f}", f"{r['high_hours']:.4f}"])
                    done += 1

            # Japan
            if rkey == 'east_asia' and has_japan:
                jp_tasks = [(fid, yr_f, t) for (fid, yr_f, t) in task_list if farms[fid]['centroid_lat'] > 41]
                if jp_tasks:
                    with ERA5Reader(japan_path) as era5:
                        jp_cache = {}
                        for fid, yr_f, turbs in jp_tasks:
                            info = farms[fid]
                            clat, clon = info['centroid_lat'], info['centroid_lon']
                            cache_key = (clat, clon)
                            if cache_key not in jp_cache:
                                u100, v100 = era5.wind_at(clat, clon)
                                ws_raw = np.sqrt(u100**2 + v100**2)
                                wd_raw = (np.degrees(np.arctan2(u100, v100)) + 180) % 360
                                mask = ws_raw >= 3.0
                                idx = np.where(mask)[0]
                                V_arr = ws_raw[idx] * (model['H'] / H_REF) ** ALPHA
                                th_arr = wd_raw[idx]
                                jp_cache[cache_key] = (V_arr, th_arr, era5.times(), idx)
                            V_arr, th_arr, at_jp, idx_jp = jp_cache[cache_key]
                            results_jp = compute_all_models(fid, yr, turbs, model, V_arr, th_arr)
                            if results_jp:
                                for wm in ['gaussian','jensen','curl']:
                                    r = results_jp[wm]
                                    w.writerow([r['farm_id'], r['year'], r['n_turb'], r['capacity_kW'],
                                               r['n_hours'], wm,
                                               f"{r['AEP_kWh']:.1f}", f"{r['CF']:.4f}",
                                               f"{r['WakeLoss']:.4f}", f"{r['Volatility_kW']:.1f}",
                                               f"{r['CV']:.4f}", f"{r['P5_kW']:.1f}", f"{r['P95_kW']:.1f}",
                                               f"{r['RampFreq']:.4f}", f"{r['low_hours']:.4f}", f"{r['high_hours']:.4f}"])
                                done += 1

            fout.flush()
            for (hf,hw) in hourly_files.values(): hf.flush()
            if total_remaining > 0:
                pct = done*100//total_remaining
                print(f"  [{datetime.now().strftime('%H:%M')}] {done}/{total_remaining} ({pct}%)")

    fout.close()
    for hf, _ in hourly_files.values(): hf.close()
    elapsed = (datetime.now() - t0).total_seconds() / 60
    print(f"\n耗时: {elapsed:.1f} min | 输出: {CSV}")
    for fid in REP_FARMS:
        hp = os.path.join(OUT_DIR, f"task2_hourly_F{fid}.csv")
        print(f"  逐时: {hp} ({os.path.getsize(hp)/1e6:.1f} MB)")
    print("完成!")

if __name__ == "__main__":
    main()
