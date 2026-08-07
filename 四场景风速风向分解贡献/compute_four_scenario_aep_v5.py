"""
=============================================================================
 compute_four_scenario_aep_v5.py — 四情景 FLORIS 直算 AEP v5
 依据: 廷显交付_四情景v5修正指导书.md (琪明, 2026-08-07)

 v4→v5 修复:
   1. 尾流模型: Numba hand-wake → FLORIS 库预计算查找表 (与S3一致)
   2. P01 WS:   np.mean(sector) → 分位数映射 (保留扇区内方差)
   3. P00 WS:   np.mean(sector) → 从历史条件分布采样
   4. P00:      1轮 → 10轮采样取平均 (降MC噪声)
   5. P11/P10:  手写Numba重算 → 从S3/counterfactual直接复制

 数据源:
   P11: S3 task3_s3_comparison.csv AEP_kWh (gauss+real) → 零误差
   P10: task2_counterfactual.csv AEP_baseWD_kWh (916对复制, 287对FLORIS新算)
   P01: FLORIS查找表 × 分位数映射WS × 实际WD
   P00: FLORIS查找表 × 10轮采样WD/WS

 用法:
   python compute_four_scenario_aep_v5.py
   python compute_four_scenario_aep_v5.py --region east_asia
=============================================================================
"""
import os, sys, csv, math, time, tempfile, shutil, argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
TASK0_DIR = os.path.join(REPO_ROOT, "offshore-task0", "output", "task0")
TASK2_DIR = os.path.join(REPO_ROOT, "offshore-task2")
TASK3_DIR = os.path.join(REPO_ROOT, "offshore-task3")
DATA_DIR  = os.path.join(TASK2_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Constants ----
ALL_YRS = list(range(2014, 2025))
H_REF = 100.0
ALPHA_DEFAULT = 0.11
N_SECTORS = 36  # 10-deg sectors

COUNTERFACTUAL_CSV = os.path.join(TASK2_DIR, "output", "task2_counterfactual.csv")
S3_CSV = os.path.join(TASK3_DIR, "output", "task3_s3_comparison.csv")
LOOKUP_CSV = os.path.join(OUT_DIR, "four_scenario_lookup_table.csv")
CONFIG_MAP_CSV = os.path.join(OUT_DIR, "four_scenario_config_map.csv")
OUTPUT_CSV = os.path.join(OUT_DIR, "four_scenario_floris_aep_v5.csv")

WS_BINS_ARR = np.array([0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                         11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
                         25.0, 30.0], dtype=np.float64)
WD_SECTORS_ARR = np.array(list(range(0, 360, 10)), dtype=np.float64)

# ---- ERA5 NC access (same as counterfactual) ----
class ERA5Reader:
    def __init__(self, path): self.path = path
    def __enter__(self):
        self.tmp = tempfile.mkdtemp(); self.tmpnc = os.path.join(self.tmp, "data.nc")
        shutil.copy2(self.path, self.tmpnc)
        import netCDF4 as nc4
        self.ds = nc4.Dataset(self.tmpnc, 'r')
        return self
    def __exit__(self, *a):
        if hasattr(self,'ds'): self.ds.close()
        if hasattr(self,'tmp'): shutil.rmtree(self.tmp)
    def wind_at(self, lat, lon):
        la = self.ds['latitude'][:]; lo = self.ds['longitude'][:]
        ila = int(np.argmin(np.abs(la-lat))); ilo = int(np.argmin(np.abs(lo-lon)))
        u = np.array(self.ds['u100'][:,ila,ilo], dtype=np.float64)
        v = np.array(self.ds['v100'][:,ila,ilo], dtype=np.float64)
        return u, v


# =========================================================================
# 1. DATA LOADING
# =========================================================================

def load_task0_coords():
    coords = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(TASK0_DIR, "turbine_coordinates.csv"), 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id']); yr = int(r['year'])
            coords[fid][yr].append({
                'x_m': float(r['x_m']), 'y_m': float(r['y_m']),
                'lon': float(r['lon']), 'lat': float(r['lat'])
            })
    return coords

def load_farms_master():
    farms = {}
    with open(os.path.join(TASK0_DIR, "farms_master.csv"), 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id'])
            farms[fid] = {
                'centroid_lat': float(r['centroid_lat']),
                'centroid_lon': float(r['centroid_lon']),
            }
    return farms

def load_wake_table():
    """Load FLORIS precomputed wake efficiency matrix per config_hash."""
    # Read entire lookup table: config_hash -> (wake_eff[36,19], p_noWake[19])
    wake = {}
    with open(LOOKUP_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            ch = r['config_hash']
            ws_bin = float(r['ws_bin_m_s'])
            wd_sector = float(r['wd_sector_deg'])
            eta = float(r['wake_efficiency'])
            pnw = float(r['p_noWake_kW'])

            if ch not in wake:
                wake[ch] = {
                    'wake_eff': np.ones((N_SECTORS, len(WS_BINS_ARR)), dtype=np.float64),
                    'p_noWake': np.zeros(len(WS_BINS_ARR), dtype=np.float64),
                }

            iwd = int(wd_sector / 10) % N_SECTORS  # wd_sector in {0,10,20,...,350}
            iws = np.where(np.abs(WS_BINS_ARR - ws_bin) < 0.001)[0]
            if len(iws) > 0:
                iws = iws[0]
                wake[ch]['wake_eff'][iwd, iws] = eta
                # p_noWake is same for all WD, just store it
                wake[ch]['p_noWake'][iws] = pnw

    return wake

def load_config_map():
    """Load farm-year -> config_hash mapping."""
    cmap = {}
    with open(CONFIG_MAP_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            cmap[(int(r['farm_id']), int(r['year']))] = r['config_hash']
    return cmap

def load_s3_p11():
    """Load P11 from S3 (gauss+real) — this is FLORIS library output, exact match target."""
    p11 = {}
    with open(S3_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('wake_model','gauss')=='gauss' and r.get('layout_type','real')=='real':
                p11[(int(r['farm_id']), int(r['year']))] = float(r['AEP_kWh'])
    return p11

def load_counterfactual_p10():
    """Load P10 from counterfactual (AEP_baseWD_kWh)."""
    p10 = {}
    if os.path.exists(COUNTERFACTUAL_CSV):
        with open(COUNTERFACTUAL_CSV, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                p10[(int(r['farm_id']), int(r['year']))] = float(r['AEP_baseWD_kWh'])
    return p10

def load_completed(csv_path):
    """Load already-computed (farm_id, year) from output CSV."""
    done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                done.add((int(r['farm_id']), int(r['year'])))
    return done

def map_farm_to_region(clat, clon):
    if 8 <= clat <= 44 and 104 <= clon <= 143: return 'east_asia'
    elif 39 <= clat <= 63 and -12 <= clon <= 32: return 'europe'
    elif 36 <= clat <= 44 and -78 <= clon <= -68: return 'us_east'
    elif 30 <= clat <= 46 and 128 <= clon <= 146: return 'japan'
    return 'east_asia'


# =========================================================================
# 2. HISTORICAL CDF CONSTRUCTION
# =========================================================================

def build_historical_cdfs(farms):
    """Build per-farm historical CDFs from 1981-2010 ERA5 daily data.

    For each farm:
      - p0_d[36]: sector frequency (WD distribution)
      - ws_hist_by_sector[36]: sorted list of WS values per sector (raw data)
    """
    import netCDF4
    from netCDF4 import num2date

    # Map farm→region
    farm_region = {}
    for fid, info in farms.items():
        clat, clon = info['centroid_lat'], info['centroid_lon']
        farm_region[fid] = map_farm_to_region(clat, clon)

    # Initialize
    ws_by_sector = {}
    wd_counts = {}
    for fid in farms:
        ws_by_sector[fid] = [[] for _ in range(N_SECTORS)]
        wd_counts[fid] = np.zeros(N_SECTORS)

    for rkey in ['east_asia', 'europe', 'us_east', 'japan']:
        region_farms = [fid for fid, r in farm_region.items() if r == rkey]
        if not region_farms: continue

        for decade in ['b1981_1990', 'b1991_2000', 'b2001_2010']:
            path = os.path.join(DATA_DIR, f"era5_baseline_daily_{rkey}_{decade}.nc")
            if not os.path.exists(path): continue

            with ERA5Reader(path) as era5:
                try:
                    vt = era5.ds['valid_time']
                    times = num2date(vt[:], units=vt.units)
                except:
                    vt = era5.ds['time']
                    times = num2date(vt[:], units=vt.units)

                for fid in region_farms:
                    clat, clon = farms[fid]['centroid_lat'], farms[fid]['centroid_lon']
                    try:
                        u, v = era5.wind_at(clat, clon)
                    except:
                        continue

                    ws = np.sqrt(u*u + v*v)
                    wd_rad = np.arctan2(v, u)
                    wd_deg = (270.0 - np.degrees(wd_rad)) % 360.0

                    valid = ws >= 3.0
                    ws_v = ws[valid]; wd_v = wd_deg[valid]

                    for i in range(len(ws_v)):
                        si = min(int(wd_v[i] / (360/N_SECTORS)), N_SECTORS-1)
                        ws_by_sector[fid][si].append(float(ws_v[i]))
                        wd_counts[fid][si] += 1

    # Post-process: sort WS arrays (for CDF lookup), compute p0_d
    hist_cdfs = {}
    for fid in farms:
        ws_data = []
        for si in range(N_SECTORS):
            if len(ws_by_sector[fid][si]) > 0:
                ws_data.append(np.sort(np.array(ws_by_sector[fid][si], dtype=np.float64)))
            else:
                ws_data.append(np.array([5.0], dtype=np.float64))
            # Also collect ALL WS values for marginal CDF
            if si == 0:
                all_ws = list(ws_by_sector[fid][si])
            else:
                all_ws.extend(ws_by_sector[fid][si])

        total = wd_counts[fid].sum()
        if total > 0:
            p0_d = wd_counts[fid] / total
        else:
            p0_d = np.ones(N_SECTORS) / N_SECTORS

        # Marginal CDF (all sectors combined)
        marginal_ws = np.sort(np.array(all_ws, dtype=np.float64))

        hist_cdfs[fid] = {
            'p0_d': p0_d,
            'ws_sorted': ws_data,       # list of sorted arrays per sector
            'marginal_ws': marginal_ws,
            'n_per_sector': np.array([len(ws_data[si]) for si in range(N_SECTORS)]),
        }

    n_farms = len(hist_cdfs)
    total_hours = sum(h['n_per_sector'].sum() for h in hist_cdfs.values())
    print(f"  历史CDF: {n_farms} farms, {total_hours/1e6:.1f}M total hours")
    return hist_cdfs


# =========================================================================
# 3. QUANTILE MAPPING
# =========================================================================

def build_year_cdfs(ws_hourly, wd_hourly):
    """Build per-sector empirical CDF from a single year's hourly data.

    Returns:
        ws_sorted[36]: sorted WS arrays per sector
        ns[36]: counts per sector
    """
    ws_by_sector = [[] for _ in range(N_SECTORS)]
    for h in range(len(ws_hourly)):
        ws = ws_hourly[h]; wd = wd_hourly[h]
        if ws < 3.0: continue
        si = min(int(wd / (360/N_SECTORS)), N_SECTORS-1)
        ws_by_sector[si].append(float(ws))

    ws_sorted = []
    ns = []
    for si in range(N_SECTORS):
        if len(ws_by_sector[si]) > 0:
            ws_sorted.append(np.sort(np.array(ws_by_sector[si], dtype=np.float64)))
        else:
            ws_sorted.append(np.array([5.0], dtype=np.float64))
        ns.append(len(ws_by_sector[si]))

    return ws_sorted, np.array(ns)


def quantile_map_ws(ws_actual, sector, year_ws_sorted, hist_ws_sorted,
                     year_ns, hist_ns, marginal_ws):
    """Map a wind speed to historical distribution via quantile matching.

    Args:
        ws_actual: actual wind speed (m/s)
        sector: WD sector index (0-35)
        year_ws_sorted: sorted WS values for this sector (current year)
        hist_ws_sorted: sorted WS values for this sector (historical baseline)
        year_ns: counts per sector (current year)
        hist_ns: counts per sector (historical)
        marginal_ws: marginal (all-sector) sorted WS for shrinkage

    Returns: v_mapped (m/s)
    """
    N_thresh = 30  # sparse sector threshold

    # Get year sector CDF with shrinkage if needed
    y_sorted = year_ws_sorted[sector]
    h_sorted = hist_ws_sorted[sector]
    N_year = year_ns[sector]

    if N_year >= N_thresh:
        # Direct quantile: rank of ws_actual in year → same rank in hist
        # Fraction of year values <= ws_actual
        rank = np.searchsorted(y_sorted, ws_actual, side='right') / len(y_sorted)
        rank = np.clip(rank, 1e-6, 1.0 - 1e-6)
    else:
        # Shrink toward marginal
        lam = N_year / (N_year + N_thresh)
        # Year rank
        rank_year = np.searchsorted(y_sorted, ws_actual, side='right') / max(1, len(y_sorted))
        # Marginal rank
        rank_marg = np.searchsorted(marginal_ws, ws_actual, side='right') / max(1, len(marginal_ws))
        rank = lam * rank_year + (1 - lam) * rank_marg
        rank = np.clip(rank, 1e-6, 1.0 - 1e-6)

    # Map to historical distribution
    idx = int(rank * (len(h_sorted) - 1))
    idx = np.clip(idx, 0, len(h_sorted) - 1)
    return float(h_sorted[idx])


# =========================================================================
# 4. HOURLY REPLAY (bin lookup)
# =========================================================================

def replay_hourly(ws_hourly, wd_hourly, wake_eff, p_noWake):
    """Replay hourly AEP from precomputed FLORIS wake table (vectorized)."""
    # Bin all hours at once
    iws = np.argmin(np.abs(WS_BINS_ARR[:, np.newaxis] - ws_hourly[np.newaxis, :]), axis=0)
    iwd = (wd_hourly / (360.0 / N_SECTORS)).astype(np.int32) % N_SECTORS
    return float(np.sum(p_noWake[iws] * wake_eff[iwd, iws]))


# =========================================================================
# 5. MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='四情景 FLORIS 直算 AEP v5')
    parser.add_argument('--region', type=str, default=None,
                        choices=['east_asia', 'europe', 'us_east', 'japan'],
                        help='只处理指定区域 (并行模式)')
    args = parser.parse_args()

    out_csv = OUTPUT_CSV if not args.region else \
        os.path.join(OUT_DIR, f"four_scenario_floris_aep_v5_{args.region}.csv")

    t_start = datetime.now()
    print("=" * 60)
    print(" 四情景 FLORIS 直算 AEP v5")
    if args.region: print(f" 区域: {args.region} (并行模式)")
    print(f" 启动: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- Load data ----
    print("\n[1/6] 加载数据...")
    coords = load_task0_coords()
    farms = load_farms_master()
    wake_table = load_wake_table()
    config_map = load_config_map()
    s3_p11 = load_s3_p11()
    cf_p10 = load_counterfactual_p10()
    completed = load_completed(out_csv)
    if args.region:
        completed |= load_completed(OUTPUT_CSV)

    print(f"  {len(farms)} farms, S3 P11: {len(s3_p11)}, CF P10: {len(cf_p10)}")
    print(f"  Wake table: {len(wake_table)} configs, config-map: {len(config_map)} mappings")
    print(f"  Already computed: {len(completed)} pairs")

    # ---- Build historical CDFs ----
    print("\n[2/6] 构建历史 CDF (1981-2010 ERA5 daily)...")
    hist_cdfs = build_historical_cdfs(farms)

    # ---- Build task list ----
    print("\n[3/6] 构建任务列表...")
    pending = []
    for key, p11_val in s3_p11.items():
        fid, yr = key
        if key in completed: continue
        if yr not in ALL_YRS: continue
        turbs = coords.get(fid, {}).get(yr, [])
        if len(turbs) < 2: continue
        pending.append((fid, yr, len(turbs)))

    if not pending:
        print("  全部已完成!")
        return

    print(f"  Pending: {len(pending)} farm-years")

    # Group by region→year for batch processing
    tasks_by_ry = defaultdict(lambda: defaultdict(list))
    for fid, yr, nt in pending:
        rkey = map_farm_to_region(
            farms[fid]['centroid_lat'], farms[fid]['centroid_lon'])
        if args.region and rkey != args.region: continue
        tasks_by_ry[rkey][yr].append((fid, yr))

    total_pending = sum(len(v) for rv in tasks_by_ry.values() for v in rv.values())
    print(f"  Pending in region: {total_pending}")

    # ---- Write header ----
    is_new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    if is_new:
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['farm_id', 'year', 'P11_kWh', 'P10_kWh', 'P01_kWh', 'P00_kWh'])

    # ---- Set up turbine power curve for hub-height correction ----
    import yaml
    with open(os.path.join(DATA_DIR, "iea_10MW.yaml"), 'r', encoding='utf-8') as f:
        tc = yaml.safe_load(f)
    H_TURB = tc['hub_height']
    NR = (H_TURB / H_REF) ** ALPHA_DEFAULT  # hub-height correction factor

    # ---- Process ----
    print(f"\n[4/6] 计算四情景...")
    done = 0
    rng = np.random.RandomState(42)

    for rkey in (['east_asia', 'europe', 'us_east', 'japan'] if not args.region else [args.region]):
        for yr in ALL_YRS:
            task_list = tasks_by_ry.get(rkey, {}).get(yr, [])
            if not task_list: continue

            nc_path = os.path.join(DATA_DIR, f"era5_{rkey}_{yr}.nc")
            if not os.path.exists(nc_path):
                print(f"  SKIP {rkey}/{yr}: no NC file")
                continue

            t_batch = time.perf_counter()

            with ERA5Reader(nc_path) as era5:
                for fid, yr_f in task_list:
                    clat, clon = farms[fid]['centroid_lat'], farms[fid]['centroid_lon']

                    # Get config and wake table
                    ch = config_map.get((fid, yr_f))
                    if ch is None or ch not in wake_table:
                        continue
                    wt = wake_table[ch]

                    # Extract ERA5
                    try:
                        u100, v100 = era5.wind_at(clat, clon)
                    except:
                        continue

                    ws_raw = np.sqrt(u100**2 + v100**2)
                    wd_rad = np.arctan2(v100, u100)
                    wd_deg = (270.0 - np.degrees(wd_rad)) % 360.0

                    mask = ws_raw >= 3.0
                    idx = np.where(mask)[0]
                    if len(idx) == 0: continue

                    ws_hub = ws_raw[idx] * NR  # height-corrected
                    wd_arr = wd_deg[idx]
                    nh = len(ws_hub)

                    # Build year CDFs
                    year_ws_sorted, year_ns = build_year_cdfs(ws_hub, wd_arr)

                    # ---- P11: from S3 ----
                    p11_val = s3_p11.get((fid, yr_f), 0.0)

                    # ---- P10: from counterfactual, or compute ----
                    if (fid, yr_f) in cf_p10:
                        p10_val = cf_p10[(fid, yr_f)]
                    else:
                        # Compute P10: actual WS + sampled hist WD (vectorized)
                        p0_d = hist_cdfs[fid]['p0_d']
                        rng_p10 = np.random.RandomState(fid * 10000 + yr_f * 100)
                        p10_si = rng_p10.choice(N_SECTORS, size=nh, p=p0_d)
                        p10_wd = (p10_si + 0.5) * (360.0 / N_SECTORS)
                        p10_val = replay_hourly(ws_hub, p10_wd, wt['wake_eff'], wt['p_noWake'])

                    # ---- P01: quantile-mapped WS + actual WD ----
                    p01_ws = np.zeros(nh)
                    hist_ws_sorted = hist_cdfs[fid]['ws_sorted']
                    hist_ns = hist_cdfs[fid]['n_per_sector']
                    marginal_ws = hist_cdfs[fid]['marginal_ws']
                    si_arr = (wd_arr / (360.0 / N_SECTORS)).astype(np.int32) % N_SECTORS

                    for h in range(nh):
                        p01_ws[h] = quantile_map_ws(
                            ws_hub[h], si_arr[h],
                            year_ws_sorted, hist_ws_sorted,
                            year_ns, hist_ns, marginal_ws
                        )
                    p01_val = replay_hourly(p01_ws, wd_arr, wt['wake_eff'], wt['p_noWake'])

                    # ---- P00: 10 rounds of sampled WD + sampled WS ----
                    p0_d = hist_cdfs[fid]['p0_d']
                    p00_sum = 0.0
                    for round_i in range(10):
                        rng_p00 = np.random.RandomState(
                            fid * 10000 + yr_f * 10 + round_i)
                        # Vectorized: sample all sectors at once
                        p00_si = rng_p00.choice(N_SECTORS, size=nh, p=p0_d)
                        p00_wd = (p00_si + 0.5) * (360.0 / N_SECTORS)
                        p00_ws = np.zeros(nh)
                        # Per-sector WS sampling (36 iterations vs 8760)
                        for si_val in range(N_SECTORS):
                            sector_mask = p00_si == si_val
                            n_in_sector = sector_mask.sum()
                            if n_in_sector > 0:
                                hs = hist_ws_sorted[si_val]
                                p00_ws[sector_mask] = hs[rng_p00.randint(0, len(hs), size=n_in_sector)]
                        p00_sum += replay_hourly(p00_ws, p00_wd, wt['wake_eff'], wt['p_noWake'])
                    p00_val = p00_sum / 10.0

                    # Write
                    with open(out_csv, 'a', newline='', encoding='utf-8-sig') as f:
                        w = csv.writer(f)
                        w.writerow([fid, yr_f,
                                   f"{p11_val:.6f}", f"{p10_val:.6f}",
                                   f"{p01_val:.6f}", f"{p00_val:.6f}"])

                    done += 1

            elapsed_batch = time.perf_counter() - t_batch
            print(f"  {rkey}/{yr}: {len(task_list)} farms in {elapsed_batch:.0f}s "
                  f"({done}/{total_pending} total)", flush=True)

    # ---- Summary ----
    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    final = load_completed(out_csv)
    if args.region:
        final |= load_completed(OUTPUT_CSV)
    print(f"\n[5/6] 完成! {elapsed/60:.0f}min")
    print(f"  Output: {out_csv}")
    print(f"  Rows: {len(final)}")

    # Self-check P11 vs S3
    print("\n[6/6] 自检 P11 vs S3...")
    print(f"  (琪明验收: 中位误差必须 = 0)")
    s3_map = s3_p11
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        diffs = []
        with open(out_csv, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                k = (int(r['farm_id']), int(r['year']))
                if k in s3_map:
                    diffs.append(abs(float(r['P11_kWh']) - s3_map[k]))
        if diffs:
            import statistics
            print(f"    P11 vs S3: max={max(diffs):.6f}, mean={statistics.mean(diffs):.6f}, "
                  f"median={statistics.median(diffs):.6f}")
            print(f"    精确匹配 (diff<1): {sum(1 for d in diffs if d < 1)}/{len(diffs)}")
            print(f"    => {'PASS' if statistics.median(diffs) < 0.01 else 'FAIL: P11≠S3!'}")


if __name__ == "__main__":
    main()
