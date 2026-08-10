"""
=============================================================================
 compute_four_scenario_aep_v6.py — 四情景 FLORIS 逐时精确 AEP v6
 依据: 廷显交付_四情景v6返工指导书(1).md (琪明, 2026-08-09 修正版)

 v5→v6 修复（三个问题，逐项对应）:
   ┌────────┬──────────────────────────────────────────────────────────┐
   │ 问题1  │ P01/P00 系统性低估 3-4pp                                  │
   │ 原因   │ replay_hourly 用分箱常量 p_noWake[bin] 代替逐时精确功率    │
   │ 修复   │ 改为 np.interp(ws, power_curve) × n_turb × wake_eff × EL  │
   │        │ 与 S3 完全同口径                                          │
   ├────────┼──────────────────────────────────────────────────────────┤
   │ 问题2  │ P10 2017→2018 有 4.6pp 断崖                               │
   │ 原因   │ counterfactual 是 Numba 手写 wake (k=0.05)，不是 FLORIS 库 │
   │        │ 2014-2017 FLORIS + 2018-2024 Numba → 两套模型拼接          │
   │ 修复   │ P10 全部 1203 对 FLORIS 逐时自算 × 10轮，不复制 CF        │
   ├────────┼──────────────────────────────────────────────────────────┤
   │ 问题3  │ 日变化混叠（加分项，不强求）                               │
   │ 处理   │ 在交付说明中标注 12:00 UTC 基准的限制                      │
   └────────┴──────────────────────────────────────────────────────────┘

 四情景统一口径（修正版）:
   P11: 实际WS × 实际WD → S3 直接复制
   P10: 实际WS × 历史WD → FLORIS 逐时 × 10轮（全部自算）
   P01: 分位数映射WS × 实际WD → FLORIS 逐时精确功率
   P00: 采样WS × 采样WD → FLORIS 逐时精确功率 × 10轮

 用法:
   python compute_four_scenario_aep_v6.py
   python compute_four_scenario_aep_v6.py --region east_asia
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
ELECTRICAL_LOSS = 0.95 * 0.97  # = 0.9215
N_SECTORS = 36  # 10-deg sectors

COUNTERFACTUAL_CSV = os.path.join(TASK2_DIR, "output", "task2_counterfactual.csv")
S3_CSV = os.path.join(TASK3_DIR, "output", "task3_s3_comparison.csv")
LOOKUP_CSV = os.path.join(OUT_DIR, "four_scenario_lookup_table.csv")
CONFIG_MAP_CSV = os.path.join(OUT_DIR, "four_scenario_config_map.csv")
OUTPUT_CSV = os.path.join(OUT_DIR, "four_scenario_floris_aep_v6.csv")

WS_BINS_ARR = np.array([0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                         11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
                         25.0, 30.0], dtype=np.float64)
WD_SECTORS_ARR = np.array(list(range(0, 360, 10)), dtype=np.float64)
WD_STEP = 360.0 / N_SECTORS  # 10.0


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
    """Load FLORIS precomputed wake efficiency matrix per config_hash.

    v6: 只加载 wake_eff[36,19]，不再加载 p_noWake（改为逐时精确插值）。
    """
    wake = {}
    with open(LOOKUP_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            ch = r['config_hash']
            ws_bin = float(r['ws_bin_m_s'])
            wd_sector = float(r['wd_sector_deg'])
            eta = float(r['wake_efficiency'])

            if ch not in wake:
                wake[ch] = {
                    'wake_eff': np.ones((N_SECTORS, len(WS_BINS_ARR)), dtype=np.float64),
                }

            iwd = int(wd_sector / 10) % N_SECTORS
            iws = np.where(np.abs(WS_BINS_ARR - ws_bin) < 0.001)[0]
            if len(iws) > 0:
                wake[ch]['wake_eff'][iwd, iws] = eta

    return wake

def load_config_map():
    """Load farm-year -> config_hash mapping."""
    cmap = {}
    with open(CONFIG_MAP_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            cmap[(int(r['farm_id']), int(r['year']))] = r['config_hash']
    return cmap

def load_s3_p11():
    """Load P11 from S3 (gauss+real) — exact FLORIS library output."""
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

def load_power_curve():
    """Load IEA 10MW power curve from turbine library YAML.
    Returns (ws_array, power_kW_array).
    """
    import yaml
    # Try FLORIS turbine library path first
    import floris as _floris_pkg
    floris_dir = os.path.dirname(_floris_pkg.__file__)
    yaml_path = os.path.join(floris_dir, "turbine_library", "iea_10MW.yaml")
    if not os.path.exists(yaml_path):
        # Fallback: local data dir
        yaml_path = os.path.join(DATA_DIR, "iea_10MW.yaml")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tbl = cfg['power_thrust_table']
    return (np.array(tbl['wind_speed'], dtype=np.float64),
            np.array(tbl['power'], dtype=np.float64))

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
      - marginal_ws: all-sector sorted WS (for shrinkage)
      - n_per_sector[36]: hours per sector
    """
    import netCDF4
    from netCDF4 import num2date

    farm_region = {}
    for fid, info in farms.items():
        clat, clon = info['centroid_lat'], info['centroid_lon']
        farm_region[fid] = map_farm_to_region(clat, clon)

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
                        si = min(int(wd_v[i] / WD_STEP), N_SECTORS-1)
                        ws_by_sector[fid][si].append(float(ws_v[i]))
                        wd_counts[fid][si] += 1

    hist_cdfs = {}
    for fid in farms:
        ws_data = []
        all_ws = []
        for si in range(N_SECTORS):
            if len(ws_by_sector[fid][si]) > 0:
                ws_data.append(np.sort(np.array(ws_by_sector[fid][si], dtype=np.float64)))
            else:
                ws_data.append(np.array([5.0], dtype=np.float64))
            all_ws.extend(ws_by_sector[fid][si])

        total = wd_counts[fid].sum()
        p0_d = wd_counts[fid] / total if total > 0 else np.ones(N_SECTORS) / N_SECTORS
        marginal_ws = np.sort(np.array(all_ws, dtype=np.float64))

        hist_cdfs[fid] = {
            'p0_d': p0_d,
            'ws_sorted': ws_data,
            'marginal_ws': marginal_ws,
            'n_per_sector': np.array([len(ws_data[si]) for si in range(N_SECTORS)]),
        }

    n_farms = len(hist_cdfs)
    total_hours = sum(h['n_per_sector'].sum() for h in hist_cdfs.values())
    print(f"  历史CDF: {n_farms} farms, {total_hours/1e6:.1f}M total hours")
    return hist_cdfs


# =========================================================================
# 3. QUANTILE MAPPING (v6: 保持分位数映射逻辑，与v5一致)
# =========================================================================

def build_year_cdfs(ws_hourly, wd_hourly):
    """Build per-sector empirical CDF from a single year's hourly data."""
    ws_by_sector = [[] for _ in range(N_SECTORS)]
    for h in range(len(ws_hourly)):
        ws = ws_hourly[h]; wd = wd_hourly[h]
        if ws < 3.0: continue
        si = min(int(wd / WD_STEP), N_SECTORS-1)
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

    稀疏扇区用 shrinkage: λ = N/(N+30), F_shrunk = λ*F_year + (1-λ)*F_marginal
    """
    N_thresh = 30

    y_sorted = year_ws_sorted[sector]
    h_sorted = hist_ws_sorted[sector]
    N_year = year_ns[sector]

    if N_year >= N_thresh:
        rank = np.searchsorted(y_sorted, ws_actual, side='right') / len(y_sorted)
        rank = np.clip(rank, 1e-6, 1.0 - 1e-6)
    else:
        lam = N_year / (N_year + N_thresh)
        rank_year = np.searchsorted(y_sorted, ws_actual, side='right') / max(1, len(y_sorted))
        rank_marg = np.searchsorted(marginal_ws, ws_actual, side='right') / max(1, len(marginal_ws))
        rank = lam * rank_year + (1 - lam) * rank_marg
        rank = np.clip(rank, 1e-6, 1.0 - 1e-6)

    idx = int(rank * (len(h_sorted) - 1))
    idx = np.clip(idx, 0, len(h_sorted) - 1)
    return float(h_sorted[idx])


# =========================================================================
# 4. HOURLY REPLAY — v6 核心修复
# =========================================================================

def replay_hourly(ws_hourly, wd_hourly, wake_eff,
                  power_curve_ws, power_curve_power, n_turbines):
    """【v6 修复】逐时精确功率 × 尾流效率 × 电气损耗。

    问题（v5）:
        p_noWake[bin] 是分箱中心功率（如 8 m/s → 627676 kW），
        实际 8.3 m/s 的功率 ≠ 8.0 m/s，累积 8760 小时 → 3-4pp 偏置。

    修复（v6）:
        p_noWake(h) = interp(ws_h, power_curve) × n_turbines  ← 逐时精确
        AEP = Σ_h p_noWake(h) × wake_eff[wd_bin, ws_bin] × ELECTRICAL_LOSS

    与 counterfactual 的 replay_hourly_from_bins() 完全同口径。
    """
    nh = len(ws_hourly)

    # 【修复1】逐时精确无尾流功率（不再用分箱常量）
    p_per_turbine = np.interp(ws_hourly, power_curve_ws, power_curve_power,
                               left=0.0, right=0.0)
    p_noWake = p_per_turbine * n_turbines  # kW, shape (nh,)

    # WS binning: nearest neighbor（与 counterfactual 的 argmin 一致）
    iws = np.argmin(np.abs(WS_BINS_ARR[:, np.newaxis] - ws_hourly[np.newaxis, :]), axis=0)

    # WD binning: nearest neighbor（与 counterfactual 的 argmin 一致）
    iwd = np.argmin(np.abs(WD_SECTORS_ARR[:, np.newaxis] - wd_hourly[np.newaxis, :]), axis=0)

    # 尾流效率查表 + 电气损耗
    p_wake = p_noWake * wake_eff[iwd, iws]  # kW
    aep = float(np.sum(p_wake)) * ELECTRICAL_LOSS  # kWh

    return aep


# =========================================================================
# 5. MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='四情景 FLORIS 逐时精确 AEP v6')
    parser.add_argument('--region', type=str, default=None,
                        choices=['east_asia', 'europe', 'us_east', 'japan'],
                        help='只处理指定区域 (并行模式)')
    args = parser.parse_args()

    out_csv = OUTPUT_CSV if not args.region else \
        os.path.join(OUT_DIR, f"four_scenario_floris_aep_v6_{args.region}.csv")

    t_start = datetime.now()
    print("=" * 60)
    print(" 四情景 FLORIS 逐时精确 AEP v6")
    print(" 修复: 逐时功率插值 + P10补算 + WD抖动")
    if args.region: print(f" 区域: {args.region} (并行模式)")
    print(f" 启动: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- Load data ----
    print("\n[1/7] 加载数据...")
    coords = load_task0_coords()
    farms = load_farms_master()
    wake_table = load_wake_table()
    config_map = load_config_map()
    s3_p11 = load_s3_p11()
    cf_p10 = load_counterfactual_p10()
    pc_ws, pc_power = load_power_curve()
    completed = load_completed(out_csv)
    if args.region:
        completed |= load_completed(OUTPUT_CSV)

    print(f"  {len(farms)} farms, S3 P11: {len(s3_p11)}, CF P10: {len(cf_p10)}")
    print(f"  Wake table: {len(wake_table)} configs, config-map: {len(config_map)} mappings")
    print(f"  功率曲线: {len(pc_ws)} points, 额定 {pc_power.max():.0f} kW")
    print(f"  Already computed: {len(completed)} pairs")

    # ---- Build historical CDFs ----
    print("\n[2/7] 构建历史 CDF (1981-2010 ERA5 daily)...")
    hist_cdfs = build_historical_cdfs(farms)

    # ---- Build task list ----
    print("\n[3/7] 构建任务列表...")
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

    # ---- Height correction factor ----
    import yaml
    import floris as _floris_pkg
    floris_dir = os.path.dirname(_floris_pkg.__file__)
    yaml_path = os.path.join(floris_dir, "turbine_library", "iea_10MW.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(DATA_DIR, "iea_10MW.yaml")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        tc = yaml.safe_load(f)
    H_TURB = tc['hub_height']
    NR = (H_TURB / H_REF) ** ALPHA_DEFAULT  # hub-height correction factor

    # ---- Process ----
    print(f"\n[4/7] 计算四情景 (v6: 逐时精确功率)...")
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
                    wake_eff = wake_table[ch]['wake_eff']
                    n_turb = len(coords.get(fid, {}).get(yr_f, []))

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

                    # Build year CDFs (for P01 quantile mapping)
                    year_ws_sorted, year_ns = build_year_cdfs(ws_hub, wd_arr)

                    # ---- P11: from S3 (exact copy, zero error) ----
                    p11_val = s3_p11.get((fid, yr_f), 0.0)

                    # ---- P10: 10轮 actual WS + sampled hist WD ----
                    # 【修正版】全部 FLORIS 逐时自算，不从 counterfactual 复制
                    # P10 和 P00 的区别只有风速来源：P10 用实际WS, P00 用采样WS
                    p0_d = hist_cdfs[fid]['p0_d']
                    p10_sum = 0.0
                    for round_i in range(10):
                        rng_p10 = np.random.RandomState(
                            fid * 10000 + yr_f * 10 + round_i)
                        p10_si = rng_p10.choice(N_SECTORS, size=nh, p=p0_d)
                        p10_wd = (p10_si + rng_p10.uniform(0, 1, size=nh)) * WD_STEP
                        p10_sum += replay_hourly(ws_hub, p10_wd, wake_eff,
                                                 pc_ws, pc_power, n_turb)
                    p10_val = p10_sum / 10.0

                    # ---- P01: quantile-mapped WS + actual WD ----
                    # 【修复1】逐时精确功率替代分箱常量
                    p01_ws = np.zeros(nh)
                    hist_ws_sorted = hist_cdfs[fid]['ws_sorted']
                    hist_ns = hist_cdfs[fid]['n_per_sector']
                    marginal_ws = hist_cdfs[fid]['marginal_ws']
                    si_arr = (wd_arr / WD_STEP).astype(np.int32) % N_SECTORS

                    for h in range(nh):
                        p01_ws[h] = quantile_map_ws(
                            ws_hub[h], si_arr[h],
                            year_ws_sorted, hist_ws_sorted,
                            year_ns, hist_ns, marginal_ws
                        )
                    p01_val = replay_hourly(p01_ws, wd_arr, wake_eff,
                                            pc_ws, pc_power, n_turb)

                    # ---- P00: 10 rounds of sampled WD + sampled WS ----
                    # 【修复1】逐时精确功率替代分箱常量
                    p0_d = hist_cdfs[fid]['p0_d']
                    p00_sum = 0.0
                    for round_i in range(10):
                        rng_p00 = np.random.RandomState(
                            fid * 10000 + yr_f * 10 + round_i)
                        # Sampled WD with in-sector uniform spread
                        p00_si = rng_p00.choice(N_SECTORS, size=nh, p=p0_d)
                        p00_wd = (p00_si + rng_p00.uniform(0, 1, size=nh)) * WD_STEP
                        # Per-sector WS sampling from historical
                        p00_ws = np.zeros(nh)
                        for si_val in range(N_SECTORS):
                            sector_mask = p00_si == si_val
                            n_in_sector = sector_mask.sum()
                            if n_in_sector > 0:
                                hs = hist_ws_sorted[si_val]
                                p00_ws[sector_mask] = hs[rng_p00.randint(0, len(hs), size=n_in_sector)]
                        p00_sum += replay_hourly(p00_ws, p00_wd, wake_eff,
                                                 pc_ws, pc_power, n_turb)
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

    # ---- Summary & self-check ----
    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    final = load_completed(out_csv)
    if args.region:
        final |= load_completed(OUTPUT_CSV)
    print(f"\n[5/7] 完成! {elapsed/60:.0f}min")
    print(f"  Output: {out_csv}")
    print(f"  Rows: {len(final)}")

    # ---- Acceptance checks ----
    print("\n[6/7] 验收检查...")

    s3_map = s3_p11
    cf_map = cf_p10

    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        rows = []
        with open(out_csv, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                rows.append(r)

        # Check 1: P11 逐字节相等（与 S3 比较）
        print("\n  [验收1] P11 vs S3 (要求: 中位误差 = 0):")
        diffs_p11 = []
        for r in rows:
            k = (int(r['farm_id']), int(r['year']))
            if k in s3_map:
                diffs_p11.append(abs(float(r['P11_kWh']) - s3_map[k]))
        if diffs_p11:
            import statistics
            print(f"    max={max(diffs_p11):.6f}, median={statistics.median(diffs_p11):.6f}")
            print(f"    精确匹配 (diff<1): {sum(1 for d in diffs_p11 if d < 1)}/{len(diffs_p11)}")
            ok = statistics.median(diffs_p11) < 0.01
            print(f"    => {'PASS' if ok else 'FAIL: P11≠S3!'}")

        # Check 2: P10 2017→2018 断点（同场配对检验）
        print("\n  [验收2] P10 2017→2018 连续性 (要求: ≤±1pp):")
        p10_2017 = {}; p10_2018 = {}
        p11_2017 = {}; p11_2018 = {}
        for r in rows:
            fid = int(r['farm_id']); yr = int(r['year'])
            if yr == 2017:
                p10_2017[fid] = float(r['P10_kWh'])
                p11_2017[fid] = float(r['P11_kWh'])
            elif yr == 2018:
                p10_2018[fid] = float(r['P10_kWh'])
                p11_2018[fid] = float(r['P11_kWh'])
        breaks = []
        for fid in set(p10_2017.keys()) & set(p10_2018.keys()):
            ratio_17 = p10_2017[fid] / p11_2017[fid] if p11_2017[fid] > 0 else 1.0
            ratio_18 = p10_2018[fid] / p11_2018[fid] if p11_2018[fid] > 0 else 1.0
            breaks.append((ratio_18 - ratio_17) * 100)  # percentage points
        if breaks:
            print(f"    N={len(breaks)}, mean={np.mean(breaks):.2f}pp, "
                  f"median={np.median(breaks):.2f}pp, "
                  f"max_abs={max(abs(np.array(breaks))):.2f}pp")
            ok = max(abs(np.array(breaks))) <= 1.0
            print(f"    => {'PASS' if ok else 'WARN: break > ±1pp'}")

        # Check 3: S_shapley sign
        print("\n  [验收3] S_shapley 逐年场均 (要求: 出现负值年份):")
        from collections import defaultdict as dd
        s_by_year = dd(list)
        for r in rows:
            yr = int(r['year'])
            p11 = float(r['P11_kWh']); p10 = float(r['P10_kWh'])
            p01 = float(r['P01_kWh']); p00 = float(r['P00_kWh'])
            s = ((p11 - p01) + (p10 - p00)) / 2.0
            d = ((p11 - p10) + (p01 - p00)) / 2.0
            delta = p11 - p00
            s_ratio = s / delta * 100 if abs(delta) > 1 else 0.0
            s_by_year[yr].append(s_ratio)
        for yr in sorted(s_by_year.keys()):
            vals = s_by_year[yr]
            print(f"    {yr}: mean S/Δ = {np.mean(vals):+.1f}%, "
                  f"median={np.median(vals):+.1f}%, "
                  f"n_pos={sum(1 for v in vals if v>0)}/{len(vals)}")

    print(f"\n[7/7] v6 脚本完成 ({elapsed/60:.0f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
