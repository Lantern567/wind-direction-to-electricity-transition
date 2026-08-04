"""
=============================================================================
 task2_floris.py — 任务二逐时尾流出力核算 (FLORIS v4.6)
 架构: 扇区分箱预计算 → 逐时查表回放 → 年度汇总
 依据: 任务书 §2.5-2.11 / 修改意见 §4.2 / 审计报告 §四
=============================================================================
"""
import os, sys, csv, math, time, tempfile, shutil, hashlib
import numpy as np
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import netCDF4
from netCDF4 import num2date

# ---- Add current dir for floris_config ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floris_config import (
    create_floris_model, load_task0_coordinates, load_farms_master,
    get_ti_for_farm, get_power_curve, get_turbine_params,
    WS_BINS, WD_SECTORS, WD_STEP,
    H_REF, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    TASK0_DIR, DATA_DIR, OUT_DIR, DEFAULT_TURBINE,
    classify_region,
)

# ---- Constants ----
ALL_YRS = list(range(2014, 2025))
N_CORES = max(1, min(cpu_count() - 2, 8))  # use up to 8 cores

# ---- ERA5 file path patterns ----
ERA5_PATTERNS = {
    "east_asia": "era5_east_asia_{year}.nc",
    "europe": "era5_europe_{year}.nc",
    "us_east": "era5_us_east_{year}.nc",
    "japan": "era5_japan_{year}.nc",
}


# =========================================================================
# 1. ERA5 WIND DATA ACCESS
# =========================================================================

def get_region_for_farm(lat, lon):
    """Map farm centroid to ERA5 region."""
    if 8 <= lat <= 44 and 104 <= lon <= 143:
        return "east_asia"
    if 39 <= lat <= 63 and -12 <= lon <= 32:
        return "europe"
    if 36 <= lat <= 42 and -78 <= lon <= -68:
        return "us_east"
    if 30 <= lat <= 46 and 128 <= lon <= 146:
        return "japan"
    return "east_asia"

def get_era5_nc_path(region, year):
    """Get ERA5 NetCDF file path for a given region and year."""
    pattern = ERA5_PATTERNS.get(region, ERA5_PATTERNS["east_asia"])
    filename = pattern.format(year=year)
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    return None

def extract_wind_series(nc_path, centroid_lat, centroid_lon):
    """Extract hourly (ws, wd) series from ERA5 NC file for a given lat/lon.

    Returns: (V_100m, theta_deg) as numpy arrays of length nhours.
    """
    # Copy to tempdir to handle Chinese path issues
    tmpdir = tempfile.mkdtemp()
    tmpnc = os.path.join(tmpdir, "data.nc")
    shutil.copy2(nc_path, tmpnc)
    ds = netCDF4.Dataset(tmpnc, 'r')

    lat_arr = ds['latitude'][:]
    lon_arr = ds['longitude'][:]
    ilat = int(np.argmin(np.abs(lat_arr - centroid_lat)))
    ilon = int(np.argmin(np.abs(lon_arr - centroid_lon)))

    u100 = np.array(ds['u100'][:, ilat, ilon], dtype=np.float64)
    v100 = np.array(ds['v100'][:, ilat, ilon], dtype=np.float64)

    ds.close()
    shutil.rmtree(tmpdir)

    ws = np.sqrt(u100**2 + v100**2)
    wd_rad = np.arctan2(v100, u100)
    wd_deg = (270.0 - np.degrees(wd_rad)) % 360.0  # meteorological convention

    return ws, wd_deg


# =========================================================================
# 2. POWER CURVE EVALUATION (no-wake)
# =========================================================================

def compute_no_wake_power(ws_array, turbine_params, n_turbines):
    """Compute no-wake farm power for each wind speed.

    Args:
        ws_array: wind speeds (m/s), shape (nhours,)
        turbine_params: dict with 'power_table'
        n_turbines: number of turbines

    Returns: P_noWake in kW, shape (nhours,)
    """
    ws_tbl = np.array(turbine_params['power_table']['wind_speed'])
    pw_tbl = np.array(turbine_params['power_table']['power'])

    # Linear interpolation of power curve
    per_turbine_power = np.interp(ws_array, ws_tbl, pw_tbl, left=0.0, right=0.0)
    return per_turbine_power * n_turbines  # kW


# =========================================================================
# 3. WAKE TABLE PRECOMPUTATION
# =========================================================================

def compute_config_hash(turbine_coords_x, turbine_coords_y,
                         turbine_type, ti, alpha,
                         wake_model_name, ws_bins, wd_sectors):
    """Generate MD5 hash for a wind farm + FLORIS configuration.

    Hash includes all factors that affect the wake efficiency matrix:
    turbine coordinates (sorted by x_m), turbine type, TI, shear,
    wake model, and wind bin definitions.

    Returns: 12-character hex string.
    """
    # Sort coordinates by x_m for deterministic ordering
    coords = sorted(zip(turbine_coords_x, turbine_coords_y), key=lambda c: c[0])
    coord_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in coords)

    parts = [
        coord_str,
        str(turbine_type),
        f"{ti:.4f}",
        f"{alpha:.4f}",
        str(wake_model_name),
        ",".join(f"{ws:.1f}" for ws in ws_bins),
        ",".join(f"{wd:.1f}" for wd in wd_sectors),
    ]
    config_str = "|".join(parts)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def precompute_wake_table(turbine_coords_x, turbine_coords_y,
                          turbine_type, ti, alpha,
                          ws_bins, wd_sectors,
                          wake_model_name="gauss",
                          return_extra=False):
    """Precompute wake efficiency for all (wd, ws) bin combinations.

    For each (wd_sector, ws_bin) combination:
      - Run FLORIS → get P_wake
      - P_noWake = n_turbines × power_curve(ws_bin)
      - η_wake = P_wake / P_noWake   (wake efficiency coefficient)

    Args:
        return_extra: if True, also returns (p_no_wake_bin, config_hash).

    Returns:
        wake_efficiency: 2D array (n_wd × n_ws) of η values
        total_time: wall time in seconds
        n_turbines: number of turbines
        (if return_extra) p_no_wake_bin: 1D array (n_ws,) of no-wake power per WS bin
        (if return_extra) config_hash: 12-char MD5 hash of this configuration
    """
    n_turbines = len(turbine_coords_x)
    n_wd = len(wd_sectors)
    n_ws = len(ws_bins)

    # Create FLORIS model once
    fm, tmp_path = create_floris_model(
        turbine_coords_x, turbine_coords_y,
        turbine_type=turbine_type, ti=ti, alpha=alpha,
        wake_model_name=wake_model_name
    )

    # Get power curve for no-wake calculation
    tp = get_turbine_params(turbine_type)
    ws_pc = np.array(tp['power_table']['wind_speed'])
    pw_pc = np.array(tp['power_table']['power'])

    # Compute P_noWake for each ws bin
    p_no_wake_bin = np.array([np.interp(ws, ws_pc, pw_pc, left=0.0, right=0.0)
                               for ws in ws_bins]) * n_turbines  # kW

    # Precompute wake efficiency matrix — one FLORIS run per WD sector
    # (batching all sectors at once is slower for large farms due to FLORIS
    #  super-linear scaling; per-sector gives best throughput across all sizes)
    wake_efficiency = np.ones((n_wd, n_ws), dtype=np.float64)

    ws_vals = np.array([float(ws) for ws in ws_bins], dtype=np.float64)
    ti_arr = np.full(n_ws, float(ti), dtype=np.float64)

    t0 = time.perf_counter()

    for iwd, wd in enumerate(wd_sectors):
        wd_arr = np.full(n_ws, float(wd), dtype=np.float64)

        fm.set(wind_directions=wd_arr, wind_speeds=ws_vals,
               turbulence_intensities=ti_arr)
        fm.run()

        farm_powers = fm.get_farm_power() / 1000.0  # W → kW, shape (n_ws,)

        # Vectorized efficiency: η = clip(P_wake / P_noWake, 0, 1), η=1 when P_noWake=0
        mask = p_no_wake_bin > 0
        wake_efficiency[iwd, mask] = np.clip(
            farm_powers[mask] / p_no_wake_bin[mask], 0.0, 1.0
        )
        # wake_efficiency[iwd, ~mask] stays 1.0 (initialized above)

    t1 = time.perf_counter()
    os.unlink(tmp_path)

    if return_extra:
        config_hash = compute_config_hash(
            turbine_coords_x, turbine_coords_y,
            turbine_type, ti, alpha, wake_model_name,
            ws_bins, wd_sectors
        )
        return wake_efficiency, (t1 - t0), n_turbines, p_no_wake_bin, config_hash

    return wake_efficiency, (t1 - t0), n_turbines


def precompute_wake_table_safe(*args, **kwargs):
    """Wrapper with error handling for multiprocessing."""
    try:
        return precompute_wake_table(*args, **kwargs)
    except Exception as e:
        print(f"  ERROR in precompute: {e}")
        return None


# =========================================================================
# 4. HOURLY REPLAY (bin lookup)
# =========================================================================

def replay_hourly_from_bins(ws_hourly, wd_hourly,
                             wake_efficiency, ws_bins, wd_sectors,
                             turbine_params, turbine_type,
                             n_turbines, capacity_kW,
                             farm_id, year):
    """Replay hourly output from precomputed wake efficiency table.

    For each hour:
      1. Find nearest ws_bin → get P_noWake
      2. Find nearest wd_sector → get η_wake
      3. P_wake(h) = P_noWake(V_h) × η_wake(WD_bin, WS_bin)
      4. Apply electrical losses

    Returns:
        results: dict with AEP, CF, WakeLoss, Volatility, CV, etc.
        hourly_rows: list of dicts for hourly CSV output
    """
    nh = len(ws_hourly)
    ws_bins_arr = np.array(ws_bins)
    wd_sectors_arr = np.array(wd_sectors)

    # Get power curve
    tp = get_turbine_params(turbine_type)
    ws_pc = np.array(tp['power_table']['wind_speed'])
    pw_pc = np.array(tp['power_table']['power'])

    # For each hour, compute P_noWake and P_wake
    p_no_wake_arr = np.zeros(nh)
    p_wake_arr = np.zeros(nh)

    for h in range(nh):
        ws = ws_hourly[h]
        wd = wd_hourly[h]

        # Bin wind speed and direction
        iws = np.argmin(np.abs(ws_bins_arr - ws))
        iwd = np.argmin(np.abs(wd_sectors_arr - wd))

        # No-wake power (direct power curve eval, don't bin it)
        p_no_wake_per_turbine = np.interp(ws, ws_pc, pw_pc, left=0.0, right=0.0)
        p_no_w = p_no_wake_per_turbine * n_turbines  # kW

        # Wake efficiency from precomputed table
        eta = wake_efficiency[iwd, iws]
        p_w = p_no_w * eta

        p_no_wake_arr[h] = p_no_w
        p_wake_arr[h] = p_w

    # Apply electrical losses
    p_no_wake_net = p_no_wake_arr * ELECTRICAL_LOSS  # kW
    p_wake_net = p_wake_arr * ELECTRICAL_LOSS  # kW

    # Annual metrics
    n_effective_hours = np.sum(ws_hourly >= 3.0)  # hours above cut-in
    aep = np.sum(p_wake_net)  # kWh
    aep_no_wake = np.sum(p_no_wake_net)
    cf = aep / (capacity_kW * n_effective_hours) if capacity_kW > 0 and n_effective_hours > 0 else 0.0
    wake_loss = (aep_no_wake - aep) / aep_no_wake if aep_no_wake > 0 else 0.0

    # Volatility metrics
    vol = float(np.std(p_wake_net))
    cv_val = vol / float(np.mean(p_wake_net)) if float(np.mean(p_wake_net)) > 0 else 0.0
    p5 = float(np.percentile(p_wake_net, 5))
    p95 = float(np.percentile(p_wake_net, 95))
    ramp = np.abs(np.diff(p_wake_net))
    ramp_freq = float(np.mean(ramp > 0.1 * capacity_kW)) if capacity_kW > 0 and len(ramp) > 0 else 0.0
    low_h = float(np.mean(p_wake_net < 0.1 * capacity_kW)) if capacity_kW > 0 else 0.0
    high_h = float(np.mean(p_wake_net > 0.9 * capacity_kW)) if capacity_kW > 0 else 0.0

    results = {
        'farm_id': farm_id, 'year': year,
        'n_turb': n_turbines, 'capacity_kW': capacity_kW,
        'n_hours': n_effective_hours, 'n_total_hours': nh,
        'AEP_kWh': aep,
        'AEP_noWake_kWh': aep_no_wake,
        'CF': cf,
        'WakeLoss': wake_loss,
        'Volatility_kW': vol,
        'CV': cv_val,
        'P5_kW': p5,
        'P95_kW': p95,
        'RampFreq': ramp_freq,
        'low_hours': low_h,
        'high_hours': high_h,
    }
    return results


# =========================================================================
# 5. MAIN COMPUTATION ORCHESTRATOR
# =========================================================================

def compute_farm_year(fid, yr, coords, farms, turbine_type, wake_models, ws_bins, wd_sectors):
    """Compute one farm-year with all wake models.

    Returns:
        list of result dicts (one per wake model)
    """
    turbs = coords.get(fid, {}).get(yr, [])
    if len(turbs) < 2:
        return []

    farm_info = farms.get(fid, {})
    lat = farm_info.get('centroid_lat', 0)
    lon = farm_info.get('centroid_lon', 0)
    country = farm_info.get('country', '')
    n_turb = len(turbs)
    # Capacity from actual turbine count for this year, not farms_master total
    tp = get_turbine_params(turbine_type)
    rated_power = tp['power_table'].get('controller_dependent_turbine_parameters', {}).get('rated_power', 10000)
    if rated_power == 0:
        rated_power = 10000
    capacity_kW = n_turb * rated_power

    ti = get_ti_for_farm(lat, lon)
    alpha = ALPHA_DEFAULT
    H_turb = tp['H']

    # Get ERA5 data
    region = get_region_for_farm(lat, lon)
    nc_path = get_era5_nc_path(region, yr)
    if nc_path is None:
        return []

    try:
        ws_100, wd_deg = extract_wind_series(nc_path, lat, lon)
    except Exception as e:
        print(f"  F{fid}/{yr}: ERA5 read error: {e}")
        return []

    # Height correction
    ws_hub = ws_100 * (H_turb / H_REF) ** alpha

    xs = [t['x_m'] for t in turbs]
    ys = [t['y_m'] for t in turbs]

    results = []
    for wm in wake_models:
        try:
            # Select bin resolution per model (jensen uses coarse bins for speed)
            from floris_config import get_bins_for_model
            ws_b, wd_b = get_bins_for_model(wm, n_turb)
            if n_turb > 400 or wm == 'jensen':
                print(f"  F{fid}/{yr}/{wm}: {len(ws_b)}x{len(wd_b)} combos ({n_turb}turb)")

            # Precompute wake table
            wake_eff, elapsed, _ = precompute_wake_table(
                xs, ys, turbine_type, ti, alpha, ws_b, wd_b, wm
            )

            # Replay hourly
            result = replay_hourly_from_bins(
                ws_hub, wd_deg, wake_eff, ws_b, wd_b,
                tp, turbine_type, n_turb, capacity_kW, fid, yr
            )
            result['wake_model'] = wm
            result['country'] = country
            result['TI'] = ti
            result['region'] = region
            result['turbine_type'] = turbine_type
            result['precompute_time_s'] = elapsed
            results.append(result)

        except Exception as e:
            print(f"  F{fid}/{yr}/{wm}: ERROR - {e}")

    return results


# =========================================================================
# 6. BATCH PROCESSING WITH CHECKPOINT/RESUME
# =========================================================================

# Representative farms for 3-model comparison and hourly output
REP_FARMS = {0, 2, 5, 15, 42, 88}  # large+medium farms across regions
NEAR_FARMS = {62, 46, 23}  # extra large farms (optional)

def main():
    t_start = datetime.now()
    print("=" * 60)
    print(" 任务二 FLORIS 逐时尾流出力核算 v2.0")
    print(f" 启动: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    print("\n[1/4] 加载任务零底座...")
    coords = load_task0_coordinates()
    farms = load_farms_master()
    print(f"  {len(farms)} farms, {sum(len(coords[fid].get(2024, [])) for fid in farms):,} turbines (2024)")

    ws_bins = np.array(WS_BINS, dtype=np.float64)
    wd_sectors = np.array(WD_SECTORS, dtype=np.float64)
    print(f"  分箱: {len(ws_bins)} WS x {len(wd_sectors)} WD = {len(ws_bins)*len(wd_sectors)} combos/farm-year")

    # Strategy: gauss + jensen for ALL farms, cc only for representative farms
    ALL_MODELS = ["gauss", "jensen"]
    EXTRA_MODELS = ["cc"]
    turbine_type = DEFAULT_TURBINE

    # Build task list — sorted by turbine count (smallest first for fast progress)
    print("\n[2/4] 构建任务列表...")
    tasks_all = []    # gauss+jensen for all farms
    tasks_extra = []  # cc for rep farms

    for fid in sorted(farms.keys()):
        for yr in ALL_YRS:
            turbs = coords.get(fid, {}).get(yr, [])
            if len(turbs) >= 2:
                tasks_all.append((fid, yr, len(turbs)))
                if fid in REP_FARMS:
                    tasks_extra.append((fid, yr, len(turbs)))

    # Sort by turbine count: fastest first
    tasks_all.sort(key=lambda x: x[2])
    tasks_extra.sort(key=lambda x: x[2])

    total_records = len(tasks_all) * len(ALL_MODELS) + len(tasks_extra) * len(EXTRA_MODELS)
    print(f"  Gauss+Jensen (全量): {len(tasks_all)} farm-years x 2 = {len(tasks_all)*2} records")
    print(f"  CC (代表场): {len(tasks_extra)} farm-years x 1 = {len(tasks_extra)} records")
    print(f"  总计: {total_records} records, {len(ws_bins)} WS × {len(wd_sectors)} WD = {len(ws_bins)*len(wd_sectors)} combos/farm-year")

    if not tasks_all:
        print("  无待计算任务!")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_annual = os.path.join(OUT_DIR, "task2_annual_floris.csv")

    # Checkpoint: load already-completed entries
    completed_set = set()
    if os.path.exists(csv_annual):
        with open(csv_annual, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed_set.add((int(r['farm_id']), int(r['year']), r['wake_model']))

    # Write header if new file
    if not os.path.exists(csv_annual) or os.path.getsize(csv_annual) == 0:
        with open(csv_annual, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['farm_id', 'year', 'n_turb', 'capacity_kW', 'n_hours',
                        'wake_model', 'turbine_type', 'country', 'region', 'TI',
                        'AEP_kWh', 'AEP_noWake_kWh', 'CF', 'WakeLoss',
                        'Volatility_kW', 'CV', 'P5_kW', 'P95_kW',
                        'RampFreq', 'low_hours', 'high_hours',
                        'precompute_time_s'])

    remaining = []
    for fid, yr, nt in tasks_all:
        for wm in ALL_MODELS:
            if (fid, yr, wm) not in completed_set:
                remaining.append((fid, yr, nt, [wm]))
    for fid, yr, nt in tasks_extra:
        for wm in EXTRA_MODELS:
            if (fid, yr, wm) not in completed_set:
                remaining.append((fid, yr, nt, [wm]))

    print(f"  已完成: {len(completed_set)}, 剩余: {len(remaining)} farm-year-model 对")
    if not remaining:
        print("  全部已完成!")
        return

    # Process
    print(f"\n[3/4] 开始计算...")
    t_batch_start = time.perf_counter()
    done = 0

    for idx, (fid, yr, n_turb, wake_list) in enumerate(remaining):
        try:
            results = compute_farm_year(fid, yr, coords, farms, turbine_type,
                                         wake_list, ws_bins, wd_sectors)
        except Exception as e:
            print(f"  F{fid}/{yr}: CRASH {e}")
            continue

        for r in results:
            with open(csv_annual, 'a', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow([r['farm_id'], r['year'], r['n_turb'],
                           r['capacity_kW'], r['n_hours'],
                           r['wake_model'], r['turbine_type'],
                           r['country'], r['region'], r['TI'],
                           r['AEP_kWh'], r['AEP_noWake_kWh'],
                           r['CF'], r['WakeLoss'],
                           r['Volatility_kW'], r['CV'],
                           r['P5_kW'], r['P95_kW'],
                           r['RampFreq'], r['low_hours'], r['high_hours'],
                           r['precompute_time_s']])

        done += len(results)
        elapsed = time.perf_counter() - t_batch_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else 0

        if (idx + 1) % 5 == 0 or idx < 3:
            print(f"  [{idx+1}/{len(remaining)}] F{fid}/{yr}: {n_turb}turb "
                  f"({','.join(wake_list)}) | {done}记录 | "
                  f"速率 {rate:.2f}/s | ETA {eta/3600:.1f}h")

    t_end = datetime.now()
    elapsed_total = (t_end - t_start).total_seconds()
    print(f"\n[4/4] 完成! {done} records in {elapsed_total/3600:.1f}h")
    print(f"  年度表: {csv_annual}")


if __name__ == "__main__":
    main()
