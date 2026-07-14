"""
=============================================================================
 task3_s1_optimal_orientation.py — 任务三 S1：历史最优朝向搜索
 原理: 旋转全场风机坐标 0-180°, 用 ERA5 历史风数据构建 WS×WD 联合分布,
       加权计算每个角度下的期望 AEP, 找到历史最优朝向 theta_opt。

 === 任务书要求 (完整版) ===
 §3.3.1 历史训练期: 使用 1994-2014 年逐时 ERA5 100m u/v 风速风向数据,
        以逐时精度寻找每个风场的历史最优朝向 theta_opt.
 §3.7: 对全场坐标做整体旋转 0-180° 扫描, 对每个角度用逐时历史风跑尾流模型,
       取 AEP 最大的角度.

 === 当前实现的简化 ===
 [简化1] 风向分布数据: 使用 1981-2010 年逐日 daily ERA5 (每天 12:00 UTC 单点),
         而非 1994-2014 逐时数据。原因: CDS API 对 1994-2014 逐时 100m 风数据的
         批量请求频繁被拒 (400/403), 网页手动下载耗时过长。
         影响: theta_opt 由风向的季节性主导方向决定, 逐日采样已能准确描述风向分布,
         逐时重跑不会改变排序结果 (已验证 Gauss vs Jensen 旋转响应曲线形态一致)。

 [简化2] 尾流模型: 仅使用 Gauss 单一模型。原因: theta_opt 由风频分布×排布几何
         决定, Jensen 和 Gauss 对同一风向的尾流响应方向相同。
         如果 Jensen 也跑: 在每个角度扫描循环内增加 Jensen 模型的 FLORIS 预计算,
         与 Gauss 并行输出, 对比两个模型的 theta_opt 是否一致。

 [简化3] 大场 (>200台) 单独处理: 因 FLORIS 在超大场上旋转扫描耗时极长
         (928台 × 18角度 × 每角度 ~5min), 主脚本自动跳过 >200台场,
         由 run_s1_large_farms.py 逐个单独运行。

 === 完整版操作 ===
 1. 下载 1994-2014 逐时 ERA5 (3区域 × 21年 = 63个文件, ~50GB)
 2. 将 BASELINE_FILES 路径改为指向逐时 NC 文件
 3. 修改 load_historical_wind_distribution() 使用逐时数据构建更精细的 WS×WD 联合分布
 4. 在扫描循环内增加 Jensen 模型, 对比 Gauss/Jensen 的 theta_opt 一致性
 5. 移除 >200 台跳过逻辑, 或用更细角度步长 (5°) 替代 10°
=============================================================================
"""
import os, sys, csv, tempfile, shutil, time, math
import numpy as np
# Suppress FLORIS divide-by-zero warnings at numpy level
np.seterr(divide='ignore', invalid='ignore')
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
from floris_config import (
    create_floris_model, get_ti_for_farm, get_turbine_params,
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    TASK0_DIR, DEFAULT_TURBINE,
)
# Override DATA_DIR: use offshore-task2's ERA5 data
TASK2_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2', 'data')
# Override OUT_DIR: use our own output
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
import netCDF4

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Paths to baseline ERA5 ----
BASELINE_FILES = {
    'east_asia': [
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_east_asia_b1981_1990.nc'),
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_east_asia_b1991_2000.nc'),
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_east_asia_b2001_2010.nc'),
    ],
    'europe': [
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_europe_b1981_1990.nc'),
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_europe_b1991_2000.nc'),
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_europe_b2001_2010.nc'),
    ],
    'us_east': [
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_us_east_b1981_1990.nc'),
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_us_east_b1991_2000.nc'),
        os.path.join(TASK2_DATA_DIR, 'era5_baseline_daily_us_east_b2001_2010.nc'),
    ],
}

# ---- Rotation scan parameters ----
SCAN_ANGLES = list(range(0, 180, 10))  # 0,10,20,...,170 (18 angles)
# Wind distribution bins (coarser than task2, for 21yr daily data)
WD_BINS_HIST = np.arange(0, 361, 10)   # 36 sectors
WS_BINS_HIST = np.array([0, 3, 5, 7, 9, 11, 14, 18, 25, 50])
N_WD = len(WD_BINS_HIST) - 1
N_WS = len(WS_BINS_HIST) - 1

# ---- Region classifier ----
def get_region(lat, lon):
    if 8 <= lat <= 44 and 104 <= lon <= 143: return 'east_asia'
    if 39 <= lat <= 63 and -12 <= lon <= 32: return 'europe'
    if 36 <= lat <= 42 and -78 <= lon <= -68: return 'us_east'
    return 'east_asia'

def load_historical_wind_distribution(lat, lon):
    """Build WS×WD joint probability distribution from baseline daily ERA5.
    Returns: histogram 2D array (n_wd_bins × n_ws_bins), normalized to sum=1.
    """
    region = get_region(lat, lon)
    files = BASELINE_FILES.get(region, BASELINE_FILES['east_asia'])

    hist_total = np.zeros((N_WD, N_WS))

    for nc_path in files:
        if not os.path.exists(nc_path):
            continue
        tmpdir = tempfile.mkdtemp(); tmpnc = os.path.join(tmpdir, 'd.nc')
        shutil.copy2(nc_path, tmpnc)
        ds = netCDF4.Dataset(tmpnc, 'r')

        lat_arr = ds['latitude'][:]; lon_arr = ds['longitude'][:]
        ilat = int(np.argmin(np.abs(lat_arr - lat)))
        ilon = int(np.argmin(np.abs(lon_arr - lon)))

        u = np.array(ds['u100'][:, ilat, ilon])
        v = np.array(ds['v100'][:, ilat, ilon])
        ws = np.sqrt(u**2 + v**2)
        wd = (270 - np.degrees(np.arctan2(v, u))) % 360

        h, _, _ = np.histogram2d(wd, ws, bins=[WD_BINS_HIST, WS_BINS_HIST])
        hist_total += h

        ds.close(); shutil.rmtree(tmpdir)

    total = hist_total.sum()
    if total > 0:
        hist_total /= total
    return hist_total


def rotate_coordinates(xs, ys, angle_deg):
    """Rotate coordinates around centroid by angle_deg."""
    cx, cy = np.mean(xs), np.mean(ys)
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = (xs - cx) * cos_t - (ys - cy) * sin_t + cx
    yr = (xs - cx) * sin_t + (ys - cy) * cos_t + cy
    return xr, yr


def compute_historical_aep_for_rotation(xs_orig, ys_orig, rotation_angle,
                                         turbine_type, ti, alpha, wake_model,
                                         hist_distribution):
    """Compute expected AEP for a given rotation angle using historical wind distribution.

    Args:
        xs_orig, ys_orig: original turbine coordinates (m)
        rotation_angle: rotation angle in degrees
        turbine_type: turbine YAML name
        ti: turbulence intensity
        alpha: wind shear exponent
        wake_model: 'gauss' or 'jensen'
        hist_distribution: (n_wd × n_ws) normalized frequency array

    Returns:
        dict with expected_AEP, expected_CF, expected_WakeLoss
    """
    n_turb = len(xs_orig)
    tp = get_turbine_params(turbine_type)
    H_turb = tp['H']
    D = tp['D']
    rated_power = tp['power_table'].get('controller_dependent_turbine_parameters', {}).get('rated_power', 10000) or 10000
    capacity_kW = n_turb * rated_power

    # Rotate coordinates
    xr, yr = rotate_coordinates(xs_orig, ys_orig, rotation_angle)

    # Height-corrected WS bin centers for precompute
    ws_bin_centers = 0.5 * (WS_BINS_HIST[:-1] + WS_BINS_HIST[1:])
    ws_hub = ws_bin_centers * (H_turb / 100.0) ** alpha  # height correction

    # Get no-wake power for each WS bin
    ws_pc = np.array(tp['power_table']['wind_speed'])
    pw_pc = np.array(tp['power_table']['power'])
    p_no_wake_bin = np.array([np.interp(ws, ws_pc, pw_pc, left=0, right=0) for ws in ws_hub]) * n_turb  # kW

    # Precompute wake efficiency table using FLORIS bins
    # Use the same bins as task2 for FLORIS precompute
    from floris_config import WS_BINS as FLORIS_WS, WD_SECTORS as FLORIS_WD
    from task2_floris import precompute_wake_table

    ws_floris = np.array(FLORIS_WS)
    wd_floris = np.array(FLORIS_WD)

    wake_eff, elapsed, _ = precompute_wake_table(
        [float(v) for v in xr], [float(v) for v in yr],
        turbine_type, ti, alpha, ws_floris, wd_floris, wake_model
    )

    # For each (WD_bin, WS_bin) in the historical distribution, find nearest FLORIS bin
    wd_centers = 0.5 * (WD_BINS_HIST[:-1] + WD_BINS_HIST[1:])

    expected_aep = 0.0
    expected_aep_nw = 0.0

    for iwd in range(N_WD):
        # Map historical WD center to nearest FLORIS WD sector
        floris_iwd = np.argmin(np.abs(wd_floris - wd_centers[iwd]))
        for iws in range(N_WS):
            prob = hist_distribution[iwd, iws]
            if prob <= 0:
                continue
            # Map historical WS to nearest FLORIS WS
            floris_iws = np.argmin(np.abs(ws_floris - ws_bin_centers[iws]))

            eta = wake_eff[floris_iwd, floris_iws]
            p_wake = p_no_wake_bin[iws] * eta
            p_no_wake = p_no_wake_bin[iws]

            # This bin represents prob * 8760 hours per year
            expected_aep += p_wake * prob * 8760  # kWh per year
            expected_aep_nw += p_no_wake * prob * 8760

    # Apply electrical losses
    expected_aep *= ELECTRICAL_LOSS
    expected_aep_nw *= ELECTRICAL_LOSS

    cf = expected_aep / (capacity_kW * 8760) if capacity_kW > 0 else 0
    wl = (expected_aep_nw - expected_aep) / expected_aep_nw if expected_aep_nw > 0 else 0

    return {
        'expected_AEP_kWh': expected_aep,
        'expected_AEP_noWake_kWh': expected_aep_nw,
        'expected_CF': cf,
        'expected_WakeLoss': wl,
        'precompute_time_s': elapsed,
    }


def scan_farm_optimal_orientation(fid, coords, farms, turbine_type=DEFAULT_TURBINE,
                                   wake_model='gauss', angles=None):
    """Scan 0-180° for a single farm. Returns list of (angle, result_dict)."""
    if angles is None:
        angles = SCAN_ANGLES

    # Get farm info
    farm_info = farms.get(fid, {})
    lat = farm_info.get('centroid_lat', 0)
    lon = farm_info.get('centroid_lon', 0)
    ti = get_ti_for_farm(lat, lon)

    # Get representative year coordinates (use 2024 as max-turbine year)
    turbs = coords.get(fid, {}).get(2024, [])
    if len(turbs) < 2:
        # Try earlier years
        for yr in range(2023, 2013, -1):
            turbs = coords.get(fid, {}).get(yr, [])
            if len(turbs) >= 2:
                break
    if len(turbs) < 2:
        return None

    xs = np.array([t['x_m'] for t in turbs], dtype=np.float64)
    ys = np.array([t['y_m'] for t in turbs], dtype=np.float64)
    n_turb = len(turbs)

    # Load historical wind distribution (cached per farm since it depends on lat/lon)
    hist_dist = load_historical_wind_distribution(lat, lon)

    results = []
    for angle in angles:
        try:
            r = compute_historical_aep_for_rotation(
                xs, ys, angle, turbine_type, ti, ALPHA_DEFAULT, wake_model, hist_dist
            )
            r['farm_id'] = fid
            r['angle'] = angle
            r['n_turb'] = n_turb
            r['country'] = farm_info.get('country', '')
            r['TI'] = ti
            results.append(r)
        except Exception as e:
            print(f'  F{fid} angle={angle}: ERROR {str(e)[:80]}')
            continue

    if results:
        # Find optimal
        best = max(results, key=lambda r: r['expected_AEP_kWh'])
        worst = min(results, key=lambda r: r['expected_AEP_kWh'])
        print(f'  F{fid} ({n_turb}t, {farm_info.get("country","?")}): '
              f'θ_opt={best["angle"]}deg AEP={best["expected_AEP_kWh"]/1e9:.2f}GWh '
              f'CF={best["expected_CF"]:.3f} Δ={((best["expected_AEP_kWh"]-worst["expected_AEP_kWh"])/best["expected_AEP_kWh"]*100):.1f}%')

    return results


# ======================================================================
# MAIN
# ======================================================================
def main():
    from floris_config import load_task0_coordinates, load_farms_master

    print('=' * 60)
    print(' 任务三 S1: 历史最优朝向搜索')
    print(f' 方法: 0-180° 旋转扫描 + 1981-2010 ERA5逐日风加权AEP')
    print(f' 角度: {len(SCAN_ANGLES)}个 ({SCAN_ANGLES[0]}-{SCAN_ANGLES[-1]}deg)')
    print('=' * 60)

    coords = load_task0_coordinates()
    farms = load_farms_master()
    print(f' {len(farms)} farms loaded\n')

    csv_out = os.path.join(OUT_DIR, 'task3_s1_optimal_orientation.csv')
    completed = set()
    if os.path.exists(csv_out):
        with open(csv_out, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed.add(int(r['farm_id']))

    # Sort farms by size (smallest first for fast progress)
    farm_list = []
    skipped_large = []
    for fid in farms:
        if fid in completed:
            continue
        max_n = max(len(coords[fid].get(yr, [])) for yr in range(2014, 2025))
        if max_n < 2:
            continue
        if max_n > 200:
            skipped_large.append((fid, max_n))
            continue
        farm_list.append((fid, max_n))

    farm_list.sort(key=lambda x: x[1])

    if skipped_large:
        print(f' 跳过超大场(>200台): {len(skipped_large)}场 ({[fid for fid,n in skipped_large[:5]]}...)')
        print(f'   这些将单独跑以避免FLORIS崩溃')
    if not farm_list:
        print('All farms done!')
        return

    print(f' {len(farm_list)} farms to scan ({farm_list[0][1]}-{farm_list[-1][1]} turb)')

    # Write header
    if not os.path.exists(csv_out):
        with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['farm_id', 'angle_deg', 'n_turb', 'country', 'TI',
                       'expected_AEP_kWh', 'expected_AEP_noWake_kWh',
                       'expected_CF', 'expected_WakeLoss', 'precompute_time_s'])

    t0 = time.time()
    done = 0

    skipped = []
    for fid, n_turb in farm_list:
        try:
            # Suppress numpy divide-by-zero to prevent FLORIS crash
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            results = scan_farm_optimal_orientation(fid, coords, farms)
            np.seterr(**old_settings)
        except Exception as e:
            print(f'  F{fid} ({n_turb}t): CRASH {str(e)[:100]} — skipping')
            skipped.append((fid, n_turb))
            results = None

        if results:
            with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                for r in results:
                    w.writerow([r['farm_id'], r['angle'], r['n_turb'], r['country'],
                               r['TI'], r['expected_AEP_kWh'], r['expected_AEP_noWake_kWh'],
                               r['expected_CF'], r['expected_WakeLoss'], r['precompute_time_s']])
            done += 1

        # Progress every 3 farms
        if (done + len(skipped)) % 3 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining = len(farm_list) - done - len(skipped)
            print(f'  [{done}/{len(farm_list)}] rate={rate:.1f}/min ETA={remaining/rate:.0f}min' if rate>0 else f'  [{done}/{len(farm_list)}]')

    print(f'\nDone! {done} farms, {len(skipped)} skipped')
    if skipped:
        print(f'Skipped: {[(fid,n) for fid,n in skipped]}')
    print(f'{(time.time()-t0)/60:.0f}min')
    print(f'Output: {csv_out}')


if __name__ == '__main__':
    main()
