"""
=============================================================================
 task2_audit_v2.py — 任务二审计证据 (FLORIS版)
 1. 旋转对照: 0-180° 扫描, 证明几何被尾流模型使用
 2. 打乱对照: 真正打乱坐标, 证明坐标空间排列影响结果
 3. 精度自检: 逐时精算 vs 分箱查表, 量化误差
 4. 产出 farm_layout_used.csv
 依据: 修改意见 §4.1 (核查清单) / 审计报告 §四 (P2)
=============================================================================
"""
import os, sys, csv, math, time, tempfile
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floris_config import (
    create_floris_model, load_task0_coordinates, load_farms_master,
    get_ti_for_farm, get_turbine_params, get_power_curve,
    WS_BINS, WD_SECTORS, WD_STEP,
    H_REF, ALPHA_DEFAULT, ELECTRICAL_LOSS, KA_GAUSS, KB_GAUSS,
    TASK0_DIR, DATA_DIR, OUT_DIR, DEFAULT_TURBINE,
    classify_region,
)
from task2_floris import (
    extract_wind_series, get_region_for_farm, get_era5_nc_path,
    precompute_wake_table, compute_no_wake_power,
    replay_hourly_from_bins,
)

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================================
# 1. ROTATION TEST
# =========================================================================

def rotation_test(turbine_coords_x, turbine_coords_y, turbine_type, ti, alpha,
                  wake_model="gauss", ws_target=8.0, n_angles=19):
    """Rotate entire wind farm and measure AEP at fixed wind speed.

    Rotates the turbine coordinates from 0 to 180 degrees in steps,
    keeps wind direction fixed at 270°, and computes total farm power.

    Returns: list of (rotation_angle, farm_power_kW)
    """
    xs = np.array(turbine_coords_x, dtype=np.float64)
    ys = np.array(turbine_coords_y, dtype=np.float64)
    n_turb = len(xs)

    angles = np.linspace(0, 180, n_angles)
    results = []

    tp = get_turbine_params(turbine_type)
    p_no_wake = compute_no_wake_power(np.array([ws_target]), tp, n_turb)[0]

    for rot in angles:
        theta = np.radians(rot)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotate coordinates around centroid
        cx, cy = xs.mean(), ys.mean()
        xr = (xs - cx) * cos_t - (ys - cy) * sin_t + cx
        yr = (xs - cx) * sin_t + (ys - cy) * cos_t + cy

        fm, tmp = create_floris_model([float(v) for v in xr], [float(v) for v in yr],
                                       turbine_type=turbine_type,
                                       ti=ti, alpha=alpha, wake_model_name=wake_model)
        fm.set(wind_speeds=[ws_target], wind_directions=[270.0],
               turbulence_intensities=[ti])
        fm.run()
        p_wake = fm.get_farm_power()[0] / 1000.0  # W → kW
        os.unlink(tmp)

        eta = p_wake / p_no_wake if p_no_wake > 0 else 1.0
        results.append((float(rot), float(p_wake), float(p_no_wake), float(eta)))

    return results


# =========================================================================
# 2. SHUFFLE TEST (FIXED VERSION)
# =========================================================================

def shuffle_test(turbine_coords_x, turbine_coords_y, turbine_type, ti, alpha,
                 wake_model="gauss", ws_target=8.0, n_shuffles=20):
    """Randomly shuffle turbine positions and measure AEP.

    The OLD version (check audit report) had a bug: pos_real[perm] = same set of points.
    THIS version truly randomizes positions within the bounding box.

    Returns: dict with statistics
    """
    xs = np.array(turbine_coords_x, dtype=np.float64)
    ys = np.array(turbine_coords_y, dtype=np.float64)
    n_turb = len(xs)
    tp = get_turbine_params(turbine_type)
    p_no_wake = compute_no_wake_power(np.array([ws_target]), tp, n_turb)[0]

    # Real layout power
    fm, tmp = create_floris_model([float(v) for v in xs], [float(v) for v in ys],
                                   turbine_type=turbine_type,
                                   ti=ti, alpha=alpha, wake_model_name=wake_model)
    fm.set(wind_speeds=[ws_target], wind_directions=[270.0],
           turbulence_intensities=[ti])
    fm.run()
    p_real = fm.get_farm_power()[0] / 1000.0
    os.unlink(tmp)

    # Shuffled layouts
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    p_shuffled = []

    for seed in range(n_shuffles):
        rng = np.random.RandomState(seed)
        xs_shuf = rng.uniform(xmin, xmax, n_turb)
        ys_shuf = rng.uniform(ymin, ymax, n_turb)

        fm, tmp = create_floris_model([float(v) for v in xs_shuf], [float(v) for v in ys_shuf],
                                       turbine_type=turbine_type,
                                       ti=ti, alpha=alpha, wake_model_name=wake_model)
        fm.set(wind_speeds=[ws_target], wind_directions=[270.0],
               turbulence_intensities=[ti])
        fm.run()
        p_shuf = fm.get_farm_power()[0] / 1000.0
        os.unlink(tmp)
        p_shuffled.append(p_shuf)

    p_shuffled = np.array(p_shuffled)
    return {
        'p_real': p_real,
        'p_no_wake': p_no_wake,
        'p_shuffled_mean': float(p_shuffled.mean()),
        'p_shuffled_std': float(p_shuffled.std()),
        'p_shuffled_min': float(p_shuffled.min()),
        'p_shuffled_max': float(p_shuffled.max()),
        'delta_pct': float((p_real - p_shuffled.mean()) / p_real * 100),
        'real_wake_loss': float((p_no_wake - p_real) / p_no_wake * 100),
        'shuffled_wake_loss': float((p_no_wake - p_shuffled.mean()) / p_no_wake * 100),
        'n_shuffles': n_shuffles,
    }


# =========================================================================
# 3. PRECISION SELF-CHECK: hourly vs bin-lookup
# =========================================================================

def precision_check(fid, yr, coords, farms, turbine_type=DEFAULT_TURBINE,
                    wake_models=["gauss"], max_hours=500):
    """Compare bin-lookup vs exact hourly FLORIS for a subset of hours.

    Runs FLORIS on the first `max_hours` hours directly (no binning),
    and compares against the bin-lookup approximation.

    Returns: dict with relative errors for AEP and WakeLoss.
    """
    turbs = coords.get(fid, {}).get(yr, [])
    if len(turbs) < 2:
        return None

    farm_info = farms.get(fid, {})
    lat = farm_info['centroid_lat']
    lon = farm_info['centroid_lon']
    ti = get_ti_for_farm(lat, lon)
    alpha = ALPHA_DEFAULT
    tp = get_turbine_params(turbine_type)
    H_turb = tp['H']
    n_t = len(turbs)

    region = get_region_for_farm(lat, lon)
    nc_path = get_era5_nc_path(region, yr)
    if nc_path is None:
        return None

    ws_100, wd_deg = extract_wind_series(nc_path, lat, lon)
    ws_hub = ws_100 * (H_turb / H_REF) ** alpha
    ws_hub = ws_hub[:max_hours]
    wd_deg = wd_deg[:max_hours]

    xs = [t['x_m'] for t in turbs]
    ys = [t['y_m'] for t in turbs]

    results = {}
    for wm in wake_models:
        # ---- Bin-lookup method ----
        ws_bins = np.array(WS_BINS, dtype=np.float64)
        wd_sectors = np.array(WD_SECTORS, dtype=np.float64)
        wake_eff, _, _ = precompute_wake_table(xs, ys, turbine_type, ti, alpha,
                                                ws_bins, wd_sectors, wm)
        capacity_kW = n_t * tp['power_table'].get('controller_dependent_turbine_parameters', {}).get('rated_power', 10000)
        if capacity_kW == 0:
            capacity_kW = n_t * 10000

        result_bin = replay_hourly_from_bins(
            ws_hub, wd_deg, wake_eff, ws_bins, wd_sectors,
            tp, turbine_type, n_t, capacity_kW, fid, yr
        )

        # ---- Exact hourly method ----
        fm, tmp_path = create_floris_model(xs, ys, turbine_type=turbine_type,
                                            ti=ti, alpha=alpha, wake_model_name=wm)
        p_no_wake_exact_arr = np.zeros(max_hours)
        p_wake_exact_arr = np.zeros(max_hours)

        for h in range(max_hours):
            ws_h = ws_hub[h]
            wd_h = wd_deg[h]
            fm.set(wind_speeds=[float(ws_h)], wind_directions=[float(wd_h)],
                   turbulence_intensities=[float(ti)])
            fm.run()
            p_wake = fm.get_farm_power()[0] / 1000.0  # W → kW
            per_turbine = np.interp(ws_h,
                                     np.array(tp['power_table']['wind_speed']),
                                     np.array(tp['power_table']['power']),
                                     left=0.0, right=0.0)
            p_no_w = per_turbine * n_t
            p_wake_exact_arr[h] = p_wake
            p_no_wake_exact_arr[h] = p_no_w

        os.unlink(tmp_path)

        # Apply losses
        p_wake_exact_net = p_wake_exact_arr * ELECTRICAL_LOSS
        p_no_wake_exact_net = p_no_wake_exact_arr * ELECTRICAL_LOSS

        aep_wake_bin = result_bin['AEP_kWh']
        aep_wake_exact = float(np.sum(p_wake_exact_net))

        aep_no_wake_bin = result_bin.get('AEP_noWake_kWh',
            float(np.sum(compute_no_wake_power(ws_hub, tp, n_t) * ELECTRICAL_LOSS)))
        aep_no_wake_exact = float(np.sum(p_no_wake_exact_net))

        wl_bin = result_bin['WakeLoss']
        wl_exact = (aep_no_wake_exact - aep_wake_exact) / aep_no_wake_exact if aep_no_wake_exact > 0 else 0.0

        delta_aep_pct = (aep_wake_bin - aep_wake_exact) / aep_wake_exact * 100 if aep_wake_exact > 0 else 0.0
        delta_wl_pp = (wl_bin - wl_exact) * 100  # percentage points

        results[wm] = {
            'aep_bin_kWh': aep_wake_bin,
            'aep_exact_kWh': aep_wake_exact,
            'delta_aep_pct': delta_aep_pct,
            'wl_bin': wl_bin,
            'wl_exact': wl_exact,
            'delta_wl_pp': delta_wl_pp,
            'n_hours_checked': max_hours,
        }

    return results


# =========================================================================
# 4. FARM LAYOUT AUDIT CSV
# =========================================================================

def write_farm_layout_used(coords, farms, output_path=None):
    """Write farm_layout_used.csv — the audit evidence file.

    Format: farm_id, turbine_id, x_m, y_m, year, utm_epsg, farm_country
    """
    if output_path is None:
        output_path = os.path.join(OUT_DIR, "farm_layout_used.csv")

    rows = []
    for fid in sorted(coords.keys()):
        country = farms.get(fid, {}).get('country', '')
        for yr in sorted(coords[fid].keys()):
            for t in coords[fid][yr]:
                rows.append({
                    'farm_id': fid,
                    'turbine_id': t['turbine_id'],
                    'x_m': t['x_m'],
                    'y_m': t['y_m'],
                    'year': yr,
                    'utm_epsg': t.get('utm_epsg', ''),
                    'country': country,
                })

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['farm_id', 'turbine_id', 'x_m', 'y_m',
                                           'year', 'utm_epsg', 'country'])
        w.writeheader()
        w.writerows(rows)

    print(f"farm_layout_used.csv: {len(rows)} rows written to {output_path}")
    return output_path


# =========================================================================
# 5. MAIN AUDIT RUNNER
# =========================================================================

def run_audits():
    """Run all audit tests and write results."""
    print("=" * 60)
    print(" 任务二 审计证据 v2.0 (FLORIS)")
    print("=" * 60)

    coords = load_task0_coordinates()
    farms = load_farms_master()

    # ---- 1. Write farm_layout_used.csv ----
    print("\n--- 1. farm_layout_used.csv ---")
    write_farm_layout_used(coords, farms)

    # ---- 2. Pick representative farms ----
    # F0 (928 large grid), F2 (572 cluster), F5 (339 belt), F15 (293 multi)
    REP_FARMS = [0, 2, 5, 15]
    TEST_YEAR = 2024

    print("\n--- 2. Rotation Test ---")
    rot_all = {}
    for fid in REP_FARMS:
        turbs = coords.get(fid, {}).get(TEST_YEAR, [])
        if len(turbs) < 3:
            continue
        farm_info = farms[fid]
        ti = get_ti_for_farm(farm_info['centroid_lat'], farm_info['centroid_lon'])
        xs = [t['x_m'] for t in turbs]
        ys = [t['y_m'] for t in turbs]
        print(f"  Rotating F{fid} ({len(turbs)} turbines)...")
        rot = rotation_test(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT,
                            wake_model="gauss", ws_target=8.0)
        rot_all[fid] = rot

        # Find best angle
        best = max(rot, key=lambda r: r[1])
        worst = min(rot, key=lambda r: r[1])
        print(f"    Best AEP at {best[0]:.0f}deg: {best[1]:.0f}kW, "
              f"Worst at {worst[0]:.0f}deg: {worst[1]:.0f}kW, "
              f"spread={((best[1]-worst[1])/best[1]*100):.1f}%")

    # Save rotation results
    rot_path = os.path.join(OUT_DIR, "audit_rotation_floris.csv")
    with open(rot_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['farm_id', 'angle_deg', 'p_wake_kW', 'p_no_wake_kW', 'wake_efficiency'])
        for fid, rows in sorted(rot_all.items()):
            for angle, pw, pn, eta in rows:
                w.writerow([fid, angle, pw, pn, eta])
    print(f"  Saved: {rot_path}")

    # ---- 3. Shuffle Test ----
    print("\n--- 3. Shuffle Test ---")
    shuffle_results = []
    for fid in REP_FARMS[:2]:  # only 2 largest for speed
        turbs = coords.get(fid, {}).get(TEST_YEAR, [])
        if len(turbs) < 5:
            continue
        farm_info = farms[fid]
        ti = get_ti_for_farm(farm_info['centroid_lat'], farm_info['centroid_lon'])
        xs = [t['x_m'] for t in turbs]
        ys = [t['y_m'] for t in turbs]
        print(f"  Shuffling F{fid} ({len(turbs)} turbines, 20 shuffles)...")
        sh = shuffle_test(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT,
                          wake_model="gauss", ws_target=8.0, n_shuffles=20)
        sh['farm_id'] = fid
        sh['n_turb'] = len(turbs)
        shuffle_results.append(sh)
        print(f"    Real WL={sh['real_wake_loss']:.1f}%, Shuffled WL={sh['shuffled_wake_loss']:.1f}%, "
              f"Delta={sh['delta_pct']:.2f}%")

    shuf_path = os.path.join(OUT_DIR, "audit_shuffle_floris.csv")
    with open(shuf_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(shuffle_results[0].keys()))
        w.writeheader()
        w.writerows(shuffle_results)
    print(f"  Saved: {shuf_path}")

    # ---- 4. Precision Self-Check ----
    print("\n--- 4. Precision Self-Check (bin-lookup vs exact hourly) ---")
    # Use a small farm for speed
    for fid in sorted(farms.keys()):
        turbs = coords.get(fid, {}).get(2024, [])
        if 5 <= len(turbs) <= 30:
            break
    print(f"  Using F{fid} ({len(turbs)} turbines), first 200 hours...")
    pc = precision_check(fid, 2024, coords, farms, DEFAULT_TURBINE,
                         wake_models=["gauss"], max_hours=200)

    prec_path = os.path.join(OUT_DIR, "audit_precision_check.csv")
    if pc:
        with open(prec_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['wake_model', 'aep_bin_kWh', 'aep_exact_kWh', 'delta_aep_pct',
                       'wl_bin', 'wl_exact', 'delta_wl_pp', 'n_hours'])
            for wm, r in pc.items():
                w.writerow([wm, r['aep_bin_kWh'], r['aep_exact_kWh'],
                           r['delta_aep_pct'], r['wl_bin'], r['wl_exact'],
                           r['delta_wl_pp'], r['n_hours_checked']])
                print(f"  {wm}: delta_AEP={r['delta_aep_pct']:.3f}%, "
                      f"delta_WL={r['delta_wl_pp']:.3f} pp")
        print(f"  Saved: {prec_path}")

    print("\n=== All audit tests complete! ===")


if __name__ == "__main__":
    run_audits()
