"""
POC 点阵优化 — Sx × Sy 参数网格扫描
策略: 保持矩形网格构型, 扫描横风向(Sx) 顺风向(Sy) 间距, FLORIS Gauss对比AEP

=== 原方案 (琪明 workplan) ===
- 时间跨度: 2014-2024 全部 11 年逐时风
- 步长: Sx 2D-8D / Sy 3D-12D, 步长 0.5D
- 组合数: 13 x 19 = 247 / 场
- 12 场 x 247 组合 x 11 年 = ~32,600 次 FLORIS 运行, 预计 ~270 小时 (11 天)

=== 当前实现 (经廷显与琪明确认的修改) ===
[修改1] 两步走策略:
  (a) 粗扫: 用 2024 单年 + 1D 步长 (7x10=70 组合), 找到最优 (Sx, Sy)
  (b) 精扫: 在粗扫最优附近 ±1.0D, 用 0.5D 步长, 2024 单年
  (c) 多年验证: 仅对精扫选出的最优 (Sx, Sy), 补跑 2014-2023 全 10 年,
      验证 AEP 提升比例跨年稳定
  依据: S1 已验证风向主导方向为季节性分布, 单年波动不改变最优间距的定性排序;
  粗扫 1D 步长足够捕获最优区域, 0.5D 精扫在其邻域内找到严格最优。

[修改2] 网格生成: 沿 PCA 主轴 + 矩形行列网格, 行列数由真实台数和长宽比决定。
  保持真实风机台数不变, 只改变间距和排列。

[修改3] 尾流模型: Gauss 全量, Jensen 在最优参数上跑数个场做稳健性验证。

=== 预期结果 ===
点阵优化 AEP 提升 3-8% (vs 朝向优化的 ~1%), 证明"几何 >> 朝向"
"""
import os, sys, csv, time, math, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    get_ti_for_farm, get_turbine_params, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master,
)
from floris_config import WS_BINS_JENSEN, WD_SECTORS_JENSEN
from task2_floris import (
    precompute_wake_table, replay_hourly_from_bins,
    get_era5_nc_path, get_region_for_farm, extract_wind_series,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# === POC 12 farms (sort by size for fast progress) ===
POC_FARMS = [
    (162, 5), (153, 10), (151, 10), (129, 27), (112, 36),
    (107, 39), (91, 55), (83, 58), (73, 67), (21, 162),
    (14, 189), (4, 364),
]
POC_FARMS.sort(key=lambda x: x[1])

# === Scan params ===
D_REF = 198.0  # IEA 10MW rotor diameter
# Stage 1: coarse (1D step)
SX_COARSE = np.arange(2.0, 8.5, 1.0)   # 2,3,4,5,6,7,8 = 7 values
SY_COARSE = np.arange(3.0, 12.5, 1.0)  # 3,4,5,6,7,8,9,10,11,12 = 10 values
# Stage 2: fine (0.5D step) around best coarse point
DELTA = 1.0  # +/- 1D around coarse optimum

def generate_grid_layout(xs_real, ys_real, n_turb, sx, sy, D=D_REF):
    """Generate rectangular grid with spacing sx*D (cross-wind) and sy*D (down-wind).
    Matches the approximate bounding box and PCA orientation of the real farm."""
    centroid_x, centroid_y = np.mean(xs_real), np.mean(ys_real)
    # Get PCA orientation
    xc = xs_real - centroid_x; yc = ys_real - centroid_y
    cov = np.cov(xc, yc)
    _, vecs = np.linalg.eigh(cov)
    theta_pca = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))

    # Bounding box in PCA coordinates
    cos_t, sin_t = math.cos(math.radians(-theta_pca)), math.sin(math.radians(-theta_pca))
    x_rot = xc * cos_t - yc * sin_t
    y_rot = xc * sin_t + yc * cos_t
    w, h = x_rot.max() - x_rot.min(), y_rot.max() - y_rot.min()

    # Grid: n_cols along cross-wind (x), n_rows along down-wind (y)
    area_per_turb = sx * sy * D * D
    aspect = max(w / max(h, 1), 0.5)
    n_cols = max(1, int(math.sqrt(n_turb * aspect)))
    n_rows = max(1, int(math.ceil(n_turb / n_cols)))

    # Generate grid centered at centroid
    dx, dy = sx * D, sy * D
    grid_x, grid_y = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(grid_x) >= n_turb:
                break
            gx = (col - (n_cols - 1) / 2.0) * dx
            gy = (row - (n_rows - 1) / 2.0) * dy
            # Rotate back to original coordinates
            x_orig = gx * math.cos(math.radians(theta_pca)) - gy * math.sin(math.radians(theta_pca)) + centroid_x
            y_orig = gx * math.sin(math.radians(theta_pca)) + gy * math.cos(math.radians(theta_pca)) + centroid_y
            grid_x.append(x_orig)
            grid_y.append(y_orig)
        if len(grid_x) >= n_turb:
            break

    return [float(v) for v in grid_x[:n_turb]], [float(v) for v in grid_y[:n_turb]]


def run_one_grid(fid, yr, xs, ys, farms, ws_bins, wd_bins, wm='gauss'):
    """Run FLORIS on a single grid layout. Returns CF, WakeLoss, AEP."""
    n = len(xs); tp = get_turbine_params(DEFAULT_TURBINE)
    rated = tp['power_table'].get('controller_dependent_turbine_parameters', {}).get('rated_power', 10000) or 10000
    cap = n * rated
    info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
    ti = get_ti_for_farm(lat, lon); H_t = tp['H']
    region = get_region_for_farm(lat, lon)
    nc = get_era5_nc_path(region, yr)
    if nc is None: return None
    ws100, wdd = extract_wind_series(nc, lat, lon)
    ws_hub = ws100 * (H_t / 100) ** ALPHA_DEFAULT

    we, et, _ = precompute_wake_table(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws_bins, wd_bins, wm)
    rr = replay_hourly_from_bins(ws_hub, wdd, we, list(ws_bins), list(wd_bins), tp, DEFAULT_TURBINE, n, cap, fid, yr)
    return rr, et


def main():
    coords = load_task0_coordinates(); farms = load_farms_master()
    ws_g = np.array(WS_BINS); wd_g = np.array(WD_SECTORS)
    YR = 2024

    csv_out = os.path.join(OUT_DIR, 'poc_layout_results.csv')

    # Real layout AEP for each POC farm (read from task2 if available)
    task2_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2', 'output', 'task2_annual_floris.csv')
    real_aep = {}
    for r in csv.DictReader(open(task2_csv, 'r', encoding='utf-8-sig')):
        if r['wake_model'] == 'gauss':
            real_aep[(int(r['farm_id']), int(r['year']))] = float(r['AEP_kWh'])

    # Check what's done
    done = set()
    if os.path.exists(csv_out):
        for r in csv.DictReader(open(csv_out, 'r', encoding='utf-8-sig')):
            done.add((int(r['farm_id']), int(r['year']), r['stage'], float(r['sx_D']), float(r['sy_D'])))

    # Write header if new
    if not os.path.exists(csv_out):
        with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(['farm_id','year','n_turb','stage','sx_D','sy_D',
                       'AEP_kWh','AEP_noWake_kWh','CF','WakeLoss','precompute_time_s'])

    print(f'POC Scan: {len(POC_FARMS)} farms, Sx={SX_COARSE}, Sy={SY_COARSE}')
    print(f'Coarse: {len(SX_COARSE)}x{len(SY_COARSE)}={len(SX_COARSE)*len(SY_COARSE)} combos/farm')

    for farm_idx, (fid, n_exp) in enumerate(POC_FARMS):
        turbs = coords[fid].get(YR, [])
        n_real = len(turbs)
        if n_real < 2:
            print(f'  F{fid}: {n_real}turb (<2), skipping')
            continue

        info = farms[fid]; country = info.get('country', '?')
        xs_real = np.array([t['x_m'] for t in turbs], dtype=np.float64)
        ys_real = np.array([t['y_m'] for t in turbs], dtype=np.float64)

        # Real layout baseline
        real_key = (fid, YR)
        aep_real = real_aep.get(real_key, 0)
        if aep_real == 0:
            # Compute real baseline
            rr, _ = run_one_grid(fid, YR, [float(x) for x in xs_real], [float(y) for y in ys_real], farms, ws_g, wd_g, 'gauss')
            aep_real = rr['AEP_kWh']

        print(f'\n[{farm_idx+1}/{len(POC_FARMS)}] F{fid} ({n_real}turb, {country}): real AEP={aep_real/1e9:.3f}GWh')
        t_farm = time.time()

        # === STAGE 1: Coarse scan ===
        best_aep = 0; best_sx = best_sy = 0
        n_coarse = 0; n_total = len(SX_COARSE) * len(SY_COARSE)

        for sx in SX_COARSE:
            for sy in SY_COARSE:
                if (fid, YR, 'coarse', sx, sy) in done:
                    continue
                try:
                    xs_g, ys_g = generate_grid_layout(xs_real, ys_real, n_real, sx, sy)
                    rr, et = run_one_grid(fid, YR, xs_g, ys_g, farms, ws_g, wd_g, 'gauss')
                    if rr is None: continue
                    with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                        csv.writer(f).writerow([fid, YR, n_real, 'coarse', sx, sy,
                            rr['AEP_kWh'], rr['AEP_noWake_kWh'], rr['CF'], rr['WakeLoss'], et])
                    n_coarse += 1
                    aep = rr['AEP_kWh']
                    if aep > best_aep:
                        best_aep = aep; best_sx = sx; best_sy = sy
                    elapsed = (time.time() - t_farm) / 60
                    eta = (n_total - n_coarse) * elapsed / max(n_coarse, 1)
                    print(f'  coarse [{n_coarse}/{n_total}] sx={sx:.0f}D sy={sy:.0f}D: AEP={aep/1e6:.1f}MWh | {elapsed:.1f}min ETA{eta:.0f}min')
                except Exception as e:
                    print(f'  coarse sx={sx}D sy={sy}D: ERR {str(e)[:80]}')

        improv = (best_aep - aep_real) / aep_real * 100 if aep_real > 0 else 0
        print(f'  Coarse best: sx={best_sx:.0f}D sy={best_sy:.0f}D AEP={best_aep/1e6:.1f}MWh ({improv:+.1f}% vs real)')

        # === STAGE 2: Fine scan around optimum ===
        sx_fine = np.arange(max(2, best_sx - DELTA), best_sx + DELTA + 0.25, 0.5)
        sy_fine = np.arange(max(3, best_sy - DELTA), best_sy + DELTA + 0.25, 0.5)
        n_fine_total = len(sx_fine) * len(sy_fine)
        best_aep_fine = best_aep; best_sx_fine = best_sx; best_sy_fine = best_sy
        n_fine = 0

        for sx in sx_fine:
            for sy in sy_fine:
                if sx == best_sx and sy == best_sy:
                    continue  # already done in coarse
                if (fid, YR, 'fine', sx, sy) in done:
                    continue
                try:
                    xs_g, ys_g = generate_grid_layout(xs_real, ys_real, n_real, sx, sy)
                    rr, et = run_one_grid(fid, YR, xs_g, ys_g, farms, ws_g, wd_g, 'gauss')
                    if rr is None: continue
                    with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                        csv.writer(f).writerow([fid, YR, n_real, 'fine', sx, sy,
                            rr['AEP_kWh'], rr['AEP_noWake_kWh'], rr['CF'], rr['WakeLoss'], et])
                    n_fine += 1
                    aep = rr['AEP_kWh']
                    if aep > best_aep_fine:
                        best_aep_fine = aep; best_sx_fine = sx; best_sy_fine = sy
                    print(f'  fine [{n_fine}/{n_fine_total}] sx={sx:.1f}D sy={sy:.1f}D: AEP={aep/1e6:.1f}MWh')
                except Exception as e:
                    print(f'  fine sx={sx:.1f}D sy={sy:.1f}D: ERR {str(e)[:80]}')

        improv_final = (best_aep_fine - aep_real) / aep_real * 100 if aep_real > 0 else 0
        print(f'  FINAL: sx={best_sx_fine:.1f}D sy={best_sy_fine:.1f}D, AEP improvement: {improv_final:+.2f}% ({(time.time()-t_farm)/60:.0f}min)')

        # === STAGE 3: 11-year validation at optimal (Sx, Sy) ===
        # Generate grid once, then run FLORIS for each year 2014-2024
        # Compare year-by-year against real layout to verify improvement is robust across years
        print(f'  Stage 3: 11-year validation at sx={best_sx_fine:.1f}D sy={best_sy_fine:.1f}D')
        xs_opt, ys_opt = generate_grid_layout(xs_real, ys_real, n_real, best_sx_fine, best_sy_fine)

        yearly = []
        for yr in range(2014, 2025):
            if (fid, yr, 'validate', best_sx_fine, best_sy_fine) in done:
                continue
            try:
                rr, et = run_one_grid(fid, yr, xs_opt, ys_opt, farms, ws_g, wd_g, 'gauss')
                if rr is None: continue
                real_key = (fid, yr)
                aep_real_yr = real_aep.get(real_key, 0)
                with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([fid, yr, n_real, 'validate', best_sx_fine, best_sy_fine,
                        rr['AEP_kWh'], rr['AEP_noWake_kWh'], rr['CF'], rr['WakeLoss'], et])
                if aep_real_yr > 0:
                    improve_yr = (rr['AEP_kWh'] - aep_real_yr) / aep_real_yr * 100
                    yearly.append(improve_yr)
                    cf_val = rr['CF']
                    print(f'    {yr}: CF={cf_val:.3f} improve={improve_yr:+.1f}%')
                else:
                    cf_val = rr['CF']
                    print(f'    {yr}: CF={cf_val:.3f}')
            except Exception as e:
                print(f'    {yr}: ERR {str(e)[:60]}')

        if yearly:
            mean_improv = sum(yearly) / len(yearly)
            print(f'  11-year mean improvement at optimal (Sx,Sy): {mean_improv:+.2f}% (std={np.std(yearly):.2f}pp)')


if __name__ == '__main__':
    main()
