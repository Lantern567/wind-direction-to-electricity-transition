"""
补算7: 朝向x几何联合优化 (F57)
比较: 建成朝向 vs theta_opt 两种角度下各跑POC点阵优化
"""
import os, sys, csv, time, numpy as np, math
sys.path.insert(0, r'D:/1风力发电实习/offshore-task2')
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS as FLORIS_WS, WD_SECTORS as FLORIS_WD,
    ALPHA_DEFAULT, ELECTRICAL_LOSS, get_ti_for_farm,
    get_turbine_params, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master,
)
from task2_floris import (
    precompute_wake_table, replay_hourly_from_bins,
    get_era5_nc_path, get_region_for_farm, extract_wind_series,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# Load S1 theta_opt
s1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'task3_s1_optimal_orientation.csv')
theta_opt = None
for r in csv.DictReader(open(s1_path, 'r', encoding='utf-8-sig')):
    if int(r['farm_id']) == 57:
        aep = float(r['expected_AEP_kWh'])
        if theta_opt is None or aep > theta_opt[1]:
            theta_opt = (int(r['angle_deg']), aep)

# Load real AEP baseline
task2_path = r'D:/1风力发电实习/offshore-task2/output/task2_annual_floris.csv'
real_aep = 0.0
for r in csv.DictReader(open(task2_path, 'r', encoding='utf-8-sig')):
    if int(r['farm_id']) == 57 and int(r['year']) == 2024 and r['wake_model'] == 'gauss':
        real_aep = float(r['AEP_kWh'])
        break

# POC scan parameters (same as original POC)
SX = np.arange(2.0, 8.5, 1.0)  # 7 values
SY = np.arange(3.0, 12.5, 1.0)  # 10 values
D = 198.0  # rotor diameter

coords = load_task0_coordinates(); farms = load_farms_master()
turbs = coords[57].get(2024, [])
n = len(turbs)
info = farms[57]
ti = get_ti_for_farm(info['centroid_lat'], info['centroid_lon'])
xs_real = np.array([float(t['x_m']) for t in turbs], dtype=np.float64)
ys_real = np.array([float(t['y_m']) for t in turbs], dtype=np.float64)
cx, cy = np.mean(xs_real), np.mean(ys_real)

# Load ERA5
region = get_region_for_farm(info['centroid_lat'], info['centroid_lon'])
nc = get_era5_nc_path(region, 2024)
ws100, wdd = extract_wind_series(nc, info['centroid_lat'], info['centroid_lon'])
tp = get_turbine_params(DEFAULT_TURBINE)
ws_hub = ws100 * (tp['H'] / 100.0) ** ALPHA_DEFAULT
ws = np.array(FLORIS_WS); wd = np.array(FLORIS_WD)
TP = get_turbine_params(DEFAULT_TURBINE); RATED = float(max(TP['power_table']['power']))

def generate_pca_grid(xs_orig, ys_orig, n_turb, sx, sy):
    centroid_x, centroid_y = np.mean(xs_orig), np.mean(ys_orig)
    xc = xs_orig - centroid_x; yc = ys_orig - centroid_y
    cov = np.cov(xc, yc)
    _, vecs = np.linalg.eigh(cov)
    theta_pca = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    cos_t, sin_t = math.cos(math.radians(-theta_pca)), math.sin(math.radians(-theta_pca))
    x_rot = xc * cos_t - yc * sin_t
    y_rot = xc * sin_t + yc * cos_t
    w, h = x_rot.max() - x_rot.min(), y_rot.max() - y_rot.min()
    dx, dy = sx * D, sy * D
    aspect = max(w / max(h, 1), 0.5)
    n_cols = max(1, int(math.sqrt(n_turb * aspect)))
    n_rows = max(1, int(math.ceil(n_turb / n_cols)))
    grid_x, grid_y = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(grid_x) >= n_turb: break
            gx = (col - (n_cols-1)/2.0) * dx
            gy = (row - (n_rows-1)/2.0) * dy
            x_orig = gx * math.cos(math.radians(theta_pca)) - gy * math.sin(math.radians(theta_pca)) + centroid_x
            y_orig = gx * math.sin(math.radians(theta_pca)) + gy * math.cos(math.radians(theta_pca)) + centroid_y
            grid_x.append(x_orig); grid_y.append(y_orig)
        if len(grid_x) >= n_turb: break
    return [float(v) for v in grid_x[:n_turb]], [float(v) for v in grid_y[:n_turb]]

def run_aep(xs, ys):
    n_t = len(xs)
    we, et, nt = precompute_wake_table(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws, wd, 'gauss')
    rr = replay_hourly_from_bins(ws_hub, wdd, we, list(ws), list(wd), TP, DEFAULT_TURBINE, n_t, n_t*RATED, 57, 2024)
    return rr['AEP_kWh'], et

# Oriented: rotate to θ_opt, then generate PCA grid
xs_rot_opt = (xs_real - cx) * math.cos(math.radians(theta_opt[0])) - (ys_real - cy) * math.sin(math.radians(theta_opt[0])) + cx
ys_rot_opt = (xs_real - cx) * math.sin(math.radians(theta_opt[0])) + (ys_real - cy) * math.cos(math.radians(theta_opt[0])) + cy

print(f'F57: {n}t, theta_opt={theta_opt[0]}deg, real AEP={real_aep/1e9:.3f}GWh')
print(f'Scan: Sx={SX}, Sy={SY} ({len(SX)}x{len(SY)}={len(SX)*len(SY)} combos each)')
print()

# Scan 1: Built orientation (default PCA grid)
best_built = 0; bsx = bsy = 0
for sx in SX:
    for sy in SY:
        xs_g, ys_g = generate_pca_grid(xs_real, ys_real, n, sx, sy)
        aep, et = run_aep(xs_g, ys_g)
        if aep > best_built: best_built = aep; bsx = sx; bsy = sy
print(f'Built orientation best: sx={bsx}D sy={bsy}D AEP={best_built/1e9:.3f}GWh (+{(best_built-real_aep)/real_aep*100:.1f}%)')

# Scan 2: Optimal orientation (rotated, then PCA grid)
best_rot = 0; rsx = rsy = 0
for sx in SX:
    for sy in SY:
        xs_g, ys_g = generate_pca_grid(xs_rot_opt, ys_rot_opt, n, sx, sy)
        aep, et = run_aep(xs_g, ys_g)
        if aep > best_rot: best_rot = aep; rsx = sx; rsy = sy
print(f'Optimal orientation best: sx={rsx}D sy={rsy}D AEP={best_rot/1e9:.3f}GWh (+{(best_rot-real_aep)/real_aep*100:.1f}%)')

# Results
geom_gain = (best_built - real_aep) / real_aep * 100
joint_gain = (best_rot - real_aep) / real_aep * 100
orient_share = (best_rot - best_built) / (best_rot - real_aep) * 100 if (best_rot - real_aep) > 0 else 0

with open(os.path.join(OUT_DIR, 'joint_optimization_F57.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerow(['scenario','sx_best_D','sy_best_D','AEP_kWh','gain_pct_vs_real','note'])
    csv.writer(f).writerow(['real', '-', '-', real_aep, 0.0, 'F57: 80 turb Vietnam, actual spacing ~1.9D'])
    csv.writer(f).writerow(['built_opt_geom', bsx, bsy, best_built, round(geom_gain,1), f'real spacing ~1.9D, optimized to {bsx}Dx{bsy}D'])
    csv.writer(f).writerow(['rotated_opt_joint', rsx, rsy, best_rot, round(joint_gain,1), f'rotated to theta_opt={theta_opt[0]}deg THEN optimized to {rsx}Dx{rsy}D'])

print(f'\nGeom alone: +{geom_gain:.1f}% | Joint (orient+geom): +{joint_gain:.1f}%')
print(f'Orientation share of joint gain: {orient_share:.0f}%')
print(f'Output: joint_optimization_F57.csv')
