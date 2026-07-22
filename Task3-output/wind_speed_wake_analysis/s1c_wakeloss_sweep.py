"""
S1c: wakeloss_sweep.csv — Controlled wind speed sweep (3-25 m/s, 1 m/s step)
Layout 1: standard 8x8 5D square grid (64 turbines, D=198m, Sx=Sy=5D=990m)
Layout 2: real dense farm (use the farm with highest WakeLoss from task2)
Single wind direction: 270deg (max wake along rows). No ERA5 needed.
"""
import os, sys, csv, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    create_floris_model, get_turbine_params, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master, get_ti_for_farm,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

D = 198.0  # IEA 10MW rotor diameter
S5D = 5 * D  # 990m
WS_SWEEP = np.arange(3.0, 26.0, 1.0)  # 3,4,...,25 m/s (23 values)
WD_FIXED = 270.0  # west wind — maximises wake along rows
TI = 0.07  # offshore typical

def build_standard_grid(n_rows=8, n_cols=8, spacing=S5D):
    xs, ys = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            xs.append(float(col) * spacing)
            ys.append(float(row) * spacing)
    return xs, ys

def compute_sweep(xs, ys, label, ti=TI):
    """Run FLORIS for each wind speed, return list of (ws, p_wake, p_noWake)"""
    n = len(xs)
    tp = get_turbine_params(DEFAULT_TURBINE)
    ws_pc = np.array(tp['power_table']['wind_speed'])
    pw_pc = np.array(tp['power_table']['power'])

    fm, tmp_path = create_floris_model(xs, ys, turbine_type=DEFAULT_TURBINE,
                                        ti=ti, alpha=0.11, wake_model_name='gauss')
    # Set all wind speeds at once
    ws_list = [float(w) for w in WS_SWEEP]
    fm.set(wind_speeds=ws_list, wind_directions=[WD_FIXED]*len(ws_list),
           turbulence_intensities=[ti]*len(ws_list))
    fm.run()
    farm_powers = fm.get_farm_power() / 1000.0  # W -> kW

    results = []
    for i, ws in enumerate(WS_SWEEP):
        p_no_wake = float(np.interp(ws, ws_pc, pw_pc, left=0, right=0)) * n
        p_wake = float(farm_powers[i])
        eta = p_wake / p_no_wake if p_no_wake > 0 else 1.0
        wl_pct = (1.0 - eta) * 100
        results.append((float(ws), round(p_wake, 1), round(p_no_wake, 1),
                        round(eta, 6), round(wl_pct, 4)))

    os.unlink(tmp_path)
    return results

def main():
    print("S1c: Controlled wind speed sweep (3-25 m/s)...")

    coords = load_task0_coordinates(); farms = load_farms_master()

    # Pick the densest real farm with highest WakeLoss from task2
    task2 = r'D:/1风力发电实习/offshore-task2/output/task2_annual_floris.csv'
    best_wl = 0; best_fid = 2  # default
    for r in csv.DictReader(open(task2,'r',encoding='utf-8-sig')):
        if r['wake_model'] == 'gauss' and r['year'] == '2024':
            wl = float(r['WakeLoss'])
            n = int(r['n_turb'])
            if wl > best_wl and 10 <= n <= 200:  # medium size, not mega
                best_wl = wl; best_fid = int(r['farm_id'])
    print(f"Real dense farm: F{best_fid} (WakeLoss={best_wl*100:.1f}%)")
    info = farms[best_fid]
    turbs = coords[best_fid].get(2024, [])
    real_xs = [float(t['x_m']) for t in turbs]
    real_ys = [float(t['y_m']) for t in turbs]
    real_ti = get_ti_for_farm(info['centroid_lat'], info['centroid_lon'])

    # Layout 1: standard 8x8 grid
    print("  Standard 8x8 grid (5D spacing)...")
    xs1, ys1 = build_standard_grid(8, 8, S5D)
    r1 = compute_sweep(xs1, ys1, 'standard_8x8_5D')

    # Layout 2: real dense farm
    print(f"  Real farm F{best_fid} ({len(turbs)} turbines)...")
    r2 = compute_sweep(real_xs, real_ys, f'real_F{best_fid}', ti=real_ti)

    # Save
    csv_out = os.path.join(OUT_DIR, 'wakeloss_sweep.csv')
    with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['layout_label', 'ws_m_s', 'p_wake_kW', 'p_noWake_kW',
                    'wake_efficiency', 'wakeloss_pct'])
        for label, results in [('standard_8x8_5D', r1), (f'real_F{best_fid}', r2)]:
            for ws_val, p_wake, p_no_wake, eta, wl in results:
                w.writerow([label, ws_val, p_wake, p_no_wake, eta, wl])

    # Show summary
    print(f"\nSweep results ({csv_out}):")
    for label, results in [('standard_8x8_5D', r1), (f'real_F{best_fid}', r2)]:
        peak_ws = max(results, key=lambda r: r[4])  # max wakeloss_pct
        print(f"  {label}: peak WL={peak_ws[4]:.1f}% at {peak_ws[0]:.0f}m/s")

if __name__ == '__main__':
    main()
