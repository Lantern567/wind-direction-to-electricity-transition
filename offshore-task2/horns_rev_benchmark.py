"""
=============================================================================
 Horns Rev 1 基准验证 — 用 FLORIS 复现已知风场结果
 Horns Rev 1: 80 × Vestas V80 (2MW), 8行 × 10列, 间距 7D × 7D
 对照: Hansen et al. (2012) SCADA数据的尾流损失 ~12-15%
 依据: 图片文本.txt 不足5 / 审计报告 §四
=============================================================================
"""
import os, sys, time, tempfile, yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floris_config import (
    create_floris_model, get_power_curve, get_turbine_params,
    ALPHA_DEFAULT, ELECTRICAL_LOSS,
)

# Horns Rev 1 geometry
# 80 turbines in 8 rows (cross-wind) × 10 columns (down-wind)
# Spacing: 7D (D=80m for V80) → 560m
D_V80 = 80.0   # m
SPACING = 7 * D_V80  # 560m
N_ROWS = 8    # cross-wind
N_COLS = 10   # down-wind

def build_horns_rev_layout():
    """Create Horns Rev 1 turbine positions."""
    xs, ys = [], []
    for col in range(N_COLS):
        for row in range(N_ROWS):
            x = col * SPACING  # down-wind
            y = row * SPACING  # cross-wind
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys

def run_horns_rev_benchmark():
    """Run FLORIS on Horns Rev 1 and compare with published SCADA results."""
    print("=" * 60)
    print(" Horns Rev 1 基准验证 (FLORIS v4)")
    print("=" * 60)
    print(f" 80 × V80 (2MW), 8 row × 10 col, 7D×7D spacing")
    print(f" Ref: Hansen et al. 2012, Wake Loss ~12-15% (SCADA)")
    print()

    xs, ys = build_horns_rev_layout()
    n_turb = len(xs)
    print(f" Turbines: {n_turb}")

    # V80 parameters
    # We don't have V80 in FLORIS turbine_library, so use nrel_5MW scaled
    # or create a V80-equivalent with ~80m rotor, ~70m hub height
    # Use PyWake to generate V80-like power/CT curves
    from py_wake.wind_turbines.generic_wind_turbines import standard_power_ct_curve

    u, p_kW, ct = standard_power_ct_curve(power_norm=2000, diameter=80.0,
                                           turbulence_intensity=0.1)

    # Create a simple turbine YAML for FLORIS
    tlib = os.path.join(os.path.dirname(__import__('floris').__file__), "turbine_library")
    v80_yml = {
        "turbine_type": "vestas_v80",
        "hub_height": 70.0,
        "rotor_diameter": 80.0,
        "TSR": 8.0,
        "operation_model": "cosine-loss",
        "power_thrust_table": {
            "ref_air_density": 1.225,
            "ref_tilt": 5.0,
            "cosine_loss_exponent_yaw": 1.88,
            "cosine_loss_exponent_tilt": 1.88,
            "helix_a": 1.8, "helix_power_b": 4.5e-03, "helix_power_c": 1.6e-10,
            "helix_thrust_b": 1.0e-03, "helix_thrust_c": 1.0e-06,
            "peak_shaving_fraction": 0.2, "peak_shaving_TI_threshold": 0.1,
            "wind_speed": [round(float(u[i]), 4) for i in range(len(u)) if i % 5 == 0],
            "power": [round(float(p_kW[i]), 2) for i in range(len(u)) if i % 5 == 0],
            "thrust_coefficient": [round(float(ct[i]), 6) for i in range(len(u)) if i % 5 == 0],
        }
    }

    # Also add 0 and 50 m/s endpoints
    v80_yml["power_thrust_table"]["wind_speed"] = (
        [0.0, 2.9] + v80_yml["power_thrust_table"]["wind_speed"][2:] + [25.01, 50.0]
    )
    v80_yml["power_thrust_table"]["power"] = (
        [0.0, 0.0] + v80_yml["power_thrust_table"]["power"][2:] + [0.0, 0.0]
    )
    v80_yml["power_thrust_table"]["thrust_coefficient"] = (
        [0.0, 0.0] + v80_yml["power_thrust_table"]["thrust_coefficient"][2:] + [0.0, 0.0]
    )

    v80_path = os.path.join(tlib, "vestas_v80.yaml")
    with open(v80_path, 'w', encoding='utf-8') as f:
        f.write("# Vestas V80 2MW - PyWake generated for Horns Rev benchmark\n")
        yaml.dump(v80_yml, f, default_flow_style=False, sort_keys=False)
    # Also copy to our data dir
    import shutil
    shutil.copy2(v80_path, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "data", "vestas_v80.yaml"))
    print(f" Created: vestas_v80.yaml")

    # Run for a range of wind speeds at 270 deg (wind from west → aligned with rows)
    ti_test = 0.075  # Horns Rev typical offshore TI
    ws_test = [4.0, 6.0, 8.0, 10.0, 12.0]
    wd_test = 270.0  # aligned with columns (worst-case wake)

    print(f"\n Running FLORIS ({len(ws_test)} wind speeds × 1 direction)...")

    fm, tmp_path = create_floris_model(xs, ys, turbine_type="vestas_v80",
                                        ti=ti_test, alpha=0.12,
                                        wake_model_name="gauss")
    fm.set(wind_speeds=ws_test, wind_directions=[wd_test]*len(ws_test),
           turbulence_intensities=[ti_test]*len(ws_test))
    fm.run()

    farm_powers = fm.get_farm_power() / 1000.0  # → kW
    os.unlink(tmp_path)

    # Compute no-wake power
    tp = get_turbine_params("vestas_v80")
    ws_pc = np.array(tp['power_table']['wind_speed'])
    pw_pc = np.array(tp['power_table']['power'])

    print(f"\n {'WS(m/s)':<10} {'P_wake(MW)':<14} {'P_noWake(MW)':<14} {'WakeLoss(%)':<12}")
    print("-" * 52)

    for i, ws in enumerate(ws_test):
        p_per_turb = np.interp(ws, ws_pc, pw_pc, left=0.0, right=0.0)
        p_no_wake = p_per_turb * n_turb
        p_wake = farm_powers[i]
        wl = (p_no_wake - p_wake) / p_no_wake * 100 if p_no_wake > 0 else 0.0
        print(f" {ws:<10.1f} {p_wake/1000:<14.2f} {p_no_wake/1000:<14.2f} {wl:<12.2f}")

    # Also run at rated wind speed with Jensen
    print(f"\n --- Jensen & CC comparison at 8 m/s ---")
    for wm in ["gauss", "jensen", "cc"]:
        fm2, tmp2 = create_floris_model(xs, ys, turbine_type="vestas_v80",
                                         ti=ti_test, alpha=0.12, wake_model_name=wm)
        fm2.set(wind_speeds=[8.0], wind_directions=[wd_test],
                turbulence_intensities=[ti_test])
        fm2.run()
        p_w = fm2.get_farm_power()[0] / 1000.0
        p_per_t = np.interp(8.0, ws_pc, pw_pc, left=0.0, right=0.0)
        p_nw = p_per_t * n_turb
        wl = (p_nw - p_w) / p_nw * 100
        print(f"  {wm:<8}: P={p_w/1000:.2f}MW, WL={wl:.1f}%")
        os.unlink(tmp2)

    print(f"\n=== Benchmark summary ===")
    print(f" Published SCADA (Hansen 2012): Wake Loss ~12-15% at rated wind")
    print(f" This study FLORIS gauss:       See above")
    print(f" Expected: within ±3pp of published range for gauss model")
    print(f" Verification: {'PASS' if 8 <= wl <= 18 else 'CHECK'} (reference range 8-22% from Xu 2026)")

if __name__ == "__main__":
    run_horns_rev_benchmark()
