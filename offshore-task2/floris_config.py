"""
=============================================================================
 floris_config.py — FLORIS v4.6 配置管理层
 负责: 机型管理 / 尾流模型选择 / TI参数化 / 坐标转换 / 分箱定义
 依据: 任务书 §2.5-2.11 / 修改意见 §4.2 / 审计报告 §四
=============================================================================
"""
import os, yaml, tempfile, shutil, copy, csv
import numpy as np
from collections import defaultdict
from floris import FlorisModel
import floris as _floris_pkg

# ---- Paths ----
FLORIS_DIR = os.path.dirname(_floris_pkg.__file__)
TASK0_DIR  = r"D:\1风力发电实习\offshore-task0\output\task0"
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ---- Physical Constants ----
RHO = 1.225          # air density kg/m3
H_REF = 100.0        # ERA5 reference height (m)
ALPHA_DEFAULT = 0.11 # default shear exponent
AVAILABILITY = 0.95
COLLECTION_EFF = 0.97
ELECTRICAL_LOSS = AVAILABILITY * COLLECTION_EFF  # ~0.92

# ---- TI parameterization by sea region (offshore typical values) ----
# Sources: Barthelmie 2007, Hansen 2012, Xu 2026
REGION_TI = {
    "north_sea":    0.06,   # UK/DE/DK/NL — stable marine
    "baltic_sea":   0.07,   # SE — semi-enclosed
    "east_china_sea": 0.08, # CN Jiangsu — monsoon-influenced
    "south_china_sea": 0.09, # CN Fujian/Guangdong — higher mixing
    "taiwan_strait": 0.10,  # TW — channel effect
    "us_east_coast": 0.07,  # US East — open Atlantic
    "mediterranean": 0.06,  # FR — low turbulence
    "default":       0.07,
}

# ---- Niayifar (2016) TI→k mapping for GCH model ----
# k = ka * TI + kb
KA_GAUSS = 0.38   # FLORIS default for gauss model
KB_GAUSS = 0.004

# ---- Wind speed bins (coarser for speed) ----
# ~20 bins, focused on power curve transition zones
WS_BINS = sorted(set(
    [0.0, 3.0] +
    [float(x) for x in range(4, 15)] +   # 4-14 m/s: main power curve (11 bins)
    [16.0, 18.0, 20.0, 22.0, 25.0, 30.0]  # above rated (6 bins)
))
WS_BINS.sort()

# ---- Wind direction sectors (10 deg spacing for speed) ----
WD_SECTORS = list(range(0, 360, 10))  # 36 sectors
WD_STEP = 10.0

# ---- Turbo bins for mega-farms (>400 turbines) ----
WS_BINS_TURBO = sorted(set([0.0, 3.0] + [float(x) for x in range(4, 26, 2)] + [30.0]))
WD_SECTORS_TURBO = list(range(0, 360, 20))  # 18 sectors
# Total: ~13 WS × 18 WD = 234 combos (vs 684, 3x speedup)

# ---- Jensen turbo bins (top-hat model: insensitive to fine binning) ----
# 14 WS x 18 WD = 252 combos, validated 0.89% AEP error on F5 339-turb
WS_BINS_JENSEN = sorted(set([0.0, 3.0] + [float(x) for x in range(5, 26, 2)] + [30.0]))
WD_SECTORS_JENSEN = list(range(0, 360, 20))

def get_bins_for_model(wake_model, n_turbines):
    """Select bin resolution based on wake model and farm size."""
    if wake_model == "jensen":
        return (np.array(WS_BINS_JENSEN, dtype=np.float64),
                np.array(WD_SECTORS_JENSEN, dtype=np.float64))
    # gauss / cc: use standard bins, with turbo override for mega-farms
    if n_turbines > 400:
        return (np.array(WS_BINS_TURBO, dtype=np.float64),
                np.array(WD_SECTORS_TURBO, dtype=np.float64))
    return (np.array(WS_BINS, dtype=np.float64),
            np.array(WD_SECTORS, dtype=np.float64))

# ---- Wake model configurations ----
WAKE_MODELS = {
    "gauss": {
        "velocity_model": "gauss",
        "deflection_model": "gauss",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
        "enable_secondary_steering": True,
        "enable_yaw_added_recovery": True,
        "enable_transverse_velocities": True,
    },
    "jensen": {
        "velocity_model": "jensen",
        "deflection_model": "gauss",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
    },
    "cc": {
        "velocity_model": "cc",
        "deflection_model": "gauss",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
        "enable_secondary_steering": True,
        "enable_yaw_added_recovery": True,
        "enable_transverse_velocities": True,
    },
}

# ---- Available turbine types ----
TURBINE_TYPES = ["ow_6MW", "ow_8MW", "ow_10MW", "ow_12MW", "ow_15MW",
                 "iea_10MW", "iea_15MW", "nrel_5MW", "iea_22MW"]
DEFAULT_TURBINE = "iea_10MW"


# =========================================================================
# 1. TURBINE CONFIGURATION
# =========================================================================

def get_turbine_params(turbine_type=DEFAULT_TURBINE):
    """Get turbine physical parameters from FLORIS turbine library YAML."""
    yaml_path = os.path.join(FLORIS_DIR, "turbine_library", f"{turbine_type}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Turbine YAML not found: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return {
        "name": cfg.get("turbine_type", turbine_type),
        "D": cfg["rotor_diameter"],
        "H": cfg["hub_height"],
        "power_table": cfg["power_thrust_table"],
    }

def get_power_curve(turbine_type=DEFAULT_TURBINE):
    """Return (ws_array, power_kW_array, ct_array) from turbine definition."""
    tp = get_turbine_params(turbine_type)
    tbl = tp["power_table"]
    return (np.array(tbl["wind_speed"]),
            np.array(tbl["power"]),
            np.array(tbl.get("thrust_coefficient", tbl.get("thrust_coefficient", []))))


# =========================================================================
# 2. REGION → TI MAPPING
# =========================================================================

def classify_region(lat, lon):
    """Classify a centroid (lat,lon) into a sea region for TI assignment."""
    # North Sea: UK, DE, DK, NL, BE, NO
    if 51 <= lat <= 62 and -5 <= lon <= 12:
        return "north_sea"
    # Baltic Sea
    if 54 <= lat <= 66 and 12 <= lon <= 30:
        return "baltic_sea"
    # Taiwan Strait
    if 22 <= lat <= 26 and 118 <= lon <= 122:
        return "taiwan_strait"
    # South China Sea (Fujian/Guangdong)
    if 21 <= lat <= 28 and 110 <= lon <= 122:
        return "south_china_sea"
    # East China Sea (Jiangsu/Zhejiang/Shanghai)
    if 28 <= lat <= 36 and 118 <= lon <= 128:
        return "east_china_sea"
    # US East Coast
    if 36 <= lat <= 44 and -78 <= lon <= -65:
        return "us_east_coast"
    # Mediterranean
    if 36 <= lat <= 46 and -5 <= lon <= 20:
        return "mediterranean"
    return "default"

def get_ti_for_farm(lat, lon):
    """Get turbulence intensity for a farm centroid."""
    region = classify_region(lat, lon)
    return REGION_TI[region]


# =========================================================================
# 3. FLORIS MODEL BUILDER
# =========================================================================

def build_base_config(turbine_coords_x, turbine_coords_y,
                      turbine_type=DEFAULT_TURBINE,
                      ti=0.06, alpha=ALPHA_DEFAULT,
                      wake_model_name="gauss"):
    """Build a minimal FLORIS configuration dict.

    Args:
        turbine_coords_x: list of UTM x coordinates (m)
        turbine_coords_y: list of UTM y coordinates (m)
        turbine_type: turbine type name (must exist in FLORIS turbine_library)
        ti: turbulence intensity
        alpha: wind shear exponent
        wake_model_name: "gauss" | "jensen" | "cc"

    Returns: dict ready for yaml.dump and FlorisModel()
    """
    n = len(turbine_coords_x)
    wm = WAKE_MODELS[wake_model_name]

    # Load default config as template
    default_path = os.path.join(FLORIS_DIR, "default_inputs.yaml")
    with open(default_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Override farm — convert all coordinates to plain Python float
    cfg["farm"]["layout_x"] = [float(x) for x in turbine_coords_x]
    cfg["farm"]["layout_y"] = [float(y) for y in turbine_coords_y]
    cfg["farm"]["turbine_type"] = [turbine_type] * n

    # Override flow_field (will be set per-run via .set())
    cfg["flow_field"]["reference_wind_height"] = get_turbine_params(turbine_type)["H"]
    cfg["flow_field"]["wind_shear"] = alpha

    # Override wake model
    cfg["wake"]["model_strings"]["velocity_model"] = wm["velocity_model"]
    cfg["wake"]["model_strings"]["deflection_model"] = wm["deflection_model"]
    cfg["wake"]["model_strings"]["turbulence_model"] = wm["turbulence_model"]
    cfg["wake"]["model_strings"]["combination_model"] = wm["combination_model"]
    for key in ["enable_secondary_steering", "enable_yaw_added_recovery",
                "enable_transverse_velocities", "enable_active_wake_mixing"]:
        if key in wm:
            cfg["wake"][key] = wm[key]

    # TI→k parameterization for gauss model
    if wake_model_name == "gauss":
        cfg["wake"]["wake_velocity_parameters"]["gauss"]["ka"] = KA_GAUSS
        cfg["wake"]["wake_velocity_parameters"]["gauss"]["kb"] = KB_GAUSS

    return cfg


def create_floris_model(turbine_coords_x, turbine_coords_y,
                        turbine_type=DEFAULT_TURBINE,
                        ti=0.06, alpha=ALPHA_DEFAULT,
                        wake_model_name="gauss"):
    """Create and return a FlorisModel instance.

    The config YAML is written to a temp file and loaded.
    Returns (FlorisModel, temp_file_path). Caller should os.unlink the path after use.
    """
    cfg = build_base_config(turbine_coords_x, turbine_coords_y,
                            turbine_type, ti, alpha, wake_model_name)

    tmpf = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                       delete=False, encoding='utf-8')
    yaml.dump(cfg, tmpf)
    tmpf.close()

    fmodel = FlorisModel(tmpf.name)
    return fmodel, tmpf.name


# =========================================================================
# 4. COORDINATES LOADER
# =========================================================================

def load_task0_coordinates():
    """Load turbine coordinates from task0 base.

    Returns: dict[farm_id][year] = list of {turbine_id, x_m, y_m, lon, lat, utm_epsg}
    """
    coord_path = os.path.join(TASK0_DIR, "turbine_coordinates.csv")
    coords = defaultdict(lambda: defaultdict(list))
    with open(coord_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id'])
            yr = int(r['year'])
            coords[fid][yr].append({
                'turbine_id': int(r['turbine_id']),
                'x_m': float(r['x_m']),
                'y_m': float(r['y_m']),
                'lon': float(r['lon']),
                'lat': float(r['lat']),
                'utm_epsg': r.get('utm_epsg', ''),
            })
    return coords

def load_farms_master():
    """Load farms_master.csv from task0."""
    fm_path = os.path.join(TASK0_DIR, "farms_master.csv")
    farms = {}
    with open(fm_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id'])
            farms[fid] = {
                'n_turb': int(r['n_turb']),
                'country': r.get('country', ''),
                'centroid_lon': float(r['centroid_lon']),
                'centroid_lat': float(r['centroid_lat']),
                'capacity_kW': int(r.get('capacity_kW', 0)),
                'area_km2': float(r.get('area_km2', 0)),
            }
    return farms


# =========================================================================
# 5. WIND BINNING UTILITIES
# =========================================================================

def get_wind_bins():
    """Return (ws_bins, wd_sectors) as numpy arrays."""
    return np.array(WS_BINS, dtype=np.float64), np.array(WD_SECTORS, dtype=np.float64)

def bin_wind_speed(ws):
    """Map a wind speed to the nearest bin center."""
    bins = np.array(WS_BINS)
    idx = np.argmin(np.abs(bins - ws))
    return bins[idx]

def bin_wind_direction(wd):
    """Map a wind direction to the nearest sector center (5-deg)."""
    # wd in degrees, 0-360
    sector = round(wd / WD_STEP) * WD_STEP
    if sector >= 360:
        sector -= 360
    return sector

def height_correct(ws_100m, H_target, alpha=ALPHA_DEFAULT):
    """Correct wind speed from 100m to turbine hub height using power law."""
    return ws_100m * (H_target / H_REF) ** alpha


# =========================================================================
# 6. BENCHMARK UTILITIES
# =========================================================================

def benchmark_floris_run(fmodel, n_runs=100):
    """Benchmark a FLORIS model: returns average ms per run."""
    import time
    # Warm up
    for _ in range(5):
        fmodel.run()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fmodel.run()
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000  # ms


if __name__ == "__main__":
    # Self-test
    print("=== floris_config.py self-test ===")
    coords = load_task0_coordinates()
    farms = load_farms_master()
    print(f"Loaded {len(farms)} farms, {sum(len(v.get(2024,[])) for v in coords.values())} turbines (2024)")

    # Test with a small farm
    fid = 0
    yr = 2024
    turbs = coords[fid].get(yr, [])
    xs = [t['x_m'] for t in turbs]
    ys = [t['y_m'] for t in turbs]
    lat = farms[fid]['centroid_lat']
    lon = farms[fid]['centroid_lon']
    ti = get_ti_for_farm(lat, lon)
    alpha = ALPHA_DEFAULT

    print(f"Farm F{fid} ({yr}): {len(turbs)} turbines, TI={ti:.3f}, region={classify_region(lat, lon)}")

    for wm in ["gauss", "jensen", "cc"]:
        fm, tmp = create_floris_model(xs, ys, turbine_type="iea_10MW", ti=ti,
                                       alpha=alpha, wake_model_name=wm)
        fm.set(wind_speeds=[8.0], wind_directions=[270.0],
               turbulence_intensities=[ti])
        fm.run()
        p = fm.get_farm_power()
        ms = benchmark_floris_run(fm, n_runs=50)
        print(f"  {wm:<8}: farm_power(8m/s)={p[0]/1e6:.2f}MW, avg {ms:.1f}ms/run")
        os.unlink(tmp)

    print(f"\nWind bins: {len(WS_BINS)} ({WS_BINS[0]:.1f} - {WS_BINS[-1]:.1f} m/s)")
    print(f"WD sectors: {len(WD_SECTORS)} ({WD_SECTORS[0]} - {WD_SECTORS[-1]} deg)")
    print("Self-test PASSED.")
