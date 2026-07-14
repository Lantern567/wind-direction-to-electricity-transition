"""
Task 3 — S2: Paradigm Layout Generation (范式排布生成)
==========================================================
For each wind farm × paradigm label, generate alternative turbine coordinates
according to the paradigm's construction logic.

Paradigm A: Crosswind grid — align perpendicular to energy wind direction
Paradigm B: Constraint priority — preserve real PCA axis, regularize spacing
Paradigm C: Expansion — keep core + add turbines crosswind
Paradigm D: High WPD — select points weighted by wind resource gradient
Paradigm E: Large spacing — same as A but with Sx/Sy increased by 25%

Author: Qiming
Inputs:  task0 (turbine_coordinates, farms_master, turbines_by_year)
         task1_output (paradigm_classification, theta_parameters, wind_metrics)
Outputs: data/task3_output/paradigm_layouts.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import spatial
from matplotlib.path import Path

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TASK0_DIR = os.path.join(DATA_DIR, 'task0')
TASK1_DIR = os.path.join(DATA_DIR, 'task1_output')
OUT_DIR = os.path.join(DATA_DIR, 'task3_output')
os.makedirs(OUT_DIR, exist_ok=True)

D_M = 198.0           # Rotor diameter (m) — IEA 10MW
MIN_TURBINES = 3      # Minimum turbines for a valid layout

# ── Data Loading ─────────────────────────────────────────────

def load_data():
    """Load all required data from task0 and task1."""
    print("Loading data...")

    data = {}

    # Task 0
    data['farms'] = pd.read_csv(os.path.join(TASK0_DIR, 'farms_master.csv'))
    data['tcoords'] = pd.read_csv(os.path.join(TASK0_DIR, 'turbine_coordinates.csv'))
    data['tby'] = pd.read_csv(os.path.join(TASK0_DIR, 'turbines_by_year.csv'))

    # Task 1
    data['paradigm'] = pd.read_csv(os.path.join(TASK1_DIR, 'task1_paradigm_classification.csv'))
    data['theta'] = pd.read_csv(os.path.join(TASK1_DIR, 'task1_theta_parameters.csv'))
    data['wind'] = pd.read_csv(os.path.join(TASK1_DIR, 'task1_wind_metrics.csv'))

    # Filter to reference years
    data['theta_2024'] = data['theta'][data['theta']['year'] == 2024].copy()
    data['wind_2024'] = data['wind'][data['wind']['year'] == 2024].copy()
    data['tcoords_2024'] = data['tcoords'][data['tcoords']['year'] == 2024].copy()

    # Task 3 — WPD gradient grid (for Paradigm D)
    wpd_grid_path = os.path.join(DATA_DIR, 'task3_output', 'era5_wpd_grid.csv')
    if os.path.exists(wpd_grid_path):
        data['wpd_grid'] = pd.read_csv(wpd_grid_path)
        print(f"  WPD grid: {len(data['wpd_grid'])} ERA5 points loaded")
    else:
        data['wpd_grid'] = None
        print("  WPD grid: NOT FOUND (Paradigm D will use fallback)")

    # Build farm_id -> (centroid_lon, centroid_lat) from farms_master
    data['farm_lonlat'] = dict(zip(
        data['farms']['farm_id'],
        zip(data['farms']['centroid_lon'], data['farms']['centroid_lat'])
    ))

    print(f"  Farms: {len(data['farms'])}")
    print(f"  Paradigm rows: {len(data['paradigm'])}")
    print(f"  Theta (2024): {len(data['theta_2024'])} farms")
    print(f"  Wind (2024): {len(data['wind_2024'])} farms")
    print(f"  Coords (2024): {len(data['tcoords_2024'])} turbines")

    return data


# ── Geometry Helpers ─────────────────────────────────────────

def get_farm_boundary(coords_xy, buffer_m=500):
    """
    Get simplified farm boundary from turbine coordinates.
    Returns: (centroid_x, centroid_y), hull_vertices, bounding_path, extent
    """
    if len(coords_xy) < 3:
        cx, cy = coords_xy[:, 0].mean(), coords_xy[:, 1].mean()
        return (cx, cy), coords_xy, None, None

    # Compute convex hull
    hull = spatial.ConvexHull(coords_xy)
    hull_verts = coords_xy[hull.vertices]

    # Compute centroid of hull (not just mean of points)
    cx = hull_verts[:, 0].mean()
    cy = hull_verts[:, 1].mean()

    # Expand hull by buffer distance (simple scaling from centroid)
    scale = 1.0 + buffer_m / max(np.ptp(hull_verts[:, 0]), np.ptp(hull_verts[:, 1]), 100)
    scaled_verts = np.array([
        [cx + (x - cx) * scale, cy + (y - cy) * scale]
        for x, y in hull_verts
    ])

    # Create matplotlib path for fast point-in-polygon testing
    # Close the polygon
    path_verts = np.vstack([scaled_verts, scaled_verts[0]])
    path = Path(path_verts)

    extent = {
        'minx': scaled_verts[:, 0].min(),
        'maxx': scaled_verts[:, 0].max(),
        'miny': scaled_verts[:, 1].min(),
        'maxy': scaled_verts[:, 1].max(),
    }

    return (cx, cy), hull_verts, path, extent


def generate_grid_in_boundary(centroid, path, extent, n_turb, Sx_D, Sy_D, theta_deg, phi_stagger=0.0):
    """
    Generate a regular grid of ~n_turb points within the farm boundary.
    Uses adaptive spacing reduction if initial grid is too sparse.
    """
    cx, cy = centroid

    if extent is None:
        # Fallback: generate circular layout
        radius = max(500, np.sqrt(n_turb) * (Sx_D + Sy_D) / 4 * D_M)
        points = np.array([
            [cx + radius * np.sqrt((i + 0.5) / n_turb) * np.cos(2 * np.pi * i / n_turb),
             cy + radius * np.sqrt((i + 0.5) / n_turb) * np.sin(2 * np.pi * i / n_turb)]
            for i in range(n_turb)
        ])
        return points, {'method': 'circular_fallback'}

    # Pre-compute rotation matrix
    theta_rad = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    # Try progressively smaller spacings until we get enough points
    spacing_ratios = [1.0, 0.7, 0.5, 0.35, 0.25]
    best_points = np.array([])

    for ratio in spacing_ratios:
        Sx_m = Sx_D * D_M * ratio
        Sy_m = Sy_D * D_M * ratio

        # Estimate grid size from area
        area_est = (extent['maxx'] - extent['minx']) * (extent['maxy'] - extent['miny'])
        cell_area = Sx_m * Sy_m
        n_est = max(n_turb * 4, int(area_est / cell_area * 3))

        n_col = max(3, int(np.sqrt(n_est * Sy_m / Sx_m)))
        n_row = max(3, int(n_est / n_col))

        # Generate grid points (vectorized)
        rows = np.arange(-n_row, n_row + 1)
        cols = np.arange(-n_col, n_col + 1)
        row_grid, col_grid = np.meshgrid(rows, cols)
        row_grid = row_grid.flatten()
        col_grid = col_grid.flatten()

        stagger = np.where(row_grid % 2 == 1, phi_stagger * Sy_m, 0.0)
        x_local = col_grid * Sy_m + stagger
        y_local = row_grid * Sx_m
        x_rot = x_local * cos_t - y_local * sin_t
        y_rot = x_local * sin_t + y_local * cos_t

        test_points = np.column_stack([cx + x_rot, cy + y_rot])
        inside = path.contains_points(test_points)
        candidate_xy = test_points[inside]

        if len(candidate_xy) >= n_turb:
            best_points = candidate_xy
            final_ratio = ratio
            break
        elif len(candidate_xy) > len(best_points):
            best_points = candidate_xy
            final_ratio = ratio

    if len(best_points) < MIN_TURBINES:
        # Last resort: use real turbine positions as template
        return np.array([]), {'method': 'failed', 'n_found': len(best_points)}

    # Subsample to target count
    if len(best_points) > n_turb:
        dists = np.sum((best_points - np.array([cx, cy])) ** 2, axis=1)
        order = np.argsort(dists)
        step = max(1, len(order) / n_turb)
        indices = order[np.floor(np.arange(n_turb) * step).astype(int)]
        best_points = best_points[indices[:n_turb]]

    return best_points, {
        'Sx_D': Sx_D * final_ratio,
        'Sy_D': Sy_D * final_ratio,
        'grid_theta': theta_deg,
        'spacing_ratio': final_ratio,
        'n_candidates_raw': len(best_points),
    }


def get_farm_theta_energy(farm_id, wind_2024, paradigm_df):
    """
    Get theta_energy for a farm. If missing, use regional median.
    """
    row = wind_2024[wind_2024['farm_id'] == farm_id]
    if len(row) > 0 and pd.notna(row['theta_energy_hist'].values[0]):
        return row['theta_energy_hist'].values[0]

    # Fallback: use country median
    farm_country = paradigm_df[paradigm_df['farm_id'] == farm_id]['country']
    if len(farm_country) > 0:
        country = farm_country.values[0]
        country_farms = paradigm_df[paradigm_df['country'] == country]['farm_id']
        country_wind = wind_2024[wind_2024['farm_id'].isin(country_farms)]
        country_theta = country_wind['theta_energy_hist'].dropna()
        if len(country_theta) > 0:
            return country_theta.median()

    # Global fallback
    global_theta = wind_2024['theta_energy_hist'].dropna()
    return global_theta.median() if len(global_theta) > 0 else 0.0


# ── Paradigm Generation Functions ────────────────────────────

def generate_paradigm_A(farm_id, n_turb, centroid, path, extent, coords, theta_params, wind_2024, paradigm_df):
    """
    Paradigm A: Crosswind layout.
    Align grid perpendicular to energy wind direction (theta_energy + 90 deg).
    """
    theta_energy = get_farm_theta_energy(farm_id, wind_2024, paradigm_df)
    crosswind_theta = (theta_energy + 90) % 360

    trow = theta_params[theta_params['farm_id'] == farm_id]
    if len(trow) > 0 and pd.notna(trow['Sx_D'].values[0]):
        Sx = trow['Sx_D'].values[0]
        Sy = trow['Sy_D'].values[0]
        phi = trow['phi_stagger'].values[0] if pd.notna(trow['phi_stagger'].values[0]) else 0.0
    else:
        Sx = 9.0; Sy = 9.0; phi = 0.0

    Sx = max(3.0, min(20.0, Sx))
    Sy = max(3.0, min(20.0, Sy))

    points, gmeta = generate_grid_in_boundary(centroid, path, extent, n_turb, Sx, Sy, crosswind_theta, phi)

    meta = {'theta_energy': theta_energy, 'grid_theta': crosswind_theta,
            'Sx_D': Sx, 'Sy_D': Sy, 'stagger': phi, 'n_grid': gmeta.get('n_candidates', 0)}
    return points, meta


def generate_paradigm_B(farm_id, n_turb, centroid, path, extent, coords, theta_params, paradigm_df):
    """
    Paradigm B: Constraint priority.
    Preserve real layout's PCA direction (proxy for constraint direction).
    """
    if len(coords) >= 3:
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        pc1_dir = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])) % 360
    else:
        pc1_dir = 0.0

    if len(coords) >= 2:
        nn_dists = spatial.distance.pdist(coords)
        nn_median = np.median(nn_dists)
        Sx = Sy = nn_median / D_M
    else:
        Sx = Sy = 7.0

    Sx = max(3.0, min(15.0, Sx))
    Sy = max(3.0, min(15.0, Sy))

    points, gmeta = generate_grid_in_boundary(centroid, path, extent, n_turb, Sx, Sy, pc1_dir)

    meta = {'pca_direction': pc1_dir, 'Sx_D': Sx, 'Sy_D': Sy}
    return points, meta


def generate_paradigm_C(farm_id, n_turb, centroid, path, extent, coords, farm_tcoords_hist, wind_2024, paradigm_df):
    """
    Paradigm C: Expansion.
    Keep earliest-phase turbines as core, add new turbines crosswind.
    """
    if farm_tcoords_hist is None or len(farm_tcoords_hist) == 0:
        return generate_paradigm_B(farm_id, n_turb, centroid, path, extent, coords, None, paradigm_df)

    yearly_counts = farm_tcoords_hist.groupby('year').size()
    first_years = yearly_counts[yearly_counts >= 3].index

    if len(first_years) == 0:
        return generate_paradigm_B(farm_id, n_turb, centroid, path, extent, coords, None, paradigm_df)

    first_year = int(first_years[0])

    core_coords = farm_tcoords_hist[(farm_tcoords_hist['year'] == first_year)][['x_m', 'y_m']].values
    n_core = len(core_coords)
    n_new = max(0, n_turb - n_core)

    if n_new == 0 or n_core < 3:
        core_points = core_coords
        meta = {'first_year': first_year, 'n_core': n_core, 'n_new': 0}
        return core_points, meta

    theta_energy = get_farm_theta_energy(farm_id, wind_2024, paradigm_df)
    crosswind_dir = (theta_energy + 90) % 360
    crosswind_rad = np.radians(crosswind_dir)

    core_center = core_coords.mean(axis=0)

    # Get core extent in crosswind-rotated coordinates
    core_rot = np.zeros_like(core_coords)
    for i in range(len(core_coords)):
        dx, dy = core_coords[i, 0] - core_center[0], core_coords[i, 1] - core_center[1]
        core_rot[i, 0] = dx * np.cos(-crosswind_rad) - dy * np.sin(-crosswind_rad)
        core_rot[i, 1] = dx * np.sin(-crosswind_rad) + dy * np.cos(-crosswind_rad)

    cross_min, cross_max = core_rot[:, 0].min(), core_rot[:, 0].max()
    along_min, along_max = core_rot[:, 1].min(), core_rot[:, 1].max()

    if n_core >= 2:
        nn_dists = spatial.distance.pdist(core_coords)
        Sx = Sy = np.median(nn_dists) / D_M
    else:
        Sx = Sy = 7.0

    new_points = []
    expansion_step = 0
    while len(new_points) < n_new and expansion_step < 50:
        expansion_step += 1
        side_sign = 1 if expansion_step % 2 == 1 else -1  # alternate sides
        offset = ((expansion_step + 1) // 2) * Sy * D_M  # increase distance each pair
        n_along = max(2, int((along_max - along_min) / (Sx * D_M)) + 2)
        for row_pos in np.linspace(along_min - Sx * D_M, along_max + Sx * D_M, n_along):
            px_local = (cross_max if side_sign > 0 else cross_min) + side_sign * offset
            px = core_center[0] + px_local * np.cos(crosswind_rad) - row_pos * np.sin(crosswind_rad)
            py = core_center[1] + px_local * np.sin(crosswind_rad) + row_pos * np.cos(crosswind_rad)
            if path is not None and path.contains_points(np.array([[px, py]]))[0]:
                new_points.append([px, py])
                if len(new_points) >= n_new:
                    break

    # Fallback: if expansion doesn't produce enough points (core too small),
    # use paradigm B (regularized layout) to fill remaining turbines
    if len(new_points) < n_new * 0.5:
        # Fill remaining with regular grid in the expanded area
        remaining = n_turb - n_core - len(new_points)
        if remaining > 0:
            # Use B-like generation for remaining turbines
            # But keep core + any new points already generated
            fill_pts, _ = generate_grid_in_boundary(
                centroid, path, extent, remaining,
                Sx, Sy, crosswind_dir, phi_stagger=0.0
            )
            if len(fill_pts) > 0:
                new_points.extend(fill_pts.tolist())

    all_points = np.vstack([core_coords, np.array(new_points)]) if new_points else core_coords

    meta = {'first_year': first_year, 'n_core': n_core, 'n_new': len(new_points),
            'crosswind_dir': crosswind_dir, 'expansion_steps': expansion_step}
    return all_points, meta


def generate_paradigm_D(farm_id, n_turb, centroid, path, extent, coords, wind_2024, paradigm_df,
                         wpd_grid=None, farm_centroid_lonlat=None):
    """
    Paradigm D: High WPD attraction.
    Generates dense candidate grid, scores positions by ERA5-IDW WPD gradient,
    selects top-WPD positions with minimum spacing constraint.
    """
    wrow = wind_2024[wind_2024['farm_id'] == farm_id]
    wpd_val = wrow['WPD_hist'].values[0] if len(wrow) > 0 and pd.notna(wrow['WPD_hist'].values[0]) else 500.0

    theta_energy = get_farm_theta_energy(farm_id, wind_2024, paradigm_df)

    # Step 1: Generate dense candidate grid (tighter spacing for more options)
    Sx_dense = 4.0
    Sy_dense = 4.0
    candidates, _ = generate_grid_in_boundary(centroid, path, extent, n_turb * 3,
                                               Sx_dense, Sy_dense, theta_energy)

    if len(candidates) < n_turb:
        # Fallback: use regular grid at moderate spacing
        points, gmeta = generate_grid_in_boundary(centroid, path, extent, n_turb, 6.5, 6.5, theta_energy)
        meta = {'WPD_hist': wpd_val, 'method': 'fallback_uniform', 'Sx_D': 6.5, 'Sy_D': 6.5}
        return points, meta

    # Step 2: Score candidates by local WPD (IDW interpolation from ERA5 grid)
    if wpd_grid is not None and farm_centroid_lonlat is not None:
        clon, clat = farm_centroid_lonlat
        # Convert each candidate from UTM meters to lon/lat (approximate)
        cos_lat = np.cos(np.radians(clat))
        cand_lons = clon + (candidates[:, 0] - centroid[0]) / (111320.0 * cos_lat)
        cand_lats = clat + (candidates[:, 1] - centroid[1]) / 111320.0

        # IDW interpolation for each candidate
        scores = idw_wpd(cand_lons, cand_lats, wpd_grid, k=8, power=2)
    else:
        # No gradient data: uniform score
        scores = np.ones(len(candidates))

    # Step 3: Greedy selection — highest WPD first, enforcing minimum spacing
    min_spacing_m = 4.0 * D_M  # 4D minimum
    selected = greedy_wpd_select(candidates, scores, n_turb, min_spacing_m)

    meta = {
        'WPD_hist': wpd_val,
        'method': 'wpd_gradient_idw',
        'n_candidates': len(candidates),
        'Sx_eff': f'{min_spacing_m / D_M:.1f}D min',
        'wpd_score_mean': float(np.mean(scores[selected])) if len(selected) > 0 else 0,
    }
    return candidates[selected], meta


def idw_wpd(lons, lats, wpd_grid, k=8, power=2):
    """
    Inverse-distance-weighted WPD interpolation.
    wpd_grid: DataFrame with columns [centroid_lat, centroid_lon, WPD_era5]
    """
    from scipy.spatial import cKDTree

    grid_pts = wpd_grid[['centroid_lat', 'centroid_lon']].values  # (lat, lon)
    grid_wpd = wpd_grid['WPD_era5'].values

    # Build KD-tree in (lat, lon) space
    tree = cKDTree(grid_pts)
    query_pts = np.column_stack([lats, lons])  # match (lat, lon) order

    k_eff = min(k, len(grid_pts))
    dists, idxs = tree.query(query_pts, k=k_eff)

    # IDW: w_i = 1/d^p, result = sum(w_i * v_i) / sum(w_i)
    if k_eff == 1:
        return grid_wpd[idxs]

    eps = 1e-6
    weights = 1.0 / (dists + eps) ** power
    weights /= weights.sum(axis=1, keepdims=True)

    if k_eff > 1:
        return (weights * grid_wpd[idxs]).sum(axis=1)
    else:
        return np.full(len(query_pts), grid_wpd[idxs])


def greedy_wpd_select(positions, scores, n_select, min_spacing):
    """
    Greedy selection: pick highest-scoring positions while respecting min spacing.
    """
    if len(positions) <= n_select:
        return np.arange(len(positions))

    order = np.argsort(scores)[::-1]  # descending score
    selected = []
    for idx in order:
        pt = positions[idx]
        # Check spacing against all selected
        too_close = False
        for s in selected:
            if np.sum((pt - positions[s]) ** 2) < min_spacing ** 2:
                too_close = True
                break
        if not too_close:
            selected.append(idx)
        if len(selected) >= n_select:
            break

    # If not enough selected (spacing too strict), relax and fill
    if len(selected) < n_select:
        remaining = [i for i in order if i not in selected]
        selected.extend(remaining[:n_select - len(selected)])

    return np.array(selected)


def generate_paradigm_E(farm_id, n_turb, centroid, path, extent, coords, theta_params, wind_2024, paradigm_df):
    """
    Paradigm E: Large spacing.
    Same as Paradigm A but with Sx, Sy increased by 25%.
    """
    theta_energy = get_farm_theta_energy(farm_id, wind_2024, paradigm_df)
    crosswind_theta = (theta_energy + 90) % 360

    trow = theta_params[theta_params['farm_id'] == farm_id]
    if len(trow) > 0 and pd.notna(trow['Sx_D'].values[0]):
        Sx = trow['Sx_D'].values[0] * 1.25
        Sy = trow['Sy_D'].values[0] * 1.25
        phi = trow['phi_stagger'].values[0] if pd.notna(trow['phi_stagger'].values[0]) else 0.0
    else:
        Sx = 7.0 * 1.25; Sy = 7.0 * 1.25; phi = 0.0

    Sx = max(5.0, min(25.0, Sx))
    Sy = max(5.0, min(25.0, Sy))

    points, gmeta = generate_grid_in_boundary(centroid, path, extent, n_turb, Sx, Sy, crosswind_theta, phi)

    meta = {'theta_energy': theta_energy, 'grid_theta': crosswind_theta,
            'Sx_D': Sx, 'Sy_D': Sy, 'spacing_increase': '+25%'}
    return points, meta


# ── Main Pipeline ────────────────────────────────────────────

def generate_all_paradigm_layouts(data):
    """
    For each farm × paradigm combination, generate turbine coordinates.
    Returns DataFrame: farm_id, paradigm, year, turbine_id, x_m, y_m, utm_epsg
    """
    paradigm_df = data['paradigm']
    theta_2024 = data['theta_2024']
    wind_2024 = data['wind_2024']
    tcoords_2024 = data['tcoords_2024']
    tcoords_all = data['tcoords']
    tby = data['tby']
    farms = data['farms']

    paradigm_labels = ['A', 'B', 'C', 'D', 'E']
    paradigm_cols = ['P_A', 'P_B', 'P_C', 'P_D', 'P_E']
    generator_map = {
        'A': generate_paradigm_A,
        'B': generate_paradigm_B,
        'C': generate_paradigm_C,
        'D': generate_paradigm_D,
        'E': generate_paradigm_E,
    }

    records = []
    stats = {p: {'generated': 0, 'skipped': 0, 'reason': []} for p in paradigm_labels}

    total_combos = int(paradigm_df[paradigm_cols].sum().sum())
    processed = 0
    print(f"  Total farm x paradigm combinations: {total_combos}")

    # Pre-group turbine coordinates by farm for fast access (avoids repeated filtering)
    tcoords_by_farm = {fid: grp for fid, grp in tcoords_all.groupby('farm_id')}
    tcoords_2024_by_farm = {fid: grp[['x_m', 'y_m']].values
                            for fid, grp in tcoords_2024.groupby('farm_id')}

    # Process each farm
    for farm_idx, (_, farm_row) in enumerate(paradigm_df.iterrows()):
        farm_id = int(farm_row['farm_id'])
        n_turb = int(farm_row['n_turb'])

        if farm_idx % 20 == 0:
            print(f"  Processing farm {farm_idx}/171 (id={farm_id})...", flush=True)

        # Get real turbine coordinates from pre-computed dict
        real_coords_raw = tcoords_2024_by_farm.get(farm_id, np.array([]).reshape(0, 2))
        if len(real_coords_raw) == 0:
            continue

        centroid, hull_verts, path, extent = get_farm_boundary(real_coords_raw, buffer_m=500)

        # Get UTM EPSG (from tcoords 2024)
        farm_epsg = 32650
        farm_utm_2024 = tcoords_2024[tcoords_2024['farm_id'] == farm_id]
        if len(farm_utm_2024) > 0 and 'utm_epsg' in farm_utm_2024.columns:
            epsg_vals = farm_utm_2024['utm_epsg'].dropna()
            if len(epsg_vals) > 0:
                epsg_str = str(epsg_vals.values[0])
                farm_epsg = int(epsg_str.split(':')[1]) if ':' in epsg_str else int(epsg_str)

        # Determine which paradigms this farm belongs to
        farm_paradigms = [pl for pl, pc in zip(paradigm_labels, paradigm_cols) if farm_row[pc] == 1]

        for paradigm in farm_paradigms:
            processed += 1
            if processed % 50 == 0:
                print(f"  Progress: {processed}/{total_combos} farm×paradigm combinations...")

            generator = generator_map[paradigm]

            try:
                if paradigm == 'C':
                    farm_tcoords_hist = tcoords_by_farm.get(farm_id)
                    points, meta = generator(
                        farm_id, n_turb, centroid, path, extent, real_coords_raw,
                        farm_tcoords_hist, wind_2024, paradigm_df
                    )
                elif paradigm == 'D':
                    lonlat = data['farm_lonlat'].get(farm_id, (0, 0))
                    points, meta = generator(
                        farm_id, n_turb, centroid, path, extent, real_coords_raw,
                        wind_2024, paradigm_df,
                        wpd_grid=data.get('wpd_grid'),
                        farm_centroid_lonlat=lonlat
                    )
                elif paradigm in ('A', 'E'):
                    points, meta = generator(
                        farm_id, n_turb, centroid, path, extent, real_coords_raw,
                        theta_2024, wind_2024, paradigm_df
                    )
                else:  # B
                    points, meta = generator(
                        farm_id, n_turb, centroid, path, extent, real_coords_raw,
                        theta_2024, paradigm_df
                    )

                if len(points) < MIN_TURBINES:
                    stats[paradigm]['skipped'] += 1
                    stats[paradigm]['reason'].append(f'farm_{farm_id}: only {len(points)} pts')
                    continue

                # Generate turbine IDs
                for ti, (px, py) in enumerate(points):
                    records.append({
                        'farm_id': farm_id,
                        'paradigm': paradigm,
                        'year': 2024,  # reference year
                        'turbine_id': f'{farm_id}_{paradigm}_{ti}',
                        'x_m': round(px, 1),
                        'y_m': round(py, 1),
                        'utm_epsg': farm_epsg,
                        'n_turb_total': len(points),
                        'Sx_D': meta.get('Sx_D', np.nan),
                        'Sy_D': meta.get('Sy_D', np.nan),
                        'grid_theta_deg': meta.get('grid_theta', meta.get('pca_direction', meta.get('crosswind_dir', np.nan))),
                    })

                stats[paradigm]['generated'] += 1

            except Exception as e:
                stats[paradigm]['skipped'] += 1
                stats[paradigm]['reason'].append(f'farm_{farm_id}: {str(e)[:80]}')

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PARADIGM LAYOUT GENERATION SUMMARY")
    print("=" * 60)
    total_generated = 0
    for p in paradigm_labels:
        g = stats[p]['generated']
        s = stats[p]['skipped']
        n_hit = int(paradigm_df[f'P_{p}'].sum())
        total_generated += g
        print(f"\n  Paradigm {p}: {g}/{n_hit} generated, {s} skipped")
        if stats[p]['reason']:
            for r in stats[p]['reason'][:3]:
                print(f"    - {r}")
            if len(stats[p]['reason']) > 3:
                print(f"    ... and {len(stats[p]['reason']) - 3} more")

    print(f"\n  Total records: {len(records)}")
    print(f"  Total farm×paradigm: {total_generated}")

    # ── Save ──
    df_out = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, 'paradigm_layouts.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Size: {len(df_out):,} rows × {len(df_out.columns)} cols")

    return df_out


# ============================================================
if __name__ == "__main__":
    print("TASK 3 — S2: Paradigm Layout Generation")
    print("=" * 60)

    data = load_data()
    df = generate_all_paradigm_layouts(data)

    print("\nDone!")
