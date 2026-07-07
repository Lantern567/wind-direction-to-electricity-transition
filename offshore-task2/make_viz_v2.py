"""
=============================================================================
 make_viz_v2.py — 任务二可视化 (FLORIS版)
 8 类图表，对齐任务书 §1.9 / 修改意见 §4.3 / 审计报告 §四
 配色沿用 dataviz 规范: categorical 8-hue, sequential blue ramp
=============================================================================
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floris_config import OUT_DIR, WAKE_MODELS

FIG_DIR = os.path.join(OUT_DIR, "figures_v2")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Palette (dataviz categorical 8-hue) ----
CAT = ['#2a78d6', '#1baf7a', '#eda100', '#008300',
       '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']
SEQ_BLUE = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#2a78d6','#1c5cab','#104281']
SURFACE = '#fcfcfb'
TEXT    = '#0b0b0b'
MUTED   = '#52514e'
GRID    = '#e0ded9'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.3,
    'grid.color': GRID,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
})


# =========================================================================
# HELPERS
# =========================================================================

def load_data(csv_path=None):
    """Load FLORIS annual summary CSV."""
    if csv_path is None:
        csv_path = os.path.join(OUT_DIR, "task2_annual_floris.csv")
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            r2 = {}
            for k, v in r.items():
                try:
                    r2[k] = float(v)
                except (ValueError, TypeError):
                    r2[k] = v
            rows.append(r2)
    return rows

def gauss_only(rows):
    return [r for r in rows if r.get('wake_model','') == 'gauss']

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"  Saved: {name}")

def stat_box(ax, data, labels, colors=None, vert=True):
    """Clean boxplot with thin marks."""
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                     widths=0.55, vert=vert,
                     flierprops=dict(marker='.', ms=3, alpha=0.4),
                     medianprops=dict(color=TEXT, lw=1.2),
                     whiskerprops=dict(lw=0.8),
                     capprops=dict(lw=0.8))
    if colors:
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(c)
            patch.set_linewidth(0.8)
    ax.tick_params(axis='both', colors=MUTED)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)


# =========================================================================
# FIG 1 — Wake Flow Overlay (3 farms, 2 wind directions)
# =========================================================================

def fig1_wake_flow():
    """尾流流场叠加图: 真实风机点位 + U_eff 色块流场
    Uses FLORIS calculate_cut_plane for selected farms.
    """
    print("\n Fig 1: Wake Flow Overlay")
    # This needs FLORIS runtime — placeholder that runs when called
    try:
        from floris import FlorisModel
        from floris_config import create_floris_model, load_task0_coordinates, load_farms_master
        from floris_config import get_ti_for_farm, DEFAULT_TURBINE

        coords = load_task0_coordinates()
        farms = load_farms_master()
        rep = [(0,0.08, "East Asia mega (928t)"), (2,0.06, "Belgium cluster (572t)"),
               (5,0.06, "UK belt (339t)")]

        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        wds = [270, 320]

        for row_idx, (fid, ti, label) in enumerate(rep):
            turbs = coords[fid].get(2024, [])[:300]  # limit for speed
            xs = [float(t['x_m'])/1000 for t in turbs]
            ys = [float(t['y_m'])/1000 for t in turbs]
            cx, cy = np.mean(xs), np.mean(ys)

            for col_idx, wd in enumerate(wds):
                ax = axes[row_idx, col_idx]
                ws_test = 8.0
                try:
                    fm, tmp = create_floris_model(
                        [float(t['x_m']) for t in turbs],
                        [float(t['y_m']) for t in turbs],
                        DEFAULT_TURBINE, ti, 0.11, 'gauss')
                    fm.set(wind_speeds=[ws_test], wind_directions=[wd],
                           turbulence_intensities=[ti])
                    fm.run()

                    # Calculate horizontal cut plane at hub height
                    cp = fm.calculate_cut_plane(
                        x_resolution=80, y_resolution=80,
                        height=fm.core.farm.hub_heights[0],
                        x_bounds=(min(xs)*1000-500, max(xs)*1000+500),
                        y_bounds=(min(ys)*1000-500, max(ys)*1000+500),
                    )
                    im = ax.pcolormesh(cp.df.x1/1000, cp.df.x2/1000,
                                        cp.df.u, cmap='YlOrRd',
                                        vmin=0, vmax=ws_test,
                                        shading='auto', rasterized=True)

                    # Turbine positions
                    ax.scatter(xs, ys, s=3, c='#2a78d6', edgecolors='none', zorder=5)
                    ax.set_title(f"{label}\nWD={wd}deg @ {ws_test}m/s", fontsize=9)
                    os.unlink(tmp)
                except Exception as e:
                    ax.text(0.5, 0.5, f"FLORIS error:\n{str(e)[:60]}",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.scatter(xs, ys, s=3, c='#2a78d6', edgecolors='none')
                    ax.set_title(f"{label} (positions only)", fontsize=9)

                ax.set_aspect('equal')
                ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')

        plt.tight_layout()
        save("fig1_wake_flow.png")
    except Exception as e:
        print(f"  Fig 1 skipped (FLORIS runtime): {e}")


# =========================================================================
# FIG 2 — Hourly Time Series
# =========================================================================

def fig2_hourly_timeseries():
    """逐时出力时间序列: P_noWake vs P_wake, 叠加风速风向"""
    print("\n Fig 2: Hourly Time Series")
    # Read from existing hourly CSV
    hourly_path = os.path.join(OUT_DIR, "task2_hourly_F0.csv")
    if not os.path.exists(hourly_path):
        print("  Skipped: no hourly CSV found")
        return

    rows = []
    with open(hourly_path, 'r', encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= 500:
                break
            rows.append(r)

    h = np.arange(len(rows))
    pn = np.array([float(r.get('P_noWake_kW',0)) for r in rows]) / 1000  # MW
    pw = np.array([float(r.get('P_wake_Gaussian_kW',0)) for r in rows]) / 1000
    ws = np.array([float(r.get('V_free_ms',0)) for r in rows])
    wd = np.array([float(r.get('theta_deg',0)) for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})

    ax1.fill_between(h, pn, pw, alpha=0.25, color=CAT[0], label='Wake Loss')
    ax1.plot(h, pn, color=MUTED, lw=1.0, label='P_noWake')
    ax1.plot(h, pw, color=CAT[0], lw=1.2, label='P_wake (Gaussian)')
    ax1.set_ylabel('Farm Power (MW)')
    ax1.legend(loc='upper right', frameon=False)
    ax1.set_title('Hourly Power: F0 (928 turbines), First 500 Hours')

    # Wind overlay
    ax2.plot(h, ws, color=CAT[1], lw=1.0, label='Wind Speed (m/s)')
    ax2.set_ylabel('Wind Speed (m/s)', color=CAT[1])
    ax2.tick_params(axis='y', colors=CAT[1])
    ax2b = ax2.twinx()
    ax2b.scatter(h[::10], wd[::10], s=2, color=CAT[2], alpha=0.5, label='WD (10-min)')
    ax2b.set_ylabel('Wind Dir (deg)', color=CAT[2])
    ax2b.tick_params(axis='y', colors=CAT[2])
    ax2.set_xlabel('Hour')

    plt.tight_layout()
    save("fig2_hourly_timeseries.png")


# =========================================================================
# FIG 3 — Wake vs NoWake + Per-Turbine Heat Map
# =========================================================================

def fig3_wake_heatmap():
    """有/无尾流对比 + 逐台尾流亏损热力图"""
    print("\n Fig 3: Wake vs NoWake + Per-Turbine Heat Map")
    rows = gauss_only(load_data())
    if not rows:
        print("  Skipped: no data")
        return

    # Left: scatter P_noWake vs P_wake
    pn = np.array([r['AEP_noWake_kWh']/1e9 for r in rows])  # TWh
    pw = np.array([r['AEP_kWh']/1e9 for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax1.scatter(pn, pw, s=8, c=CAT[0], alpha=0.4, edgecolors='none')
    mm = min(pn.min(), pw.min())
    mx = max(pn.max(), pw.max())
    ax1.plot([mm, mx], [mm, mx], '--', color=MUTED, lw=0.8, label='1:1')
    ax1.set_xlabel('AEP_noWake (TWh)'); ax1.set_ylabel('AEP_wake (TWh)')
    ax1.set_title('Wake vs No-Wake AEP')
    ax1.legend(frameon=False)

    # Right: WakeLoss histogram by region
    regions = defaultdict(list)
    for r in rows:
        regions[r.get('region','other')].append(float(r['WakeLoss'])*100)

    region_order = ['east_asia','europe','us_east','japan']
    data = [regions.get(ro, [0]) for ro in region_order]
    labels = ['E.Asia','Europe','US East','Japan']
    colors = CAT[:4]
    stat_box(ax2, data, labels, colors)
    ax2.set_ylabel('Wake Loss (%)')
    ax2.set_title('Wake Loss by Region')

    plt.tight_layout()
    save("fig3_wake_heatmap.png")


# =========================================================================
# FIG 4 — Power Curves (multi-turbine)
# =========================================================================

def fig4_power_curves():
    """功率曲线 + Ct 曲线 (多机型叠加)"""
    print("\n Fig 4: Power Curves")
    from floris_config import get_power_curve, get_turbine_params

    turbine_types = ["ow_6MW", "ow_8MW", "ow_10MW", "ow_12MW", "ow_15MW"]
    colors = CAT[:5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ttype, c in zip(turbine_types, colors):
        tp = get_turbine_params(ttype)
        ws = np.array(tp['power_table']['wind_speed'])
        pw = np.array(tp['power_table']['power'])
        ct = np.array(tp['power_table']['thrust_coefficient'])

        ax1.plot(ws, pw/1000, color=c, lw=1.2, label=f"{ttype}")
        ax2.plot(ws[1:], ct[1:], color=c, lw=1.2, label=f"{ttype}")

    ax1.axvline(3, color=MUTED, ls='--', lw=0.8, alpha=0.5)
    ax1.axvline(25, color=MUTED, ls='--', lw=0.8, alpha=0.5)
    ax1.set_xlabel('Wind Speed (m/s)'); ax1.set_ylabel('Power (MW)')
    ax1.set_title('Power Curves')
    ax1.legend(frameon=False, loc='lower right')
    ax1.set_xlim(2, 26)

    ax2.set_xlabel('Wind Speed (m/s)'); ax2.set_ylabel('Thrust Coefficient')
    ax2.set_title('Ct Curves')
    ax2.set_xlim(3, 25)

    plt.tight_layout()
    save("fig4_power_curves.png")


# =========================================================================
# FIG 5 — dAEP Global Map
# =========================================================================

def fig5_daep_map():
    """ΔAEP_WD 全球地图 (反事实风向变化)"""
    print("\n Fig 5: dAEP Global Map")
    cf_path = os.path.join(OUT_DIR, "task2_counterfactual.csv")
    if not os.path.exists(cf_path):
        print("  Skipped: no counterfactual CSV (will generate after batch)")
        return

    rows = []
    with open(cf_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Join with farm locations
    from floris_config import load_farms_master
    farms = load_farms_master()

    lons, lats, deltas, sizes = [], [], [], []
    for r in rows:
        fid = int(r.get('farm_id', -1))
        if fid in farms:
            lons.append(farms[fid]['centroid_lon'])
            lats.append(farms[fid]['centroid_lat'])
            delta = float(r.get('Delta_AEP_GWh', r.get('dAEP_GWh', 0)))
            deltas.append(delta)
            sizes.append(farms[fid].get('capacity_kW', 0)/1000 * 0.5)

    lons = np.array(lons); lats = np.array(lats)
    deltas = np.array(deltas); sizes = np.array(sizes)

    fig, ax = plt.subplots(figsize=(14, 7))
    pos_mask = deltas > 0
    neg_mask = deltas < 0

    ax.scatter(lons[neg_mask], lats[neg_mask], s=np.abs(sizes[neg_mask])*2,
               c=CAT[5], alpha=0.55, edgecolors='none', label=f'ΔAEP<0 ({neg_mask.sum()})')
    ax.scatter(lons[pos_mask], lats[pos_mask], s=np.abs(sizes[pos_mask])*2,
               c=CAT[0], alpha=0.55, edgecolors='none', label=f'ΔAEP>0 ({pos_mask.sum()})')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('ΔAEP_WD: Real vs Baseline Wind Direction')
    ax.legend(frameon=False)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 70)

    plt.tight_layout()
    save("fig5_daep_map.png")


# =========================================================================
# FIG 6 — 3-Model Comparison
# =========================================================================

def fig6_model_comparison():
    """三尾流模型对比 (Gaussian/Jensen/CC)"""
    print("\n Fig 6: 3-Model Comparison")
    rows = load_data()
    if not rows:
        print("  Skipped: no data")
        return

    models_data = defaultdict(lambda: {'CF':[], 'WL':[], 'Vol':[]})
    for r in rows:
        wm = r.get('wake_model','gauss')
        models_data[wm]['CF'].append(float(r['CF'])*100)
        models_data[wm]['WL'].append(float(r['WakeLoss'])*100)
        models_data[wm]['Vol'].append(float(r.get('Volatility_kW',0))/1e3)

    model_order = ['gauss','jensen','cc']
    colors_m = [CAT[0], CAT[1], CAT[2]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, metric in enumerate(['CF','WL','Vol']):
        ax = axes[idx]
        data = [models_data.get(m,{}).get(metric,[0]) for m in model_order]
        labels = ['Gauss','Jensen','CC']
        stat_box(ax, data, labels, colors_m)
        titles = ['Capacity Factor (%)', 'Wake Loss (%)', 'Volatility (MW)']
        ax.set_title(titles[idx])
        ax.set_ylabel(titles[idx])

    plt.suptitle('3 Wake Models: Robustness Check', y=1.01, fontsize=12)
    plt.tight_layout()
    save("fig6_model_comparison.png")


# =========================================================================
# FIG 7 — Rotation-AEP Response Curve
# =========================================================================

def fig7_rotation_response():
    """旋转-AEP 响应曲线 (排布敏感性自检)"""
    print("\n Fig 7: Rotation-AEP Response")
    rot_path = os.path.join(OUT_DIR, "audit_rotation_floris.csv")
    if not os.path.exists(rot_path):
        print("  Skipped: no rotation CSV")
        return

    rot_data = defaultdict(list)
    with open(rot_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid = int(r['farm_id'])
            rot_data[fid].append((float(r['angle_deg']),
                                   float(r['wake_efficiency'])))

    rep = sorted(rot_data.keys())[:4]
    colors = CAT[:4]

    fig, ax = plt.subplots(figsize=(9, 5))

    for fid, c in zip(rep, colors):
        angles, etas = zip(*sorted(rot_data[fid]))
        ax.plot(angles, np.array(etas)*100, '-', color=c, lw=1.5,
                marker='o', ms=4, label=f'F{fid}')

    ax.set_xlabel('Rotation Angle (deg)'); ax.set_ylabel('Wake Efficiency (%)')
    ax.set_title('Farm Rotation → AEP Response (proves geometry used by FLORIS)')
    ax.legend(frameon=False, title='Farm ID')
    ax.set_xlim(0, 180)

    plt.tight_layout()
    save("fig7_rotation_response.png")


# =========================================================================
# FIG 8 — CF/WakeLoss Distribution
# =========================================================================

def fig8_distribution():
    """CF / WakeLoss / 波动指标 分布图 (分区域)"""
    print("\n Fig 8: CF/WakeLoss Distribution by Region")
    rows = gauss_only(load_data())
    if not rows:
        print("  Skipped: no data")
        return

    regions = defaultdict(list)
    for r in rows:
        reg = r.get('region','other')
        regions[reg].append({
            'cf': float(r['CF'])*100,
            'wl': float(r['WakeLoss'])*100,
            'cv': float(r.get('CV',0))*100,
        })

    region_order = ['east_asia','europe','us_east','japan']
    labels = ['E.Asia','Europe','US East','Japan']
    reg_colors = CAT[:4]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (metric, ylabel) in enumerate([('cf','CF (%)'), ('wl','WL (%)'), ('cv','CV (%)')]):
        ax = axes[idx]
        data = [[d[metric] for d in regions.get(ro, [{'cf':0,'wl':0,'cv':0}])]
                for ro in region_order]
        stat_box(ax, data, labels, reg_colors)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)

    plt.suptitle('CF / WakeLoss / CV by Region (Gauss, all years)', y=1.01)
    plt.tight_layout()
    save("fig8_distribution.png")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 56)
    print(" 任务二 可视化 v2.0 (FLORIS)")
    print(f" Output: {FIG_DIR}")
    print("=" * 56)

    fig1_wake_flow()
    fig2_hourly_timeseries()
    fig3_wake_heatmap()
    fig4_power_curves()
    fig5_daep_map()
    fig6_model_comparison()
    fig7_rotation_response()
    fig8_distribution()

    print(f"\nDone! {len(os.listdir(FIG_DIR))} figures in {FIG_DIR}")

if __name__ == "__main__":
    main()
