"""v3 完整可视化 — 使用 task1_summary_v3.csv（三尾流模型 + Volatility/RampFreq）"""
import os, csv, math, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cartopy.crs as ccrs; import cartopy.feature as cfeature
from sklearn.cluster import DBSCAN

CSV = "output/task1_summary_v3.csv"
CF_CSV = "output/task1_counterfactual.csv"
FIG_DIR = "output/figures"
os.makedirs(FIG_DIR, exist_ok=True)

rows = []
with open(CSV, 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f): rows.append(r)
gauss = [r for r in rows if r['wake_model'] == 'gaussian']
print(f"Records: {len(rows)} ({len(gauss)} Gaussian)")

cf_rows = []
if os.path.exists(CF_CSV):
    with open(CF_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): cf_rows.append(r)

turbs = []
with open("data/turbine_yearly.csv", 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        turbs.append({'id': int(r['id']), 'lon': float(r['lon']), 'lat': float(r['lat'])})

REGIONS = {
    "east_asia": {"bbox": [41, 104, 8, 142], "extent": [104, 142, 6, 42], "name": "East Asia"},
    "europe":    {"bbox": [63, -10, 39, 22], "extent": [-12, 24, 37, 64], "name": "Europe"},
    "us_east":   {"bbox": [42, -77, 36, -69], "extent": [-78, -68, 34, 43], "name": "US East"},
}
clr = {'east_asia': '#3498db', 'europe': '#e67e22', 'us_east': '#2ecc71'}

# ===== Fig 1: Wake model comparison =====
print("1/6 Wake model comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for pi, (key, lbl) in enumerate([('CF', 'CF (%)'), ('WakeLoss', 'Wake Loss (%)'), ('Volatility_kW', 'Vol (kW)')]):
    ax = axes[pi]; wms = ['gaussian', 'jensen', 'curl']
    vals = [[float(r[key]) * (100 if key != 'Volatility_kW' else 1) for r in rows if r['wake_model'] == wm] for wm in wms]
    ax.bar(wms, [np.mean(v) for v in vals], color=['steelblue', 'darkorange', 'seagreen'])
    ax.set_ylabel(lbl); ax.set_title(f'Mean {lbl}')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'v3_fig1_wake_cmp.png'), dpi=150)
plt.close(fig)

# ===== Fig 2: Global distributions (Gaussian) =====
print("2/6 Global distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, key, lbl, c in [
    (axes[0,0], 'CF', 'CF (%)', 'steelblue'),
    (axes[0,1], 'WakeLoss', 'Wake Loss (%)', 'coral'),
    (axes[1,0], 'Volatility_kW', 'Volatility (kW)', 'mediumseagreen'),
    (axes[1,1], 'RampFreq', 'Ramp Frequency', 'mediumpurple'),
]:
    mult = 100 if key in ('CF', 'WakeLoss') else 1
    vals = [float(r[key]) * mult for r in gauss]
    ax.hist(vals, bins=40, color=c, edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(vals), color='red', linestyle='--', label=f'Mean={np.mean(vals):.1f}')
    ax.set_xlabel(lbl); ax.set_title(lbl); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'v3_fig2_global_dists.png'), dpi=150)
plt.close(fig)

# ===== Fig 3: Regional maps (Gaussian, 3-panel per region) =====
print("3/6 Regional maps...")
for rk, cfg in REGIONS.items():
    active = [t for t in turbs if cfg['bbox'][2] <= t['lat'] <= cfg['bbox'][0] and cfg['bbox'][1] <= t['lon'] <= cfg['bbox'][3]]
    if len(active) < 3: continue
    xy = np.array([[t['lon'], t['lat']] for t in active])
    mlat = np.mean(xy[:, 1]); kx, ky = 111.32 * math.cos(math.radians(mlat)), 111.32
    xyk = xy.copy(); xyk[:, 0] *= kx; xyk[:, 1] *= ky
    labels = DBSCAN(eps=5.0, min_samples=3, metric='euclidean').fit_predict(xyk)
    farms = defaultdict(list)
    for i, t in enumerate(active):
        if labels[i] >= 0: farms[int(labels[i])].append(t)
    centers = {fid: (np.mean([t['lon'] for t in ft]), np.mean([t['lat'] for t in ft])) for fid, ft in farms.items()}

    rk_rows = [r for r in gauss if r['region'] == rk]
    fdata = {}
    for r in rk_rows:
        fid = int(r['farm_id']); yr = int(r['year'])
        cf = float(r['CF'])*100; wl = float(r['WakeLoss'])*100; vol = float(r['Volatility_kW'])
        if fid not in fdata or yr > fdata[fid][0]:
            fdata[fid] = (yr, cf, wl, vol)

    fig = plt.figure(figsize=(24, 8))
    for pi, (title, vm, cmap, vmin, vmax, lbl) in enumerate([
        ('CF (%)', {f: fdata[f][1] for f in fdata}, 'RdYlGn', 15, 65, 'CF (%)'),
        ('Wake Loss (%)', {f: fdata[f][2] for f in fdata}, 'coolwarm', 0, 25, 'WL (%)'),
        ('Vol (MW)', {f: fdata[f][3]/1000 for f in fdata}, 'YlOrRd', 0, 10, 'Vol (MW)'),
    ]):
        ax = plt.subplot(1, 3, pi+1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
        ax.set_extent(cfg['extent'])
        for fid, val in vm.items():
            if fid in centers:
                sz = max(15, min(250, len(farms[fid])*2.5))
                ax.scatter(*centers[fid], s=sz, c=val, cmap=cmap, vmin=vmin, vmax=vmax,
                           edgecolors='#333', linewidths=0.3, transform=ccrs.PlateCarree())
        plt.colorbar(ax.collections[0], ax=ax, shrink=0.6).set_label(lbl)
        ax.set_title(title)
    fig.suptitle(cfg['name'], fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'v3_fig3_map_{rk}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {cfg['name']} OK")

# ===== Fig 4: Counterfactual =====
print("4/6 Counterfactual...")
if cf_rows:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    das = [float(r['Delta_AEP_WD_kWh'])/1e6 for r in cf_rows]
    dws = [float(r['Delta_WakeLoss_WD'])*100 for r in cf_rows]
    lbs = [f"{r['region'][:3]}-F{r['farm_id']}-{r['year']}" for r in cf_rows]
    ax1.barh(range(len(das)), das, color=['green' if v>=0 else 'red' for v in das])
    ax1.set_yticks(range(len(das))); ax1.set_yticklabels(lbs, fontsize=7)
    ax1.axvline(0, color='black'); ax1.set_xlabel('Delta AEP (GWh)'); ax1.set_title('Delta AEP: Real vs Baseline')
    ax2.barh(range(len(dws)), dws, color=['red' if v>0 else 'green' for v in dws])
    ax2.set_yticks(range(len(dws))); ax2.set_yticklabels(lbs, fontsize=7)
    ax2.axvline(0, color='black'); ax2.set_xlabel('Delta WakeLoss (%)'); ax2.set_title('Delta WakeLoss: Real vs Baseline')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'v3_fig4_counterfactual.png'), dpi=150)
    plt.close(fig)

# ===== Fig 5: Regional overlay histograms =====
print("5/6 Regional overlay...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for (key, lbl), ax in zip([('CF', 'CF (%)'), ('WakeLoss', 'WL (%)'), ('Volatility_kW', 'Vol (kW)'), ('RampFreq', 'RampFreq')], axes.flat):
    mult = 100 if key in ('CF', 'WakeLoss') else 1
    for rk in REGIONS:
        vals = [float(r[key])*mult for r in gauss if r['region'] == rk]
        if vals: ax.hist(vals, alpha=0.4, bins=30, label=f"{REGIONS[rk]['name']} (n={len(vals)})", color=clr[rk])
    ax.set_xlabel(lbl); ax.legend(fontsize=8); ax.set_title(lbl)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'v3_fig5_regional_dists.png'), dpi=150)
plt.close(fig)

# ===== Fig 6: Scatter CF/WL per model =====
print("6/6 Scatter...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for wi, wm in enumerate(['gaussian', 'jensen', 'curl']):
    ax = axes[wi]; wm_rows = [r for r in rows if r['wake_model'] == wm]
    for rk in REGIONS:
        rk_r = [r for r in wm_rows if r['region'] == rk]
        if rk_r:
            ax.scatter([float(r['CF'])*100 for r in rk_r], [float(r['WakeLoss'])*100 for r in rk_r],
                       s=5, alpha=0.4, c=clr[rk], label=REGIONS[rk]['name'])
    ax.set_xlabel('CF (%)'); ax.set_ylabel('Wake Loss (%)'); ax.set_title(wm.capitalize()); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'v3_fig6_scatter_by_model.png'), dpi=150)
plt.close(fig)

print(f"\nDone! {FIG_DIR}/")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('v3'):
        print(f"  {f} ({os.path.getsize(os.path.join(FIG_DIR,f))//1024}KB)")
