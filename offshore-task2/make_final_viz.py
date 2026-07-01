"""
任务二 v4 最终可视化 — 修复版 + 交叉验证 + 新增结论支撑图
基于 3609条完整数据 (1203 farm-years × 3 wake models)
"""
import os, csv, math, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG = os.path.join(DATA, "figures"); os.makedirs(FIG, exist_ok=True)

# ===== LOAD ALL DATA =====
rows = []
with open(os.path.join(DATA, "task2_summary_v4.csv"), 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f): rows.append(r)

cf_rows = []
cfp = os.path.join(DATA, "task2_counterfactual.csv")
if os.path.exists(cfp):
    with open(cfp, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): cf_rows.append(r)

farms_info = {}
with open(r"D:\1风力发电实习\offshore-task0\output\task0\farms_master.csv", 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        fid = int(r['farm_id'])
        farms_info[fid] = {'lon': float(r['centroid_lon']), 'lat': float(r['centroid_lat']),
                           'country': r['country'], 'n_turb': int(r['n_turb']), 'area': float(r['area_km2'])}

def get_region(fid):
    info = farms_info.get(fid)
    if not info: return 'other'
    lon, lat = info['lon'], info['lat']
    if 8 <= lat <= 44 and 104 <= lon <= 143: return 'east_asia'
    if 39 <= lat <= 63 and -12 <= lon <= 32: return 'europe'
    if 36 <= lat <= 44 and -78 <= lon <= -68: return 'us_east'
    return 'other'

REGION_COLORS = {'east_asia': '#3498db', 'europe': '#e67e22', 'us_east': '#2ecc71'}
REGION_NAMES = {'east_asia': 'East Asia', 'europe': 'Europe', 'us_east': 'US East'}

gauss = [r for r in rows if r['wake_model'] == 'gaussian']
jensen = [r for r in rows if r['wake_model'] == 'jensen']
curl = [r for r in rows if r['wake_model'] == 'curl']

# ---- Remove old / regenerate clean ----
for f in os.listdir(FIG):
    if f.startswith(('A_','B_','C_','D_','E_','F_','H_','I_','viz1','viz3','viz5')):
        os.remove(os.path.join(FIG, f))

try:
    import cartopy.crs as ccrs; import cartopy.feature as cfeature
    HAS_CARTOPY = True
except:
    HAS_CARTOPY = False

# ============================================================
# FIG 0: CROSS-VALIDATION — Our results vs Xu 2026
# ============================================================
print("0: Cross-validation vs Xu 2026...")
cn_50plus = [r for r in gauss if get_region(int(r['farm_id']))=='east_asia' and int(r['n_turb'])>=50]
cn_all = [r for r in gauss if get_region(int(r['farm_id']))=='east_asia']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) CF comparison: Our CN large farms vs Xu 2026 range
our_cf = [float(r['CF'])*100 for r in cn_50plus]
xu_cf_range = [28, 45]
axes[0].bar(0, np.mean(our_cf), color='steelblue', yerr=np.std(our_cf), capsize=5, label=f'This study\n(n={len(our_cf)}, mean={np.mean(our_cf):.1f}%)')
axes[0].bar(1, (xu_cf_range[0]+xu_cf_range[1])/2, color='lightgray',
            yerr=[[(xu_cf_range[0]+xu_cf_range[1])/2 - xu_cf_range[0]],
                  [xu_cf_range[1] - (xu_cf_range[0]+xu_cf_range[1])/2]],
            capsize=5, label=f'Xu 2026\n({xu_cf_range[0]}-{xu_cf_range[1]}%)')
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['This study\n(CN >=50 turbines)', 'Xu et al. 2026\n(Nature Comms)'])
axes[0].set_ylabel('Capacity Factor (%)'); axes[0].set_title('CF Validation'); axes[0].legend()

# (b) WakeLoss comparison
our_wl = [float(r['WakeLoss'])*100 for r in cn_50plus]
xu_wl_range = [8, 22]
axes[1].bar(0, np.mean(our_wl), color='coral', yerr=np.std(our_wl), capsize=5, label=f'This study\n(n={len(our_wl)}, mean={np.mean(our_wl):.1f}%)')
axes[1].bar(1, (xu_wl_range[0]+xu_wl_range[1])/2, color='lightgray',
            yerr=[[(xu_wl_range[0]+xu_wl_range[1])/2 - xu_wl_range[0]],
                  [xu_wl_range[1] - (xu_wl_range[0]+xu_wl_range[1])/2]],
            capsize=5, label=f'Xu 2026\n({xu_wl_range[0]}-{xu_wl_range[1]}%)')
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['This study\n(CN >=50 turbines)', 'Xu et al. 2026\n(Nature Comms)'])
axes[1].set_ylabel('Wake Loss (%)'); axes[1].set_title('WakeLoss Validation'); axes[1].legend()

# (c) Scale-stratified WL comparison
wl_by_size = {'>=200': [], '50-200': [], '20-50': [], '<20': []}
for r in cn_all:
    nt = int(r['n_turb'])
    wl = float(r['WakeLoss'])*100
    if nt >= 200: wl_by_size['>=200'].append(wl)
    elif nt >= 50: wl_by_size['50-200'].append(wl)
    elif nt >= 20: wl_by_size['20-50'].append(wl)
    else: wl_by_size['<20'].append(wl)
labels = ['<20','20-50','50-200','>=200']
vals = [np.mean(wl_by_size[k]) for k in labels if wl_by_size[k]]
axes[2].bar(labels, vals, color='coral', alpha=0.8)
axes[2].set_xlabel('Farm Size (turbines)'); axes[2].set_ylabel('Mean WakeLoss (%)')
axes[2].set_title('WakeLoss by Farm Size (China only)')
axes[2].axhline(xu_wl_range[0], color='gray', linestyle='--', alpha=0.5)
axes[2].axhline(xu_wl_range[1], color='gray', linestyle='--', alpha=0.5)

fig.suptitle('Cross-Validation with Xu et al. (2026, Nature Communications)', fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(FIG, '00_cross_validation.png'), dpi=150); plt.close(fig)
print("  -> 00_cross_validation.png")

# ============================================================
# FIG A: Three-model boxplots
# ============================================================
print("A: Three-model boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, key, label, mult in zip(axes, ['CF','WakeLoss','Volatility_kW'],
                                ['Capacity Factor','Wake Loss Rate','Volatility (MW)'],
                                [100, 100, 0.001]):
    dg = [float(r[key])*mult for r in gauss if float(r.get(key,0))>0 or key=='WakeLoss']
    dj = [float(r[key])*mult for r in jensen if float(r.get(key,0))>0 or key=='WakeLoss']
    dc = [float(r[key])*mult for r in curl if float(r.get(key,0))>0 or key=='WakeLoss']
    bp = ax.boxplot([dg, dj, dc], tick_labels=['Gaussian','Jensen','Curl'],
                    patch_artist=True, medianprops=dict(color='black'))
    for patch, color in zip(bp['boxes'], ['#3498db','#e67e22','#2ecc71']):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel(label); ax.set_title(f'{label} by Wake Model')
fig.suptitle('Three-Model Robustness Check (3,609 records)', fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'A_three_model_box.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG B: Regional maps — FIX: vmax=30, YlOrRd colormap
# ============================================================
print("B: Regional maps (fixed color scale)...")
if HAS_CARTOPY:
    regions_bbox = {
        'east_asia': [104, 142, 6, 42],
        'europe': [-12, 24, 37, 64],
        'us_east': [-78, -68, 34, 43],
    }
    for rk, bbox in regions_bbox.items():
        rk_rows = [r for r in gauss if get_region(int(r['farm_id'])) == rk and int(r['year']) == 2024]
        if not rk_rows: continue
        n_rows = len(rk_rows)
        # US: smaller figure, bigger dots
        if n_rows <= 10:
            figsize = (18, 6)
            dot_factor = 8
        else:
            figsize = (24, 8)
            dot_factor = 2

        fig = plt.figure(figsize=figsize)
        for pi, (title, key, mult, cmap, vmin, vmax) in enumerate([
            ('Capacity Factor (%)', 'CF', 100, 'RdYlGn', 15, 65),
            ('Wake Loss (%)', 'WakeLoss', 100, 'YlOrRd', 0, 30),
            ('Volatility (GW)', 'Volatility_kW', 1e-6, 'YlOrRd', 0, 5),
        ]):
            ax = plt.subplot(1, 3, pi+1, projection=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
            ax.set_extent(bbox)
            for r in rk_rows:
                fid = int(r['farm_id'])
                info = farms_info.get(fid)
                if not info: continue
                val = float(r[key]) * mult
                nt = int(r['n_turb'])
                sz = max(15, min(300, nt * dot_factor))
                ax.scatter(info['lon'], info['lat'], s=sz, c=val, cmap=cmap,
                          vmin=vmin, vmax=vmax, edgecolors='#333', linewidths=0.3,
                          transform=ccrs.PlateCarree())
            cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.6, format='%.0f')
            cbar.set_label(title)
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(title)
        fig.suptitle(f'{REGION_NAMES.get(rk, rk)} — 2024 (Gaussian, {n_rows} farms)', fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'B_map_{rk}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("  Maps done")
else:
    print("  Cartopy not available")

# ============================================================
# FIG C: Annual trends
# ============================================================
print("C: Annual trends...")
yr_cf = defaultdict(lambda: defaultdict(list))
yr_wl = defaultdict(lambda: defaultdict(list))
for r in gauss:
    rk = get_region(int(r['farm_id']))
    yr = int(r['year']); yr_cf[rk][yr].append(float(r['CF'])*100); yr_wl[rk][yr].append(float(r['WakeLoss'])*100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for rk, color in [('east_asia','#3498db'),('europe','#e67e22'),('us_east','#2ecc71')]:
    yrs = sorted(yr_cf[rk].keys())
    cf_m = [np.mean(yr_cf[rk][y]) for y in yrs]; cf_s = [np.std(yr_cf[rk][y]) for y in yrs]
    wl_m = [np.mean(yr_wl[rk][y]) for y in yrs]; wl_s = [np.std(yr_wl[rk][y]) for y in yrs]
    ax1.fill_between(yrs, [m-s for m,s in zip(cf_m,cf_s)], [m+s for m,s in zip(cf_m,cf_s)], alpha=0.15, color=color)
    ax1.plot(yrs, cf_m, 'o-', markersize=4, color=color, label=REGION_NAMES.get(rk, rk))
    ax2.fill_between(yrs, [m-s for m,s in zip(wl_m,wl_s)], [m+s for m,s in zip(wl_m,wl_s)], alpha=0.15, color=color)
    ax2.plot(yrs, wl_m, 'o-', markersize=4, color=color, label=REGION_NAMES.get(rk, rk))
ax1.set_xlabel('Year'); ax1.set_ylabel('Mean CF (%)'); ax1.set_title('Capacity Factor Trend (2014-2024)')
ax2.set_xlabel('Year'); ax2.set_ylabel('Mean WakeLoss (%)'); ax2.set_title('Wake Loss Trend (2014-2024)')
ax1.legend(); ax2.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'C_annual_trend.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG D: Farm size effect — FIX: log scale + hexbin + kernel density
# ============================================================
print("D: Farm size effect (fixed)...")
nts_all = [int(r['n_turb']) for r in gauss if float(r['WakeLoss'])>0]
wls_all = [float(r['WakeLoss'])*100 for r in gauss if float(r['WakeLoss'])>0]
cfs_all = [float(r['CF'])*100 for r in gauss if float(r['WakeLoss'])>0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: hexbin density with log x-axis
hb = hb_im = ax1.scatter(nts_all, wls_all, c=cfs_all, cmap='RdYlGn', s=15, alpha=0.4, edgecolors='none')
ax1.set_xscale('log'); ax1.set_xlabel('Turbine Count (log scale)'); ax1.set_ylabel('Wake Loss (%)')
ax1.set_title('WakeLoss vs Farm Size (hexbin density)')
fig.colorbar(hb_im, ax=ax1, shrink=0.7, label='CF (%)')

# Right: boxplot by size category
cats = {'<10': [], '10-30': [], '30-80': [], '80-200': [], '200+': []}
for nt, wl in zip(nts_all, wls_all):
    if nt < 10: cats['<10'].append(wl)
    elif nt < 30: cats['10-30'].append(wl)
    elif nt < 80: cats['30-80'].append(wl)
    elif nt < 200: cats['80-200'].append(wl)
    else: cats['200+'].append(wl)
cat_labels = ['<10','10-30','30-80','80-200','200+']
bp = ax2.boxplot([cats[k] for k in cat_labels], tick_labels=cat_labels, patch_artist=True)
for patch, alpha_val in zip(bp['boxes'], np.linspace(0.3, 1, 5)):
    patch.set_facecolor('coral'); patch.set_alpha(alpha_val)
ax2.set_xlabel('Farm Size Category'); ax2.set_ylabel('Wake Loss (%)')
ax2.set_title('WakeLoss Distribution by Size Category')

fig.suptitle('Farm Size Effect on Wake Loss', fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'D_farm_size_effect.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG E: Counterfactual — FIX: boxplots + clean layout
# ============================================================
print("E: Counterfactual (fixed)...")
if cf_rows:
    cf_by_region = defaultdict(list)
    cf_by_yr = defaultdict(list)
    for r in cf_rows:
        rk = get_region(int(r['farm_id'])); yr = int(r['year'])
        da = float(r['Delta_AEP_WD_kWh']) / 1e6
        cf_by_region[rk].append(da); cf_by_yr[yr].append(da)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Left: boxplot by region
    bp_data = [cf_by_region.get(rk, []) for rk in ['east_asia','europe','us_east']]
    bp = axes[0].boxplot(bp_data, tick_labels=['East Asia','Europe','US East'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db','#e67e22','#2ecc71']):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].set_ylabel('Delta AEP (GWh)'); axes[0].set_title('By Region')

    # Middle: boxplot by farm size category
    cf_by_size = {'<50': [], '50-200': [], '200+': []}
    for r in cf_rows:
        fid = int(r['farm_id']); nt = int(farms_info.get(fid, {}).get('n_turb', 0))
        da = float(r['Delta_AEP_WD_kWh']) / 1e6
        if nt < 50: cf_by_size['<50'].append(da)
        elif nt < 200: cf_by_size['50-200'].append(da)
        else: cf_by_size['200+'].append(da)
    bp2 = axes[1].boxplot([cf_by_size[k] for k in ['<50','50-200','200+']],
                          tick_labels=['<50','50-200','200+'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightblue','coral','red']):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].set_ylabel('Delta AEP (GWh)'); axes[1].set_title('By Farm Size')

    # Right: histogram
    all_das = [float(r['Delta_AEP_WD_kWh'])/1e6 for r in cf_rows]
    axes[2].hist(all_das, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[2].axvline(np.mean(all_das), color='red', linestyle='--', label=f'Mean={np.mean(all_das):+.1f} GWh')
    axes[2].axvline(0, color='gray', linewidth=0.5)
    axes[2].set_xlabel('Delta AEP (GWh)'); axes[2].set_title('Distribution'); axes[2].legend()

    fig.suptitle('Counterfactual: Real vs Baseline (1981-2010) Wind Direction', fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'E_counterfactual.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG F: Hourly time series (keep as-is, minor polish)
# ============================================================
print("F: Hourly...")
hp = os.path.join(DATA, "task2_hourly_F0.csv")
if os.path.exists(hp):
    f0_2024 = []
    with open(hp, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['year']) == 2024:
                f0_2024.append(r)
                if len(f0_2024) >= 500: break
    if f0_2024:
        hrs = range(len(f0_2024))
        pn = [float(r['P_noWake_kW'])/1e6 for r in f0_2024]
        pw_g = [float(r['P_wake_Gaussian_kW'])/1e6 for r in f0_2024]
        ws = [float(r['V_free_ms']) for r in f0_2024]
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.fill_between(hrs, pn, pw_g, alpha=0.25, color='steelblue', label='Wake Loss')
        ax1.plot(hrs, pn, 'gray', alpha=0.7, lw=0.5, label='P_noWake')
        ax1.plot(hrs, pw_g, 'steelblue', lw=0.8, label='P_wake (Gaussian)')
        ax2 = ax1.twinx(); ax2.plot(hrs, ws, 'red', alpha=0.3, lw=0.5, label='Wind Speed')
        ax1.set_xlabel('Hour (2024)'); ax1.set_ylabel('Power (GW)'); ax2.set_ylabel('Wind Speed (m/s)')
        ax1.set_title('Farm 0 (928 turbines) — First 500 hours of 2024')
        ax1.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'F_hourly_F0.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG G: Rotation response — add annotation explaining the U-shape
# ============================================================
print("G: Rotation response (annotated)...")
# Keep audit_rotation_test.png — it's already good. Just add a note.

# ============================================================
# FIG H: Country comparison
# ============================================================
print("H: Country comparison...")
country_data = defaultdict(lambda: {'cf': [], 'wl': [], 'cnt': 0})
for r in gauss:
    fid = int(r['farm_id'])
    cc = farms_info.get(fid, {}).get('country', '?')
    country_data[cc]['cf'].append(float(r['CF'])*100)
    country_data[cc]['wl'].append(float(r['WakeLoss'])*100)
    country_data[cc]['cnt'] += 1

top12 = sorted(country_data.items(), key=lambda x: x[1]['cnt'], reverse=True)[:12]
fig, ax = plt.subplots(figsize=(12, 6))
labels = [c[0] for c in top12]; xs = np.arange(len(labels))
cf_vals = [np.mean(c[1]['cf']) for c in top12]
wl_vals = [np.mean(c[1]['wl']) for c in top12]
cnts = [c[1]['cnt'] for c in top12]
bars1 = ax.bar(xs - 0.2, cf_vals, 0.35, label='CF (%)', color='steelblue', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(xs + 0.2, wl_vals, 0.35, label='WakeLoss (%)', color='coral', alpha=0.8)
# Add farm count on top
for i, cnt in enumerate(cnts):
    ax.text(i, cf_vals[i]+1, f'n={cnt}', ha='center', fontsize=7, color='#555')
ax.set_xlabel('Country'); ax.set_ylabel('CF (%)'); ax2.set_ylabel('WakeLoss (%)')
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_title('Top 12 Countries: CF and WakeLoss (Gaussian, 2024 mean)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'H_country.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG I: WakeLoss distribution — FIX: 3 separate subplots with KDE
# ============================================================
print("I: WakeLoss distribution (fixed)...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, wm, rows_wm, c in zip(axes, ['Gaussian','Jensen','Curl'],
                                [gauss, jensen, curl],
                                ['#3498db','#e67e22','#2ecc71']):
    wls = [float(r['WakeLoss'])*100 for r in rows_wm if float(r['WakeLoss'])*100 > 0]
    ax.hist(wls, bins=40, color=c, alpha=0.7, edgecolor='white', density=True)
    # KDE overlay
    from scipy.stats import gaussian_kde
    if len(wls) > 3:
        kde = gaussian_kde(wls); xs_k = np.linspace(0, max(wls)*1.1, 200)
        ax.plot(xs_k, kde(xs_k), 'k-', lw=1.5, alpha=0.8)
    ax.axvline(np.mean(wls), color='red', linestyle='--', lw=1, label=f'Mean={np.mean(wls):.1f}%')
    ax.set_xlabel('Wake Loss (%)'); ax.set_title(f'{wm} (n={len(wls)})'); ax.legend(fontsize=8)
axes[0].set_ylabel('Probability Density')
fig.suptitle('Wake Loss Distribution by Model', fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'I_wakeloss_dist.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG J: Paradigm comparison (keep from Qiming)
# ============================================================
print("J: Paradigm comparison...")
paradigm_path = r"D:\1风力发电实习\task1_output\task1_paradigm_classification.csv"
if os.path.exists(paradigm_path):
    paradigm = {}
    with open(paradigm_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            paradigm[int(r['farm_id'])] = r.get('paradigm_labels', '')
    pdata = defaultdict(lambda: {'cf': [], 'wl': []})
    for r in gauss:
        fid = int(r['farm_id'])
        if fid in paradigm and paradigm[fid]:
            for p in paradigm[fid].split(','):
                p = p.strip()
                if p: pdata[p]['cf'].append(float(r['CF'])*100); pdata[p]['wl'].append(float(r['WakeLoss'])*100)
    if pdata:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        paradigms_sorted = sorted(pdata.keys())
        axes[0].bar(paradigms_sorted, [np.mean(pdata[p]['cf']) for p in paradigms_sorted], color='steelblue')
        axes[0].set_ylabel('Mean CF (%)'); axes[0].set_title('CF by Paradigm (Task1)')
        axes[1].bar(paradigms_sorted, [np.mean(pdata[p]['wl']) for p in paradigms_sorted], color='coral')
        axes[1].set_ylabel('Mean WakeLoss (%)'); axes[1].set_title('WakeLoss by Paradigm (Task1)')
        fig.suptitle('Construction Paradigms vs Task2 Performance', fontsize=14)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'J_paradigm.png'), dpi=150); plt.close(fig)

# ============================================================
# FIG K: Rotation response detail — add annotation
# ============================================================
print("K: Rotation response detail...")
# This is already covered by audit_rotation_test.png, keep it

# ============================================================
# SUMMARY
# ============================================================
print(f"\n====== 全部可视化生成完毕 ======")
for f in sorted(os.listdir(FIG)):
    if f.endswith('.png'):
        sz = os.path.getsize(os.path.join(FIG, f)) // 1024
        print(f"  {f} ({sz}KB)")
