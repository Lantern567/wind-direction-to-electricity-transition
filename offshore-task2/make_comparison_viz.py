"""新旧对比可视化: A)CF散点图 B)WL分布叠加"""
import csv, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OLD = r'D:\1风力发电实习\offshore-task2\output\task2_summary_v4.csv'
NEW = r'D:\1风力发电实习\offshore-task2\output\task2_annual_floris.csv'
OUT_DIR = r'D:\1风力发电实习\offshore-task2\output\figures_v2'

# Load
old_rows = list(csv.DictReader(open(OLD,'r',encoding='utf-8-sig')))
new_rows = list(csv.DictReader(open(NEW,'r',encoding='utf-8-sig')))

# Build common keys (farm_id, year, model)
by_key_old = {}
for r in old_rows:
    wm = r.get('wake_model','')
    if wm == 'gaussian': wm = 'gauss'
    elif wm == 'curl': wm = 'cc'
    by_key_old[(int(r['farm_id']), int(r['year']), wm)] = r

by_key_new = {}
for r in new_rows:
    by_key_new[(int(r['farm_id']), int(r['year']), r['wake_model'])] = r

common = set(by_key_old.keys()) & set(by_key_new.keys())
print(f'Common pairs: {len(common)}')

# ---- FIG A: CF scatter (gauss only) ----
keys_gauss = [(f,y,w) for f,y,w in common if w=='gauss']
cf_old = [float(by_key_old[k]['CF'])*100 for k in keys_gauss]
cf_new = [float(by_key_new[k]['CF'])*100 for k in keys_gauss]
wl_old_g = [float(by_key_old[k]['WakeLoss'])*100 for k in keys_gauss]
wl_new_g = [float(by_key_new[k]['WakeLoss'])*100 for k in keys_gauss]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# A-left: CF scatter
ax = axes[0]
ax.scatter(cf_old, cf_new, s=8, c='#2a78d6', alpha=0.35, edgecolors='none')
mn = min(min(cf_old), min(cf_new)); mx = max(max(cf_old), max(cf_new))
ax.plot([mn, mx], [mn, mx], '--', color='#52514e', lw=0.8, label='1:1')
ax.fill_between([mn, mx], [mn, mx], [mn*0.9, mx*0.9], alpha=0.05, color='#e34948')
ax.set_xlabel('OLD Numba CF (%)'); ax.set_ylabel('NEW FLORIS CF (%)')
ax.set_title(f'CF: OLD vs NEW (Gauss, n={len(keys_gauss)})')
ax.legend(frameon=False, fontsize=8)
# Stats annotation
bias = np.mean([n-o for n,o in zip(cf_new, cf_old)])
ax.text(0.05, 0.95, f'Mean bias: {bias:+.1f}pp\nMost points below 1:1 line',
        transform=ax.transAxes, va='top', fontsize=8, color='#52514e')

# A-right: WL scatter
ax2 = axes[1]
ax2.scatter(wl_old_g, wl_new_g, s=8, c='#e34948', alpha=0.35, edgecolors='none')
mn2 = max(0, min(min(wl_old_g), min(wl_new_g)))
mx2 = max(max(wl_old_g), max(wl_new_g))
ax2.plot([mn2, mx2], [mn2, mx2], '--', color='#52514e', lw=0.8, label='1:1')
ax2.set_xlabel('OLD Numba WakeLoss (%)'); ax2.set_ylabel('NEW FLORIS WakeLoss (%)')
ax2.set_title(f'WakeLoss: OLD vs NEW (Gauss, n={len(keys_gauss)})')
ax2.legend(frameon=False, fontsize=8)
bias2 = np.mean([n-o for n,o in zip(wl_new_g, wl_old_g)])
ax2.text(0.05, 0.95, f'Mean bias: {bias2:+.1f}pp\nMost points above 1:1 line',
        transform=ax2.transAxes, va='top', fontsize=8, color='#52514e')

plt.tight_layout()
out_a = os.path.join(OUT_DIR, 'figA_cf_scatter.png')
plt.savefig(out_a, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'FigA saved: {os.path.getsize(out_a)/1024:.0f} KB')

# ---- FIG B: WL distribution overlay ----
fig2, ax3 = plt.subplots(figsize=(10, 5))

for wm, emoji, color in [('gauss', 'Gauss', '#2a78d6'), ('jensen', 'Jensen', '#1baf7a')]:
    keys = [(f,y,w) for f,y,w in common if w==wm]
    if not keys: continue
    wl_o = [float(by_key_old[k]['WakeLoss'])*100 for k in keys]
    wl_n = [float(by_key_new[k]['WakeLoss'])*100 for k in keys]

    ax3.hist(wl_o, bins=40, alpha=0.3, color=color, density=True,
             label=f'OLD {emoji} (mean={np.mean(wl_o):.1f}%)')
    ax3.hist(wl_n, bins=40, alpha=0.5, color=color, density=True, histtype='step', linewidth=2,
             label=f'NEW {emoji} (mean={np.mean(wl_n):.1f}%)')

ax3.set_xlabel('Wake Loss (%)'); ax3.set_ylabel('Density')
ax3.set_title('WakeLoss Distribution: OLD Numba vs NEW FLORIS')
ax3.legend(frameon=False, fontsize=9)

plt.tight_layout()
out_b = os.path.join(OUT_DIR, 'figB_wl_distribution.png')
plt.savefig(out_b, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'FigB saved: {os.path.getsize(out_b)/1024:.0f} KB')
print('Done!')
