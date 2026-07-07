"""修复图5: 加海岸线 + 正确经纬度范围"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

cf = r'D:\1风力发电实习\offshore-task2\output\task2_counterfactual.csv'
fm_path = r'D:\1风力发电实习\offshore-task0\output\task0\farms_master.csv'

farms = {}
for r in csv.DictReader(open(fm_path,'r',encoding='utf-8-sig')):
    farms[int(r['farm_id'])] = (float(r['centroid_lon']), float(r['centroid_lat']), int(r.get('capacity_kW',0)))

lons, lats, deltas, caps = [], [], [], []
for r in csv.DictReader(open(cf,'r',encoding='utf-8-sig')):
    fid = int(r['farm_id'])
    if fid in farms:
        lon, lat, cap = farms[fid]
        d = float(r.get('Delta_AEP_WD_kWh', 0)) / 1e9  # GWh
        lons.append(lon); lats.append(lat)
        deltas.append(d); caps.append(max(20, cap/100000))

fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#e8f4f8', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888888', zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#cccccc', zorder=1)

pos_mask = [d > 0 for d in deltas]
neg_mask = [d < 0 for d in deltas]

ax.scatter([lons[i] for i in range(len(lons)) if neg_mask[i]],
           [lats[i] for i in range(len(lats)) if neg_mask[i]],
           s=[caps[i]*0.5 for i in range(len(caps)) if neg_mask[i]],
           c='#e34948', alpha=0.6, edgecolors='none', transform=ccrs.PlateCarree(),
           label=f'Real > Baseline ({sum(neg_mask)})')

ax.scatter([lons[i] for i in range(len(lons)) if pos_mask[i]],
           [lats[i] for i in range(len(lats)) if pos_mask[i]],
           s=[caps[i]*0.5 for i in range(len(caps)) if pos_mask[i]],
           c='#2a78d6', alpha=0.6, edgecolors='none', transform=ccrs.PlateCarree(),
           label=f'Baseline > Real ({sum(pos_mask)})')

ax.legend(frameon=False, loc='lower left', fontsize=10)
ax.set_title('dAEP_WD: Real Wind Direction vs 1981-2010 Baseline (Gauss model)', fontsize=13)

out = r'D:\1风力发电实习\offshore-task2\output\figures_v2\fig5_daep_map.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'fig5 regenerated with coastlines: {os.path.getsize(out)/1024:.0f} KB')
