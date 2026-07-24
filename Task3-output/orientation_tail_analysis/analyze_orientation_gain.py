"""Analyze orientation gain farms — geographic patterns"""
import csv, numpy as np
from collections import Counter, defaultdict

fm = {}
for r in csv.DictReader(open(r'D:/1风力发电实习/offshore-task0/output/task0/farms_master.csv','r',encoding='utf-8-sig')):
    fm[int(r['farm_id'])] = dict(r)

og = defaultdict(list)
og_path = r'D:/1风力发电实习/wind-direction-to-electricity-transition-main/分析材料_几何主导杠杆/data_derived/orientation_gain.csv'
for r in csv.DictReader(open(og_path,'r',encoding='utf-8-sig')):
    og[int(r['farm_id'])].append(float(r['gain_pct']))

farm_gain = {fid: sum(gains)/len(gains) for fid, gains in og.items()}
top = sorted(farm_gain.items(), key=lambda x: -x[1])

print("Top 20 farms by mean orientation gain:")
print('FID    Gain    Country              Lat   Lon    n_turb area_km2')
for fid, gain in top[:20]:
    info = fm.get(fid, {})
    print(f'{fid:>4} {gain:>+6.1f}% {info.get("country","?"):<20} {float(info.get("centroid_lat",0)):>5.1f} {float(info.get("centroid_lon",0)):>6.1f} {info.get("n_turb","?"):>6} {info.get("area_km2","?"):>8}')

print('\n=== High gain (>5%) farms by country ===')
high = [(fid, gain) for fid, gain in farm_gain.items() if gain > 5]
countries = Counter(fm.get(fid,{}).get('country','?') for fid,_ in high)
for c, n in countries.most_common():
    fids = [(fid, gain) for fid, gain in high if fm.get(fid,{}).get('country')==c]
    print(f'  {c}: {n} — {[(fid, round(gain)) for fid,gain in fids]}')

print('\n=== Geographic regions ===')
for fid, gain in sorted(high, key=lambda x: -x[1]):
    info = fm.get(fid, {})
    lat = float(info.get('centroid_lat',0)); lon = float(info.get('centroid_lon',0))
    region = 'East Asia' if 8<=lat<=44 and 104<=lon<=143 else ('Europe' if 39<=lat<=63 and -12<=lon<=32 else 'Other')
    print(f'  F{fid}: {gain:+.1f}% {info.get("country","?")} ({lat:.1f},{lon:.1f}) [{region}]')

print('\n=== Gain distribution ===')
all_gains = [v for v in farm_gain.values()]
print(f'Mean={np.mean(all_gains):+.2f}% Median={np.median(all_gains):+.2f}%')
print(f'P75={np.percentile(all_gains,75):+.1f}% P90={np.percentile(all_gains,90):+.1f}% P95={np.percentile(all_gains,95):+.1f}%')
print(f'>2%: {sum(1 for g in all_gains if g>2)} farms')
print(f'>5%: {sum(1 for g in all_gains if g>5)} farms')
print(f'>10%: {sum(1 for g in all_gains if g>10)} farms')

# Read wind stats to join
ws_path = r'D:/1风力发电实习/offshore-task4/output/farm_ws_stats.csv'
ws_mean = {}
for r in csv.DictReader(open(ws_path,'r',encoding='utf-8-sig')):
    fid = int(r['farm_id']); yr = int(r['year'])
    if fid not in ws_mean: ws_mean[fid] = []
    ws_mean[fid].append(float(r['ws_mean']))

print('\n=== High gain farms with wind speed ===')
for fid, gain in high:
    info = fm.get(fid, {})
    ws = np.mean(ws_mean.get(fid, [0])) if fid in ws_mean else 0
    n = int(info.get('n_turb', 0))
    area = float(info.get('area_km2', 0))
    density = n/area if area > 0 else 0
    print(f'  F{fid} ({info.get("country","?")}): gain={gain:+.1f}% ws={ws:.1f}m/s n={n} density={density:.2f}/km2')
