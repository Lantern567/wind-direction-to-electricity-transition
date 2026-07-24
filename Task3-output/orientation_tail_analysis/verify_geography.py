"""Verify high-gain farm geography, wind concentration, and features"""
import csv, numpy as np
from collections import defaultdict

# Load data
fm = {}
for r in csv.DictReader(open(r'D:/1风力发电实习/offshore-task0/output/task0/farms_master.csv','r',encoding='utf-8-sig')):
    fm[int(r['farm_id'])] = dict(r)

og = defaultdict(list)
for r in csv.DictReader(open(r'D:/1风力发电实习/wind-direction-to-electricity-transition-main/分析材料_几何主导杠杆/data_derived/orientation_gain.csv','r',encoding='utf-8-sig')):
    og[int(r['farm_id'])].append(float(r['gain_pct']))

wci = {}
for r in csv.DictReader(open(r'D:/1风力发电实习/task1_output/task1_wind_metrics.csv','r',encoding='utf-8-sig')):
    try: wci[int(r['farm_id'])] = float(r.get('WCI_hist', 0))
    except: pass

ws_data = defaultdict(list)
for r in csv.DictReader(open(r'D:/1风力发电实习/offshore-task4/output/farm_ws_stats.csv','r',encoding='utf-8-sig')):
    ws_data[int(r['farm_id'])].append(float(r['ws_mean']))

high = sorted([(fid, sum(gains)/len(gains)) for fid, gains in og.items() if sum(gains)/len(gains) > 5], key=lambda x: -x[1])

geo_notes = {
    57: 'Mekong Delta — NE monsoon corridor, single-peak wind rose',
    155: 'Taranto Gulf, Italy — Adriatic channel wind (Bora/Mistral)',
    66: 'Hangzhou Bay mouth, Zhejiang — East Asian monsoon, bay funnel',
    157: 'Denmark Straits — Baltic-North Sea transition, channel wind',
    91: 'Pearl River Delta, Guangdong — South China Sea monsoon',
    126: 'Mekong Delta south, Vietnam — NE monsoon',
    159: 'Mekong Delta nearshore, Vietnam — NE monsoon + micro-farm amplification',
}

print("=== HIGH GAIN FARM GEOGRAPHIC VERIFICATION ===\n")
headings = ['FID','Gain','Country','Lat','Lon','n_t','ws','WCI','Geographic note']
print(f"{headings[0]:>5} {headings[1]:>7} {headings[2]:<20} {headings[3]:>6} {headings[4]:>7} {headings[5]:>5} {headings[6]:>5} {headings[7]:>6} {headings[8]}")
print('-' * 100)

for fid, gain in high:
    info = fm.get(fid, {})
    lat = float(info.get('centroid_lat', 0)); lon = float(info.get('centroid_lon', 0))
    n = int(info.get('n_turb', 0)); c = info.get('country', '?')
    w = wci.get(fid, -1); ws_m = np.mean(ws_data.get(fid, [0]))
    geo = geo_notes.get(fid, '?')
    print(f'{fid:>5} {gain:>+6.1f}% {c:<20} {lat:>5.1f} {lon:>6.1f} {n:>5} {ws_m:>4.1f} {w:>5.1f}   {geo}')

# Correlation: gain vs WCI across all farms
pairs = []
for fid, gains in og.items():
    g = sum(gains)/len(gains)
    if fid in wci and not np.isnan(wci[fid]):
        pairs.append((g, wci[fid]))

gs, ws_list = zip(*pairs)
r_val = np.corrcoef(gs, ws_list)[0,1]
print(f'\n=== OVERALL STATISTICS ===')
print(f'r(gain, WCI) = {r_val:.3f} (n={len(gs)} farms with valid WCI)')
print(f'Gain: mean={np.mean(gs):+.2f}%  median={np.median(gs):+.2f}%  P95={np.percentile(gs,95):+.1f}%')
print(f'>2%: {sum(1 for g in gs if g>2)} farms')
print(f'>5%: {sum(1 for g in gs if g>5)} farms')
print(f'>10%: {sum(1 for g in gs if g>10)} farms')

# Key insight: WCI threshold
print(f'\nWCI distribution:')
for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
    high_wci = [(g, w) for g, w in pairs if w > thresh]
    if high_wci:
        avg_g = np.mean([g for g,_ in high_wci])
        print(f'  WCI>{thresh}: {len(high_wci)} farms, mean gain={avg_g:+.2f}%')
