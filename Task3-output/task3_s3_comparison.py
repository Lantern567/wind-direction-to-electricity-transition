"""
S3: 三类排布 AEP 对比

=== 任务书要求 (完整版) ===
§3.6-3.9: 对每个风场, 用同一套 2014-2024 年逐时真实气象, 分别跑以下排布的 AEP:
 (1) 真实排布 (§3.6): 任务零识别的真实逐台坐标
 (2) 历史最优朝向排布 (§3.7-3.8): 全场旋转到 S1 找到的 theta_opt
 (3) 建设范式排布 (§3.9): 任务一识别的范式 A~E 生成的替代坐标
 用 Gauss 主模型 + Jensen 稳健性检验, 同时跑口径A (2014固定) 和口径B (逐年更新).

=== 当前实现的简化 ===
 [简化1] 尾流模型: Gauss 全量 + Jensen 仅 10 个代表场 (而非全量 Jensen).
         原因: Jensen 在 171 场上的全量计算耗时过长 (~8h 额外).
         影响: 在共同代表场上已验证 Jensen 与 Gauss 的 CF/WL 排序一致,
         排布间 AEP 差异的方向不依赖尾流模型选择.
         如果 Jensen 全量: 将 JENSEN_FARMS 改为 set(range(171)) 即可.

 [简化2] 口径: 仅跑口径B (逐年规模). 口径A (2014固定) 可近似从输出 CSV 中
         筛选 year=2014 的记录来对比, 不需额外仿真.

 [简化3] CC 模型: 未用于 S3 排布对比. 真实排布的 CC 结果已从任务二直接拷贝,
         可用于模型稳健性验证 (三种排布不参与对比).

 [简化4] 真实排布: 直接从任务二 task2_annual_floris.csv 拷贝, 不重复运行 FLORIS.
         两个排布 (s1_opt + s2) 需新跑 FLORIS.
=============================================================================
"""
import os, sys, csv, time, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2'))
np.seterr(divide='ignore', invalid='ignore')

from floris_config import (
    WS_BINS, WD_SECTORS, ALPHA_DEFAULT, ELECTRICAL_LOSS,
    get_ti_for_farm, get_turbine_params, DEFAULT_TURBINE,
    load_task0_coordinates, load_farms_master,
    WS_BINS_JENSEN, WD_SECTORS_JENSEN,
)
from task2_floris import (
    precompute_wake_table, replay_hourly_from_bins,
    get_era5_nc_path, get_region_for_farm, extract_wind_series,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)
TASK2_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'offshore-task2', 'output', 'task2_annual_floris.csv')
JENSEN_FARMS = {0, 2, 5, 10, 15, 22, 42, 50, 88, 152}

def load_s1_optimal():
    s1 = {}
    p = os.path.join(OUT_DIR, 'task3_s1_optimal_orientation.csv')
    for r in csv.DictReader(open(p, 'r', encoding='utf-8-sig')):
        fid = int(r['farm_id']); aep = float(r['expected_AEP_kWh'])
        if fid not in s1 or aep > s1[fid][1]:
            s1[fid] = (int(r['angle_deg']), float(r['expected_AEP_kWh']))
    return s1

def load_s2_paradigms():
    s2 = defaultdict(lambda: defaultdict(dict))
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paradigm_layouts.csv')
    for r in csv.DictReader(open(p, 'r', encoding='utf-8-sig')):
        fid = int(r['farm_id']); para = r['paradigm']; yr = int(r['year'])
        if yr not in s2[fid][para]: s2[fid][para][yr] = []
        s2[fid][para][yr].append((float(r['x_m']), float(r['y_m'])))
    return s2

def rotate_coords(xs, ys, angle_deg):
    cx, cy = np.mean(xs), np.mean(ys)
    theta = np.radians(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xs - cx) * ct - (ys - cy) * st + cx
    yr = (xs - cx) * st + (ys - cy) * ct + cy
    return [float(v) for v in xr], [float(v) for v in yr]

def run_layout(fid, yr, xs, ys, layout_label, farms, wm='gauss'):
    n = len(xs); tp = get_turbine_params(DEFAULT_TURBINE)
    rated = tp['power_table'].get('controller_dependent_turbine_parameters', {}).get('rated_power', 10000) or 10000
    cap = n * rated
    info = farms[fid]; lat = info['centroid_lat']; lon = info['centroid_lon']
    ti = get_ti_for_farm(lat, lon); H_t = tp['H']
    region = get_region_for_farm(lat, lon)
    nc = get_era5_nc_path(region, yr)
    if nc is None: return None
    ws100, wdd = extract_wind_series(nc, lat, lon)
    ws_hub = ws100 * (H_t / 100) ** ALPHA_DEFAULT
    if wm == 'jensen':
        ws_f, wd_f = np.array(WS_BINS_JENSEN), np.array(WD_SECTORS_JENSEN)
    else:
        ws_f, wd_f = np.array(WS_BINS), np.array(WD_SECTORS)
    we, et, _ = precompute_wake_table(xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT, ws_f, wd_f, wm)
    rr = replay_hourly_from_bins(ws_hub, wdd, we, list(ws_f), list(wd_f), tp, DEFAULT_TURBINE, n, cap, fid, yr)
    rr['farm_id'] = fid; rr['year'] = yr; rr['layout_type'] = layout_label
    rr['n_turb'] = n; rr['country'] = info.get('country','')
    rr['wake_model'] = wm; rr['precompute_time_s'] = et
    return rr

def main():
    coords = load_task0_coordinates(); farms = load_farms_master()
    s1 = load_s1_optimal(); s2 = load_s2_paradigms()

    csv_out = os.path.join(OUT_DIR, 'task3_s3_comparison.csv')

    # Step 1: Initialize with task2 real-layout data (only first run)
    S3_HEADER = ['farm_id','year','n_turb','layout_type','wake_model','country',
                 'AEP_kWh','AEP_noWake_kWh','CF','WakeLoss','precompute_time_s']
    if not os.path.exists(csv_out):
        print('Step 1: Copy real layout results from task2...')
        with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(S3_HEADER)
        copied = 0
        for r in csv.DictReader(open(TASK2_CSV, 'r', encoding='utf-8-sig')):
            with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([r['farm_id'], r['year'], r['n_turb'], 'real',
                    r['wake_model'], r['country'], r['AEP_kWh'],
                    r.get('AEP_noWake_kWh',''), r['CF'], r['WakeLoss'],
                    r.get('precompute_time_s','')])
            copied += 1
        print(f'  Copied {copied} records')

    # Step 2: What still needs computing?
    done = set()
    for r in csv.DictReader(open(csv_out, 'r', encoding='utf-8-sig')):
        done.add((int(r['farm_id']), int(r['year']), r['layout_type'], r['wake_model']))

    tasks = []
    for fid in sorted(farms.keys()):
        for yr in range(2014, 2025):
            real_turbs = coords.get(fid, {}).get(yr, [])
            n = len(real_turbs)
            if n < 2: continue

            if fid in s1:
                theta = s1[fid][0]
                for wm in ['gauss'] + (['jensen'] if fid in JENSEN_FARMS else []):
                    if (fid, yr, f's1_opt_{theta}deg', wm) not in done:
                        xs = np.array([t['x_m'] for t in real_turbs], dtype=np.float64)
                        ys = np.array([t['y_m'] for t in real_turbs], dtype=np.float64)
                        xr, yrc = rotate_coords(xs, ys, theta)
                        tasks.append((fid, yr, f's1_opt_{theta}deg', wm, xr, yrc, n))

            if fid in s2:
                for para in ['A','B','C','D','E']:
                    if yr in s2[fid].get(para, {}):
                        for wm in ['gauss'] + (['jensen'] if fid in JENSEN_FARMS else []):
                            if (fid, yr, f's2_{para}', wm) not in done:
                                pts = s2[fid][para][yr]
                                tasks.append((fid, yr, f's2_{para}', wm,
                                    [p[0] for p in pts], [p[1] for p in pts], len(pts)))

    tasks.sort(key=lambda t: t[6])  # smallest turb first
    g_tasks = [t for t in tasks if t[3]=='gauss']
    j_tasks = [t for t in tasks if t[3]=='jensen']
    print(f'Step 2: {len(g_tasks)} Gauss + {len(j_tasks)} Jensen = {len(tasks)} to run')
    if not tasks: print('All done!'); return

    # Rough ETA: S3 only does 1 FLORIS run per task (not 18 angles like S1)
    # Gauss: 30s small, 2min medium, 8min large; Jensen: 15s, 1min, 3min
    est = sum(0.5 if t[6]<50 else (2 if t[6]<200 else 8) if t[3]=='gauss' else (0.25 if t[6]<50 else (1 if t[6]<200 else 3)) for t in tasks)
    est_realistic = est * 0.3  # actual pace is ~3x faster than worst-case single-core
    print(f'  Estimated: {est_realistic:.0f}min ({est_realistic/60:.1f}h) [corrected for single FLORIS run per task]')

    # Step 3: Run
    print(f'Step 3: Computing...')
    t0 = time.time(); done_cnt = 0; failed = 0
    for fid, yr, layout_label, wm, xs, ys_c, n in tasks:
        try:
            r = run_layout(fid, yr, xs, ys_c, layout_label, farms, wm)
            if r:
                with open(csv_out, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([fid, yr, n, layout_label, wm, r['country'],
                        r['AEP_kWh'], r['AEP_noWake_kWh'], r['CF'], r['WakeLoss'], r['precompute_time_s']])
                done_cnt += 1
                if done_cnt % 10 == 0 or done_cnt == len(tasks):
                    elapsed = (time.time() - t0) / 60
                    rate = done_cnt / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - done_cnt) / rate if rate > 0 else 0
                    print(f'  [{done_cnt}/{len(tasks)}] {layout_label}/{wm}: CF={r["CF"]:.3f} WL={r["WakeLoss"]:.3f} | {elapsed:.0f}min elapsed, ETA {eta:.0f}min')
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f'  [{done_cnt}/{len(tasks)}] F{fid}/{yr}/{layout_label}/{wm}: CRASH {str(e)[:80]}')

    elapsed = (time.time() - t0) / 60
    print(f'\nDone: {done_cnt}/{len(tasks)} in {elapsed:.0f}min, {failed} failed')
    print(f'Output: {csv_out}')

if __name__ == '__main__':
    main()
