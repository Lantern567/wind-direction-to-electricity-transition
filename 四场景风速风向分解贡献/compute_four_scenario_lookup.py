"""
=============================================================================
 compute_four_scenario_lookup.py — 四情景风速风向贡献分解查找表
 依据: 廷显交付_四情景查找表指导书.md (琪明, 2026-08-03)

 功能:
   1. 加载 S3 反事实覆盖的 (farm_id, year) 配对
   2. 按 config_hash 去重（同排布复用）
   3. 多进程并行预计算 FLORIS Gauss 尾流效率矩阵
   4. 输出查找表 CSV + 配置映射 CSV

 用法:
   python compute_four_scenario_lookup.py              # 默认 4 进程
   python compute_four_scenario_lookup.py --workers 6  # 6 进程
   python compute_four_scenario_lookup.py --serial      # 单进程 (调试用)
   python compute_four_scenario_lookup.py --dry-run     # 仅显示待计算清单

 输出:
   补算/output/four_scenario_lookup_table.csv  — WS×WD 效率矩阵 (684行/配置)
   补算/output/four_scenario_config_map.csv    — farm↔config 映射
 =============================================================================
"""
import os, sys, csv, time, argparse, glob as globmod, tempfile, traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process
from collections import Counter
import numpy as np

# ---- Path setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = os.path.join(OUT_DIR, ".tmp_lookup")
TASK2_DIR = r"D:\1风力发电实习\offshore-task2"
sys.path.insert(0, TASK2_DIR)

# Import config — do this at module level so workers inherit via spawn
from floris_config import (
    load_task0_coordinates, load_farms_master,
    get_ti_for_farm, WS_BINS, WD_SECTORS,
    ALPHA_DEFAULT, DEFAULT_TURBINE,
)
from task2_floris import compute_config_hash

# ---- Constants ----
COUNTERFACTUAL_PATH = os.path.join(TASK2_DIR, "output", "task2_counterfactual.csv")
LOOKUP_CSV = os.path.join(OUT_DIR, "four_scenario_lookup_table.csv")
CONFIG_MAP_CSV = os.path.join(OUT_DIR, "four_scenario_config_map.csv")

WS_BINS_ARR = np.array(WS_BINS, dtype=np.float64)
WD_SECTORS_ARR = np.array(WD_SECTORS, dtype=np.float64)
WAKE_MODEL = "gauss"
ROWS_PER_CONFIG = len(WS_BINS) * len(WD_SECTORS)  # 19 × 36 = 684


# =========================================================================
# Worker function — runs in subprocess, writes to its own temp file
# =========================================================================

def _worker_init():
    """Limit BLAS threading in worker processes."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def _compute_one_config(ch, xs, ys, ti):
    """Compute wake efficiency matrix for one config.

    Called via Pool.starmap(). Each worker writes its own .csv.part file
    into TMP_DIR, avoiding any cross-process file contention.

    Returns: (config_hash, n_rows, elapsed_s, n_turbines, tmpfile_path|error_string)
    """
    n_turbines = len(xs)
    tmp_path = os.path.join(TMP_DIR, f"{ch}.part")

    try:
        # Each worker imports FLORIS independently (needed for spawn)
        _sys = __import__('sys')
        _sys.path.insert(0, TASK2_DIR)

        from task2_floris import precompute_wake_table
        from floris_config import (
            DEFAULT_TURBINE as _DT, ALPHA_DEFAULT as _AD,
            WS_BINS as _WB, WD_SECTORS as _WD,
        )
        import numpy as _np

        ws_b = _np.array(_WB, dtype=_np.float64)
        wd_b = _np.array(_WD, dtype=_np.float64)

        wake_eff, elapsed, _, p_no_wake_bin, _hash = precompute_wake_table(
            xs, ys,
            turbine_type=_DT, ti=ti, alpha=_AD,
            ws_bins=ws_b, wd_sectors=wd_b,
            wake_model_name='gauss',
            return_extra=True,
        )

        n_wd = len(_WD)
        n_ws = len(_WB)

        # Write directly to temp file — no shared state
        with open(tmp_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            for iwd in range(n_wd):
                wd_str = f"{_WD[iwd]:.0f}"
                for iws in range(n_ws):
                    w.writerow([
                        ch,
                        f"{_WB[iws]:.1f}",
                        wd_str,
                        f"{wake_eff[iwd, iws]:.8f}",
                        f"{p_no_wake_bin[iws]:.2f}",
                        str(n_turbines),
                    ])

        return (ch, n_wd * n_ws, elapsed, n_turbines, tmp_path)

    except Exception as e:
        # Write error info to tmp file so main process can see it
        err_msg = f"{e}\n{traceback.format_exc()}"
        with open(tmp_path + ".err", 'w', encoding='utf-8') as f:
            f.write(err_msg)
        return (ch, 0, 0, n_turbines, f"ERROR: {e}")


# =========================================================================
# Helpers
# =========================================================================

def load_counterfactual_pairs():
    pairs = set()
    with open(COUNTERFACTUAL_PATH, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            pairs.add((int(r['farm_id']), int(r['year'])))
    return sorted(pairs)


def load_completed_hashes():
    """Load hashes from the main lookup CSV AND any existing .part files."""
    completed = set()
    # From main CSV
    if os.path.exists(LOOKUP_CSV) and os.path.getsize(LOOKUP_CSV) > 0:
        with open(LOOKUP_CSV, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed.add(r['config_hash'])
    # From partial files (from previous interrupted run)
    if os.path.isdir(TMP_DIR):
        for part in globmod.glob(os.path.join(TMP_DIR, "*.part")):
            h = os.path.basename(part).replace(".part", "")
            # Quick check: count lines
            with open(part, 'r', encoding='utf-8-sig') as f:
                n = sum(1 for _ in f)
            if n == ROWS_PER_CONFIG:
                completed.add(h)
    return completed


def merge_parts_to_main(completed_before):
    """Merge all .part files into the main lookup CSV and clean up.

    Reads all part files + existing main CSV, deduplicates by config_hash,
    and writes a clean merged output.
    """
    # Collect all rows, keyed by config_hash
    all_hashes = {}  # config_hash -> list of row tuples

    # Read existing main CSV
    if os.path.exists(LOOKUP_CSV) and os.path.getsize(LOOKUP_CSV) > 0:
        with open(LOOKUP_CSV, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader:
                ch = r['config_hash']
                if ch not in all_hashes:
                    all_hashes[ch] = []
                all_hashes[ch].append((
                    r['config_hash'], r['ws_bin_m_s'], r['wd_sector_deg'],
                    r['wake_efficiency'], r['p_noWake_kW'], r['n_turbines'],
                ))

    # Read all .part files
    if os.path.isdir(TMP_DIR):
        for part in sorted(globmod.glob(os.path.join(TMP_DIR, "*.part"))):
            ch = os.path.basename(part).replace(".part", "")
            with open(part, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = [tuple(row) for row in reader]
            if ch in all_hashes:
                # Overwrite with newer (part file takes precedence)
                all_hashes[ch] = rows
            else:
                all_hashes[ch] = rows

    # Write merged output
    with open(LOOKUP_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['config_hash', 'ws_bin_m_s', 'wd_sector_deg',
                    'wake_efficiency', 'p_noWake_kW', 'n_turbines'])
        for ch in sorted(all_hashes.keys()):
            w.writerows(all_hashes[ch])

    # Clean up part files
    if os.path.isdir(TMP_DIR):
        for part in globmod.glob(os.path.join(TMP_DIR, "*")):
            os.unlink(part)

    return len(all_hashes)


def save_config_map(mappings):
    with open(CONFIG_MAP_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['farm_id', 'year', 'config_hash'])
        for farm_id, year, config_hash in sorted(mappings):
            w.writerow([farm_id, year, config_hash])


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='四情景查找表预计算 (FLORIS Gauss)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help=f'并行进程数 (默认: min(cpu-2, 6))')
    parser.add_argument('--serial', action='store_true',
                        help='单进程模式 (调试或避免多进程问题时使用)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示待计算清单, 不实际计算')
    args = parser.parse_args()

    n_workers = args.workers
    if n_workers is None:
        n_workers = min(max(1, cpu_count() - 2), 6)
    if args.serial:
        n_workers = 1

    os.makedirs(OUT_DIR, exist_ok=True)
    if n_workers > 1:
        os.makedirs(TMP_DIR, exist_ok=True)

    t_start = datetime.now()

    print("=" * 60, flush=True)
    print(" 四情景查找表预计算", flush=True)
    print(f" 启动: {t_start.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f" 尾流模型: {WAKE_MODEL}", flush=True)
    print(f" 分箱: {len(WS_BINS)} WS x {len(WD_SECTORS)} WD = {ROWS_PER_CONFIG} rows/config", flush=True)
    print(f" 模式: {'单进程' if n_workers == 1 else f'{n_workers} 进程并行'}", flush=True)
    print("=" * 60, flush=True)

    # ---- 1. Load data ----
    print("\n[1/4] 加载数据...", flush=True)
    pairs = load_counterfactual_pairs()
    coords = load_task0_coordinates()
    farms = load_farms_master()
    print(f"  反事实配对: {len(pairs)} 个 (farm_id, year)", flush=True)

    # ---- 2. Build config map ----
    print("\n[2/4] 构建配置映射 & 去重...", flush=True)
    configs = {}
    farm_config_map = []
    skipped_no_coords = 0
    skipped_lt2 = 0

    for fid, yr in pairs:
        turbs = coords.get(fid, {}).get(yr, [])
        if len(turbs) == 0:
            skipped_no_coords += 1
            continue
        if len(turbs) < 2:
            skipped_lt2 += 1
            continue

        farm_info = farms.get(fid, {})
        lat = farm_info.get('centroid_lat', 0)
        lon = farm_info.get('centroid_lon', 0)
        ti = get_ti_for_farm(lat, lon)
        xs = [t['x_m'] for t in turbs]
        ys = [t['y_m'] for t in turbs]

        ch = compute_config_hash(
            xs, ys, DEFAULT_TURBINE, ti, ALPHA_DEFAULT,
            WAKE_MODEL, WS_BINS_ARR, WD_SECTORS_ARR,
        )
        farm_config_map.append((fid, yr, ch))

        if ch not in configs:
            configs[ch] = {
                'xs': xs, 'ys': ys, 'ti': ti,
                'n_turbines': len(turbs), 'farms': [(fid, yr)],
            }
        else:
            configs[ch]['farms'].append((fid, yr))

    n_unique = len(configs)
    print(f"  有效配对: {len(farm_config_map)}", flush=True)
    print(f"  独特配置: {n_unique} (去重率: {(1-n_unique/max(1,len(farm_config_map)))*100:.1f}%)", flush=True)

    # ---- 3. Compute ----
    print(f"\n[3/4] 计算尾流效率矩阵...", flush=True)

    # Always merge first to get accurate completed count
    completed_before = load_completed_hashes()
    if n_workers > 1 and os.path.isdir(TMP_DIR):
        # Merge existing parts into main CSV for clean state
        print("  合并已有临时文件到主CSV...", flush=True)
        merge_parts_to_main(completed_before)
        completed_before = load_completed_hashes()

    pending = [(ch, cfg) for ch, cfg in configs.items() if ch not in completed_before]

    if not pending:
        print("  全部已完成!", flush=True)
    elif args.dry_run:
        print(f"  DRY RUN — 待计算: {len(pending)} 配置", flush=True)
        pending_sorted = sorted(pending, key=lambda x: x[1]['n_turbines'])
        for ch, cfg in pending_sorted:
            nt = cfg['n_turbines']
            est = max(1, nt * 0.3 + 0.5)
            nf = len(cfg['farms'])
            print(f"    {ch}: {nt:4d}机 {nf}场 (~{est:.0f}s)", flush=True)
    else:
        print(f"  已完成: {len(completed_before)}, 待计算: {len(pending)}", flush=True)

        # Size distribution of pending
        size_cnt = Counter()
        for ch, cfg in pending:
            nt = cfg['n_turbines']
            if nt <= 30:
                size_cnt['≤30'] += 1
            elif nt <= 60:
                size_cnt['31-60'] += 1
            elif nt <= 100:
                size_cnt['61-100'] += 1
            else:
                size_cnt['>100'] += 1
        size_str = " ".join(f"{k}:{v}" for k, v in sorted(size_cnt.items()))
        print(f"  机队分布: {size_str}", flush=True)

        # Build work items, sorted by size (small first for fast feedback)
        work_items = [
            (ch, cfg['xs'], cfg['ys'], cfg['ti'])
            for ch, cfg in pending
        ]
        work_items.sort(key=lambda x: len(x[1]))

        n_total = len(work_items)
        n_done = 0
        n_error = 0
        t_batch_start = time.perf_counter()

        if n_workers == 1:
            # ---- Serial mode ----
            _worker_init()
            for ch, xs, ys, ti in work_items:
                t0 = time.perf_counter()
                result = _compute_one_config(ch, xs, ys, ti)
                t1 = time.perf_counter()

                ch2, n_rows, elapsed, n_turb, tmp_or_err = result

                if isinstance(tmp_or_err, str) and tmp_or_err.startswith("ERROR"):
                    n_error += 1
                    print(f"  [{n_done+1}/{n_total}] {ch}: {tmp_or_err}", flush=True)
                    continue

                n_done += 1
                elapsed_total = t1 - t_batch_start
                rate = n_done / elapsed_total if elapsed_total > 0 else 0
                eta_s = (n_total - n_done) / rate if rate > 0 else 0

                print(f"  [{n_done}/{n_total}] {ch}: {n_turb:4d}机 "
                      f"{elapsed:.1f}s | "
                      f"速率 {rate*3600:.1f}/h | "
                      f"ETA {eta_s/60:.0f}min",
                      flush=True)
        else:
            # ---- Multiprocessing mode ----
            with Pool(processes=n_workers, initializer=_worker_init) as pool:
                for result in pool.starmap(_compute_one_config, work_items, chunksize=1):
                    ch2, n_rows, elapsed, n_turb, tmp_or_err = result

                    if isinstance(tmp_or_err, str) and tmp_or_err.startswith("ERROR"):
                        n_error += 1
                        print(f"  [{n_done+1}/{n_total}] {ch2}: {tmp_or_err}", flush=True)
                        n_done += 1
                        continue

                    n_done += 1
                    elapsed_total = time.perf_counter() - t_batch_start
                    rate = n_done / elapsed_total if elapsed_total > 0 else 0
                    eta_s = (n_total - n_done) / rate if rate > 0 else 0

                    print(f"  [{n_done}/{n_total}] {ch2}: {n_turb:4d}机 "
                          f"{elapsed:.1f}s | "
                          f"速率 {rate*3600:.1f}/h | "
                          f"ETA {eta_s/60:.0f}min",
                          flush=True)

        # Merge all parts into main CSV
        if n_workers > 1:
            print(f"\n  合并临时文件...", flush=True)
            merge_parts_to_main(completed_before)
        elif n_workers == 1:
            # Serial mode: merge temp parts too
            merge_parts_to_main(completed_before)

        if n_error:
            print(f"\n  !! {n_error} 配置计算失败", flush=True)

    # ---- 4. Save config map ----
    print(f"\n[4/4] 保存配置映射...", flush=True)
    save_config_map(farm_config_map)

    # ---- Summary ----
    t_end = datetime.now()
    elapsed_total = (t_end - t_start).total_seconds()
    final_hashes = load_completed_hashes()

    print(f"\n{'='*60}", flush=True)
    print(f" 完成! 总耗时 {elapsed_total/60:.1f}min", flush=True)
    n_complete = len(final_hashes)
    status = "OK" if n_complete == n_unique else "MISMATCH!"
    print(f" 独特配置: {n_complete}/{n_unique} ({status})", flush=True)
    n_rows = n_complete * ROWS_PER_CONFIG
    print(f" 查找表行数: {n_rows:,}", flush=True)
    print(f" 配置映射: {len(farm_config_map)} farm-year 对", flush=True)
    print(f" 输出文件:", flush=True)
    print(f"   {LOOKUP_CSV}", flush=True)
    print(f"   {CONFIG_MAP_CSV}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
