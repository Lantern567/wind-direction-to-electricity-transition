"""
ERA5 批量下载 — 3 区域，美国按年分片
东亚/欧洲：已完成（跳过），美国：11 次排队（每年约 30 MB）
"""
import cdsapi, os, time, json, glob
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_log.json")

REGIONS = {
    "east_asia": [41, 104,   8, 142],
    "europe":    [63, -10,  39,  22],
    "us_east":   [42, -77,  36, -69],
}
YEARS = list(range(2014, 2025))
DATASET = "reanalysis-era5-single-levels"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": [], "failed": {}, "downloads": []}

def save_log(log):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

def progress(log):
    comp = len(log["completed"])
    dls = log["downloads"]
    if not dls:
        return f"{comp}"
    avg_m = sum(d.get("elapsed_sec", 0) for d in dls[-5:]) / len(dls[-5:]) / 60
    return f"{comp} | 均{avg_m:.0f}分/个"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    log = load_log()

    # 清理残留
    done_set = set(log["completed"])
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "era5_*.nc"))):
        jid = os.path.basename(f).replace("era5_", "").replace(".nc", "")
        if jid not in done_set:
            os.remove(f)
            print(f"[清理残留] {os.path.basename(f)}")
    for jid in list(done_set):
        if not os.path.exists(os.path.join(DATA_DIR, f"era5_{jid}.nc")):
            log["completed"].remove(jid)
            save_log(log)

    tf = os.path.join(DATA_DIR, "test_202401_us.nc")
    if os.path.exists(tf):
        os.remove(tf)

    total = len(REGIONS) * len(YEARS)
    print(f"{datetime.now():%H:%M:%S}  开始 | 已:{len(done_set)} 待:{total - len(done_set)} 共:{total}")
    print("-" * 50)

    for rkey, area in REGIONS.items():
        for yr in YEARS:
            jid = f"{rkey}_{yr}"
            out = os.path.join(DATA_DIR, f"era5_{jid}.nc")

            if jid in done_set and os.path.exists(out):
                continue

            if os.path.exists(out):
                os.remove(out)

            req = {
                "product_type": ["reanalysis"],
                "variable": ["100m_u_component_of_wind", "100m_v_component_of_wind"],
                "year": [str(yr)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area,
            }

            ok = False
            for attempt in range(1, 4):
                t0 = time.time()
                try:
                    print(f"{datetime.now():%H:%M:%S}  请求 {jid} ...", end=" ", flush=True)
                    cdsapi.Client(quiet=True).retrieve(DATASET, req).download(out)
                    e = time.time() - t0
                    sz = os.path.getsize(out) / 1e6
                    log["completed"].append(jid)
                    log["downloads"].append({"job_id": jid, "size_mb": round(sz,1), "elapsed_sec": round(e)})
                    save_log(log)
                    print(f"OK {sz:.0f}MB {e/60:.0f}分 | {progress(log)}")
                    ok = True
                    break
                except Exception as ex:
                    if os.path.exists(out):
                        os.remove(out)
                    if attempt < 3:
                        print(f"重试{attempt+1}...", end=" ", flush=True)
                        time.sleep(attempt * 30)
                    else:
                        log["failed"][jid] = str(ex)[:300]
                        save_log(log)
                        # 打印完整错误便于排查
                        print(f"\n  完整错误: {str(ex)[:500]}")

            if ok:
                time.sleep(3)

    log = load_log()
    c = len(log["completed"])
    f = len(log["failed"])
    print(f"\n{'='*50}")
    print(f"完成: {c}  失败: {f}")


if __name__ == "__main__":
    main()
