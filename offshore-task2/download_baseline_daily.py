"""
ERA5 1981-2010 基准期下载 — 每天12:00 UTC仅1时刻，30年合成1次请求
目标：3区域 × 1次 = 3次排队，总约330 MB
"""
import cdsapi, os

OUT = r"D:\1风力发电实习\offshore-task2\data"
os.makedirs(OUT, exist_ok=True)

REGIONS = {
    "east_asia": [44, 104, 8, 143],  # 扩展北界到44N 含日本北海道
    "europe":    [63, -10, 39, 22],
    "us_east":   [42, -77, 36, -69],
}

client = cdsapi.Client(quiet=True)

# 30年拆3批, 每批10年 — 共9次排队
BATCHES = [
    ("b1981_1990", [str(y) for y in range(1981, 1991)]),
    ("b1991_2000", [str(y) for y in range(1991, 2001)]),
    ("b2001_2010", [str(y) for y in range(2001, 2011)]),
]

for rkey, area in REGIONS.items():
    for bname, years in BATCHES:
        out_file = os.path.join(OUT, f"era5_baseline_daily_{rkey}_{bname}.nc")
        if os.path.exists(out_file):
            sz = os.path.getsize(out_file) / 1e6
            print(f"{rkey} {bname}: 已存在 {sz:.0f} MB, 跳过")
            continue

        print(f"{rkey} {bname}: 请求 {len(years)}年 (每天12:00, 排队中)...", flush=True)
        result = client.retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"],
            "variable": ["100m_u_component_of_wind", "100m_v_component_of_wind"],
            "year": years,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["12:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        })
        result.download(out_file)
        sz = os.path.getsize(out_file) / 1e6
        print(f"  OK {sz:.0f} MB", flush=True)

print("全部完成!")
