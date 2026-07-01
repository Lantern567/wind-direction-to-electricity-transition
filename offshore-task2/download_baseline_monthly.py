"""
下载 ERA5 月度平均 100m u/v 风分量 (1981-2010)
用于任务二 §2.8 反事实分析的基准期风向分布
一次请求覆盖全30年, 几MB极快
"""
import cdsapi, os

OUT = r"D:\1风力发电实习\offshore-task2\data"
os.makedirs(OUT, exist_ok=True)

REGIONS = {
    "east_asia": [41, 104, 8, 142],
    "europe":    [63, -10, 39, 22],
    "us_east":   [42, -77, 36, -69],
    "japan":     [44, 139, 41, 143],
}

client = cdsapi.Client(quiet=True)

for rkey, area in REGIONS.items():
    out_file = os.path.join(OUT, f"era5_monthly_{rkey}_1981_2010.nc")
    if os.path.exists(out_file):
        print(f"{rkey}: 已存在, 跳过")
        continue

    print(f"{rkey}: 下载 1981-2010 月度均值...")
    result = client.retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["100m_u_component_of_wind", "100m_v_component_of_wind"],
        "year": [str(y) for y in range(1981, 2011)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    })
    result.download(out_file)
    sz = os.path.getsize(out_file) / 1e6
    print(f"  OK {sz:.1f} MB")

print("完成!")
