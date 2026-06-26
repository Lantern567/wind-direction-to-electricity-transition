"""补下载日本北海道区域 ERA5 — 填补 F145 的缺口 (lat 41°-44°N, lon 139°-143°E)"""
import cdsapi, os

OUT = r"D:\1风力发电实习\offshore-task1\data"
os.makedirs(OUT, exist_ok=True)

client = cdsapi.Client(quiet=True)
for yr in range(2014, 2025):
    out = os.path.join(OUT, f"era5_japan_{yr}.nc")
    if os.path.exists(out):
        print(f"{yr}: 已存在，跳过")
        continue
    print(f"{yr}: 请求...")
    result = client.retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"],
        "variable": ["100m_u_component_of_wind", "100m_v_component_of_wind"],
        "year": [str(yr)],
        "month": [f"{m:02d}" for m in range(1,13)],
        "day": [f"{d:02d}" for d in range(1,32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [44, 139, 41, 143],  # 北纬44, 西经139, 南纬41, 东经143
    })
    result.download(out)
    sz = os.path.getsize(out)/1e6
    print(f"  OK {sz:.0f} MB")

print("完成")
