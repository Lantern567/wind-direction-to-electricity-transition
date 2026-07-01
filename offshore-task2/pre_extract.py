"""
预提取脚本 —— 将 ERA5 NC 文件转换为风场级逐时风速 Parquet
只读 NC 文件，不做任何尾流计算，输出轻量级表格

产出：
  output/farm_wind_east_asia.parquet
  output/farm_wind_europe.parquet
  output/farm_wind_us_east.parquet

每行: farm_id | year | month | day | hour | V_ms | theta_deg
V_ms    = 全场平均自由来流风速 (100m, 未修正高度)
theta   = 全场平均来流风向 (气象角, 0=北)

后续 task1_complete.py 直接读 Parquet，不再打开 NC 文件
"""
import os, sys, math, json, yaml, csv, tempfile, shutil
import numpy as np
import netCDF4
from netCDF4 import num2date
from collections import defaultdict
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

REGIONS = {
    "east_asia": [41, 104,  8, 142],
    "europe":    [63, -10, 39,  22],
    "us_east":   [42, -77, 36, -69],
}

TEST_MODE = False   # True=只提取东亚2024年7月，快速验证
ALPHA = 0.11
H_REF = 100.0

# ============================================================
class ERA5File:
    """单次打开一个 NC 文件（临时目录方案）"""
    def __init__(self, path): self.path = path
    def __enter__(self):
        self.tmp = tempfile.mkdtemp()
        self.tmpnc = os.path.join(self.tmp, "data.nc")
        shutil.copy2(self.path, self.tmpnc)
        self.ds = netCDF4.Dataset(self.tmpnc, 'r')
        return self
    def __exit__(self, *a):
        if hasattr(self, 'ds'): self.ds.close()
        if hasattr(self, 'tmp'): shutil.rmtree(self.tmp)
    def times(self):
        return num2date(self.ds['valid_time'][:], units=self.ds['valid_time'].units)
    def wind_at(self, lat, lon):
        la = self.ds['latitude'][:]; lo = self.ds['longitude'][:]
        ila = int(np.argmin(np.abs(la - lat)))
        ilo = int(np.argmin(np.abs(lo - lon)))
        u = np.array(self.ds['u100'][:, ila, ilo])
        v = np.array(self.ds['v100'][:, ila, ilo])
        return u, v

# ============================================================
def load_farms():
    """加载风场分组（基于 turbine_yearly.csv + DBSCAN）"""
    from sklearn.cluster import DBSCAN

    # 读风机
    turbines = []
    with open(os.path.join(DATA_DIR, "turbine_yearly.csv"), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            turbines.append({
                'id': int(row['id']),
                'lon': float(row['lon']),
                'lat': float(row['lat']),
                'active': {y: row.get(f'active_{y}') == 'True' for y in range(2014, 2025)},
            })

    # 按区域聚类
    all_farms = {}
    for rkey, bbox in REGIONS.items():
        lmn, lmx, lnn, lnx = bbox[2], bbox[0], bbox[1], bbox[3]
        turbs = [t for t in turbines
                 if t['active'].get(2024, False)
                 and lmn <= t['lat'] <= lmx and lnn <= t['lon'] <= lnx]

        if len(turbs) < 3:
            all_farms[rkey] = {}
            continue

        # DBSCAN
        coords = np.array([[t['lon'], t['lat']] for t in turbs])
        mlat = np.mean(coords[:, 1])
        kx = 111.32 * math.cos(math.radians(mlat))
        ky = 111.32
        ckm = coords.copy()
        ckm[:, 0] *= kx
        ckm[:, 1] *= ky

        labels = DBSCAN(eps=5.0, min_samples=3, metric='euclidean').fit_predict(ckm)

        # 分组
        farms = defaultdict(list)
        for i, t in enumerate(turbs):
            if labels[i] >= 0:
                farms[int(labels[i])].append(t)

        all_farms[rkey] = dict(farms)
        print(f"  {rkey}: {len(farms)} 个风场, {sum(len(v) for v in farms.values())} 台风机")

    return all_farms

# ============================================================
def extract_region(rkey, farms, era5_paths, out_dir):
    """提取一个区域所有风场的逐时风速，写入 Parquet"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = os.path.join(out_dir, f"farm_wind_{rkey}.parquet")
    if os.path.exists(out_path):
        os.remove(out_path)

    writer = None
    batch_rows = []
    batch_size = 500000  # 每 50 万行写一次

    total_rows = 0

    for yr in range(2014, 2025):
        key = (rkey, yr)
        if key not in era5_paths:
            print(f"  {yr}: 无数据，跳过")
            continue

        print(f"  {yr}...", end=" ", flush=True)
        t0 = datetime.now()

        with ERA5File(era5_paths[key]) as era5:
            times = era5.times()

            if TEST_MODE:
                # 只取 7 月
                idx = [i for i, t in enumerate(times) if t.month == 7]
            else:
                idx = list(range(len(times)))

            n_steps = len(idx)
            yr_rows = 0

            for fid, ft in farms.items():
                nt = len(ft)

                # 只在风机当年活跃时才输出
                active_this_year = [t for t in ft if t['active'].get(yr, False)]
                if not active_this_year:
                    continue

                # 提取每台风机逐时风速
                tws = np.zeros((nt, n_steps))
                twd = np.zeros((nt, n_steps))

                for j, t in enumerate(ft):
                    u, v = era5.wind_at(t['lat'], t['lon'])
                    us, vs = u[idx], v[idx]
                    tws[j] = np.sqrt(us * us + vs * vs)
                    twd[j] = (np.degrees(np.arctan2(us, vs)) + 180) % 360

                # 全场平均
                for s, ti in enumerate(idx):
                    dt = times[ti]
                    Vf = float(np.mean(tws[:, s]))
                    th = float(np.mean(twd[:, s]))
                    batch_rows.append({
                        'farm_id': fid, 'year': dt.year, 'month': dt.month,
                        'day': dt.day, 'hour': dt.hour,
                        'V_ms': round(Vf, 2), 'theta_deg': round(th, 1),
                    })
                    yr_rows += 1

                if len(batch_rows) >= batch_size:
                    table = pa.Table.from_pylist(batch_rows)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                    batch_rows.clear()

            total_rows += yr_rows
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"{yr_rows//1000}k行 {elapsed:.0f}s", flush=True)

    # 最后一批
    if batch_rows:
        table = pa.Table.from_pylist(batch_rows)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()

    sz = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  → {out_path} ({sz:.0f} MB, {total_rows//1000}k 行)")
    return out_path

# ============================================================
def main():
    t0 = datetime.now()
    print("=" * 60)
    mode = "测试" if TEST_MODE else "全量"
    print(f"ERA5 → Parquet 预提取 ({mode}模式)")
    print("=" * 60)

    # [1] 加载风场
    print("\n[1] 风场聚类...")
    all_farms = load_farms()

    # [2] 扫描 NC 文件
    print("\n[2] 扫描 NC 文件...")
    era5p = {}
    for rk in REGIONS:
        for yr in range(2014, 2025):
            p = os.path.join(DATA_DIR, f"era5_{rk}_{yr}.nc")
            if os.path.exists(p):
                era5p[(rk, yr)] = p
    print(f"  {len(era5p)} 个文件")

    # [3] 逐区域提取
    for rk in REGIONS:
        farms = all_farms.get(rk, {})
        if not farms:
            continue
        print(f"\n[3] 提取 {rk} ({len(farms)} 风场)...")
        extract_region(rk, farms, era5p, OUT_DIR)

    print(f"\n总耗时: {(datetime.now()-t0).total_seconds()/60:.1f} 分钟")
    print("完成！")


if __name__ == "__main__":
    main()
