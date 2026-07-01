"""
=============================================================================
 任务一 §1.8 风向变化反事实分析
 情景A: 真实风速 + 真实风向 → AEP_real, WakeLoss_real
 情景B: 真实风速 + 基准期(2014-2017)风向分布 → AEP_baseWD, WakeLoss_baseWD

 每区域取前2个代表风场，7年分析期(2018-2024)
 Numba JIT 加速全循环，边算边写 CSV
=============================================================================
"""
import os, math, time, csv, numpy as np
from collections import defaultdict
from datetime import datetime
from numba import njit

# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE = [2014, 2015, 2016, 2017]
ANALYSIS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
REGIONS = {"east_asia": [41,104,8,142], "europe": [63,-10,39,22], "us_east": [42,-77,36,-69]}
REG_NAMES = {"east_asia": "东亚", "europe": "欧洲", "us_east": "美国"}
H_REF=100.0; ALPHA=0.11; K=0.05; WAKE_LIMIT=90.0

N_FARMS_PER_REGION = 2  # 每区选前2个代表风场

# ============================================================
# 1. LOAD
# ============================================================
def load_models():
    import yaml
    m = {}
    for name in ["nrel_5MW","iea_10MW","iea_15MW"]:
        with open(os.path.join(DATA_DIR,f"{name}.yaml"),'r',encoding='utf-8') as f:
            c = yaml.safe_load(f)
        pt = c['power_thrust_table']
        m[name] = {
            'D':c['rotor_diameter'],'H':c['hub_height'],
            'Pr':pt['controller_dependent_turbine_parameters']['rated_power'],
            'ws':np.array(pt['wind_speed'],dtype=np.float64),
            'power':np.array(pt['power'],dtype=np.float64),
            'ct':np.array(pt['thrust_coefficient'],dtype=np.float64),
        }
    return m

def load_turbines():
    rows=[]
    with open(os.path.join(DATA_DIR,"turbine_yearly.csv"),'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append({'id':int(r['id']),'lon':float(r['lon']),'lat':float(r['lat']),
                         'first_year':int(r.get('first_year',2018)),
                         'active':{2024:r.get('active_2024')=='True'}})
    return rows

def load_wind(rk):
    import pyarrow.parquet as pq
    p=os.path.join(OUT_DIR,f"farm_wind_{rk}.parquet")
    return pq.read_table(p).to_pandas() if os.path.exists(p) else None

def assign_model(yr):
    if yr<=2017: return "nrel_5MW"
    elif yr<=2021: return "iea_10MW"
    return "iea_15MW"

# ============================================================
# 2. CLUSTER
# ============================================================
def cluster_farms(turbs):
    from sklearn.cluster import DBSCAN
    if len(turbs)<3: return {}
    xy=np.array([[t['lon'],t['lat']] for t in turbs])
    mlat=np.mean(xy[:,1]); kx=111.32*math.cos(math.radians(mlat)); ky=111.32
    xyk=xy.copy(); xyk[:,0]*=kx; xyk[:,1]*=ky
    labels=DBSCAN(eps=5.0,min_samples=3,metric='euclidean').fit_predict(xyk)
    farms=defaultdict(list)
    for i,t in enumerate(turbs):
        if labels[i]>=0: farms[int(labels[i])].append(t)
    return dict(farms)

def local_xy(turbs):
    rlat=np.mean([t['lat'] for t in turbs]); rlon=np.mean([t['lon'] for t in turbs])
    kx=111.32*math.cos(math.radians(rlat)); ky=111.32
    return np.array([[(t['lon']-rlon)*kx,(t['lat']-rlat)*ky] for t in turbs])

# ============================================================
# 3. NUMBA ENGINES
# ============================================================
@njit
def wake_gauss(pos,wdr,ws,D,wsn,ctn,k=0.05,al=90.0):
    n=len(pos)
    if n<=1: return ws
    wx,wy=math.cos(wdr),math.sin(wdr); proj=pos[:,0]*wx+pos[:,1]*wy
    order=np.argsort(proj); ve=ws.copy()
    for ii in range(1,n):
        i=order[ii]; l=0.0
        for jj in range(ii):
            j=order[jj]; dx=(proj[i]-proj[j])*1000.0
            if dx<=0: continue
            dy=abs((pos[i,0]-pos[j,0])*wy-(pos[i,1]-pos[j,1])*wx)*1000.0
            if math.degrees(math.atan2(dy,dx))>al: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0 or ct>=1: continue
            s=k*dx+0.1414213562373095*D
            df=(1-math.sqrt(1-ct))*math.exp(-0.5*(dy/s)**2)*(D/(D+2*k*dx))**2
            if df>0: l+=df*df
        if l>0: ve[i]=ws[i]*max(0,1-math.sqrt(l))
    return ve

@njit
def power_nb(ws_arr,wsn,pwn):
    out=np.zeros_like(ws_arr)
    for i in range(len(ws_arr)):
        v=ws_arr[i]
        if v<wsn[1] or v>=wsn[-2]: out[i]=0.0
        else: out[i]=np.interp(v,wsn,pwn)
    return out

# ============================================================
# 4. Numba 反事实核心循环
# ============================================================
@njit
def cf_core(pos_arr, nt, V_arr, th_arr_real, th_arr_base, D, wsn, pwn, ctn):
    """
    单农场一年的反事实计算。
    返回: (pn_sum, pr_sum, pb_sum)
    """
    pn = 0.0; pr = 0.0; pb = 0.0
    nh = len(V_arr)
    for s in range(nh):
        ws = np.full(nt, V_arr[s])
        # No-wake 基准（两种情景相同）
        pn_arr = power_nb(ws, wsn, pwn)
        pn += pn_arr.sum()
        # 情景A: 真实风向
        ve_r = wake_gauss(pos_arr, math.radians(270.0 - th_arr_real[s]), ws, D, wsn, ctn)
        pr += power_nb(ve_r, wsn, pwn).sum()
        # 情景B: 基准期风向
        ve_b = wake_gauss(pos_arr, math.radians(270.0 - th_arr_base[s]), ws, D, wsn, ctn)
        pb += power_nb(ve_b, wsn, pwn).sum()
    return pn, pr, pb

# ============================================================
# 5. 风向分布构建
# ============================================================
def build_baseline(fid, wind_df):
    """构建基准期(2014-2017)逐月风向频率分布"""
    dfb = wind_df[(wind_df['year'].isin(BASELINE)) & (wind_df['farm_id'] == fid)]
    se = np.linspace(0, 360, 17)
    ms = {m: np.zeros(16) for m in range(1, 13)}
    if len(dfb) == 0:
        return {m: np.ones(16)/16 for m in range(1, 13)}, se
    dfb = dfb[dfb['V_ms'] >= 3.0]
    if len(dfb) == 0:
        return {m: np.ones(16)/16 for m in range(1, 13)}, se
    sectors = np.clip((dfb['theta_deg'].values / 22.5).astype(np.int32), 0, 15)
    months_arr = dfb['month'].values
    for m in range(1, 13):
        mask = months_arr == m
        cnt = mask.sum()
        if cnt > 0:
            for s in range(16):
                ms[m][s] = (sectors[mask] == s).sum()
            ms[m] /= cnt
        else:
            ms[m] = np.ones(16) / 16
    return ms, se

def sample_wind(hourly_rows, wd_dist, se, seed_hash):
    """从基准分布中为每个小时采样风向"""
    np.random.seed(seed_hash)
    th_base = np.zeros(len(hourly_rows))
    for i, (_, row) in enumerate(hourly_rows.iterrows()):
        m = int(row['month'])
        probs = wd_dist[m]
        si = np.random.choice(len(probs), p=probs)
        th_base[i] = (se[si] + se[si+1]) / 2 + np.random.uniform(-5, 5)
    return th_base

# ============================================================
# 6. MAIN
# ============================================================
def main():
    t0 = datetime.now()
    print("任务一 §1.8 反事实分析")
    print(f"基准期: {BASELINE}, 检验期: {ANALYSIS}")
    print(f"每区 {N_FARMS_PER_REGION} 个代表风场")
    print("=" * 60)

    # [1] Load
    print("[1/4] 加载...")
    models = load_models(); turbs = load_turbines()
    wind_dfs = {rk: load_wind(rk) for rk in REGIONS}
    wind_dfs = {k: v for k, v in wind_dfs.items() if v is not None}
    print(f"  {len(turbs)} 风机, {len(models)} 机型, {len(wind_dfs)} 区域")

    # [2] Cluster + warm Numba
    print("[2/4] 聚类 + Numba 预热...")
    farm_cache = {}
    for rk in REGIONS:
        bbox = REGIONS[rk]
        a24 = [t for t in turbs if t['active'][2024]
               and bbox[2] <= t['lat'] <= bbox[0] and bbox[1] <= t['lon'] <= bbox[3]]
        if len(a24) >= 3:
            farm_cache[rk] = cluster_farms(a24)
    for rk in REGIONS:
        print(f"  {REG_NAMES[rk]}: {len(farm_cache.get(rk,{}))} 风场")

    tp = np.random.rand(10, 2) * 10; tws = np.full(10, 8.0); m0 = models['iea_10MW']
    _ = wake_gauss(tp, math.radians(180), tws, m0['D'], m0['ws'], m0['ct'])
    print("  Numba 预热完成")

    # [3] Compute
    CSV = os.path.join(OUT_DIR, "task1_counterfactual.csv")
    completed = set()
    if os.path.exists(CSV):
        with open(CSV, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed.add((r['region'], int(r['farm_id']), int(r['year'])))

    tasks = []
    for rk in REGIONS:
        if rk not in farm_cache: continue
        fids = sorted(farm_cache[rk].keys())[:N_FARMS_PER_REGION]
        for fid in fids:
            for yr in ANALYSIS:
                if (rk, fid, yr) not in completed:
                    tasks.append((rk, fid, yr))

    total = len(completed) + len(tasks)
    print(f"\n[3/4] 反事实: {len(completed)}/{total} 已完成, {len(tasks)} 待算")

    if not tasks:
        print("  全部已完成!")
    else:
        fout = open(CSV, 'a', newline='', encoding='utf-8-sig')
        w = csv.writer(fout)
        if not os.path.exists(CSV) or os.path.getsize(CSV) == 0:
            w.writerow(['region', 'farm_id', 'year', 'AEP_real_kWh', 'AEP_baseWD_kWh',
                        'Delta_AEP_WD_kWh', 'WakeLoss_real', 'WakeLoss_baseWD', 'Delta_WakeLoss_WD'])

        # 预构建风向分布（每个代表风场做一次）
        wd_caches = {}
        for rk in REGIONS:
            if rk not in farm_cache: continue
            fids = sorted(farm_cache[rk].keys())[:N_FARMS_PER_REGION]
            for fid in fids:
                wd_caches[(rk, fid)], _ = build_baseline(fid, wind_dfs[rk])

        for rk, fid, yr in tasks:
            ft = farm_cache[rk][fid]; n_turb = len(ft); pos = local_xy(ft)
            bys = [t.get('first_year', 2018) for t in ft if t.get('first_year')]
            myr = int(np.median(bys)) if bys else 2018
            model = models[assign_model(myr)]
            H, D, Pr = model['H'], model['D'], model['Pr']
            wsn, pwn, ctn = model['ws'], model['power'], model['ct']
            cap = n_turb * Pr

            dfy = wind_dfs[rk]
            dfy = dfy[(dfy['year'] == yr) & (dfy['farm_id'] == fid)]
            dfy = dfy[dfy['V_ms'] >= 3.0]
            nh = len(dfy)
            if nh == 0: continue

            V_arr = dfy['V_ms'].values * (H / H_REF) ** ALPHA
            th_real = dfy['theta_deg'].values

            # 构建该年情景B的采样风向
            wd_dist = wd_caches[(rk, fid)]
            se = np.linspace(0, 360, len(wd_dist[1]) + 1)
            months_arr = dfy['month'].values
            np.random.seed(fid * 10000 + yr)
            th_base = np.zeros(nh)
            for i in range(nh):
                m = months_arr[i]
                probs = wd_dist[m]
                si = np.random.choice(len(probs), p=probs)
                th_base[i] = (se[si] + se[si+1]) / 2 + np.random.uniform(-5, 5)

            # Numba 核心循环
            pn, pr, pb = cf_core(pos, n_turb, V_arr, th_real, th_base, D, wsn, pwn, ctn)

            wl_real = (pn - pr) / pn if pn > 0 else 0
            wl_base = (pn - pb) / pn if pn > 0 else 0

            w.writerow([rk, fid, yr,
                       f"{pr:.1f}", f"{pb:.1f}", f"{pr - pb:.1f}",
                       f"{wl_real:.4f}", f"{wl_base:.4f}", f"{wl_real - wl_base:.4f}"])
            fout.flush()
            print(f"  {REG_NAMES[rk]} F{fid} {yr}: "
                  f"ΔAEP={(pr-pb)/1e6:+.1f}GWh  ΔWL={(wl_real-wl_base)*100:+.2f}%",
                  flush=True)

        fout.close()

    # [4] 汇总
    print(f"\n[4/4] 汇总...")
    if os.path.exists(CSV):
        with open(CSV, 'r', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        das = [float(r['Delta_AEP_WD_kWh']) / 1e6 for r in rows]
        dws = [float(r['Delta_WakeLoss_WD']) * 100 for r in rows]

        n_pos_aep = sum(1 for d in das if d > 0)
        n_neg_aep = sum(1 for d in das if d < 0)
        n_pos_wl = sum(1 for d in dws if d > 0)
        n_neg_wl = sum(1 for d in dws if d < 0)

        print(f"  {len(rows)} 条记录")
        print(f"  ΔAEP: 范围={min(das):+.1f}~{max(das):+.1f} GWh  "
              f"均值={sum(das)/len(das):+.2f} GWh")
        print(f"        真实更优: {n_pos_aep}次, 基准更优: {n_neg_aep}次")
        print(f"  ΔWakeLoss: 范围={min(dws):+.2f}~{max(dws):+.2f}%  "
              f"均值={sum(dws)/len(dws):+.3f}%")
        print(f"              真实加重尾流: {n_pos_wl}次, 基准加重尾流: {n_neg_wl}次")

    el = (datetime.now() - t0).total_seconds() / 60
    print(f"\n耗时: {el:.1f} min | DONE")


if __name__ == "__main__":
    main()
