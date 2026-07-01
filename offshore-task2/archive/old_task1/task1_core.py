"""
=============================================================================
 任务一 FINAL v2 — 边算边写 + 可续传
 Numba JIT Gaussian wake + 进度实时输出
 反事实分析待后续优化启用
=============================================================================
"""
import os, math, time, csv, numpy as np
from collections import defaultdict
from datetime import datetime
from numba import njit

# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ALL_YRS   = list(range(2014, 2025))
REGIONS   = {"east_asia": [41,104,8,142], "europe": [63,-10,39,22], "us_east": [42,-77,36,-69]}
REG_NAMES = {"east_asia": "东亚", "europe": "欧洲", "us_east": "美国"}
BASELINE  = [2014,2015,2016,2017]; ANALYSIS = [2018,2019,2020,2021,2022,2023,2024]
H_REF=100.0; ALPHA=0.11; K=0.05; WAKE_LIMIT=90.0

TEST_MODE = False; TEST_FARMS = 3

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
                         'active':{y:r.get(f'active_{y}')=='True' for y in ALL_YRS}})
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
# 3. NUMBA WAKE + POWER
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
def wake_jensen(pos,wdr,ws,D,wsn,ctn,k=0.075,al=90.0):
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
            r_wake=k*dx+D/2
            if abs(dy)>=r_wake: continue
            if math.degrees(math.atan2(dy,dx))>al: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0: continue
            df=(1-math.sqrt(1-ct))*(D/(D+2*k*dx))**2
            if df>0: l+=df*df
        if l>0: ve[i]=ws[i]*max(0,1-math.sqrt(l))
    return ve

@njit
def wake_curl(pos,wdr,ws,D,wsn,ctn,k=0.05,al=90.0):
    n=len(pos)
    if n<=1: return ws
    wx,wy=math.cos(wdr),math.sin(wdr); proj=pos[:,0]*wx+pos[:,1]*wy
    order=np.argsort(proj); ve=ws.copy()
    for ii in range(1,n):
        i=order[ii]; l=0.0
        for jj in range(ii):
            j=order[jj]; dx=(proj[i]-proj[j])*1000.0
            if dx<=0: continue
            dy_raw=abs((pos[i,0]-pos[j,0])*wy-(pos[i,1]-pos[j,1])*wx)*1000.0
            curl_off=0.02*dx; dy=abs(dy_raw-curl_off)
            if math.degrees(math.atan2(dy,dx))>al: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0 or ct>=1: continue
            s=k*dx+0.1767766952966369*D
            df=(1-math.sqrt(1-ct))*math.exp(-0.5*(dy/s)**2)*(D/(D+2.5*k*dx))**2
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
# 4. MAIN
# ============================================================
def main():
    t0=datetime.now()
    mode="TEST" if TEST_MODE else "FULL"
    print(f"任务一 FINAL v2 ({mode}) — 边算边写,可续传")
    print("="*60)

    # [1]
    print("[1/4] 加载...")
    models=load_models(); turbs=load_turbines()
    print(f"  {len(turbs)} 风机, {len(models)} 机型")

    # [2]
    print("[2/4] 风速 Parquet + 聚类...")
    wind_dfs={}; farm_cache={}
    for rk in REGIONS:
        df=load_wind(rk)
        if df is not None: wind_dfs[rk]=df
        bbox=REGIONS[rk]
        a24=[t for t in turbs if t['active'][2024] and bbox[2]<=t['lat']<=bbox[0] and bbox[1]<=t['lon']<=bbox[3]]
        if len(a24)>=3: farm_cache[rk]=cluster_farms(a24)
    for rk in REGIONS:
        nf=len(farm_cache.get(rk,{}))
        print(f"  {REG_NAMES[rk]}: {nf} 风场")

    # Warm Numba
    tp=np.random.rand(10,2)*10; tws=np.full(10,8.0); m0=models['iea_10MW']
    _=wake_gauss(tp,math.radians(180),tws,m0['D'],m0['ws'],m0['ct'])
    _=wake_jensen(tp,math.radians(180),tws,m0['D'],m0['ws'],m0['ct'])
    _=wake_curl(tp,math.radians(180),tws,m0['D'],m0['ws'],m0['ct'])
    print("  Numba OK (Gauss+Jensen+Curl)")

    # [3] Compute — 边算边写（年度 + 小时级）, v3 adds Jensen/Curl + Volatility
    CSV = os.path.join(OUT_DIR,"task1_summary_v3.csv")
    completed=set()
    if os.path.exists(CSV):
        with open(CSV,'r',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                completed.add((r['region'],int(r['farm_id']),int(r['year']),r['wake_model']))

    # 构建任务列表 — 每个农场年算一次（一次性出三种尾流）
    tasks=[]
    for rk in REGIONS:
        if rk not in farm_cache: continue
        fids=sorted(farm_cache[rk].keys())
        if TEST_MODE: fids=fids[:TEST_FARMS]
        yrs=[2024] if TEST_MODE else ALL_YRS
        for fid in fids:
            for yr in yrs:
                k=(rk,fid,yr,'gaussian')  # 用 gaussian 做完成标记
                if k not in completed: tasks.append((rk,fid,yr))

    total=len(completed)+len(tasks)
    print(f"\n[3/4] 计算: {len(completed)}/{total} 已完成, {len(tasks)} 待算")

    if not tasks:
        print("  全部已完成!")
    else:
        fout=open(CSV,'a',newline='',encoding='utf-8-sig')
        w=csv.writer(fout)
        if not os.path.exists(CSV) or os.path.getsize(CSV)==0:
            w.writerow(['region','farm_id','year','n_turb','wake_model','AEP_kWh','CF','WakeLoss','Volatility_kW','RampFreq'])

        # 小时级输出 — 新文件 v3，含三列尾流
        hourly_files = {}
        hourly_writers = {}
        for rk in REGIONS:
            hp = os.path.join(OUT_DIR, f"task1_hourly_v3_{rk}.csv")
            hf = open(hp, 'a', newline='', encoding='utf-8-sig')
            hw = csv.writer(hf)
            if os.path.getsize(hp) == 0:
                hw.writerow(['farm_id','year','month','day','hour','V_free_ms','theta_deg',
                             'P_noWake_kW','P_wake_Gaussian_kW','P_wake_Jensen_kW','P_wake_Curl_kW','n_turb'])
            hourly_files[rk] = hf
            hourly_writers[rk] = hw

        t_start=datetime.now(); done=0

        for rk,fid,yr in tasks:
            ft=farm_cache[rk][fid]; n_turb=len(ft); pos=local_xy(ft)
            bys=[t.get('first_year',2018) for t in ft if t.get('first_year')]
            myr=int(np.median(bys)) if bys else 2018
            model=models[assign_model(myr)]
            H,D,Pr=model['H'],model['D'],model['Pr']
            wsn,pwn,ctn=model['ws'],model['power'],model['ct']; cap=n_turb*Pr
            hw = hourly_writers[rk]

            dfy=wind_dfs[rk]
            dfy=dfy[(dfy['year']==yr)&(dfy['farm_id']==fid)]
            dfy=dfy[dfy['V_ms']>=3.0]; nh=len(dfy)
            if nh==0: continue
            V_arr=dfy['V_ms'].values*(H/H_REF)**ALPHA
            th_arr=dfy['theta_deg'].values

            pn_sum=0.0; gauss_sum=0.0; jensen_sum=0.0; curl_sum=0.0
            gauss_pw=[]; jensen_pw=[]; curl_pw=[]  # 收集小时级出力用于Volatility

            for s in range(nh):
                ws=np.full(n_turb,V_arr[s])
                pn_hour=float(power_nb(ws,wsn,pwn).sum())
                pn_sum+=pn_hour

                wdr=math.radians(270-th_arr[s])
                ve_gauss=wake_gauss(pos,wdr,ws,D,wsn,ctn,K,WAKE_LIMIT)
                pw_gauss_h=float(power_nb(ve_gauss,wsn,pwn).sum())
                gauss_sum+=pw_gauss_h; gauss_pw.append(pw_gauss_h)

                ve_jensen=wake_jensen(pos,wdr,ws,D,wsn,ctn,0.075,WAKE_LIMIT)
                pw_jensen_h=float(power_nb(ve_jensen,wsn,pwn).sum())
                jensen_sum+=pw_jensen_h; jensen_pw.append(pw_jensen_h)

                ve_curl=wake_curl(pos,wdr,ws,D,wsn,ctn,K,WAKE_LIMIT)
                pw_curl_h=float(power_nb(ve_curl,wsn,pwn).sum())
                curl_sum+=pw_curl_h; curl_pw.append(pw_curl_h)

                row_data = dfy.iloc[s]
                hw.writerow([fid, yr, int(row_data['month']), int(row_data['day']),
                             int(row_data['hour']),
                             f"{V_arr[s]:.2f}", f"{th_arr[s]:.1f}",
                             f"{pn_hour:.1f}", f"{pw_gauss_h:.1f}",
                             f"{pw_jensen_h:.1f}", f"{pw_curl_h:.1f}", n_turb])

            # Write annual rows for all three wake models
            for wname, pw_sum, pw_arr in [
                ('gaussian', gauss_sum, gauss_pw),
                ('jensen', jensen_sum, jensen_pw),
                ('curl', curl_sum, curl_pw)]:
                cf_val=pw_sum/(cap*nh) if cap>0 else 0
                wl_val=(pn_sum-pw_sum)/pn_sum if pn_sum>0 else 0
                vol=float(np.std(np.array(pw_arr)))
                pw_a=np.array(pw_arr)
                ramp_freq=float(np.mean(np.abs(np.diff(pw_a))>0.2*cap)) if len(pw_a)>1 else 0.0
                w.writerow([rk,fid,yr,n_turb,wname,
                           f"{pw_sum:.1f}",f"{cf_val:.4f}",f"{wl_val:.4f}",
                           f"{vol:.1f}",f"{ramp_freq:.4f}"])
            fout.flush()
            if done % 50 == 0:
                for hf_obj in hourly_files.values(): hf_obj.flush()
            done+=1

            if done%50==0:
                el=(datetime.now()-t_start).total_seconds(); rate=done/el if el>0 else 0
                rem=len(tasks)-done; eta=rem/rate/60 if rate>0 else 0
                print(f"  {done}/{len(tasks)} ({done*100//len(tasks)}%) ~{eta:.0f}min {datetime.now().strftime('%H:%M:%S')}")

        fout.close()
        for hf_obj in hourly_files.values(): hf_obj.close()
        el=(datetime.now()-t_start).total_seconds()/60
        print(f"  完成! 耗时: {el:.1f} min")

    # [4] Viz
    print(f"\n[4/4] 图表...")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except: pass
    all_r=[]
    if os.path.exists(CSV):
        with open(CSV,'r',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                all_r.append({'region':row['region'],'farm_id':int(row['farm_id']),
                              'year':int(row['year']),'n_turb':int(row['n_turb']),
                              'wake_model':row.get('wake_model','gaussian'),
                              'AEP':float(row['AEP_kWh']),'CF':float(row['CF']),
                              'WakeLoss':float(row['WakeLoss'])})
    if all_r:
        cfs=[o['CF']*100 for o in all_r if o.get('CF',0)>0]
        if cfs:
            fig,ax=plt.subplots(figsize=(8,5))
            ax.hist(cfs,bins=30,color='steelblue',edgecolor='white')
            ax.set_xlabel('CF (%)'); ax.set_title(f'Capacity Factor (n={len(cfs)})')
            fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_cf.png'),dpi=150); plt.close(fig)

        wls=[o['WakeLoss']*100 for o in all_r if o.get('WakeLoss',0)>0]
        if wls:
            fig,ax=plt.subplots(figsize=(8,5))
            ax.hist(wls,bins=30,color='coral',edgecolor='white')
            ax.set_xlabel('Wake Loss (%)'); ax.set_title(f'Wake Loss (n={len(wls)})')
            fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_wl.png'),dpi=150); plt.close(fig)

        reg_stats={}
        for o in all_r:
            rk=o.get('region','?')
            if rk not in reg_stats: reg_stats[rk]=[]
            reg_stats[rk].append(o.get('WakeLoss',0)*100)
        if reg_stats:
            fig,ax=plt.subplots(figsize=(8,5))
            rks=sorted(reg_stats.keys()); vals=[np.mean(reg_stats[k]) for k in rks]
            ax.bar(rks,vals,color=['steelblue','darkorange','seagreen'])
            ax.set_ylabel('Mean Wake Loss (%)'); ax.set_title('Wake Loss by Region')
            fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_region.png'),dpi=150); plt.close(fig)

    el=(datetime.now()-t0).total_seconds()/60
    print(f"总耗时: {el:.1f} min | DONE")


if __name__=="__main__":
    main()
