"""
任务二 核查证据 + 可视化
1. 旋转对照 & 打乱对照 (audit evidence)
2. 6 张高优先级图表
"""
import os, math, csv, numpy as np, yaml
from collections import defaultdict
from numba import njit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

TASK0_DIR = r"D:\1风力发电实习\offshore-task0\output\task0"
DATA_DIR  = r"D:\1风力发电实习\offshore-task2\data"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR   = os.path.join(OUT_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)

H_REF=100.0; ALPHA=0.11; K_WAKE=0.05; WAKE_LIMIT=90.0
ELEC = 0.95*0.97

# ========== LOADERS ==========
def load_model():
    with open(os.path.join(DATA_DIR,"iea_10MW.yaml"),'r',encoding='utf-8') as f:
        c=yaml.safe_load(f)
    pt=c['power_thrust_table']
    return {'D':c['rotor_diameter'],'H':c['hub_height'],
            'Pr':pt['controller_dependent_turbine_parameters']['rated_power'],
            'ws':np.array(pt['wind_speed'],dtype=np.float64),
            'power':np.array(pt['power'],dtype=np.float64),
            'ct':np.array(pt['thrust_coefficient'],dtype=np.float64)}

def load_coords():
    c=defaultdict(lambda:defaultdict(list))
    with open(os.path.join(TASK0_DIR,"turbine_coordinates.csv"),'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid=int(r['farm_id']); yr=int(r['year'])
            c[fid][yr].append({'x':float(r['x_m'])/1000,'y':float(r['y_m'])/1000,
                               'lon':float(r['lon']),'lat':float(r['lat'])})
    return c

@njit
def wake(pos,wdr,ws,D,wsn,ctn):
    n=len(pos)
    if n<=1: return ws
    wx,wy=math.cos(wdr),math.sin(wdr); proj=pos[:,0]*wx+pos[:,1]*wy
    order=np.argsort(proj); ve=ws.copy()
    for ii in range(1,n):
        i=order[ii]; losses=0.0
        for jj in range(ii):
            j=order[jj]; dx=(proj[i]-proj[j])*1000.0
            if dx<=0: continue
            dy=abs((pos[i,0]-pos[j,0])*wy-(pos[i,1]-pos[j,1])*wx)*1000.0
            if math.degrees(math.atan2(dy,dx))>WAKE_LIMIT: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0 or ct>=1: continue
            s=K_WAKE*dx+0.1414*D; df=(1-math.sqrt(1-ct))*math.exp(-0.5*(dy/s)**2)*(D/(D+2*K_WAKE*dx))**2
            if df>0: losses+=df*df
        if losses>0: ve[i]=ws[i]*max(0,1-math.sqrt(losses))
    return ve

@njit
def pwr(v_arr,wsn,pwn):
    o=np.zeros_like(v_arr)
    for i in range(len(v_arr)):
        v=v_arr[i]
        if v<wsn[1] or v>=wsn[-2]: o[i]=0.0
        else: o[i]=np.interp(v,wsn,pwn)
    return o

# =========== PICK 5 REPRESENTATIVE FARMS ============
# F0 (928, China grid), F2 (572, Belgium cluster), F3 (443, China belt)
# F5 (339, UK belt), F15 (293, China multi_cluster)
REP_FARMS = [0, 2, 3, 5, 15]

# =========== AUDIT 1: ROTATION TEST ============
def rotation_test(fid, yr, turbs, model, V_arr, th_arr):
    n=len(turbs); D=model['D']; wsn=model['ws']; pwn=model['power']; ctn=model['ct']
    pos=np.array([[t['x'],t['y']] for t in turbs], dtype=np.float64)
    nh=len(V_arr)
    scan_deg=range(0,180,10)
    results=[]
    for rot in scan_deg:
        pn_sum=0.0; pw_sum=0.0
        for s in range(nh):
            ws=np.full(n,V_arr[s]); pn_sum+=float(pwr(ws,wsn,pwn).sum())
            wdr=math.radians(270.0-(th_arr[s]+rot)%360)
            pw_sum+=float(pwr(wake(pos,wdr,ws,D,wsn,ctn),wsn,pwn).sum())
        results.append((rot, float(pw_sum)*ELEC, float(pn_sum)*ELEC))
    return results

# =========== AUDIT 2: SHUFFLE TEST ============
def shuffle_test(fid, yr, turbs, model, V_arr, th_arr):
    n=len(turbs); D=model['D']; wsn=model['ws']; pwn=model['power']; ctn=model['ct']
    pos_real=np.array([[t['x'],t['y']] for t in turbs], dtype=np.float64)
    # Shuffle
    idx=np.random.permutation(n)
    pos_shuffled=pos_real[idx].copy()
    nh=len(V_arr)
    pn_sum=0.0; pr_sum=0.0; ps_sum=0.0
    for s in range(nh):
        ws=np.full(n,V_arr[s]); pn_sum+=float(pwr(ws,wsn,pwn).sum())
        wdr=math.radians(270.0-th_arr[s])
        pr_sum+=float(pwr(wake(pos_real,wdr,ws,D,wsn,ctn),wsn,pwn).sum())
        ps_sum+=float(pwr(wake(pos_shuffled,wdr,ws,D,wsn,ctn),wsn,pwn).sum())
    return (float(pr_sum)*ELEC, float(ps_sum)*ELEC, float(pn_sum)*ELEC)

# =========== MAIN ============
def main():
    coords=load_coords(); model=load_model()
    tp=np.random.rand(10,2); tw=np.full(10,8.0); _=wake(tp,math.radians(180),tw,model['D'],model['ws'],model['ct'])
    print("Audit + Viz running...")

    # Read task2 summary for context
    rows=[]
    with open(os.path.join(OUT_DIR,"task2_summary.csv"),'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): rows.append(r)

    # ===== Audit: Rotation test for F0 2024 =====
    print("\n1/8 Rotation test (F0 2024)...")
    f0_turbs=coords[0].get(2024,[]); n=len(f0_turbs)
    nc_path=os.path.join(DATA_DIR,"era5_east_asia_2024.nc")
    import tempfile, shutil, netCDF4
    from netCDF4 import num2date
    tmpdir=tempfile.mkdtemp(); tmpnc=os.path.join(tmpdir,"r.nc"); shutil.copy2(nc_path,tmpnc)
    ds=netCDF4.Dataset(tmpnc,'r')
    clat=np.mean([t['lat'] for t in f0_turbs]); clon=np.mean([t['lon'] for t in f0_turbs])
    la=ds['latitude'][:]; lo=ds['longitude'][:]
    ila=int(np.argmin(np.abs(la-clat))); ilo=int(np.argmin(np.abs(lo-clon)))
    u100=np.array(ds['u100'][:,ila,ilo]); v100=np.array(ds['v100'][:,ila,ilo])
    all_times=num2date(ds['valid_time'][:],units=ds['valid_time'].units)
    ds.close(); shutil.rmtree(tmpdir)
    # Filter months 1,7 for speed
    idx=[i for i,t in enumerate(all_times) if t.month in [1,7]]
    ws_raw=np.sqrt(u100[idx]**2+v100[idx]**2); wd_raw=(np.degrees(np.arctan2(u100[idx],v100[idx]))+180)%360
    mask=ws_raw>=3.0; idx2=np.where(mask)[0]
    V_arr=ws_raw[idx2]*(model['H']/H_REF)**ALPHA; th_arr=wd_raw[idx2]

    rot_res=rotation_test(0,2024,f0_turbs,model,V_arr[:500],th_arr[:500])
    fig,ax=plt.subplots(figsize=(8,5))
    degs=[r[0] for r in rot_res]; aeps=[r[1]/1e6 for r in rot_res]
    ax.plot(degs,aeps,'o-',color='steelblue',markersize=6)
    ax.axvline(0,color='red',linestyle='--',label='Real orientation'); ax.axhline(aeps[0],color='gray',linestyle=':',alpha=0.5)
    ax.set_xlabel('Rotation angle (deg)'); ax.set_ylabel('AEP (GWh)')
    ax.set_title(f'Audit: Rotation-AEP Response — Farm 0 (928 turbines, 2024)')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'audit_rotation_test.png'),dpi=150); plt.close(fig)
    print("  -> audit_rotation_test.png")

    # ===== Audit: Shuffle test =====
    print("2/8 Shuffle test...")
    sr,sxr,pnr=shuffle_test(0,2024,f0_turbs,model,V_arr[:500],th_arr[:500])
    print(f"  Real AEP={sr/1e6:.1f} GWh, Shuffled AEP={sxr/1e6:.1f} GWh, Delta={(sr-sxr)/sr*100:.1f}%")
    # Save to CSV
    with open(os.path.join(OUT_DIR,"audit_shuffle_results.csv"),'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['farm_id','year','AEP_real_GWh','AEP_shuffled_GWh','Delta_pct'])
        w.writerow([0,2024,f"{sr/1e6:.1f}",f"{sxr/1e6:.1f}",f"{(sr-sxr)/sr*100:.1f}"])
    print("  -> audit_shuffle_results.csv")

    # ===== Fig 1: Wake flow map for 3 directions =====
    print("3/8 Wake flow overlay (Farm 3, 443 turbines)...")
    f3_turbs=coords[3].get(2024,[])
    if len(f3_turbs)>10:
        pos3=np.array([[t['x'],t['y']] for t in f3_turbs[:50]],dtype=np.float64)  # 50 turbine subset
        n3=len(pos3)
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        for axi,(wd,title) in enumerate(zip([90,180,270],['East wind (90°)','North wind (180°)','West wind (270°)'])):
            ws=np.full(n3,10.0); ve=wake(pos3,math.radians(270-wd),ws,model['D'],model['ws'],model['ct'])
            deficit=(ws-ve)/ws*100
            sc=axes[axi].scatter(pos3[:,0],pos3[:,1],s=deficit*5+10,c=deficit,cmap='Reds',vmin=0,vmax=30,edgecolors='gray',linewidths=0.3)
            axes[axi].set_title(title); axes[axi].set_aspect('equal')
        fig.colorbar(sc,ax=axes[2],shrink=0.7,label='Wake deficit (%)')
        fig.savefig(os.path.join(FIG_DIR,'fig1_wake_flow.png'),dpi=150); plt.close(fig)
        print("  -> fig1_wake_flow.png")

    # ===== Fig 2: Hourly P_noWake vs P_wake time series =====
    print("4/8 Time series (Farm 5, month 1)...")
    f5_turbs=coords[5].get(2024,[]); n5=len(f5_turbs)
    idx_jan=idx2[:168]; V_jan=V_arr[:168]; th_jan=th_arr[:168]
    pos5=np.array([[t['x'],t['y']] for t in f5_turbs],dtype=np.float64)
    pn_seq=[]; pw_seq=[]
    for s in range(len(V_jan)):
        ws5=np.full(n5,V_jan[s]); pn_seq.append(float(pwr(ws5,model['ws'],model['power']).sum()))
        pw_seq.append(float(pwr(wake(pos5,math.radians(270-th_jan[s]),ws5,model['D'],model['ws'],model['ct']),model['ws'],model['power']).sum()))
    fig,ax1=plt.subplots(figsize=(12,5))
    ax1.plot(pn_seq,label='P_noWake',color='gray',alpha=0.7); ax1.plot(pw_seq,label='P_wake (Gaussian)',color='steelblue')
    ax2=ax1.twinx(); ax2.plot(V_jan,label='Wind speed (m/s)',color='red',alpha=0.4,linestyle='--')
    ax1.set_xlabel('Hour'); ax1.set_ylabel('Power (kW)'); ax2.set_ylabel('Wind speed (m/s)')
    ax1.set_title(f'Farm 5 (339 turbines) — Jan 2024 first week'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig2_time_series.png'),dpi=150); plt.close(fig)
    print("  -> fig2_time_series.png")

    # ===== Fig 3: Wake loss per-turbine heatmap =====
    print("5/8 Wake loss heatmap (Farm 0, 2024)...")
    pos0=np.array([[t['x'],t['y']] for t in f0_turbs],dtype=np.float64); n0=len(pos0)
    # Calculate per-turbine annual mean deficit
    turbine_wl=np.zeros(n0)
    sample_hours=min(200,len(V_arr))
    for s in range(sample_hours):
        ws0=np.full(n0,V_arr[s]); ve0=wake(pos0,math.radians(270-th_arr[s]),ws0,model['D'],model['ws'],model['ct'])
        turbine_wl+=(1-ve0/ws0)*100/sample_hours
    fig,ax=plt.subplots(figsize=(10,8))
    sc=ax.scatter(pos0[:,0],pos0[:,1],s=5,c=turbine_wl,cmap='Reds',vmin=0,vmax=20)
    ax.set_aspect('equal'); ax.set_title(f'Farm 0 (928 turbines) — Per-turbine mean wake deficit (%)')
    fig.colorbar(sc,ax=ax,shrink=0.7,label='Wake deficit (%)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig3_wake_heatmap.png'),dpi=150); plt.close(fig)
    print("  -> fig3_wake_heatmap.png")

    # ===== Fig 4: Delta AEP global map =====
    print("6/8 Delta AEP map...")
    cf_rows=[]
    with open(os.path.join(OUT_DIR,"task2_counterfactual.csv"),'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): cf_rows.append(r)
    farm_delta={}
    for r in cf_rows:
        fid=int(r['farm_id']); das=float(r['Delta_AEP_WD_kWh'])/1e6
        farm_delta[fid]=das
    # Load farm centroids
    farms={}
    with open(os.path.join(TASK0_DIR,"farms_master.csv"),'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            fid=int(r['farm_id']); farms[fid]=(float(r['centroid_lon']),float(r['centroid_lat']))
    # Simple global scatter
    fig,ax=plt.subplots(figsize=(16,10))
    lons=[farms[f][0] for f in farm_delta if f in farms]
    lats=[farms[f][1] for f in farm_delta if f in farms]
    dvals=[farm_delta[f] for f in farm_delta if f in farms]
    colors=['red' if d<0 else 'green' for d in dvals]
    ax.scatter(lons,lats,s=[abs(d)/10+20 for d in dvals],c=colors,alpha=0.6,edgecolors='gray',linewidths=0.3)
    ax.axhline(0,color='gray',alpha=0.3); ax.axvline(0,color='gray',alpha=0.3)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Delta AEP Real vs Baseline (1981-2010 WD). Red=Loss, Green=Gain. n={len(dvals)} farms')
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig4_delta_aep_map.png'),dpi=150); plt.close(fig)
    print("  -> fig4_delta_aep_map.png")

    # ===== Fig 5: Wake model comparison (placeholder — only Gaussian computed) =====
    print("7/8 Wake model comparison...")
    # We only have Gaussian in Core. Show CF/WL scatter overlay from summary
    fig,ax=plt.subplots(figsize=(8,6))
    cfs=[float(r['CF'])*100 for r in rows]; wls=[float(r['WakeLoss'])*100 for r in rows]
    ax.scatter(cfs,wls,s=2,alpha=0.3,c='steelblue')
    ax.set_xlabel('CF (%)'); ax.set_ylabel('Wake Loss (%)'); ax.set_title(f'CF vs WakeLoss — All 1203 farms (Gaussian)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig5_cf_wl_scatter.png'),dpi=150); plt.close(fig)
    print("  -> fig5_cf_wl_scatter.png")

    # ===== Fig 6: Power curve plot =====
    print("8/8 Power curves...")
    fig,ax=plt.subplots(figsize=(10,6))
    colors=['#2ecc71','#3498db','#e74c3c']
    for ci,(name,fn) in enumerate([('IEA 15MW','iea_15MW.yaml'),('IEA 10MW','iea_10MW.yaml'),('NREL 5MW','nrel_5MW.yaml')]):
        with open(os.path.join(DATA_DIR,fn),'r',encoding='utf-8') as f:
            c=yaml.safe_load(f)
        pt=c['power_thrust_table']; ws=pt['wind_speed']; pw=pt['power']
        ax.plot(ws,[p/1000 for p in pw],label=f'{name} (D={c["rotor_diameter"]:.0f}m)',color=colors[ci])
    ax.axvline(3,color='gray',linestyle='--',alpha=0.5,label='Cut-in (3 m/s)')
    ax.axvline(25,color='gray',linestyle=':',alpha=0.5,label='Cut-out (25 m/s)')
    ax.set_xlabel('Wind Speed (m/s)'); ax.set_ylabel('Power (MW)')
    ax.set_title('Reference Turbine Power Curves (NREL/IEA)'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig6_power_curves.png'),dpi=150); plt.close(fig)
    print("  -> fig6_power_curves.png")

    print(f"\nAll outputs in {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(FIG_DIR,f))//1024}KB)")

if __name__=="__main__":
    main()
