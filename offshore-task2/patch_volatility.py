"""修复 v4 CSV 中 Volatility=0 的行：只重算 >3 turbines 的农场年"""
import csv, os, math, numpy as np, tempfile, shutil
from collections import defaultdict
from numba import njit
import netCDF4
from netCDF4 import num2date

TASK0_DIR = r"D:\1风力发电实习\offshore-task0\output\task0"
DATA_DIR  = r"D:\1风力发电实习\offshore-task2\data"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
H_REF=100.0; ALPHA=0.11; K_WAKE=0.05; WAKE_LIMIT=90.0
ELEC = 0.95*0.97

# ---- Numba (same as task2_core) ----
@njit
def wake_gauss(pos,wdr,ws,D,wsn,ctn,k=0.05,al=90.0):
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
            if math.degrees(math.atan2(dy,dx))>al: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0 or ct>=1: continue
            s=k*dx+0.1414*D; df=(1-math.sqrt(1-ct))*math.exp(-0.5*(dy/s)**2)*(D/(D+2*k*dx))**2
            if df>0: losses+=df*df
        if losses>0: ve[i]=ws[i]*max(0,1-math.sqrt(losses))
    return ve
@njit
def wake_jensen(pos,wdr,ws,D,wsn,ctn,k=0.075,al=90.0):
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
            if abs(dy)>=k*dx+D/2: continue
            if math.degrees(math.atan2(dy,dx))>al: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0: continue
            df=(1-math.sqrt(1-ct))*(D/(D+2*k*dx))**2
            if df>0: losses+=df*df
        if losses>0: ve[i]=ws[i]*max(0,1-math.sqrt(losses))
    return ve
@njit
def wake_curl(pos,wdr,ws,D,wsn,ctn,k=0.05,al=90.0):
    n=len(pos)
    if n<=1: return ws
    wx,wy=math.cos(wdr),math.sin(wdr); proj=pos[:,0]*wx+pos[:,1]*wy
    order=np.argsort(proj); ve=ws.copy()
    for ii in range(1,n):
        i=order[ii]; losses=0.0
        for jj in range(ii):
            j=order[jj]; dx=(proj[i]-proj[j])*1000.0
            if dx<=0: continue
            dy_raw=abs((pos[i,0]-pos[j,0])*wy-(pos[i,1]-pos[j,1])*wx)*1000.0; dy=abs(dy_raw-0.02*dx)
            if math.degrees(math.atan2(dy,dx))>al: continue
            ct=np.interp(ve[j],wsn,ctn)
            if ct<=0 or ct>=1: continue
            s=k*dx+0.1768*D; df=(1-math.sqrt(1-ct))*math.exp(-0.5*(dy/s)**2)*(D/(D+2.5*k*dx))**2
            if df>0: losses+=df*df
        if losses>0: ve[i]=ws[i]*max(0,1-math.sqrt(losses))
    return ve
@njit
def power_nb(ws_arr,wsn,pwn):
    o=np.zeros_like(ws_arr)
    for i in range(len(ws_arr)):
        v=ws_arr[i]
        if v<wsn[1] or v>=wsn[-2]: o[i]=0.0
        else: o[i]=np.interp(v,wsn,pwn)
    return o

class ERA5Reader:
    def __init__(self,p): self.path=p
    def __enter__(self):
        self.tmp=tempfile.mkdtemp(); self.tmpnc=os.path.join(self.tmp,"d.nc")
        shutil.copy2(self.path,self.tmpnc); self.ds=netCDF4.Dataset(self.tmpnc,'r'); return self
    def __exit__(self,*a):
        if hasattr(self,'ds'): self.ds.close()
        if hasattr(self,'tmp'): shutil.rmtree(self.tmp)
    def wind_at(self,lat,lon):
        la=self.ds['latitude'][:]; lo=self.ds['longitude'][:]
        ila=int(np.argmin(np.abs(la-lat))); ilo=int(np.argmin(np.abs(lo-lon)))
        return np.array(self.ds['u100'][:,ila,ilo]),np.array(self.ds['v100'][:,ila,ilo])

print("Loading v4 CSV...")
rows=[]
with open(os.path.join(OUT_DIR,"task2_summary_v4.csv"),'r',encoding='utf-8-sig') as f:
    reader=csv.DictReader(f); fieldnames=reader.fieldnames
    for r in reader: rows.append(r)

# Find faulty rows
faulty=defaultdict(list)
for i,r in enumerate(rows):
    if float(r['Volatility_kW'])==0 and int(r['n_turb'])>3:
        faulty[(int(r['farm_id']),int(r['year']))].append(i)

print(f"{len(faulty)} farm-years need repair ({sum(len(v) for v in faulty.values())} rows)")

# Load coords + model
coords=defaultdict(lambda:defaultdict(list))
with open(os.path.join(TASK0_DIR,"turbine_coordinates.csv"),'r',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        coords[int(r['farm_id'])][int(r['year'])].append({'x_m':float(r['x_m']),'y_m':float(r['y_m']),'lon':float(r['lon']),'lat':float(r['lat'])})

import yaml
with open(os.path.join(DATA_DIR,"iea_10MW.yaml"),'r',encoding='utf-8') as f:
    c=yaml.safe_load(f)
pt=c['power_thrust_table']
model={'D':c['rotor_diameter'],'H':c['hub_height'],'Pr':pt['controller_dependent_turbine_parameters']['rated_power'],
       'ws':np.array(pt['wind_speed'],dtype=np.float64),'power':np.array(pt['power'],dtype=np.float64),'ct':np.array(pt['thrust_coefficient'],dtype=np.float64)}

# Repair loop
repaired=0
for (fid,yr), indices in faulty.items():
    turbs=coords.get(fid,{}).get(yr,[])
    if len(turbs)<2: continue
    clat=np.mean([t['lat'] for t in turbs]); clon=np.mean([t['lon'] for t in turbs])
    if 8<=clat<=44 and 104<=clon<=143:
        rkey='east_asia' if clat<=41 else 'japan'
    elif 39<=clat<=63 and -12<=clon<=32: rkey='europe'
    else: rkey='us_east'
    nc=os.path.join(DATA_DIR,f"era5_{rkey}_{yr}.nc")
    if not os.path.exists(nc): nc=os.path.join(DATA_DIR,f"era5_japan_{yr}.nc")
    if not os.path.exists(nc): continue

    with ERA5Reader(nc) as era:
        u100,v100=era.wind_at(clat,clon)
    ws=np.sqrt(u100**2+v100**2); wd=(np.degrees(np.arctan2(u100,v100))+180)%360
    mask=ws>=3.0; idx=np.where(mask)[0]; nh=len(idx)
    if nh==0: continue
    V=ws[idx]*(model['H']/H_REF)**ALPHA; th=wd[idx]
    n_turb=len(turbs); cap=n_turb*model['Pr']; pos=np.array([[t['x_m']/1000,t['y_m']/1000] for t in turbs],dtype=np.float64)

    wnames=['gaussian','jensen','curl']; wfuncs=[wake_gauss,wake_jensen,wake_curl]; wks=[K_WAKE,0.075,K_WAKE]
    for wi,(wn,wfunc,wk) in enumerate(zip(wnames,wfuncs,wks)):
        pw_arr=np.zeros(nh)
        for s in range(nh):
            ws_arr=np.full(n_turb,V[s]); wdr=math.radians(270-th[s])
            pw_arr[s]=float(power_nb(wfunc(pos,wdr,ws_arr,model['D'],model['ws'],model['ct'],wk,WAKE_LIMIT),model['ws'],model['power']).sum())
        AEP=float(pw_arr.sum())*ELEC; CF=AEP/(cap*nh); Vol=float(np.std(pw_arr*ELEC))
        CV=Vol/(float(np.mean(pw_arr))*ELEC) if float(np.mean(pw_arr))>0 else 0
        P5=float(np.percentile(pw_arr,5))*ELEC; P95=float(np.percentile(pw_arr,95))*ELEC
        RF=float(np.mean(np.abs(np.diff(pw_arr))>0.2*cap)) if len(pw_arr)>1 else 0
        lo=float(np.mean(pw_arr<0.1*cap)); hi=float(np.mean(pw_arr>0.9*cap))
        # Update the row
        ri=[idx for idx in indices if rows[idx]['wake_model']==wn]
        for i in ri:
            rows[i]['Volatility_kW']=f"{Vol:.1f}"; rows[i]['CV']=f"{CV:.4f}"
            rows[i]['P5_kW']=f"{P5:.1f}"; rows[i]['P95_kW']=f"{P95:.1f}"
            rows[i]['RampFreq']=f"{RF:.4f}"; rows[i]['low_hours']=f"{lo:.4f}"; rows[i]['high_hours']=f"{hi:.4f}"
            repaired+=1

# Write back
out=os.path.join(OUT_DIR,"task2_summary_v4.csv")
with open(out,'w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
print(f"Done. Repaired {repaired} rows. Output: {out}")
