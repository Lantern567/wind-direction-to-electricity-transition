\"\"\"朝向长尾效应地理分析 — Fig 7 生成。
数据: orientation_gain_all_farms.csv (171场) + S1旋转扫描 + task1_wind_metrics
输出: fig7_orientation_long_tail.png
\"\"\"
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib import colors

plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

# ---- data ----
BASE = r'd:\01学习资料\wind-direction-to-electricity-transition\task3'
df = pd.read_csv(f'{BASE}/output/orientation_gain_all_farms.csv')
s1 = pd.read_csv(f'{BASE}/task3_s1_optimal_orientation.csv')
wm = pd.read_csv(f'{BASE}/../data/task1_output/task1_wind_metrics.csv')
high = df[df['mean_gain']>5]
IDS = [57,155,66,157,91,126,159]

# ---- figure ----
fig = plt.figure(figsize=(22,11))

# (a) global map
ax = fig.add_axes([0.02,0.42,0.70,0.56], projection=ccrs.PlateCarree())
ax.set_extent([-80,145,5,62]); ax.add_feature(cfeature.LAND,facecolor='#f0f0f0')
ax.add_feature(cfeature.OCEAN,facecolor='#e8f0f8')
ax.add_feature(cfeature.COASTLINE,linewidth=0.4,edgecolor='#888')
ax.add_feature(cfeature.BORDERS,linewidth=0.2,edgecolor='#bbb')
norm = colors.SymLogNorm(linthresh=1,linscale=1,vmin=-0.5,vmax=20)
sc = ax.scatter(df['lon'],df['lat'],c=df['mean_gain'],cmap='RdYlGn_r',norm=norm,
    s=np.clip(df['n_turb']*2,10,400),alpha=0.7,edgecolors='white',linewidth=0.3,
    transform=ccrs.PlateCarree(),zorder=2)
ax.scatter(high['lon'],high['lat'],s=np.clip(high['n_turb']*3,50,500),
    facecolors='none',edgecolors='red',linewidth=2,transform=ccrs.PlateCarree(),zorder=3)
for _,r in high.iterrows():
    off = 14 if r['farm_id']!=57 else -16
    ax.annotate(f'F{int(r[\"farm_id\"])}\n{r[\"mean_gain\"]:+.1f}%',(r['lon'],r['lat']),
        xytext=(8,off),textcoords='offset points',fontsize=7,fontweight='bold',color='darkred',
        bbox=dict(boxstyle='round,pad=0.15',facecolor='white',alpha=0.85),
        transform=ccrs.PlateCarree(),zorder=4)
ax.set_title('(a) 朝向优化 AEP 增益全球分布',fontsize=13,fontweight='bold')

# (b) histogram
ax = fig.add_axes([0.76,0.42,0.22,0.56])
ax.hist(df['mean_gain'],bins=50,color='#e74c3c',alpha=0.7,edgecolor='white')
ax.axvline(x=5,color='red',linestyle='--',lw=2,label='>5% (7场)')
ax.axvline(x=df['mean_gain'].median(),color='darkblue',linestyle='-',lw=1.5,
    label=f'中位 {df[\"mean_gain\"].median():.2f}%')
ax.set_xlabel('朝向优化增益(%)'); ax.set_ylabel('风场数')
ax.set_title('(b) 长尾分布 (n=171)',fontsize=12,fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)

# (c) 7 sensitivity curves
for i,fid in enumerate(IDS):
    ax = fig.add_axes([0.02+i*0.095,0.05,0.088,0.32])
    sub = s1[s1['farm_id']==fid].sort_values('angle_deg')
    aep = sub['expected_AEP_kWh'].values
    aep_n = (aep-aep.min())/(aep.max()-aep.min())*100
    ax.plot(sub['angle_deg'].values,aep_n,'o-',color='#e74c3c',lw=1.8,ms=4)
    ax.axvline(x=sub['angle_deg'].values[aep.argmax()],color='darkred',linestyle='--',lw=1)
    wci = wm[(wm['farm_id']==fid)&(wm['year']==2024)]['WCI_yearly'].iloc[0]
    country = s1[s1['farm_id']==fid]['country'].iloc[0]
    gain = high[high['farm_id']==fid]['mean_gain'].iloc[0]
    ax.set_title(f'F{fid}({country[:6]}) {gain:+.1f}%\nWCI={wci:.2f}',fontsize=7)
    ax.set_xlim(0,170); ax.set_ylim(0,108); ax.set_xticks([0,85,170])
    ax.tick_params(labelsize=6); ax.grid(alpha=0.3)

fig.suptitle('Fig 7: 朝向优化价值的长尾分布与地理归因',fontsize=16,fontweight='bold',y=0.99)
fig.savefig(f'{BASE}/output/fig7_orientation_long_tail.png',dpi=250,bbox_inches='tight')
print('Saved: fig7_orientation_long_tail.png')
