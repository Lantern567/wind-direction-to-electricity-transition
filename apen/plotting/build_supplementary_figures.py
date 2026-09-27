from pathlib import Path
import json
import numpy as np
import pandas as pd

from plot_config import DATA_ROOT, OUTPUT_ROOT, PLOT_INPUTS, save_600dpi
HERE = OUTPUT_ROOT
ROOT = DATA_ROOT
from figure_common import style, plt, BLUE, ORANGE, LIGHT, GREY, INK, Line2D

DATA = ROOT/'analysis_data'
style()
with np.load(PLOT_INPUTS/'wp4c_jensen.npz') as z:
    xy = z['xy']/198
    names = z['pid'].tolist()
fig, axes = plt.subplots(2,3,figsize=(7.9,5.25))
fig.subplots_adjust(left=.075,right=.97,top=.93,bottom=.105,wspace=.25,hspace=.30)
mins = []
extent = max(abs(xy).max(),1)+3
for i, ax in enumerate(axes.flat):
    point = xy[i]
    dist = np.linalg.norm(point[:,None]-point[None,:],axis=-1)
    np.fill_diagonal(dist,np.inf)
    mins.append(float(dist.min()))
    ax.scatter(point[:,0],point[:,1],s=13,c=BLUE,edgecolors='white',lw=.3)
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal')
    ax.spines[['top','right']].set_visible(False)
    ax.text(0,1.05,f'{names[i]}  ·  minimum {dist.min():.2f}D',transform=ax.transAxes,fontsize=8.2)
    ax.grid(color='#E8ECEE',lw=.4)
    ax.set_xticks([-40,0,40]);ax.set_yticks([-40,0,40])
    if i>=3:ax.set_xlabel('x / D')
    if i%3==0:ax.set_ylabel('y / D')
save_600dpi(fig, 'Figure_S1_600dpi')
plt.close(fig)

summary = pd.read_csv(DATA/'model_summary_screened.csv').query("scenario == 'baseline'")
sites = pd.read_csv(DATA/'mechanism_sites.csv')
fig, axes = plt.subplots(1,2,figsize=(7.9,3.25))
fig.subplots_adjust(left=.11,right=.97,bottom=.21,top=.86,wspace=.40)
a,b=axes
for j,scope in enumerate(['all','unflagged']):
    r = summary[(summary.screen==scope)&(summary.group=='all155')].iloc[0]
    a.errorbar(j,r.weighted_gain_pct,yerr=[[r.weighted_gain_pct-r.weather_year_bootstrap_lower_pct],
                 [r.weather_year_bootstrap_upper_pct-r.weighted_gain_pct]],fmt='o',ms=7,color=[BLUE,ORANGE][j],capsize=5,lw=1.5)
    a.text(j,r.weather_year_bootstrap_upper_pct+.02,f'{r.weighted_gain_pct:.3f}%',ha='center',fontsize=9)
a.set(xlim=(-.55,1.55),ylim=(0,.68),ylabel='Total generation gain (%)')
a.set_xticks([0,1],['All projects\nn = 155','No flagged site\nn = 139'])
for mask,col in [(~sites.coastal_group,LIGHT),(sites.coastal_group,ORANGE)]:
    b.scatter(sites.loc[mask,'spacing_D'],sites.loc[mask,'A_half_pct'],s=14,c=col,edgecolors='white',lw=.3)
flag=~sites.negative_rotor_free
b.scatter(sites.loc[flag,'spacing_D'],sites.loc[flag,'A_half_pct'],s=35,facecolors='none',edgecolors=INK,lw=.6)
b.set(xlabel='Nearest-neighbour spacing (D)',ylabel='Orientation sensitivity, A (%)',ylim=(-.8,27))
b.legend(handles=[Line2D([],[],marker='o',mfc='none',mec=INK,ls='',label='Flagged site')],fontsize=7.5,loc='upper right')
for ax,letter in zip(axes,['a','b']):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',lw=.5,color='#E8ECEE');ax.set_axisbelow(True)
    ax.text(-.20,1.09,letter,transform=ax.transAxes,fontsize=12,fontweight='bold')
save_600dpi(fig, 'Figure_S2_600dpi')
plt.close(fig)
(HERE/'supplementary_figure_checks.json').write_text(json.dumps({'layout_minimum_spacing_D':dict(zip(names,mins)),
    'flagged_sites':int(flag.sum()),'outputs':['figureS1_layouts.png','figureS2_screening.png']},indent=2))
print(dict(zip(names,mins)))
