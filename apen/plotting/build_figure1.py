from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

from plot_config import DATA_ROOT, OUTPUT_ROOT, PLOT_INPUTS, MAP_ROOT, save_600dpi
ROOT = DATA_ROOT
OUT = OUTPUT_ROOT
import shapefile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

BLUE, ORANGE, PURPLE, TEAL = '#244B68', '#C45D32', '#7C658C', '#7FA69E'
INK, GRAY, LIGHT = '#24282D', '#9AA4AC', '#E6E9EB'
CASE_COLORS = {57: ORANGE, 155: '#B38A3D', 157: '#4E817C', 66: BLUE, 91: PURPLE}
CASE_NAMES = {57:'F57 · Vietnam coast', 155:'F155 · Gulf of Taranto',
              157:'F157 · Danish straits', 66:'F66 · Hangzhou Bay',
              91:'F91 · Pearl River mouth'}
INPUTS = {
 'annual_gains': ROOT/'server/final_consistency/outputs/historical_annual.csv',
 'site_summary': ROOT/'server/final_consistency/outputs/historical_sites.csv',
 'farm_metadata': PLOT_INPUTS/'wp9c_farm_metrics.csv',
}
og = pd.read_csv(INPUTS['annual_gains'])
assert not og.duplicated(['farm_id','year']).any()
meta = pd.read_csv(INPUTS['farm_metadata']).set_index('farm_id', verify_integrity=True)
farm = og.groupby('farm_id').agg(G=('gain_pct','mean'), years=('year','count'))
farm['positive_years_pct'] = og.assign(positive=og.gain_pct > 1e-8).groupby('farm_id').positive.mean()*100
farm = farm.join(meta[['cent_lon','cent_lat','is_corr','n_turb']], validate='one_to_one')
order = farm.sort_values('G',ascending=False).index
farm['rank'] = pd.Series(np.arange(1,len(order)+1),index=order)
cases = list(order[:5])
FLAGGED_CASES = {57, 155}

def case_label(fid):
    return f'F{fid}' + ('†' if fid in FLAGGED_CASES else '')

farm['top5'] = farm.index.isin(cases)
CASE_COLORS = dict(zip(cases,[ORANGE,'#B38A3D',BLUE,'#4E817C',PURPLE]))
assert len(og)==1203 and len(farm)==171
assert int((farm.years>=3).sum())==146 and int((farm.years>=5).sum())==108
gp = og.pivot(index='year',columns='farm_id',values='gain_pct').reindex(columns=order)
assert gp.shape == (11,171) and gp.notna().sum().sum()==1203

paired_gain = 100*(og.selected_GWh/og.baseline_GWh-1)
gain_difference = float(np.max(abs(paired_gain-og.gain_pct)))
assert gain_difference < 1e-8
common = pd.read_csv(INPUTS['site_summary']).set_index('farm_id')
de = common['fixed_geometry_annual_delta_GWh']
positive = de.clip(lower=0)
negative_share = -de.clip(upper=0).sum()/positive.sum()*100
farm['n_turb'] = common.n_turb
capacity = common.reference_capacity_GW
farm['common_period_annual_delta_GWh'] = de
farm['reference_capacity_GW'] = capacity
cumulative = pd.DataFrame({'farm_id':order, 'rank':np.arange(1,172),
 'capacity_pct':capacity.loc[order].cumsum().to_numpy()/capacity.sum()*100,
 'energy_pct':positive.loc[order].cumsum().to_numpy()/positive.sum()*100})
x5, y5 = cumulative.iloc[4][['capacity_pct','energy_pct']]

# Read cached Natural Earth shapefiles. These are lon/lat coordinates, plotted
# directly in the same Plate Carree projection as the original figure.
MAP_FILES = {'land':MAP_ROOT/'physical/ne_50m_land.shp',
             'borders':MAP_ROOT/'cultural/ne_50m_admin_0_boundary_lines_land.shp'}
def read_shapes(path):
    records=[]
    with shapefile.Reader(str(path)) as reader:
        for shape in reader.iterShapes():
            points=np.asarray(shape.points)
            starts=list(shape.parts)+[len(points)]
            records.append((shape.bbox,[points[a:b] for a,b in zip(starts[:-1],starts[1:])]))
    return records
land, borders = read_shapes(MAP_FILES['land']), read_shapes(MAP_FILES['borders'])

plt.rcParams.update({'font.family':'Arial','font.size':9.2,'axes.labelsize':9.2,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.2,
 'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',
 'mathtext.bf':'Arial:bold'})
fig = plt.figure(figsize=(7.9,9.62))
AXES=[]
def axis(bounds, letter=None):
    ax=fig.add_axes(bounds); AXES.append(ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(pad=3)
    if letter:
        fig.text(bounds[0]-.070,bounds[1]+bounds[3]+.010,letter,fontsize=12,weight='bold')
    return ax
def note(ax,text,xy,xytext,color=INK,align='left'):
    return ax.annotate(text,xy=xy,xytext=xytext,color=color,fontsize=8.5,
        ha=align,va='center',arrowprops={'arrowstyle':'-','color':color,'lw':.65},
        annotation_clip=False)

max_gain=max(22,float(np.ceil(og.gain_pct.max())))
min_gain=min(-4,float(np.floor(og.gain_pct.min())))
norm=SymLogNorm(linthresh=.5,linscale=1,vmin=min_gain,vmax=max_gain,base=10)
cmap=LinearSegmentedColormap.from_list('gain',[
 (0,BLUE),(float(norm(-1)),'#A6C3CF'),(float(norm(0)),'#FAF6EF'),
 (float(norm(1)),'#E1B388'),(1,'#A7442B')])
cmap.set_bad('#D8DDE1')

# a: the same ranked means with the full, aligned farm-year evidence below.
aa=axis([.105,.854,.775,.115],'a')
ranked=farm.loc[order]
aa.plot(ranked['rank'],ranked.G,color='#BAC4CA',lw=.65,zorder=1)
aa.scatter(ranked['rank'],ranked.G,c=ranked.G,cmap=cmap,norm=norm,
           s=13,edgecolors='white',linewidths=.3,zorder=3)
aa.scatter(farm.loc[cases,'rank'],farm.loc[cases,'G'],facecolors='none',
           edgecolors=ORANGE,s=30,lw=.8,zorder=4)
aa.set(xlim=(.5,171.5),ylim=(min(-2.0,farm.G.min()*1.1),max(23,farm.G.max()*1.2)),yscale='symlog',ylabel='Mean gain, G (%)')
aa.set_yscale('symlog',linthresh=1)
aa.set_yticks([-1,0,1,5,20]); aa.yaxis.set_major_formatter(ScalarFormatter())
aa.tick_params(axis='x',bottom=False,labelbottom=False)
aa.axhline(0,color=GRAY,lw=.45,zorder=0)
aa.axhline(farm.G.median(),color=ORANGE,lw=.7,ls='--',zorder=0)
aa.axhline(farm.G.mean(),color=BLUE,lw=.65,ls=':',zorder=0)
aa.text(.43,.89,f'Median {farm.G.median():.2f}%   ·   Mean {farm.G.mean():.2f}%',transform=aa.transAxes,fontsize=8.7,va='top')
note(aa,case_label(cases[0]),(1,farm.loc[cases[0],'G']),(12,farm.loc[cases[0],'G']*.90),ORANGE)
note(aa,case_label(cases[1]),(2,farm.loc[cases[1],'G']),(22,farm.loc[cases[1],'G']*.88),ORANGE)
ah=axis([.105,.712,.775,.112])
im=ah.imshow(np.ma.masked_invalid(gp.to_numpy()),aspect='auto',interpolation='none',
    extent=(.5,171.5,2024.5,2013.5),cmap=cmap,norm=norm,rasterized=True)
ah.set(xlim=(.5,171.5),ylabel='Year',xlabel='Farms ranked by multi-year mean gain (n = 171)')
ah.set_yticks([2014,2016,2018,2020,2022,2024]); ah.set_xticks([1,30,60,90,120,150,171])
ah.spines[['top','right']].set_visible(True)
for spine in ah.spines.values(): spine.set_linewidth(.4); spine.set_edgecolor('#BAC4CA')
cax=fig.add_axes([.905,.712,.012,.257])
cb=fig.colorbar(im,cax=cax,ticks=[-4,-1,0,1,5,22]); cb.ax.minorticks_off()
cb.set_ticklabels(['−4','−1','0','1','5','22'])
cb.set_label('Gain (%)',labelpad=4,fontsize=9)
cb.outline.set_linewidth(.45)
fig.legend([Patch(facecolor='#D8DDE1')],['No record'],loc='center',bbox_to_anchor=(.827,.682),
           handlelength=1.0,fontsize=8.1)

# 年度持续性与多年收益共同展示，记录较短的场址采用空心点。
ab=axis([.105,.458,.35,.183],'b')
for mask,face in [(farm.years>=5,'#9AB1C1'),(farm.years<5,'white')]:
    ab.scatter(farm.loc[mask,'G'],farm.loc[mask,'positive_years_pct'],s=17,facecolors=face,
               edgecolors='#6D899A',lw=.5,alpha=.8,zorder=2)
ab.scatter(farm.loc[cases,'G'],farm.loc[cases,'positive_years_pct'],s=32,facecolors='none',edgecolors=ORANGE,lw=1,zorder=4)
ab.set_xscale('symlog',linthresh=1)
ab.set(xlim=(min(-1.5,farm.G.min()-.2),max(22,farm.G.max()*1.15)),ylim=(-4,111),xlabel='Mean gain, G (%)',ylabel='Years with positive gain (%)')
ab.set_xticks([-1,0,1,5,20]);ab.xaxis.set_major_formatter(ScalarFormatter());ab.set_yticks([0,25,50,75,100])
ab.legend([Line2D([],[],marker='o',ls='',mfc='#9AB1C1',mec='#6D899A',ms=4),
           Line2D([],[],marker='o',ls='',mfc='white',mec='#6D899A',ms=4)],
          [f'≥5 yr (n = {(farm.years>=5).sum()})',f'<5 yr (n = {(farm.years<5).sum()})'],
          loc='lower right',fontsize=7.8,handletextpad=.3)

# c: preserve the cumulative energy definition and all negative contributions.
ac=axis([.59,.458,.335,.183],'c')
xx=np.r_[0,cumulative.capacity_pct.to_numpy()]; yy=np.r_[0,cumulative.energy_pct.to_numpy()]
ac.plot([0,100],[0,100],color=GRAY,lw=.75,ls='--')
ac.fill_between(xx,xx,yy,color='#C8D9DF',alpha=.65,lw=0)
ac.plot(xx,yy,color=BLUE,lw=1.35)
ac.scatter(cumulative.capacity_pct[:5],cumulative.energy_pct[:5],s=24,color=ORANGE,
           edgecolors='white',lw=.5,zorder=4)
ac.set(xlim=(0,101),ylim=(0,max(103,float(yy.max())+2)),
       xlabel='Cumulative reference capacity (%)',ylabel='Cumulative positive\nadded energy (%)')
ac.set_xticks([0,25,50,75,100]); ac.set_yticks([0,25,50,75,100])
note(ac,f'Top 5 sites\n{x5:.1f}% capacity\n{y5:.1f}% added energy',
     (x5,y5),(31,24),ORANGE)
ac.text(73,67,'1:1',color=GRAY,fontsize=8.3,rotation=42)
ac.text(.03,.98,f'Common 2014–2024 weather\nNegative changes: {negative_share:.1f}% of positive total',
        transform=ac.transAxes,va='top',fontsize=6.5,color=GRAY,
        bbox={'facecolor':'white','edgecolor':'none','pad':1.5,'alpha':.9})

# d/e: two original geographical windows, now with separate panel letters.
extents={}
def draw_map(bounds,letter,select,xticks,yticks):
    ax=axis(bounds,letter)
    chosen=farm[select]
    dy=max(chosen.cent_lat.max()-chosen.cent_lat.min()+5,
           (chosen.cent_lon.max()-chosen.cent_lon.min()+5)/1.2985)
    dx=min(dy*1.2985,45)
    cx=(chosen.cent_lon.max()+chosen.cent_lon.min())/2
    cy=(chosen.cent_lat.max()+chosen.cent_lat.min())/2
    xmin,xmax,ymin,ymax=(cx-dx/2,cx+dx/2,cy-dy/2,cy+dy/2)
    extents[letter]=[xmin,xmax,ymin,ymax]
    for bbox,parts in land:
        if bbox[2]<xmin or bbox[0]>xmax or bbox[3]<ymin or bbox[1]>ymax: continue
        vertices=[]; codes=[]
        for part in parts:
            vertices.extend(part); codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(part)-2)+[MplPath.CLOSEPOLY])
        patch=PathPatch(MplPath(np.asarray(vertices),codes),facecolor='#E9ECEE',
                        edgecolor='#AAB2B7',lw=.35,zorder=0)
        ax.add_patch(patch)
    for bbox,parts in borders:
        if bbox[2]<xmin or bbox[0]>xmax or bbox[3]<ymin or bbox[1]>ymax: continue
        for part in parts: ax.plot(part[:,0],part[:,1],color='#BAC1C6',lw=.3,zorder=1)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal')
    ax.set_xticks([x for x in xticks if xmin<x<xmax]); ax.set_yticks([y for y in yticks if ymin<y<ymax])
    ax.set_xticklabels([f'{abs(x):g}°'+('E' if x>0 else 'W' if x<0 else '') for x in ax.get_xticks()])
    ax.set_yticklabels([f'{y:g}°N' for y in ax.get_yticks()])
    ax.grid(color='#D8DEE2',lw=.35,ls=':',zorder=-1)
    ax.spines[['top','right','bottom','left']].set_visible(False)
    ax.tick_params(length=0)
    # Keep the original three groups; a fixed marker size avoids an unexplained
    # second magnitude encoding. Stars identify precisely the same five sites.
    for mask,col,size in [(select & ~farm.is_corr & ~farm.top5,'#91A2AD',14),
                          (select & farm.is_corr & ~farm.top5,TEAL,22)]:
        ax.scatter(farm.loc[mask,'cent_lon'],farm.loc[mask,'cent_lat'],s=size,color=col,
                   edgecolors='white',lw=.3,zorder=4,alpha=.95)
    for fid in farm[select & farm.top5].index:
        r=farm.loc[fid]
        ax.scatter(r.cent_lon,r.cent_lat,s=100,marker='*',color=ORANGE,edgecolors='white',lw=.6,zorder=7)
    return ax

east=farm.cent_lon.between(100,130)&farm.cent_lat.between(5,38)
europe=farm.cent_lon.between(-6,24)&farm.cent_lat.between(36,62)
ad=draw_map([.105,.239,.35,.181],'d',east,np.arange(95,136,10),np.arange(10,39,10))
ae=draw_map([.59,.239,.335,.181],'e',europe,np.arange(-10,32,10),np.arange(40,61,10))
offsets={57:(110,10),66:(123.6,29.5),91:(115.6,20.8),155:(19.2,39),157:(15.3,55.5)}
for fid in cases:
    r=farm.loc[fid]
    if east.loc[fid] or europe.loc[fid]:
        ax=ad if east.loc[fid] else ae
        xytext=offsets.get(fid,(r.cent_lon+3,r.cent_lat+1))
        note(ax,case_label(fid),(r.cent_lon,r.cent_lat),xytext,ORANGE)
fig.legend([Line2D([],[],marker='*',ls='',mfc=ORANGE,mec='white',ms=9),
            Line2D([],[],marker='o',ls='',mfc=TEAL,mec='white',ms=5),
            Line2D([],[],marker='o',ls='',mfc='#91A2AD',mec='white',ms=5)],
           ['Top five mean gains','Other regional-group site','Other site'],
           loc='center',bbox_to_anchor=(.53,.216),ncol=3,columnspacing=1.6,
           handletextpad=.35,fontsize=8.1)

# f: aligned small multiples preserve the annual series without crossing labels.
p50=gp.median(axis=1); p25=gp.quantile(.25,axis=1); p75=gp.quantile(.75,axis=1)
fig.text(.035,.192,'f',fontsize=12,fontweight='bold')
short={57:'Southern Vietnam',155:'Gulf of Taranto',66:'Hangzhou Bay',157:'Danish straits',91:'Pearl River mouth'}
for j,fid in enumerate(cases):
    af=axis([.105+j*.170,.051,.139,.119])
    af.fill_between(gp.index,p25,p75,color='#D8DEE3',alpha=.9,lw=0)
    af.plot(gp.index,p50,color='#8998A4',lw=.85)
    series=gp[fid].dropna()
    af.plot(series.index,series.values,color=CASE_COLORS[fid],lw=1.35,
            ls='--' if farm.loc[fid,'years']<5 else '-',marker='o',ms=2.6,
            mfc='white',mec=CASE_COLORS[fid],mew=.65,zorder=4)
    af.set(xlim=(2013.3,2024.7),ylim=(min(-.5,gp[cases].min().min()-1),max(24,gp[cases].max().max()+2)))
    af.set_xticks([2014,2024]);af.set_yticks([0,10,20]);af.tick_params(labelsize=7.1)
    af.grid(axis='y',color='#E8ECEE',lw=.5)
    if j==0:af.set_ylabel('Annual gain (%)',fontsize=8.5)
    else:af.tick_params(labelleft=False);af.spines['left'].set_visible(False)
    af.text(.5,1.15,case_label(fid),ha='center',va='bottom',transform=af.transAxes,fontsize=8.2,color=CASE_COLORS[fid],fontweight='bold')
    af.text(.5,1.035,short.get(fid,meta.loc[fid,'country']),ha='center',va='bottom',transform=af.transAxes,fontsize=6.8)
fig.text(.52,.015,'Year',ha='center',fontsize=8.5)
fig.legend([Patch(facecolor='#D8DEE3'),Line2D([],[],color='#8998A4',lw=1.1)],
           ['All-site annual IQR','Median'],loc='center',bbox_to_anchor=(.78,.015),
           ncol=2,handlelength=1.1,columnspacing=1,fontsize=7.4)

# Save source tables and a deterministic validation record alongside the figure.
farm.sort_values('rank').to_csv(OUT/'figure1_farm_data.csv',index=True)
gp.to_csv(OUT/'figure1_annual_gain_matrix.csv',index=True)
cumulative.to_csv(OUT/'figure1_cumulative_data.csv',index=False)
record={
 'input_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS.values()},
 'map_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in MAP_FILES.values()},
 'n_sites':len(farm),'n_site_years':len(og),'missing_heatmap_cells':int(gp.isna().sum().sum()),
 'mean_G':float(farm.G.mean()),'median_G':float(farm.G.median()),
 'positive_site_years_pct':float((og.gain_pct>1e-8).mean()*100),
 'top5':list(map(int,cases)),
 'dagger_marked_sites': sorted(FLAGGED_CASES),
 'dagger_scope':'Baseline-model rotor-node audit of terminal geometry; not a year-specific classification',
 'max_gain_difference_from_paired_energy':gain_difference,
 'top5_capacity_pct':float(x5),'top5_added_energy_pct':float(y5),
 'cumulative_energy_max_pct':float(yy.max()),
 'heatmap_range':[float(og.gain_pct.min()),float(og.gain_pct.max())],
 'heatmap_colour_limits':[min_gain,max_gain],'heatmap_linear_threshold_pct':.5,
 'maps_projection':'Plate Carree, same extent algorithm and 50m Natural Earth as original',
 'map_extents':extents,'new_analyses':'Panel c uses common-period annual energy and terminal geometry; positive and negative contributions separated',
 'negative_to_positive_energy_pct':float(negative_share),
 'panel_mapping':{'old a':'new a upper','old b':'new b','old c east Asia':'new d',
                  'old c Europe':'new e','old d':'new f','old e':'new c'},
 'visual_qa':'pending'}
save_600dpi(fig, 'Figure_1_600dpi')
fig.savefig(OUT/'figure1_preview.png',dpi=170)
(OUT/'figure1_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
plt.close(fig)
