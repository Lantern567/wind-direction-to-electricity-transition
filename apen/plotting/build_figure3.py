from pathlib import Path
from figure_common import *
HERE = OUTPUT_ROOT
INPUTS=[ROOT/'server/final_consistency/outputs/standard_baseline.npz', ROOT/'analysis_data/baseline_sites.csv',
 PLOT_INPUTS/'wp9c_farm_metrics.csv']
z=np.load(INPUTS[0]);ids=z['farm_ids'].astype(int)
f=pd.read_csv(INPUTS[2]).set_index('farm_id').loc[ids]
current = pd.read_csv(INPUTS[1]).set_index('farm_id').loc[ids]
assert f.index.is_unique and len(f)==171 and set(f.index)==set(ids)
A=z['A'][:,[0,1,3,4,5]].astype(float);mean=A.mean(1);real=current.A_half_pct.to_numpy();sel=f.is_corr.to_numpy(bool)
assert A.shape==(171,5) and sel.sum()==23
rank=pd.DataFrame(A).rank().corr().to_numpy()
rankmed=float(np.median(rank[np.triu_indices(5,1)]));rho=spearman(mean,real)
top=A>=np.quantile(A,.75,axis=0)
counts=(top&sel[:,None]).sum(0);share=counts/top.sum(0)*100
assert np.array_equal(top.sum(0),[43]*5)
meds=np.median(A[sel],axis=0)/np.median(A[~sel],axis=0)
sv=f['corr'].eq('Vietnam').to_numpy() & (ids!=56)
ch=f['corr'].eq('China strait').to_numpy()
region_medians=[float(np.median(mean[m])) for m in [sv,ch,~sel]]
assert sv.sum()==12 and ch.sum()==8

style();fig=plt.figure(figsize=(7.9,9.62));fig.text(.027,.972,'a',fontsize=12,fontweight='bold')
shapes=[read_shapes(p) for p in MAP_PATHS]
norm=LogNorm(.15,4.3)
cmap=LinearSegmentedColormap.from_list('regional_A',['#C6D8E1','#F1EAD9','#DA926C',ORANGE,'#8E3720'])
map_specs=[('Europe',[.085,.761,.29,.202],[-12,25,37,64],[0,10,20],[40,50,60]),
 ('US Atlantic',[.084,.677,.13,.064],[-73,-69,40,43],[-72,-70],[41,43]),
 ('East Asia',[.405,.677,.47,.29],[98,146,5,46],[100,120,140],[10,20,30,40])]
covered=np.zeros(len(f),bool);map_records=[]
for name,bounds,ext,xt,yt in map_specs:
    ax=axis(fig,bounds,grid=False);basemap(ax,ext,xt,yt,shapes)
    m=f.cent_lon.between(ext[0],ext[1]).to_numpy() & f.cent_lat.between(ext[2],ext[3]).to_numpy()
    covered|=m
    im=ax.scatter(f.cent_lon.to_numpy()[m],f.cent_lat.to_numpy()[m],c=mean[m],cmap=cmap,norm=norm,
                  s=22 if name!='US Atlantic' else 15,edgecolors='white',lw=.45,zorder=4)
    ax.text(.02,.99,f'{name} (n = {m.sum()})',transform=ax.transAxes,ha='left',va='top',fontsize=7.9)
    if name=='US Atlantic':ax.tick_params(labelsize=6.8)
    if name=='East Asia':
        for label,xy,xytext in [('Southern Vietnam',(106.4,9.2),(115,8)),('China coast',(119,25),(127,27))]:
            ax.annotate(label,xy=xy,xytext=xytext,color=ORANGE,fontsize=7.5,
                        arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.65})
    map_records.append({'region':name,'extent':ext,'n':int(m.sum())})
assert covered.all()
cax=fig.add_axes([.915,.697,.013,.254]);cb=fig.colorbar(im,cax=cax,ticks=[.2,.5,1,2,4])
cb.ax.set_yticklabels(['0.2','0.5','1','2','4']);cb.ax.minorticks_off();cb.outline.set_linewidth(.55)
cb.set_label('Mean A across five geometries (%)',fontsize=8.1,labelpad=5)
# a inset: the same regional medians, encoded by zero-based column height.
fig.text(.244,.742,'Regional medians',fontsize=7.7,color=INK)
am=axis(fig,[.244,.684,.150,.048],grid=False)
am.bar(range(3),region_medians,width=.56,color=[ORANGE,ORANGE,BLUE],zorder=3)
for j,value in enumerate(region_medians):
    am.text(j,value+.18,f'{value:.2f}%',ha='center',va='bottom',fontsize=6.9,color=INK)
am.set(xlim=(-.65,2.65),ylim=(0,5.0))
am.set_xticks(range(3),['S.\nVietnam','China\ncoast','Other\nsites'])
am.set_yticks([])
am.tick_params(axis='x',length=0,pad=3,labelsize=6.6)
am.spines['left'].set_visible(False)
am.spines['bottom'].set_color(GREY)
am.spines['bottom'].set_linewidth(.55)

# b: all points, group medians and interquartile ranges on a stated log scale.
ab=axis(fig,[.100,.455,.825,.153],'b');rng=np.random.default_rng(20260910)
for m,shift,col,label in [(~sel,.14,LIGHT,'Other sites (n = 148)'),(sel,-.14,ORANGE,'Coastal comparison group (n = 23)')]:
    for j in range(5):
        vals=A[m,j];pos=j+shift
        ab.scatter(pos+rng.uniform(-.065,.065,len(vals)),vals,s=3.8,c=col,alpha=.25,lw=0,zorder=2)
        q1,med,q3=np.quantile(vals,[.25,.5,.75])
        ab.plot([pos,pos],[q1,q3],c=col,lw=2.2,zorder=4)
        ab.scatter(pos,med,s=24,c=col,edgecolors='white',lw=.45,zorder=5,label=label if j==0 else None)
for j,value in enumerate(meds):ab.text(j,8.0,f'{value:.1f}×',ha='center',fontsize=8,color=ORANGE)
ab.set_yscale('log');ab.set(ylim=(.045,12),xlim=(-.5,4.5),ylabel='Orientation sensitivity, A (%)')
ab.set_xticks(range(5),[f'{n}\n{s:.1f}D' for n,s in zip(NAMES,SPACING)])
ab.set_yticks([.1,1,10],['0.1','1','10']);ab.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ab.legend(loc='lower center',bbox_to_anchor=(.53,1.08),ncol=2,handletextpad=.1,columnspacing=1.2,fontsize=7.9)

# c: lower triangle removes redundant cells while retaining every distinct pair.
ac=axis(fig,[.095,.235,.322,.158],'c',False)
cm=LinearSegmentedColormap.from_list('ranks',['#F3F6F7','#99BDB9',BLUE]);cm.set_bad('white')
masked=np.ma.masked_where(np.triu(np.ones((5,5),dtype=bool)),rank)
rank_im=ac.imshow(masked,vmin=.9,vmax=1,cmap=cm,aspect='equal')
for i in range(5):
    for j in range(i):ac.text(j,i,f'{rank[i,j]:.2f}',ha='center',va='center',fontsize=7.2,color='white' if rank[i,j]>.975 else INK)
ac.set_xticks(range(4),range(1,5));ac.set_yticks(range(1,5),range(2,6));ac.tick_params(length=0)
ac.set_xticks(np.arange(-.5,5,1),minor=True);ac.set_yticks(np.arange(-.5,5,1),minor=True)
ac.grid(which='minor',color='white',lw=2.3);ac.tick_params(which='minor',length=0)
ac.set(xlabel='Layout',ylabel='Layout');ac.spines[:].set_visible(False)
ac.text(.73,.88,'Median ρ',transform=ac.transAxes,ha='center',fontsize=7.5,color=GREY)
ac.text(.73,.72,f'{rankmed:.2f}',transform=ac.transAxes,ha='center',fontsize=14,color=BLUE)
rank_cax=fig.add_axes([.425,.247,.009,.139]);rank_cb=fig.colorbar(rank_im,cax=rank_cax,ticks=[.9,.95,1])
rank_cb.ax.set_yticklabels(['0.90','0.95','1.00']);rank_cb.ax.tick_params(labelsize=7,length=2)
rank_cb.outline.set_visible(False);rank_cb.set_label('Spearman ρ',fontsize=7.4,labelpad=3)

# d: the 23-member group's composition within each 43-site upper quartile.
ad=axis(fig,[.585,.235,.34,.158],'d');x=np.arange(5)
ad.vlines(x,0,share,color='#E7BCA5',lw=5,zorder=2)
ad.scatter(x,share,c=ORANGE,s=30,zorder=3)
for j,(v,n) in enumerate(zip(share,counts)):ad.text(j,v+2,f'{n}/43',ha='center',fontsize=7.1,color=ORANGE)
ad.axhline(sel.mean()*100,c=GREY,ls='--',lw=.8)
ad.text(4.35,sel.mean()*100+1.4,'23/171 = 13.5%',ha='right',fontsize=7,color=GREY)
ad.set(xlim=(-.45,4.5),ylim=(0,57),xlabel='Layout',ylabel='Coastal-group share\nof top-quartile sites (%)')
ad.set_xticks(x,range(1,6));ad.set_yticks([0,20,40])

# e: retain every site with independent linear axes and a true 1:1 reference.
ae=axis(fig,[.100,.047,.355,.135],'e')
for m,col in [(~sel,LIGHT),(sel,ORANGE)]:ae.scatter(mean[m],real[m],s=15,c=col,alpha=.85,edgecolors='white',lw=.35,zorder=3)
ae.plot([0,4.4],[0,4.4],ls='--',c=GREY,lw=.7)
ae.text(.03,.96,f'ρ = {rho:.3f}; n = 171',transform=ae.transAxes,fontsize=7.8,va='top')
ae.set(xlim=(-.05,4.4),ylim=(-.5,27),xlabel='Mean A across five geometries (%)',ylabel='Real-layout A (%)')
ae.set_xticks([0,1,2,3,4]);ae.set_yticks([0,10,20])

# f: direct cross-layout persistence by geography; predictive results remain in SI.
af=axis(fig,[.695,.061,.230,.121],grid=False);fig.text(.515,.191,'f',fontsize=12,fontweight='bold')
regional_rates=[]
for j,(mask,name,col) in enumerate([(sv,'Southern Vietnam',ORANGE),(ch,'China coast',ORANGE),
        (sel&~sv&~ch,'Beibu + Europe',ORANGE),(~sel,'Other sites',LIGHT)]):
    count=int((top.all(1)&mask).sum());n=int(mask.sum());rate=100*count/n
    af.barh(j,rate,height=.45,color=col,alpha=.85)
    af.text(rate+3,j,f'{count}/{n}',va='center',fontsize=7.1,color=INK)
    regional_rates.append({'group':name,'n':n,'all_five_top_quartile_n':count,'pct':rate})
af.set_yticks(range(4),[x['group'] for x in regional_rates]);af.tick_params(axis='y',length=0,labelsize=7,pad=5)
af.spines['left'].set_visible(False)
af.set(xlim=(0,126),ylim=(3.6,-.6),xlabel='Sites in the top quartile\nin all five geometries (%)')
af.set_xticks([0,50,100]);af.tick_params(axis='x',labelsize=7.4)
pd.DataFrame(regional_rates).to_csv(HERE/'figure3_regional_persistence.csv',index=False)
save_600dpi(fig, 'Figure_3_600dpi')
fig.savefig(HERE/'figure3_preview.png',dpi=170,facecolor='white')
plt.close(fig)
f.assign(A_five_mean=mean,selected_region=sel).to_csv(HERE/'figure3_site_data.csv',encoding='utf-8-sig')
pd.DataFrame(rank,index=PIDS,columns=PIDS).to_csv(HERE/'figure3_rank_correlations.csv')
pd.DataFrame({'layout':PIDS,'n_top':top.sum(0),'n_selected_top':counts,'selected_share_pct':share,'median_ratio':meds}).to_csv(HERE/'figure3_layout_summary.csv',index=False)
meta={'inputs':hashes(INPUTS),'map_inputs':hashes(MAP_PATHS),'n_sites':171,'n_selected':23,
      'rho_real':rho,'median_rank_correlation':rankmed,'regional_medians':region_medians,
      'all_sites_on_maps':bool(covered.all()),'map_windows':map_records,
      'projection':'Plate Carree with cached Natural Earth 50m land and borders; existing site locations only',
      'intervals':'Panel b: within-group 25th-75th percentiles, not confidence intervals; all site dots retained',
      'panel_mapping':{'old a':'b','old b':'c','old c':'d','old d':'e','old e':'a','old f':'SI only; new f = direct regional persistence'},
      'no_wake_simulation':True,'visual_qa':'pending'}
(HERE/'figure3_validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in meta.items() if k not in ['inputs','map_inputs']},ensure_ascii=False))
