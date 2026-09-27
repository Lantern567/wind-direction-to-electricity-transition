from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

from plot_config import DATA_ROOT, OUTPUT_ROOT, save_600dpi
HERE = OUTPUT_ROOT
ROOT = DATA_ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

DATA = ROOT / 'analysis_data'
OLD_MODEL = ROOT / 'server/outputs'
INPUTS = [DATA/'mechanism_sites.csv', ROOT/'server/final_consistency/outputs/standard_baseline.npz',
          DATA/'concentration_correlations.csv', OLD_MODEL/'models/baseline/F066_5deg.npz']
df = pd.read_csv(INPUTS[0]).set_index('farm_id').sort_index()
cross = np.load(INPUTS[1])
ids = df.index.to_numpy(dtype=int)
assert df.index.is_unique and len(ids) == 171
np.testing.assert_array_equal(cross['farm_ids'], ids)
A = df.A_half_pct.to_numpy()
theta = df.pooled_half_peak_deg.to_numpy()
S = df.spacing_D.to_numpy()
group = df.coastal_group.to_numpy(bool)
high = A > 5.2
delta = df.phase_distance_deg.to_numpy()
Gdiag = df.Gdiag_pct.to_numpy()
ratio = df.Gdiag_over_A_pct.to_numpy()
independent = [0,1,3,4,5]
cmeans = cross['A_c'][:,independent].astype(float).mean(axis=(0,1))
drop = 100*(1-cmeans[1]/cmeans[0])
def spearman(x,y):
    return float(np.corrcoef(pd.Series(x).rank(),pd.Series(y).rank())[0,1])
project = df.n_turb >= 10
r_spacing, r_phase = spearman(S,A), spearman(delta[high],ratio[high])
r_gain = spearman(A[project], df.loc[project, 'S1_gain_pct'])
assert group.sum() == 23 and project.sum() == 155
stats = pd.read_csv(INPUTS[2])
panel = pd.DataFrame({'farm_id':ids,'A_halfcircle_pct':A,'spacing_D':S,'corridor_group':group,
                     'high_response':high,'pooled_peak_deg':theta,'phase_distance_deg':delta,
                     'Gdiag_pct':Gdiag,'100Gdiag_over_A_pct':ratio,'S1_gain_pct':df.S1_gain_pct.to_numpy()})
f66_model = np.load(INPUTS[3])
p0 = np.load(ROOT/'server/inputs/legacy_eta.npz')['P0']
all_ws, wd = f66_model['ws'].astype(float), f66_model['wd'].astype(float)
valid_power = p0 >= 1e-3
ws = all_ws[valid_power]
efficiency = f66_model['power_W'][valid_power]/(p0[valid_power, None]*int(f66_model['n_turb']))*100
i9 = np.flatnonzero(np.isclose(ws,9))[0]
eta9 = efficiency[i9]
assert np.isfinite(efficiency).all() and efficiency.min() >= 0 and efficiency.max() <= 100

BLUE = '#244B68'; ORANGE = '#C45D32'; PURPLE = '#7C658C'; TEAL = '#7FA69E'
LIGHT = '#A4BAC8'; INK = '#252C32'; GREY = '#8B99A5'; GRID = '#E8ECEE'
plt.rcParams.update({'font.family':'Arial','font.size':8.8,'axes.labelsize':8.8,
    'xtick.labelsize':8.1,'ytick.labelsize':8.1,'axes.linewidth':.7,'axes.edgecolor':INK,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'legend.frameon':False,'legend.fontsize':7.9,'pdf.fonttype':42,'ps.fonttype':42,
    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
fig = plt.figure(figsize=(7.9,9.62))
def axis(bounds, grid=True):
    ax = fig.add_axes(bounds)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.8,width=.65,pad=3)
    if grid: ax.grid(axis='y',color=GRID,lw=.55,zorder=0)
    ax.set_axisbelow(True)
    return ax
def label(letter,x,y):
    fig.text(x,y,letter,fontsize=12,fontweight='bold',va='bottom',ha='left')
def scatter(ax,x,y,mask=None,marker='o'):
    if mask is None: mask = np.ones(len(ids),dtype=bool)
    for sel,col,s in [(~group,LIGHT,15),(group,ORANGE,23)]:
        keep = sel & mask
        ax.scatter(x[keep],y[keep],s=s,c=col,marker=marker,alpha=.88,
                   edgecolor='white',lw=.4,zorder=3)

# a: the full existing F66 efficiency grid; numeric speed spacing is preserved.
label('a',.028,.968)
ah = axis([.100,.825,.775,.145],False)
speed_edges = np.r_[ws[0]-(ws[1]-ws[0])/2,(ws[:-1]+ws[1:])/2,ws[-1]+(ws[-1]-ws[-2])/2]
direction_edges = np.arange(-2.5,360,5)
cmap = LinearSegmentedColormap.from_list('efficiency',['#FBF5EA','#D8E5DF',TEAL,BLUE])
mesh = ah.pcolormesh(direction_edges,speed_edges,efficiency,vmin=0,vmax=100,cmap=cmap,shading='flat',rasterized=True)
ah.set(xlim=(-2.5,357.5),ylim=(speed_edges[0],speed_edges[-1]),ylabel='Wind speed (m s$^{-1}$)')
ah.set_yticks([3,9,16,22,25])
ah.set_xticks([0,90,180,270,355])
ah.tick_params(axis='x',labelbottom=False,bottom=False)
ah.axhline(9,color=ORANGE,lw=1,ls=(0,(3,2)))
ah.text(.985,.045,'F66',transform=ah.transAxes,ha='right',va='bottom',color=INK,fontsize=8)
cax = fig.add_axes([.895,.825,.012,.145])
cb = fig.colorbar(mesh,cax=cax,ticks=[0,25,50,75,100])
cb.outline.set_linewidth(.55)
cb.ax.tick_params(length=2,width=.6,labelsize=7.4)
cb.set_label('Farm efficiency (%)',fontsize=8.3,labelpad=5)
al = axis([.100,.714,.775,.082])
al.plot(np.r_[wd,360],np.r_[eta9,eta9[0]],c=BLUE,lw=1.5)
al.fill_between(np.r_[wd,360],np.r_[eta9,eta9[0]],0,color=LIGHT,alpha=.17)
lo,hi = eta9.argmin(),eta9.argmax()
al.scatter([wd[lo],wd[hi]],[eta9[lo],eta9[hi]],s=24,c=ORANGE,edgecolors='white',lw=.35,zorder=4)
al.annotate(f'{eta9[lo]:.1f}%',(wd[lo],eta9[lo]),xytext=(24,29),fontsize=8,color=ORANGE,
            arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.7},va='center')
al.annotate(f'{eta9[hi]:.1f}%',(wd[hi],eta9[hi]),xytext=(70,106),fontsize=8,color=ORANGE,
            arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.7},va='center')
al.text(.985,.08,'9 m s$^{-1}$',transform=al.transAxes,ha='right',color=ORANGE,fontsize=8)
al.set(xlim=(-2.5,360),ylim=(0,112),ylabel='Efficiency (%)',xlabel='Wind direction (°)')
al.set_xticks([0,90,180,270,360]); al.set_yticks([0,50,100])

# b: spacing and generation differences, retaining every site.
label('b',.028,.635)
ab = axis([.100,.457,.355,.170])
scatter(ab,S,A)
ab.axvline(4,c=GREY,lw=.8,ls=(0,(3,2)))
ab.text(4.14,24,'4D',fontsize=7.9,color=GREY)
ab.text(.98,.95,f'ρ = {r_spacing:.3f}\nn = 171',transform=ab.transAxes,ha='right',va='top',fontsize=8)
ab.set(xlim=(S.min()-.3,S.max()+.3),ylim=(-.7,27),ylabel='Orientation sensitivity, A (%)',xlabel='Nearest-neighbour spacing (D)')
ab.set_yticks([0,5,10,15,20,25])

# c: four controlled wind distributions; a linked annotation quantifies C0 -> C1.
label('c',.514,.635)
ac = axis([.703,.457,.222,.170],False)
for j,value in enumerate(cmeans):
    ac.hlines(j,0,value,color=LIGHT,lw=1.8)
    ac.scatter(value,j,s=38,c=BLUE,edgecolor='white',lw=.5,zorder=3)
    ac.text(value+.045,j,f'{value:.2f}',va='center',fontsize=8.2,color=BLUE)
ac.set_yticks(range(4),['C0  Joint winds','C1  Speed–direction\nindependent','C2  Common speed\ndistribution','C3  Uniform direction'])
ac.tick_params(axis='y',length=0,labelsize=7.5,pad=7)
ac.spines['left'].set_visible(False)
ac.set(xlim=(-.03,1.40),ylim=(3.45,-.68),xlabel='Mean A (%)')
ac.set_xticks([0,.5,1.0])
ac.annotate('',(cmeans[1],.83),(cmeans[0],.18),arrowprops={'arrowstyle':'->','color':ORANGE,'lw':1})
ac.text(1.15,.53,f'−{drop:.1f}%',color=ORANGE,fontsize=8,ha='center',va='center')

# d: connected coefficients replace paired bars; no confidence intervals implied.
label('d',.028,.370)
ad = axis([.310,.278,.615,.094],False)
raw, partial = stats.spearman.to_numpy(),stats.partial_spearman.to_numpy()
for j,(r,p) in enumerate(zip(raw,partial)):
    ad.plot([r,p],[j,j],c=GREY,lw=1.1,zorder=1)
ad.scatter(raw,np.arange(4),s=32,facecolors='white',edgecolors=TEAL,lw=1.2,zorder=3)
ad.scatter(partial,np.arange(4),s=30,c=PURPLE,marker='D',edgecolor='white',lw=.4,zorder=4)
for j,value in enumerate(partial):
    ad.text(max(.028,max(raw[j],value)+.038),j,f'{value:.3f}',va='center',fontsize=8,color=PURPLE)
ad.axvline(0,color=GREY,lw=.7,ls=(0,(3,2)))
ad.set_yticks(range(4),['Single direction · frequency','Common axis · frequency','Single direction · V³ weighted','Common axis · V³ weighted'])
ad.tick_params(axis='y',length=0,labelsize=7.6,pad=7)
ad.spines['left'].set_visible(False)
ad.set(xlim=(-.15,.74),ylim=(3.50,-.52),xlabel='Correlation with A')
ad.set_xticks([-.1,0,.2,.4,.6])
ad.legend(handles=[Line2D([],[],marker='o',ms=4.6,mfc='white',mec=TEAL,lw=0,label='Spearman'),
                   Line2D([],[],marker='D',ms=4.5,mfc=PURPLE,mec='white',lw=0,label='Controlling for spacing')],
          loc='lower right',bbox_to_anchor=(1,1.06),ncol=2,fontsize=7.8,handletextpad=.3,columnspacing=1.1)

# e and f: original-angle effects and baseline wake loss, with unchanged metrics.
label('e',.028,.206)
ae = axis([.100,.047,.355,.151])
scatter(ae,delta,ratio,high,marker='D')
ae.text(.025,.97,f'ρ = {r_phase:.3f}\nn = {high.sum()}; A > 5.2%',transform=ae.transAxes,ha='left',va='top',fontsize=7.9)
ae.set(xlim=(-2,90),ylim=(-3,max(ratio[high])*1.12),xlabel='Original-to-optimal angle (°)',ylabel='Gain relative to A (%)\n100G$_{diag}$/A')
ae.set_xticks([0,30,60,90]); ae.set_yticks([0,40,80,120,160])
label('f',.514,.206)
af = axis([.585,.047,.340,.151])
scatter(af,A,df.S1_gain_pct.to_numpy(),project.to_numpy())
af.axhline(1,color=GREY,lw=.7,ls='--')
af.axvline(5.2,color=GREY,lw=.7,ls='--')
af.text(.03,.97,f'ρ = {r_gain:.3f}; n = 155',transform=af.transAxes,ha='left',va='top',fontsize=7.9)
af.set(xlim=(-.7,27),ylim=(-1.2,17.5),xlabel='Orientation sensitivity, A (%)',ylabel='Generation gain in\n2020–2024 (%)')
af.set_yticks([0,5,10,15]); af.set_xticks([0,10,20])
fig.legend(handles=[Line2D([],[],marker='o',ms=4.6,mfc=ORANGE,mec='white',lw=0,label='Selected-region sites (n = 23)'),
                    Line2D([],[],marker='o',ms=4.6,mfc=LIGHT,mec='white',lw=0,label='Other sites (n = 148)')],
           loc='center',bbox_to_anchor=(.51,.673),ncol=2,handletextpad=.2,columnspacing=1.4,fontsize=8)

save_600dpi(fig, 'Figure_2_600dpi')
fig.savefig(HERE/'figure2_preview.png',dpi=170,facecolor='white')
plt.close(fig)
panel.to_csv(HERE/'figure2_panel_data.csv',index=False,encoding='utf-8-sig')
stats.to_csv(HERE/'figure2_correlations.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(efficiency,index=ws,columns=wd).rename_axis('wind_speed_m_s').to_csv(HERE/'figure2_f66_efficiency_pct.csv',encoding='utf-8-sig')
pd.DataFrame({'scenario':['C0','C1','C2','C3'],'mean_A_pct':cmeans}).to_csv(HERE/'figure2_scenario_means.csv',index=False)
meta = {'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in INPUTS],
        'n_sites':171,'n_selected_regions':23,'n_panel_e':int(high.sum()),
        'template_aggregation':'Five distinct geometries; rotated S_B45 excluded from average',
        'rho_spacing':r_spacing,'rho_phase':r_phase,'rho_A_gain':r_gain,
        'scenario_means':cmeans.tolist(),'C1_reduction_pct':float(drop),
        'main_panel':{'farm':66,'speeds':ws.tolist(),'directions':wd.tolist(),'shape':list(efficiency.shape),
                      'numeric_speed_axis':True,'interpolation':False,'missing':0,
                      'excluded_speeds_zero_reference_power':all_ws[~valid_power].tolist(),
                      'nine_m_s_range_pct':[float(eta9.min()),float(eta9.max())]},
        'reference_style':'Figure 1 v5.7 and previously selected neighbouring-project style; blue, orange, teal, purple; Arial; no panel titles',
        'layout':'Dominant full-width heatmap and 9 m/s section; five supporting panels; original a-f numbering retained',
        'visual_qa':'pending'}
(HERE/'figure2_validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(HERE),'data_comparison':'passed','n':171,'main_panel_grid':list(efficiency.shape)},ensure_ascii=False))
