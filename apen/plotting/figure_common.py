from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from plot_config import DATA_ROOT, OUTPUT_ROOT, PLOT_INPUTS, MAP_ROOT, save_600dpi
HERE=OUTPUT_ROOT
ROOT=DATA_ROOT
import shapefile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
BLUE,ORANGE,PURPLE,TEAL='#244B68','#C45D32','#7C658C','#7FA69E'
INK,GREY,LIGHT,GRID='#252C32','#8B99A5','#A4BAC8','#E8ECEE'
PIDS=['S_A','S_B0','S_C','S_D','S_E']
NAMES=['Aligned','Dense','Phased','Compact','Wide']
SPACING=[9.44,3.33,5.,4.,11.8]
def style():
    plt.rcParams.update({'font.family':'Arial','font.size':8.8,'axes.labelsize':8.8,
      'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,'axes.edgecolor':INK,
      'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
      'legend.frameon':False,'legend.fontsize':7.8,'pdf.fonttype':42,'ps.fonttype':42,
      'svg.fonttype':'none','mathtext.fontset':'custom','mathtext.rm':'Arial',
      'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
def axis(fig,bounds,letter=None,grid=True):
    ax=fig.add_axes(bounds);ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.8,width=.65,pad=3);ax.set_axisbelow(True)
    if grid:ax.grid(axis='y',color=GRID,lw=.55)
    if letter:fig.text(bounds[0]-.07,bounds[1]+bounds[3]+.009,letter,fontsize=12,fontweight='bold')
    return ax
def hashes(paths):return [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
def spearman(x,y):return float(np.corrcoef(pd.Series(x).rank(),pd.Series(y).rank())[0,1])
def read_shapes(path):
    records=[]
    with shapefile.Reader(str(path)) as reader:
        for shape in reader.iterShapes():
            points=np.asarray(shape.points)
            starts=list(shape.parts)+[len(points)]
            records.append((shape.bbox,[points[a:b] for a,b in zip(starts[:-1],starts[1:])]))
    return records
MAP_PATHS=[MAP_ROOT/'physical/ne_50m_land.shp',MAP_ROOT/'cultural/ne_50m_admin_0_boundary_lines_land.shp']
def basemap(ax,extent,xticks,yticks,shapes):
    xmin,xmax,ymin,ymax=extent
    for bbox,parts in shapes[0]:
        if bbox[2]<xmin or bbox[0]>xmax or bbox[3]<ymin or bbox[1]>ymax:continue
        vs=[];cs=[]
        for part in parts:
            vs.extend(part);cs.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(part)-2)+[MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MplPath(np.asarray(vs),cs),facecolor='#E9ECEE',edgecolor='#AAB2B7',lw=.35,zorder=0))
    for bbox,parts in shapes[1]:
        if bbox[2]<xmin or bbox[0]>xmax or bbox[3]<ymin or bbox[1]>ymax:continue
        for part in parts:ax.plot(part[:,0],part[:,1],c='#BAC1C6',lw=.3,zorder=1)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal')
    ax.set_xticks(xticks);ax.set_yticks(yticks)
    ax.set_xticklabels([f'{abs(x):g}°'+('E' if x>0 else 'W' if x<0 else '') for x in xticks])
    ax.set_yticklabels([f'{y:g}°N' for y in yticks]);ax.tick_params(length=0)
    ax.grid(color='#D8DEE2',lw=.35,ls=':',zorder=-1);ax.spines[:].set_visible(False)
