"""
wp9f：按 v3.4 新口径（27/171、wp7a 冻结管线）重绘论文图2 —— 复刻原图 fig02 版式
================================================================
版式：完全按原图《结论三、四重算结果/figures/extracted/fig02.png》逐像素测量坐标复刻
  (a) 左上：F66 终期真实机位 9 m/s 高尾流方向（θ=0°）FLORIS 水平面速度亏损场
  (b) 右上：低尾流方向（θ=35°）；（b）右缘贴竖色条（YlOrRd，0–50%，上=高值）
  (c) 左中：整幅小提琴图——高响应场（A>5.2%）vs 其余场的有效间距 S/D 分布
            （v36 起替换原 S/D–A 散点与国家留出插图，学长反馈"换成小提琴图"）
  (d) 右中：27 场 Δφ vs 100·G/A（点色=走廊归属：红=台湾海峡/橙=越南/灰=其他，
            红菱=分箱中位±IQR，图例盒内）
  (e) 下排全宽：二阶谐波重构 vs 2024 逐时尾流模拟增益（171 场，1:1 线）
v35/v36 改动（学长反馈 2026-08-25）：
  1. (c)/(d) 两个中排面板左右对调：(c) 间距在左、(d) 相位在右（与图注阅读顺序一致），
     LOO 插图随 (c) 移到左面板右上角（x 坐标镜像）
  2. (b) 删除 farm 凸包浅灰填充层（"框不要"），地图白底（亏损<15% 不着色）
  3. v35 (c) 顶部新增横向小提琴；v36 按用户要求 (c) 改为整幅小提琴图
     （高响应 vs 其余 × S/D，红/蓝配色+中位菱形+四分位须线），删除国家留出插图
  4. (d) 散点按走廊归属着色（红=台湾海峡、橙=越南、灰=其他，意大利并入其他），图例同步
  5. 数据路径修正：/结论三、四重算结果/output → /补算/output（目录已迁移）
v37 改动（用户 2026-08-27：板面单调，换个表示方法丰富版面）：
  1. (c) 小提琴上叠加抖动个体点（雨云样式，组色半透明，垫在须线/中位之下）
  2. (d) 顶部空白区新增走廊堆叠 Δφ 直方图条（位置经数据核查，不遮任何散点）
  3. (e) 散点下加蓝色 KDE 密度等高线（展示 171 场在 1:1 线附近的聚集结构）
  （c/d 面板顺序 v36 起已是 (c) 左、(d) 右；docx 内嵌仍为旧图，待批准后替换）
v38 改动（用户 2026-08-28：论文图字体统一）：
  1. 字体族 DejaVu Sans → Arial（与 nc_style_nat 主稿图1/3/4 同族，Helvetica/DejaVu 兜底）
  2. 字号体例统一到主稿尺度：基底 8、轴标签 8、刻度 7、图例 7、面板字母 11.5 粗体
     （原 15.5 轴标签 / 13 字母 / 12 标题 / 9 刻度全部下调）
  3. 版式、数据、坐标全部不动（v37 用户已认可的内容零改动）
原图取证要点（本版全部落实）：
  - 原图 PNG 坐标系自底向上：所有 y 用 y_fig = 1 − y_measured − h 换算
  - 面板脊线仅左+下（纯黑 ~1.1pt=3px），顶/右无脊线；刻度黑；地图刻度向内、散点图向外
  - 色条盒 x[2359,2385]px，黑色细框；YlOrRd 0–50，上=深红
  - 插图：红(REDMED)左脊线 + 浅灰(235,238,240)右脊线 + 4 条等距浅灰横网格线
    + 红菱形（分国留出中位）+ 顶部标题，无刻度无 1:1 线
  - 图内蓝色风向箭头 lw≈2pt mutation_scale≈8（原图盒内蓝色像素 ~850）
风坐标：rot_wind 自实现 wind_delta 旋转（风沿 +x），平面网格与涡轮点同一变换。
    尾流场用 fm.sample_flow_at_points 在农场坐标规则网格上直接采样
    （CutPlane df 的 x1/x2 与 u 框架错位，已实证弃用）。
输出：结论三、四重算结果/figures-new/fig2_v38.png（200 dpi，2555×2718 px，与 v34 同尺寸）
"""
import os, io, sys, warnings, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from scipy.ndimage import uniform_filter
from pyproj import Transformer, CRS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyArrowPatch

REPO = 'd:/01学习资料/wind-direction-to-electricity-transition'
OUTDIR = REPO + '/结论三、四重算结果/figures-new'
sys.path.insert(0, REPO + '/offshore-task2')
from floris_config import create_floris_model, get_ti_for_farm
from floris.utilities import wind_delta

# ── 复刻原图的样式参数（全部来自 fig02.png 像素测量） ──
FIG_W, FIG_H = 12.775, 13.59        # 英寸 → 200 dpi 下恰为 2555×2718 px
ORANGE = '#FD7C37'                  # 高响应场橙
SPBLK  = '#000000'                  # 面板脊线/刻度：纯黑（原图取样 (0,0,0) 3px 宽）
DARK   = '#232323'                  # 文字
GRAYD  = '#A8A8A8'                  # 散点灰
REDMED = '#B03B3B'                  # (d)/插图 中位菱形红（测得 179,58,58）
TURB   = '#25313B'                  # 涡轮小点深蓝灰（测得 37,49,59）
GRIDLG = '#EBEEF0'                  # 插图横网格线浅灰（测得 235,238,240）
BLUE   = '#3973A5'                  # 图内风向箭头蓝（测得 57,115,165）
ARROW  = '#363636'                  # 边距小箭头深灰（测得 54,54,54）
CMAP   = 'YlOrRd'                   # 尾流场色（色条 LUT 573.7 距离决定性匹配）
VMAX   = 35.0                       # 亏损色标上限 %（原图带内深红≈29-34% 亏损，反推色标 15-35）
VMIN   = 15.0                       # 亏损色标下限 %（<15% 不着色=白底，原图取证）
D = 198.0                           # iea_10MW 风轮直径 m

plt.rcParams.update({
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],   # v38：与主稿图1/3/4 统一（nc_style_nat）
    'font.size': 8, 'axes.linewidth': 0.7,
    'figure.dpi': 200, 'savefig.dpi': 200,
})

# ═══════════════════════════════════════════════════════════════════════
# 一、数据（与 wp9e 完全同口径，统计量应与此前一致）
# ═══════════════════════════════════════════════════════════════════════
cur = np.load(REPO + '/补算/output/wp7a_real_curves.npz', allow_pickle=True)
eta_npz = np.load(REPO + '/补算/output/wp7a_real_eta.npz', allow_pickle=True)
A = cur['A']; th_star = cur['th_star']; E_y = cur['E_y']; fids = cur['farm_ids']
eta = eta_npz['eta']; ws = eta_npz['ws']; wd = eta_npz['wd']; fids_eta = eta_npz['farm_ids']
loo = np.load(REPO + '/补算/output/wp6c_loo.npz', allow_pickle=True)

metrics = pd.read_csv(REPO + '/补算/output/wp9c_farm_metrics.csv', encoding='utf-8-sig')
metrics = metrics.set_index('farm_id').loc[fids].reset_index()
S = metrics['spacing_D_med'].values.astype(float)

# 走廊归属（v35：图 d 点色；意大利并入 other，走廊统计口径只含台湾海峡+越南）
tb7c = pd.read_csv(REPO + '/补算/output/wp7c_scenario_table.csv', encoding='utf-8-sig')
c2corr = dict(zip(tb7c['farm_id'], tb7c['corridor']))
CORR_COL = {'China_strait': '#D9361E', 'Vietnam': ORANGE, 'other': GRAYD}
CORR_NAMES = {'China_strait': 'Taiwan Strait', 'Vietnam': 'Vietnam', 'other': 'Other'}

hi = A > 5.2
n_hi = int(hi.sum())
print(f'高响应场数（A>5.2%, wp7a）: {n_hi}')

# ── 相位与增益 ──
dphi_deg = th_star % 180
dphi = np.minimum(dphi_deg, 180 - dphi_deg)
idx = np.arange(len(th_star))
Esel = E_y[idx, :, th_star // 5]
E0 = E_y[:, :, 0]
G = 100 * np.nanmean((Esel - E0) / np.maximum(E0, 1e-9), axis=1)
ratio = 100 * G / A
ok_ratio = hi & np.isfinite(ratio) & (A > 0)

# ── 重构 vs G2024 ──
a_ = A / 100.0
Ghat = 100 * 2 * a_ * np.sin(np.deg2rad(dphi)) ** 2 / (1 + a_ * np.cos(2 * np.deg2rad(dphi)))
G2024 = 100 * (E_y[idx, 10, th_star // 5] / E_y[:, 10, 0] - 1)
m = np.isfinite(Ghat) & np.isfinite(G2024)
rs, rp = spearmanr(Ghat[m], G2024[m]).statistic, pearsonr(Ghat[m], G2024[m]).statistic
bias = np.mean(Ghat[m] - G2024[m]); mae = np.mean(np.abs(Ghat[m] - G2024[m]))
neg = (G2024[m] < 0).mean() * 100
r_phase = spearmanr(dphi[ok_ratio], ratio[ok_ratio]).statistic
print(f'相位 ρ（{int(ok_ratio.sum())} 场）= {r_phase:.3f}')
print(f'重构（{int(m.sum())} 场）: Spearman {rs:.3f} / Pearson {rp:.3f} / '
      f'偏差 {bias:+.2f} pp / MAE {mae:.2f} pp / 负增益 {neg:.0f}%')
print(f'S 范围 [{np.nanmin(S):.2f}, {np.nanmax(S):.2f}], A 范围 '
      f'[{np.nanmin(A):.2f}, {np.nanmax(A):.2f}]')

# ═══════════════════════════════════════════════════════════════════════
# 二、F66 端元（9 m/s 高/低尾流方向，FLORIS 实算）
# ═══════════════════════════════════════════════════════════════════════
tc = pd.read_csv(REPO + '/data/task0/turbine_coordinates.csv', encoding='utf-8-sig')
tc = tc.sort_values(['farm_id', 'year'])
last_year = tc.groupby('farm_id')['year'].max().reset_index()
tc_last = tc.merge(last_year, on=['farm_id', 'year'])
g66 = tc_last[tc_last.farm_id == 66]
lon66 = g66['lon'].values.astype(float); lat66 = g66['lat'].values.astype(float)
clon, clat = float(np.median(lon66)), float(np.median(lat66))
tf = Transformer.from_crs(CRS.from_epsg(4326),
                          CRS.from_proj4(f'+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m'),
                          always_xy=True)
x66, y66 = tf.transform(lon66, lat66)
x66 = np.asarray(x66) / 1000.0          # km
y66 = np.asarray(y66) / 1000.0

i66 = list(fids_eta).index(66)
j9 = int(np.argmin(np.abs(ws - 9.0)))
e66 = eta[i66, j9, :]
th_hi, th_lo = float(wd[np.argmin(e66)]), float(wd[np.argmax(e66)])
eta_hi, eta_lo = float(e66.min()), float(e66.max())
print(f'F66 9 m/s 端元: 高尾流 θ={th_hi:.0f}° η={eta_hi*100:.1f}% | '
      f'低尾流 θ={th_lo:.0f}° η={eta_lo*100:.1f}% | A_wp7a={A[list(fids).index(66)]:.2f}%')

ti66 = get_ti_for_farm(clat, clon)

# FLORIS 官方风坐标旋转（风沿 +x），旋转中心 = 涡轮包围盒中心
_coords = np.column_stack([x66 * 1000.0, y66 * 1000.0, np.zeros(len(x66))])
XC_M, YC_M = float(np.mean(x66)) * 1000.0, float(np.mean(y66)) * 1000.0

def rot_wind(x_m, y_m, theta, xc=XC_M, yc=YC_M):
    """农场坐标(m) → 风坐标(m)：wind_delta(θ)=(θ−270)%360 使风自西(+x)吹来。
    网格与涡轮用同一变换，保证尾流场与涡轮点严格对齐。"""
    d = np.deg2rad(float(wind_delta(np.array([theta]))[0]))
    xo = x_m - xc; yo = y_m - yc
    xr = xo * np.cos(d) - yo * np.sin(d) + xc
    yr = xo * np.sin(d) + yo * np.cos(d) + yc
    return xr, yr

def turb_wind_frame(theta):
    """涡轮位置（km）→ 风坐标系（风沿 +x）"""
    xr, yr = rot_wind(x66 * 1000.0, y66 * 1000.0, theta)
    return xr / 1000.0, yr / 1000.0

def run_floris(theta):
    fm, tmp = create_floris_model(x66 * 1000.0, y66 * 1000.0, turbine_type='iea_10MW',
                                  ti=ti66, alpha=0.11, wake_model_name='gauss')
    fm.set(wind_speeds=[9.0], wind_directions=[theta], turbulence_intensities=[ti66])
    fm.run()
    u = np.asarray(fm.turbine_average_velocities).ravel()
    os.unlink(tmp)
    return u

def run_floris_plane(theta, margin_m=0.0, res_pts=40):
    """FLORIS 水平面（轮毂高 119 m）：自建农场坐标规则网格 + sample_flow_at_points 采样，
    再整体旋转到风坐标系（风沿 +x），返回规则网格与 u 场（km）。
    （CutPlane df 的 x1/x2 与 u 存在框架错位——实测无法对齐涡轮速度，已弃用；
      sample_flow_at_points 在涡轮位置与涡轮速度相关 0.99+，200×200 仅 1.45 s。）"""
    fm, tmp = create_floris_model(x66 * 1000.0, y66 * 1000.0, turbine_type='iea_10MW',
                                  ti=ti66, alpha=0.11, wake_model_name='gauss')
    fm.set(wind_speeds=[9.0], wind_directions=[theta], turbulence_intensities=[ti66])
    fm.run()                                   # ← 必须先 run 才有尾流
    x0 = float(x66.min()) * 1000.0 - margin_m
    x1 = float(x66.max()) * 1000.0 + margin_m
    y0 = float(y66.min()) * 1000.0 - margin_m
    y1 = float(y66.max()) * 1000.0 + margin_m
    xs = np.linspace(x0, x1, res_pts)
    ys = np.linspace(y0, y1, res_pts)
    Xf, Yf = np.meshgrid(xs, ys)
    U = np.asarray(fm.sample_flow_at_points(Xf.ravel(), Yf.ravel(),
                                            np.full(Xf.size, 119.0))).reshape(Xf.shape)
    os.unlink(tmp)
    Xr, Yr = rot_wind(Xf, Yf, theta)           # 网格与涡轮同一旋转变换 → 风坐标系
    return Xr / 1000.0, Yr / 1000.0, U

u_hi = run_floris(th_hi)
u_lo = run_floris(th_lo)
print(f'θ={th_hi:.0f}° 涡轮 u min {u_hi.min():.2f} | θ={th_lo:.0f}° 涡轮 u min {u_lo.min():.2f}')
# (a) 原图带横贯面板、带内连续深红：margin 1200m + 100×100 网格
#    （cell 沿风 158m×横向 67m：核与核间平均后带内 ~29-34% → 深红连续，原图同此结构）
Xp_hi, Yp_hi, Up_hi = run_floris_plane(th_hi, margin_m=1200.0, res_pts=100)
# (b) 原图 farm 菱形缩至面板 ~40%：margin 5000m + 粗网格（cell~244m，配合平滑避开尾流核）
Xp_lo, Yp_lo, Up_lo = run_floris_plane(th_lo, margin_m=5000.0, res_pts=100)
print(f'平面场 u min: θ={th_hi:.0f}° {Up_hi.min():.2f} | θ={th_lo:.0f}° {Up_lo.min():.2f}')
xt_hi, yt_hi = turb_wind_frame(th_hi)
xt_lo, yt_lo = turb_wind_frame(th_lo)

# ═══════════════════════════════════════════════════════════════════════
# 三、画布（原图 PNG 自底向上 → y_fig = 1 − y_measured − h）
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(FIG_W, FIG_H))

def ax_at(x0, x1, y0, y1):
    return fig.add_axes([x0, y0, x1 - x0, y1 - y0])

axa = ax_at(0.1448, 0.4356, 0.6865, 0.9592)      # (a) 显示于图上排左
axb = ax_at(0.6215, 0.9233, 0.6865, 0.9592)      # (b) 显示于图上排右
cax = ax_at(0.9233, 0.9335, 0.6865, 0.9592)      # 色条（盒 x[2359,2385]px，紧贴 b 右缘）
axc = ax_at(0.1166, 0.4638, 0.3547, 0.6060)      # (c) 间距显示于中排左（v35 与 (d) 对调）
axd = ax_at(0.5652, 0.9119, 0.3547, 0.6060)      # (d) 相位显示于中排右（v35 与 (c) 对调）
axe = ax_at(0.1166, 0.9119, 0.0647, 0.2531)      # (e) 显示于下排全宽

def style_spines(ax, ticks_in=False):
    """原图样式：仅左+下黑色脊线，顶/右无；刻度黑；地图向内、散点图向外"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(SPBLK)
        ax.spines[sp].set_linewidth(1.1)         # 3px @200dpi
    kw = dict(labelsize=7, labelcolor=SPBLK, colors=SPBLK)   # v38 刻度 7（主稿体例）
    if ticks_in:
        kw.update(direction='in', length=3)
    ax.tick_params(axis='both', which='major', **kw)

def draw_map(ax, u, Xp, Yp, Up, xt, yt, ttl1, ttl2, x0_arrow, x1_arrow, farm_fill=False):
    """尾流亏损场 + 涡轮点 + 图内蓝色风向箭头（全部在风坐标系）。
    原图取证：地图为白底（<15% 亏损不着色），(a) 深红带=各排尾流核（亏损封顶 ~47%），
    (b) 淡黄低亏损斑块；粗网格 nearest 斑块质感。
    v35：删除 (b) farm 凸包浅灰填充层（学长反馈"框不要"），其余不变。"""
    defi = np.clip(100 * (1 - Up / 9.0), 0, 3 * VMAX)
    if farm_fill:
        # (b) 14×14 平滑：θ=35 近尾流核被粗网格+平滑抹平，最大亏损 ~20-25%（原图 (b) 最深淡黄）
        defi = uniform_filter(defi, size=14)
    else:
        # (a) 沿风轴 9 窗口平滑：核与核间平均后带内连续 ~29-34%（原图带内纯深红结构）
        defi = uniform_filter(defi, size=(1, 9))
    cmap_w = plt.get_cmap(CMAP).copy()
    cmap_w.set_under('white')
    ax.pcolormesh(Xp, Yp, defi, cmap=cmap_w, norm=Normalize(VMIN, VMAX),
                  shading='nearest', rasterized=True, zorder=1)
    ax.scatter(xt, yt, s=2.5, color=TURB, lw=0, zorder=3)
    ax.set_xlim(Xp.min(), Xp.max()); ax.set_ylim(Yp.min(), Yp.max())
    # 原图：脊线在完整面板（y=852px），地图等比居中、上下留白 → adjustable='datalim'
    # （'box' 会收缩轴盒使脊线上移、箭头跑位）
    ax.set_aspect('equal', adjustable='datalim')
    style_spines(ax, ticks_in=True)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    # 两行内嵌标题（左上角）
    ax.text(0.034, 0.968, ttl1, transform=ax.transAxes, fontsize=8.5,
            va='top', color=DARK, zorder=6)
    ax.text(0.034, 0.925, ttl2, transform=ax.transAxes, fontsize=8.5,
            va='top', color=DARK, zorder=6)
    # 图内蓝色风向箭头（底部，指向 +x = 风向；原图盒内蓝色像素 ~850 → lw2/mut8）
    ax.annotate('', xy=(x1_arrow, 0.082), xytext=(x0_arrow, 0.082),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.0, mutation_scale=8))

draw_map(axa, u_hi, Xp_hi, Yp_hi, Up_hi, xt_hi, yt_hi,
         'High-wake direction', f'θ = {th_hi:.0f}°, η = {eta_hi*100:.1f}%', 0.678, 0.930)
draw_map(axb, u_lo, Xp_lo, Yp_lo, Up_lo, xt_lo, yt_lo,
         'Low-wake direction', f'θ = {th_lo:.0f}°, η = {eta_lo*100:.1f}%', 0.654, 0.895,
         farm_fill=True)

# ── 色条（盒 x[2359,2385]px，黑色细框，上=高值，无标题） ──
cb = fig.colorbar(ScalarMappable(norm=Normalize(VMIN, VMAX), cmap=CMAP), cax=cax)
cb.set_ticks([15, 20, 25, 30, 35])
cb.ax.yaxis.set_ticks_position('right')
cb.ax.tick_params(labelsize=7, labelcolor=SPBLK, colors=SPBLK, length=3)
cb.outline.set_color(SPBLK); cb.outline.set_linewidth(1.1)

# ── 边距风向小箭头（显示于两图上方左侧空白，深灰） ──
for x0, x1, xt in ((0.0900, 0.1033, 0.106), (0.5663, 0.5797, 0.583)):
    fig.add_artist(FancyArrowPatch((x0, 0.9290), (x1, 0.9290),
                                   arrowstyle='-|>', mutation_scale=16,
                                   color=ARROW, lw=1.5, transform=fig.transFigure,
                                   shrinkA=0, shrinkB=0))
    fig.text(xt, 0.9290, '9 m s⁻¹', fontsize=8, va='center', ha='left', color=DARK)

# ═══════════════════════════════════════════════════════════════════════
# (d) 左中：27 场 Δφ vs 100·G/A
# ═══════════════════════════════════════════════════════════════════════
# v35：散点按走廊归属着色（红=台湾海峡、橙=越南、灰=其他；意大利并入其他）
corr_hi = [c2corr.get(int(f), 'other') for f in fids[ok_ratio]]
corr_hi = ['other' if c == 'Italy' else c for c in corr_hi]
n_corr = {cg: int(sum(c == cg for c in corr_hi)) for cg in ('China_strait', 'Vietnam', 'other')}
for cg, col in CORR_COL.items():
    sel = np.array([c == cg for c in corr_hi])
    axd.scatter(dphi[ok_ratio][sel], ratio[ok_ratio][sel], s=24, marker='D',
                color=col, lw=0, zorder=3)
bins = [0, 15, 30, 45, 60, 75, 90]
for k in range(len(bins) - 1):
    sel = (dphi[ok_ratio] >= bins[k]) & (dphi[ok_ratio] < bins[k + 1])
    if sel.sum() == 0:
        continue
    xs = dphi[ok_ratio][sel]; ys = ratio[ok_ratio][sel]
    xc = float(np.median(xs)); yc = float(np.median(ys))
    lo_, hi_ = np.percentile(ys, 25), np.percentile(ys, 75)
    axd.plot(xc, yc, 'D', color=REDMED, ms=5, zorder=4)
    axd.plot([xc, xc], [lo_, hi_], color=DARK, lw=1.1, zorder=4)
axd.text(0.04, 0.94, f'Spearman ρ = {r_phase:.3f}  (n = {int(ok_ratio.sum())})',
         transform=axd.transAxes, fontsize=8, va='top', color=DARK)
axd.set_xlim(0, 90); axd.set_xticks(np.arange(0, 91, 15))
# ratio 上至 145.3%（7/27 场 >30%），y 范围按数据自适应（原图 [-30,30] 会裁点）
axd.set_ylim(0, 150); axd.set_yticks(np.arange(0, 151, 30))
style_spines(axd)
# 图例盒（v35：走廊归属 + 分箱中位；四条目纵向排布）
handles = [plt.Line2D([], [], marker='D', color=CORR_COL[cg], ls='', ms=7,
                      label=f'{CORR_NAMES[cg]} (n = {n_corr[cg]})')
           for cg in ('China_strait', 'Vietnam', 'other')] + \
          [plt.Line2D([], [], marker='D', color=REDMED, ls='', ms=6,
                      label='Bin median ± IQR')]
leg = axd.legend(handles=handles, loc='upper left', mode='expand',
                 bbox_to_anchor=(0.58, 0.30, 0.34, 0.42),
                 frameon=True, edgecolor=SPBLK, facecolor='white', fontsize=7,
                 handletextpad=0.4, borderpad=0.5)
leg.get_frame().set_linewidth(1.1)

# v37 顶部空白区：走廊堆叠 Δφ 直方图（位置经数据核查：Δφ∈[29.7,60.3]°、
# ratio∈[114,142.5] 区域无任何散点，图例盒与 ρ 注记亦不相交）
axdh = axd.inset_axes([0.33, 0.76, 0.34, 0.19])
_hbins = np.arange(0, 91, 15)
_hbtm = np.zeros(len(_hbins) - 1)
for cg in ('China_strait', 'Vietnam', 'other'):
    _sel = np.array([c == cg for c in corr_hi])
    _cnt, _ = np.histogram(dphi[ok_ratio][_sel], bins=_hbins)
    axdh.bar(_hbins[:-1], _cnt, width=13.5, bottom=_hbtm, color=CORR_COL[cg],
             edgecolor='white', lw=0.5)
    _hbtm += _cnt
axdh.set_xlim(-7.5, 97.5); axdh.set_ylim(0, max(3, _hbtm.max()))
axdh.set_xticks([0, 45, 90])
axdh.yaxis.set_major_locator(MaxNLocator(3))
axdh.tick_params(labelsize=7, colors=SPBLK, length=2)
for sp in ('top', 'right'):
    axdh.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axdh.spines[sp].set_color(SPBLK); axdh.spines[sp].set_linewidth(0.8)
axdh.set_xlabel('Δφ (°)', fontsize=8)
axdh.set_ylabel('n', fontsize=8)

# ═══════════════════════════════════════════════════════════════════════
# (c) 左中：整幅小提琴图（v36：高响应组 vs 其余组的有效间距 S/D 分布；
#      学长反馈"换成小提琴图"——原 S/D–A 散点与国家留出插图整体替换）
# ═══════════════════════════════════════════════════════════════════════
S_lo_v = S[~hi & ~np.isnan(S)]
S_hi_v = S[hi & ~np.isnan(S)]
ymax_v = max(12.0, 2.0 * math.ceil(np.nanmax(S) / 2))
vparts = axc.violinplot([S_hi_v, S_lo_v], positions=[0, 1], vert=True,
                        widths=0.60, showmedians=True)
for body, col in zip(vparts['bodies'], (ORANGE, BLUE)):
    body.set_facecolor(col); body.set_edgecolor('none'); body.set_alpha(0.85)
vparts['cmedians'].set_color(SPBLK)
vparts['cmedians'].set_linewidth(1.3)
vparts['cmedians'].set_zorder(6)
for xi, arr in ((0, S_hi_v), (1, S_lo_v)):
    med = float(np.median(arr)); q1, q3 = np.percentile(arr, [25, 75])
    axc.plot([xi, xi], [q1, q3], color=DARK, lw=1.2, zorder=5)
    axc.plot(xi, med, 'D', color='white', mec=DARK, mew=1.1, ms=6.5, zorder=7)
    axc.text(xi, q3 + 0.045 * ymax_v, f'{med:.1f}D', ha='center', va='bottom',
             fontsize=7.5, color=DARK, fontweight='bold', zorder=8)
# v37 雨云：个体点抖动叠加（组色半透明，zorder 垫在四分位须/中位之下）
rng = np.random.default_rng(7)
for xi, arr, col in ((0, S_hi_v, ORANGE), (1, S_lo_v, BLUE)):
    axc.scatter(xi + rng.uniform(-0.17, 0.17, len(arr)), arr, s=7,
                color=col, alpha=0.45, lw=0, zorder=4)
axc.set_xlim(-0.72, 1.72)
axc.set_xticks([0, 1])
axc.set_xticklabels([f'High response\n(A > 5.2%, n = {len(S_hi_v)})',
                     f'Others\n(n = {len(S_lo_v)})'], fontsize=7.5)
axc.set_ylim(0, ymax_v)
axc.set_yticks(np.arange(0, ymax_v + 0.01, 2))
axc.set_ylabel('Effective spacing, S/D (D)', fontsize=8)
axc.text(0.035, 0.955, 'n = 171 farms', transform=axc.transAxes, va='top',
         fontsize=8, color=DARK)
style_spines(axc)

# ═══════════════════════════════════════════════════════════════════════
# (e) 下排：二阶谐波重构 vs 2024 逐时尾流模拟增益
# ═══════════════════════════════════════════════════════════════════════
hi_e = max(20.0, 5.0 * math.ceil(np.nanmax([Ghat[m].max(), G2024[m].max()]) / 5))
lo_e = min(-5.0, -5.0 * math.ceil(max(0.0, -np.nanmin([Ghat[m].min(), G2024[m].min()])) / 5))
axe.scatter(G2024[m], Ghat[m], s=12, color=GRAYD, alpha=0.8, lw=0, zorder=2)
axe.scatter(G2024[m & hi], Ghat[m & hi], s=18, color=ORANGE, lw=0, zorder=3)
axe.plot([lo_e, hi_e], [lo_e, hi_e], color='#444444', ls='--', lw=0.9, zorder=1)
# v37 密度等高线（KDE）：171 场在 1:1 线附近的聚集结构（垫在散点之下）
kde = gaussian_kde(np.vstack([G2024[m], Ghat[m]]))
_gx = np.linspace(lo_e, hi_e, 100)
_gxx, _gyy = np.meshgrid(_gx, _gx)
_gz = kde(np.vstack([_gxx.ravel(), _gyy.ravel()])).reshape(_gxx.shape)
axe.contour(_gxx, _gyy, _gz, levels=6, colors=BLUE, alpha=0.45,
            linewidths=0.7, zorder=1.2)
axe.text(0.037, 0.945,
         f'Spearman ρ = {rs:.3f} · Pearson r = {rp:.3f} · '
         f'Bias {bias:+.2f} pp · MAE {mae:.2f} pp',
         transform=axe.transAxes, fontsize=7.5, va='top', color=DARK)
# '1:1' 标注（原图位于对角线右端下方，沿对角线方向旋转）
ang = math.degrees(math.atan2(axe.get_position().height * FIG_H,
                              axe.get_position().width * FIG_W))
axe.text(0.945, 0.655, '1:1', transform=axe.transAxes, fontsize=8,
         rotation=-ang, va='center', ha='left', color=DARK)
axe.text(0.645, 0.555, f'n = {int(m.sum())}', transform=axe.transAxes,
         fontsize=8, color=DARK, va='top')
axe.set_xlim(lo_e, hi_e); axe.set_ylim(lo_e, hi_e)
axe.set_xticks(np.arange(lo_e, hi_e + 0.01, 5))
axe.set_yticks(np.arange(lo_e, hi_e + 0.01, 5))
style_spines(axe)

# ═══════════════════════════════════════════════════════════════════════
# 面板字母（原图像素位置 → 图坐标 y = 1 − py/2718）
# ═══════════════════════════════════════════════════════════════════════
# v37b：c/d/e 字母上移至面板上方空白带（原位置与 y 轴顶部刻度标签重合），
# 与 a/b 字母同体例；py 为距图顶像素：c/d 1077→992、e 2016→1943
for ch, px, py in (('a', 277, 65), ('b', 1496, 57), ('c', 290, 992),
                   ('d', 1434, 992), ('e', 288, 1943)):
    fig.text(px / 2555.0, 1.0 - py / 2718.0, ch, fontsize=11.5, fontweight='bold',
             va='top', ha='left', color=DARK)

# ── 坐标轴标签（图坐标，原图位置换算；v38 字号 8 = 主稿轴标签体例） ──
fig.text(0.2900, 0.6497, 'Along-wind streamwise distance (km)',
         ha='center', va='center', fontsize=8, color=DARK)
fig.text(0.7708, 0.6497, 'Along-wind streamwise distance (km)',
         ha='center', va='center', fontsize=8, color=DARK)
fig.text(0.2526, 0.3177, 'S/D (–)', ha='center', va='center',
         fontsize=8, color=DARK)
fig.text(0.7391, 0.3177, 'Δφ (°)', ha='center', va='center',
         fontsize=8, color=DARK)
fig.text(0.5143, 0.0226, '2024 hourly wake-simulation gain (%)',
         ha='center', va='center', fontsize=8, color=DARK)

os.makedirs(OUTDIR, exist_ok=True)
PNG = OUTDIR + '/fig2_v38.png'
fig.savefig(PNG)
print(f'已保存: {PNG}（{FIG_W}×{FIG_H} in @200dpi = 2555×2718 px）')
print('v38: 字体统一 Arial/主稿字号 | (c) 雨云个体点 | (d) Δφ 走廊堆叠直方图 | (e) KDE 密度等高线')
print(f'走廊分组（d 面板 n={int(ok_ratio.sum())}）:', n_corr)
print(f'小提琴组: 高响应 n={len(S_hi_v)}, 其余 n={len(S_lo_v)}')
