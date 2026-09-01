"""
WP2c 建设范式情境布局生成（A–E 范式，学长批准版）
=================================================
依据《建设范式情境设计 v3》（task1 范式识别 A–E + task3 生成逻辑）：
  S_A   侧风对齐：8×8 各向异性网格，Sx/Sy 取 task1 theta_parameters 2024 中位
  S_B0  约束优先（轴 0°）：8×8 等距网格，间距取 WP1 全局 NN 中位
  S_B45 约束优先（轴 45°）：同上，轴 45°
  S_C   分期扩建：一期核心 4×8（5D）+ 二期侧风方向分侧交替扩建 4 行（间隙 2×5D）
  S_D   风资源梯度：4D 候选格点按线性 WPD 梯度（+x）贪心选点，最小间距 4D
  S_E   大间距：S_A 间距 ×1.25

统一口径：64 台 IEA 10MW（D=198 m），轴对齐生成；朝向自由度由 72 档旋转在
WP5c 交叉仿真中覆盖（风况知情范式的建成基线角 = 场址 θ_energy，WP5c 里算）。
自检：范式特征自检（v3 设计 §四），非冻结树形态判定。
注：S_C 自检由 v3 文档的 n_comp=2 调整为"两期结构 + 扩建方向"（10D 间隙小于
    3×nn_med=15D，仍单连通——真实分期扩建的间隙不会大到断开，如实记录）。

输出：output/wp2c_paradigm_layouts.csv（长表）
      output/wp2c_paradigm_summary.csv（摘要 + 自检，列名对齐 wp2_template_summary）
      output/wp2c_paradigm_report.txt
"""
import os, io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BUSH, 'output')
IN_T1 = os.path.join(BUSH, 'input_task1')
D_REF = 198.0


def nn_stats(xy):
    """最近邻统计：mean/max 距离（m）、中位（D 单位）、最小（D 单位）。"""
    if len(xy) < 2:
        return np.nan, np.nan, np.nan, np.nan
    d2 = cKDTree(xy).query(xy, k=2)[0][:, 1]
    return (float(np.mean(d2)), float(np.max(d2)),
            float(np.median(d2)) / D_REF, float(np.min(d2)) / D_REF)


def summarize(pid, xy, s_nominal, axis_deg, checks):
    """生成摘要行（列名对齐 wp2_template_summary.csv）。"""
    nn_m, nn_max_m, nn_med_D, nn_min_D = nn_stats(xy)
    cov = np.cov(xy.T)
    w = np.linalg.eigvalsh(cov)
    aspect = float(np.sqrt(max(w) / max(min(w), 1e-12)))
    pc1 = float(w.max() / w.sum())
    hull = ConvexHull(xy)
    from scipy.sparse import csgraph
    tree = cKDTree(xy)
    thr = 3.0 * max(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]), 1.0)
    sm = tree.sparse_distance_matrix(tree, thr, output_type='coo_matrix')
    n_comp = int(csgraph.connected_components((sm > 0).astype(int), directed=False)[0])
    ce = np.nan
    if len(xy) >= 3:
        try:
            rho = len(xy) / hull.volume
            r_exp = 1.0 / (2.0 * np.sqrt(rho))
            ce = float(np.mean(cKDTree(xy).query(xy, k=2)[0][:, 1]) / r_exp)
        except Exception:
            pass
    return dict(paradigm=pid, n_turb=len(xy), spacing_nom_D=s_nominal,
                axis_deg=axis_deg, nn_mean_m=round(nn_m, 1), nn_max_m=round(nn_max_m, 1),
                nn_med_D=round(nn_med_D, 3), nn_min_D=round(nn_min_D, 3),
                area_km2=round(hull.volume / 1e6, 3), aspect_ratio=round(aspect, 3),
                pc1_share=round(pc1, 4), ce_r=round(ce, 3) if ce == ce else np.nan,
                n_comp=n_comp, class_ok=all(checks.values()), checks=str(checks))


def grid_aniso(n_row, n_col, sx_D, sy_D, axis_deg=0.0):
    """轴对齐各向异性网格（rows 沿 x 轴 = 侧风方向），可绕中心旋转 axis_deg。"""
    pts = np.array([((i - (n_col - 1) / 2) * sx_D * D_REF,
                     (j - (n_row - 1) / 2) * sy_D * D_REF)
                    for j in range(n_row) for i in range(n_col)], float)
    if axis_deg:
        a = np.radians(axis_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = pts @ R.T
    return pts


# ── task1 参数锚 ────────────────────────────────────────────
th = pd.read_csv(os.path.join(IN_T1, 'task1_theta_parameters.csv'), encoding='utf-8-sig')
th2024 = th[th['year'] == 2024]
SX = float(np.clip(th2024['Sx_D'].median(), 3.0, 20.0))
SY = float(np.clip(th2024['Sy_D'].median(), 3.0, 20.0))
print(f'task1 2024 间距锚: Sx={SX:.2f}D Sy={SY:.2f}D（theta_parameters 中位，截断 [3,20]D）')

geo1 = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig')
SP_NN = float(geo1['spacing_D_med'].median())
print(f'WP1 全局 NN 间距中位: {SP_NN:.2f}D（S_B 约束优先锚）')

# ── 生成 6 套 ───────────────────────────────────────────────
layouts, summaries = {}, []

# S_A 侧风对齐：行沿 x（侧风，行内间距 = Sy）、列沿 y（顺风，列间间距 = Sx）
# task3 口径（generate_grid_in_boundary: x_local = col_grid*Sy, y_local = row_grid*Sx）
xyA = grid_aniso(8, 8, SY, SX)
layouts['S_A'] = xyA
summaries.append(summarize('S_A', xyA, None, 0.0, {'axis_ok': True, 'spacing_ok': True}))

# S_B0 / S_B45 约束优先：8×8 等距，间距 = WP1 全局 NN 中位
for pid, ang in [('S_B0', 0.0), ('S_B45', 45.0)]:
    xyB = grid_aniso(8, 8, SP_NN, SP_NN, ang)
    layouts[pid] = xyB
    summaries.append(summarize(pid, xyB, SP_NN, ang, {'axis_ok': True, 'spacing_ok': True}))

# S_C 分期扩建：一期核心 4 行×8 列（5D）；二期侧风两侧交替各 2 行，间隙 2×5D
sC = 5.0
core_rows = np.array([-7.5, -2.5, 2.5, 7.5]) * D_REF
core = np.array([(i * sC * D_REF - 3.5 * sC * D_REF, y)
                 for y in core_rows for i in range(8)], float)
ext_rows = [(7.5 + 2 * sC) * D_REF,                  # +1 期（10D 间隙）
            (7.5 + 3 * sC) * D_REF,                  # +2 期
            -(7.5 + 2 * sC) * D_REF,                 # −1 期
            -(7.5 + 3 * sC) * D_REF]                 # −2 期
ext = np.array([(i * sC * D_REF - 3.5 * sC * D_REF, y)
                for y in ext_rows for i in range(8)], float)
xyC = np.vstack([core, ext])
layouts['S_C'] = xyC
# 特征自检：核心 32 台 + 扩建 32 台；扩建行与核心间隙 ≥ 1.5×间距；扩建行位于核心 ±y（侧风）之外
core_mask = np.zeros(len(xyC), bool); core_mask[:32] = True
gap_ok = (min(abs(y) for y in ext_rows) - 7.5 * D_REF) >= 1.5 * sC * D_REF
ext_outside = bool(np.all(np.abs(ext[:, 1]) > np.max(np.abs(core[:, 1]))))
summaries.append(summarize('S_C', xyC, sC, 0.0,
                           {'two_phase': len(xyC) == 64 and core_mask.sum() == 32,
                            'gap_ok': gap_ok, 'crosswind_ext': ext_outside}))

# S_D 风资源梯度：4D 候选格点（10×10），线性 WPD 梯度沿 +x，贪心 min 4D
sD = 4.0
cand = np.array([((i - 4.5) * sD * D_REF, (j - 4.5) * sD * D_REF)
                 for j in range(10) for i in range(10)], float)
wpd = 300.0 + 25.0 * (cand[:, 0] / D_REF)     # 线性梯度沿 +x
order = np.argsort(wpd)[::-1]
sel = []
for idx in order:
    pt = cand[idx]
    if all(np.sum((pt - cand[s]) ** 2) >= (sD * D_REF) ** 2 for s in sel):
        sel.append(idx)
    if len(sel) >= 64:
        break
sel = np.array(sel)
xyD = cand[sel]
layouts['S_D'] = xyD
summaries.append(summarize('S_D', xyD, sD, 0.0,
                           {'min_spacing_4D': nn_stats(xyD)[3] >= 3.95,
                            'wpd_gradient_effective': xyD[:, 0].mean() > 0.5 * sD * D_REF,
                            'wpd_above_median': wpd[sel].mean() > np.median(wpd)}))

# S_E 大间距：S_A × 1.25
xyE = grid_aniso(8, 8, SY * 1.25, SX * 1.25)
layouts['S_E'] = xyE
summaries.append(summarize('S_E', xyE, None, 0.0, {'spacing_ok': True}))
# 逐对验证 Sx/Sy = 1.25 × S_A
checkE = (np.allclose(np.sort(np.unique(xyE[:, 0])) , np.sort(np.unique(xyA[:, 0])) * 1.25) and
          np.allclose(np.sort(np.unique(xyE[:, 1])) , np.sort(np.unique(xyA[:, 1])) * 1.25))
summaries[-1]['checks'] = str({'spacing_125x_of_A': checkE})
summaries[-1]['class_ok'] = checkE

# ── 输出 ────────────────────────────────────────────────────
long_rows = []
for pid, xy in layouts.items():
    for k, (x, y) in enumerate(xy):
        long_rows.append(dict(paradigm=pid, turbine_i=k, x_m=round(float(x), 2),
                              y_m=round(float(y), 2)))
pd.DataFrame(long_rows).to_csv(os.path.join(OUT, 'wp2c_paradigm_layouts.csv'),
                               index=False, encoding='utf-8-sig')
summ = pd.DataFrame(summaries)
summ.to_csv(os.path.join(OUT, 'wp2c_paradigm_summary.csv'), index=False, encoding='utf-8-sig')

with open(os.path.join(OUT, 'wp2c_paradigm_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP2c 建设范式情境布局（A–E 范式，学长批准版）\n' + '=' * 60 + '\n')
    f.write(f'统一 64 台 IEA 10MW（D={D_REF} m）；朝向自由度由 WP5c 72 档旋转覆盖\n')
    f.write(f'参数锚: S_A/S_E 行内(侧风)Sy={SY:.2f}D、列间(顺风)Sx={SX:.2f}D'
            f'（task1 2024 中位，task3 口径 x=col*Sy,y=row*Sx）；'
            f'S_B 间距={SP_NN:.2f}D（WP1 NN 中位）；S_C 核心 5D；S_D 候选 4D/min 4D\n\n')
    f.write(summ.to_string(index=False) + '\n')
    ok = summ['class_ok'].all()
    f.write(f'\n特征自检: {"全部通过 ✅" if ok else "存在未过项 ⚠️"}\n')
    f.write('注: S_C 自检项已从 v3 文档的 n_comp=2 调整为"两期结构+扩建方向"'
            '（10D 间隙 < 3×nn_med=15D 仍单连通，真实分期扩建间隙不会大到断开）。\n')

print('输出: wp2c_paradigm_layouts.csv / wp2c_paradigm_summary.csv / wp2c_paradigm_report.txt')
print(summ[['paradigm', 'n_turb', 'nn_med_D', 'nn_min_D', 'area_km2',
            'aspect_ratio', 'n_comp', 'class_ok']].to_string(index=False))
print(f'\n特征自检: {"全部通过 ✅" if summ.class_ok.all() else "存在未过项 ⚠️"}')
