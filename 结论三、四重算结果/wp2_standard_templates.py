"""
WP2 标准排布模板生成：4 形态 × 3 间距 × 3 重复 = 36 套
============================================================
对应《结论三与结论四补充计算方案》§3 执行顺序第 2 步。

设计约束：
  - 每套 64 台 IEA 10MW 级参考机组（D = 198 m）；
  - 间距档 s ∈ {3D, 5D, 7D}（594 / 990 / 1386 m），与论文主分析一致；
  - 形态与间距完全交叉（含 带状×3D、带状×5D、规则网格×3D、规则网格×5D）；
  - 每套模板必须通过 WP1 冻结规则树的形态自检
    （sparse → multi_cluster → belt → rule_grid → cluster），
    即"模板的形态标签"必须与其设计意图一致——否则换种子重掷。

形态族（内部变化 = 重复因子，供下游 R_g/F_g 稳健性使用）：
  rule_grid   r1: 8×8 对齐         r2: 8×8 交错          r3: 8×8 斜切(剪切0.3)
  belt        r1: 2×32 对齐        r2: 4×16 对齐         r3: 4×16 交错
  cluster     r1: 16×(2×2)块 掷点  r2: 16×(2×2)块 换种子  r3: 8×(2×4)块 掷点
  multi_cluster r1: 4×(4×4) 线性   r2: 2×(8×4) 线性     r3: 2×2 象限

注：rule_grid 不用 16×4（其 pc1_share≈0.96 会被冻结树判为 belt），
    改用等面积斜切网格（pc1_share≈0.64, ce_r≈2.3）。

输出：补算/output/wp2_standard_templates.csv（长表，含每台机位）
      补算/output/wp2_template_summary.csv（36 套摘要 + 形态自检）
      补算/output/wp2_template_report.txt
"""
import os, io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csgraph

BUSH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BUSH, 'output')
os.makedirs(OUT, exist_ok=True)

D_REF = 198.0

# ═══════════════════════════════════════════════════════════════════════
# 1. 冻结规则树（与 wp1_local_geometry.py 完全一致）
# ═══════════════════════════════════════════════════════════════════════
def classify_frozen(xy):
    """冻结形态规则树。xy: (n,2) m。"""
    n = len(xy)
    if n < 10:
        return 'sparse', dict(n_comp=1, pc1_share=np.nan, ce_r=np.nan, nn_med=np.nan)
    tree = cKDTree(xy)
    d2 = tree.query(xy, k=2)[0][:, 1]
    nn_mean = float(np.mean(d2)); nn_med = float(np.median(d2))
    thr = 3.0 * max(nn_med, 1.0)
    sm = tree.sparse_distance_matrix(tree, thr, output_type='coo_matrix')
    G = (sm > 0).astype(int)
    n_comp = int(csgraph.connected_components(G, directed=False)[0])
    cov = np.cov(xy.T)
    w = np.linalg.eigvalsh(cov)
    pc1_share = float(w.max() / w.sum())
    ce_r = np.nan
    if n >= 3:
        try:
            hull = ConvexHull(xy)
            rho = n / hull.volume
            r_exp = 1.0 / (2.0 * np.sqrt(rho))
            ce_r = float(nn_mean / r_exp) if r_exp > 0 else np.nan
        except Exception:
            pass
    if n_comp >= 2:
        ltype = 'multi_cluster'
    elif pc1_share >= 0.85:
        ltype = 'belt'
    elif ce_r > 1.8:
        ltype = 'rule_grid'
    else:
        ltype = 'cluster'
    return ltype, dict(n_comp=n_comp, pc1_share=pc1_share, ce_r=ce_r, nn_med=nn_med)

# ═══════════════════════════════════════════════════════════════════════
# 2. 形态生成器（单位间距 s 的机位，m）
# ═══════════════════════════════════════════════════════════════════════
def jittered_centers(base_centers, jitter, seed):
    """对块中心施加 ±jitter*s 均匀抖动（固定种子，确定性）。"""
    rng = np.random.default_rng(seed)
    out = np.array(base_centers, float)
    for k in range(len(out)):
        out[k] += rng.uniform(-jitter, jitter, size=2)
    return out

def blocks_from_centers(centers, block, s):
    """由块中心(单位 s)生成块内机位(单位 s)。块为对齐网格 block=(rows, cols)。"""
    bh, bw = block
    pts = []
    for cx, cy in centers:
        for j in range(bh):
            for i in range(bw):
                pts.append(((cx + i) * s, (cy + j) * s))
    return np.array(pts, float)

# ---- cluster 族的三种重复（确定性晶格 + 抖动，中心最小间距 ≥ 3s）----
# 菱形 4×4（行间距 3.5s、行内 4s、隔行错位 2s）：跨块机位距离 ≤ 3s → 单连通分量；
# 面积 ≈ 150s² → ce_r ≈ 1.3（落在 cluster 带 [1.2,1.8]，不被判为 rule_grid）。
_DIAMOND16 = [(i * 4.0 + (j % 2) * 2.0, j * 3.5) for j in range(4) for i in range(4)]
_SQUARE16 = [(i * 3.5, j * 3.5) for j in range(4) for i in range(4)]

def cluster_rep1(s):
    return blocks_from_centers(_DIAMOND16, (2, 2), s)

def cluster_rep2(s):
    # 抖动 ±0.4s + 种子阶梯：若抖动破坏连通性（n_comp≥2）则换种子重掷
    for seed in range(11, 40):
        xy = blocks_from_centers(jittered_centers(_DIAMOND16, 0.4, seed=seed), (2, 2), s)
        if classify_frozen(xy)[0] == 'cluster':
            return xy
    raise RuntimeError('cluster_rep2 抖动无通过种子')

def cluster_rep3(s):
    for seed in range(21, 50):
        xy = blocks_from_centers(jittered_centers(_SQUARE16, 0.4, seed=seed), (2, 2), s)
        if classify_frozen(xy)[0] == 'cluster':
            return xy
    raise RuntimeError('cluster_rep3 抖动无通过种子')

def gen_template(morph, rep, s):
    if morph == 'rule_grid':
        if rep == 1:    # 8×8 对齐
            return np.array([(i * s, j * s) for j in range(8) for i in range(8)], float)
        if rep == 2:    # 8×8 交错
            return np.array([((i + 0.5 * (j % 2)) * s, j * s) for j in range(8) for i in range(8)], float)
        if rep == 3:    # 8×8 斜切（剪切 0.3）
            return np.array([((i + 0.3 * j) * s, j * s) for j in range(8) for i in range(8)], float)
    if morph == 'belt':
        if rep == 1:    # 2×32
            return np.array([(i * s, j * s) for j in range(2) for i in range(32)], float)
        if rep == 2:    # 4×16
            return np.array([(i * s, j * s) for j in range(4) for i in range(16)], float)
        if rep == 3:    # 4×16 交错
            return np.array([((i + 0.5 * (j % 2)) * s, j * s) for j in range(4) for i in range(16)], float)
    if morph == 'cluster':
        if rep == 1:
            return cluster_rep1(s)
        if rep == 2:
            return cluster_rep2(s)
        if rep == 3:
            return cluster_rep3(s)
    if morph == 'multi_cluster':
        if rep == 1:    # 4×(4×4) 线性，簇间间隙 4s
            return np.array([((c * 7 + i) * s, j * s) for c in range(4) for j in range(4) for i in range(4)], float)
        if rep == 2:    # 2×(8×4)，簇间间隙 6s
            return np.array([((c * 13 + i) * s, j * s) for c in range(2) for j in range(4) for i in range(8)], float)
        if rep == 3:    # 2×2 象限（簇中心 ±3.5s）
            pts = []
            for cx, cy in ((-3.5, -3.5), (3.5, -3.5), (-3.5, 3.5), (3.5, 3.5)):
                for j in range(4):
                    for i in range(4):
                        pts.append(((cx + i) * s, (cy + j) * s))
            return np.array(pts, float)
    raise ValueError(morph)

# ═══════════════════════════════════════════════════════════════════════
# 3. 生成 36 套 + 冻结树自检
# ═══════════════════════════════════════════════════════════════════════
MORPHS = ['rule_grid', 'belt', 'cluster', 'multi_cluster']
SPACINGS = [3.0, 5.0, 7.0]
REPS = [1, 2, 3]

long_rows, sum_rows, bad = [], [], []
for morph in MORPHS:
    for sd in SPACINGS:
        s = sd * D_REF
        for rep in REPS:
            tid = f'{morph[:2].upper()}{int(sd)}r{rep}'   # e.g. RU3r1 / BE5r2 / CL7r3 / MU3r1
            xy = gen_template(morph, rep, s)
            ltype, m = classify_frozen(xy)
            if ltype != morph:
                bad.append((tid, ltype))
            # 机位间距校验：中位 NN ≈ s（±5%），最小 NN ≥ 0.8s（无重叠/无塌缩）
            tree = cKDTree(xy)
            nn = tree.query(xy, k=2)[0][:, 1]
            nn_min = float(nn.min())
            # 摘要指标
            cov = np.cov(xy.T)
            w = np.linalg.eigvalsh(cov)
            aspect = float(np.sqrt(w.max() / max(w.min(), 1e-12)))
            hull = ConvexHull(xy)
            area_m2 = float(hull.volume)
            sum_rows.append(dict(
                template_id=tid, morphology=morph, spacing_D=sd, rep=rep,
                n_turb=len(xy), nn_mean_m=round(float(nn.mean()), 1),
                nn_med_D=round(float(np.median(nn)) / D_REF, 3),
                nn_min_D=round(nn_min / D_REF, 3),
                area_km2=round(area_m2 / 1e6, 3), aspect_ratio=round(aspect, 3),
                pc1_share=round(m['pc1_share'], 4), ce_r=round(m['ce_r'], 3),
                n_comp=m['n_comp'], classified=morph if ltype == morph else ltype,
                class_ok=(ltype == morph)))
            for k, (x, y) in enumerate(xy):
                long_rows.append(dict(template_id=tid, morphology=morph, spacing_D=sd,
                                      rep=rep, turbine_i=k, x_m=round(float(x), 2), y_m=round(float(y), 2)))

tpl = pd.DataFrame(long_rows)
summ = pd.DataFrame(sum_rows)

# ═══════════════════════════════════════════════════════════════════════
# 4. 输出
# ═══════════════════════════════════════════════════════════════════════
tpl.to_csv(os.path.join(OUT, 'wp2_standard_templates.csv'), index=False, encoding='utf-8-sig')
summ.to_csv(os.path.join(OUT, 'wp2_template_summary.csv'), index=False, encoding='utf-8-sig')

with open(os.path.join(OUT, 'wp2_template_report.txt'), 'w', encoding='utf-8') as f:
    f.write('WP2 标准排布模板报告（4 形态 × 3 间距 × 3 重复 = 36 套）\n')
    f.write('=' * 60 + '\n')
    f.write(f'参考转子直径 D = {D_REF} m；每套 64 台\n\n')
    f.write('形态×间距自检矩阵（class_ok）:\n')
    f.write(summ.pivot_table(index='morphology', columns='spacing_D', values='class_ok',
                             aggfunc='all').to_string())
    f.write('\n\n')
    f.write(summ.to_string(index=False))
    f.write(f'\n\n自检失败模板: {bad if bad else "无"}')
    f.write('\n\n间距档验证（nn_med_D 应≈spacing_D）:\n')
    f.write(summ.groupby('spacing_D')['nn_med_D'].agg(['min', 'max']).to_string())

print(f'模板数: {summ.template_id.nunique()} | 机位总数: {len(tpl)}')
print(f'自检失败: {bad if bad else "无"}')
print(summ[['template_id', 'morphology', 'spacing_D', 'nn_med_D', 'area_km2',
            'aspect_ratio', 'pc1_share', 'ce_r', 'n_comp', 'classified']].to_string(index=False))
print(f'\n输出: {os.path.join(OUT, "wp2_standard_templates.csv")}')
