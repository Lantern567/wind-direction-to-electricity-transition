"""
机制重算 v2：把"尾流池 × 窄玫瑰"换成可前向计算的三因子
===========================================================
诊断：原 §2.3 用 wake_pool（场均尾流损失）当"尾流池"。但旋转不改变损失总量、
只重新分配，因此可被朝向回收的从来不是池子多大，而是池子随来流方向的起伏多大。

本脚本用**本地真实机位**直接计算方向响应曲线 L(θ)，并给出三个可算因子：
  F1 几何各向异性   L_range      = L(θ) 的谷峰差            （无需风玫瑰）
  F2 能量转化率     已隐含在 Lw(θ) 中（按各场 Weibull 风速分布加权）
  F3 玫瑰选择性     A_pred       = 玫瑰 ⊛ Lw 的振幅          （风玫瑰为弱项，见 README）

输入（全部本地）：
  offshore-task0-HuTingxian/output/task0/turbine_coordinates.csv
  offshore-task2/data/iea_10MW.yaml
  补算/output/task1_training_data.csv        （A_actual, Weibull, WCI …）
  task1_output/task1_wind_metrics.csv        （WCI_yearly, theta_energy_yearly）
输出：
  补算/output/mechanism_v2_curves.csv        逐场 L(θ) / Lw(θ) 曲线（72 个方向）
  补算/output/mechanism_v2_metrics.csv       逐场机制指标 + 回测实测 A
"""
import os, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, yaml
from scipy.special import i0, i1
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)

COORD = os.path.join(ROOT, 'offshore-task0-HuTingxian', 'output', 'task0', 'turbine_coordinates.csv')
YAML = os.path.join(ROOT, 'offshore-task2', 'data', 'iea_10MW.yaml')
TRAIN = os.path.join(OUT, 'task1_training_data.csv')
WMET = os.path.join(ROOT, 'task1_output', 'task1_wind_metrics.csv')

# ---------------- 模型常数（与 offshore-task2 主口径一致） ----------------
K_WAKE = 0.05          # Jensen 尾流扩张系数（海上）
TH = np.arange(0, 360, 5.0)          # 72 个来流方向
WS = np.arange(3.0, 26.0, 1.0)       # 23 个风速箱
YEAR = 2024                          # 终期配置

with open(YAML, 'r', encoding='utf-8') as f:
    ty = yaml.safe_load(f)
D_ROTOR = float(ty['rotor_diameter'])
pt = ty['power_thrust_table']
PC_WS = np.array(pt['wind_speed'], dtype=float)
PC_P = np.array(pt['power'], dtype=float)
PC_CT = np.array(pt['thrust_coefficient'], dtype=float)
P_RATED = float(PC_P.max())


def power_of(u):
    return np.interp(u, PC_WS, PC_P, left=0.0, right=0.0)


def ct_of(u):
    return np.interp(u, PC_WS, PC_CT, left=0.0, right=0.0)


def overlap_frac(dy, rw):
    """转子盘(半径 0.5D)与尾流盘(半径 rw)的面积重叠份额，dy/rw 单位=D。"""
    r0 = 0.5
    d = np.abs(dy)
    out = np.zeros_like(d)
    full = d <= np.abs(rw - r0)
    out[full] = np.minimum(1.0, (rw[full] / r0) ** 2)
    part = (~full) & (d < rw + r0)
    if part.any():
        dd, R, r = d[part], rw[part], r0
        c1 = np.clip((dd ** 2 + R ** 2 - r ** 2) / (2 * dd * R), -1, 1)
        c2 = np.clip((dd ** 2 + r ** 2 - R ** 2) / (2 * dd * r), -1, 1)
        area = (R ** 2 * np.arccos(c1) + r ** 2 * np.arccos(c2)
                - 0.5 * np.sqrt(np.maximum(0.0, (-dd + R + r) * (dd + R - r)
                                           * (dd - R + r) * (dd + R + r))))
        out[part] = area / (np.pi * r ** 2)
    return np.clip(out, 0.0, 1.0)


def geometry_kernel(xs_D, ys_D):
    """返回 Q[theta, i] —— 与风速无关的几何亏损核（Jensen，平方和叠加，部分重叠加权）。
       实际亏损 = A0(ws) * Q ，其中 A0 = 1-sqrt(1-Ct(ws))。"""
    n = len(xs_D)
    Q = np.zeros((len(TH), n))
    for it, t in enumerate(np.radians(TH)):
        xr = xs_D * np.cos(t) + ys_D * np.sin(t)      # 顺风坐标
        yr = -xs_D * np.sin(t) + ys_D * np.cos(t)     # 侧风坐标
        dx = xr[:, None] - xr[None, :]                # >0: j 在 i 上游
        dy = yr[:, None] - yr[None, :]
        up = dx > 1e-6
        rw = 0.5 + K_WAKE * np.where(up, dx, 0.0)
        ov = np.where(up, overlap_frac(dy, rw), 0.0)
        amp = np.where(up, 1.0 / (1.0 + 2.0 * K_WAKE * dx) ** 2, 0.0) * ov
        Q[it] = np.sqrt((amp ** 2).sum(axis=1))       # 平方和叠加
    return Q


def directional_response(Q, weib_A, weib_k):
    """返回 (L_geo, L_energy)：
       L_geo    —— 固定 9 m/s 下的功率损失份额随方向（纯几何）
       L_energy —— 按各场 Weibull 风速分布能量加权的 AEP 损失份额随方向
                   （自动包含"只有低于额定的小时尾流才折算成损失"这一项）"""
    n = Q.shape[1]
    # 纯几何：9 m/s
    u0 = 9.0
    d0 = (1 - np.sqrt(1 - ct_of(u0))) * Q
    p0 = power_of(u0 * (1 - np.clip(d0, 0, 0.95))).sum(axis=1)
    L_geo = 1.0 - p0 / (n * power_of(u0))

    # 能量加权
    w = (weib_k / weib_A) * (WS / weib_A) ** (weib_k - 1) * np.exp(-(WS / weib_A) ** weib_k)
    w = w / w.sum()
    num = np.zeros(len(TH)); den = 0.0
    for iw, u in enumerate(WS):
        a0 = 1 - np.sqrt(1 - ct_of(u))
        pw = power_of(u * (1 - np.clip(a0 * Q, 0, 0.95))).sum(axis=1)
        num += w[iw] * pw
        den += w[iw] * n * power_of(u)
    L_en = 1.0 - num / den
    return L_geo, L_en


def kappa_from_rbar(r):
    r = float(np.clip(r, 1e-3, 0.95))
    return brentq(lambda k: i1(k) / i0(k) - r, 1e-4, 200.0)


def heading_curve(L, rose):
    """年均损失随阵列朝向 phi 的变化 = 玫瑰与 L 的循环卷积。"""
    return np.array([(rose * np.roll(L, -j)).sum() for j in range(len(L))])


# ======================== 主流程 ========================
print('加载数据 ...')
co = pd.read_csv(COORD)
co = co[co.year == YEAR]
tr = pd.read_csv(TRAIN)
wm = pd.read_csv(WMET).dropna(subset=['WCI_yearly', 'theta_energy_yearly'])
g = wm.groupby('farm_id')
WCI = g['WCI_yearly'].mean()
MU = g['theta_energy_yearly'].apply(
    lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(),
                                    np.cos(np.radians(s)).mean())) % 360)
wb = tr.set_index('farm_id')[['weibull_A', 'weibull_k']]
wbA_med, wbk_med = wb.weibull_A.median(), wb.weibull_k.median()
print(f'  {co.farm_id.nunique()} 场 / {len(co)} 台机位（{YEAR} 终期配置）'
      f'，转子 {D_ROTOR:.0f} m，额定 {P_RATED/1000:.1f} MW')

rows, curves = [], []
t0 = time.time()
for i, (fid, sub) in enumerate(co.groupby('farm_id')):
    if len(sub) < 2:
        continue
    xs = (sub.x_m.values - sub.x_m.mean()) / D_ROTOR
    ys = (sub.y_m.values - sub.y_m.mean()) / D_ROTOR
    Q = geometry_kernel(xs, ys)
    wA = float(wb.weibull_A.get(fid, wbA_med)); wk = float(wb.weibull_k.get(fid, wbk_med))
    L_geo, L_en = directional_response(Q, wA, wk)

    rec = dict(farm_id=fid, n_turb=len(sub),
               L_geo_mean=L_geo.mean() * 100, L_geo_range=(L_geo.max() - L_geo.min()) * 100,
               Lw_mean=L_en.mean() * 100, Lw_range=(L_en.max() - L_en.min()) * 100,
               Lw_cv=L_en.std() / max(L_en.mean(), 1e-9))
    # 朝向可回收（需玫瑰）：von Mises(kappa<-WCI, mu<-theta_energy)
    if fid in WCI.index:
        kap = kappa_from_rbar(WCI[fid])
        rose = np.exp(kap * np.cos(np.radians(TH) - np.radians(MU[fid])))
        rose /= rose.sum()
        ann = heading_curve(L_en, rose)
        aep = 1.0 - ann
        rec.update(kappa=kap, WCI=float(WCI[fid]), theta_energy=float(MU[fid]),
                   A_pred=(aep.max() - aep.mean()) / aep.mean() * 100,
                   amp_pred=(aep.max() - aep.min()) / aep.mean() * 100,
                   theta_opt_pred=float(TH[np.argmax(aep)]))
        # 罚分平坦度 / 等效偏差（方案 §6 的〔待补 等效偏差指标〕）
        rel = (aep - aep.min()) / max(aep.max() - aep.min(), 1e-12)
        rec['flat_frac_within_1pct'] = float(
            ((aep.max() - aep) / aep.max() < 0.01).mean())
        rec['penalty_flatness'] = float(rel.mean())
    rows.append(rec)
    curves.append(pd.DataFrame({'farm_id': fid, 'theta_deg': TH,
                                'L_geo': L_geo, 'L_energy': L_en}))
    if (i + 1) % 40 == 0:
        print(f'  [{i+1}/{co.farm_id.nunique()}] {time.time()-t0:.0f}s')

met = pd.DataFrame(rows)
met = met.merge(tr[['farm_id', 'A', 'wake_pool', 'WCI', 'spacing_D', 'aspect_ratio',
                    'pc1_share', 'ws_mean', 'ws_std', 'weibull_A', 'weibull_k',
                    'frac_below_rated', 'wd_entropy_norm', 'orient_sensitivity',
                    'lat', 'lon', 'country']].rename(columns={'WCI': 'WCI_train'}),
                on='farm_id', how='left')
met.to_csv(os.path.join(OUT, 'mechanism_v2_metrics.csv'), index=False, encoding='utf-8-sig')
pd.concat(curves).to_csv(os.path.join(OUT, 'mechanism_v2_curves.csv'),
                         index=False, encoding='utf-8-sig')
print(f'\n完成 {len(met)} 场，用时 {time.time()-t0:.0f}s')
print(f'  -> output/mechanism_v2_metrics.csv')
print(f'  -> output/mechanism_v2_curves.csv')
print(met[['L_geo_range', 'Lw_range', 'Lw_mean', 'A_pred', 'A']].describe().round(2).to_string())
