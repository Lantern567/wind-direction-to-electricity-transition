"""
WP6b 验证（FLORIS 直接计算部分）：旋转等价 + 1° 直接验证 + p 重分箱
====================================================================
验收红线（方案 §4.6）：
  R1 旋转排布与反向旋转风况等价（旋转坐标直算 vs 查表平移）
  R2 5° 相对 1° 加密 |ΔA| ≤ 0.3 pp（三条独立证据链）
     a. η 傅里叶插值全量敏感性（wp6a）
     b. 8 组合（4 形态 × 2 高 A 场址）FLORIS 1° 风向直算 A vs 查表 5° A
     c. p 的 1° 方向重分箱（20 场址抽样，重提取 ERA5 逐时）

必须在 wp4 + wp5 完成后运行（依赖查表与交叉仿真输出）。
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import netCDF4

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import create_floris_model, get_ti_for_farm, get_turbine_params

HOURS = 8760.0
NC_DIR = os.path.join(REPO, 'data', '111')

z4 = np.load(os.path.join(OUT, 'wp4_wake_lookup.npz'))
ETA = z4['eta'].astype(np.float64)          # (36,5,18,72)
P0 = z4['P0'].astype(np.float64)
WS = z4['ws']; WD = z4['wd']; TI_VALS = z4['ti']
TIDS = list(z4['tid']); XY = z4['xy']
summ = pd.read_csv(os.path.join(OUT, 'wp2_template_summary.csv'), encoding='utf-8-sig')
MORPH = summ['morphology'].values; SPAC = summ['spacing_D'].values; REP = summ['rep'].values

z5 = np.load(os.path.join(OUT, 'wp5_cross_farms.npz'))
A_farm = z5['A']; th_star = z5['th_star']; E_cf = z5['E_cf']
farm_ids = z5['farm_ids'].astype(int)
z3 = np.load(os.path.join(OUT, 'wp3_climate_joint.npz'))
p_farm = z3['p_fy'].astype(np.float64)
geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')

def ti_idx(f):
    return int(np.argmin(np.abs(np.array(TI_VALS) - get_ti_for_farm(geo.loc[f, 'cent_lat'], geo.loc[f, 'cent_lon']))))

def floris_eta(x, y, ti, dirs, n_speed=None):
    """FLORIS 直算 η：返回 (n_speed, len(dirs))。dirs 一维、n_speed=len(WS) 时按速度批量。"""
    n_speed = len(WS) if n_speed is None else n_speed
    fm, tmp = create_floris_model(x, y, turbine_type='iea_10MW', ti=ti,
                                  alpha=0.11, wake_model_name='gauss')
    P = np.zeros((n_speed, len(dirs)))
    for i in range(n_speed):
        fm.set(wind_speeds=np.full(len(dirs), float(WS[i])),
               wind_directions=dirs.astype(float),
               turbulence_intensities=np.full(len(dirs), ti))
        fm.run()
        P[i] = fm.get_farm_power().flatten()
    os.unlink(tmp)
    with np.errstate(divide='ignore', invalid='ignore'):
        e = P / (len(x) * P0[:n_speed, None])
    e[~np.isfinite(e)] = 1.0
    e[P0[:n_speed] < 1e-3, :] = 1.0
    return e

report = {}

# ═══════════════════════════════════════════════════════════════════════
# R1 旋转等价：RU5r1 / BE5r1 旋转 45°、90° 后直算 vs 查表 η(u,d−θ)
# ═══════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
lines = []
for tid in ['RU5r1', 'BE5r1']:
    k = TIDS.index(tid)
    x0 = XY[k, :, 0]; y0 = XY[k, :, 1]
    cx, cy = x0.mean(), y0.mean()
    j_ti = int(np.argmin(np.abs(np.array(TI_VALS) - 0.07)))
    for th in [45.0, 90.0]:
        rad = np.radians(th)
        xr = (x0 - cx) * np.cos(rad) - (y0 - cy) * np.sin(rad)
        yr = (x0 - cx) * np.sin(rad) + (y0 - cy) * np.cos(rad)
        e_rot = floris_eta(xr, yr, 0.07, WD)
        # 查表参考：排布逆时针 θ ≡ 风况平移 +θ → η(u, (d+θ) mod 72)
        # （45° belt 决定性检验：+shift 匹配 0.003，−shift 失配 0.469）
        shift = int(round(th / 5.0))
        e_ref = ETA[k, j_ti][:, np.roll(np.arange(72), -shift)]   # 先取 2D，避免高级索引轴前移
        diff = np.abs(e_rot - e_ref)
        rel = diff / np.maximum(np.abs(e_ref).mean(), 1e-6)
        lines.append(f'{tid} θ={th:5.0f}°: max|Δη|={diff.max():.5f}  '
                     f'mean|Δη|={diff.mean():.5f}  max|Δη|/η̄={rel.max():.4f}')
        print(lines[-1])
report['R1'] = '\n'.join(lines) + '\n验收: max|Δη| < 0.005（0.5%）'

# ═══════════════════════════════════════════════════════════════════════
# R2b 1° 直接验证：8 组合 FLORIS 360 风向直算 A vs 查表 5° A
# ═══════════════════════════════════════════════════════════════════════
A_mean = A_farm.mean(axis=1)
top2 = farm_ids[np.argsort(-A_mean)[:2]]           # 两个最高 A 场址
WD1 = np.arange(360.0)
IDX5 = (np.arange(72)[None, :] * 5 + np.arange(180)[:, None]) % 360   # (180,72) θ=排布逆时针
lines = []
for f in top2:
    i = list(farm_ids).index(int(f))
    p = p_farm[i].mean(axis=0)                     # (18,72) 多年平均
    j_ti = ti_idx(int(f))
    for m in ['rule_grid', 'belt', 'cluster', 'multi_cluster']:
        cand = [k for k in range(36) if MORPH[k] == m and SPAC[k] == 5.0 and REP[k] == 1]
        k = cand[0]
        x = XY[k, :, 0]; y = XY[k, :, 1]
        e1 = floris_eta(x, y, TI_VALS[j_ti], WD1)              # (18,360) 1° 直算
        Hp = e1[:, IDX5] * P0[:, None, None]                   # (18,180,72)
        E1 = np.tensordot(Hp, p, axes=([0, 2], [0, 1])) * HOURS / 1000.0   # (180,)
        A1 = 100 * (E1.max() - E1.mean()) / np.maximum(E1.mean(), 1e-9)
        A5 = A_farm[i, k]
        th1 = int(np.argmax(E1))
        lines.append(f'farm {int(f):3d} {m:>12s} 5D: A(1°)={A1:.3f}  A(5°)={A5:.3f}  '
                     f'ΔA={A1-A5:+.3f}  θ*(1°)={th1} vs {int(th_star[i,k])}')
        print(lines[-1])
report['R2b'] = '\n'.join(lines) + '\n验收: |ΔA| ≤ 0.3 pp'

# ═══════════════════════════════════════════════════════════════════════
# R2c p 的 1° 重分箱敏感性：20 场址抽样（按 A 均值分层）
# ═══════════════════════════════════════════════════════════════════════
H_TURB = get_turbine_params('iea_10MW')['H']
os.chdir(NC_DIR)                                   # netCDF4 中文路径 workaround
years = list(range(2014, 2025))
order = np.argsort(A_mean)
samp = np.sort(np.concatenate([order[np.linspace(0, len(order)-1, 20, dtype=int)]]))
WS_EDGES = np.concatenate([[3.0 - 0.5], (WS[:-1] + WS[1:]) / 2.0, [np.inf]])
WD1_EDGES = np.arange(-0.5, 360.5, 1.0)
def region_of(lat, lon):
    if 8 <= lat <= 44 and 104 <= lon <= 143: return 'east_asia'
    if 39 <= lat <= 63 and -12 <= lon <= 32: return 'europe'
    if 36 <= lat <= 42 and -78 <= lon <= -68: return 'us_east'
    if 30 <= lat <= 46 and 128 <= lon <= 146: return 'japan'
    return 'east_asia'
dA_p = []
for i in samp:
    f = int(farm_ids[i])
    g = geo.loc[f]
    reg = region_of(g.cent_lat, g.cent_lon)
    H = None
    for yr in years:
        nc = f'era5_{reg}_{yr}.nc'
        if not os.path.exists(nc):
            continue
        ds = netCDF4.Dataset(nc, 'r')
        la = np.array(ds['latitude'][:], float); lo = np.array(ds['longitude'][:], float)
        ilat = int(np.argmin(np.abs(la - g.cent_lat))); ilon = int(np.argmin(np.abs(lo - g.cent_lon)))
        u = np.array(ds['u100'][:, ilat, ilon], float); v = np.array(ds['v100'][:, ilat, ilon], float)
        ds.close()
        ws_100 = np.sqrt(u**2 + v**2)
        wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        ws_hub = ws_100 * (H_TURB / 100.0) ** 0.11
        hh, _, _ = np.histogram2d(wd, ws_hub, bins=[WD1_EDGES, WS_EDGES])
        H = hh.T if H is None else H + hh.T
    p1 = H / H.sum()                                # (18,360) 1° 方向
    j_ti = ti_idx(f)
    IDX72 = ((np.arange(360)[None, :] + 5 * np.arange(36)[:, None]) % 360) // 5   # (36,360) θ=排布逆时针
    for k in range(36):
        e5 = ETA[k, j_ti]                           # (18,72)
        G = p1[:, IDX72] * e5[:, IDX72]             # (18,36,360)
        E = np.einsum('u,utd->t', P0, G)            # E[θ] = Σ_u P0[u]·Σ_d1 p1[u,d1]·e5[u,(d1+θ)/5 mod 72]
        A1p = 100 * (E.max() - E.mean()) / np.maximum(E.mean(), 1e-9)
        dA_p.append(A1p - A_farm[i, k])
dA_p = np.array(dA_p)
report['R2c'] = (f'p 1° 重分箱（20 场址 × 36 模板）: |ΔA| p50={np.percentile(np.abs(dA_p),50):.3f} '
                 f'p95={np.percentile(np.abs(dA_p),95):.3f} max={np.abs(dA_p).max():.3f} pp')
print(report['R2c'])

with open(os.path.join(OUT, 'wp6b_report.txt'), 'w', encoding='utf-8') as fp:
    fp.write('WP6b FLORIS 直接验证报告\n' + '=' * 60 + '\n')
    fp.write('R1 旋转等价:\n' + report['R1'] + '\n\n')
    fp.write('R2b 1° 直算（8 组合）:\n' + report['R2b'] + '\n\n')
    fp.write('R2c p 重分箱:\n' + report['R2c'] + '\n')
print(f'\n完成 | 总耗时 {(time.perf_counter()-t0)/60:.1f} min | 输出 wp6b_report.txt')
