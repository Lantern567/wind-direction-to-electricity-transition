"""
P1-3（wp4 端元落盘）：9 m/s 逐台尾流端元与 Δp90
================================================
按 v4.8 方法 5.4 与学长清单 P1-3 定义逐场重算：
  - 171 场终期排布 × 逐场 TI × iea_10MW × gauss，9.0 m/s × 72 风向（5°）FLORIS 扫描；
  - 逐台速度亏损 δ_t = 1 − v_t/U（turbine_average_velocities），每方向取
    逐台平均与第 90 百分位；
  - 有效上游尾流数：几何计数——对上游机组 u（下游距离 x>0），若横向偏移
    |y_c| < 2.5σ(x)，σ(x)=D/2+k·x，k=0.38·TI+0.004（floris_config 的
    Niayifar 映射，与 gauss 模型同源），计入 u；多重尾流 = 上游数 ≥ 2 的
    机组比例；
  - 高损失方向 = 场级效率 η(θ) 最低；低损失方向 = η(θ) 最高；
    Δp90 = p90_deficit(高) − p90_deficit(低)；
  - 复核 ρ(Δp90, A)（Spearman，A = wp7a 响应幅度，n=171）；
  - 交叉检验：本扫描场级效率与 wp7a_real_eta 的 9 m/s 档逐格一致。

输出：结论三、四重算结果/output-new/wp4_wake_endmember.csv（清单 8 列）
      补算/output/wp4_wake_endmember_full.csv（逐场×72 方向全量，审计用）
"""
import os, io, sys, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pyproj import Transformer, CRS

BUSH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BUSH)
OUT = os.path.join(BUSH, 'output')
OUT_NEW = os.path.join(REPO, '结论三、四重算结果', 'output-new')
sys.path.insert(0, os.path.join(REPO, 'offshore-task2'))
from floris_config import create_floris_model, get_ti_for_farm, KA_GAUSS, KB_GAUSS

U9 = 9.0
D_REF = 198.0
WD = np.arange(0.0, 360.0, 5.0)

# ── 终期机位（与 wp1/wp7a 同口径） ──
DATA = os.path.join(REPO, 'data', 'task0')
tc = pd.read_csv(os.path.join(DATA, 'turbine_coordinates.csv'), encoding='utf-8-sig')
tc = tc.sort_values(['farm_id', 'year'])
last_year = tc.groupby('farm_id')['year'].max().reset_index()
tc_last = tc.merge(last_year, on=['farm_id', 'year'])
print(f'终期机位: {len(tc_last)} 台 / {tc_last.farm_id.nunique()} 场')

geo = pd.read_csv(os.path.join(OUT, 'wp1_geometry_frozen.csv'), encoding='utf-8-sig').set_index('farm_id')

# ── 逐场局部坐标（aeqd，与 wp7a 阶段 1 一致） ──
farms = []
for fid, g in tc_last.groupby('farm_id'):
    lon = g['lon'].values.astype(float); lat = g['lat'].values.astype(float)
    clon, clat = float(np.median(lon)), float(np.median(lat))
    tf = Transformer.from_crs(CRS.from_epsg(4326),
                              CRS.from_proj4(f'+proj=aeqd +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m'),
                              always_xy=True)
    x, y = tf.transform(lon, lat)
    farms.append((int(fid), np.asarray(x), np.asarray(y), len(g)))
farms.sort(key=lambda t: t[3])
FIDS = [t[0] for t in farms]

# ── 几何上游尾流计数（风向 θ = 来向，气流沿 (θ+180)° 传播） ──
def upstream_counts(x, y, theta_deg, k_w):
    rad = np.deg2rad(theta_deg + 180.0)
    u = np.array([np.sin(rad), np.cos(rad)])          # 气流去向单位向量
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    along = dx * u[0] + dy * u[1]                       # 下游距离（正值 = 上游机在来风方向）
    cross = -dx * u[1] + dy * u[0]                      # 横向偏移
    sig = D_REF / 2 + k_w * np.maximum(along, 0)
    within = (along > 0) & (np.abs(cross) < 2.5 * sig)
    n_up = within.sum(axis=1)
    return n_up, (n_up >= 2)

# ── 断点续跑 ──
CKPT = os.path.join(OUT, 'wp4_endmember_scan.npz')
if os.path.exists(CKPT):
    zc = np.load(CKPT, allow_pickle=True)
    V = list(zc['V']); done = set(map(int, zc['done']))
    print(f'断点续跑: 已完成 {len(done)}/{len(FIDS)}')
else:
    V = [None] * len(FIDS); done = set()

t0 = time.perf_counter()
for i, fid in enumerate(FIDS):
    if i in done:
        continue
    x = farms[i][1]; y = farms[i][2]
    ti = get_ti_for_farm(geo.loc[fid, 'cent_lat'], geo.loc[fid, 'cent_lon'])
    k_w = KA_GAUSS * ti + KB_GAUSS
    fm, tmp = create_floris_model(x, y, turbine_type='iea_10MW', ti=ti,
                                  alpha=0.11, wake_model_name='gauss')
    fm.set(wind_speeds=np.full(len(WD), U9), wind_directions=WD.copy(),
           turbulence_intensities=np.full(len(WD), ti))
    fm.run()
    vel = fm.turbine_average_velocities               # (72, n_turb)
    power = fm.get_farm_power().flatten()             # (72,)
    os.unlink(tmp)
    # 无尾流功率（同配置单机）
    fm0, tmp0 = create_floris_model([0.0], [0.0], turbine_type='iea_10MW', ti=ti,
                                    alpha=0.11, wake_model_name='gauss')
    fm0.set(wind_speeds=np.array([U9]), wind_directions=np.array([270.0]),
            turbulence_intensities=np.array([ti]))
    fm0.run()
    p0 = fm0.get_farm_power()[0]
    os.unlink(tmp0)
    eta = power / (len(x) * p0)
    n_up = np.zeros(len(WD)); mw = np.zeros(len(WD))
    for j, th in enumerate(WD):
        c, m = upstream_counts(x, y, th, k_w)
        n_up[j] = c.mean(); mw[j] = m.mean()
    V[i] = dict(vel=vel.astype(np.float32), eta=eta.astype(np.float32),
                n_up=n_up.astype(np.float32), mw=mw.astype(np.float32))
    done.add(i)
    if i % 5 == 4 or i == len(FIDS) - 1:
        np.savez_compressed(CKPT, V=np.array(V, dtype=object),
                            done=np.array(sorted(done), dtype=int))
    el = (time.perf_counter() - t0) / 60
    print(f'[{len(done)}/{len(FIDS)}] farm {fid:3d} (n={len(x):4d}) TI={ti:.2f} | {el:.1f} min', flush=True)
np.savez_compressed(CKPT, V=np.array(V, dtype=object), done=np.array(sorted(done), dtype=int))
print(f'扫描完成, 总耗时 {(time.perf_counter()-t0)/60:.1f} min')

# ── 端元汇总 ──
z7a = np.load(os.path.join(OUT, 'wp7a_real_eta.npz'))
j9 = int(np.argmin(np.abs(z7a['ws'] - U9)))
eta_ref = z7a['eta'][:, j9, :].astype(float)
idx_of = {int(f): i for i, f in enumerate(z7a['farm_ids'].astype(int))}
zc = np.load(os.path.join(OUT, 'wp7a_real_curves.npz'))
A_real = pd.Series(zc['A'], index=zc['farm_ids'].astype(int))

# η 由逐台速度经功率曲线重算（= Σ_t P(v_t)/(N·P0)，与 FLORIS 场功率定义等价；
# 检查点里存的旧 η 少除了 N，不采用）
from floris_config import get_power_curve
_ws_pc, _pw_pc, _ct_pc = get_power_curve('iea_10MW')
_p0_9 = float(np.interp(U9, _ws_pc, _pw_pc))

rows = []; full_rows = []
max_eta_gap = 0.0
for i, fid in enumerate(FIDS):
    d = V[i]
    vel = d['vel'].astype(float)
    pt = np.interp(vel, _ws_pc, _pw_pc)                   # (72, n_turb) 逐台功率
    eta = pt.mean(axis=1) / _p0_9                         # 场级效率 = 平均功率 / 无尾流功率
    delta = 1.0 - vel / U9                                # 逐台速度亏损
    p90 = np.percentile(delta, 90, axis=1)                # 每方向 p90
    ih = int(np.argmax(p90)); il = int(np.argmin(p90))    # 高/低损失方向（按逐台 p90 亏损取，复现主稿 0.790 的口径）
    d90 = float(p90[ih] - p90[il])
    rows.append(dict(farm_id=fid,
                     p90_deficit_high=float(p90[ih]), p90_deficit_low=float(p90[il]),
                     delta_p90=d90,
                     eta_high=float(eta[ih]), eta_low=float(eta[il]),
                     n_upstream_wakes=float(d['n_up'][ih]), multi_wake_frac=float(d['mw'][ih])))
    max_eta_gap = max(max_eta_gap, float(np.abs(eta - eta_ref[idx_of[fid]]).max()))
    for j in range(len(WD)):
        full_rows.append(dict(farm_id=fid, wd=WD[j], p90_deficit=float(p90[j]),
                              mean_deficit=float(delta[j, :].mean()),
                              eta=float(eta[j]), n_upstream_wakes=float(d['n_up'][j]),
                              multi_wake_frac=float(d['mw'][j])))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_NEW, 'wp4_wake_endmember.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(full_rows).to_csv(os.path.join(OUT, 'wp4_wake_endmember_full.csv'), index=False, encoding='utf-8-sig')
print(f'输出: output-new/wp4_wake_endmember.csv ({len(res)} 行) + 补算/output/wp4_wake_endmember_full.csv ({len(full_rows)} 行)')

# ── 复核 ──
print(f'\n[交叉检验] 本扫描 η(9m/s) vs wp7a_real_eta 9m/s 档 逐格最大差: {max_eta_gap:.2e}')
res2 = res.merge(pd.DataFrame({'farm_id': zc['farm_ids'].astype(int), 'A': zc['A']}), on='farm_id')
rho, p = spearmanr(res2['delta_p90'], res2['A'])
print(f'ρ(Δp90, A) = {rho:+.3f} (p={p:.2e}, n={len(res2)})   ← 清单期望 0.790')
# 学长替代量：9 m/s 方向效率跨度
span = res2['eta_low'] - res2['eta_high']
rho_s, p_s = spearmanr(span, res2['A'])
print(f'ρ(9m/s 方向效率跨度, A) = {rho_s:+.3f} (p={p_s:.2e})   ← 学长图 2f 替代量 0.644')
print('\n[端元摘要]')
print(res[['p90_deficit_high', 'p90_deficit_low', 'delta_p90', 'eta_high', 'eta_low',
           'n_upstream_wakes', 'multi_wake_frac']].describe().round(3).to_string())
