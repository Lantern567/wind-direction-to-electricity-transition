"""
wp9e：把图2相关支撑统计在 wp7a 冻结口径（171 场、iea_10MW、逐场 TI、72 角度）下重算
====================================================================================
目的：正文统一 27/171 后，原来在 task1 口径（29 场/167 场）上算的支撑数字需同步：
  ① 9 m/s 方向端元效率跨度（原：29 场中位 23.0%→95.8%、跨度 72.8 pp）
  ② 建成—最优相位差与 100G/A 的 Spearman（原：29 场 0.775）
  ③ 二阶谐波结构重构 vs G2024（原：167 场 Spearman 0.757 / Pearson 0.787 /
     偏差 +0.82 pp / MAE 1.06 pp；A 单独 0.388、相位单独 0.510；负增益 ~18%）
  ④ A>5.2% 高响应场数（A 与 A_full 两口径核对 → 应为 27）
输出：补算/output/wp9e_27stats.txt
"""
import io, sys, os
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
REPO = 'd:/01学习资料/wind-direction-to-electricity-transition'
CUR = np.load(REPO + '/结论三、四重算结果/output/wp7a_real_curves.npz')
ETA = np.load(REPO + '/结论三、四重算结果/output/wp7a_real_eta.npz')
OUT = REPO + '/补算/output/wp9e_27stats.txt'

A = CUR['A']; A_full = CUR['A_full']
th_star = CUR['th_star']                 # 最优角索引（72 档 5°）
E_y = CUR['E_y']                         # (171, 11, 72) 2014–2024
G_so = CUR['G_so']                       # (171, 5) 样本外逐年增益 %
eta = ETA['eta']                         # (171, 18, 72) 效率
ws = ETA['ws']

lines = []
def w(t): lines.append(t)

w(f'# A>5.2% 场数: A 口径 {int((A>5.2).sum())} / A_full 口径 {int((A_full>5.2).sum())}')
hi = A > 5.2
n_hi = int(hi.sum())
w(f'采用 A 口径，高响应场 n={n_hi}')

# ── ① 9 m/s 方向端元 ──
j9 = int(np.argmin(np.abs(ws - 9.0)))
w(f'# 9 m/s 风速档实际为 {ws[j9]:.2f} m/s')
emin = eta[hi, j9, :].min(axis=1)
emax = eta[hi, j9, :].max(axis=1)
w(f'9 m/s 端元（{n_hi} 场）: 高损失方向效率中位 {np.median(emin)*100:.1f}% → '
  f'低损失方向 {np.median(emax)*100:.1f}%，跨度 {np.median(emax-emin)*100:.1f} pp')

# ── ② 相位差 vs 100G/A ──
# 注：G_so 是相对半圆均值的增益（非相对建成朝向）；正文 G 定义为建成→最优，
# 故此处 G 由 E_y 重算：G = mean_y 100×(E_y(θ*)−E_y(0))/E_y(0)
dphi_deg = th_star % 180                              # th_star 为角度（0–175°，5° 步长）
dphi = np.minimum(dphi_deg, 180 - dphi_deg)           # 折叠到 [0, 90]
idx = np.arange(len(th_star))
Esel = E_y[idx, :, th_star // 5]                      # (171, 11) 逐场逐年最优角能量
E0 = E_y[:, :, 0]
G = 100 * np.nanmean((Esel - E0) / np.maximum(E0, 1e-9), axis=1)   # 多年平均建成→最优增益 %
ratio = 100 * G / A
ok = hi & np.isfinite(ratio) & (A > 0)
r_phase = spearmanr(dphi[ok], ratio[ok]).statistic
w(f'# 相位: 建成—最优相位差 vs 100G/A 的 Spearman（{int(ok.sum())} 场）= {r_phase:.3f}')

# ── ③ 二阶谐波重构 vs G2024 ──
a = A / 100.0
Ghat = 100 * 2 * a * np.sin(np.deg2rad(dphi))**2 / (1 + a * np.cos(2*np.deg2rad(dphi)))
G2024 = 100 * (E_y[np.arange(len(th_star)), 10, th_star // 5] / E_y[:, 10, 0] - 1)   # 2024 年（末档）样本外增益
m = np.isfinite(Ghat) & np.isfinite(G2024)
rs, rp = spearmanr(Ghat[m], G2024[m]).statistic, pearsonr(Ghat[m], G2024[m]).statistic
bias = np.mean(Ghat[m] - G2024[m])
mae = np.mean(np.abs(Ghat[m] - G2024[m]))
neg = (G2024[m] < 0).mean() * 100
w(f'# 重构（{int(m.sum())} 场）: Spearman {rs:.3f} / Pearson {rp:.3f} / '
  f'平均偏差 {bias:+.2f} pp / MAE {mae:.2f} pp / G2024 负增益比例 {neg:.0f}%')
rA = spearmanr(A[m], G2024[m]).statistic
rP = spearmanr(dphi[m], G2024[m]).statistic
w(f'A 单独 vs G2024: {rA:.3f}；相位单独 vs G2024: {rP:.3f}')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('\n已写入', OUT)
