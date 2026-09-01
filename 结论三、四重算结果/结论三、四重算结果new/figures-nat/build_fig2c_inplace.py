# -*- coding: utf-8 -*-
"""图 2 (c) 面板内容替换（2026-08-30，用户："直接在原坐标轴改，不要另加坐标轴"）
====================================================================
不重画坐标轴：保留原图 (c) 面板的边框、刻度线、y 轴刻度标签、x 轴下方情景名，
只擦除内容区旧曲线/旧文字，按原图坐标标定（实测刻度像素）重画数据内容。

坐标标定（原图 4196×5120，面板框 x 282-2484 / y 1339-3113）：
  y 刻度线 8 条：1912/2074/2236/2397/2559/2721/2882/3044（间隔 162px）
  主刻度 0/0.5/1.0/1.5 对应像素 y 3047/2723/2399/2075 → y_px = 3047 - 648*v
  x 刻度线 4 条：422/815/1208/1602（间隔 393px）→ x_px = 422 + 393*i

数据（wp5c_farm_cross.csv，与 SI 表 S4 对账）：总平均 C0 1.0897 / C1 0.7531
/ C2 0.7233 / C3 0.0；灰线 = 六范式各自跨场均值（最高 1.681）。

输出：figures-nat/_fig2c_inplace.png（整幅 4196×5120，仅内容区变化）
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from nc_style_nat import RED, INK, DEEP_BLUE

# ── 数据 ──
d = pd.read_csv(os.path.join(REPO, '补算', 'output', 'wp5c_farm_cross.csv'))
SCEN = ['A_C0', 'A_C1', 'A_C2', 'A_C3']
para = d.groupby('paradigm')[SCEN].mean()
means = d[SCEN].mean().values
assert abs(means[0] - 1.0897) < 0.001 and abs(means[1] - 0.7531) < 0.001
assert abs(means[2] - 0.7233) < 0.001 and means[3] < 1e-6
assert abs(para.values.max() - 1.681) < 0.01
print('对账: 总平均 C0=%.4f C1=%.4f C2=%.4f C3=%.4f' % tuple(means))

# ── 坐标映射（实测刻度） ──
def xpx(i):  return 422 + 393 * i
def ypx(v):  return 3047 - 648 * v

# ── 擦除原内容区（x 303..2474, y 1348..3089，避开边框/刻度/刻度标签） ──
ORIG = os.path.join(REPO, '结论三、四重算结果', '_fig2_orig_backup.png')
img = Image.open(ORIG).convert('RGB')
a = np.asarray(img).copy()
a[1348:3089, 303:2474] = 255
img = Image.fromarray(a)

# ── 新画布：像素坐标直接作图 ──
W, H, DPI = 4196, 5120, 450
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 40})
fig = plt.figure(figsize=(W / DPI, H / DPI))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0)   # 像素坐标（y 向下）
ax.axis('off')

ST = dict(path_effects=[pe.withStroke(linewidth=6, foreground='white')])

# 六范式灰线
X = [xpx(i) for i in range(4)]
for p in para.index:
    ax.plot(X, [ypx(v) for v in para.loc[p].values],
            color='#9aa5ad', lw=2.0, zorder=2)
# 蓝线总平均
ax.plot(X, [ypx(v) for v in means], color=DEEP_BLUE, lw=3.5, zorder=4,
        marker='o', ms=7, mfc=DEEP_BLUE, mec='white', mew=1.2)

# 数值标签（白描边，放各点上方/右侧）
ax.text(xpx(0.02) + 0, ypx(1.12) - 12, '1.09', ha='center', va='bottom',
        fontsize=44, color='#333333', **ST)
ax.text(xpx(0.98), ypx(0.81) - 12, '0.75', ha='center', va='bottom',
        fontsize=44, color='#333333', **ST)
ax.text(xpx(2.02), ypx(0.78) - 12, '0.72', ha='center', va='bottom',
        fontsize=44, color='#333333', **ST)
ax.text(xpx(3.02), ypx(0.07) + 4, 'A = 0', ha='left', va='top',
        fontsize=44, color='#333333', **ST)

# −30.9% 标注（C0–C1 段上方空白）
ax.text(xpx(0.5), ypx(1.47) - 10, '−0.34 pp\n(−30.9%)', ha='center',
        va='bottom', fontsize=40, color=RED, **ST)

# 图例盒（右上角，白底黑边两行）
bx, by, bw, bh = 1880, 1365, 580, 170
ax.add_patch(Rectangle((bx, by), bw, bh, facecolor='white',
                       edgecolor='black', lw=2, zorder=5))
ax.plot([bx + 28, bx + 62], [by + 62, by + 62], color=DEEP_BLUE, lw=3.5, zorder=6)
ax.text(bx + 78, by + 64, 'Mean across all farms', ha='left', va='center',
        fontsize=38, color='#222222', zorder=6)
ax.plot([bx + 28, bx + 62], [by + 128, by + 128], color='#9aa5ad', lw=2.0, zorder=6)
ax.text(bx + 78, by + 130, 'Each of six paradigms', ha='left', va='center',
        fontsize=38, color='#222222', zorder=6)

out = os.path.join(HERE, '_fig2c_inplace.png')
fig.savefig(out, dpi=DPI, facecolor='none', transparent=True)
new = Image.open(out).convert('RGBA')
assert new.size == (W, H)
composed = Image.alpha_composite(img.convert('RGBA'), new)
final = os.path.join(HERE, '_fig2c_final.png')
composed.convert('RGB').save(final)
print('saved:', final)

# ── 换入 docx（r+b 原地写，兼容 Word 打开状态） ──
import zipfile, hashlib
for docx in [os.path.join(REPO, '结论三、四重算结果', 'ADAPEN_manuscript_zh_v4.9.docx'),
             os.path.join(REPO, '重算交付_20260829', '论文', 'ADAPEN_manuscript_zh_v4.9.docx')]:
    if not os.path.exists(docx):
        print('skip missing:', docx); continue
    with open(docx, 'rb') as f:
        zin = zipfile.ZipFile(f)
        png = zin.read('word/media/image2.png')
        zinfo = zin.getinfo('word/media/image2.png')
        items = []
        for it in zin.infolist():
            items.append((it, zin.read(it.filename) if it.filename != 'word/media/image2.png' else None))
    with open(docx, 'r+b') as f:
        f.seek(0); f.truncate(0)
        zout = zipfile.ZipFile(f, 'w')
        for it, data in items:
            if it.filename == 'word/media/image2.png':
                zout.writestr(it, final and open(final, 'rb').read())
            else:
                zout.writestr(it, data)
        zout.close()
    with open(docx, 'rb') as f:
        zi = zipfile.ZipFile(f); im = zi.read('word/media/image2.png')
    print('swapped:', os.path.basename(docx), hashlib.md5(im).hexdigest()[:12])
