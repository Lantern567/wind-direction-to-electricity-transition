# -*- coding: utf-8 -*-
"""v4.9 图3 colorbar 完整重绘 v6（2026-08-31）
====================================================================
用户："colorbar要延申到f结束，不是在f开始处结束"
渐变条改为**全高有色**重绘（消除浅色端接近白色/稀疏的观感）：
- 从原图渐变中轴（y 5130, x 1937-2924）提取颜色序列（33 采样）
- 映射到新位置 x 462-3415（f 面板右框），每列填满 y 5105-5164 全高
- 刻度短线（原 19 个）按比例移到新位置
- 刻度标签（原 2 组）按比例移到新位置
- 大标签（两行标题文字）移到新 colorbar 中心（x 1938）居中
其余像素不动。
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import zipfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(HERE, '_fig3_restored.png')     # 原图
BASE = os.path.join(HERE, '_fig3c_v5.png')         # v5（底线已修）
DOCX = os.path.join(REPO, '结论三、四重算结果', 'ADAPEN_manuscript_zh_v4.9.docx')

orig = np.asarray(Image.open(SRC).convert('RGB')).astype(np.uint8)
a = np.asarray(Image.open(BASE).convert('RGB')).astype(np.uint8)

# ══════════ colorbar 渐变重绘 ══════════
NEW0, NEW1 = 462, 3415
GY0, GY1 = 5105, 5164     # 渐变条高度
# 原图渐变颜色序列（y 5130, x 1937-2920，去掉右端白色边缘）
src_row = orig[5130, 1937:2921].astype(float)   # 984 像素
# 用中轴颜色插值生成新渐变（每列）
n_new = NEW1 - NEW0 + 1
x_src = np.linspace(0, 983, n_new)
colors = np.empty((n_new, 3), dtype=np.uint8)
for c in range(3):
    colors[:, c] = np.interp(x_src, np.arange(984), src_row[:, c])
# 填充渐变条（每列全高）
a[GY0:GY1, NEW0:NEW1+1] = colors[None, :, :]

# ══════════ 刻度短线重排 ══════════
# 原图刻度 x 位置（y 5160-5178 检测的 19 个）
OLD_TICKS = [1937, 1976, 2009, 2037, 2062, 2084, 2233, 2319, 2381, 2428,
             2467, 2500, 2529, 2554, 2575, 2724, 2811, 2872, 2920]
SCALE = (NEW1 - NEW0) / (2924 - 1937)
tk_y0, tk_y1 = 5160, 5178
# 先涂白旧刻度区
a[tk_y0:tk_y1, 400:3480] = 255
for tx in OLD_TICKS:
    new_tx = int(round(NEW0 + (tx - 1937) * SCALE))
    # 画刻度短线（3px 宽, 18px 高）
    a[tk_y0:tk_y1, new_tx-1:new_tx+2] = (31, 41, 55)

# ══════════ 刻度标签重排 ══════════
# 原图刻度标签（y 5180-5260）：2 组（x 2040-2140 + x 2540-2620）
lab_old = [(2040, 2140, 5180, 5260), (2540, 2620, 5180, 5260)]
a[5180:5260, 400:3480] = 255
for x0, x1, y0, y1 in lab_old:
    new_x0 = int(round(NEW0 + (x0 - 1937) * SCALE))
    new_x1 = new_x0 + (x1 - x0)     # 宽度不变（文字不变形）
    block = orig[y0:y1, x0:x1]
    a[y0:y1, new_x0:new_x1] = block

# ══════════ 大标签移到新中心 ══════════
# 原大标签（y 5265-5320, x 1860-3002）
big = orig[5265:5320, 1860:3002]
big_dark = big.max(axis=2) < 245
# 涂白旧大标签区
a[5265:5320, 1860:3002] = 255
# 新中心 x 1938（colorbar 中心），宽度不变
new_big_x0 = 1938 - (3002 - 1860) // 2
a[5265:5320, new_big_x0:new_big_x0 + (3002-1860)] = big

out = os.path.join(HERE, '_fig3c_v6.png')
Image.fromarray(a).save(out)
print('saved:', out)

# 验证
row = a[5130]
print('新渐变采样:')
for x in [462, 600, 900, 1200, 1500, 1938, 2200, 2500, 2800, 3100, 3415]:
    print(f'  x{x}: {tuple(row[x])}')
# 全高检查
reg = a[5105:5164, 462:3416]
nw = reg.max(axis=2) < 245
print('渐变条全高非白比例:', round(nw.mean(), 3))

# 换入 docx
png = open(out, 'rb').read()
for docx in [DOCX,
             os.path.join(REPO, '重算交付_20260829', '论文', 'ADAPEN_manuscript_zh_v4.9.docx')]:
    if not os.path.exists(docx):
        continue
    with open(docx, 'rb') as f:
        zin = zipfile.ZipFile(f)
        items = [(it, zin.read(it.filename)) for it in zin.infolist()]
    with open(docx, 'r+b') as f:
        f.seek(0); f.truncate(0)
        zout = zipfile.ZipFile(f, 'w')
        for it, data in items:
            zout.writestr(it, png if it.filename == 'word/media/image3.png' else data)
        zout.close()
    with open(docx, 'rb') as f:
        zi = zipfile.ZipFile(f)
        im = zi.read('word/media/image3.png')
    print('swapped:', os.path.basename(docx), hashlib.md5(im).hexdigest()[:12])
