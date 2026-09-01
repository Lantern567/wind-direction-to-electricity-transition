# -*- coding: utf-8 -*-
"""v4.9 图3 两处修改 v2（2026-08-31，用户纠正后）
====================================================================
① a 子图横坐标间距缩小：**平移组**（组本身不变形），组间距 625→580（缩 7%）。
   组定义（数据簇+红点标注+x轴标签）：
     组0: x 660-1130 | 组1: x 1270-1750 | 组2: x 1860-2370 | 组3: x 2470-3000 | 组4: x 3060-3620
   平移：组0 不动，组1-4 依次左移 45/90/135/180px。
② colorbar 从 e 的第一个子图（x 462）到 f 子图（x 3411）结束：
   渐变本体条（y 5100-5160）水平拉伸到 x 462-3411；
   刻度短线（y 5160-5178）与刻度标签（y 5180-5250）按比例平移（文字不变形）。
其余像素一律不动。
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import zipfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BLOCK = os.path.dirname(HERE)
SRC = os.path.join(HERE, '_fig3_restored.png')     # 已恢复的原图
DOCX = os.path.join(BLOCK, 'ADAPEN_manuscript_zh_v4.9.docx')

img = Image.open(SRC).convert('RGB')
a = np.asarray(img).copy()
nonwhite = a.max(axis=2) < 240

# ══════════ ① a 面板组平移 ══════════
GROUPS = [  # (x0, x1, shift)
    (660, 1130, 0),
    (1270, 1750, 45),
    (1860, 2370, 90),
    (2470, 3000, 135),
    (3060, 3620, 180),
]
Y0, Y1 = 100, 1460   # 数据区+标签区（含红点标注、小提琴、x轴标签）
for x0, x1, sh in GROUPS:
    if sh == 0:
        continue
    crop = img.crop((x0, Y0, x1, Y1))
    a[Y0:Y1, x0:x1] = 255
    a[Y0:Y1, x0 - sh: x1 - sh] = np.asarray(crop)

# ══════════ ② colorbar 拉伸 + 刻度/标签平移 ══════════
CB0, CB1, CBY0, CBY1 = 1937, 2924, 5100, 5160   # 渐变本体
TK0, TK1, TKY0, TKY1 = 1937, 2924, 5160, 5178   # 刻度短线
NL0, NL1, NLY0, NLY1 = 1937, 2924, 5180, 5250   # 刻度标签
NEW0, NEW1 = 462, 3411                          # 新范围
SCALE = (NEW1 - NEW0) / (CB1 - CB0)             # 2.99

# 渐变本体：水平拉伸
cb = img.crop((CB0, CBY0, CB1, CBY1))
cb_new = cb.resize((int(round((CB1 - CB0) * SCALE)), CBY1 - CBY0), Image.LANCZOS)
a[CBY0:CBY1, CB0:CB1] = 255
a[CBY0:CBY1, NEW0:NEW0 + cb_new.size[0]] = np.asarray(cb_new)

# 刻度短线：按比例平移（抠出每个刻度块）
tk_region = img.crop((TK0, TKY0, TK1, TKY1))
tk_dark = np.asarray(tk_region.convert('L')) < 200
from scipy import ndimage
lab, n = ndimage.label(tk_dark)
a[TKY0:TKY1, TK0:TK1] = 255
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(ys) < 3:
        continue
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    if w > 40 or h > 30:      # 只移短线（排除文字）
        continue
    old_x = xs.min() + TK0
    new_x = int(round(NEW0 + (old_x - TK0) * SCALE))
    block = tk_region.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    a[TKY0 + ys.min(): TKY0 + ys.max() + 1, new_x: new_x + w] = np.asarray(block)

# 刻度标签：抠出文字块平移
nl_region = img.crop((NL0, NLY0, NL1, NLY1))
nl_dark = np.asarray(nl_region.convert('L')) < 200
lab2, n2 = ndimage.label(nl_dark)
a[NLY0:NLY1, NL0:NL1] = 255
for i in range(1, n2 + 1):
    ys, xs = np.where(lab2 == i)
    if len(ys) < 10:
        continue
    w = xs.max() - xs.min() + 1
    old_x = xs.min() + NL0
    new_x = int(round(NEW0 + (old_x - NL0) * SCALE))
    block = nl_region.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    a[NLY0 + ys.min(): NLY0 + ys.max() + 1, new_x: new_x + w] = np.asarray(block)

out = os.path.join(HERE, '_fig3c_v2.png')
Image.fromarray(a).save(out)
print('saved:', out)

# 换入 docx（工作版 + 交付版）
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
