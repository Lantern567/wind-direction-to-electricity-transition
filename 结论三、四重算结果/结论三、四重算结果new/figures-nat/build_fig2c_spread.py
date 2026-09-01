# -*- coding: utf-8 -*-
"""图 2 (c) 顶部公式行上下标拉开（2026-08-30 最终版）
====================================================================
用户："就上边两个灰色折线的文字重叠" —— 面板上部 y 1490-1600 的
公式文字行（带上下标）看起来挤在一起。把上标上移、下标下移，
主体行与箭头（x 1480-1660 的指向箭头）保持不动。

实现：从备份原图抠出三个 y 带（上标 1490-1530 / 主体 1530-1567 /
下标 1567-1600）的非白像素，涂白原位置，按新 y 粘贴。
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ORIG = os.path.join(REPO, '结论三、四重算结果', '_fig2_orig_backup.png')
img = Image.open(ORIG).convert('RGB')
a = np.asarray(img).copy()
nonwhite = a.max(axis=2) < 245

X0, X1 = 600, 1660          # 公式行 x 范围
TOP_SUP_Y0, TOP_SUP_Y1 = 1490, 1530    # 上标带
BODY_Y0, BODY_Y1 = 1530, 1567          # 主体带
SUB_Y0, SUB_Y1 = 1567, 1600            # 下标带
SHIFT = 18

# 上标带里 x 1480-1660 是箭头（不动）；只移动 x 600-1480 的上标
sup = nonwhite[TOP_SUP_Y0:TOP_SUP_Y1, X0:1480]
sub = nonwhite[SUB_Y0:SUB_Y1, X0:X1]

# 涂白原位置
a[TOP_SUP_Y0:TOP_SUP_Y1, X0:1480][sup] = 255
a[SUB_Y0:SUB_Y1, X0:X1][sub] = 255
# 粘贴新位置（上标上移 SHIFT、下标下移 SHIFT）
a[TOP_SUP_Y0-SHIFT:TOP_SUP_Y1-SHIFT, X0:1480][sup] = np.asarray(img)[TOP_SUP_Y0:TOP_SUP_Y1, X0:1480][sup]
a[SUB_Y0+SHIFT:SUB_Y1+SHIFT, X0:X1][sub] = np.asarray(img)[SUB_Y0:SUB_Y1, X0:X1][sub]

out = os.path.join(HERE, '_fig2c_spread.png')
Image.fromarray(a).save(out)
print('saved:', out)

# 换入 docx
import zipfile, hashlib
for docx in [os.path.join(REPO, '结论三、四重算结果', 'ADAPEN_manuscript_zh_v4.9.docx'),
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
            zout.writestr(it, open(out, 'rb').read() if it.filename == 'word/media/image2.png' else data)
        zout.close()
    with open(docx, 'rb') as f:
        zi = zipfile.ZipFile(f)
        im = zi.read('word/media/image2.png')
    print('swapped:', os.path.basename(docx), hashlib.md5(im).hexdigest()[:12])
