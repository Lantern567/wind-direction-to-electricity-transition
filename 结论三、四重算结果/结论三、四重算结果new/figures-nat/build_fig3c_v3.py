# -*- coding: utf-8 -*-
"""v4.9 图3 修复 v3（2026-08-31）
====================================================================
在 v2 基础上修复：
① x 轴底线（y 1288-1299）被组平移涂白/搬走 → 从原图整行恢复；
② a 面板顶部图例/标题（y 85-225）被组平移波及 → 从原图恢复原位
   （红点标注 y 225-300 保持 v2 已平移的位置，不在恢复范围）；
③ colorbar 右端补到 x 3411（f 右缘）。
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
SRC = os.path.join(HERE, '_fig3_restored.png')     # 原图（恢复版）
V2 = os.path.join(HERE, '_fig3c_v2.png')           # v2（组平移 + colorbar 拉伸）
DOCX = os.path.join(BLOCK, 'ADAPEN_manuscript_zh_v4.9.docx')

orig = np.asarray(Image.open(SRC).convert('RGB')).copy()
a = np.asarray(Image.open(V2).convert('RGB')).copy()

# ① x 轴底线（y 1288-1299）整行恢复
a[1288:1299, 380:3560] = orig[1288:1299, 380:3560]
# ② 顶部图例/标题（y 85-225）恢复原位
a[85:225, 380:3560] = orig[85:225, 380:3560]

# ③ colorbar 右端：本体已到 3410，补 3411（f 右缘）
# 若右端差 1px，从原图补
if (a.max(axis=2) < 240)[5100:5160, 3411].any():
    pass
else:
    a[5100:5160, 3411] = orig[5100:5160, 3411]

out = os.path.join(HERE, '_fig3c_v3.png')
Image.fromarray(a).save(out)
print('saved:', out)

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
