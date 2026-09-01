# -*- coding: utf-8 -*-
"""图 2 (c) 面板最小手术（2026-08-30 第三轮，用户："字体特别大 / 图乱了"）
====================================================================
前两轮整面板重建/替换均被否。本轮**从备份恢复原图**，只做一件事：
给内容区里被曲线遮挡的文字（数值标签 1.09/0.75/0.72/A=0、红色标注）加
白色描边（halo），其余像素（数据线、红箭头、右侧文本列、坐标轴）完全不动。

文字判定：内容区 (303,1348)-(2474,3089) 内 暗色(R,G,B<160) 或 红色
(R>140,G<110,B<110) 的**小块连通域**（w 15-220、h 15-90，排除长曲线/大元素）。
白描边：文字掩码膨胀 6px → 在原图上涂白 → 文字像素贴回。

输出：_fig2c_halo.png（整幅 4196×5120），换入工作版+交付版 v4.9。
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
from scipy import ndimage
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

ORIG = os.path.join(REPO, '结论三、四重算结果', '_fig2_orig_backup.png')
img = Image.open(ORIG).convert('RGB')
a = np.asarray(img).copy()
R, G, B = a[:,:,0].astype(int), a[:,:,1].astype(int), a[:,:,2].astype(int)

X0, X1, Y0, Y1 = 303, 2474, 1348, 3089   # 内容区

dark = (R < 160) & (G < 160) & (B < 160)
red  = (R > 140) & (G < 110) & (B < 110)
mask = (dark | red)[Y0:Y1, X0:X1]

lab, n = ndimage.label(mask)
keep = np.zeros_like(mask)
n_txt = 0
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    if 15 <= w <= 220 and 15 <= h <= 90 and len(ys) >= 40:
        keep[lab == i] = True
        n_txt += 1
print('文字块数量:', n_txt)

# 白描边：膨胀 6px，在原图上涂白，再贴回文字
dil = ndimage.binary_dilation(keep, iterations=6)
a[Y0:Y1, X0:X1][dil] = 255
a[Y0:Y1, X0:X1][keep] = np.asarray(img)[Y0:Y1, X0:X1][keep]

out = os.path.join(HERE, '_fig2c_halo.png')
Image.fromarray(a).save(out)
print('saved:', out)

# 换入 docx（r+b 原地写）
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
