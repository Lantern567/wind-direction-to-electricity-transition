# -*- coding: utf-8 -*-
"""图 2 (c) 两处文字原位微调（2026-08-30 用户指令："直接在原图调，不要改变什么"）
====================================================================
① 红色标注文字（两行，y 2766-2870，x 378-930）整体上移 20px；
② 最上面两条灰色折线的标注文字（y 2156-2224，x 1270-1645）上下两行
   紧贴/粘连 → 上排上移 18px、下排原位不动。
其余像素（曲线、箭头、坐标轴、其他标注）一律不动。

像素级实现：只移动"目标颜色"的像素（红=R>G+50且R>B+50；中灰标注=
(100,116,139)/(154,165,173)±30），核心掩码膨胀 1px 带抗锯齿边，
旧位置涂白、新位置写入原像素值。粘连字符（上下行连成一个连通域）按
"颈线"切开：y≤2195 归上排、y≥2196 归下排；comp5（x1578-1609 的"9"与
下行"5"粘连）颈线 2186。
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
from scipy import ndimage
import zipfile, hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(REPO, '_fig2_orig_backup.png')

img = Image.open(ORIG).convert('RGB')
src = np.asarray(img).copy()
a = np.asarray(img).copy()
R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
nonwhite = a.max(axis=2) < 250
blue = (abs(R-37) < 45) & (abs(G-99) < 50) & (abs(B-235) < 35)
ink = (R < 60) & (G < 70) & (B < 80)

# ══════════ ① 红色标注上移 20px ══════════
reddish = (R > G + 50) & (R > B + 50) & nonwhite
red_zone = np.zeros_like(reddish)
red_zone[2740:2900, 345:985] = reddish[2740:2900, 345:985]
red_core = red_zone
red_cap = ndimage.binary_dilation(red_core, iterations=1) & reddish
ys, xs = np.where(red_cap)
print('红色像素:', len(ys), 'bbox x%d-%d y%d-%d' % (xs.min(), xs.max(), ys.min(), ys.max()))
# 目标位置冲突检查（落在非红系内容上=浅灰线/标签，覆盖之，但打印核对）
dest_r = np.zeros_like(red_cap)
dest_r[ys - 12, xs] = True
reddishish = (R > G + 25) & (R > B + 25)
coll = dest_r & nonwhite & ~reddishish
cy, cx = np.where(coll)
print('红字目标位真冲突:', len(cy),
      ('bbox x%d-%d y%d-%d' % (cx.min(), cx.max(), cy.min(), cy.max())) if len(cy) else '')
a[red_cap] = 255
a[ys - 12, xs] = src[red_cap]

# ══════════ ② 灰色标注上排上移 18px ══════════
Z = (slice(2090, 2280), slice(1240, 1690))
med = ((abs(R - 100) < 30) & (abs(G - 116) < 30) & (abs(B - 139) < 30)) | \
      ((abs(R - 154) < 30) & (abs(G - 165) < 30) & (abs(B - 173) < 30))
zmed = med[Z]
lab, n = ndimage.label(zmed)
core_move = np.zeros_like(zmed)
for i in range(1, n + 1):
    cys, cxs = np.where(lab == i)
    ymin, ymax = cys.min() + 2090, cys.max() + 2090
    xmin, xmax = cxs.min() + 1240, cxs.max() + 1240
    comp = lab == i
    if ymax <= 2195:                      # 纯上排：整体上移
        core_move |= comp
    elif ymin >= 2196:                    # 纯下排：不动
        pass
    else:                                 # 粘连：按颈线切开
        cut = 2186 if (xmin >= 1578 and xmax <= 1610) else 2196
        rows = (cys + 2090) <= cut - 1
        core_move[cys[rows], cxs[rows]] = True
cap = ndimage.binary_dilation(core_move, iterations=1)
# 颈线下方 1px 抗锯齿边也带上；comp5 区域颈线 2187
keep_y = np.zeros_like(cap)
keep_y[:] = (np.arange(2090, 2280) <= 2196)[:, None]
keep_y[: 2187 - 2090, 1575 - 1240: 1612 - 1240] = True
cap = cap & keep_y
cap = cap & nonwhite[Z] & ~blue[Z] & ~ink[Z] & ~reddish[Z]
# 排除浅灰曲线色（防止误抓曲线本体；标注 AA 最外层 1px 损失可忽略）
curve = (abs(R - 203) < 10) & (abs(G - 213) < 10) & (abs(B - 225) < 10)
cap = cap & ~curve[Z]
mys, mxs = np.where(cap)
print('灰标注上排像素:', len(mys), 'bbox x%d-%d y%d-%d' %
      (mxs.min() + 1240, mxs.max() + 1240, mys.min() + 2090, mys.max() + 2090))
dest_m = np.zeros_like(cap)
dest_m[mys - 18, mxs] = True
print('灰标注目标位冲突:', (dest_m & nonwhite[Z] & ~cap).sum())
a[Z][cap] = 255
a[mys - 18 + 2090, mxs + 1240] = src[Z][cap]

# ── 保存并换入 docx ──
out = os.path.join(HERE, '_fig2c_moved.png')
Image.fromarray(a).save(out)
print('saved:', out)

for docx in [os.path.join(REPO, 'ADAPEN_manuscript_zh_v4.9.docx'),
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
            zout.writestr(it, open(out, 'rb').read()
                          if it.filename == 'word/media/image2.png' else data)
        zout.close()
    with open(docx, 'rb') as f:
        zi = zipfile.ZipFile(f)
        im = zi.read('word/media/image2.png')
    print('swapped:', os.path.basename(docx), hashlib.md5(im).hexdigest()[:12])
