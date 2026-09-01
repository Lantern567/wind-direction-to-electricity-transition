# -*- coding: utf-8 -*-
"""v4.9 图3 两处修改（2026-08-30 用户指令）
====================================================================
① a 子图（顶部小提琴/散点）横坐标间距缩小一点：
   数据区 x 700-3560（y 100-1291）+ x 轴标签 x 700-3560（y 1330-1460）
   整体水平压缩 6%（resize 0.94），居中粘贴（x 786 起，两侧各留 86px）。
   x 轴底线（y 1291 全宽横线）与 y 轴刻度标签（x<700）不动。
② colorbar（y 5100-5160, x 1937-2924）右端拉长到 x 3411
   （与右列面板 e/f 宽度一致）：本体条水平拉伸（resize），
   刻度短线（y 5160-5178）与刻度标签（y 5180-5250）按比例平移。
其余像素一律不动。
"""
import os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import zipfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # 项目根
BLOCK = os.path.dirname(HERE)                          # 结论三、四重算结果
DOCX = os.path.join(BLOCK, 'ADAPEN_manuscript_zh_v4.9.docx')

z = zipfile.ZipFile(DOCX)
img = Image.open(io.BytesIO(z.read('word/media/image3.png'))).convert('RGB')
src = np.asarray(img).copy()
a = np.asarray(img).copy()
H, W = a.shape[:2]

# ══════════ ① a 面板数据区水平压缩 6% ══════════
AX0, AX1, AY0, AY1 = 700, 3560, 100, 1291   # 数据区
LX0, LX1, LY0, LY1 = 700, 3560, 1330, 1460  # x 轴标签区
F = 0.94
data_region = img.crop((AX0, AY0, AX1, AY1))
lab_region = img.crop((LX0, LY0, LX1, LY1))
new_w = int(round((AX1 - AX0) * F))
new_wl = int(round((LX1 - LX0) * F))
off = (AX1 - AX0 - new_w) // 2           # 居中偏移
off_l = (LX1 - LX0 - new_wl) // 2

# 涂白原区域
a[AY0:AY1, AX0:AX1] = 255
a[LY0:LY1, LX0:LX1] = 255
# 粘贴压缩后
data_small = data_region.resize((new_w, AY1 - AY0), Image.LANCZOS)
lab_small = lab_region.resize((new_wl, LY1 - LY0), Image.LANCZOS)
a[AY0:AY1, AX0 + off: AX0 + off + new_w] = np.asarray(data_small)
a[LY0:LY1, LX0 + off_l: LX0 + off_l + new_wl] = np.asarray(lab_small)

# ══════════ ② colorbar 拉长 ══════════
CX0, CX1, CY0, CY1 = 1937, 2924, 5100, 5160   # colorbar 本体
TX0, TX1, TY0, TY1 = 1937, 2924, 5160, 5178   # 刻度短线
NX0, NX1, NY0, NY1 = 1937, 2924, 5180, 5250   # 刻度标签
TARGET_R = 3411                                # 拉长到右列右端
SCALE = (TARGET_R - CX0) / (CX1 - CX0)         # 1.497

cb = img.crop((CX0, CY0, CX1, CY1))
ticks = img.crop((TX0, TY0, TX1, TY1))
nums = img.crop((NX0, NY0, NX1, NY1))
new_cw = int(round((CX1 - CX0) * SCALE))
a[CY0:CY1, CX0:CX1] = 255
a[TY0:TY1, TX0:TX1] = 255
a[NY0:NY1, NX0:NX1] = 255
cb_small = cb.resize((new_cw, CY1 - CY0), Image.LANCZOS)
ticks_small = ticks.resize((new_cw, TY1 - TY0), Image.LANCZOS)
nums_small = nums.resize((new_cw, NY1 - NY0), Image.LANCZOS)
a[CY0:CY1, CX0:CX0 + new_cw] = np.asarray(cb_small)
a[TY0:TY1, CX0:CX0 + new_cw] = np.asarray(ticks_small)
a[NY0:NY1, CX0:CX0 + new_cw] = np.asarray(nums_small)

out = os.path.join(HERE, '_fig3c_edited.png')
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
