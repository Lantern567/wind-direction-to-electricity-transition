# -*- coding: utf-8 -*-
"""Nature-Communications 风格模块 v3（2026-08-24，严格按用户指定参考图
figure-model/微信图片_20260824115717_244_105.jpg（图1 全球地图 + 7 insets）
与 微信图片_20260824115714_241_105.jpg（图2 华北平原十panel 组版）复刻体例：

  - 面板：白底 + 细黑边框（axes.linewidth 1.0 / edgecolor #1a1a1a），
    淡灰网格仅作辅助，#E8E8E8。
  - 面板字母：加粗黑体，置于框内左上角（参考图2 inset 组版体例）。
  - 数值注记：彩色加粗 + 白色描边（halo，参考图1 "0.84 m/year" 白描边字样）。
  - 地图：青蓝海面 #9ad3e2 + 白陆 + 深灰海岸线 + 黑框；
    大陆轮廓细、国界浅灰、网格虚线浅灰；图例黑细框白底。
  - 地图可加北向箭头 N 与比例尺（参考图2 右图体例）。
  - 主色板（参考图1 底部五色）：#e73618 红 / #f6bf5c 橙黄 /
    #def1e4 淡绿 / #7bcbf1 浅蓝 / #1e4e9e 深蓝（配 #5299cc 中蓝过渡）。
只供 figures-nat/ 下脚本引用。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 参考图五色板 ──
RED = "#e73618"          # 高亮强调 / 负值 / 走廊
ORANGE = "#f6bf5c"       # 次强调橙黄
PALE_GREEN = "#def1e4"   # 浅青绿（地图低值端）
LIGHT_BLUE = "#7bcbf1"   # 浅蓝（层级低端）
MID_BLUE = "#5299cc"     # 中蓝（层级中端）
DEEP_BLUE = "#1e4e9e"    # 深蓝（层级高端 / 参照组）
GREY = "#8e99a4"
LIGHT_GREY = "#c9d2da"   # 浅灰（其它项目、连接线）
INK = "#1a1a1a"
BOX_EC = "#555555"       # 图例/文字框边框

# 三级情景：由浅入深的顺序蓝（同色系 = 同问题域不同自由度）
SCOL = {"S1": LIGHT_BLUE, "S2": MID_BLUE, "S3": DEEP_BLUE}

# 走廊分组色（图4a 地图 / 分走廊柱）
CORRIDOR_COL = {
    "China_strait": RED,
    "Vietnam": ORANGE,
    "Italy": "#a577ad",
    "Denmark": "#73c79e",
    "other": "#b8c2cc",
}

# 图3 二分组：走廊 vs 其他（红 vs 深蓝 / 灰蓝）
GROUP2 = {"corridor": RED, "other": "#1e4e9e"}
GROUP2_SOFT = {"corridor": RED, "other": "#aeb9c4"}

# 六范式显示名与间距（间距数值来自 wp7d，格式化后随数据渲染）
PARADIGM_ORDER = ["S_A", "S_B0", "S_B45", "S_C", "S_D", "S_E"]
PARADIGM_NAME = {"S_A": "Aligned", "S_B0": "Dense", "S_B45": "Dense 45°",
                 "S_C": "Phased", "S_D": "Compact", "S_E": "Wide"}

# 地图体例（参考图1：青蓝海 + 白陆 + 深灰海岸线）
LAND_COL = "#ffffff"
SEA_COL = "#9ad3e2"
COAST_COL = "#6b6b6b"
GRIDLINE_COL = "#c9c9c9"
COAST_LW = 0.5


def apply_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 1.0,           # 细黑边框（参考体例）
        "axes.edgecolor": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelcolor": "#111111",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": BOX_EC,
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.35,
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.45,
        "grid.alpha": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 450,
        "savefig.bbox": None,          # 关键：不紧致裁剪，保持版面
        "axes.axisbelow": True,
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",
    })


def panel_label(ax, letter, dx=0.012, dy=1.0, fs=11.5):
    """框内左上角加粗小写面板字母（参考图2 inset 组版体例）。"""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=fs, fontweight="bold", va="top", ha="left",
            color="#111111", zorder=40, clip_on=False)


def halo(ax, x, y, s, color=INK, fs=7.2, weight="bold", ha="center",
         va="center", zorder=30, **kw):
    """彩色加粗数值注记 + 白色描边（参考图1 白描边数值字样）。"""
    import matplotlib.patheffects as pe
    t = ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
                fontweight=weight, zorder=zorder, **kw)
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    return t


def north_arrow(ax, x=0.955, y0=0.94, y1=0.80, fs=8.5):
    """图内黑边 N 箭头（参考图2 右图）。"""
    ax.annotate("N", xy=(x, y0), xytext=(x, y1),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                fontsize=fs, fontweight="bold", color="#111111",
                ha="center", va="center", annotation_clip=False, zorder=35,
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.1,
                                mutation_scale=7))


def scale_bar(ax, lon0, lat0, km=1000.0, fs=7, color="#111111"):
    """地图内比例尺（近似，PlateCarree 低纬）。"""
    dlon = km / (111.32 * max(abs(np.cos(np.radians(lat0))), 0.15))
    h = 0.35
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=color, lw=1.4, zorder=35)
    ax.plot([lon0, lon0], [lat0 - h, lat0 + h], color=color, lw=1.1, zorder=35)
    ax.plot([lon0 + dlon, lon0 + dlon], [lat0 - h, lat0 + h],
            color=color, lw=1.1, zorder=35)
    ax.text(lon0 + dlon / 2, lat0 + 0.9, f"{km:.0f} km", fontsize=fs,
            ha="center", va="bottom", color=color, zorder=35)


def save_fig(fig, path):
    fig.savefig(path, dpi=450, bbox_inches=None, pad_inches=0.02)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches=None, pad_inches=0.02)
    print("saved:", path)
