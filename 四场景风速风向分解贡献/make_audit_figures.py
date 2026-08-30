# -*- coding: utf-8 -*-
"""四情景 v3 复核用图（Fig A1-A5）。

用法：python make_audit_figures.py
输入：本目录的 v3 结果、tmp/oldv2 下从 commit 7702497 取出的 v2 结果、
      offshore-task2/output 的既有 Gauss 主表与反事实表。
输出：figures/FigA1..A5.(png|pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.edgecolor": "#4b5563",
    "axes.linewidth": 0.8,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OK, PART, BAD, GREY = "#2a9d5c", "#d98c1f", "#c0392b", "#94a3b8"
NEW, OLD = "#1d4ed8", "#9ca3af"

io = lambda p: pd.read_csv(p, encoding="utf-8-sig")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    try:  # 矢量版可选：中文字体子集化偶发 MemoryError，不影响 PNG 交付
        fig.savefig(FIG / f"{name}.pdf")
    except Exception as exc:
        print(f"  (skip vector for {name}: {type(exc).__name__})")
    plt.close(fig)
    print("saved:", name)


# ── 数据 ──────────────────────────────────────────────────────────────
aep = io(HERE / "four_scenario_aep_farmyear.csv")
summ = io(HERE / "four_scenario_farm_summary.csv")
aep_old = io(REPO / "tmp/oldv2/aep_old.csv")

gauss = io(REPO / "offshore-task2/output/task2_annual_floris.csv")
gauss = gauss[gauss.wake_model == "gauss"].drop_duplicates(["farm_id", "year"])
cf = io(REPO / "offshore-task2/output/task2_counterfactual.csv")

gcols = gauss[["farm_id", "year", "AEP_kWh", "region", "n_hours"]].rename(
    columns={"AEP_kWh": "g_AEP", "region": "g_region", "n_hours": "g_hours"})


def p11_err(df):
    m = df.merge(gcols, on=["farm_id", "year"], how="inner")
    return (m.P11_kWh - m.g_AEP).abs() / m.g_AEP * 100, m


def p10_err(df):
    m = df.merge(cf[["farm_id", "year", "AEP_baseWD_kWh"]], on=["farm_id", "year"], how="inner")
    return (m.P10_kWh - m.AEP_baseWD_kWh).abs() / m.AEP_baseWD_kWh * 100


e11_new, m_new = p11_err(aep)
e11_old, _ = p11_err(aep_old)
e10_new, e10_old = p10_err(aep), p10_err(aep_old)


# ── Fig A1：验收看板 ──────────────────────────────────────────────────
items = [
    ("时间维（唯一时间戳、≤8784h）", 2, "8752–8784 h；>9000h 记录 337 → 0"),
    ("物理边界（CF≤1、尾流≤无尾流）", 2, "无尾流 CF 上限 1.288 → 0.66"),
    ("分解闭合（<1e-8 个百分点）", 2, "2.8e-14 个百分点"),
    ("主脚本可交付", 2, "four_scenario_decomposition.py 已上传"),
    ("样本覆盖（对齐 1203 场-年）", 1, "897 → 1153；2018–2024 → 2014–2024；仍缺 50 且无原因字段"),
    ("配置一致（四情景同哈希）", 1, "结构满足；无 QA 文件可交叉核验"),
    ("$P_{11}$ 复现（中位≤1.5%、P95≤3%）", 0, "中位 20.2% → 18.1%；达标 18/1153"),
    ("$P_{10}$ 复现（中位≤1.5%、P95≤3%）", 0, "中位 23.45% → 23.45%；达标 7/897"),
    ("有效小时与 Gauss 主表同规则", 0, "1153/1153 不一致，中位差 661 h"),
    ("区域匹配 100% 一致", 0, "归一大小写后仍 659/1153（57%）错配 + 25 条空"),
    ("历史基准（1981–2010 逐小时）", 0, "仍为 2014–2024 池化，且含目标年自身，未做留一"),
    ("可复现性（无个人绝对路径 + QA）", 0, "写死 d:\\01学习资料\\ 与 C:\\Users\\beyqm\\；QA/日志/概率表仍缺"),
]
colors = {2: OK, 1: PART, 0: BAD}
labels = {2: "已通过", 1: "部分改善", 0: "未通过"}

fig, ax = plt.subplots(figsize=(10.4, 4.9))
y = np.arange(len(items))[::-1]
for yi, (name, st, note) in zip(y, items):
    ax.barh(yi, 1, color=colors[st], height=0.62, zorder=2)
    ax.text(-0.015, yi, name, ha="right", va="center", fontsize=9)
    ax.text(1.03, yi, note, ha="left", va="center", fontsize=8, color="#374151")
ax.set_xlim(0, 1)
ax.set_ylim(-0.7, len(items) - 0.3)
ax.axis("off")
ax.legend(handles=[Patch(color=colors[k], label=labels[k]) for k in (2, 1, 0)],
          loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
ax.set_title("图 A1  返工验收看板：12 项硬性标准的 v3 状态（4 项通过 / 2 项部分 / 6 项未通过）",
             fontsize=10.5, pad=12, loc="left")
save(fig, "FigA1_acceptance_board")


# ── Fig A2：P11 / P10 对拍误差 CDF ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
for ax, (en, eo, tag, ref) in zip(axes, [
        (e11_new, e11_old, r"$P_{11}$ vs 既有 Gauss 实际 AEP", "task2_annual_floris.csv"),
        (e10_new, e10_old, r"$P_{10}$ vs 固定历史风向反事实", "task2_counterfactual.csv")]):
    for e, c, lab in [(eo, OLD, f"v2 旧版 (n={len(eo)})"), (en, NEW, f"v3 新版 (n={len(en)})")]:
        s = np.sort(e.values)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=c, lw=1.8, label=lab)
    ax.axvline(1.5, color=BAD, ls="--", lw=1, zorder=1)
    ax.axvline(3.0, color=BAD, ls=":", lw=1, zorder=1)
    ax.text(1.7, 45, "验收线\n1.5% / 3%", color=BAD, fontsize=7.5, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 600)
    ax.set_ylim(0, 100)
    ax.set_xlabel("绝对相对误差（%，对数轴）")
    ax.set_ylabel("累积占比（%）")
    ax.set_title(tag + f"\n对照：{ref}", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
fig.suptitle("图 A2  两项 P0 对拍：v3 误差分布仍整体位于验收线右侧两个数量级",
             fontsize=10.5, y=1.03, x=0.02, ha="left")
save(fig, "FigA2_p11_p10_cdf")


# ── Fig A3：时间维修复 + 有效小时口径 ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
ax = axes[0]
bins = np.linspace(8000, 27000, 96)
ax.hist(aep_old.n_hours, bins=bins, color=OLD, label=f"v2 旧版 (n={len(aep_old)})")
ax.hist(aep.n_hours, bins=bins, color=NEW, alpha=0.9, label=f"v3 新版 (n={len(aep)})")
ax.set_yscale("log")
ax.set_ylim(0.7, 3000)
ax.axvline(8784, color=BAD, ls="--", lw=1)
ax.annotate("闰年上限 8784 h", xy=(8784, 700), xytext=(11500, 700), color=BAD, fontsize=7.5,
            va="center", arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
ax.annotate("旧版约两年\n(17520 h) 324 条", xy=(17520, 330), xytext=(19200, 60),
            color="#4b5563", fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#4b5563", lw=0.8))
ax.set_xlabel("每场-年小时数 n_hours")
ax.set_ylabel("记录数（对数轴）")
ax.set_title("(a) 时间维重复已修复：8752–26352 h → 8752–8784 h", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="upper center")
ax.grid(alpha=0.25, lw=0.6)

ax = axes[1]
ax.scatter(m_new.g_hours, m_new.n_hours, s=7, color=NEW, alpha=0.45, edgecolors="none")
lo, hi = 6800, 8900
ax.plot([lo, hi], [lo, hi], color=BAD, ls="--", lw=1.1, label="同口径应落在此线")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("既有 Gauss 主表有效小时数")
ax.set_ylabel("v3 四情景 n_hours")
ax.set_title("(b) 有效小时口径未对齐：1153/1153 不一致，中位差 661 h", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(alpha=0.25, lw=0.6)
fig.suptitle("图 A3  时间维已修复，但进入 AEP 积分的有效小时规则仍与主表不同",
             fontsize=10.5, y=1.03, x=0.02, ha="left")
save(fig, "FigA3_hours")


# ── Fig A4：区域错配混淆矩阵 + P11 偏差结构 ──────────────────────────
norm = lambda s: s.fillna("<空>").astype(str).str.lower()
ct = pd.crosstab(norm(m_new.g_region), norm(m_new.region))
order_r = [c for c in ["east_asia", "europe", "us_east"] if c in ct.index]
order_c = [c for c in ["east_asia", "europe", "us_east", "<空>"] if c in ct.columns]
ct = ct.loc[order_r, order_c]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
ax = axes[0]
im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        diag = ct.index[i] == ct.columns[j]
        ax.text(j, i, v, ha="center", va="center", fontsize=9,
                color="white" if v > ct.values.max() * 0.55 else "#111827",
                fontweight="bold" if diag else "normal")
ax.set_xticks(range(ct.shape[1]), ct.columns, fontsize=8)
ax.set_yticks(range(ct.shape[0]), ct.index, fontsize=8)
ax.set_xlabel("v3 四情景表标注的 region")
ax.set_ylabel("既有 Gauss 主表 region")
ax.set_title("(a) 区域错配 659/1153（57%）；美东 45 条 0 条标对", fontsize=9.5)

ax = axes[1]
ratio = (m_new.P11_kWh / m_new.g_AEP)
ax.hist(ratio.clip(0, 4), bins=np.linspace(0, 4, 70), color=NEW)
ax.axvline(1, color=BAD, ls="--", lw=1.1)
ax.text(1.05, ax.get_ylim()[1] * 0.9, "应为 1.0", color=BAD, fontsize=8)
ax.set_xlabel(r"$P_{11}$ / 既有 Gauss 实际 AEP")
ax.set_ylabel("记录数")
ax.set_title(f"(b) 偏差是双向的：中位 {ratio.median():.3f}，而 F57 系列高至 3.6 倍", fontsize=9.5)
ax.grid(alpha=0.25, lw=0.6)
fig.suptitle("图 A4  两个未修复根因：区域来源错误，且 $P_{11}$ 偏差不是单一尺度问题",
             fontsize=10.5, y=1.03, x=0.02, ha="left")
save(fig, "FigA4_region_bias")


# ── Fig A5：场-年覆盖 ────────────────────────────────────────────────
years = list(range(2014, 2025))
tgt = set(zip(gauss.farm_id, gauss.year))
old = set(zip(aep_old.farm_id, aep_old.year))
new = set(zip(aep.farm_id, aep.year))
cnt = pd.DataFrame({
    "目标（Gauss 主表）": [sum(1 for f, y in tgt if y == yr) for yr in years],
    "v2 旧版已算": [sum(1 for f, y in old if y == yr) for yr in years],
    "v3 新版已算": [sum(1 for f, y in new if y == yr) for yr in years],
}, index=years)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                         gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
w, x = 0.27, np.arange(len(years))
ax.bar(x - w, cnt["目标（Gauss 主表）"], w, color="#cbd5e1", label="目标：Gauss 主表 1203")
ax.bar(x, cnt["v2 旧版已算"], w, color=OLD, label=f"v2 旧版 {len(old)}")
ax.bar(x + w, cnt["v3 新版已算"], w, color=NEW, label=f"v3 新版 {len(new)}")
ax.set_xticks(x, years, rotation=45, fontsize=8)
ax.set_ylabel("场-年记录数")
ax.set_title("(a) 2014–2017 已从 0 补齐，各年仍略低于目标", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.25, lw=0.6)

ax = axes[1]
miss = sorted(tgt - new)
mf = pd.Series([f for f, y in miss]).value_counts().sort_values(ascending=False)
ax.barh([f"F{i}" for i in mf.index[:14]][::-1], mf.values[:14][::-1], color=BAD, height=0.65)
ax.set_xlabel("该场缺失的年数")
ax.set_title(f"(b) 50 个缺失场-年集中在 {mf.size} 个风场（前 14 名）", fontsize=9.5)
ax.grid(axis="x", alpha=0.25, lw=0.6)
ax.tick_params(labelsize=8)
fig.suptitle("图 A5  样本覆盖：897 → 1153 场-年，距 1203 目标仍缺 50 且未给出缺失原因",
             fontsize=10.5, y=1.03, x=0.02, ha="left")
save(fig, "FigA5_coverage")
print("缺失场-年明细（前若干）：", miss[:12], "... 共", len(miss))

# ── Fig A6：误差随风场规模变化（否定"统一系数可修正"） ────────────────
m_new["ratio"] = m_new.P11_kWh / m_new.g_AEP
m_new["err"] = e11_new.values
band = pd.cut(m_new.n_turb, [0, 100, 400, 10000], labels=["≤100 台", "101–400 台", ">400 台"])
st = m_new.groupby(band, observed=True).agg(n=("err", "size"), err=("err", "median"), r=("ratio", "median"))

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
ax = axes[0]
x = np.arange(len(st))
ax.bar(x, st.err, 0.55, color=BAD)
for xi, (v, n) in enumerate(zip(st.err, st.n)):
    ax.text(xi, v + 1.2, f"{v:.2f}%\n(n={n})", ha="center", fontsize=8.5)
ax.axhline(1.5, color="#111827", ls="--", lw=1)
ax.text(2.45, 3.2, "验收线 1.5%", ha="right", fontsize=7.5)
ax.set_xticks(x, st.index)
ax.set_ylim(0, 55)
ax.set_ylabel(r"$P_{11}$ 中位绝对相对误差（%）")
ax.set_title("(a) 误差随风场规模单调放大", fontsize=9.5)
ax.grid(axis="y", alpha=0.25, lw=0.6)

ax = axes[1]
ax.scatter(m_new.n_turb, m_new.ratio, s=8, color=NEW, alpha=0.35, edgecolors="none")
ax.axhline(1, color=BAD, ls="--", lw=1.1)
for xi, r in zip([60, 200, 700], st.r):
    ax.plot([xi * 0.55, xi * 1.8], [r, r], color="#111827", lw=2, solid_capstyle="butt")
    ax.text(xi * 1.9, r, f"分层中位 {r:.3f}", fontsize=7.5, va="center")
ax.set_xscale("log")
ax.set_ylim(0, 3.8)
ax.set_xlim(1.3, 6000)
ax.set_xlabel("当年机组数 n_turb（对数轴）")
ax.set_ylabel(r"$P_{11}$ / 既有 Gauss 实际 AEP")
ax.set_title("(b) 比值从 0.848 降到 0.548，不是统一系数", fontsize=9.5)
ax.grid(alpha=0.25, lw=0.6)
fig.suptitle("图 A6  $P_{11}$ 偏差与风场规模强相关，无法用单一比例因子校正",
             fontsize=10.5, y=1.03, x=0.02, ha="left")
save(fig, "FigA6_scale_dependence")


# ── Fig A7：F66 逐年翻转 + 极端 S_pct ────────────────────────────────
f66 = m_new[m_new.farm_id == 66].sort_values("year")
fig = plt.figure(figsize=(10.4, 4.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.12, wspace=0.28)
axt, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
yrs = f66.year.tolist()
axt.bar(yrs, f66.ratio, 0.5, color=NEW)
axt.axhline(1, color=BAD, ls="--", lw=1)
axt.set_ylabel(r"$P_{11}$ / 参考 AEP", fontsize=8.5)
axt.set_ylim(0, 1.75)
axt.set_xticks(yrs, [""] * len(yrs))
axt.grid(axis="y", alpha=0.25, lw=0.6)
axt.set_title("(a) F66：$P_{11}$ 由高估 1.49 倍翻转为低估 0.72 倍，风速贡献随之崩塌",
              fontsize=9.5)
for xi, v in zip(yrs, f66.ratio):
    axt.text(xi, v + 0.05, f"{v:.3f}", ha="center", fontsize=8)
axb.bar(yrs, f66.S_shapley, 0.5, color=PART)
axb.axhline(0, color="#111827", lw=0.8)
axb.set_ylabel("S_shapley（%）", fontsize=8.5)
axb.set_ylim(-100, 25)
axb.set_xticks(yrs, [int(v) for v in yrs], fontsize=8.5)
axb.grid(axis="y", alpha=0.25, lw=0.6)
for xi, v in zip(yrs, f66.S_shapley):
    axb.text(xi, v + (3 if v >= 0 else -9), f"{v:.1f}", ha="center", fontsize=8)

ax = fig.add_subplot(gs[:, 1])
ext = eff_ext = pd.DataFrame({
    "场-年": ["F13–2022\n英国", "F66–2023\n中国", "F73–2024\n英国"],
    "S_pct": [-102.77, -102.84, -115.08],
    "S_shapley": [-87.20, -87.52, -89.47],
})
x = np.arange(3)
ax.bar(x - 0.18, ext.S_pct, 0.34, color=BAD, label="S_pct")
ax.bar(x + 0.18, ext.S_shapley, 0.34, color="#e8a598", label="S_shapley")
ax.axhline(-100, color="#111827", ls="--", lw=1)
ax.text(2.45, -97, "−100% 线", ha="right", fontsize=7.5)
ax.set_xticks(x, ext["场-年"], fontsize=8)
ax.set_ylabel("风速贡献（%）")
ax.set_ylim(-125, 5)
ax.set_title("(b) 3 条 S_pct < −100% 的记录，无异常清单可核查", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(axis="y", alpha=0.25, lw=0.6)
fig.suptitle("图 A7  偏差在同一风场内年际反号，说明不是固定损耗遗漏（F66 的 M_S_rms=50.0% 即由此驱动）",
             fontsize=10.5, y=1.03, x=0.02, ha="left")
save(fig, "FigA7_f66_extremes")


# ── Fig A8：交互项与结论文档不符 ─────────────────────────────────────
I = io(HERE / "four_scenario_effects_farmyear.csv").I_pct
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.hist(I.clip(-15, 35), bins=np.linspace(-15, 35, 110), color=NEW)
for v, c, lab, dy in [(0.1, BAD, "结论文档声称\n“约 0.1%”", 0.95),
                      (I.median(), "#111827", f"实际中位\n{I.median():.3f}%", 0.72),
                      (I.mean(), PART, f"实际均值\n{I.mean():.3f}%", 0.5)]:
    ax.axvline(v, color=c, ls="--", lw=1.2)
    ax.annotate(lab, xy=(v, ax.get_ylim()[1] * dy), xytext=(v + 5, ax.get_ylim()[1] * dy),
                color=c, fontsize=8, va="center",
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
ax.set_xlabel("交互项 I_pct（%）")
ax.set_ylabel("记录数")
ax.set_title("图 A8  交互项：绝对值 P95 达 21.69%、最大 65.94%，\n"
             "“交互很小、二乘二分解足够稳定”不成立",
             fontsize=10, loc="left", pad=8)
ax.grid(alpha=0.25, lw=0.6)
save(fig, "FigA8_interaction")


print("\n复核数字：")
print(f"  P11 v2 中位 {e11_old.median():.2f}% → v3 {e11_new.median():.2f}%")
print(f"  P10 v2 中位 {e10_old.median():.2f}% → v3 {e10_new.median():.2f}%")
print(f"  区域错配 {int((norm(m_new.g_region) != norm(m_new.region)).sum())}/{len(m_new)}")
print(f"  覆盖 {len(new)}/{len(tgt)}，缺 {len(tgt - new)}")
