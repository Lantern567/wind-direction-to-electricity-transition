# -*- coding: utf-8 -*-
"""四情景 v6 派生层重建（§2.7 / 图 7 的唯一数据入口）

背景：廷显 2026-08-10 交付 v6 时只更新了原始 AEP 表
`output/four_scenario_floris_aep_v6.csv`；同目录下的
`four_scenario_{effects_farmyear,aep_farmyear,farm_summary,threshold_farms}.csv`
与 v5 逐字节相同（git 显示为纯重命名），因此**不可直接使用**。
本脚本从 v6 原始 AEP 重算派生层，输出到本目录，供图 7 与正文引用。

分解公式由 v5 的 (aep_farmyear, effects_farmyear) 配对反推，
并以 v5 数据自校验通过（最大偏差 2.6e-14）：
    total = (P11-P00)/P11*100
    S     = (P10-P00)/P11*100        # 只换风速
    D     = (P01-P00)/P11*100        # 只换风向
    I     = total - S - D
    S_shapley = S + I/2 ;  D_shapley = D + I/2
分母取 P11（该场-年实际 AEP）。

噪声尺度口径（见正文 §2.7 与 §7 红线）：
    M_S_std = 逐年 S_shapley 的样本标准差（ddof=1）—— 主口径
    M_S_rms = sqrt(mean(S_shapley^2))            —— 仅作对照，含基准偏置
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(
    HERE, "..", "..", "wind-direction-to-electricity-transition",
    "四场景风速风向分解贡献", "output"))


def decompose(df):
    d = df.copy()
    d["total_pct"] = (d.P11_kWh - d.P00_kWh) / d.P11_kWh * 100
    d["S_pct"] = (d.P10_kWh - d.P00_kWh) / d.P11_kWh * 100
    d["D_pct"] = (d.P01_kWh - d.P00_kWh) / d.P11_kWh * 100
    d["I_pct"] = d.total_pct - d.S_pct - d.D_pct
    d["S_shapley"] = d.S_pct + d.I_pct / 2
    d["D_shapley"] = d.D_pct + d.I_pct / 2
    return d


def summarize(eff):
    g = eff.groupby("farm_id")
    s = pd.DataFrame({
        "n_years": g.size(),
        "country": g.country.first(),
        "G_mean": g.gain_pct.mean(),
        "M_S_mean": g.S_shapley.mean(),
        "M_S_std": g.S_shapley.std(ddof=1),
        "M_S_rms": g.S_shapley.apply(lambda x: np.sqrt((x ** 2).mean())),
        "M_D_rms": g.D_shapley.apply(lambda x: np.sqrt((x ** 2).mean())),
    }).reset_index()
    s["R_std"] = s.G_mean / s.M_S_std
    s["R_rms"] = s.G_mean / s.M_S_rms
    return s


def main():
    # --- 公式自校验：用 v5 原始 AEP 复算 v5 effects ---
    v5a = pd.read_csv(os.path.join(SRC, "four_scenario_aep_farmyear.csv"), encoding="utf-8-sig")
    v5e = pd.read_csv(os.path.join(SRC, "four_scenario_effects_farmyear.csv"), encoding="utf-8-sig")
    chk = decompose(v5a).merge(v5e, on=["farm_id", "year"], suffixes=("_c", "_r"))
    err = max(abs(chk[f"{c}_c"] - chk[f"{c}_r"]).max()
              for c in ["S_pct", "D_pct", "I_pct", "total_pct", "S_shapley", "D_shapley"])
    assert err < 1e-6, f"分解公式自校验失败: {err}"
    print(f"[自校验] 分解公式复现 v5 effects，最大偏差 {err:.2e}  OK")

    # --- v6 派生 ---
    v6 = pd.read_csv(os.path.join(SRC, "four_scenario_floris_aep_v6.csv"), encoding="utf-8-sig")
    assert not v6.duplicated(["farm_id", "year"]).any(), "v6 存在重复场-年"
    eff = decompose(v6).merge(v5e[["farm_id", "year", "country", "gain_pct"]],
                              on=["farm_id", "year"], how="left")
    assert eff.gain_pct.notna().all(), "gain_pct 有缺失"
    summ = summarize(eff)

    eff.to_csv(os.path.join(HERE, "v6_effects_farmyear.csv"), index=False, encoding="utf-8-sig")
    summ.to_csv(os.path.join(HERE, "v6_farm_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"[输出] v6_effects_farmyear.csv ({len(eff)} 场-年) / v6_farm_summary.csv ({len(summ)} 场)")

    d5 = summ[summ.n_years >= 5]
    print(f"\n主样本 n>=5: {len(d5)} 场")
    print(f"  M_S_std 中位 {d5.M_S_std.median():.2f}% | M_S_rms 中位 {d5.M_S_rms.median():.2f}%")
    print(f"  M_S_mean 场均 {d5.M_S_mean.mean():+.2f}pp")
    print(f"  R_std>1: {(d5.R_std>1).sum()}/{len(d5)} -> {sorted(d5[d5.R_std>1].farm_id)}")
    print(f"  R_rms>1: {(d5.R_rms>1).sum()}/{len(d5)} -> {sorted(d5[d5.R_rms>1].farm_id)}")
    d3 = summ[summ.n_years >= 3]
    print(f"次样本 n>=3: {len(d3)} 场 | R_std>1: {(d3.R_std>1).sum()} -> {sorted(d3[d3.R_std>1].farm_id)}")
    print(f"偏置占比 mean^2/rms^2 = {eff.S_shapley.mean()**2/(eff.S_shapley**2).mean()*100:.1f}%")
    ys = eff.groupby("year").S_shapley.mean()
    print(f"逐年场均为负: {[int(y) for y in ys[ys<0].index]}")


if __name__ == "__main__":
    main()
