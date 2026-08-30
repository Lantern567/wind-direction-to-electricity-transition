# 四情景风速—风向贡献分解：数据说明

**2026-08-07 初版 | 2026-08-09 更新（学长审计 + 交付物补齐） | 琪明**

## 版本

| 版本 | 日期 | P11 中位误差 | M_S 中位 | R_i>1 | 状态 |
|------|------|-------------|---------|--------|------|
| v3 | 08-05 | ~18% | 7.2% | F91, F157 | 废弃 |
| v4 | 08-07 | 1.98% | 11.3% | F66, F157 | 废弃 |
| v5 | 08-07 | 0.000% | ~6.6%（含偏置） | F66, F91, F157 | 废弃（输入已删除，脚本留档勿重跑） |
| **v6** | **08-29** | **<0.01%** | **权威 M_S_std（逐场）** | **越线 n>=5: F57, F66, F91, F157（+n>=3: F155）** | **当前** |

⚠️ **v5 已知问题（学长审计发现，v6 已修复）：**
- P01/P00 使用查找表，与 P11/P10 逐时 FLORIS 不在同一口径 → MS 含 ~52-57% 偏置能量
- P10 2014-2017 与 2018-2024 算法混搭 → 4.6pp 人造断点

✅ **v6 修复（2026-08-29，权威口径）：**
- P10 全部改为 FLORIS 逐时自算（×10 轮），消除算法混搭
- P01/P00 改为逐时精确功率曲线（np.interp），消除查找表偏置
- 修复只改变 M_S_std → 越线场 6→5（F160 比值 1.03→0.96 掉线；比值 1.14–3.19）
- 权威表：`output/four_scenario_farm_summary_AUTHORITATIVE.csv`

## 交付文件

### 主数据文件 (output/)

| 文件 | 行数 | 说明 |
|------|------|------|
| `output/four_scenario_aep_farmyear_v6.csv` | 1,203 | 每场-年四情景 AEP (P00/P10/P01/P11) + S3 参考 AEP（v6 口径） |
| `output/four_scenario_effects_farmyear_v6.csv` | 1,203 | 每场-年 S/D/I 效应 + Shapley + 朝向收益（v6 口径） |
| `output/four_scenario_farm_summary_AUTHORITATIVE.csv` | 171 | **权威表**：风场级 G, M_S_std, M_D, R_i（学长清单 P0-1 裁决一致） |
| `output/four_scenario_farm_summary_v6.csv` | 171 | v6 派生表（与 AUTHORITATIVE 同源） |
| `output/four_scenario_floris_aep_v6.csv` | 1,203 | v6 输入：FLORIS 逐时四情景 AEP |

### 运行日志与验证 (output/)

| 文件 | 说明 |
|------|------|
| `output/run_log.csv` | 分解步骤日志（步骤、时间、状态、详情） |
| `output/probability_checks.csv` | 分位数映射验证清单（廷显侧生成，当前为 v6 占位符） |

### 诊断图 (output/figures/)

| 文件 | 说明 |
|------|------|
| `output/figures/FigA1_acceptance_board_v5.png` | v5 验收板（PASS/FAIL 八项检查） |
| `output/figures/FigD1_four_scenario_timeseries.png` | 6 代表农场的四情景 AEP 年序列 |
| `output/figures/FigD2_shapley_decomposition.png` | 6 代表农场的 S/D Shapley 分解 |
| `output/figures/FigD3_MS_vs_G_scatter.png` | 108 场 M_S vs G 散点图 |
| `output/figures/FigD4_S_shapley_boxplot.png` | S_shapley 逐年箱线图（全样本） |
| `output/figures/FigD5_P10_discontinuity_check.png` | P10/P11 2017→2018 断点检验 |
| `output/figures/FigD6_MS_bias_decomposition.png` | MS 偏置分解 (mean² vs rms²) |

### 文档

| 文件 | 说明 |
|------|------|
| `four_scenario_qa.md` | 验收清单（含学长审计修正） |
| `four_scenario_conclusions.md` | 结论（中文，含 provisional 标注） |
| `four_scenario_data_README.md` | 本文件 |

## 列定义

### four_scenario_aep_farmyear.csv

| 列 | 含义 |
|----|------|
| `farm_id` | 风场编号 (1-171) |
| `year` | 年份 (2014-2024) |
| `country` | 国家 |
| `P00_kWh` | 历史风速 × 历史风向 (基准 AEP) |
| `P10_kWh` | 实际风速 × 历史风向 |
| `P01_kWh` | 历史风速 × 实际风向 |
| `P11_kWh` | 实际风速 × 实际风向 (= S3 AEP) |
| `AEP_kWh` | S3 Gauss+real 参考 AEP (应 = P11) |
| `P11_err_pct` | P11 与 S3 相对误差 (%) |

### four_scenario_effects_farmyear.csv

| 列 | 含义 |
|----|------|
| `S_pct` | 风速主效应 = (P10-P00)/P11 × 100 |
| `D_pct` | 风向主效应 = (P01-P00)/P11 × 100 |
| `I_pct` | 交互效应 = (P11-P10-P01+P00)/P11 × 100 |
| `total_pct` | 总效应 = (P11-P00)/P11 × 100 |
| `S_shapley` | Shapley 风速贡献 = 50[(P10-P00)+(P11-P01)]/P11 |
| `D_shapley` | Shapley 风向贡献 = 50[(P01-P00)+(P11-P10)]/P11 |
| `gain_pct` | 朝向优化收益 G (%) |

### four_scenario_farm_summary.csv

| 列 | 含义 |
|----|------|
| `G_mean` / `G_ew_mean` | 朝向收益 (算术平均 / 能源加权) |
| `M_S_rms` / `M_S_mae` | 风速噪声 (RMS / MAE) |
| `M_S_std` / `M_S_mean` | S_shapley 标准差 / 均值 |
| `M_D_rms` / `M_D_mae` | 风向效应 (RMS / MAE) |
| `R_i` / `H_i` | G/M_S 比率 / R_i>1 标记 |

## 计算方法 (v6，当前权威口径)

1. **P11 (实际 WS × 实际 WD)**: 从 S3 直接复制 Gauss+real 的 `AEP_kWh`（逐字节相等）
2. **P10 (实际 WS × 历史 WD)**: 全部 FLORIS 逐时自算 ×10 轮（Numba 手写尾流废弃）
3. **P01 (历史 WS × 实际 WD)**: 逐时精确功率曲线（np.interp 替代查找表）
4. **P00 (历史 WS × 历史 WD)**: 逐时精确功率曲线（np.interp 替代查找表）

所有 FLORIS 计算均使用 FLORIS 库 Gauss 尾流模型 (与 S3 一致), IEA 10MW 机型, 电气损耗 0.9215。

## 可复现性

### 分解脚本

```bash
cd 四场景风速风向分解贡献
python run_floris_decomposition_v6.py    # 主分解（当前权威）
python validate_v6.py                    # v6 验收（P11 与 S3 逐字节比对等）
```

### 依赖数据

- v6 输入: `output/four_scenario_floris_aep_v6.csv`（重建生成）
- S3 Gauss 实际 AEP: `task3/task3_s3_comparison.csv`
- 最优朝向: `task3/task3_s1_optimal_orientation.csv`

### 历史留档（勿重跑）

- `run_floris_decomposition_v5.py` / `generate_v5_deliverables.py`：v5 口径生成器，
  其输入 `four_scenario_floris_aep_v5.csv` 已删除；保留仅为审计溯源。

### 环境

- Python ≥ 3.10
- pandas, numpy, scipy
- matplotlib（仅诊断图脚本需要）
