# 四情景风速—风向贡献分解：数据说明 (v5 最终版)

**2026-08-07 | 琪明**

## 版本

| 版本 | 日期 | P11 中位误差 | M_S 中位 | R_i>1 | 状态 |
|------|------|-------------|---------|--------|------|
| v3 | 08-05 | ~18% | 7.2% | F91, F157 | 废弃 |
| v4 | 08-07 | 1.98% | 11.3% | F66, F157 | 废弃 |
| **v5** | **08-07** | **0.000%** | **6.6%** | **F66, F91, F157** | **当前** |

## 交付文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `four_scenario_aep_farmyear.csv` | 1,203 | 每场-年四情景 AEP (P00/P10/P01/P11) + S3 参考 AEP |
| `four_scenario_effects_farmyear.csv` | 1,203 | 每场-年 S/D/I 效应 + Shapley + 朝向收益 |
| `four_scenario_farm_summary.csv` | 171 | 风场级 G, M_S, M_D, R_i (108 场 ≥5 年) |
| `four_scenario_threshold_farms.csv` | 171 | 越线标记 (全球 M_S / 自身 M_S) |
| `four_scenario_qa.md` | — | 验收清单 |
| `four_scenario_conclusions.md` | — | 结论 (中文) |
| `four_scenario_data_README.md` | — | 本文件 |

## 列定义

与 [v3 README](四场景风速风向分解贡献/four_scenario_data_README.md) 一致，新增列：

| 列 | 含义 |
|----|------|
| `P11_err_pct` | P11 与 S3 Gauss 实际 AEP 的相对误差 (%) |
| `G_ew_mean` | 能源加权朝向收益 (以 P11 为权重) |
| `M_S_mae` | 逐年 Shapley 风速贡献的平均绝对值 |
| `M_S_std` | 逐年 Shapley 风速贡献的标准差 |
| `P11_median_err` | 该场 P11 中位误差 |

## 计算方法 (v5)

1. **P11 (实际 VS × 实际 WD)**: 从 S3 `task3_s3_comparison.csv` 直接复制 Gauss+real 的 `AEP_kWh`
2. **P10 (实际 WS × 历史 WD)**: 916 对从 counterfactual `AEP_baseWD_kWh` 复制, 287 对 FLORIS 库新算
3. **P01 (历史 WS × 实际 WD)**: FLORIS 库 36×19 查找表 × 分位数映射 WS × 实际年 WD 频率
4. **P00 (历史 WS × 历史 WD)**: FLORIS 库查找表 × 10 轮历史 WD/WS 随机采样均值

所有 FLORIS 计算均使用 FLORIS 库 Gauss 尾流模型 (与 S3 一致), IEA 10MW 机型, 电气损耗 0.9215。

## 参考数据

- S3 Gauss 实际 AEP: `task3/task3_s3_comparison.csv`
- 最优朝向: `task3/task3_s1_optimal_orientation.csv`
- 廷显 v5 输入: `四场景风速风向分解贡献/four_scenario_floris_aep_v5.csv`
- 分解脚本: `补算/output/run_floris_decomposition_v5.py`
