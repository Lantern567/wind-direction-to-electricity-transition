# 四情景风速—风向贡献分解：数据说明

生成日期：2026-08-04

## 文件清单

### （FLORIS 查找表）

| 文件 | 行数 | 说明 |
|------|------|------|
| `four_scenario_lookup_table.csv` | 227,772 | 廷显：每个独特农场配置的尾流效率矩阵（已含电气损耗 0.9215） |
| `four_scenario_config_map.csv` | 916 | 场-年 → config_hash 的去重映射 |
| `compute_four_scenario_lookup.py` | — | 编排脚本（含多进程 FLORIS 计算逻辑） |

### （四情景分解）

| 文件 | 行数 | 说明 |
|------|------|------|
| `four_scenario_aep_farmyear.csv` | 897 | 每场-年四情景 AEP（P00/P10/P01/P11）及无尾流 AEP |
| `four_scenario_effects_farmyear.csv` | 897 | 每场-年风速主效应 S、风向主效应 D、交互 I、Shapley 贡献、朝向收益 gain_pct |
| `four_scenario_farm_summary.csv` | 171 | 风场级汇总：G、M_S、M_D、R_i |
| `four_scenario_threshold_farms.csv` | 171 | 带越线标记 |
| `four_scenario_decomposition.py` | — | 主计算脚本 |
| `four_scenario_data_README.md` | — | 本文件 |
| `four_scenario_conclusions.md` | — | 结论文档 |

## 列定义

### four_scenario_lookup_table.csv（廷显）

| 列 | 含义 |
|----|------|
| config_hash | 排布配置哈希（MD5 前 12 位） |
| ws_bin_m_s | 风速档中心值（m/s，19 档） |
| wd_sector_deg | 风向扇区中心角度（°，36 扇区，气象方向） |
| wake_efficiency | 该 (ws, wd) 下的尾流效率（0-1） |
| p_noWake_kW | 该 (ws, wd) 下全场无尾流功率（kW） |
| n_turbines | 该配置的机组台数 |

### four_scenario_config_map.csv（廷显）

| 列 | 含义 |
|----|------|
| farm_id, year | 风场标识和运营年份 |
| config_hash | 对应的排布配置哈希 |

### four_scenario_aep_farmyear.csv

| 列 | 含义 |
|----|------|
| farm_id, year | 风场标识和运营年份 |
| config_hash | FLORIS 配置哈希（相同排布共享） |
| n_turb, n_hours | 机组台数、该年有效 ERA5 时数 |
| P00_kWh | 历史基准风速 × 历史基准风向下的 AEP |
| P10_kWh | 实际年份风速 × 历史基准风向下的 AEP |
| P01_kWh | 历史基准风速 × 实际年份风向下的 AEP |
| P11_kWh | 实际年份风速 × 实际年份风向下的 AEP（= 实际 AEP） |
| P00_noWake, P11_noWake | 对应无尾流 AEP |

### four_scenario_effects_farmyear.csv

| 列 | 含义 |
|----|------|
| S_pct | 风速主效应 = (P10-P00)/P11×100 |
| D_pct | 风向主效应 = (P01-P00)/P11×100 |
| I_pct | 风速—风向交互 = (P11-P10-P01+P00)/P11×100 |
| total_pct | 总效应 = (P11-P00)/P11×100 = S+D+I |
| S_shapley | Shapley 风速贡献（两条替换路径均值） |
| D_shapley | Shapley 风向贡献 |
| gain_pct | 朝向优化收益 = (s1_opt_AEP - real_AEP)/real_AEP×100 |

### four_scenario_farm_summary.csv

| 列 | 含义 |
|----|------|
| G_mean | 朝向收益多年算术均值 |
| M_S_rms | 逐年 Shapley 风速贡献的均方根（RMS） |
| M_D_rms | 逐年 Shapley 风向贡献的均方根 |
| R_i | G_mean / M_S_rms（>1 表示朝向收益超过风速年际噪声） |
| H_i | R_i > 1 的指示变量 |
| n_years | 该场有效年份数（主分析要求 ≥5） |
| country, region | 国家和海域 |

## 历史基准口径

历史基准采用该农场全部可用 ERA5 年份（2014–2024，约 11 年）的池化联合分布 p_hist(d) × p_hist(v|d)，而非任务书首选的 1981–2010 年逐小时基准。原因：1981–2010 年 ERA5 仅为逐日采样（12:00 UTC），且仅覆盖东亚区域。本口径在方法节中应标注为"2014–2024 多年代理基准"。

## 验收

- 分解闭合：S+D+I 与 total 误差均值 7.5×10⁻¹⁶ 个百分点
- Shapley 闭合：S_shapley + D_shapley 与 total 误差均值 4.4×10⁻¹⁶ 个百分点
- 概率表和：每场-年四情景概率和归一化误差 <10⁻¹⁰
