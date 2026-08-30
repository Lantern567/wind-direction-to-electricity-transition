# 重算交付物（2026-08-29 学长审计）

本次交付围绕学长《ADAPEN_v41_重算与确认清单_20260829》的审计重算：
全部数字来自实算，脚本内置对账断言；交付物含论文工作版、SI v4、图件、
v6 权威数据与 P1 落盘文件。

## 目录结构

```
重算交付_20260829/
├── 论文/
│   ├── ADAPEN_manuscript_zh_v4.8.docx           学长原件（未动）
│   ├── ADAPEN_manuscript_zh_v4.9.docx           工作版：41 处 SI 交叉引用 + v6 数字 + 新图1
│   ├── 补充信息_Supplementary_Information_v4.docx  SI v4：25 表 + 新图 S1/S2/S3
│   └── 重算回执_20260829.md                     审计回执（P0-1 裁决、待拍板 4 项）
├── 四场景v6权威数据/
│   ├── four_scenario_farm_summary_AUTHORITATIVE.csv  权威表（P0-1 裁决一致）
│   ├── four_scenario_aep_farmyear_v6.csv / effects_farmyear_v6.csv / farm_summary_v6.csv
│   ├── four_scenario_floris_aep_v6.csv          v6 输入
│   ├── run_floris_decomposition_v6.py           主分解脚本
│   ├── validate_v6.py                           v6 验收
│   └── four_scenario_data_README.md             数据说明（已更新 v6 口径）
├── 重算脚本/
│   ├── update_manuscript_v49.py                 正文 41 处引用 + 数字同步
│   ├── swap_fig1_into_v49.py                    图1 换入 v4.9（同 rId + 图注）
│   └── build_si_docx_v4.py                      SI v4 构建
├── 图件/
│   ├── 构建脚本/  build_fig1_senior_nat.py、build_figS2_basic_data.py、
│   │              build_figS3_sensitivity.py、build_si_figure_s1_en.py、
│   │              nc_style_nat.py（依赖）
│   └── 成品/      Fig1_senior_nat.png（4342×5286）、FigS1_paradigm_layouts.png、
│                  FigS2_basic_data.png、FigS3_sensitivity.png
├── output-new/                                   P1 落盘 27 个文件
│    wp1_geometry_frozen.csv、wp3_climate_weights.npz、wp3_farm_concentration.csv、
│    wp4_wake_endmember.csv、wp5_shape_climate_model.csv、wp5c_*（7）、wp5d_*（2）、
│    wp6_harmonic_reconstruction.csv、wp6c_*（2）、wp7a_*（2）、wp7c_*（3）、
│    wp7d_*（2）、wp7e_orientation_value_by_template.csv
└── 补算输入/                                     重算输入（溯源）
     orientation_gain.csv、wp9c_farm_metrics.csv、wp9g_windwindow_diagnosis.csv
```

## 关键口径（v6 权威）

- 越线：n≥5 = 4 场（F57/F66/F91/F157，4/108=3.7%）；放宽 n≥3 加 F155（5/146=3.4%）
- 比值 1.14–3.19；F160 比值 0.96 掉线（美国东海岸移出图 1 的 b/c/d）
- 面板 (e)：学长原图 5.6%/37% 不可复现，实算 1.5%/16.8%（待学长裁决）

## 未包含（本地备份/参考）

- 所有 `*_旧图备份.docx`、`*_本轮前备份.docx`（在 结论三、四重算结果/ 原位置保留）
- 参考材料（Nature Energy 框架 v3.x、汇报 docx、s41560 PDF 等）
- 图 2/3/4 成品（本次仅图 1 + SI 三图按用户要求重画）
