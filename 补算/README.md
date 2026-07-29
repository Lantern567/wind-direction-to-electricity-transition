# 补算（Supplementary Analysis）

几何胜于对齐——基于全球171个真实海上风场的朝向与几何联合后验分析。

## 目录结构

```
补算/
├── README.md                          # 本文件
├── 补算分工.docx                       # 任务分工说明
├── 走廊图谱_Phase1返修意见_20260728.docx # Phase 1 返修意见
├── README_数据说明.docx                # 廷显交付数据说明
├── Supplementary_Analysis_Report.docx  # 最终报告（output/）
│
├── 廷显交付数据/
│   ├── reference_wake_table.csv       # 参考尾流表（36风向×19风速）
│   ├── high_gain_wind_roses.csv       # 7场30年逐日风向（ERA5 baseline）
│   ├── jensen_rotation_56farms.csv    # Jensen旋转重跑75场
│   ├── ow_sensitivity_3farms.csv      # 机型敏感性（3场×3机型）
│   └── joint_optimization_F57.csv     # F57联合优化
│
├── 分析脚本/
│   ├── task1_corridor_atlas_v3.py     # 走廊图谱（Fig 4）
│   ├── task4_8_remaining.py           # 乘积回归+置换+赌注核算
│   ├── task5_wind_rose_verification.py # 风玫瑰+扰动测试
│   ├── task6_bootstrap_ci.py          # 聚类Bootstrap
│   └── generate_final_report.py       # 最终报告生成器
│
└── output/
    ├── Supplementary_Analysis_Report.docx  # 最终报告
    ├── fig4_corridor_atlas.png            # Fig 4 走廊图谱
    ├── task1_corridor_grid.csv            # 全局网格预测
    ├── task1_training_data.csv            # 训练集
    ├── task1_loo_predictions.csv          # LOO预测
    ├── task5_wind_rose_verification.csv   # 风玫瑰WCI
    ├── task5_perturbation_test.csv        # 扰动测试
    ├── task6_bootstrap_ci.csv             # Bootstrap CI
    ├── task6_sign_test_by_country.csv     # 分层符号检验
    └── task8_global_bet.csv               # 全球赌注核算
```

## 核心结论

1. 朝向优化全球均值增益+0.9%，聚类Bootstrap下仍显著（3区域CI [0.87%, 2.31%]）
2. 仅7个农场（4.1%）超过5%，集中于季风走廊和海峡地形整流区
3. WCI×尾流池交互项通过乘积回归（p=0.042）和置换检验（p=0.000）
4. F57联合优化：几何+63.0% vs 朝向+3.7%（17:1）
5. 走廊图谱识别四大高敏感海域，计划装机225GW，年损失上限23.5TWh
