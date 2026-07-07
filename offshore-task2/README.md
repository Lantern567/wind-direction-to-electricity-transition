# 任务二：全球海上风电场逐时出力核算（FLORIS v4.6）

## 一句话

用 FLORIS v4.6 标准库计算全球 171 个海上风场 2014-2024 年的逐时出力、尾流损失和容量因子，三模型（Gauss/Jensen/CC）对比验证稳健性。

## 核心结果

| 指标 | 数值 |
|------|------|
| Gauss CF 均值 | 43.1% |
| Gauss WakeLoss 均值 | 13.5% |
| Jensen CF 均值 | 42.1% |
| Jensen WakeLoss 均值 | 15.7% |
| 风向变化影响（dAEP_WD） | -0.4%（净负面影响） |
| 三模型 CF 最大差异 | < 5 pp |
| 机型敏感性 | 6MW→15MW CF 单调递增，结论稳健 |

详见 `output/task2_report_FINAL.docx`。

## 文件结构

```
├── floris_config.py          FLORIS 配置层（TI参数化、机型管理、分箱）
├── task2_floris.py           主计算引擎（分箱预计算 + 逐时查表）
├── task2_audit_v2.py         审计证据（旋转测试、打乱对照、精度自检）
├── horns_rev_benchmark.py    Horns Rev 1 基准验证
├── run_sensitivity.py        机型敏感性分析（5场×5机型×2模型）
├── make_report_final.py      Word 报告生成脚本
├── make_viz_v2.py            可视化生成
├── make_comparison_viz.py    新旧版本对比图生成
├── fix_fig5.py               图5（Cartopy 海岸线）再生脚本
├── data/                     机型 YAML 配置文件
│   ├── iea_10MW.yaml         IEA 10MW 参考机型（本研究默认）
│   ├── ow_6MW.yaml           PyWake 动量理论推导 6MW
│   ├── ow_8MW.yaml           PyWake 动量理论推导 8MW
│   ├── ow_10MW.yaml          PyWake 动量理论推导 10MW
│   ├── ow_12MW.yaml          PyWake 动量理论推导 12MW
│   ├── ow_15MW.yaml          PyWake 动量理论推导 15MW
│   ├── iea_15MW.yaml         FLORIS 内置 IEA 15MW
│   ├── nrel_5MW.yaml         FLORIS 内置 NREL 5MW
│   └── vestas_v80.yaml       Horns Rev 基准验证用 V80
├── output/
│   ├── task2_annual_floris.csv       主数据（3,546行，171场×11年×3模型）
│   ├── task2_summary_v4.csv          旧版参考数据（Numba 自研引擎）
│   ├── task2_counterfactual.csv      反事实分析（916次风向对照实验）
│   ├── turbine_sensitivity.csv       机型敏感性结果（50次运行）
│   ├── audit_rotation_floris.csv     旋转测试结果
│   ├── audit_shuffle_floris.csv      打乱对照结果
│   ├── audit_precision_check.csv     分箱精度验证
│   ├── task2_report_FINAL.docx       ★ 终版 Word 报告
│   ├── task2_report_FLORIS.docx      备用报告
│   └── figures_v2/                   11 张可视化图（PNG, 200dpi）
└── README.md               本文件
```

## 环境要求

### Python 库

```
pip install floris==4.6       # 尾流仿真引擎（核心）
pip install py_wake           # 机型 YAML 生成（run_sensitivity.py 用）
pip install numpy scipy matplotlib netCDF4
pip install cartopy           # 图5 需要（fix_fig5.py）
pip install python-docx       # 报告生成（make_report_final.py）
pip install pyyaml            # YAML 读写
```

Python 3.10+ 推荐。

### 数据依赖（不包含在本仓库中）

1. **任务零底座**：`offshore-task0/output/task0/` 下的 `farms_master.csv`、`turbine_coordinates.csv`。路径需在 `floris_config.py` 中修改 `TASK0_DIR` 变量。

2. **ERA5 逐小时风速数据**：`data/era5_{region}_{year}.nc`（四区域×11年=44个文件，总计 ~22GB）。下载来源见任务零 `caliber_config.yaml`。

3. **GEBCO 水深数据**：`data/GEBCO_2024.tif`（4.6GB），用于风机角色标注。已经由任务零处理完，风力核算本身上下游不直接依赖。

## 快速开始

### 1. 修改路径

打开 `floris_config.py`，修改：

```python
TASK0_DIR = r"你的路径/offshore-task0/output/task0"
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
```

### 2. 跑一个小场验证环境

```bash
python -c "
from floris_config import *
from task2_floris import *
coords = load_task0_coordinates()
farms = load_farms_master()
print(f'{len(farms)} farms loaded')
"
```

### 3. 跑全量 Gauss+Jensen

```bash
python task2_floris.py
```

支持断点续传：中断后重跑自动从已完成记录继续。

### 4. 跑审计证据

```bash
python task2_audit_v2.py
```

产出 `farm_layout_used.csv`、旋转测试、打乱对照、精度自检。

### 5. 跑机型敏感性

```bash
python run_sensitivity.py
```

5个代表场 × 5档机型 × 2个尾流模型 = 50次运行，约30-60分钟。

### 6. 生成图表和报告

```bash
python make_viz_v2.py          # 8张核心图
python make_comparison_viz.py  # 新旧对比图
python fix_fig5.py             # 图5 Cartopy 版
python make_report_final.py    # Word 报告
```

## 分箱加速说明

为避免逐小时跑 FLORIS（171场×11年×8760h × ~50ms ≈ 228,000小时），采用扇区分箱方案：

- Gauss: 19 WS bins × 36 WD sectors = 684 combos/farm-year
- Jensen: 14 WS bins × 18 WD sectors = 252 combos/farm-year（顶帽模型对粗化不敏感，已验证误差 < 1%）
- CC: 同 Gauss 分箱，但大场（>200台）计算代价极高易崩溃

精度自检结果：AEP 误差 1.35%，WakeLoss 误差 -1.1 个百分点。

## 已知限制

1. **CC 模型大场不完整**：F0(928台)、F1(589台)、F3(443台) 等大场部分年份的 Cumulative Curl 模型因 FLORIS 计算代价过高未完成。中小场全部覆盖。
2. **统一 IEA 10MW 机型**：机型敏感性已证明 6MW→15MW 结论定性不变。
3. **ERA5 分辨率限制**：0.25°≈25km，大多数风场（中位12km）小于单格点，无法解析风场内部风速梯度。
4. **电气损耗全局固定**：0.95×0.97=0.92，未按国家/海域差异化。

## 引用

本代码基于 FLORIS v4.6.6（NREL 2024）和 PyWake（DTU 2023）。如使用本研究数据或代码，请引用任务书及本仓库。
