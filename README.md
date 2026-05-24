# Wind-direction-to-electricity transition

本仓库用于推进“气候变化下风向格局演变对风电出力的影响研究”。项目参考 `Lantern567/renewable-energy-climate-risk-insurance` 的研究骨架，但这是一个独立新仓库：用于管理本项目的 Word 原稿、研究方案、数据设计、文献检索、图表规划和后续脚本。

## Research Question

风向不是只描述天气的气象变量，而是决定风电场有效入流方向、风机排布效率、尾流损失和出力稳定性的关键设计参数。本研究关注：

在气候非平稳背景下，风电场按历史或当年最优风玫瑰建设，是否会出现“建成时最优、运行中失效”的问题？

## Core Logic

1. 证明风向变化会显著影响风电出力，并且影响不只是风速变化的附属结果。
2. 使用近 20 年风速风向数据做逐年反事实，检验“当年最优建设方案”在后续真实风场中的失效速度。
3. 建立动态鲁棒风电规划框架，把风向不确定性转化为方向冗余、智能偏航、储能、跨区消纳和技改窗口等工程规则。

## Repository Map

- `manuscript/source-word/`: 本地 Word 原稿。
- `manuscript/research-plan.md`: Result 框架、研究背景、识别模型和完整研究方案。
- `manuscript/one-page-concept.md`: 一页研究构想。
- `manuscript/figure-plan.md`: 预期图表、面板设计、输入数据和产出指标。
- `data/docs/wind-direction-data-sources.md`: 数据源和处理规则。
- `data/docs/wind-direction-data-dictionary.csv`: 核心变量、构造方法和用途。
- `literature/wind-direction-search-strategy.md`: 文献检索关键词、筛选标准和证据表设计。
- `references/wind-direction-reference-map.md`: 参考文献分组和后续 BibTeX 管理框架。

## Current Milestones

| Milestone | Target Date | Output |
| --- | --- | --- |
| 研究背景与问题界定 | 2026-06-02 | 完成风向作为气候风险变量的论证 |
| Result 1 | 2026-06-30 | 风向变化对出力影响的显著性和效果大小 |
| Result 2 | 2026-07-21 | 逐年最优建设方案的反事实失效曲线 |
| Result 3 | 2026-08-18 | 动态鲁棒风电规划框架和区域规则 |
| 讨论与定价含义 | 2026-08-31 | 风向稳定性作为气候敏感型设计服务的价值解释 |

## Working Rules

- 大体量原始数据不直接提交到 Git，优先放在 `data/raw/`、`data/interim/`、`data/processed/` 等本地目录。
- 需要长期维护的内容优先使用 Markdown、CSV、BibTeX 和脚本。
- Word 原稿保留在 `manuscript/source-word/`，后续正式写作内容同步拆分到 Markdown。
