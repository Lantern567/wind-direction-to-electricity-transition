# Wind Direction Literature Search Strategy

## Review Purpose

为论文建立四类证据链：

1. 气候变化是否正在改变区域风向、季节风玫瑰和风向集中度。
2. 风向如何通过有效入流、尾流损失、排布效率和出力波动影响风电发电量。
3. 非平稳气候条件下，使用历史最优风资源评估是否会导致风电规划失效。
4. 动态鲁棒规划、智能偏航、储能和跨区消纳如何降低风向不确定性带来的生命周期风险。

## Search Blocks

### Block A：气候变化与风向非平稳

Example queries:

- climate change wind direction shift
- prevailing wind direction trend climate change
- wind rose change climate variability
- East Asian monsoon wind direction change
- westerlies shift wind direction renewable energy

### Block B：风向、尾流与风电出力

Example queries:

- wind direction wake loss wind farm power output
- wind rose turbine layout annual energy production
- wind direction variability wind farm performance
- yaw control wind direction uncertainty AEP
- wind direction concentration wind power variability

### Block C：风资源评估与非平稳风险

Example queries:

- nonstationary wind resource assessment
- climate change wind energy resource assessment
- wind farm design under climate uncertainty
- robust wind farm layout optimization climate
- historical wind climate bias wind farm planning

### Block D：动态鲁棒规划与适应性设计

Example queries:

- robust wind farm layout optimization
- stochastic optimization wind farm planning
- adaptive wind energy planning climate risk
- wind power storage grid integration climate variability
- climate resilience renewable energy infrastructure

## Screening Criteria

Include:

- 直接讨论风向、风玫瑰、主导风向、方向离散度或尾流方向敏感性的研究。
- 使用观测、再分析或气候模式讨论长期风场变化的研究。
- 将风电工程设计、AEP、尾流、排布或偏航控制与方向不确定性连接的研究。
- 提供可借鉴识别方法、反事实设计或鲁棒优化模型的研究。

Exclude:

- 只讨论平均风速且没有风向信息的研究，除非其方法可用于气候非平稳识别。
- 只做短期功率预测且不涉及风向结构变化的研究。
- 缺少数据来源、模型设定或可复核结果的材料。

## Evidence Table Fields

| Field | Description |
| --- | --- |
| citation_key | BibTeX key |
| topic_block | A/B/C/D |
| region | Study area |
| data_source | Observation, reanalysis, climate model, wind farm data |
| time_coverage | Study period |
| wind_direction_metric | WDI, wind rose, circular variance, dominant direction, etc. |
| power_metric | AEP, capacity factor, wake loss, output volatility, forecast error |
| method | Fixed effects, event study, simulation, optimization, etc. |
| main_finding | Key result |
| relevance_to_project | How it supports Result 1, Result 2, or Result 3 |
| limitations | Data or identification limitations |
