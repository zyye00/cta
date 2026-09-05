# 国内商品市场环境与基金匹配分析

本项目使用国内商品期货行情、板块映射和南华商品指数，构建板块等权的波动—相关性市场环境，并描述不同市场状态下南华商品指数及私募基金净值的历史表现。

## 项目结构

```text
.
├── config/
│   ├── config.yaml
│   └── commodity_sectors.csv
├── notebooks/
│   ├── 01_download.ipynb
│   ├── 02_analysis.ipynb
│   ├── 03_fund_analysis.ipynb
│   └── 04_sector_volatility_contributions.ipynb
├── src/
│   └── cta_research/
│       ├── config.py
│       ├── cta.py
│       ├── environment.py
│       ├── funds.py
│       ├── histories.py
│       ├── performance.py
│       ├── plotting.py
│       ├── regimes.py
│       ├── research.py
│       └── rqdata.py
└── tests/
```

## 使用流程

1. 运行 `notebooks/01_download.ipynb`，获取商品期货原始行情。
2. 在 `config/config.yaml` 的 `paths.external_nhci_file` 中配置南华商品指数文件。
3. 运行 `notebooks/02_analysis.ipynb`，查看板块等权市场环境与 NHCI 结果。
4. 将私募基金净值 Excel 放入 `data/`，运行 `notebooks/03_fund_analysis.ipynb`，查看基金与 NHCI 的匹配结果。
5. 可选运行 `notebooks/04_sector_volatility_contributions.ipynb`，查看各板块对波动率的贡献结构。

三本分析 Notebook 可以独立运行，不需要先运行另一份分析 Notebook。执行结果已保存于 Notebook，打开即可查看正式图表和统计表。

03 Notebook 会将基金产品名显示为 `基金 01`、`基金 02` 等别名，并按 `config/config.yaml` 中的 `fund_categories` 分为“截面”和“趋势”两类，仅用于组织展示，不构造类别组合。真实名称与别名的映射保存在本地 `data/fund_name_mapping.yaml`，未写入 Notebook 输出或上传。

## 数据格式

商品行情至少包含：

```text
underlying_symbol | date | close
```

南华商品指数支持 Excel、CSV 或 Parquet，至少包含日期列和指数点位列；中文列名 `日期`、`指数点位` 可直接读取。

私募基金净值支持放在 `data/` 下的 Excel 文件，至少包含日期列和净值列；支持 `复权净值`、`累计净值`、`单位净值`、`净值` 等列名。

## 研究口径

- 每日以完成 ramp-in 的有效品种为等权基准形成板块组合收益；品种和板块首次通过40条最小观测门槛并形成有效波动率后，按六个月线性 ramp-in 纳入，并按当日有效权重重新归一化。
- 每月使用此前三个自然月的板块组合收益，计算板块等权年化波动率。
- 月度指标只使用已经完成的月份；行情停在月中时不生成该月月末指标。
- 板块组合两两相关性取绝对值，并按两个板块的 ramp-in 权重乘积加权平均；权重缺失的板块对不参与当月聚合。
- 高低状态阈值使用当月之前已完整结束的 36 个月滚动中位数，状态从下一个交易日起生效。
- `INSUFFICIENT_HISTORY` 区间不进入图表的正式区间和条件绩效统计。
- 02中的 NHCI 使用252个交易日年化。03中每只基金和 NHCI 使用同一组共同日期及同一个年化因子：日频252、周频52、月频12，更低频按自然日间隔折算。
- 非连续基金净值的区间收益归入区间结束日对应的状态，跨状态区间不拆分。
