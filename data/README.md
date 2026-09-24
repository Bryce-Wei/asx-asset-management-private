# 数据放置与格式

我将四份原始输入放在 `data/` 目录中，随仓库公开；将便于检查程序流程的模拟样本放在 `data/sample/`。价格文件只包含 Bloomberg 导出的股票价格，客户资料不含客户身份信息。运行生成的结果不提交。

## 完整分析输入

四份原始文件位于 `data/`，保留原文件名及工作簿布局。程序默认从 `data/raw/` 读取，因此运行完整分析或回测时用 `--data-dir data` 指定该目录；把文件复制到 `data/raw/` 则不需要这个参数。各文件的内容与读取约定如下：

| 文件 | 内容与读取约定 | 用途 |
|---|---|---|
| `ASX200top10.xlsx` | 优先读取 `Bloomberg raw` 工作表，以两行表头识别各股票的 `PX_LAST` 列，第一列为日期 | 价格清洗、收益计算、蒙特卡洛模拟与组合优化 |
| `Client_Details.xlsx` | 优先读取 `Data` 工作表，包含唯一的 `client_ID`、`risk_profile`、`age_group` 和数值型持仓特征 | 客户特征标准化与 KMeans 分群 |
| `Economic_Indicators.xlsx` | 读取第一张工作表中的 `Monthly Indicators` 区域；列为月份、行为指标，季度指标不参与当前流程 | 月度宏观指标对齐、OLS 回归与调整量计算 |
| `news_dump.json` | 新闻记录列表，包含 `Equity`、`Source`、`Date/Time`、`Headline` | VADER 标题情绪评分与按股票汇总的权重调整 |

股票代码在载入时统一为资产标签，并用于对齐客户持仓、新闻和最终权重。股票日期与客户编号必须唯一；价格必须为正数，客户特征必须为完整的有限数值。加载器不会改写输入文件。

价格中的缺失值保留为缺失，不以零值代替。客户字段不接受未经处理的缺失记录。宏观数据保留缺失观测，分析阶段按所属月份对齐并向前填充；这一处理不代表指标在当月交易时已发布，具体限制见 [方法与限制](../docs/methodology.md)。

新闻按照输入顺序读取，`Date/Time` 保留为文本。当前情绪模块把每只股票的前 5 条记录视为近期新闻，程序不会自动按时间排序，因此我在整理输入时需要确保每只股票的记录由新到旧排列。每只待配置股票都需要至少一条有效新闻；不足 5 条时使用已有记录。滚动回测则会解析 `Date/Time`（例如 `22 Jul '20 10:05 AM`），按时间排序并只使用信号日收盘前发布的新闻，不依赖输入顺序。

接入其他数据源时，需要同时核对工作表、表头、资产标签、字段单位和日期含义，并相应调整 `asset_management/load_data.py`。NLP 使用的 `vader_lexicon` 是 NLTK 的外部语言资源，首次运行前需安装：`python -m nltk.downloader vader_lexicon`。

## 模拟演示输入

`sample/prices.csv` 第一列为 `date`，其余列为虚构资产价格；`sample/allocations.csv` 第一列为 `ticker`，其余列为基础权重、情绪调整、宏观调整和最终权重。字段及样本含义见 [sample/README.md](sample/README.md)。

`python main.py --mode demo` 只读取模拟样本，用于检查收益计算、权重校验和报告输出。完整分析和滚动回测分别由 `python main.py --data-dir data` 和 `python main.py --mode backtest --data-dir data` 启动，需要上述四份原始输入。

## 结果目录

完整模式的结果表写入 `outputs/tables/`，图像写入 `outputs/figures/`；演示和回测结果分别写入 `outputs/demo/`、`outputs/backtest/`。这些结果由 `.gitignore` 排除，保留在本地。`data/raw/` 同样被 `.gitignore` 排除，适合存放不打算公开的本地数据。
