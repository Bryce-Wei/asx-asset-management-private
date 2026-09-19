# 模拟演示数据

我在此目录保留一组模拟样本，便于在没有原始输入文件时检查程序流程。样本不包含真实股票、客户或新闻记录。

- `prices.csv`：20 个示例日期、3 个虚构资产 `SYN_A`、`SYN_B`、`SYN_C` 的正价格；字段为 `date` 和各资产价格列，用于展示读取、收益计算和绘图。
- `allocations.csv`：字段为 `ticker`、`base_weight`、`sentiment_adjustment`、`macro_adjustment`、`final_weight`；基础与最终权重均为人工固定的 40%、35%、25%，情绪与宏观调整均为零。

这些权重不是蒙特卡洛模拟、优化器、NLP 或宏观模型的输出。日期只用于展示时间索引，不代表真实市场的完整交易日历，样本不能用于评估投资表现或模型有效性。

在项目根目录运行 `python main.py --mode demo` 即可使用这组样本，结果写入本地 `outputs/demo/`。完整分析使用 `data/raw/` 中的四份输入文件，不使用本目录的示例权重。
