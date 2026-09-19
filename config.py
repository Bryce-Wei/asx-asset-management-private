"""集中配置数据路径、风险等级、蒙特卡洛及新闻情绪分析参数。

路径以本文件所在目录为基准，切换终端工作目录不会改变默认输入位置。
权重和收益率统一使用小数比例，例如 0.01 表示 1%，而不是数值 1。
本文件保存跨模块设置；优化器、客户分群及宏观回归的专用默认值
分别保留在对应模块的默认配置中，调用时可通过设置字典覆盖。
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent  # 项目根目录，使用解析后的绝对路径。
RAW_DIR = PROJECT_DIR / "data" / "raw"  # 四份本地原始输入文件，读取时不修改。
SAMPLE_DIR = PROJECT_DIR / "data" / "sample"  # 模拟价格及固定权重，供演示模式使用。
OUTPUT_DIR = PROJECT_DIR / "outputs"  # 表格、图表和运行摘要的默认输出根目录。

# 各分析模块共用此配置；其他默认参数在对应模块中定义。
SETTINGS = {
    # 组合优化的年化周期数：日度均值与协方差均乘以 252，波动率取平方根。
    "annualization": 252,
    # 本次采用的风险档位；1、2、3 对应模型内逐步降低的波动率惩罚。
    # 此值选择基础组合，与客户分群生成的簇编号是不同概念。
    "selected_risk_level": 2,
    # 固定独立随机状态，使同一收益数据与参数下的抽样结果可以复现。
    "monte_carlo_seed": 5545,
    # 随机组合的数量，单位为组；增加数量会增加计算及结果表的规模。
    "monte_carlo_samples": 5545,
    # 蒙特卡洛单独使用的年化周期数，应与输入收益频率和优化口径一致。
    "monte_carlo_annualization": 252,
    # 对 VADER 的正面、负面、中性分值作线性加权，得到每条标题的策略分值。
    # 这些系数不作用于 compound 综合分；compound 另行保留用于报告展示。
    "sentiment_weights": {"pos": 1.0, "neg": -1.0, "neu": 0.1},
    # 每只股票取输入顺序中的前 5 条作为近期窗口，少于 5 条时使用全部。
    # 新闻加载器不按日期排序，因而输入文件应事先按需要的时间顺序排列。
    "sentiment_recent_articles": 5,
    # （近期均分 - 全部新闻均分）除以此值后作为加法权重调整量。
    # 例如分差 0.1 除以 10 得 0.01，即归一化之前的 1 个百分点调整。
    "sentiment_adjustment_divisor": 10.0,
    # 新闻调整量保留的小数位数；最终权重的取整精度由配置合并模块决定。
    "sentiment_adjustment_decimals": 4,
}
