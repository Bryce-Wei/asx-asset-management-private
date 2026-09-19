"""提供主分析流程和预测实验共用的价格及收益率转换。

输入统一为日期索引、资产代码列的价格矩阵；输出保留资产列顺序，
收益率采用小数比例。本模块不年化、不插补缺失价格，也不转换币种，
以便组合模型和宏观模型分别使用其需要的频率与收益定义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validated_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """检查价格矩阵，并返回按日期升序排列的浮点副本。

    日期和资产标签必须唯一，才能让滞后、重采样和资产对齐具有确定
    含义。允许缺失价格传播到收益计算，但拒绝无限值、零及负价格；
    后三者无法形成有效的价格比值或对数收益。不更改输入表格。
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a nonempty DataFrame with dates and asset columns.")
    if prices.columns.has_duplicates:
        raise ValueError("Price columns must have unique asset labels.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must use a DatetimeIndex.")
    if prices.index.has_duplicates or prices.index.isna().any():
        raise ValueError("Price dates must be unique and nonmissing.")
    try:
        frame = prices.astype(float).sort_index()
    except (ValueError, TypeError) as exc:
        raise ValueError("Prices must contain numeric values or missing values.") from exc
    if np.isinf(frame.to_numpy()).any() or (frame <= 0).any().any():
        raise ValueError("Observed prices must be finite and strictly positive.")
    return frame


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """按相邻输入日期计算对数收益率，并对齐所有资产。

    每列按 ln(P_t / P_{t-1}) 计算，行索引代表区间结束日期，输出单位
    为每个相邻观测区间的小数对数收益；不在这里进行年化。

    在滞后计算前保留缺失价格，使缺失日期以及后一观测日受影响的
    收益率均记为缺失，再只保留所有资产收益率完整的共同日期。
    共同样本让后续均值与协方差估计使用一致的观测范围。
    不补充输入中缺少的整个日期，也不额外应用交易日历；若源表直接
    省略某日，仍按现有相邻行计算，输入频率应由数据提供方先确认。
    """
    frame = _validated_prices(prices)
    # 先求比值再统一删行；提前删去缺失价格会把跨缺口变化误当成单期收益。
    returns = np.log(frame / frame.shift(1)).dropna(how="any")
    if returns.empty:
        raise ValueError("Prices contain no adjacent, complete observation pairs for all assets.")
    return returns


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """使用每月最后一个有效价格计算简单收益率。

    先将每列按自然月末重采样，再计算 P_m / P_{m-1} - 1；输出日期
    为自然月末，小数 0.01 表示当月增长 1%，不是对数收益。
    ``last`` 对每只资产取该月最后一个非缺失价格，所以不同资产的
    实际末次观测日可能不同，输入数据的覆盖情况仍需自行确认。

    重采样保留完全空缺的月份，滞后后再删除任一资产收益缺失的行，
    避免将跨月累计变化误计为单月收益。结果不跨月前向填充。
    """
    frame = _validated_prices(prices)
    month_end_prices = frame.resample(pd.offsets.MonthEnd()).last()
    returns = (month_end_prices / month_end_prices.shift(1) - 1.0).dropna(how="any")
    if returns.empty:
        raise ValueError("Prices contain no adjacent months with complete prices for all assets.")
    return returns
