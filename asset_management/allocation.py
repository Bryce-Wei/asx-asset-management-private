"""按股票标签合并基础配置、情绪调整和宏观调整，并报告权重约束状态。

合并采用加法叠加后统一归一化；此处不重新优化、不截断负数，也不实施
单一持仓上限。因此基础组合禁止做空，不代表调整后的最终组合仍禁止做空。
"""

import numpy as np
import pandas as pd

DEFAULT_SETTINGS = {"final_weight_decimals": 4}


def combine_weights(base, sentiment_adjustments, macro_adjustments, settings):
    """合并三个配置分量，返回可追溯的最终权重明细表。

    参数：
        base：非空 Series，索引为唯一股票代码，值为基础权重的小数表示。
        sentiment_adjustments、macro_adjustments：同样以唯一股票代码
            为索引的 Series，值为加法权重调整量；必须与 base 覆盖完全
            相同的股票，可以具有不同顺序。0.01 表示增加一个百分点的
            未归一化配置量，不表示把现有权重乘以 1.01。
        settings：final_weight_decimals 指定最终权重保留的小数位数。

    返回：
        以基础权重股票顺序为索引的 DataFrame，含 base_weight、
        sentiment_adjustment、macro_adjustment 和 final_weight。
        attrs 保存取整精度，供 validate_weights 计算合计容差。

    公式为 a_i = 基础权重_i + 情绪调整_i + 宏观调整_i，随后
    w_i = a_i / Σa_i，最后按指定精度取整。归一化只处理总预算，
    不保证权重非负；分母为负时还会反转全部符号。分母很小但非零时
    权重可能被显著放大，函数不会自动更改策略规则，需检查返回结果。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    if not isinstance(base, pd.Series) or base.empty or not base.index.is_unique:
        raise ValueError("Base weights must be a nonempty Series with unique tickers.")
    pieces = {"base_weight": base}
    for name, adjustment in (
        ("sentiment_adjustment", sentiment_adjustments),
        ("macro_adjustment", macro_adjustments),
    ):
        if not isinstance(adjustment, pd.Series) or not adjustment.index.is_unique:
            raise ValueError(f"{name} must have unique ticker labels.")
        # 股票缺失或多出都直接报错，避免静默填零或丢弃股票改变配置含义。
        if set(adjustment.index) != set(base.index):
            raise ValueError(f"{name} tickers must exactly match the base weights.")
        # 按股票标签重新排列，确保相同位置相加的是同一只股票，而非依赖行序。
        pieces[name] = adjustment.reindex(base.index)
    frame = pd.DataFrame(pieces).astype(float).rename_axis("ticker")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Weights and adjustments must contain only finite values.")
    # 每列均为小数尺度的权重或调整量，先叠加再按所有股票的合计缩放。
    combined = frame.sum(axis=1)
    denominator = float(combined.sum())
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("Combined weights have a zero or non-finite sum; cannot normalize.")
    # 逐只股票取整会使合计略偏离 1；不把差额强制加到某一只股票上。
    frame["final_weight"] = (combined / denominator).round(int(cfg["final_weight_decimals"]))
    if not np.isfinite(frame["final_weight"].to_numpy()).all():
        raise ValueError("Normalization produced non-finite weights.")
    frame.attrs["final_weight_decimals"] = int(cfg["final_weight_decimals"])
    return frame


def validate_weights(frame):
    """报告最终权重的有限性、预算约束与负持仓状态，不修改权重。

    参数：
        frame：含 final_weight 数值列的 DataFrame；通常直接使用
            combine_weights 的结果。若 attrs 没有 final_weight_decimals，
            按模块默认的四位小数精度确定容差。

    返回：
        字典包含 finite（非空且全部有限）、sum（权重合计）、
        sum_to_one（是否在取整容差内合计为 1）、sum_tolerance、
        nonnegative（全部非负）及 has_short_positions（是否存在负权重）。

    每只股票四舍五入最多带来约半个末位单位的误差，因此总容差采用
    持仓数 × 0.5 × 10^(-小数位数)，再加入浮点机器精度。该检查只
    评估输出数值约束，不评估换手成本、集中度、杠杆上限或投资适配性。
    """
    if "final_weight" not in frame:
        raise ValueError("A final_weight column is required.")
    weights = frame["final_weight"].to_numpy(dtype=float)
    decimals = int(frame.attrs.get("final_weight_decimals", DEFAULT_SETTINGS["final_weight_decimals"]))
    # 根据取整精度而非固定阈值判断预算误差，避免将正常舍入误差判为失败。
    tolerance = len(weights) * 0.5 * 10.0 ** (-decimals) + np.finfo(float).eps
    finite = bool(len(weights) and np.isfinite(weights).all())
    total = float(weights.sum())
    return {
        "finite": finite,
        "sum": total,
        "sum_to_one": bool(finite and abs(total - 1.0) <= tolerance),
        "sum_tolerance": float(tolerance),
        "nonnegative": bool(finite and (weights >= 0).all()),
        "has_short_positions": bool((weights < 0).any()),
    }
