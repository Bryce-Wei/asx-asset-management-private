"""按自然月对齐价格与经济指标，以无截距回归生成宏观权重调整量。

经济指标按观测所属月份使用，没有按实际发布日期或历史修订版本还原
可交易信息集。因此结果用于样本内月度关系分析，不构成避免前视偏差的回测。
"""

import numpy as np
import pandas as pd

DEFAULT_SETTINGS = {
    "macro_start_date": "2019-01-31",
    "macro_recent_months": 3,
    "macro_exclude_last_month": True,
    "macro_features": (
        "Money Supply, M1 (%y/y)",
        "Consumer Sentiment Index",
        "Trade Balance (Mil. AUD)",
        "CPI, TD-MI Inflation Gauge Idx (%m/m)",
        "Job Advertisements (%m/m, sa)",
        "Retail Sales (%m/m)",
    ),
    "macro_drop_indicators": (
        "Housing Finance Commitments, Number (%m/m, sa)",
        "Sales of New Motor Vehicles (%y/y)",
        "Unemployment Rate (sa)",
        "Dwellings approved, Tot, (%m/m, sa)",
        "Dwellings approved, Private Sector, (%m/m, sa)",
        "Housing Finance Commitments, Value (%m/m, sa)",
    ),
    "macro_baseline_indicator": "Consumer Sentiment Index",
}


def analyze_macro(prices_df, indicators_df, settings):
    """估计各股票对宏观指标变化的关系，并按统一基准计算调整量。

    参数：
        prices_df：日期索引、股票代码列的价格 DataFrame；日期及列名唯一。
            月度价格取自然月内最后一条有效价格，随后计算简单环比收益。
        indicators_df：日期索引、指标名称列的 DataFrame；日期及列名唯一。
            必须覆盖 macro_features 配置中的全部指标。日期表示指标所属
            月份；指标原始单位由字段定义，如指数、百万澳元或百分比。
        settings：覆盖默认配置。macro_start_date 限定建模区间；
            macro_recent_months 控制近期均值窗口；macro_exclude_last_month
            决定是否舍弃最后一条完整月度变化记录；macro_features 指定
            回归解释变量；macro_baseline_indicator 指定公共基准指标。

    返回：
        adjustments：股票代码为索引的宏观权重调整量 Series。
        diagnostics：每只股票的观测数、非中心化 R²、条件数、预测值、
            基准值及调整量，便于检查拟合程度与数值稳定性。

    回归为 r = Xβ，不添加常数项，也不标准化各列。X 是指标数值的
    环比变化，即使某指标本来以百分比表示，仍计算其相对变化而非百分点差。
    当指标接近零或跨越正负值时，这种变换可能产生很大波动，需要结合
    经济含义解释。样本数量校验仅保证基本可估计性，不保证无共线性或
    样本外预测能力；无截距模型的 R² 也不能直接与含截距模型比较。
    """
    from statsmodels.api import OLS

    cfg = {**DEFAULT_SETTINGS, **settings}
    feature_names = list(cfg["macro_features"])
    missing = set(feature_names).difference(indicators_df.columns)
    if missing:
        raise ValueError(f"Economic data is missing indicators: {sorted(missing)}")
    if prices_df.empty or indicators_df.empty:
        raise ValueError("Price and economic data must be nonempty.")
    if not prices_df.index.is_unique or not indicators_df.index.is_unique:
        raise ValueError("Price and economic date indexes must be unique.")
    if not prices_df.columns.is_unique or not indicators_df.columns.is_unique:
        raise ValueError("Price and indicator labels must be unique.")
    prices = prices_df.copy()
    indicators = indicators_df.drop(columns=list(cfg["macro_drop_indicators"]), errors="ignore").copy()
    prices.index = pd.to_datetime(prices.index)
    indicators.index = pd.to_datetime(indicators.index)
    prices = prices.sort_index()
    indicators = indicators.sort_index()
    # 经济指标日期代表观测所属月份，并非实际发布日期。
    # 按自然月对齐价格与指标，避免周末月末的指标被错误移入次月。
    # 该对齐方式用于月度关系分析，不代表指标在当月最后交易日已可获取。
    monthly_prices = prices.resample(pd.offsets.MonthEnd()).last()
    monthly_indicators = indicators.resample(pd.offsets.MonthEnd()).last()
    # 先保留两个来源的完整月末日历，再向前填充指标；只使用此前所属月份
    # 的最近记录，不向后填充。但发布日期未被建模，仍不能视作实时可用数据。
    calendar = monthly_prices.index.union(monthly_indicators.index).sort_values()
    aligned = monthly_indicators.reindex(calendar).ffill().reindex(monthly_prices.index)
    monthly = monthly_prices.join(aligned)
    monthly = monthly.loc[cfg["macro_start_date"]:]
    # 统一计算价格和指标的相对变化，再删除含缺失/无穷值的整行，确保每只
    # 股票与各解释变量使用相同月份。此处完整行筛选也包括保留的其他指标列。
    changes = monthly.ffill().pct_change(fill_method=None)
    changes = changes.replace([np.inf, -np.inf], np.nan).dropna()
    # 这是显式的样本截尾规则：无论最后月份是否已完整结束，开启时均去掉
    # 最后一条有效变化记录，不依赖当前日期判断该月是否仍在更新。
    if cfg["macro_exclude_last_month"]:
        changes = changes.iloc[:-1]
    if len(changes) < len(feature_names) + 1:
        raise ValueError("Insufficient complete monthly observations for macro regression.")
    recent_count = int(cfg["macro_recent_months"])
    if recent_count <= 0:
        raise ValueError("macro_recent_months must be positive.")
    # 按月份倒序后取近期窗口；窗口大于可用记录数时，均值使用全部可用记录。
    feature_frame = changes[feature_names].sort_index(ascending=False)
    recent = feature_frame.iloc[:recent_count].mean()
    # 基准规则：最新一期指标变化 × (1 + 近期平均变化)，再取指定指标的
    # 结果作为所有股票共同扣减的基准。这是策略缩放规则，并非无风险利率。
    projected_features = feature_frame.iloc[0] * (1.0 + recent)
    baseline = float(projected_features[cfg["macro_baseline_indicator"]])
    # 回归预测输入使用“近期平均变化”本身，而非上面的 projected_features。
    # 分开保留两个量，便于核对预测与基准在公式中的不同作用。
    prediction_input = pd.DataFrame([recent], columns=feature_names)
    adjustments = {}
    diagnostics = []
    for ticker in prices.columns:
        # statsmodels 的 OLS 不会自动添加截距；传入哪些列就拟合哪些系数。
        fitted = OLS(changes[ticker], changes[feature_names]).fit()
        prediction = float(fitted.predict(prediction_input).iloc[0])
        # 股票预测收益减去统一指标基准后直接作为加法调整量，未做幅度约束。
        # 这一经验规则可能产生较大或负的调整，最终风险由配置结果检查反映。
        adjustment = prediction - baseline
        if not np.isfinite(adjustment):
            raise ValueError(f"Non-finite macro adjustment for {ticker}.")
        adjustments[ticker] = adjustment
        diagnostics.append({
            "ticker": ticker, "observations": int(fitted.nobs),
            "r_squared_uncentered": float(fitted.rsquared),
            "condition_number": float(fitted.condition_number),
            "predicted_return": prediction, "baseline": baseline,
            "macro_adjustment": adjustment,
        })
    return {
        "adjustments": pd.Series(adjustments, name="macro_adjustment").rename_axis("ticker"),
        "diagnostics": pd.DataFrame(diagnostics).set_index("ticker"),
    }
