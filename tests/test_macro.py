"""检查月末落在非交易日时，经济指标与股价仍按同月对齐。"""

import unittest

import numpy as np
import pandas as pd

from asset_management.macro import DEFAULT_SETTINGS, analyze_macro


class MacroAlignmentTests(unittest.TestCase):
    """构造已知系数的月度关系，防止月末映射错误使回归意外跨月。"""

    def test_weekend_month_ends_preserve_same_month_relationship(self):
        """指标用自然月末、价格用交易月末，仍应恢复相同月份的六因子关系。

        人工收益由固定线性系数生成，因此期数、预测值及调整量都有
        独立可计算的预期值，能捕捉周末月末被错移至下个月的问题。
        """
        # 2019 年 3 月和 6 月的月末为周日。
        # 指标采用自然月末日期，股价采用当月最后一个工作日。
        month_ends = pd.date_range("2019-01-31", periods=18, freq=pd.offsets.MonthEnd())
        trading_ends = pd.date_range("2019-01-31", periods=18, freq=pd.offsets.BMonthEnd())
        month_number = np.arange(18, dtype=float)[:, None]
        feature_number = np.arange(1, 7, dtype=float)[None, :]
        indicator_changes = 0.01 + 0.005 * np.sin(month_number * feature_number)
        indicators = pd.DataFrame(
            100 * np.cumprod(1 + indicator_changes, axis=0),
            index=month_ends,
            columns=DEFAULT_SETTINGS["macro_features"],
        )
        coefficients = np.array([0.5, -0.2, 0.1, 0.3, 0.05, 0.07])
        price_returns = indicator_changes @ coefficients
        prices = pd.DataFrame(
            {"AAA": 100 * np.cumprod(1 + price_returns)}, index=trading_ends,
        )
        # 计算变化率后排除首期缺失值与末期观测。
        # 即使月末落在周末，回归也应恢复预设的同月线性关系。
        training_changes = indicator_changes[1:-1]
        recent = training_changes[-3:].mean(axis=0)
        expected_prediction = float(recent @ coefficients)
        expected_baseline = float(training_changes[-1, 1] * (1 + recent[1]))
        result = analyze_macro(prices, indicators, {})
        diagnostics = result["diagnostics"].loc["AAA"]
        self.assertEqual(diagnostics["observations"], 16)
        self.assertAlmostEqual(diagnostics["predicted_return"], expected_prediction, places=10)
        self.assertAlmostEqual(
            result["adjustments"]["AAA"], expected_prediction - expected_baseline, places=10,
        )


if __name__ == "__main__":
    unittest.main()
