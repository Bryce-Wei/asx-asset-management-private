"""检查收益率计算的数值结果、日期对齐和缺失值处理。"""

import unittest

import numpy as np
import pandas as pd

from asset_management.features import log_returns, monthly_returns


class ReturnTests(unittest.TestCase):
    """用可手算的价格验证收益定义、日期顺序及缺失处理边界。"""

    def test_chronological_returns_preserve_asset_labels(self):
        """日期故意乱序、资产列故意不按字母排序，检查时间排序不会错换资产。"""
        prices = pd.DataFrame(
            {"B": [121, 100, 110], "A": [81, 100, 90]},
            index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
        )
        actual = log_returns(prices)
        self.assertEqual(list(actual.columns), ["B", "A"])
        self.assertEqual(list(actual.index), list(pd.to_datetime(["2020-01-02", "2020-01-03"])))
        np.testing.assert_allclose(actual.to_numpy(), [[np.log(1.1), np.log(0.9)]] * 2)

    def test_missing_price_does_not_bridge_or_fill_returns(self):
        """缺失价格应同时影响本期和下一期收益，只保留最后共同有效的区间。"""
        prices = pd.DataFrame(
            {"A": [100, np.nan, 121, 133.1], "B": [50, 55, 60.5, 66.55]},
            index=pd.date_range("2020-01-01", periods=4),
        )
        actual = log_returns(prices)
        self.assertEqual(list(actual.index), [pd.Timestamp("2020-01-04")])
        np.testing.assert_allclose(actual.iloc[0].to_numpy(), [np.log(1.1)] * 2)

    def test_nonpositive_prices_are_rejected(self):
        """零、负数和无限价格都不能形成有效的对数收益。"""
        for value in (0, -1, np.inf):
            prices = pd.DataFrame({"A": [100, value]}, index=pd.date_range("2020-01-01", periods=2))
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                log_returns(prices)

    def test_duplicate_dates_are_rejected(self):
        """同日重复记录会使相邻区间不明确，因此必须在计算前被拒绝。"""
        prices = pd.DataFrame({"A": [100, 110]}, index=pd.to_datetime(["2020-01-01", "2020-01-01"]))
        with self.assertRaisesRegex(ValueError, "unique"):
            log_returns(prices)

    def test_monthly_returns_are_simple_and_keep_month_gaps(self):
        """三月完全缺失时不能把二月至四月的累计涨幅当作四月单月收益。

        二月和五月均应为简单收益 10%，日期标签统一到自然月末，
        从而同时区分对数收益算法和错误的跨缺口计算。
        """
        prices = pd.DataFrame(
            {"A": [90, 100, 110, 121, 133.1]},
            index=pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-28", "2020-04-30", "2020-05-29"]),
        )
        actual = monthly_returns(prices)
        self.assertEqual(list(actual.index), [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-05-31")])
        np.testing.assert_allclose(actual["A"].to_numpy(), [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
