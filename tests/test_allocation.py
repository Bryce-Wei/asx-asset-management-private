"""检查权重合并是否保留资产标签，并识别无效输入。"""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from asset_management.allocation import combine_weights, validate_weights


class AllocationTests(unittest.TestCase):
    """围绕资产对齐、归一化前提和空头保留规则验证组合配置。"""

    def setUp(self):
        """构造合计为 1 的固定权重及零调整，作为各边界场景的共同基准。"""
        self.base = pd.Series([0.4, 0.35, 0.25], index=["SYN_A", "SYN_B", "SYN_C"])
        self.zero = pd.Series(0.0, index=self.base.index)

    def test_adjustments_align_by_asset_not_row_position(self):
        """打乱两类调整的行序后结果应完全相同，防止把调整加到错误股票上。"""
        adjustments = pd.Series([0.02, -0.01, 0.0], index=self.base.index)
        expected = combine_weights(self.base, adjustments, self.zero, {})
        actual = combine_weights(
            self.base, adjustments.reindex(["SYN_C", "SYN_A", "SYN_B"]),
            self.zero.reindex(["SYN_B", "SYN_C", "SYN_A"]), {},
        )
        assert_frame_equal(actual, expected)
        self.assertEqual(list(actual.index), list(self.base.index))
        self.assertTrue(validate_weights(actual)["sum_to_one"])

    def test_fixed_demo_weights_remain_valid(self):
        """零调整不应改变固定演示配置，归一化后仍保持原来的非负比例。"""
        actual = combine_weights(self.base, self.zero, self.zero, {})
        np.testing.assert_allclose(actual["final_weight"], [0.4, 0.35, 0.25])
        self.assertTrue(validate_weights(actual)["nonnegative"])

    def test_missing_extra_and_duplicate_assets_are_rejected(self):
        """分别覆盖缺少代码、混入陌生代码和重复代码，禁止静默补零或截断。"""
        invalid = [
            self.zero.drop("SYN_C"),
            pd.Series([0, 0, 0], index=["SYN_A", "SYN_B", "UNKNOWN"]),
            pd.Series([0, 0, 0], index=["SYN_A", "SYN_A", "SYN_C"]),
        ]
        for adjustments in invalid:
            with self.subTest(labels=list(adjustments.index)):
                with self.assertRaises(ValueError):
                    combine_weights(self.base, adjustments, self.zero, {})

    def test_nonfinite_values_and_zero_total_are_rejected(self):
        """非有限调整不能进入合并，总额被完全抵消时也不能继续做归一化。"""
        for invalid in [np.nan, np.inf, -np.inf]:
            adjustments = self.zero.copy()
            adjustments.iloc[0] = invalid
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    combine_weights(self.base, adjustments, self.zero, {})
        with self.assertRaises(ValueError):
            combine_weights(self.base, -self.base, self.zero, {})

    def test_negative_weights_are_reported_without_silent_clipping(self):
        """加法调整可产生空头；检查应报告负权重，同时保留合计约束。"""
        adjustment = pd.Series([-0.5, 0.5, 0.0], index=self.base.index)
        actual = combine_weights(self.base, adjustment, self.zero, {})
        self.assertLess(actual.loc["SYN_A", "final_weight"], 0)
        checks = validate_weights(actual)
        self.assertTrue(checks["has_short_positions"])
        self.assertFalse(checks["nonnegative"])
        self.assertTrue(checks["sum_to_one"])


if __name__ == "__main__":
    unittest.main()
