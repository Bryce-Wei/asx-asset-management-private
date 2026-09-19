"""检查蒙特卡洛模拟和新闻情绪调整的关键数值规则。"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from config import SETTINGS
from asset_management.portfolio import simulate_portfolios
from asset_management.sentiment import analyze_sentiment


class ModelTests(unittest.TestCase):
    """用确定性输入隔离抽样统计量与新闻汇总规则，不依赖市场拟合结果。"""

    def test_simulation_reproducibility_and_identical_asset_statistics(self):
        """完全相同的两列收益使任何权重组合都有相同、可手算的统计量。

        配置 12 个年化周期后收益应为 0.08，方差应为 0.0028；重复
        调用结果一致，且调用前后的进程全局随机状态应保持不变。
        """
        returns = pd.DataFrame({"A": [0.01, -0.01, 0.02], "B": [0.01, -0.01, 0.02]})
        settings = {**SETTINGS, "monte_carlo_samples": 7, "monte_carlo_annualization": 12}
        before = np.random.get_state()
        actual = simulate_portfolios(returns, settings)
        after = np.random.get_state()
        assert_frame_equal(actual, simulate_portfolios(returns, settings))
        self.assertEqual(len(actual), 7)
        np.testing.assert_allclose(actual["return"], 0.08)
        np.testing.assert_allclose(actual["volatility"], np.sqrt(0.0028))
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_sentiment_preserves_input_order_and_requested_asset_alignment(self):
        """以输入中的首条新闻作为近期窗口，按指定股票顺序输出调整量。

        使用人为指定的三类词典分数，将词典行为与汇总算法分开验证；
        股票 B、A 的输出顺序与新闻首次出现顺序相反，以检查代码对齐。
        """
        news = pd.DataFrame({
            "Equity": ["A", "B", "A", "B"],
            "Headline": ["positive", "neutral", "negative", "positive"],
        })
        polarities = {
            "positive": {"pos": 1.0, "neg": 0.0, "neu": 0.0, "compound": 0.8},
            "negative": {"pos": 0.0, "neg": 1.0, "neu": 0.0, "compound": -0.8},
            "neutral": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
        }
        settings = {**SETTINGS, "sentiment_recent_articles": 1}
        # 固定词典评分，单独验证汇总规则；测试运行不需要下载词典。
        with patch("nltk.sentiment.vader.SentimentIntensityAnalyzer") as analyzer:
            analyzer.return_value.polarity_scores.side_effect = polarities.__getitem__
            actual = analyze_sentiment(news, ["B", "A"], settings)
        self.assertEqual(list(actual["adjustments"].index), ["B", "A"])
        np.testing.assert_allclose(actual["adjustments"], [-0.045, 0.1])
        np.testing.assert_allclose(actual["scores"]["history_score"], [0.55, 0.0])
        np.testing.assert_allclose(actual["scores"]["recent_score"], [0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
