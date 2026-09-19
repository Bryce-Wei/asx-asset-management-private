"""使用 VADER 分析英文新闻标题，并将近期情绪变化映射为权重调整量。

这里采用词典情绪模型，没有训练专用金融语言模型。标题的情绪分值不等于
股价涨跌概率；讽刺、金融术语以及缺少正文上下文均可能影响识别结果。
"""

import numpy as np
import pandas as pd


def analyze_sentiment(news_df, tickers, settings):
    """汇总每只股票的标题情绪，并比较近期窗口与全部样本。

    参数：
        news_df：至少含 Equity（股票代码）和 Headline（文本标题）列的
            DataFrame，新闻记录顺序会被保留。本函数不读取日期或排序，
            因而调用前应将每只股票的新闻按时间从新到旧排列。
        tickers：需要输出的唯一股票代码序列；每只股票都必须有新闻记录。
            输出表和调整量按该序列排列，其他股票新闻不进入输出汇总。
        settings：sentiment_weights 为 pos、neg、neu 三类分数的系数；
            sentiment_recent_articles 指定近期窗口篇数；
            sentiment_adjustment_divisor 控制情绪差转换为权重调整量的
            缩放幅度；sentiment_adjustment_decimals 控制调整量精度。

    返回：
        adjustments：股票代码为索引的权重增减量 Series，以小数表示。
        scores：股票代码为索引的 DataFrame，包含 neg、neu、pos、
            compound 的全部标题平均分，以及 recent_score、history_score。

    每篇标题的加权分为 pos×正向系数 + neg×负向系数 + neu×中性系数。
    调整量为（近期平均分 - 全部平均分）/缩放除数；全部样本包含近期
    窗口，两个均值并非独立样本。除数绝对值越大，调整幅度越小；负除数
    会反转方向。权重系数是策略参数，不由模型在此处学习。
    """
    required = {"Equity", "Headline"}
    missing = required.difference(news_df.columns)
    if missing:
        raise ValueError(f"News data is missing columns: {sorted(missing)}")
    if news_df.empty or news_df[["Equity", "Headline"]].isna().any().any():
        raise ValueError("News records require nonempty Equity and Headline fields.")
    if not news_df["Headline"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every headline must be text.")
    tickers = pd.Index(tickers, name="ticker")
    if not tickers.is_unique:
        raise ValueError("Ticker labels must be unique.")
    missing_tickers = tickers.difference(news_df["Equity"].unique())
    if len(missing_tickers):
        raise ValueError(f"No news records for tickers: {missing_tickers.tolist()}")
    # 延迟加载词典模型；本地缺少词典时给出安装说明，不在运行中自动联网下载。
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise RuntimeError(
            "NLTK vader_lexicon is missing. Install it explicitly with "
            "`python -m nltk.downloader vader_lexicon` before a local full run. "
            "This program does not download resources automatically."
        ) from exc
    # VADER 的 pos、neg、neu 是情绪组成分数，compound 是 [-1, 1] 综合分。
    # 策略用前三项计算加权分，compound 仅用于结果汇总，不参与调整量公式。
    polarity = pd.DataFrame(
        [analyzer.polarity_scores(text) for text in news_df["Headline"]],
        index=news_df.index,
    )
    scored = pd.concat([news_df[["Equity"]], polarity], axis=1)
    recent_count = int(settings["sentiment_recent_articles"])
    divisor = float(settings["sentiment_adjustment_divisor"])
    if recent_count <= 0 or not np.isfinite(divisor) or divisor == 0:
        raise ValueError("Recent article count must be positive and divisor finite/nonzero.")
    weights = settings["sentiment_weights"]
    summaries = []
    adjustments = []
    for ticker in tickers:
        part = scored.loc[scored["Equity"] == ticker]
        weighted = sum(part[key] * float(weights[key]) for key in ("pos", "neg", "neu"))
        # “近期”取该股票输入记录的前若干篇，不能在未排序数据上解释为最近新闻。
        # 新闻不足窗口篇数时使用全部记录，此时近期均值与历史均值相同，调整为零。
        recent_score = float(weighted.iloc[:recent_count].mean())
        history_score = float(weighted.mean())
        # 输出是加到基础配置上的权重差值，最终组合的预算约束由配置模块处理。
        change = np.round(
            (recent_score - history_score) / divisor,
            int(settings["sentiment_adjustment_decimals"]),
        )
        row = part[["neg", "neu", "pos", "compound"]].mean().to_dict()
        row.update(recent_score=recent_score, history_score=history_score)
        summaries.append(row)
        adjustments.append(change)
    return {
        "adjustments": pd.Series(adjustments, index=tickers, name="sentiment_adjustment"),
        "scores": pd.DataFrame(summaries, index=tickers),
    }
