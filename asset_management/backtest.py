"""按月滚动回测：每个调仓日只使用当时已经可以获得的数据重新计算权重。

信息集规则均以信号日 s（每月最后一个交易日）为准：

- 价格：只使用日期不晚于 s 的收盘价；组合优化只取最近
  ``backtest_lookback_days`` 个收益区间。
- 新闻：只使用时间戳不晚于 s 当日 ``backtest_news_cutoff``（默认 16:00，
  即 ASX 收盘）的标题，并在每只股票内按时间由新到旧排列。
- 宏观指标：所属月份不晚于 s 所在月份减去 ``backtest_macro_release_lag_months``，
  以统一的保守滞后近似实际发布日期；宏观回归使用的价格也截止到该月。

信号在 s 收盘后形成，于 ``backtest_execution_lag_days`` 个交易日后的收盘成交，
此后才获得新权重的收益；成交日当天的涨跌仍归属原持仓。

蒙特卡洛和客户分群不影响权重，回测中不运行。股票池固定为输入中的股票，
价格为 PX_LAST（不含分红），指标取文件中的最终修订值；这些偏差无法由
截取日期消除，解读结果时需一并考虑。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management import allocation, macro, portfolio, sentiment
from asset_management.features import log_returns

DEFAULT_SETTINGS = {
    # 组合优化的滚动窗口，单位为交易日收益个数；None 表示使用 s 之前的全部历史。
    "backtest_lookback_days": 504,
    # 窗口为 None 时，首个信号日之前至少需要的收益个数。
    "backtest_min_history_days": 252,
    # 信号日之后第几个交易日收盘成交；0 表示假设能以信号日收盘价成交。
    "backtest_execution_lag_days": 1,
    # 单边交易成本（基点），按换手 Σ|目标权重 - 当前权重| 计收。
    "backtest_cost_bps": 10.0,
    # 年化无风险利率（小数），仅用于夏普比率。
    "backtest_risk_free_rate": 0.0,
    # 宏观发布滞后：所属月份 m 的指标视为在 m + 滞后月数 的月末才可用。
    "backtest_macro_release_lag_months": 2,
    # 宏观回归使用的最近月份数，替代全样本模式中固定的 macro_start_date。
    "backtest_macro_window_months": 36,
    # 新闻可用截止时刻（当地时间，HH:MM）；晚于信号日该时刻的标题不进入当期。
    "backtest_news_cutoff": "16:00",
    # 带时区的新闻时间先换算到该时区，再去掉时区与交易日期比较。
    "backtest_news_timezone": "Australia/Sydney",
    # 新闻时间格式；None 时先按 ISO 8601 解析，失败后再按日在前的澳洲格式解析。
    "news_datetime_format": None,
    # 可选的回测起止日期，按信号日筛选；None 表示使用全部可用区间。
    "backtest_start": None,
    "backtest_end": None,
}

# 各策略版本启用的调整分量，用于分别衡量情绪和宏观调整的贡献。
VARIANTS = {
    "base": (),
    "base_sentiment": ("sentiment",),
    "base_macro": ("macro",),
    "full": ("sentiment", "macro"),
}
# 不依赖任何模型的等权基准，每期同样按成交规则调仓并计入成本。
BENCHMARK = "equal_weight"


def prepare_news(news_df, settings):
    """解析新闻时间戳，返回带 timestamp 列、按时间由新到旧排列的副本。

    时间戳是判断“当时是否已发布”的唯一依据，因此无法解析的记录直接报错，
    不静默丢弃。先按 ISO 8601 解析：pandas 在 dayfirst 模式下会把
    2020-03-04 读成 4 月 3 日，所以只有 ISO 解析失败时才按日在前解析；
    格式不统一或存在歧义时应在 news_datetime_format 中显式指定。
    带时区的时间换算到 backtest_news_timezone，不带时区的视为当地时间。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    if "Date/Time" not in news_df:
        raise ValueError("News data requires a Date/Time column for point-in-time filtering.")
    raw = news_df["Date/Time"].astype(str).str.strip()
    try:
        if cfg["news_datetime_format"]:
            stamps = pd.to_datetime(raw, format=cfg["news_datetime_format"], errors="raise")
        else:
            try:
                stamps = pd.to_datetime(raw, format="ISO8601", errors="raise")
            except (ValueError, TypeError):
                stamps = pd.to_datetime(raw, dayfirst=True, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Cannot parse news Date/Time values; set news_datetime_format explicitly."
        ) from exc
    if not pd.api.types.is_datetime64_any_dtype(stamps):
        raise ValueError("News Date/Time values mix time zones or formats; set news_datetime_format.")
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(cfg["backtest_news_timezone"]).dt.tz_localize(None)
    if stamps.isna().any():
        raise ValueError("Every news record needs a parseable Date/Time value.")
    frame = news_df.copy()
    frame["timestamp"] = stamps
    # 稳定排序保留同一时刻记录的原始相对顺序。
    return frame.sort_values("timestamp", ascending=False, kind="mergesort").reset_index(drop=True)


def rebalance_schedule(price_index, settings):
    """列出每月的信号日和成交日。

    信号日为每个自然月最后一个有价格数据的日期；数据末月若未到当月最后
    一个工作日，视为月份尚未结束，不生成信号。信号日之前须有足够的收益
    历史；成交日为其后第 backtest_execution_lag_days 个交易日，超出数据
    末尾的信号不再交易。返回含 signal_date、execution_date 的 DataFrame。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    index = pd.DatetimeIndex(price_index)
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Price dates must be nonempty, unique and sorted.")
    lag = int(cfg["backtest_execution_lag_days"])
    if lag < 0:
        raise ValueError("backtest_execution_lag_days must be nonnegative.")
    history = int(cfg["backtest_lookback_days"] or cfg["backtest_min_history_days"])
    if history < 2:
        raise ValueError("The optimizer needs at least two historical returns.")
    positions = pd.Series(np.arange(len(index)), index=index)
    month_last = positions.groupby(index.to_period("M")).max()
    if index[-1] < index[-1] + pd.offsets.BMonthEnd(0):
        month_last = month_last.iloc[:-1]
    start = pd.Timestamp(cfg["backtest_start"]) if cfg["backtest_start"] else None
    end = pd.Timestamp(cfg["backtest_end"]) if cfg["backtest_end"] else None
    rows = []
    for position in month_last:
        signal = index[position]
        # position 个收益需要 position + 1 个价格，因此位置即可用的收益个数。
        if position < history or position + lag >= len(index):
            continue
        if (start is not None and signal < start) or (end is not None and signal > end):
            continue
        rows.append({"signal_date": signal, "execution_date": index[position + lag]})
    return pd.DataFrame(rows, columns=["signal_date", "execution_date"])


def point_in_time_inputs(data, signal_date, settings):
    """截取信号日当时已可获得的价格、新闻和宏观指标。

    data 含 prices、news（须经 prepare_news）和 indicators。返回的表格
    均为副本，另附 news_cutoff 与 macro_cutoff 两个截止时点，便于逐期审计
    每个模型实际看到的数据范围。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    signal = pd.Timestamp(signal_date)
    hour, minute = (int(part) for part in str(cfg["backtest_news_cutoff"]).split(":"))
    news_cutoff = signal.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    lag = int(cfg["backtest_macro_release_lag_months"])
    if lag < 0:
        raise ValueError("backtest_macro_release_lag_months must be nonnegative.")
    macro_cutoff = (signal.to_period("M") - lag).end_time.normalize()
    news = data["news"]
    return {
        "prices": data["prices"].loc[:signal].copy(),
        "news": news.loc[news["timestamp"] <= news_cutoff].reset_index(drop=True),
        "indicators": data["indicators"].loc[:macro_cutoff].copy(),
        "news_cutoff": news_cutoff,
        "macro_cutoff": macro_cutoff,
    }


def compute_signals(pit, settings):
    """用单期信息集计算基础权重、两类调整量和各策略版本的目标权重。

    返回 targets（版本名 → 目标权重）、failures（本期未给出目标的版本及
    原因）、components（按股票列出三个分量）和 status（各分量状态）。
    基础组合优化失败时，所有依赖优化器的版本本期沿用原持仓；情绪或宏观
    分量因数据不足无法计算时按零调整处理并记录原因，不中断回测。缺少
    VADER 词典等环境问题仍直接报错，避免整段回测在无提示的情况下失效。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    prices = pit["prices"]
    tickers = pd.Index(prices.columns, name="ticker")
    zero = pd.Series(0.0, index=tickers)
    targets = {BENCHMARK: pd.Series(1.0 / len(tickers), index=tickers, name="weight")}
    status = {"base": "ok", "sentiment": "ok", "macro": "ok"}
    lookback = cfg["backtest_lookback_days"]
    window = prices.iloc[-(int(lookback) + 1):] if lookback else prices
    try:
        # 回测不使用有效前沿，点数设为 0，避免每期额外求解数十次。
        base = portfolio.analyze_portfolios(
            log_returns(window), {**cfg, "frontier_points": 0},
        )["base_weights"]
    except (ValueError, RuntimeError) as exc:
        status["base"] = f"failed: {exc}"
        components = pd.DataFrame(
            np.nan, index=tickers,
            columns=["base_weight", "sentiment_adjustment", "macro_adjustment"],
        )
        failures = {name: "base optimization failed" for name in VARIANTS}
        return {"targets": targets, "failures": failures, "components": components, "status": status}

    sentiment_adjustment = zero.copy()
    news = pit["news"]
    published = set(news["Equity"])
    covered = [ticker for ticker in tickers if ticker in published]
    if not covered:
        status["sentiment"] = "zero: no news published yet"
    else:
        try:
            result = sentiment.analyze_sentiment(news, covered, cfg)
            sentiment_adjustment.loc[covered] = result["adjustments"].reindex(covered).to_numpy()
            if len(covered) < len(tickers):
                status["sentiment"] = f"partial: {len(tickers) - len(covered)} tickers without news"
        except ValueError as exc:
            status["sentiment"] = f"zero: {exc}"

    macro_adjustment = zero.copy()
    indicators = pit["indicators"]
    if indicators.empty:
        status["macro"] = "zero: no indicators released yet"
    else:
        last_month = indicators.index.max()
        # 回归按同月配对价格与指标；价格若超过最近已发布月份，未发布月份的
        # 指标会被向前填充成零变化混入回归，因此价格同样截止到该月。
        start = (last_month.to_period("M") - int(cfg["backtest_macro_window_months"])).end_time.normalize()
        try:
            result = macro.analyze_macro(
                prices.loc[:last_month], indicators, {**cfg, "macro_start_date": start},
            )
            macro_adjustment = result["adjustments"].reindex(tickers)
        except ValueError as exc:
            status["macro"] = f"zero: {exc}"

    parts = {"sentiment": sentiment_adjustment, "macro": macro_adjustment}
    failures = {}
    for name, used in VARIANTS.items():
        try:
            frame = allocation.combine_weights(
                base,
                parts["sentiment"] if "sentiment" in used else zero,
                parts["macro"] if "macro" in used else zero,
                cfg,
            )
            targets[name] = frame["final_weight"].rename("weight")
        except ValueError as exc:
            failures[name] = str(exc)
    components = pd.DataFrame({
        "base_weight": base,
        "sentiment_adjustment": sentiment_adjustment,
        "macro_adjustment": macro_adjustment,
    }).rename_axis("ticker")
    return {"targets": targets, "failures": failures, "components": components, "status": status}


def simulate(prices, targets, cost_bps, start_date):
    """按日模拟一个策略版本的净值，目标权重在成交日收盘时生效。

    参数：
        prices：完整的日期 × 股票价格表。
        targets：以成交日为键、股票代码为索引的目标权重。
        cost_bps：单边成本（基点），乘以换手 Σ|目标 - 当前| 从净值中扣除。
        start_date：净值起点；此前及首次调仓前资金为零收益现金。

    成交日当天的收益先按原持仓计入，再按收盘价调仓；两次调仓之间权重随
    价格漂移。权重合计与 1 的差额视为零收益现金。缺失价格按最近已知价格
    估值（当天收益为零，恢复报价后一次性体现累计涨跌），只用过去的价格。

    返回（净值 Series，调仓记录 DataFrame）；初始资金为 1。
    """
    tickers = prices.columns
    valuation = prices.ffill()
    daily = valuation.pct_change(fill_method=None).fillna(0.0).to_numpy()
    dates = prices.index
    start = dates.get_loc(pd.Timestamp(start_date))
    holdings = np.zeros(len(tickers))
    cash = 1.0
    nav, log = [], []
    for i in range(start, len(dates)):
        if i > start:
            holdings = holdings * (1.0 + daily[i])
        value = holdings.sum() + cash
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Portfolio value became non-positive on {dates[i].date()}.")
        target = targets.get(dates[i])
        if target is not None:
            weights = target.reindex(tickers).to_numpy(dtype=float)
            if not np.isfinite(weights).all():
                raise ValueError("Target weights must be finite and cover every ticker.")
            unpriced = (weights != 0) & np.isnan(valuation.iloc[i].to_numpy())
            if unpriced.any():
                raise ValueError(f"Target weight on unpriced tickers {list(tickers[unpriced])} on {dates[i].date()}.")
            turnover = float(np.abs(weights - holdings / value).sum())
            cost = turnover * float(cost_bps) / 1e4
            value *= 1.0 - cost
            holdings = weights * value
            cash = value - holdings.sum()
            log.append({"execution_date": dates[i], "turnover": turnover, "cost": cost, "initial": not log})
        nav.append(value)
    return (
        pd.Series(nav, index=dates[start:], name="nav"),
        pd.DataFrame(log, columns=["execution_date", "turnover", "cost", "initial"]),
    )


def performance_metrics(nav, rebalances, settings):
    """计算各版本的收益、风险、回撤、换手及相对等权基准的超额表现。

    nav 为日期 × 版本的净值表，初始资金为 1；rebalances 为 run_backtest
    生成的调仓记录。波动率与夏普比率按 annualization 个交易日年化，复合
    年化收益按自然日折算年数。夏普比率扣除 backtest_risk_free_rate；
    超额收益与信息比率以同期等权组合为基准。首笔建仓不计入平均换手。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    periods = float(cfg.get("annualization", 252))
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    daily = nav.pct_change().iloc[1:]
    if years <= 0 or len(daily) < 2:
        raise ValueError("Backtest period is too short for performance metrics.")
    rf_daily = (1.0 + float(cfg["backtest_risk_free_rate"])) ** (1.0 / periods) - 1.0
    rows = {}
    for name in nav.columns:
        returns = daily[name]
        volatility = returns.std(ddof=1)
        trades = rebalances.loc[(rebalances["variant"] == name) & (rebalances["status"] == "rebalanced")]
        regular = trades.loc[~trades["initial"].astype(bool)]
        row = {
            "total_return": nav[name].iloc[-1] - 1.0,
            "cagr": nav[name].iloc[-1] ** (1.0 / years) - 1.0,
            "annual_volatility": volatility * np.sqrt(periods),
            "sharpe": (returns.mean() - rf_daily) / volatility * np.sqrt(periods) if volatility > 0 else np.nan,
            "max_drawdown": float((nav[name] / nav[name].cummax() - 1.0).min()),
            "avg_turnover": regular["turnover"].mean(),
            "annual_turnover": regular["turnover"].sum() / years,
            "annual_cost_drag": trades["cost"].sum() / years,
            "rebalances": len(trades),
            "held_periods": int(((rebalances["variant"] == name) & (rebalances["status"] == "held")).sum()),
        }
        if name != BENCHMARK and BENCHMARK in daily:
            active = returns - daily[BENCHMARK]
            tracking = active.std(ddof=1) * np.sqrt(periods)
            row["active_return_vs_equal_weight"] = active.mean() * periods
            row["information_ratio"] = row["active_return_vs_equal_weight"] / tracking if tracking > 0 else np.nan
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("variant")


def run_backtest(inputs, settings):
    """逐月执行信号计算和调仓模拟，返回报告表格、绩效指标和运行摘要。

    inputs 与 load_inputs 的返回值一致（不使用 clients）。每个信号日先截取
    信息集，再计算所有版本的目标权重；全部信号完成后统一模拟净值。表格：
    backtest_nav（每日净值）、backtest_metrics（绩效）、backtest_weights
    （每期目标权重）、backtest_signals（每期三个分量）、backtest_periods
    （每期数据截止时点与分量状态）、backtest_rebalances（每期每版本的
    调仓或沿用记录）。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    prices = inputs["prices"]
    data = {
        "prices": prices,
        "news": prepare_news(inputs["news"], cfg),
        "indicators": inputs["indicators"],
    }
    schedule = rebalance_schedule(prices.index, cfg)
    if schedule.empty:
        raise ValueError("No rebalance date has enough history; shorten backtest_lookback_days or extend the data.")
    names = [BENCHMARK, *VARIANTS]
    targets = {name: {} for name in names}
    period_rows, status_rows, signal_frames, weight_rows = [], [], [], []
    for signal_date, execution_date in schedule.itertuples(index=False):
        pit = point_in_time_inputs(data, signal_date, cfg)
        result = compute_signals(pit, cfg)
        period_rows.append({
            "signal_date": signal_date, "execution_date": execution_date,
            "price_data_through": pit["prices"].index.max(),
            "news_cutoff": pit["news_cutoff"], "news_available": len(pit["news"]),
            "macro_cutoff": pit["macro_cutoff"],
            "macro_data_through": pit["indicators"].index.max() if not pit["indicators"].empty else pd.NaT,
            **{f"{key}_status": value for key, value in result["status"].items()},
        })
        signal_frames.append(result["components"].assign(signal_date=signal_date).reset_index())
        for name in names:
            target = result["targets"].get(name)
            if target is None:
                status_rows.append({"signal_date": signal_date, "execution_date": execution_date,
                                    "variant": name, "status": "held", "reason": result["failures"][name]})
                continue
            targets[name][execution_date] = target
            status_rows.append({"signal_date": signal_date, "execution_date": execution_date,
                                "variant": name, "status": "rebalanced", "reason": ""})
            weight_rows.append({"signal_date": signal_date, "variant": name, **target.to_dict()})

    start = schedule["execution_date"].iloc[0]
    navs, logs = {}, []
    for name in names:
        nav, log = simulate(prices, targets[name], cfg["backtest_cost_bps"], start)
        navs[name] = nav
        if not log.empty:
            logs.append(log.assign(variant=name))
    nav = pd.DataFrame(navs).rename_axis("date")
    trades = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(
        columns=["execution_date", "turnover", "cost", "initial", "variant"])
    rebalances = pd.DataFrame(status_rows).merge(trades, on=["execution_date", "variant"], how="left")
    rebalances["initial"] = rebalances["initial"].astype("boolean").fillna(False).astype(bool)
    metrics = performance_metrics(nav, rebalances, cfg)
    periods = pd.DataFrame(period_rows).set_index("signal_date")
    count = len(periods)
    active = {
        name: int((~periods[f"{name}_status"].str.startswith("zero")).sum())
        for name in ("sentiment", "macro")
    }
    summary = {
        "first_signal": str(schedule["signal_date"].iloc[0].date()),
        "last_signal": str(schedule["signal_date"].iloc[-1].date()),
        "nav_start": str(nav.index[0].date()), "nav_end": str(nav.index[-1].date()),
        "periods": count,
        "execution_lag_days": int(cfg["backtest_execution_lag_days"]),
        "cost_bps": float(cfg["backtest_cost_bps"]),
        "lookback_days": cfg["backtest_lookback_days"],
        "macro_release_lag_months": int(cfg["backtest_macro_release_lag_months"]),
        # 调整分量实际生效的期数；若为 0，对应版本与 base 相同，比较没有意义。
        "overlay_coverage": {name: f"{value}/{count}" for name, value in active.items()},
        "base_failures": int((periods["base_status"] != "ok").sum()),
    }
    tables = {
        "backtest_nav": nav,
        "backtest_metrics": metrics,
        "backtest_weights": pd.DataFrame(weight_rows).set_index(["signal_date", "variant"]),
        "backtest_signals": pd.concat(signal_frames, ignore_index=True).set_index(["signal_date", "ticker"]),
        "backtest_periods": periods,
        "backtest_rebalances": rebalances.set_index(["signal_date", "variant"]),
    }
    return {"tables": tables, "metrics": metrics, "summary": summary}
