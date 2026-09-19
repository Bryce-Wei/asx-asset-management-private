"""独立的 ARIMA 留出集预测实验，预测结果不参与组合权重调整。

在项目根目录运行：
    python experiments/arima_forecast.py --data-dir data/raw --ticker BHP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# 将项目根目录加入导入路径，支持直接运行实验脚本。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asset_management.features import log_returns
from asset_management.load_data import load_prices


def run_experiment(data_dir: Path, ticker: str | None, output_dir: Path, train_ratio: float = 0.7) -> dict:
    """以时间顺序划分样本，用训练集拟合 ARIMA 并预测整个留出区间。

    输入来自指定目录的股票价格工作簿；ticker 未给定时取第一只股票。
    预测对象为该股票相邻有效观测的对数收益，小数 0.01 表示对数价格
    变化 0.01，不是价格点数，也未转换为年化收益。train_ratio 控制
    训练样本比例，取整后至少保留 30 个训练收益和 2 个留出收益。

    模型选择与拟合只使用训练集，预测从训练末端一次产生全部留出期，
    不使用留出真实值滚动更新。以训练均值的常数预测作为对照，返回
    两者的留出均方误差和模型阶数，并保存预测明细、指标及时间序列图。
    此独立实验不由主入口调用，预测也不参与组合权重调整。
    """
    if not 0.1 <= train_ratio <= 0.9:
        raise ValueError("train_ratio must be between 0.1 and 0.9.")
    try:
        # 实验依赖按需导入，主分析及模块读取不因缺少 pmdarima 而失败。
        from pmdarima import auto_arima
    except ImportError as exc:
        raise RuntimeError(
            "ARIMA requires the optional experiment dependencies. "
            "Install with: python -m pip install -r experiments/requirements.txt"
        ) from exc
    prices = load_prices(Path(data_dir) / "ASX200top10.xlsx")
    ticker = ticker or str(prices.columns[0])
    if ticker not in prices.columns:
        raise ValueError(f"Unknown ticker {ticker!r}; choose from {', '.join(prices.columns)}.")
    # 先选定单只股票，再计算收益，避免其他股票缺失导致无关日期被删除。
    series = log_returns(prices.loc[:, [ticker]])[ticker]
    split = int(len(series) * train_ratio)
    if split < 30 or len(series) - split < 2:
        raise ValueError("ARIMA needs at least 30 training returns and two holdout returns.")
    # 按时间切分且不打乱样本，使未来留出区间不会进入训练信息集合。
    train, test = series.iloc[:split], series.iloc[split:]
    # 在训练收益上搜索非季节性模型；差分阶数自动确定，p、q 上限为 5。
    # 候选拟合失败时继续搜索，由库的默认信息准则选择最终阶数。
    model = auto_arima(
        train.to_numpy(), start_p=1, start_q=1, max_p=5, max_q=5,
        seasonal=False, d=None, D=0, stepwise=True,
        suppress_warnings=True, error_action="ignore",
    )
    predictions = np.asarray(model.predict(n_periods=len(test)), dtype=float)
    # 多步预测必须与留出区间一一对应，拒绝长度不符或非有限的预测结果。
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise RuntimeError("ARIMA produced an invalid holdout forecast.")
    # 两种预测使用同一留出集比较；MSE 的单位为小数对数收益的平方。
    # 基准均值只从训练集估计，不能用留出均值以免引入未来信息。
    error = float(np.mean((test.to_numpy() - predictions) ** 2))
    baseline_error = float(np.mean((test.to_numpy() - float(train.mean())) ** 2))
    metrics = {
        "ticker": ticker, "target": "log_return", "train_observations": len(train),
        "holdout_observations": len(test), "train_ratio": train_ratio,
        "order": [int(value) for value in model.order], "holdout_mse": error,
        "training_mean_baseline_mse": baseline_error,
        "forecast_method": "从固定预测起点进行多步预测，不使用留出集重新拟合",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 真实值保留留出期日期，预测按相同期序写入；同名输出文件会覆盖。
    results = test.to_frame("actual_log_return")
    results["predicted_log_return"] = predictions
    results.to_csv(output_dir / "arima_holdout.csv", index_label="date")
    (output_dir / "arima_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(11, 4.5))
    axes.plot(train.index, train, label="Training returns", linewidth=0.7, alpha=0.7)
    axes.plot(test.index, test, label="Holdout returns", linewidth=0.7)
    axes.plot(test.index, predictions, label="ARIMA forecast", linewidth=1.3)
    # 竖线标出训练/留出边界；曲线全部是收益序列，不作累计价格重建。
    axes.axvline(test.index[0], color="gray", linestyle="--", linewidth=1)
    axes.set(title=f"{ticker}: ARIMA log-return holdout forecast", xlabel="Date", ylabel="Log return")
    axes.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "arima_forecast.png", dpi=160)
    plt.close(figure)
    return metrics


def main() -> int:
    """读取实验参数并打印指标，失败时以命令行错误状态退出。

    默认从项目 data/raw 读取数据，结果写入 outputs/arima；也可显式
    指定输入和输出目录，与完整分析报告分别保存以便比较。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--ticker", default=None, help="股票代码，默认使用工作簿中的第一只股票。")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "arima")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()
    try:
        metrics = run_experiment(args.data_dir, args.ticker, args.output_dir, args.train_ratio)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ARIMA experiment failed: {exc}\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
