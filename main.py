"""执行完整资产管理分析流程，支持按月滚动回测，也支持使用模拟数据运行演示。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from config import RAW_DIR, SAMPLE_DIR, OUTPUT_DIR, SETTINGS
from asset_management.features import log_returns
from asset_management.load_data import load_inputs, load_sample_prices
from asset_management.reporting import save_reports


def run_demo(sample_dir: Path, output_dir: Path) -> dict:
    """使用模拟价格和固定示例权重验证读取、检查和报告生成流程。

    ``sample_dir`` 中的 ``prices.csv`` 按日期排列资产价格，
    ``allocations.csv`` 以资产代码为索引，至少包含 ``final_weight``。
    两份文件必须覆盖同一组资产，避免图中的配置与价格对象不一致。
    本模式直接展示文件中的固定权重，不调用客户分群、组合优化、
    蒙特卡洛、新闻情绪或宏观模型，也不将示例权重解释为投资建议。
    返回模式说明、权重检查结果及已生成的文件路径，供运行摘要使用。
    """
    prices = load_sample_prices(sample_dir)
    allocations = pd.read_csv(sample_dir / "allocations.csv", index_col=0)
    if set(allocations.index) != set(prices.columns):
        raise ValueError("Demo price tickers and allocation tickers must match.")
    from asset_management.allocation import validate_weights
    # 检查只描述现有权重的合计与空头情况，不自动修正示例输入。
    checks = validate_weights(allocations)
    tables = {"allocations": allocations, "prices": prices, "log_returns": log_returns(prices)}
    files = save_reports(tables, output_dir, mode="demo")
    return {"mode": "demo", "description": "使用模拟数据和固定示例权重生成演示结果。",
            "weight_checks": checks, "files": [str(p) for p in files]}


def run_full(data_dir: Path, output_dir: Path, risk_level: int | None = None) -> dict:
    """执行完整分析，并将中间结果与最终配置一并保存。

    ``data_dir`` 提供股票价格、客户记录、新闻标题和月度经济指标。
    ``risk_level`` 为可选的单次运行覆盖值；未提供时使用配置文件。
    客户分群用于描述客户结构，组合优化使用所选风险等级生成基础
    权重；蒙特卡洛用于展示收益与风险分布，其结果不参与最终权重。
    新闻与宏观模块分别输出按股票代码索引的调整量，再与基础权重
    对齐、相加、归一化。最后输出检查摘要以及各表格和图表的路径。
    """
    # 延后导入完整模型，使演示入口不必初始化这些模型及其额外资源。
    from asset_management import clients, portfolio, sentiment, macro, allocation

    # 复制配置后覆盖本次风险等级，避免一次调用影响后续调用的默认值。
    settings = dict(SETTINGS)
    if risk_level is not None:
        settings["selected_risk_level"] = risk_level
    inputs = load_inputs(data_dir)
    # 先统一价格日期及资产列，再以共同有效日期的收益率估计组合统计量。
    returns = log_returns(inputs["prices"])
    client_results = clients.analyze_clients(inputs["clients"], settings)
    portfolio_results = portfolio.analyze_portfolios(returns, settings)
    simulations = portfolio.simulate_portfolios(returns, settings)
    sentiment_results = sentiment.analyze_sentiment(inputs["news"], list(returns.columns), settings)
    macro_results = macro.analyze_macro(inputs["prices"], inputs["indicators"], settings)
    # 三类权重按代码匹配，不能按行号直接相加；调整后可能出现负权重。
    allocations = allocation.combine_weights(portfolio_results["base_weights"],
                                              sentiment_results["adjustments"],
                                              macro_results["adjustments"], settings)
    checks = allocation.validate_weights(allocations)
    # 仅汇集表格对象交给报告层；客户结果字典中的其他元数据不强制转表。
    tables = {
        "allocations": allocations,
        "portfolio_metrics": portfolio_results["metrics"],
        "simulations": simulations,
        "frontier": portfolio_results["frontier"],
        "sentiment_scores": sentiment_results["scores"],
        "macro_diagnostics": macro_results["diagnostics"],
        **{"clients_" + k: v for k, v in client_results.items() if isinstance(v, pd.DataFrame)},
    }
    files = save_reports(tables, output_dir, mode="full")
    return {"mode": "full", "description": "已完成客户分群、组合优化、蒙特卡洛、新闻情绪和宏观分析。",
            "risk_level": settings["selected_risk_level"], "weight_checks": checks,
            "files": [str(p) for p in files]}


def run_backtest(data_dir: Path, output_dir: Path, risk_level: int | None = None) -> dict:
    """按月滚动回测完整配置流程，每期只使用信号日当时可获得的数据。

    读取与完整模式相同的原始输入；每个月末重新优化基础组合并计算情绪、
    宏观调整，按设定的成交滞后与交易成本模拟每日净值。同时输出等权、
    仅基础组合、仅加情绪、仅加宏观和完整策略五个版本，用于判断各调整
    分量是否带来增益。回测参数见 backtest.DEFAULT_SETTINGS，可在
    config.SETTINGS 中覆盖。返回回测区间、各版本绩效和生成文件路径。
    """
    from asset_management import backtest

    settings = dict(SETTINGS)
    if risk_level is not None:
        settings["selected_risk_level"] = risk_level
    result = backtest.run_backtest(load_inputs(data_dir), settings)
    files = save_reports(result["tables"], output_dir, mode="backtest")
    # NaN 转为 JSON null，避免输出非标准 JSON。
    metrics = result["metrics"].round(6).astype(object)
    metrics = metrics.where(metrics.notna(), None).to_dict(orient="index")
    return {"mode": "backtest", "description": "按月滚动、逐期截取当时可得数据的历史回测。",
            "risk_level": settings["selected_risk_level"], **result["summary"],
            "metrics": metrics, "files": [str(p) for p in files]}


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数、选择运行模式并写出机器可读的运行摘要。

    完整模式默认输出到 ``outputs``，演示和回测模式默认输出到其 ``demo``、
    ``backtest`` 子目录；指定 ``--output-dir`` 时各模式都使用该显式路径。
    分析失败时将错误写到标准错误并返回 1；成功时保存 UTF-8 JSON，
    同时在标准输出打印摘要并返回 0，便于终端或外部脚本判断结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full", "backtest"), default="full",
                        help="运行模式，默认执行完整分析；backtest 为按月滚动回测")
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR, help="完整分析与回测的原始数据目录")
    parser.add_argument("--sample-dir", type=Path, default=SAMPLE_DIR, help="演示数据目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录，完整模式默认为 outputs")
    parser.add_argument("--risk-level", type=int, choices=(1, 2, 3), default=None, help="覆盖配置中的风险等级：1、2 或 3")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (OUTPUT_DIR / args.mode if args.mode != "full" else OUTPUT_DIR)
    try:
        if args.mode == "demo":
            result = run_demo(args.sample_dir, output_dir)
        elif args.mode == "backtest":
            result = run_backtest(args.data_dir, output_dir, args.risk_level)
        else:
            result = run_full(args.data_dir, output_dir, args.risk_level)
    except (OSError, ValueError, RuntimeError, ImportError, LookupError) as exc:
        print(f"无法完成 {args.mode} 模式：{exc}", file=sys.stderr)
        return 1
    # 所有模型及报告步骤完成后才写本次成功摘要；同名结果文件会被覆盖。
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
