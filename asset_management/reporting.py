"""将分析结果保存为表格和图表，不重新估计模型或调整组合。

报告层只消费传入的表格，按可用字段决定生成哪些图。表格保留计算
结果和索引，图中对缺失或无限坐标的过滤仅用于显示，不回写结果表。
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

# 使用非交互后端，命令行或没有图形桌面的环境也可直接生成 PNG 文件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def save_reports(
    tables: dict[str, pd.DataFrame], output_dir: str | Path, *, mode: str
) -> list[Path]:
    """保存带字段名及索引的 CSV 和可用图表，返回实际生成文件的路径。

    ``tables`` 的键用作文件名，值必须是表格；所有表格保存到 tables
    子目录，图像保存到 figures 子目录。目录不存在时创建，同名文件
    会覆盖；不会删除此次没有生成的旧文件，因此跨次比较应另设目录。

    可选图表约定：allocations 以股票代码为索引并含 final_weight；
    prices 以日期为索引、资产为列；simulations、frontier 及
    portfolio_metrics 含 return 和 volatility，数值已经由模型年化；
    clients_counts 含 clients 人数列；sentiment_scores 含 compound
    综合分。缺少对应结果时跳过该图，不补算模型或构造替代结果。

    演示模式只展示传入配置与价格，并注明模拟数据和固定示例权重；
    完整模式还可展示模拟组合、前沿、客户分群和新闻情绪。
    """
    if mode not in {"demo", "full"}:
        raise ValueError("mode must be 'demo' or 'full'")
    for name, frame in tables.items():
        # 限制键的字符集合，防止文件名意外包含路径分隔符或目录跳转。
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid report table name: {name!r}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Report {name!r} must be a DataFrame")

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, frame in tables.items():
        target = tables_dir / f"{name}.csv"
        # 写入 UTF-8 字节序标记方便 Excel 识别中文；索引保留股票、日期或策略标签。
        frame.to_csv(target, encoding="utf-8-sig", index=True)
        saved.append(target)

    note = (
        "DEMO | Synthetic data and fixed illustrative weights."
        if mode == "demo"
        else "FULL | Local analysis output; interpretation depends on input data and assumptions."
    )

    def finish(fig: plt.Figure, filename: str) -> None:
        """统一添加模式说明和边距，保存图像并释放图形资源。"""
        fig.text(0.04, 0.025, note, fontsize=8, color="#526175")
        fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.97))
        target = figures_dir / filename
        fig.savefig(target, dpi=160, facecolor="white")
        plt.close(fig)
        saved.append(target)

    allocations = tables.get("allocations")
    if allocations is not None and "final_weight" in allocations and not allocations.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        weights = allocations["final_weight"]
        bars = ax.bar(weights.index.astype(str), weights, color="#267F86", width=0.58)
        ax.bar_label(bars, labels=[f"{weight:.1%}" for weight in weights], padding=5)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Portfolio weight")
        ax.set_title(
            "Illustrative allocation | synthetic demo"
            if mode == "demo"
            else "Final portfolio allocation",
            loc="left", fontweight="bold", pad=16,
        )
        # 坐标轴保留负权重，直观呈现调整后的空头，而不是在显示时截断。
        ax.set_ylim(min(0, float(weights.min()) * 1.2), max(0.1, float(weights.max()) * 1.22))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5EAF0")
        ax.spines[["top", "right"]].set_visible(False)
        finish(fig, "allocation.png")

    prices = tables.get("prices")
    if prices is not None and not prices.empty:
        numeric = prices.select_dtypes(include="number")
        # 只有首期各资产都有正价格时才统一缩放到 100，避免无效除数。
        # 这是资产价格走势对比，没有使用组合权重，也不是组合回测净值。
        if not numeric.empty and (numeric.iloc[0] > 0).all():
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            (numeric / numeric.iloc[0] * 100).plot(ax=ax, linewidth=2)
            ax.set_title(
                "Synthetic price paths | demo data"
                if mode == "demo"
                else "Indexed price paths",
                loc="left", fontweight="bold", pad=16,
            )
            ax.set_ylabel("Indexed price (first observation = 100)")
            ax.set_xlabel("Date")
            ax.grid(color="#E5EAF0")
            ax.spines[["top", "right"]].set_visible(False)
            finish(fig, "prices.png")

    # 完整模式的图表只展示传入的计算结果，模型计算和评分由各分析模块负责。
    if mode == "full":
        simulations = tables.get("simulations")
        frontier = tables.get("frontier")
        metrics = tables.get("portfolio_metrics")
        has_simulations = simulations is not None and {"return", "volatility"}.issubset(simulations)
        has_frontier = frontier is not None and {"return", "volatility"}.issubset(frontier)
        has_metrics = metrics is not None and {"return", "volatility"}.issubset(metrics)
        if has_simulations or has_frontier or has_metrics:
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            if has_simulations:
                # 只移除无法绘制坐标的行；颜色直接使用传入比率，不另算无风险利率。
                points = simulations.replace([np.inf, -np.inf], np.nan).dropna(subset=["return", "volatility"])
                if not points.empty:
                    scatter_kwargs = {"s": 10, "alpha": 0.45, "label": "Simulated portfolios"}
                    if "ratio" in points:
                        scatter_kwargs.update(c=points["ratio"], cmap="viridis")
                    else:
                        scatter_kwargs["color"] = "#93ACBC"
                    scatter = ax.scatter(points["volatility"], points["return"], **scatter_kwargs)
                    if "ratio" in points:
                        fig.colorbar(scatter, ax=ax, label="Supplied return / volatility ratio")
            if has_frontier:
                curve = frontier
                # 优化失败的目标收益点保留在 CSV 中，但不连入可行前沿曲线。
                if "success" in curve:
                    curve = curve.loc[curve["success"].eq(True)]
                curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["return", "volatility"])
                curve = curve.sort_values("return")
                if not curve.empty:
                    ax.plot(curve["volatility"], curve["return"], color="#273B58", linewidth=2,
                            label="Optimized frontier")
            if has_metrics:
                # 每个策略单独标记，便于与随机模拟点云和有效前沿对比。
                points = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["return", "volatility"])
                markers = ["*", "D", "s", "^", "P", "X"]
                for i, (strategy, row) in enumerate(points.iterrows()):
                    ax.scatter(row["volatility"], row["return"], marker=markers[i % len(markers)],
                               s=95, edgecolors="white", linewidths=0.8, zorder=5,
                               label=str(strategy))
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("Annualized volatility")
            ax.set_ylabel("Annualized return")
            ax.set_title("Portfolio simulation and optimized frontier", loc="left", fontweight="bold", pad=16)
            ax.grid(color="#E5EAF0")
            ax.spines[["top", "right"]].set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=8, frameon=False, loc="best")
            finish(fig, "efficient_frontier.png")

        counts = tables.get("clients_counts")
        if counts is not None and "clients" in counts and not counts.empty:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            bars = ax.bar(counts.index.astype(str), counts["clients"], color="#267F86", width=0.58)
            ax.bar_label(bars, padding=5)
            ax.set_xlabel("Cluster label")
            ax.set_ylabel("Number of clients")
            ax.set_title("Client cluster distribution", loc="left", fontweight="bold", pad=16)
            ax.set_ylim(0, max(1, float(counts["clients"].max()) * 1.18))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E5EAF0")
            ax.spines[["top", "right"]].set_visible(False)
            finish(fig, "client_clusters.png")

        scores = tables.get("sentiment_scores")
        if scores is not None and "compound" in scores and not scores.empty:
            compound = scores["compound"].replace([np.inf, -np.inf], np.nan).dropna()
            # 接受已汇总或逐行的分数；按股票索引求均值，仅展示 compound，
            # 不将该图的柱高等同于情绪模块实际输出的加法权重调整量。
            compound = compound.groupby(level=0, sort=False).mean()
            if not compound.empty:
                fig, ax = plt.subplots(figsize=(9.5, 4.8))
                colors = ["#267F86" if value >= 0 else "#C66A54" for value in compound]
                ax.bar(compound.index.astype(str), compound, color=colors, width=0.62)
                ax.axhline(0, color="#526175", linewidth=0.8)
                ax.set_ylim(min(-1, float(compound.min()) * 1.1), max(1, float(compound.max()) * 1.1))
                ax.set_ylabel("Mean supplied compound score")
                ax.set_title("News sentiment by asset", loc="left", fontweight="bold", pad=16)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="#E5EAF0")
                ax.spines[["top", "right"]].set_visible(False)
                finish(fig, "sentiment.png")

    return saved
