"""基于历史对数收益计算组合配置、目标收益曲线和蒙特卡洛模拟结果。

收益与风险估计均来自传入样本的均值和协方差，没有加入交易成本、税费、
换手约束或未来收益预测。收益波动比未扣除无风险利率，不能直接视为
使用超额收益计算的夏普比率。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULT_SETTINGS = {
    "annualization": 252,
    "selected_risk_level": 2,
    "risk_levels": (1, 2, 3),
    "risk_penalty_intercept": 11.0,
    "risk_penalty_slope": 3.33,
    "base_weight_decimals": 3,
    "frontier_min_return": 0.0,
    "frontier_max_return": 0.3,
    "frontier_points": 50,
    "optimizer_tolerance": 1e-6,
    "optimizer_max_iterations": 100,
    "constraint_tolerance": 1e-5,
}


def simulate_portfolios(log_returns, settings):
    """抽样生成只做多、权重合计为 1 的组合，并计算年化统计量。

    参数：
        log_returns：日期为行索引、唯一股票代码为列的非空 DataFrame。
            数值为单期对数收益，使用小数表示；至少需要两期完整有限观测。
        settings：需提供 monte_carlo_samples（模拟次数）、
            monte_carlo_annualization（每年期数）和 monte_carlo_seed
            （随机种子）。日频数据通常采用每年 252 个交易日的假设。

    返回：
        每行对应一次抽样的 DataFrame，包含 return（年化期望对数收益）、
        volatility（年化波动率）和 ratio（收益除以波动率）。收益和
        波动率均以小数表示；零波动组合的 ratio 为 NaN，不返回组合权重。

    年化规则为均值乘以每年期数、协方差乘以每年期数，再对组合方差
    开平方。这依赖收益在时间上的稳定性及线性方差缩放假设，并非复利
    净值收益的直接预测。增加模拟次数主要提高散点覆盖密度，不保证找到
    最优解；随机正数归一化也不等于在权重单纯形上均匀抽样。
    """
    if not isinstance(log_returns, pd.DataFrame) or log_returns.empty:
        raise ValueError("log_returns must be a nonempty DataFrame.")
    if len(log_returns) < 2 or not log_returns.columns.is_unique:
        raise ValueError("At least two return observations and unique tickers are required.")
    values = log_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns must contain only finite numbers.")
    count = int(settings["monte_carlo_samples"])
    periods = float(settings["monte_carlo_annualization"])
    if count <= 0 or periods <= 0:
        raise ValueError("Simulation count and annualization must be positive.")
    # 使用独立随机状态确保模拟可复现，避免修改进程的全局随机状态。
    rng = np.random.RandomState(settings["monte_carlo_seed"])
    # 先对各资产独立抽取 [0, 1) 随机数，再逐行归一化。
    # 这种生成方式只用于探索可行组合，不代表所有可行权重具有相同抽中概率。
    weights = rng.random_sample((count, values.shape[1]))
    weights /= weights.sum(axis=1, keepdims=True)
    mean = log_returns.mean().to_numpy() * periods
    covariance = log_returns.cov().to_numpy() * periods
    expected_returns = weights @ mean
    # 每个组合的方差为 wᵀΣw；向量化运算同时计算全部模拟组合。
    variances = np.einsum("ij,jk,ik->i", weights, covariance, weights)
    # 协方差计算的浮点误差可能产生极小负方差，开平方前仅按零截断方差。
    volatilities = np.sqrt(np.maximum(variances, 0))
    ratios = np.divide(
        expected_returns,
        volatilities,
        out=np.full(count, np.nan),
        where=volatilities > 0,
    )
    return pd.DataFrame({
        "return": expected_returns,
        "volatility": volatilities,
        "ratio": ratios,
    })


def analyze_portfolios(log_returns, settings):
    """优化基础组合，并按指定收益网格求解最小波动率组合。

    参数：
        log_returns：以日期为行、唯一股票代码为列的单期对数收益表，
            所有数值需有限且至少有两期。列顺序也是优化权重的资产顺序。
        settings：覆盖 DEFAULT_SETTINGS 的配置。annualization 指定
            每年期数；selected_risk_level 选择用于后续调整的基础组合；
            risk_levels 指定需要报告的风险级别；有效前沿的收益范围、
            点数、优化精度和最大迭代次数也由配置控制。

    返回：
        base_weights：所选风险级别的权重 Series，以股票代码为索引，
            按 base_weight_decimals 取整，取整后合计可能略偏离 1。
        metrics：以策略名称为索引，报告未取整解的年化 return、
            volatility 与 ratio；ratio 未扣除无风险利率。
        frontier：各目标年化收益对应的最小波动率、成功标记及求解信息。
            不可行或求解失败的目标保留记录，volatility 设为 NaN。

    各资产权重约束为 [0, 1] 且合计为 1。风险级别组合最小化
    -年化收益 + 风险惩罚系数 × 年化波动率，惩罚系数按配置随风险级别
    线性变化。较低惩罚通常允许更高风险；若自定义配置使系数为负，
    目标会奖励波动率，因此参数的业务含义需要另行检查。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    if not isinstance(log_returns, pd.DataFrame) or log_returns.empty:
        raise ValueError("log_returns must be a nonempty DataFrame.")
    if len(log_returns) < 2 or not log_returns.columns.is_unique:
        raise ValueError("At least two observations and unique tickers are required.")
    if not np.isfinite(log_returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns must contain only finite numbers.")
    annualization = float(cfg["annualization"])
    if annualization <= 0:
        raise ValueError("annualization must be positive.")
    # 对数收益按线性规则年化；这些样本估计不会自动修正收益自相关或结构变化。
    expected = log_returns.mean().to_numpy() * annualization
    covariance = log_returns.cov().to_numpy() * annualization
    count = len(expected)
    # 用等权组合作为共同初值，预算约束确保优化过程始终针对全额投资组合。
    initial = np.full(count, 1.0 / count)
    bounds = [(0.0, 1.0)] * count
    budget = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    tolerance = float(cfg["constraint_tolerance"])

    def statistics(weights):
        """计算给定权重的年化收益、波动率和未扣无风险利率的收益波动比。"""
        ret = float(expected @ weights)
        vol = float(np.sqrt(max(float(weights @ covariance @ weights), 0.0)))
        return ret, vol, ret / vol if vol > 0 else np.nan

    def solve(objective, extra_constraints=()):
        """在共同预算与边界约束下求解；额外约束用于指定目标收益。"""
        return minimize(
            objective, initial, method="SLSQP", bounds=bounds,
            constraints=(budget, *extra_constraints),
            options={
                "ftol": float(cfg["optimizer_tolerance"]),
                "maxiter": int(cfg["optimizer_max_iterations"]),
            },
        )

    def valid(result):
        """同时检查求解状态、数值有限性、权重预算和上下界容差。"""
        return (
            result.success and np.isfinite(result.x).all()
            and np.isfinite(result.fun)
            and abs(result.x.sum() - 1.0) <= tolerance
            and np.min(result.x) >= -tolerance
            and np.max(result.x) <= 1.0 + tolerance
        )

    def negative_ratio(weights):
        """把最大收益波动比转为最小化问题，并排除无法计算比率的解。"""
        ratio = statistics(weights)[2]
        return -ratio if np.isfinite(ratio) else np.inf

    solutions = {
        "max_ratio": solve(negative_ratio),
        "min_variance": solve(lambda weights: statistics(weights)[1] ** 2),
    }
    # 同时求解报告所需风险级别及最终选择的级别，去重后保留配置顺序。
    risks = list(dict.fromkeys([*cfg["risk_levels"], cfg["selected_risk_level"]]))
    for level in risks:
        penalty = float(cfg["risk_penalty_intercept"]) - float(cfg["risk_penalty_slope"]) * level
        solutions[f"risk_{level}"] = solve(
            lambda weights, coefficient=penalty:
                -statistics(weights)[0] + coefficient * statistics(weights)[1]
        )
    # 关键组合失败时停止，避免后续配置阶段把失败求解器输出当作有效权重。
    for name, result in solutions.items():
        if not valid(result):
            raise RuntimeError(f"Portfolio optimization failed for {name}: {result.message}")
    metrics = pd.DataFrame(
        [statistics(result.x) for result in solutions.values()],
        index=pd.Index(solutions, name="strategy"),
        columns=["return", "volatility", "ratio"],
    )
    # 在给定目标收益下最小化波动率。全部目标构成最小方差边界；
    # 若包含全局最小方差组合以下的收益区间，其中部分点并非有效的上半支。
    frontier_rows = []
    for target in np.linspace(
        cfg["frontier_min_return"], cfg["frontier_max_return"], int(cfg["frontier_points"])
    ):
        # 禁止做空时，组合期望收益是资产收益的凸组合，不能超出其最小/最大值。
        if target < expected.min() - tolerance or target > expected.max() + tolerance:
            frontier_rows.append({
                "return": target, "volatility": np.nan, "success": False,
                "message": "Target outside attainable long-only return range.",
            })
            continue
        constraint = {"type": "eq", "fun": lambda weights, goal=target: expected @ weights - goal}
        result = solve(lambda weights: statistics(weights)[1], (constraint,))
        # 求解器成功标记之外，还要单独验证目标收益等式是否在允许误差内。
        success = valid(result) and abs(expected @ result.x - target) <= tolerance
        frontier_rows.append({
            "return": target,
            "volatility": statistics(result.x)[1] if success else np.nan,
            "success": bool(success), "message": str(result.message),
        })
    # 仅对交给配置模块的基础权重取整；指标仍对应未取整的优化解。
    selected = solutions[f"risk_{cfg['selected_risk_level']}"]
    return {
        "base_weights": pd.Series(
            selected.x.round(int(cfg["base_weight_decimals"])),
            index=log_returns.columns.rename("ticker"), name="base_weight",
        ),
        "metrics": metrics,
        "frontier": pd.DataFrame(frontier_rows),
    }
