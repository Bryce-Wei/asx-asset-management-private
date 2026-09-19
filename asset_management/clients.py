"""将客户数值特征标准化后进行 KMeans 聚类，并按配置映射风险等级。

聚类识别的是特征相似性，簇编号本身没有风险高低顺序。风险等级映射属于
另行设定的业务规则，不能把聚类结果直接解释为已经验证的风险承受能力。
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_SETTINGS = {
    "client_clusters": 3,
    "client_random_state": 42,
    "client_n_init": 10,
    "cluster_to_risk": {0: 1, 1: 2, 2: 3},
}


def analyze_clients(clients_df, settings):
    """生成客户分组、聚类中心及各组人数。

    参数：
        clients_df：每行一位客户的 DataFrame，索引随结果保留。必须含
            risk_profile 列；该列与已有 cluster 列不参与建模。其余列
            全部作为特征，因此需事先剔除标识符和无关字段，并完成数值编码。
        settings：可覆盖默认配置。client_clusters 指定分组数；
            client_random_state 固定随机初始化；client_n_init 控制尝试
            的初始聚类次数；cluster_to_risk 将簇编号映射为风险等级。

    返回：
        assignments：保留客户输入列，并增加 cluster 与 mapped_risk_level。
        centers：以簇编号为索引、以特征名为列的中心，已恢复为输入特征单位。
        counts：以簇编号为索引，clients 列记录每组客户数。

    标准化避免收入、年龄等量纲不同的特征仅因数值范围较大而主导距离。
    这里不以 risk_profile 训练分类器，也不评估风险标签预测准确率。
    类别编码会参与欧氏距离计算，其编码方式会影响分组解释。
    """
    cfg = {**DEFAULT_SETTINGS, **settings}
    if "risk_profile" not in clients_df:
        raise ValueError("Client data must contain risk_profile.")
    # 排除已有风险标签，避免把用于解释分组的标签本身作为聚类依据。
    features = clients_df.drop(columns=["risk_profile", "cluster"], errors="ignore")
    if features.empty or len(features) < cfg["client_clusters"]:
        raise ValueError("Not enough client rows or features for clustering.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Clustering features must be finite numeric values.")
    # 缺失值、无穷值不能直接用于距离计算；校验通过后再按列标准化。
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    model = KMeans(
        n_clusters=int(cfg["client_clusters"]),
        random_state=int(cfg["client_random_state"]),
        n_init=int(cfg["client_n_init"]),
    ).fit(scaled)
    # 簇编号只是模型内部标签；映射表缺少某个编号时，该组风险等级会为空。
    # 调整特征或样本后，需要重新检查中心特征与映射规则是否仍然一致。
    assignments = clients_df.copy()
    assignments["cluster"] = model.labels_
    assignments["mapped_risk_level"] = assignments["cluster"].map(cfg["cluster_to_risk"])
    # 将中心从标准化空间还原，便于用收入、年龄等原始单位解释每组特征。
    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=features.columns,
    ).rename_axis("cluster")
    counts = assignments.groupby("cluster").size().rename("clients").to_frame()
    return {"assignments": assignments, "centers": centers, "counts": counts}
