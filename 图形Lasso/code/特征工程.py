# ===================================================================
# 阶段三-A：特征工程
# 从 L2-B 输出的 GMVP 权重和资产网络中构建 VARX 预测所需的
# 预测变量矩阵 X 与目标变量矩阵 Y。
#
# 论文对应：§3.2 网络正则化 VARX
#   滞后权重 w̃_{t-1}...w̃_{t-p}
#   外生变量 x_t (VIX/行业价差等)
#   网络拓扑特征 (度中心性/PageRank)
#   滚动网络均值 Ā (公式7)
# ===================================================================
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

from 共享模块 import (
    K, P_LAGS, NETWORK_THRESHOLD, L_TRAIN_GLASSO,
    OUT_DIR, log, set_log_file,
)


# ===================================================================
# 外生变量加载（基础模板，可按需扩展为实际数据源）
# ===================================================================
def load_exogenous(dates: list) -> pd.DataFrame:
    """加载市场状态外生变量。

    当前为模板——返回占位 DataFrame。接入真实数据时替换此函数。

    Args:
        dates: 交易日日期列表 (YYYYMMDD)

    Returns:
        DataFrame, index=dates, 列包含: vix, sector_spread, lag_turnover, ...
    """
    n = len(dates)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date":             dates,
        "vix":              rng.normal(20, 5, n).clip(8),      # VIX 波动率指数
        "sector_spread":    rng.normal(0.02, 0.01, n),         # 行业收益率差
        "lag_market_vol":   rng.normal(0.01, 0.005, n).clip(0),# 滞后市场波动率
        "lag_ba_spread":    rng.normal(0.0005, 0.0002, n).clip(0),# 滞后买卖价差
    }, index=dates)
    return df


# ===================================================================
# 网络拓扑特征
# ===================================================================
def compute_network_features(adj_mat: np.ndarray) -> dict:
    """从邻接矩阵提取拓扑特征。

    Args:
        adj_mat: (K,K) 布尔邻接矩阵

    Returns:
        dict: degree_centrality(K,), pagerank(K,), clustering(K,), modularity(float)
    """
    # 度中心性
    degree = adj_mat.sum(axis=0).astype(np.float64)
    degree /= (K - 1) if K > 1 else 1.0

    # 简化 PageRank (阻尼因子 0.85, 50 次迭代)
    d = 0.85
    pagerank = np.ones(K) / K
    out_deg = adj_mat.sum(axis=1).astype(np.float64)
    out_deg[out_deg == 0] = -1  # 沉没节点

    for _ in range(50):
        new_pr = np.ones(K) * (1 - d) / K
        for i in range(K):
            if out_deg[i] > 0:
                new_pr += d * (adj_mat[i] @ pagerank) * pagerank[i] / out_deg[i]
            else:
                new_pr += d * pagerank[i] / K
        pagerank = new_pr / new_pr.sum()

    # 聚类系数
    clustering = np.zeros(K)
    for i in range(K):
        neighbors = np.where(adj_mat[i] > 0)[0]
        if len(neighbors) < 2:
            clustering[i] = 0.0
        else:
            sub = adj_mat[np.ix_(neighbors, neighbors)]
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            clustering[i] = np.triu(sub, k=1).sum() / possible if possible > 0 else 0.0

    return {
        "degree_centrality": degree,
        "pagerank":          pagerank,
        "clustering":        clustering,
    }


# ===================================================================
# 滚动网络均值 (论文公式7)
# ===================================================================
def rolling_network_mean(
    adj_mats: dict,        # {day_idx: adj_mat(K,K)}
    t: int,
    window: int = 20
) -> np.ndarray:
    """Ā_{ij,t} = (1/window) Σ_{ℓ=0}^{window-1} A_{ij,t-ℓ}

    Args:
        adj_mats: {日期索引: (K,K) 邻接矩阵}
        t: 当前日期索引
        window: 滚动窗口长度

    Returns:
        (K,K) 滚动平均网络矩阵
    """
    rolling_mean = np.zeros((K, K), dtype=np.float64)
    count = 0
    for lag in range(window):
        idx = t - lag
        if idx in adj_mats:
            rolling_mean += adj_mats[idx]
            count += 1
    if count > 0:
        rolling_mean /= count
    return rolling_mean


# ===================================================================
# 构建特征矩阵 (主入口)
# ===================================================================
def build_feature_matrix(
    weights: pd.DataFrame,          # (T, K) GMVP 权重矩阵
    adj_mats: dict,                 # {day_idx: (K,K) 邻接矩阵}
    exogenous: pd.DataFrame,        # 外生变量
    date_list: list,                # 交易日列表
) -> Tuple[np.ndarray, np.ndarray, list]:
    """构建 VARX 预测的 X (特征) 和 Y (目标)。

    对每日 t, 构建:
      X[t] = [ vec(w̃_{t-1}), ..., vec(w̃_{t-p}),         ← 滞后权重
               x_t,                                      ← 外生变量
               deg_centrality(t-1), pagerank(t-1), ... ] ← 网络特征

      Y[t] = w̃_t    ← 需预测的 GMVP 权重

    Args:
        weights:   (T, K) GMVP 权重
        adj_mats:  {day_idx: (K,K) 邻接矩阵}
        exogenous: 外生变量 DataFrame
        date_list: 交易日序列

    Returns:
        X: (n_samples, n_features) 特征矩阵
        Y: (n_samples, K) 目标矩阵
        valid_indices: 可用日期索引列表
    """
    T, K_ = weights.shape
    assert K_ == K, f"权重列数 {K_} 不等于 K={K}"

    # 外生变量对齐到 date_list
    x_df = exogenous.reindex(date_list).fillna(method='ffill').fillna(0)
    n_exog = x_df.shape[1]

    # 总特征维度
    n_lag_features    = P_LAGS * K          # 滞后权重
    n_network_features = 3 * K             # 度/PageRank/聚类 (按资产)
    n_total           = n_lag_features + n_exog + n_network_features

    features = []
    targets  = []
    valid_indices = []

    for t in range(P_LAGS + 1, T):
        # 检查所有滞后日有数据
        skip = False
        for lag in range(1, P_LAGS + 1):
            if t - lag not in weights.index:
                skip = True
                break
        if skip:
            continue

        # --- 滞后权重 ---
        lag_vecs = []
        for lag in range(1, P_LAGS + 1):
            lag_vecs.append(weights.iloc[t - lag].values)
        lag_block = np.concatenate(lag_vecs)          # (P_LAGS * K,)

        # --- 外生变量 ---
        if t < len(date_list):
            exog_vec = x_df.iloc[t].values            # (n_exog,)
        else:
            exog_vec = np.zeros(n_exog)

        # --- 网络特征 ---
        net_features = np.zeros(n_network_features)
        idx_prev = t - 1
        if idx_prev in adj_mats:
            nf = compute_network_features(adj_mats[idx_prev])
            pos = 0
            for key in ["degree_centrality", "pagerank", "clustering"]:
                net_features[pos:pos + K] = nf[key]
                pos += K

        # --- 组装 ---
        x_vec = np.concatenate([lag_block, exog_vec, net_features])
        features.append(x_vec)

        # --- 目标 ---
        targets.append(weights.iloc[t].values)

        valid_indices.append(t)

    X = np.array(features, dtype=np.float64)
    Y = np.array(targets, dtype=np.float64)

    return X, Y, valid_indices


# ===================================================================
# 独立运行入口
# ===================================================================
def main():
    """加载 L2-B 输出，构建特征矩阵，保存供 L3-B 使用。"""
    set_log_file(OUT_DIR / "feature_engineering_log.txt")
    log("=" * 60)
    log("  阶段三-A: 特征工程")
    log("=" * 60)
    log(f"K={K}  P_LAGS={P_LAGS}  NETWORK_THRESHOLD={NETWORK_THRESHOLD}")

    # 加载 L2-B 输出
    weights_path = OUT_DIR / "code" / "输出数据" / "reg_weights_2436.csv"
    adj_path     = OUT_DIR / "code" / "输出数据"

    if not weights_path.exists():
        log("错误: 权重文件不存在，请先运行步骤二")
        return

    log(f"加载权重: {weights_path}")
    weights_df = pd.read_csv(weights_path, index_col=0)
    log(f"  权重矩阵: {weights_df.shape}")

    # 加载日频诊断数据获取日期序列
    daily_path = OUT_DIR / "code" / "输出数据" / "每日的描述性数据.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        date_list = daily["date"].astype(str).tolist()
    else:
        date_list = [str(i) for i in range(len(weights_df))]

    # 外生变量
    exogenous = load_exogenous(date_list)

    # 构建特征
    log("构建特征矩阵...")
    X, Y, valid_idx = build_feature_matrix(weights_df, {}, exogenous, date_list)

    log(f"特征矩阵: X={X.shape}  Y={Y.shape}")
    log(f"有效样本: {len(valid_idx)}/{len(weights_df)}")

    # 保存
    out_dir = OUT_DIR / "code" / "输出数据"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_features.npy", X)
    np.save(out_dir / "Y_targets.npy", Y)
    log(f"已保存: X_features.npy, Y_targets.npy")


if __name__ == "__main__":
    main()
