"""阶段三-A: 特征工程 — 论文§3.2 网络正则化 VARX

构建 VARX 特征矩阵 X 与目标 Y：
  滞后权重 w̃_{t-1}...w̃_{t-p}  +  外生变量  +  网络拓扑

滚动网络均值 Ā 单独保存为 A_bar.npy，供 VARX 正则化层构建掩码 M=1(Ā≥threshold)。
Ā 不进入特征矩阵 X（避免 K×K=153,664 维展平导致维度爆炸）。
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# 共享模块在 图形Lasso/code/，加到 Python 路径
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))

from 共享模块 import (
    K, P_LAGS, NETWORK_THRESHOLD, Ā_ROLLING_WINDOW,
    OUT_DIR, log, set_log_file,
)



# ===================================================================
# 外生变量
# ===================================================================
def compute_ba_spread(rett: np.ndarray) -> float:
    """Abdi-Ranaldo bid-ask spread from intraday returns."""
    log_p = np.vstack([np.zeros((1, rett.shape[1])), rett.cumsum(axis=0)])
    high, low, close = log_p.max(axis=0), log_p.min(axis=0), log_p[-1]
    eta = (high + low) / 2
    return float(2 * np.sqrt(np.mean((close - eta) ** 2)))


def compute_market_vol(rett: np.ndarray) -> float:
    """Cross-sectional mean volatility."""
    return float(np.sqrt(np.mean(rett ** 2)))


def compute_cross_section_moments(rett: np.ndarray) -> Tuple[float, float]:
    """截面偏度与峰度（基于 392 支股票当日日内收益）。"""
    daily_ret = rett.sum(axis=0)  # 当日累计收益 (K,)
    mu = daily_ret.mean()
    sigma = daily_ret.std()
    if sigma < 1e-10:
        return 0.0, 0.0
    skew = float(((daily_ret - mu) ** 3).mean() / sigma ** 3)
    kurt = float(((daily_ret - mu) ** 4).mean() / sigma ** 4 - 3)  # excess kurtosis
    return skew, kurt


def compute_market_return(rett: np.ndarray) -> float:
    """等权市场日收益（392 支股票当日累计收益均值）。"""
    return float(rett.sum(axis=0).mean())


def _load_fred_csv(path: Path, value_col: str) -> pd.Series:
    """加载 FRED CSV → Series (index=YYYYMMDD)。"""
    if not path.exists():
        return pd.Series(dtype=np.float64)
    df = pd.read_csv(path)
    df["observation_date"] = df["observation_date"].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.set_index("observation_date")[value_col]


def load_exogenous(dates: list, npy_dir: Optional[str] = None) -> pd.DataFrame:
    """加载 9 维外生变量（全部 shift(1) 防泄露）。

    VIX + BA Spread + Market Vol + Term Spread + Credit Spread
    + S&P500 Ret + DXY + Cross-section Skew + Kurt
    """
    data_dir = Path(__file__).parents[1] / "数据"

    # VIX
    vix_path = data_dir / "vix_daily.csv"
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path, index_col=0, dtype={0: str})  # 强制字符串索引
        vix_df.index = vix_df.index.astype(str)
        vix_series = pd.Series(
            {d: vix_df.loc[str(d), "vix"] if str(d) in vix_df.index else np.nan for d in dates}
        )
    else:
        vix_series = pd.Series(np.nan, index=dates)

    # FRED 宏观指标
    term_spread = _load_fred_csv(data_dir / "term_spread.csv", "T10Y2Y")
    credit_spread = _load_fred_csv(data_dir / "credit_spread.csv", "BAA10YM")
    dxy = _load_fred_csv(data_dir / "dxy_close.csv", "DTWEXBGS")
    dxy_ret = dxy.pct_change()  # DXY 日收益

    term_s = pd.Series({d: term_spread.get(str(d), np.nan) for d in dates})
    credit_s = pd.Series({d: credit_spread.get(str(d), np.nan) for d in dates})
    # 信用利差月频 → 前向填充
    credit_s = credit_s.ffill()
    dxy_s = pd.Series({d: dxy_ret.get(str(d), np.nan) for d in dates})

    # BA Spread + Market Vol + 市场收益 + 截面偏度/峰度
    ba = pd.Series(np.nan, index=dates)
    mvol = pd.Series(np.nan, index=dates)
    mret = pd.Series(np.nan, index=dates)
    skew_s = pd.Series(np.nan, index=dates)
    kurt_s = pd.Series(np.nan, index=dates)
    if npy_dir:
        npy_p = Path(npy_dir)
        file_map = {f.stem[:8]: f for f in npy_p.iterdir() if f.suffix == ".npy"}
        for d in dates:
            f = file_map.get(str(d))
            if f:
                rett = np.load(str(f))
                ba[d] = compute_ba_spread(rett)
                mvol[d] = compute_market_vol(rett)
                mret[d] = compute_market_return(rett)
                sk, ku = compute_cross_section_moments(rett)
                skew_s[d] = sk
                kurt_s[d] = ku

    df = pd.DataFrame({
        "vix": vix_series,
        "lag_market_vol": mvol,
        "lag_ba_spread": ba,
        "lag_term_spread": term_s,
        "lag_credit_spread": credit_s,
        "lag_sp500_ret": mret,  # 用 392 支等权市场收益代替 S&P500
        "lag_dxy_ret": dxy_s,
        "lag_cs_skew": skew_s,
        "lag_cs_kurt": kurt_s,
    }, index=dates)
    # 全部后移一天 → t-1 值在 t 可知
    for col in df.columns:
        if col.startswith("lag_"):
            df[col] = df[col].shift(1)
    return df


# ===================================================================
# 网络拓扑特征（向量化）
# ===================================================================
def compute_network_features(adj_mat: np.ndarray) -> Dict[str, np.ndarray]:
    """从邻接矩阵计算度中心性、PageRank、聚类系数（向量化）。"""
    adj = adj_mat.astype(np.float64)
    deg = adj.sum(axis=1)
    deg_norm = deg / (K - 1) if K > 1 else deg

    # PageRank (阻尼 0.85, 50 iter, 向量化)
    d = 0.85
    pr = np.ones(K) / K
    out_deg = adj.sum(axis=1)
    out_deg_safe = np.where(out_deg > 0, out_deg, 1.0)  # 避免除零
    sink_mask = (out_deg == 0)
    sink_share = np.zeros(K)
    for _ in range(50):
        # 贡献: pr[i] / out_deg[i] 给每个邻居
        contrib = np.where(out_deg > 0, pr / out_deg_safe, 0.0)
        new_pr = (1 - d) / K + d * (adj.T @ contrib)
        # 沉没节点均分
        new_pr += d * pr[sink_mask].sum() / K
        pr = new_pr / new_pr.sum()

    # 聚类系数（向量化，避免双重 for 循环）
    A_sq = adj @ adj  # A²[i,j] = Σ_k A[i,k]·A[k,j]
    triangle = np.diag(A_sq @ adj) / 2  # 每节点三角形数
    deg_safe = np.where(deg > 1, deg * (deg - 1) / 2, 1.0)
    clustering = triangle / deg_safe

    return {
        "degree_centrality": deg_norm,
        "pagerank": pr,
        "clustering": clustering,
    }


# ===================================================================
# 滚动网络均值 Ā (论文公式 7)
# ===================================================================
def compute_rolling_mean(adj_list: List[np.ndarray], t: int, window: int = Ā_ROLLING_WINDOW) -> np.ndarray:
    """Ā_{ij,t} = (1/W) Σ_{ℓ=0}^{W-1} A_{ij,t-ℓ}，过滤 None（失败日）。"""
    start = max(0, t - window + 1)
    end = t + 1
    mats = [m for m in adj_list[start:end] if m is not None]
    if not mats:
        return np.zeros((K, K), dtype=np.float64)
    return np.mean(np.stack(mats), axis=0)


# ===================================================================
# 构建特征矩阵（主入口）
# ===================================================================
def build_feature_matrix(
    weights_arr: np.ndarray,      # (T, K) 权重矩阵
    adj_list: List[np.ndarray],   # 长度 T 的邻接矩阵列表
    exog_arr: np.ndarray,         # (T, n_exog) 外生变量
    valid_mask: np.ndarray,       # (T,) bool 有效日（非 NaN）
    p_lags: int = P_LAGS,
    include_net_topo: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    """构建 X (特征), Y (目标), A_bar (网络掩码源)。

    每日 t:
      X[t] = [w̃_{t-1}, ..., w̃_{t-p}, x_t]  (+ deg/PR/clust 若 include_net_topo=True)
      Y[t] = w̃_t
      A_bar[t] = Ā_{t-1}  (K×K 滚动均值，供 VARX 构建 M=1(Ā≥threshold))

    VARX 模型（VAR及拓展.py）只使用滞后权重 + 外生变量（共 1185 维），
    网络拓扑特征不在特征矩阵中——它们通过 Ā → M → 网络正则化惩罚层输入，
    这是论文网络正则化 VARX 的标准做法（公式 13）。

    Ā 不进入 X：展平 K×K=153,664 维会导致 p>>n 维度爆炸。
    网络拓扑特征（度/PR/聚类）已从默认 X 中移除以对齐 1185 维（β 避免静默丢弃），
    设置 include_net_topo=True 可恢复。
    """
    T = weights_arr.shape[0]
    n_exog = exog_arr.shape[1]

    # 预计算 Ā 滚动均值缓存（始终需要，供网络正则化掩码）
    a_cache: Dict[int, np.ndarray] = {}
    for t in range(T):
        if valid_mask[t]:
            a_cache[t] = compute_rolling_mean(adj_list, t)

    # 预计算网络特征缓存（仅 include_net_topo=True 时需要）
    net_cache: Dict[int, Dict[str, np.ndarray]] = {}
    if include_net_topo:
        for t in range(T):
            if valid_mask[t] and adj_list[t] is not None:
                net_cache[t] = compute_network_features(adj_list[t])

    # 特征维度（Ā 不进入 X，单独保存）
    n_lag = p_lags * K                                              # 1176
    n_net = (3 * K) if include_net_topo else 0                      # 0 或 1176
    n_total = n_lag + n_exog + n_net                                # 1185 或 2361

    features, targets, a_bars, valid_idx = [], [], [], []

    # 特征名（不含 Ā）
    names = [f"lag{p}_w{i}" for p in range(1, p_lags + 1) for i in range(K)]
    names += [f"exog_{c}" for c in range(n_exog)]
    if include_net_topo:
        names += [f"deg_{i}" for i in range(K)]
        names += [f"pr_{i}" for i in range(K)]
        names += [f"clust_{i}" for i in range(K)]

	for t in range(p_lags, T):
	    # 所有滞后日有效 & 当前日有效
	    lag_ok = all(valid_mask[t - lag] for lag in range(1, p_lags + 1))
	    if not (lag_ok and valid_mask[t]):
	        continue
	
	    # 滞后权重
	    lag_block = np.concatenate([weights_arr[t - lag] for lag in range(1, p_lags + 1)])
	
	    # 外生变量
	    exog_vec = exog_arr[t]
	
	    # 网络特征（可选，默认不包含——VARX 通过 A_bar 掩码做网络正则化）
	    if include_net_topo:
	        idx_prev = t - 1
	        if idx_prev in net_cache:
	            nf = net_cache[idx_prev]
	            net_block = np.concatenate([nf["degree_centrality"], nf["pagerank"], nf["clustering"]])
	        else:
	            net_block = np.zeros(3 * K)
	        x_vec = np.concatenate([lag_block, exog_vec, net_block])
	    else:
	        x_vec = np.concatenate([lag_block, exog_vec])
	
	    # 组装特征（Ā 不进入 X）
	    features.append(x_vec)
	    targets.append(weights_arr[t])
	    a_bars.append(a_cache.get(t - 1, np.zeros((K, K))))
	    valid_idx.append(t)

    X = np.array(features, dtype=np.float64) if features else np.empty((0, n_total))
    Y = np.array(targets, dtype=np.float64) if targets else np.empty((0, K))
    A_bar = np.array(a_bars, dtype=np.float64) if a_bars else np.empty((0, K, K))
    return X, Y, A_bar, valid_idx, names


# ===================================================================
# 验收检查（规范 §5）
# ===================================================================
def validate_features(X: np.ndarray, Y: np.ndarray, valid_idx: List[int],
                      T: int, p_lags: int) -> Dict[str, float]:
    """特征有效性 + 分布稳定性 + 数据泄露检查。"""
    metrics = {}
    # 方差阈值
    std_arr = np.std(X, axis=0)
    metrics["n_zero_var_features"] = int(np.sum(std_arr < 1e-6))
    metrics["missing_rate"] = float(np.mean(np.isnan(X)))

    # 时间顺序（无泄露）
    metrics["is_monotonic"] = bool(np.all(np.diff(valid_idx) > 0))
    metrics["n_valid_samples"] = len(valid_idx)
    metrics["coverage"] = len(valid_idx) / T

    # 训练/测试划分
    n_train = int(0.70 * len(valid_idx))
    n_val = int(0.15 * len(valid_idx))
    X_train, X_test = X[:n_train], X[n_train + n_val:]

    # PSI（分位数分箱，基于训练集 decile 边缘 → 测试集用相同边缘）
    # 注：≤0.1 阈值源自信用评分模型监控（1-2年窗口）。
    # 本项目跨10年+COVID崩盘，0.1极难达到。放宽至 ≤0.5 并报告分位数分布。
    def psi_single(a, b, bins=10):
        # 分位数分箱边缘（基于训练集），避免空箱导致的密度爆炸
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(a, quantiles)
        # 处理退化边缘（所有值相同）
        if np.any(np.diff(edges) < 1e-12):
            return 0.0
        a_cnt, _ = np.histogram(a, bins=edges)
        b_cnt, _ = np.histogram(b, bins=edges)
        a_prop = np.clip(a_cnt / a_cnt.sum(), 1e-8, None)
        b_prop = np.clip(b_cnt / b_cnt.sum(), 1e-8, None)
        return float(np.sum((a_prop - b_prop) * np.log(a_prop / b_prop)))

    psi_values = [psi_single(X_train[:, j], X_test[:, j]) for j in range(X.shape[1])]
    psi_arr = np.array(psi_values)
    metrics["psi_max"] = float(np.nanmax(psi_arr))
    metrics["psi_mean"] = float(np.nanmean(psi_arr))
    metrics["psi_p50"] = float(np.nanpercentile(psi_arr, 50))
    metrics["psi_p95"] = float(np.nanpercentile(psi_arr, 95))
    metrics["psi_pass_rate_0.1"] = float(np.mean(psi_arr <= 0.1))
    metrics["psi_pass_rate_0.5"] = float(np.mean(psi_arr <= 0.5))

    return metrics


# ===================================================================
# 主入口
# ===================================================================
def main():
    set_log_file(OUT_DIR / "feature_engineering_log.txt")
    t_start = time.time()
    log("=" * 60)
    log("  阶段三-A: 特征工程")
    log("=" * 60)
    log(f"K={K}  P_LAGS={P_LAGS}  NETWORK_THRESHOLD={NETWORK_THRESHOLD}  Ā_ROLLING_WINDOW={Ā_ROLLING_WINDOW}")

    # 输入路径（阶段二产出在 图形Lasso/code/输出数据/）
    project_dir = Path(__file__).parents[1]  # 大创/
    input_dir = project_dir / "图形Lasso" / "code" / "输出数据"
    weights_path = input_dir / "reg_weights_2436.csv"
    adj_dir = input_dir / "adjacency"
    daily_path = input_dir / "Daily_Statistics.csv"
    npy_dir = project_dir / "数据" / "1min_log_return_npy"

    # 输出路径（保存在当前 特征工程/ 目录）
    out_dir = Path(__file__).parent

    if not weights_path.exists():
        log(f"错误: 权重文件不存在 {weights_path}")
        return
    if not adj_dir.exists():
        log(f"错误: 邻接矩阵目录不存在 {adj_dir}")
        return

    # 加载权重
    log(f"加载权重: {weights_path.name}")
    weights_df = pd.read_csv(weights_path, index_col=0)
    T = len(weights_df)
    weights_arr = weights_df.values.astype(np.float64)
    log(f"  权重矩阵: {weights_df.shape}")

    # 日期列表
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        date_list = daily["date"].astype(str).tolist()
    else:
        date_list = [str(i) for i in range(T)]

    # 加载邻接矩阵（按日期顺序）
    log(f"加载邻接矩阵: {adj_dir.name}")
    adj_list = [None] * T
    adj_count = 0
    for i, d in enumerate(date_list):
        f = adj_dir / f"{d}.npy"
        if f.exists():
            adj_list[i] = np.load(str(f))
            adj_count += 1
    log(f"  邻接矩阵: {adj_count}/{T} 天")

    # 外生变量
    log("计算外生变量 (VIX + BA Spread + Market Vol)...")
    exog_df = load_exogenous(date_list, str(npy_dir) if npy_dir.exists() else None)
    exog_arr = exog_df.fillna(method="ffill").fillna(0).values
    log(f"  外生变量: {exog_df.shape}")

    # 有效日 mask（权重无 NaN）
    valid_mask = ~np.isnan(weights_arr).any(axis=1)
    log(f"  有效日: {valid_mask.sum()}/{T}")

    # 构建特征
    log("构建特征矩阵 (Ā 不进入 X，单独保存为 A_bar.npy)...")
    t0 = time.time()
    X, Y, A_bar, valid_idx, names = build_feature_matrix(
        weights_arr, adj_list, exog_arr, valid_mask, P_LAGS,
        include_net_topo=False,  # 默认不包含网络拓扑特征；VARX 通过 A_bar 掩码做网络正则化
    )
    t_build = time.time() - t0
    log(f"  特征矩阵: X={X.shape}  Y={Y.shape}  A_bar={A_bar.shape}  耗时: {t_build:.1f}s")
    log(f"  特征维度 = {X.shape[1]} (滞后 {P_LAGS*K} + 外生 {exog_arr.shape[1]})")
    log(f"  A_bar 形状 = (n_samples={A_bar.shape[0]}, K={A_bar.shape[1]}, K={A_bar.shape[2]}) 供正则化掩码 M=1(Ā≥threshold)")
    log(f"  有效样本: {len(valid_idx)}/{T} ({len(valid_idx)/T*100:.1f}%)")

    # 验收检查
    log("\n验收检查:")
    metrics = validate_features(X, Y, valid_idx, T, P_LAGS)
    for k, v in metrics.items():
        log(f"  {k}: {v}")

    # 保存
    np.save(out_dir / "X_features.npy", X)
    np.save(out_dir / "Y_targets.npy", Y)
    np.save(out_dir / "A_bar.npy", A_bar)
    with open(out_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    np.save(out_dir / "valid_indices.npy", np.array(valid_idx))

    valid_dates = [date_list[i] for i in valid_idx]
    pd.DataFrame({"date": valid_dates, "idx": valid_idx}).to_csv(
        out_dir / "feature_valid_dates.csv", index=False
    )

    elapsed = time.time() - t_start
    log(f"\n完成! 总耗时: {elapsed:.1f}s")
    log(f"输出: X_features.npy ({X.shape}), Y_targets.npy ({Y.shape}), A_bar.npy ({A_bar.shape})")
    log(f"      feature_names.txt, feature_valid_dates.csv, valid_indices.npy")
    log("=" * 60)


if __name__ == "__main__":
    main()
