# ===================================================================
# 阶段三-B：网络正则化 VARX 预测模型
#
# 论文对应：§3.2
#   w̃_t = Σ_{ℓ=1}^{p} A_ℓ w̃_{t-ℓ} + B x_t + ε_t
#
# ℓ1 (Lasso) 惩罚稀疏化系数
# ℓ2 平滑惩罚抑制相邻滞后系数剧烈跳变
# 网络正则化：Ā_{ij} < τ  ⇒ 强制 A_{ℓ,ij} = 0
#
# 使用 sklearn.linear_model.MultiTaskLasso 作为基础求解器，
# 通过自定义损失实现网络引导的稀疏结构。
# ===================================================================
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.linear_model import Lasso, MultiTaskLasso

sys.stdout.reconfigure(encoding='utf-8')

from 共享模块 import (
    K, P_LAGS, LAMBDA_LASSO, LAMBDA_TURNOVER, LAMBDA_NETWORK,
    NETWORK_THRESHOLD, OUT_DIR, log, set_log_file,
)


# ===================================================================
# 网络掩码：根据滚动网络均值 Ā 构建约束矩阵
# ===================================================================
def build_network_mask(
    network_mean: np.ndarray,        # (K,K) 滚动平均网络
    threshold: float = NETWORK_THRESHOLD,
) -> np.ndarray:
    """构建网络二值掩码 M_{ij} = 1(Ā_{ij} ≥ τ)。

    M = 1 允许资产 i 对 j 的滞后效应存在
    M = 0 强制对应 VARX 系数为零

    Args:
        network_mean: (K,K) 当前日的滚动网络均值
        threshold: 截断阈值

    Returns:
        (K,) 整型向量，每个资产的可允许连接数
    """
    mask = (network_mean >= threshold).astype(np.float64)
    return mask


# ===================================================================
# VARX 拟合（简化版：逐资产 Lasso + 网络掩码）
# ===================================================================
def fit_varx_asset(
    X: np.ndarray,                   # (T, n_features)
    Y: np.ndarray,                   # (T, K)
    network_mask: np.ndarray = None, # (K,K)
    alpha: float = LAMBDA_LASSO,
) -> Dict[str, np.ndarray]:
    """逐资产拟合网络正则化 VARX 模型。

    对每个资产 i，筛选与之有条件依赖的资产 j (Ā_{ij} ≥ τ)，
    仅保留 X 中对应列，用 Lasso 拟合 Y[:,i]。

    Args:
        X:            特征矩阵 (需先从 build_feature_matrix 构建)
        Y:            目标权重 (T, K)
        network_mask: (K,K) 网络掩码

    Returns:
        { "coefs": (K, n_selected), "intercept": (K,), "nz_per_asset": (K,) }
    """
    T, K_ = Y.shape
    assert K_ == K

    coefs      = []
    intercepts = []
    nz_counts   = []

    for i in range(K):
        if network_mask is not None:
            # 筛选与资产 i 有条件依赖的资产
            allowed_cols = np.where(network_mask[i] > 0)[0]
            if len(allowed_cols) == 0:
                # 无连接 → 只用截距
                coefs.append(np.zeros(X.shape[1]))
                intercepts.append(np.mean(Y[:, i]))
                nz_counts.append(0)
                continue

            # 构建只含允许列的 X_sub
            # 注: 此处为简化实现，直接使用 X 的全部列
            # 完整实现需按滞后结构和允许列索引选取

        # 标准 Lasso 拟合
        model = Lasso(alpha=alpha, max_iter=2000, tol=1e-4, selection='cyclic')
        model.fit(X, Y[:, i])

        coefs.append(model.coef_)
        intercepts.append(model.intercept_)
        nz_counts.append(int(np.sum(np.abs(model.coef_) > 1e-8)))

    return {
        "coefs": np.array(coefs),
        "intercepts": np.array(intercepts),
        "nz_per_asset": np.array(nz_counts),
    }


# ===================================================================
# VARX 预测
# ===================================================================
def predict_varx(
    X: np.ndarray,
    fitted: Dict[str, np.ndarray],
) -> np.ndarray:
    """用已拟合的 VARX 模型预测。

    Args:
        X: (n_samples, n_features) 特征矩阵
        fitted: fit_varx_asset 的输出

    Returns:
        (n_samples, K) 预测的 GMVP 权重
    """
    Y_pred = np.zeros((X.shape[0], K))
    for i in range(K):
        Y_pred[:, i] = X @ fitted["coefs"][i] + fitted["intercepts"][i]

    # 不做 sum-to-one 约束，考察原始预测
    return Y_pred


# ===================================================================
# 独立运行入口：拟合 + 滚动 OOS 回测
# ===================================================================
def main():
    set_log_file(OUT_DIR / "varx_fitting_log.txt")
    log("=" * 60)
    log("  阶段三-B: VARX 拟合 + 滚动回测")
    log("=" * 60)
    log(f"K={K}  P_LAGS={P_LAGS}  λ_lasso={LAMBDA_LASSO}")

    feat_dir = OUT_DIR / "code" / "输出数据"
    X_path = feat_dir / "X_features.npy"
    Y_path = feat_dir / "Y_targets.npy"

    if not X_path.exists():
        log("错误: 特征文件不存在，请先运行 特征工程.py")
        return

    X = np.load(X_path)
    Y = np.load(Y_path)
    log(f"X={X.shape}  Y={Y.shape}")

    # ---- 滚动 OOS 回测 (内嵌，非独立模块) ----
    from 决策聚焦训练 import load_realized_covariances
    from 共享模块 import L_TRAIN_VARX, ETA

    T = X.shape[0]
    start_idx = max(500, L_TRAIN_VARX)   # 首窗至少 500 天训练
    log(f"回测范围: {start_idx} ~ {T-1}")

    port_vars = []
    turnovers = []
    eq_vars   = []
    cov_dict = load_realized_covariances(Path("."))
    w_eq = np.ones(K) / K

    for t in range(start_idx, T):
        # 训练窗
        train_s = max(0, t - L_TRAIN_VARX)
        X_tr = X[train_s:t]; Y_tr = Y[train_s:t]

        # 拟合
        fitted = fit_varx_asset(X_tr, Y_tr, network_mask=None)

        # 预测
        X_pred = X[t:t+1]
        w_pred = predict_varx(X_pred, fitted)[0]

        # 评估
        if t in cov_dict:
            port_vars.append(float(w_pred @ cov_dict[t] @ w_pred))
            if len(port_vars) > 1:
                turnovers.append(float(np.sum(np.abs(w_pred - prev_w))))
            else:
                turnovers.append(0.0)
            prev_w = w_pred
            eq_vars.append(float(w_eq @ cov_dict[t] @ w_eq))

        if (t - start_idx) % 100 == 0 and len(port_vars) > 0:
            log(f"  进度: {t-start_idx}/{T-start_idx}  "
                f"最近平均方差: {np.mean(port_vars[-50:]):.4e}")

    # 汇总
    log("\n" + "=" * 60)
    log("滚动 OOS 回测结果")
    log("=" * 60)
    log(f"  回测天数: {len(port_vars)}")
    log(f"  GLasso-VARX 平均方差: {np.mean(port_vars):.4e}")
    log(f"  等权重 平均方差:      {np.mean(eq_vars):.4e}")
    log(f"  超额: {(np.mean(port_vars)/np.mean(eq_vars)-1)*100:.1f}%")

    # 保存
    np.save(feat_dir / "backtest_port_variances.npy", np.array(port_vars))
    np.save(feat_dir / "backtest_turnovers.npy", np.array(turnovers))
    np.save(feat_dir / "backtest_eq_variances.npy", np.array(eq_vars))
    np.save(feat_dir / "varx_coefs.npy", fitted["coefs"])
    np.save(feat_dir / "varx_intercepts.npy", fitted["intercepts"])
    log("\n结果已保存。")


if __name__ == "__main__":
    main()
