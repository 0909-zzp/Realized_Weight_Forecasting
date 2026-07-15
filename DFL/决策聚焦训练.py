# ===================================================================
# 阶段三-C：决策聚焦训练
#
# 论文对应：§3.3 Decision-Focused Loss
#   L(w̃_pred, w̃_{t-1}, Σ̂_realized_t) =
#       w̃_predᵀ · Σ̂_realized_t · w̃_pred           ← 预测组合方差
#     + η · ‖w̃_pred - w̃_{t-1}‖₁                   ← 交易成本
#     + α · ‖β‖₁ + γ · ‖β‖₂²                       ← 系数正则化
#
# 与标准 MSE 损失不同：决策聚焦损失直接最小化预测权重对应的
# 已实现组合方差，而非预测权重与实际权重的偏差。
# ===================================================================
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional

sys.stdout.reconfigure(encoding='utf-8')

# 共享模块路径
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import (
    K, P_LAGS, ETA, LAMBDA_LASSO, OUT_DIR, log, set_log_file,
)


# ===================================================================
# 已实现协方差加载
# ===================================================================
def load_realized_covariances(
    daily_path: Path,
) -> Dict[int, np.ndarray]:
    """加载已实现协方差矩阵序列。

    从步骤二输出中读取逐日的 realized covariance。
    若未直接保存，则从原始数据重新计算。

    Returns:
        {day_idx: (K,K) 已实现协方差}
    """
    from 共享模块 import DATA_DIR, load_day, compute_raw_cov

    log("从原始数据加载协方差...")
    covs = {}
    # 遍历全部交易日
    from 共享模块 import _ensure_file_list, _files_raw
    _ensure_file_list()

    for idx in range(len(_files_raw)):
        rett = load_day(idx)
        covs[idx] = compute_raw_cov(rett)

    return covs


# ===================================================================
# 决策聚焦损失函数
# ===================================================================
def decision_focused_loss(
    w_pred: np.ndarray,           # (K,) 预测权重
    w_prev: np.ndarray,           # (K,) 上期实际权重
    realized_cov: np.ndarray,     # (K,K) 已实现协方差
    eta: float = ETA,
) -> float:
    """计算决策聚焦损失。

    Args:
        w_pred:       预测的 GMVP 权重向量
        w_prev:       上一期的真实权重
        realized_cov: 当日的已实现协方差
        eta:          交易成本系数

    Returns:
        标称损失值
    """
    # 组合方差
    port_var = float(w_pred @ realized_cov @ w_pred)

    # 交易成本 (换手率)
    turnover = float(np.sum(np.abs(w_pred - w_prev)))

    return port_var + eta * turnover


# ===================================================================
# 组合后评估
# ===================================================================
def evaluate_predictions(
    w_pred_seq: np.ndarray,        # (T, K) 预测权重序列
    w_actual_seq: np.ndarray,      # (T, K) 实际权重序列
    cov_dict: Dict[int, np.ndarray],
    start_idx: int,
    eta: float = ETA,
) -> dict:
    """对预测权重序列计算决策聚焦评估指标。

    Returns:
        dict with avg_var, avg_turnover, avg_loss, total_loss
    """
    T = w_pred_seq.shape[0]
    losses    = []
    variances = []
    turnovers = []

    for t in range(T):
        real_idx = start_idx + t
        if real_idx not in cov_dict:
            continue

        w_p = w_pred_seq[t]
        w_a = w_actual_seq[t] if w_actual_seq is not None else w_p
        cov = cov_dict[real_idx]

        var = float(w_p @ cov @ w_p)
        to  = float(np.sum(np.abs(w_p - w_a))) if t > 0 else 0.0

        variances.append(var)
        turnovers.append(to)
        losses.append(var + eta * to)

    return {
        "avg_variance":  np.mean(variances),
        "avg_turnover":  np.mean(turnovers),
        "avg_loss":      np.mean(losses),
        "total_loss":    np.sum(losses),
        "variances":     np.array(variances),
        "turnovers":     np.array(turnovers),
    }


# ===================================================================
# 独立运行入口
# ===================================================================
def main():
    set_log_file(OUT_DIR / "decision_focused_log.txt")
    log("=" * 60)
    log("  阶段三-C: 决策聚焦训练")
    log("=" * 60)
    log(f"K={K}  ETA={ETA}")

    # ---- 加载 VARX 输出（从 VARX/ 目录） ----
    varx_dir = Path(__file__).parents[1] / "VARX"
    feat_dir = Path(__file__).parents[1] / "特征工程"

    # 检查 VARX 系数文件是否存在（来自 VARX/fitted_models/ 目录）
    coefs_path = varx_dir / "fitted_models" / "coefs_model4.npy"
    scaler_path = varx_dir / "fitted_models" / "scaler_model4.pkl"
    cols_path = varx_dir / "fitted_models" / "feat_cols_model4.npy"
    intercepts_path = varx_dir / "fitted_models" / "intercepts_model4.npy"

    if not coefs_path.exists():
        log(f"错误: VARX 系数文件不存在 ({coefs_path})")
        log("请先运行 VARX/VAR及拓展.py 的 main() 生成系数文件")
        return

    coefs      = np.load(coefs_path)                   # (K, n_feat_sel)
    intercepts = np.load(intercepts_path)               # (K,)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    feat_cols  = np.load(cols_path)                     # 选择的特征列索引

    # 加载特征数据
    X_all = np.load(feat_dir / "X_features.npy")       # (n_samples, n_total_feat)
    Y_all = np.load(feat_dir / "Y_targets.npy")        # (n_samples, K)

    log(f"VARX 系数: coefs={coefs.shape}  intercepts={intercepts.shape}")
    log(f"特征数据: X={X_all.shape}  Y={Y_all.shape}  使用列={feat_cols.shape}")

    # 用全部样本做一次全量预测（使用 M4 模型）
    log("全量预测 (Model 4 Network VARX)...")
    X_sel = X_all[:, feat_cols]
    X_std = scaler.transform(X_sel)
    Y_pred = X_std @ coefs.T + intercepts              # (n_samples, K)

    # 归一化（sum-to-one）
    s = Y_pred.sum(axis=1, keepdims=True)
    s = np.where(np.abs(s) < 1e-10, 1.0, s)
    Y_pred = Y_pred / s

    # ---- 加载已实现协方差 ----
    cov_dict = load_realized_covariances(Path("."))

    # ---- 决策聚焦评估 ----
    # 跳过初始期（与特征工程的 lag 效应对齐）
    start_idx = P_LAGS   # 特征工程从第 P_LAGS 个有效日开始
    log(f"评估起始样本索引: {start_idx}")
    metrics = evaluate_predictions(
        Y_pred[start_idx:],
        Y_all[start_idx:],
        cov_dict,
        start_idx,
    )

    log("\n" + "-" * 50)
    log("决策聚焦评估结果 (Model 4 Network VARX)")
    log("-" * 50)
    log(f"  平均组合方差:  {metrics['avg_variance']:.4e}")
    log(f"  平均换手率:    {metrics['avg_turnover']:.4f}")
    log(f"  平均决策损失:  {metrics['avg_loss']:.4e}")
    log(f"  总决策损失:    {metrics['total_loss']:.4e}")

    # ---- 等权重 benchmark ----
    w_eq = np.ones(K) / K
    eq_vars = []
    for t in range(start_idx, len(Y_all)):
        real_idx = start_idx + (t - start_idx)
        if real_idx in cov_dict:
            eq_vars.append(float(w_eq @ cov_dict[real_idx] @ w_eq))
    eq_avg = np.mean(eq_vars) if eq_vars else 0
    log(f"\n  等权重 benchmark 方差: {eq_avg:.4e}")
    log(f"  GLasso-VARX 超额: {(metrics['avg_variance'] / eq_avg - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
