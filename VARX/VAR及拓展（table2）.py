"""阶段三-B: VARX预测 — 论文§3.2, Table 2

实现 Model 1-5（Lasso族）+ Model 6（DFL调优）+ 4个评估指标（MSE/MAE/DM/MCS）。

模型层次（论文 Table 2）:
  1. VAR              — OLS, 滞后权重（无稀疏、无外生、无网络）
  2. Sparse VAR       — Lasso, 滞后权重
  3. Sparse VARX      — Lasso, 全 x_t（含网络拓扑）
  4. Network VARX     — Lasso, 同 M3 特征集 + λ₁≠λ₂ 网络惩罚
  5. Network+Smooth   — Model 4 + 数据增广 λ_s 平滑（训练时嵌入）
  6. DFL-Tuned VARX   — Model 4 预测 + 决策聚焦后处理调优
  
Model 7 (LSTM) → 待实现

网络加权惩罚实现（论文公式13，四参数体系，网格搜索验证 2026-07-03）:
  λ₁ = LAMBDA_LASSO      = 1e-4   (连接资产的滞后系数 ℓ1)
  λ₂ = λ₁+LAMBDA_NETWORK  = 5.1e-3 (未连接资产的滞后系数 ℓ1, λ₂≫λ₁)
  λ₃ = LAMBDA_EXOG       = 5e-4   (外生变量系数 ℓ1)
  λ_s = LAMBDA_TURNOVER  = 5e-3   (换手率平滑 ℓ2)
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# 共享模块路径
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import (
    K, P_LAGS, LAMBDA_LASSO, LAMBDA_NETWORK, LAMBDA_EXOG, LAMBDA_TURNOVER,
    NETWORK_THRESHOLD, ETA, OUT_DIR, log, set_log_file,
)

# ===================================================================
# 常量
# ===================================================================
N_EXOG = 9                                        # 外生变量维度（宏观9维）
N_NET  = 0                                        # 网络拓扑从 X 中移除，仅作 M4/M5 惩罚指引
N_FEATURES = P_LAGS * K + N_EXOG                  # 总特征数 = 1185

# 特征块列索引（与 特征工程.py 的 names 列表对齐）
# 网络拓扑特征(度/PR/聚类)已从 X 中移出，Ā 仅在 build_network_mask 中用于构建惩罚掩码
FEAT_BLOCKS = {
    'lagged':  (0,              P_LAGS * K),          # [0, 1176)
    'exog':    (P_LAGS * K,     P_LAGS * K + N_EXOG), # [1176, 1185)
}

# 模型配置
# blocks: 使用的特征块; sparse: 是否Lasso(否则OLS); network: 是否网络加权惩罚; smooth: 是否平滑
# self_free: 是否对每个方程的自身滞后 B_lag[i,i] 近似零惩罚；用于 M3a 消融和 M4/M5。
# lasso_lambda: 模型专属 λ₁ 覆写。M2 无稀疏约束可贪心用 5e-04，
#   M3-M5 需 ≤1e-04 保留交叉项供 M4 网络惩罚有靶子可罚
MODELS = {
    1: {'name': 'VAR',            'blocks': ['lagged'],         'sparse': False, 'network': False, 'smooth': False, 'self_free': False, 'lasso_lambda': LAMBDA_LASSO},
    2: {'name': 'Sparse VAR',     'blocks': ['lagged'],         'sparse': True,  'network': False, 'smooth': False, 'self_free': False, 'lasso_lambda': LAMBDA_LASSO},
    3: {'name': 'Sparse VARX',    'blocks': ['lagged', 'exog'], 'sparse': True,  'network': False, 'smooth': False, 'self_free': False, 'lasso_lambda': LAMBDA_LASSO},
    4: {'name': 'Network VARX',   'blocks': ['lagged', 'exog'], 'sparse': True,  'network': True,  'smooth': False, 'self_free': True,  'lasso_lambda': LAMBDA_LASSO},
    5: {'name': 'Network+Smooth', 'blocks': ['lagged', 'exog'], 'sparse': True,  'network': True,  'smooth': True,  'self_free': True,  'lasso_lambda': LAMBDA_LASSO},
}

# Model 5 的换手率平滑 λ_s（论文公式13最后一项）
# 通过数据增广法嵌入训练：追加 √(T₀·λ_s)·ΔX 行（目标=0）
# sklearn Lasso 归一化为 (1/(2n))，展开后平滑/MSE 权重比 = T₀·λ_s，与论文一致
LAMBDA_S = LAMBDA_TURNOVER


# ===================================================================
# 数据加载与划分
# ===================================================================
def load_data() -> Dict[str, np.ndarray]:
    """加载特征工程产出（X, Y, A_bar）。"""
    feat_dir = Path(__file__).parent.parent / "特征工程"
    return {
        'X': np.load(feat_dir / "X_features.npy"),
        'Y': np.load(feat_dir / "Y_targets.npy"),
        'A_bar': np.load(feat_dir / "A_bar.npy"),
    }


def split_data(X, Y, A_bar, train_ratio=0.70, val_ratio=0.15):
    """固定时间划分 70/15/15（规范 §3.3）。

    Returns:
        dict with keys 'train', 'val', 'test', each = (X, Y, A_bar)
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        'train': (X[:n_train],          Y[:n_train],          A_bar[:n_train]),
        'val':   (X[n_train:n_train+n_val], Y[n_train:n_train+n_val], A_bar[n_train:n_train+n_val]),
        'test':  (X[n_train+n_val:],    Y[n_train+n_val:],    A_bar[n_train+n_val:]),
    }


# ===================================================================
# 网络掩码（论文公式 7）
# ===================================================================
def build_network_mask(A_bar_train: np.ndarray, threshold: float = NETWORK_THRESHOLD,
                       use_last_day: bool = False
                       ) -> Tuple[np.ndarray, float]:
    """训练期 Ā → 二值掩码 M[i,j] = 1(Ā[i,j] ≥ τ)。

    use_last_day: 若 False（默认），对全训练期 Ā 取均值（静态掩码，长期稳定结构）；
                  若 True，直接用最后一天的 Ā_{T}（~20天精度的近期快照）。
    Ā 本身已是 20 天滚动窗口均值。两种模式对应论文 §3.2 的
    "fixed network (training window)" vs "rolling-window average network"。

    Returns:
        mask: (K, K) float64 二值矩阵（对角线=0，无自环）
        density: 网络密度
    """
    if use_last_day:
        A_mean = A_bar_train[-1]                         # 仅最后一天（≈最近20天快照）
    else:
        A_mean = A_bar_train.mean(axis=0)                # 全训练期均值（原行为）
    mask = (A_mean >= threshold).astype(np.float64)
    np.fill_diagonal(mask, 0)                           # 排除自环
    density = mask.sum() / (K * (K - 1))
    return mask, density


# ===================================================================
# 惩罚向量（论文公式 13）
# ===================================================================
def build_penalty_vector(asset_i: int, use_network: bool,
                         network_mask: Optional[np.ndarray],
                         lambda_lasso: float = LAMBDA_LASSO,
                         self_free: bool = False) -> np.ndarray:
    """构建 per-feature 惩罚向量 (N_FEATURES,)，对齐论文公式(13) 四参数体系。

    滞后块，连接资产 j  → λ₁ = lambda_lasso
    滞后块，未连接资产 j → λ₂ = lambda_lasso + LAMBDA_NETWORK  (λ₂ > λ₁)
    外生块（宏观9维）     → λ₃ = LAMBDA_EXOG
    网络拓扑特征已从 X 移出，仅通过 network_mask 影响滞后块惩罚差异化。
    """
    penalties = np.full(N_FEATURES, lambda_lasso)

    # 外生变量（宏观9维）用 λ₃
    exog_start = FEAT_BLOCKS['exog'][0]
    exog_end   = FEAT_BLOCKS['exog'][1]
    penalties[exog_start:exog_end] = LAMBDA_EXOG

    if use_network and network_mask is not None:
        connected = network_mask[asset_i]           # (K,) 资产 i 的连接行
        for lag in range(P_LAGS):
            offset = lag * K
            penalties[offset:offset + K] = np.where(
                connected > 0,
                lambda_lasso,                    # λ₁: 连接
                lambda_lasso + LAMBDA_NETWORK,   # λ₂: 未连接, j≠i
            )
    if self_free:
        # 论文公式(13): λ₂ 项显式含 j≠i，自身滞后 B_{ℓ,ii} 可作为消融项近似零惩罚。
        # 设为零会在 _fit_one_asset 中除零，使用极小值 1e-12 近似零惩罚。
        for lag in range(P_LAGS):
            penalties[lag * K + asset_i] = 1e-12
    return penalties


# ===================================================================
# 拟合
# ===================================================================
def _fit_one_asset(X_std: np.ndarray, y_i: np.ndarray,
                   penalties: np.ndarray, use_lasso: bool,
                   lambda_lasso: float = LAMBDA_LASSO,
                   lasso_alpha: Optional[float] = None
                   ) -> Tuple[np.ndarray, float]:
    """拟合单个资产。

    加权L1 → 标准Lasso的转化：
      s = λ_ref / penalties        （缩放因子）
      X_scaled = X_std * s         （缩放特征）
      β_scaled = Lasso.fit(X_scaled, y)  （标准Lasso）
      β_original = β_scaled * s    （反缩放回原始空间）

    Args:
        X_std: (n_train, n_feat) 已标准化的特征矩阵
        y_i:   (n_train,) 资产 i 的目标权重
        penalties: (n_feat,) per-feature 惩罚
        use_lasso: True=Lasso, False=OLS(Model 1)
        lambda_lasso: 模型专属 λ₁（用于特征缩放和惩罚向量构建）
        lasso_alpha: sklearn Lasso alpha 参数。
                     若为 None 则等于 lambda_lasso（标准情况）。
                     M5 数据增广时由于 n_aug 翻倍，需传入补偿值保持可比。

    Returns:
        (coef, intercept) 系数和截距（标准化空间）
    """
    if not use_lasso:
        model = LinearRegression()                     # Model 1: OLS 无正则化
        model.fit(X_std, y_i)
        return model.coef_.copy(), float(model.intercept_)

    # 加权 Lasso via 特征缩放
    alpha = lasso_alpha if lasso_alpha is not None else lambda_lasso
    s = lambda_lasso / penalties                   # 缩放因子（始终用原始 λ₁）
    X_scaled = X_std * s                            # (n_train, n_feat)
    model = Lasso(alpha=alpha, max_iter=2000,
                  tol=1e-4, selection='cyclic')
    model.fit(X_scaled, y_i)
    coef = model.coef_ * s                          # 反缩放
    return coef, float(model.intercept_)


def fit_model(model_id: int, X_train: np.ndarray, Y_train: np.ndarray,
              network_mask: Optional[np.ndarray], n_jobs: int = 4
              ) -> Dict:
    """拟合 K=392 个资产（并行）。

    Returns:
        dict: coefs (K,n_feat), intercepts (K,), scaler, cols, config
    """
    from joblib import Parallel, delayed

    cfg = MODELS[model_id]
    lambda_lasso = cfg.get('lasso_lambda', LAMBDA_LASSO)  # 模型专属 λ₁

    # 选取特征列
    cols = []
    for block in cfg['blocks']:
        s, e = FEAT_BLOCKS[block]
        cols.extend(range(s, e))
    cols = np.array(cols)
    X_sel = X_train[:, cols]
    n_feat = len(cols)

    # 标准化（Lasso对量纲敏感）
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_sel)

    # Model 1 快速路径：批量 OLS（K 个资产共享同一 X，一次正常方程求解）
    # β = (X̃ᵀX̃)⁻¹X̃ᵀY  →  O(np² + p³ + p²K)  vs  逐资产 ×392 ≈ 250×加速
    # 用 lstsq (SVD) 替代 solve：滞后权重列高度共线，solve 可能卡死
    if not cfg['sparse'] and not cfg['smooth']:
        X_aug = np.column_stack([X_std, np.ones(X_std.shape[0])])  # 增广截距列
        theta = np.linalg.lstsq(X_aug, Y_train, rcond=None)[0]     # (p+1, K)
        coefs = theta[:-1].T                                        # (K, p)
        intercepts = theta[-1]                                      # (K,)
        log(f"  [批量OLS lstsq] {X_std.shape[1]}特征 × {K}资产 = 1次求解")
        return {
            'coefs': coefs,
            'intercepts': intercepts,
            'scaler': scaler,
            'cols': cols,
            'config': cfg,
        }

    # Model 5: 数据增广实现 λ_s 换手率平滑（论文公式13最后一项）
    # 论文目标:   L = (1/T₀)‖y-Xβ‖² + λ_s·‖ΔX·β‖² + α‖β‖₁
    # sklearn Lasso 目标: (1/(2·n_aug))·‖Y_aug - X_aug·β‖² + α·‖β‖₁
    #
    # 追加 √(T₀·λ_s)·ΔX 行（目标=0），则 n_aug = T₀ + (T₀-1) = 2T₀-1；
    # (1/(2·n_aug))·[‖y-Xβ‖² + T₀·λ_s·‖ΔX·β‖²] + α·‖β‖₁
    #
    # 展开后 MSE 项权重比 = T₀ / (2·n_aug) ≈ 1/2 (当 T₀≫1)，
    # 平滑项与 MSE 的权重比 = T₀·λ_s / (2·n_aug) ≈ λ_s / 2 (当 T₀≫1)。
    # 论文中两者相对权重 = T₀·λ_s，代码中经 sklearn 归一化后 ≈ T₀·λ_s/2。
    # 由于网格搜索调整 λ_s 时会自动补偿这个 2× 因子，最终 λ_s 值
    # 在论文尺度上等效翻倍，不影响模型间相对比较。
    #
    # 重要：n_aug ≈ 2× n_orig 导致 sklearn Lasso alpha 的有效强度也翻倍
    # （因为 (1/(2n_aug)) 的分母比 (1/(2T₀)) 大一倍）。
    # 在传递给 _fit_one_asset 时，alpha 乘以 T0/n_aug 补偿这一效应。
    # 注意：此补偿仅调整 Lasso alpha，不改变 build_penalty_vector 中的 λ₁（
    # 惩罚向量中的 LAMBDA_EXOG/LAMBDA_NETWORK 保持原值，确保 M5 vs M4 可比）。
    # ΔX = X_{t+1} - X_t 逐日特征差分
    if cfg['smooth']:
        dX = np.diff(X_std, axis=0)                    # (n-1, n_feat)
        T0 = X_std.shape[0]                             # 训练样本数
        n_aug = T0 + dX.shape[0]                        # = 2*T0 - 1
        alpha_ratio = T0 / n_aug                        # n_aug 补偿因子
        scale = np.sqrt(LAMBDA_S * T0)                  # √(T₀·λ_s) 对齐论文
        X_aug = np.vstack([X_std, scale * dX])         # (n + n-1, n_feat)
        Y_aug = np.vstack([Y_train, np.zeros((dX.shape[0], K))])
    else:
        X_aug = X_std
        Y_aug = Y_train
        alpha_ratio = 1.0                          # 非M5无补偿

    # 构建每个资产的参数
    tasks = []
    for i in range(K):
        if cfg['network']:
            full_pen = build_penalty_vector(
                i, True, network_mask, lambda_lasso, self_free=cfg.get('self_free', False))
        else:
            full_pen = build_penalty_vector(
                i, False, None, lambda_lasso, self_free=cfg.get('self_free', False))
        pen_sel = full_pen[cols]
        # M5 需补偿 n_aug 翻倍导致的 alpha 放大
        m5_alpha = None  # None → 标准情况（alpha = lambda_lasso）
        if cfg['smooth']:
            m5_alpha = lambda_lasso * alpha_ratio
        tasks.append((X_aug, Y_aug[:, i], pen_sel, cfg['sparse'], lambda_lasso, m5_alpha))

    # 并行拟合
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_fit_one_asset)(X_std=task[0], y_i=task[1], penalties=task[2],
                                use_lasso=task[3], lambda_lasso=task[4], lasso_alpha=task[5])
        for task in tasks
    )

    coefs = np.array([r[0] for r in results])       # (K, n_feat)
    intercepts = np.array([r[1] for r in results])  # (K,)

    return {
        'coefs': coefs,
        'intercepts': intercepts,
        'scaler': scaler,
        'cols': cols,
        'config': cfg,
    }


# ===================================================================
# 预测
# ===================================================================
def predict_model(X_test: np.ndarray, fitted: Dict,
                  normalize: bool = True) -> np.ndarray:
    """预测 GMVP 权重（论文公式 11 归一化）。"""
    X_sel = X_test[:, fitted['cols']]
    X_std = fitted['scaler'].transform(X_sel)

    Y_pred = X_std @ fitted['coefs'].T + fitted['intercepts']  # (n_test, K)

    if normalize:
        s = Y_pred.sum(axis=1, keepdims=True)
        s = np.where(np.abs(s) < 1e-10, 1.0, s)
        Y_pred = Y_pred / s
    return Y_pred




# ===================================================================
# 评估指标（论文 Table 2）
# ===================================================================
def compute_mse(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """MSE = mean((w_pred - w_true)²)。

    规范: mean((w̃_t^pred - w̃_t)² / K) = mean over (T,K) of diff²
    """
    return float(np.mean((Y_pred - Y_true) ** 2))


def compute_mae(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """MAE = mean(|w_pred - w_true|)。"""
    return float(np.mean(np.abs(Y_pred - Y_true)))


def dm_test(Y_pred_model: np.ndarray, Y_pred_baseline: np.ndarray,
            Y_true: np.ndarray) -> Tuple[float, float, float]:
    """Diebold-Mariano 检验（1步前向，vs baseline Model 1）。

    标准 DM 方向（Diebold & Mariano, 1995）:
      d_t = loss_model_t - loss_baseline_t   （正 = baseline 更好）
      DM_stat = mean(d) / sqrt(var(d)/n)
      负且统计显著 → model 显著优于 baseline（损失更低）。

    Returns:
        (dm_stat, p_value, log10_p_value)
        log10_p_value: p值的 log10 表示，用于 DM 统计量极大时避免 float64 下溢。
                       显示时写作 10^{log10_p_value}。
    """
    # 每个时间步的MSE（对K维取平均）
    loss_model = ((Y_pred_model - Y_true) ** 2).mean(axis=1)      # (n_test,)
    loss_base  = ((Y_pred_baseline - Y_true) ** 2).mean(axis=1)   # (n_test,)
    d = loss_model - loss_base                                     # 正=baseline更好, 负=model更好
    n = len(d)
    if n < 2 or d.var(ddof=1) < 1e-15:
        return 0.0, 1.0, np.nan
    dm = d.mean() / np.sqrt(d.var(ddof=1) / n)
    # 用 logsf 避免 float64 下溢（DM≈40 时 p≈10^-350）
    logsf = sp_stats.norm.logsf(abs(dm))          # log(1 - Φ(|DM|))
    log10_p = (np.log(2) + float(logsf)) / np.log(10)
    pval = 2 * (1 - sp_stats.norm.cdf(abs(dm)))
    return float(dm), float(pval), float(log10_p)


def compute_mcs_rank(mse_dict: Dict[int, float]) -> Dict[int, int]:
    """简化 MCS：按 MSE 升序排名（小=好，rank=1最优）。

    完整 Hansen(2011) MCS 程序在 性能评估.py 中实现。
    """
    ranked = sorted(mse_dict.items(), key=lambda x: x[1])
    return {m: i + 1 for i, (m, _) in enumerate(ranked)}


def compute_turnover(Y_pred: np.ndarray,
                    Y_actual_prev: np.ndarray = None) -> float:
    """平均日换手率 = mean_t Σ_i |w_t - w_{t-1}|。

    Args:
        Y_pred:         (T, K) 预测权重序列
        Y_actual_prev:  (T, K) 上一期实际权重（经济换手率）。
                        若为 None，则退化为预测序列自身差分。
    """
    if Y_actual_prev is not None:
        # 经济换手率: |w_pred_t - w_actual_{t-1}|，需对齐长度
        n = min(len(Y_pred), len(Y_actual_prev))
        return float(np.mean(np.abs(Y_pred[:n] - Y_actual_prev[:n]).sum(axis=1)))
    if len(Y_pred) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(Y_pred, axis=0)).sum(axis=1)))


# ===================================================================
RHO_DFL = 1e-3
DFL_BOX = 0.05

def _solve_dfl_day(w_stat, w_prev, Sigma, eta_vec, rho=1e-3):
    """L2 闭式解 — 网络差异化交易成本。

    min  ½w^T·Σ·w + ½Σ_i η_i·(w_i−w_prev_i)² + ½ρ·||w−w_stat||²   s.t. Σw=1
    闭式: A = Σ + diag(η_vec+ρ),  b = η_vec⊙w_prev + ρ·w_stat
    """
    K = len(w_stat)
    eta_vec = np.asarray(eta_vec, dtype=np.float64)
    A = Sigma + np.diag(eta_vec + rho)
    b = eta_vec * w_prev + rho * w_stat
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A += 1e-8 * np.eye(K)
        A_inv = np.linalg.inv(A)
    ones = np.ones(K)
    A_inv_b = A_inv @ b
    A_inv_1 = A_inv @ ones
    lam = (A_inv_b.sum() - 1.0) / A_inv_1.sum()
    w_opt = A_inv_b - lam * A_inv_1
    w_opt = np.clip(w_opt, -DFL_BOX, DFL_BOX)
    w_opt = w_opt / w_opt.sum()
    return w_opt

def compute_model6(Y_pred_m4, Y_actual, valid_train_days, n_train=200, eta=1e-4, rho=1e-3):
    """M6: DFL 事后调优 (盒约束 + 滚动协方差 + 网络差异化 η)。"""
    log("\n--- Model 6: DFL Post-hoc (box=" + str(DFL_BOX) + ", rolling cov) ---")
    log(f"  参数: eta={eta:.0e}  rho={rho:.0e}")
    sys.path.insert(0, str(Path(__file__).parents[1]/"图形Lasso"/"code"))
    from 共享模块 import load_day, compute_raw_cov, EPS_RIDGE
    t0=time.time()
    # 滚动协方差: 只用训练期末40天 (最近市场状态)
    roll_window=40
    si=valid_train_days[-roll_window:]
    cov_sum=np.zeros((K,K)); loaded=0
    for idx in si:
        try: rett=load_day(idx); cov=compute_raw_cov(rett); cov.flat[::K+1]+=EPS_RIDGE; cov_sum+=cov; loaded+=1
        except: pass
    Sigma_rolling=cov_sum/loaded
    A_bar=np.load(Path(__file__).parent.parent/"特征工程"/"A_bar.npy")[:len(valid_train_days)].mean(0)
    deg=(A_bar.sum(1)/(K-1)).clip(0,1)
    eta_vec=eta*(1.5-0.5*deg)
    A_bar=np.load(Path(__file__).parent.parent/"特征工程"/"A_bar.npy")[:len(valid_train_days)].mean(0)
    deg=(A_bar.sum(1)/(K-1)).clip(0,1)
    eta_vec=eta*(1.5-0.5*deg)
    log(f"  滚动Sigma({roll_window}d): {loaded}天, deg=[{deg.min():.2f},{deg.max():.2f}], eta=[{eta_vec.min():.1e},{eta_vec.max():.1e}] ({time.time()-t0:.1f}s)")
    T=Y_pred_m4.shape[0]; Y_dfl=np.zeros_like(Y_pred_m4)
    for t in range(T):
        ws=Y_pred_m4[t]; wp=Y_actual[t-1] if t>0 else Y_actual[0]
        Y_dfl[t]=_solve_dfl_day(ws,wp,Sigma_rolling,eta_vec,rho)
    log(f"  DFL 完成 ({time.time()-t0:.1f}s)")
    return Y_dfl


"""阶段三-B: VARX预测 — 论文§3.2, Table 2

实现 Model 1-5（Lasso族）+ Model 6（DFL调优）+ 4个评估指标（MSE/MAE/DM/MCS）。

模型层次（论文 Table 2）:
  1. VAR              — OLS, 滞后权重（无稀疏、无外生、无网络）
  2. Sparse VAR       — Lasso, 滞后权重
  3. Sparse VARX      — Lasso, 全 x_t（含网络拓扑）
  4. Network VARX     — Lasso, 同 M3 特征集 + λ₁≠λ₂ 网络惩罚
  5. Network+Smooth   — Model 4 + 数据增广 λ_s 平滑（训练时嵌入）
  6. DFL-Tuned VARX   — Model 4 预测 + 决策聚焦后处理调优
  
Model 7 (LSTM) → 待实现

网络加权惩罚实现（论文公式13，四参数体系，网格搜索验证 2026-07-03）:
  λ₁ = LAMBDA_LASSO      = 1e-4   (连接资产的滞后系数 ℓ1)
  λ₂ = λ₁+LAMBDA_NETWORK  = 5.1e-3 (未连接资产的滞后系数 ℓ1, λ₂≫λ₁)
  λ₃ = LAMBDA_EXOG       = 5e-4   (外生变量系数 ℓ1)
  λ_s = LAMBDA_TURNOVER  = 5e-3   (换手率平滑 ℓ2)
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# 共享模块路径
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import (
    K, P_LAGS, LAMBDA_LASSO, LAMBDA_NETWORK, LAMBDA_EXOG, LAMBDA_TURNOVER,
    NETWORK_THRESHOLD, ETA, OUT_DIR, log, set_log_file,
)

# ===================================================================
# 常量
# ===================================================================
N_EXOG = 9                                        # 外生变量维度（宏观9维）
N_NET  = 0                                        # 网络拓扑从 X 中移除，仅作 M4/M5 惩罚指引
N_FEATURES = P_LAGS * K + N_EXOG                  # 总特征数 = 1185

# 特征块列索引（与 特征工程.py 的 names 列表对齐）
# 网络拓扑特征(度/PR/聚类)已从 X 中移出，Ā 仅在 build_network_mask 中用于构建惩罚掩码
FEAT_BLOCKS = {
    'lagged':  (0,              P_LAGS * K),          # [0, 1176)
    'exog':    (P_LAGS * K,     P_LAGS * K + N_EXOG), # [1176, 1185)
}

# 模型配置
# blocks: 使用的特征块; sparse: 是否Lasso(否则OLS); network: 是否网络加权惩罚; smooth: 是否平滑
# self_free: 是否对每个方程的自身滞后 B_lag[i,i] 近似零惩罚；用于 M3a 消融和 M4/M5。
# lasso_lambda: 模型专属 λ₁ 覆写。M2 无稀疏约束可贪心用 5e-04，
#   M3-M5 需 ≤1e-04 保留交叉项供 M4 网络惩罚有靶子可罚
MODELS = {
    1: {'name': 'VAR',            'blocks': ['lagged'],         'sparse': False, 'network': False, 'smooth': False, 'self_free': False, 'lasso_lambda': LAMBDA_LASSO},
    2: {'name': 'Sparse VAR',     'blocks': ['lagged'],         'sparse': True,  'network': False, 'smooth': False, 'self_free': False, 'lasso_lambda': LAMBDA_LASSO},
    3: {'name': 'Sparse VARX',    'blocks': ['lagged', 'exog'], 'sparse': True,  'network': False, 'smooth': False, 'self_free': False, 'lasso_lambda': LAMBDA_LASSO},
    4: {'name': 'Network VARX',   'blocks': ['lagged', 'exog'], 'sparse': True,  'network': True,  'smooth': False, 'self_free': True,  'lasso_lambda': LAMBDA_LASSO},
    5: {'name': 'Network+Smooth', 'blocks': ['lagged', 'exog'], 'sparse': True,  'network': True,  'smooth': True,  'self_free': True,  'lasso_lambda': LAMBDA_LASSO},
}

# Model 5 的换手率平滑 λ_s（论文公式13最后一项）
# 通过数据增广法嵌入训练：追加 √(T₀·λ_s)·ΔX 行（目标=0）
# sklearn Lasso 归一化为 (1/(2n))，展开后平滑/MSE 权重比 = T₀·λ_s，与论文一致
LAMBDA_S = LAMBDA_TURNOVER


# ===================================================================
# 数据加载与划分
# ===================================================================
def load_data() -> Dict[str, np.ndarray]:
    """加载特征工程产出（X, Y, A_bar）。"""
    feat_dir = Path(__file__).parent.parent / "特征工程"
    return {
        'X': np.load(feat_dir / "X_features.npy"),
        'Y': np.load(feat_dir / "Y_targets.npy"),
        'A_bar': np.load(feat_dir / "A_bar.npy"),
    }


def split_data(X, Y, A_bar, train_ratio=0.70, val_ratio=0.15):
    """固定时间划分 70/15/15（规范 §3.3）。

    Returns:
        dict with keys 'train', 'val', 'test', each = (X, Y, A_bar)
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        'train': (X[:n_train],          Y[:n_train],          A_bar[:n_train]),
        'val':   (X[n_train:n_train+n_val], Y[n_train:n_train+n_val], A_bar[n_train:n_train+n_val]),
        'test':  (X[n_train+n_val:],    Y[n_train+n_val:],    A_bar[n_train+n_val:]),
    }


# ===================================================================
# 网络掩码（论文公式 7）
# ===================================================================
def build_network_mask(A_bar_train: np.ndarray, threshold: float = NETWORK_THRESHOLD,
                       use_last_day: bool = False
                       ) -> Tuple[np.ndarray, float]:
    """训练期 Ā → 二值掩码 M[i,j] = 1(Ā[i,j] ≥ τ)。

    use_last_day: 若 False（默认），对全训练期 Ā 取均值（静态掩码，长期稳定结构）；
                  若 True，直接用最后一天的 Ā_{T}（~20天精度的近期快照）。
    Ā 本身已是 20 天滚动窗口均值。两种模式对应论文 §3.2 的
    "fixed network (training window)" vs "rolling-window average network"。

    Returns:
        mask: (K, K) float64 二值矩阵（对角线=0，无自环）
        density: 网络密度
    """
    if use_last_day:
        A_mean = A_bar_train[-1]                         # 仅最后一天（≈最近20天快照）
    else:
        A_mean = A_bar_train.mean(axis=0)                # 全训练期均值（原行为）
    mask = (A_mean >= threshold).astype(np.float64)
    np.fill_diagonal(mask, 0)                           # 排除自环
    density = mask.sum() / (K * (K - 1))
    return mask, density


# ===================================================================
# 惩罚向量（论文公式 13）
# ===================================================================
def build_penalty_vector(asset_i: int, use_network: bool,
                         network_mask: Optional[np.ndarray],
                         lambda_lasso: float = LAMBDA_LASSO,
                         self_free: bool = False) -> np.ndarray:
    """构建 per-feature 惩罚向量 (N_FEATURES,)，对齐论文公式(13) 四参数体系。

    滞后块，连接资产 j  → λ₁ = lambda_lasso
    滞后块，未连接资产 j → λ₂ = lambda_lasso + LAMBDA_NETWORK  (λ₂ > λ₁)
    外生块（宏观9维）     → λ₃ = LAMBDA_EXOG
    网络拓扑特征已从 X 移出，仅通过 network_mask 影响滞后块惩罚差异化。
    """
    penalties = np.full(N_FEATURES, lambda_lasso)

    # 外生变量（宏观9维）用 λ₃
    exog_start = FEAT_BLOCKS['exog'][0]
    exog_end   = FEAT_BLOCKS['exog'][1]
    penalties[exog_start:exog_end] = LAMBDA_EXOG

    if use_network and network_mask is not None:
        connected = network_mask[asset_i]           # (K,) 资产 i 的连接行
        for lag in range(P_LAGS):
            offset = lag * K
            penalties[offset:offset + K] = np.where(
                connected > 0,
                lambda_lasso,                    # λ₁: 连接
                lambda_lasso + LAMBDA_NETWORK,   # λ₂: 未连接, j≠i
            )
    if self_free:
        # 论文公式(13): λ₂ 项显式含 j≠i，自身滞后 B_{ℓ,ii} 可作为消融项近似零惩罚。
        # 设为零会在 _fit_one_asset 中除零，使用极小值 1e-12 近似零惩罚。
        for lag in range(P_LAGS):
            penalties[lag * K + asset_i] = 1e-12
    return penalties


# ===================================================================
# 拟合
# ===================================================================
def _fit_one_asset(X_std: np.ndarray, y_i: np.ndarray,
                   penalties: np.ndarray, use_lasso: bool,
                   lambda_lasso: float = LAMBDA_LASSO,
                   lasso_alpha: Optional[float] = None
                   ) -> Tuple[np.ndarray, float]:
    """拟合单个资产。

    加权L1 → 标准Lasso的转化：
      s = λ_ref / penalties        （缩放因子）
      X_scaled = X_std * s         （缩放特征）
      β_scaled = Lasso.fit(X_scaled, y)  （标准Lasso）
      β_original = β_scaled * s    （反缩放回原始空间）

    Args:
        X_std: (n_train, n_feat) 已标准化的特征矩阵
        y_i:   (n_train,) 资产 i 的目标权重
        penalties: (n_feat,) per-feature 惩罚
        use_lasso: True=Lasso, False=OLS(Model 1)
        lambda_lasso: 模型专属 λ₁（用于特征缩放和惩罚向量构建）
        lasso_alpha: sklearn Lasso alpha 参数。
                     若为 None 则等于 lambda_lasso（标准情况）。
                     M5 数据增广时由于 n_aug 翻倍，需传入补偿值保持可比。

    Returns:
        (coef, intercept) 系数和截距（标准化空间）
    """
    if not use_lasso:
        model = LinearRegression()                     # Model 1: OLS 无正则化
        model.fit(X_std, y_i)
        return model.coef_.copy(), float(model.intercept_)

    # 加权 Lasso via 特征缩放
    alpha = lasso_alpha if lasso_alpha is not None else lambda_lasso
    s = lambda_lasso / penalties                   # 缩放因子（始终用原始 λ₁）
    X_scaled = X_std * s                            # (n_train, n_feat)
    model = Lasso(alpha=alpha, max_iter=2000,
                  tol=1e-4, selection='cyclic')
    model.fit(X_scaled, y_i)
    coef = model.coef_ * s                          # 反缩放
    return coef, float(model.intercept_)


def fit_model(model_id: int, X_train: np.ndarray, Y_train: np.ndarray,
              network_mask: Optional[np.ndarray], n_jobs: int = 4
              ) -> Dict:
    """拟合 K=392 个资产（并行）。

    Returns:
        dict: coefs (K,n_feat), intercepts (K,), scaler, cols, config
    """
    from joblib import Parallel, delayed

    cfg = MODELS[model_id]
    lambda_lasso = cfg.get('lasso_lambda', LAMBDA_LASSO)  # 模型专属 λ₁

    # 选取特征列
    cols = []
    for block in cfg['blocks']:
        s, e = FEAT_BLOCKS[block]
        cols.extend(range(s, e))
    cols = np.array(cols)
    X_sel = X_train[:, cols]
    n_feat = len(cols)

    # 标准化（Lasso对量纲敏感）
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_sel)

    # Model 1 快速路径：批量 OLS（K 个资产共享同一 X，一次正常方程求解）
    # β = (X̃ᵀX̃)⁻¹X̃ᵀY  →  O(np² + p³ + p²K)  vs  逐资产 ×392 ≈ 250×加速
    # 用 lstsq (SVD) 替代 solve：滞后权重列高度共线，solve 可能卡死
    if not cfg['sparse'] and not cfg['smooth']:
        X_aug = np.column_stack([X_std, np.ones(X_std.shape[0])])  # 增广截距列
        theta = np.linalg.lstsq(X_aug, Y_train, rcond=None)[0]     # (p+1, K)
        coefs = theta[:-1].T                                        # (K, p)
        intercepts = theta[-1]                                      # (K,)
        log(f"  [批量OLS lstsq] {X_std.shape[1]}特征 × {K}资产 = 1次求解")
        return {
            'coefs': coefs,
            'intercepts': intercepts,
            'scaler': scaler,
            'cols': cols,
            'config': cfg,
        }

    # Model 5: 数据增广实现 λ_s 换手率平滑（论文公式13最后一项）
    # 论文目标:   L = (1/T₀)‖y-Xβ‖² + λ_s·‖ΔX·β‖² + α‖β‖₁
    # sklearn Lasso 目标: (1/(2·n_aug))·‖Y_aug - X_aug·β‖² + α·‖β‖₁
    #
    # 追加 √(T₀·λ_s)·ΔX 行（目标=0），则 n_aug = T₀ + (T₀-1) = 2T₀-1；
    # (1/(2·n_aug))·[‖y-Xβ‖² + T₀·λ_s·‖ΔX·β‖²] + α·‖β‖₁
    #
    # 展开后 MSE 项权重比 = T₀ / (2·n_aug) ≈ 1/2 (当 T₀≫1)，
    # 平滑项与 MSE 的权重比 = T₀·λ_s / (2·n_aug) ≈ λ_s / 2 (当 T₀≫1)。
    # 论文中两者相对权重 = T₀·λ_s，代码中经 sklearn 归一化后 ≈ T₀·λ_s/2。
    # 由于网格搜索调整 λ_s 时会自动补偿这个 2× 因子，最终 λ_s 值
    # 在论文尺度上等效翻倍，不影响模型间相对比较。
    #
    # 重要：n_aug ≈ 2× n_orig 导致 sklearn Lasso alpha 的有效强度也翻倍
    # （因为 (1/(2n_aug)) 的分母比 (1/(2T₀)) 大一倍）。
    # 在传递给 _fit_one_asset 时，alpha 乘以 T0/n_aug 补偿这一效应。
    # 注意：此补偿仅调整 Lasso alpha，不改变 build_penalty_vector 中的 λ₁（
    # 惩罚向量中的 LAMBDA_EXOG/LAMBDA_NETWORK 保持原值，确保 M5 vs M4 可比）。
    # ΔX = X_{t+1} - X_t 逐日特征差分
    if cfg['smooth']:
        dX = np.diff(X_std, axis=0)                    # (n-1, n_feat)
        T0 = X_std.shape[0]                             # 训练样本数
        n_aug = T0 + dX.shape[0]                        # = 2*T0 - 1
        alpha_ratio = T0 / n_aug                        # n_aug 补偿因子
        scale = np.sqrt(LAMBDA_S * T0)                  # √(T₀·λ_s) 对齐论文
        X_aug = np.vstack([X_std, scale * dX])         # (n + n-1, n_feat)
        Y_aug = np.vstack([Y_train, np.zeros((dX.shape[0], K))])
    else:
        X_aug = X_std
        Y_aug = Y_train
        alpha_ratio = 1.0                          # 非M5无补偿

    # 构建每个资产的参数
    tasks = []
    for i in range(K):
        if cfg['network']:
            full_pen = build_penalty_vector(
                i, True, network_mask, lambda_lasso, self_free=cfg.get('self_free', False))
        else:
            full_pen = build_penalty_vector(
                i, False, None, lambda_lasso, self_free=cfg.get('self_free', False))
        pen_sel = full_pen[cols]
        # M5 需补偿 n_aug 翻倍导致的 alpha 放大
        m5_alpha = None  # None → 标准情况（alpha = lambda_lasso）
        if cfg['smooth']:
            m5_alpha = lambda_lasso * alpha_ratio
        tasks.append((X_aug, Y_aug[:, i], pen_sel, cfg['sparse'], lambda_lasso, m5_alpha))

    # 并行拟合
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_fit_one_asset)(X_std=task[0], y_i=task[1], penalties=task[2],
                                use_lasso=task[3], lambda_lasso=task[4], lasso_alpha=task[5])
        for task in tasks
    )

    coefs = np.array([r[0] for r in results])       # (K, n_feat)
    intercepts = np.array([r[1] for r in results])  # (K,)

    return {
        'coefs': coefs,
        'intercepts': intercepts,
        'scaler': scaler,
        'cols': cols,
        'config': cfg,
    }


# ===================================================================
# 预测
# ===================================================================
def predict_model(X_test: np.ndarray, fitted: Dict,
                  normalize: bool = True) -> np.ndarray:
    """预测 GMVP 权重（论文公式 11 归一化）。"""
    X_sel = X_test[:, fitted['cols']]
    X_std = fitted['scaler'].transform(X_sel)

    Y_pred = X_std @ fitted['coefs'].T + fitted['intercepts']  # (n_test, K)

    if normalize:
        s = Y_pred.sum(axis=1, keepdims=True)
        s = np.where(np.abs(s) < 1e-10, 1.0, s)
        Y_pred = Y_pred / s
    return Y_pred




# ===================================================================
# 评估指标（论文 Table 2）
# ===================================================================
def compute_mse(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """MSE = mean((w_pred - w_true)²)。

    规范: mean((w̃_t^pred - w̃_t)² / K) = mean over (T,K) of diff²
    """
    return float(np.mean((Y_pred - Y_true) ** 2))


def compute_mae(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """MAE = mean(|w_pred - w_true|)。"""
    return float(np.mean(np.abs(Y_pred - Y_true)))


def dm_test(Y_pred_model: np.ndarray, Y_pred_baseline: np.ndarray,
            Y_true: np.ndarray) -> Tuple[float, float, float]:
    """Diebold-Mariano 检验（1步前向，vs baseline Model 1）。

    标准 DM 方向（Diebold & Mariano, 1995）:
      d_t = loss_model_t - loss_baseline_t   （正 = baseline 更好）
      DM_stat = mean(d) / sqrt(var(d)/n)
      负且统计显著 → model 显著优于 baseline（损失更低）。

    Returns:
        (dm_stat, p_value, log10_p_value)
        log10_p_value: p值的 log10 表示，用于 DM 统计量极大时避免 float64 下溢。
                       显示时写作 10^{log10_p_value}。
    """
    # 每个时间步的MSE（对K维取平均）
    loss_model = ((Y_pred_model - Y_true) ** 2).mean(axis=1)      # (n_test,)
    loss_base  = ((Y_pred_baseline - Y_true) ** 2).mean(axis=1)   # (n_test,)
    d = loss_model - loss_base                                     # 正=baseline更好, 负=model更好
    n = len(d)
    if n < 2 or d.var(ddof=1) < 1e-15:
        return 0.0, 1.0, np.nan
    dm = d.mean() / np.sqrt(d.var(ddof=1) / n)
    # 用 logsf 避免 float64 下溢（DM≈40 时 p≈10^-350）
    logsf = sp_stats.norm.logsf(abs(dm))          # log(1 - Φ(|DM|))
    log10_p = (np.log(2) + float(logsf)) / np.log(10)
    pval = 2 * (1 - sp_stats.norm.cdf(abs(dm)))
    return float(dm), float(pval), float(log10_p)


def compute_mcs_rank(mse_dict: Dict[int, float]) -> Dict[int, int]:
    """简化 MCS：按 MSE 升序排名（小=好，rank=1最优）。

    完整 Hansen(2011) MCS 程序在 性能评估.py 中实现。
    """
    ranked = sorted(mse_dict.items(), key=lambda x: x[1])
    return {m: i + 1 for i, (m, _) in enumerate(ranked)}


def compute_turnover(Y_pred: np.ndarray,
                    Y_actual_prev: np.ndarray = None) -> float:
    """平均日换手率 = mean_t Σ_i |w_t - w_{t-1}|。

    Args:
        Y_pred:         (T, K) 预测权重序列
        Y_actual_prev:  (T, K) 上一期实际权重（经济换手率）。
                        若为 None，则退化为预测序列自身差分。
    """
    if Y_actual_prev is not None:
        # 经济换手率: |w_pred_t - w_actual_{t-1}|，需对齐长度
        n = min(len(Y_pred), len(Y_actual_prev))
        return float(np.mean(np.abs(Y_pred[:n] - Y_actual_prev[:n]).sum(axis=1)))
    if len(Y_pred) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(Y_pred, axis=0)).sum(axis=1)))


# ===================================================================
# Model 6: 决策聚焦训练（论文 §3.3 — 学习范式, 非事后调优）
#
# 论文公式 (14)-(15) + 系数正则化:
#   L(Θ) = (1/T₀) Σ_t [ ŵ_t^T·Σ̂_t·ŵ_t + η·||ŵ_t − w̃_{t-1}||₁ ]
#          + α·||β||₁ + γ·||β||₂² + λ_s·||ΔX·β||²
#
def main():
    out_dir = Path(__file__).parent
    set_log_file(out_dir / "varx_log.txt")

    log("=" * 60)
    log("  阶段三-B: VARX预测 — Table 2")
    log("=" * 60)
    log(f"K={K}  P_LAGS={P_LAGS}  λ_lasso={LAMBDA_LASSO}  λ_network={LAMBDA_NETWORK}")
    log(f"  NETWORK_THRESHOLD={NETWORK_THRESHOLD}  LAMBDA_S={LAMBDA_S}")

    # ---- 加载数据 ----
    data = load_data()
    X, Y, A_bar = data['X'], data['Y'], data['A_bar']
    log(f"\n数据: X={X.shape}  Y={Y.shape}  A_bar={A_bar.shape}")

    # ---- 划分 ----
    splits = split_data(X, Y, A_bar)
    X_tr, Y_tr, A_tr = splits['train']
    X_te, Y_te, A_te = splits['test']
    n_train, n_val, n_test = X_tr.shape[0], splits['val'][0].shape[0], X_te.shape[0]
    log(f"划分: 训练={n_train}  验证={n_val}  测试={n_test}")

    # ---- 网络掩码 ----
    net_mask, density = build_network_mask(A_tr)
    avg_degree = net_mask.sum(axis=1).mean()
    log(f"网络掩码: 密度={density:.1%}  平均度={avg_degree:.0f}")

    # ---- 逐模型拟合 + 预测 + 评估 ----
    predictions = {}
    fitted_models = {}        # 保存拟合结果供 DFL 等下游使用
    results = []

    for model_id in range(1, 6):
        cfg = MODELS[model_id]
        log(f"\n--- Model {model_id}: {cfg['name']} ---")
        log(f"  特征块: {cfg['blocks']}  稀疏: {cfg['sparse']}  网络: {cfg['network']}  平滑: {cfg['smooth']}")

        t0 = time.time()
        fitted = fit_model(model_id, X_tr, Y_tr, net_mask, n_jobs=4)
        t_fit = time.time() - t0

        Y_pred = predict_model(X_te, fitted)

        predictions[model_id] = Y_pred
        fitted_models[model_id] = fitted

        mse = compute_mse(Y_pred, Y_te)
        mae = compute_mae(Y_pred, Y_te)
        sparsity = float(np.mean(np.abs(fitted['coefs']) < 1e-8))
        turnover = compute_turnover(Y_pred)

        log(f"  耗时: {t_fit:.1f}s")
        log(f"  MSE: {mse:.6e}  MAE: {mae:.6e}")
        log(f"  稀疏度: {sparsity:.1%}  换手率: {turnover:.6f}")

        results.append({
            'Model': model_id,
            'Name': cfg['name'],
            'MSE': mse,
            'MAE': mae,
            'Sparsity': sparsity,
            'Turnover': turnover,
            'FitTime_s': round(t_fit, 1),
        })

    # ---- DM 检验（vs Model 1）----
    log("\n--- DM 检验 (vs Model 1 VAR) ---")
    Y_base = predictions[1]
    for model_id in [2, 3, 4, 5, 6]:
        if model_id not in predictions:
            continue
        dm, pval, log10p = dm_test(predictions[model_id], Y_base, Y_te)
        if pval == 0.0 and not np.isnan(log10p):
            p_fmt = "10^{" + f"{log10p:.0f}" + "}"
        else:
            p_fmt = f"{pval:.4e}"
        sig = '***' if (pval < 0.01 or (pval == 0.0 and not np.isnan(log10p))) else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        log(f"  Model {model_id} vs 1: DM={dm:+.3f}  p={p_fmt} {sig}")
        results[model_id - 1]['DM_stat'] = dm
        results[model_id - 1]['DM_pvalue'] = pval
        results[model_id - 1]['DM_log10_p'] = log10p
    results[0]['DM_stat'] = np.nan
    results[0]['DM_pvalue'] = np.nan

    # ---- Model 6: DFL Post-hoc (盒约束) ----
    feat_dir = Path(__file__).parent.parent / "特征工程"
    valid_indices = np.load(feat_dir / "valid_indices.npy")
    train_day_indices = valid_indices[:n_train]
    Y_pred_m6 = compute_model6(predictions[4], Y_te, train_day_indices, eta=ETA)

    predictions[6] = Y_pred_m6
    m6_mse = compute_mse(Y_pred_m6, Y_te)
    m6_mae = compute_mae(Y_pred_m6, Y_te)
    m6_to  = compute_turnover(Y_pred_m6)
    log(f"  MSE: {m6_mse:.6e}  MAE: {m6_mae:.6e}  换手率: {m6_to:.6f}")

    predictions[6] = Y_pred_m6
    m6_mse = compute_mse(Y_pred_m6, Y_te)
    m6_mae = compute_mae(Y_pred_m6, Y_te)
    m6_to  = compute_turnover(Y_pred_m6)
    log(f"  MSE: {m6_mse:.6e}  MAE: {m6_mae:.6e}  换手率: {m6_to:.6f}")

    results.append({
        'Model': 6, 'Name': 'DFL VARX',
        'MSE': m6_mse, 'MAE': m6_mae,
        'Sparsity': np.nan, 'Turnover': m6_to, 'FitTime_s': np.nan,
    })

    # ---- Model 7 占位 ----
    results.append({
        'Model': 7, 'Name': 'LSTM', 'MSE': np.nan, 'MAE': np.nan,
        'DM_stat': np.nan, 'DM_pvalue': np.nan, 'DM_log10_p': np.nan,
        'MCS_rank': np.nan, 'Sparsity': np.nan, 'Turnover': np.nan, 'FitTime_s': np.nan,
    })

    # ---- M6 DM 检验（补在 results 之后）----
    dm_m6, pval_m6, log10p_m6 = dm_test(predictions[6], Y_base, Y_te)
    if pval_m6 == 0.0 and not np.isnan(log10p_m6):
        p_fmt_m6 = "10^{" + f"{log10p_m6:.0f}" + "}"
    else:
        p_fmt_m6 = f"{pval_m6:.4e}"
    sig_m6 = '***' if (pval_m6 < 0.01 or (pval_m6 == 0.0 and not np.isnan(log10p_m6))) else ('**' if pval_m6 < 0.05 else ('*' if pval_m6 < 0.1 else ''))
    log(f"  Model 6 vs 1: DM={dm_m6:+.3f}  p={p_fmt_m6} {sig_m6}")
    results[5]['DM_stat'] = dm_m6
    results[5]['DM_pvalue'] = pval_m6
    results[5]['DM_log10_p'] = log10p_m6

    # ---- Model 7: LSTM (pure NumPy, BPTT + Adam) ----
    log(f"\n--- Model 7: LSTM ---")
    log(f"  架构: 序列长度={20}, 隐藏维度={64}, LSTM×1 → Dense({K})")
    m4_cols = fitted_models[4]['cols']
    X_m7 = X_tr[:, m4_cols]

    # hyperparams
    seq_len, hid, lr = 20, 64, 0.01     # lr=0.01 补偿 1/K 梯度缩放修复
    n_feat = X_m7.shape[1]
    batch, epochs, patience = 32, 100, 10

    # ---------- LSTM cell ----------
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    d_tanh    = lambda z, a: 1 - a**2
    d_sigmoid = lambda a: a * (1 - a)

    # Xavier init
    scale = np.sqrt(2.0 / (n_feat + hid))
    W = {g: scale * np.random.randn(hid, n_feat + hid).astype(np.float64) for g in 'f i o c'.split()}
    b = {g: (np.ones(hid) if g == 'f' else np.zeros(hid)).astype(np.float64) for g in 'f i o c'.split()}
    Wy = scale * np.random.randn(K, hid).astype(np.float64)
    by = np.zeros(K, dtype=np.float64)

    # Adam accumulators
    M, V = {}, {}
    for g in 'f i o c'.split():
        M[f'W_{g}'], V[f'W_{g}'] = np.zeros_like(W[g]), np.zeros_like(W[g])
        M[f'b_{g}'], V[f'b_{g}'] = np.zeros_like(b[g]), np.zeros_like(b[g])
    M['Wy'], V['Wy'] = np.zeros_like(Wy), np.zeros_like(Wy)
    M['by'], V['by'] = np.zeros_like(by), np.zeros_like(by)

    # ---------- make sequences ----------
    def make_seqs(X, Y):  # (n, feat), (n, K) → (n-seq+1, seq, feat), (n-seq+1, K)
        n = len(X); xs, ys = [], []
        for i in range(seq_len-1, n):
            xs.append(X[i-seq_len+1:i+1]); ys.append(Y[i])
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    # normalize
    x_mu = X_m7.mean(0); x_sig = np.std(X_m7, 0).clip(1e-8)
    X_s = (X_m7 - x_mu) / x_sig
    # 验证集从 splits 获取（X_tr/Y_tr 仅含训练部分，不能从中取 val）
    X_val_s = (splits['val'][0][:, m4_cols] - x_mu) / x_sig
    Y_val_d = splits['val'][1]
    X_te_s  = (X_te[:, m4_cols] - x_mu) / x_sig

    Xs_tr, Ys_tr = make_seqs(X_s, Y_tr)
    Xs_val, Ys_val = make_seqs(X_val_s, Y_val_d) if len(Y_val_d) >= seq_len else (Xs_tr[:1], Ys_tr[:1])

    # ---------- forward pass ----------
    def forward(X_batch):  # (B, seq, feat) → (B, K), plus caches
        B = len(X_batch)
        h = np.zeros((B, hid)); c = np.zeros((B, hid))
        cache = []  # [(h_prev, c_prev, i, f, o, ct, x_t)]
        for t in range(seq_len):
            x_t = X_batch[:, t, :]; h_prev, c_prev = h.copy(), c.copy()
            xh = np.hstack([x_t, h])
            i = sigmoid(xh @ W['i'].T + b['i'])
            f = sigmoid(xh @ W['f'].T + b['f'])
            o = sigmoid(xh @ W['o'].T + b['o'])
            ct = np.tanh(xh @ W['c'].T + b['c'])
            c = f * c + i * ct
            h = o * np.tanh(c)
            cache.append((h_prev, c_prev, i, f, o, ct, x_t))
        y = h @ Wy.T + by
        return y, cache, h, c

    # ---------- backward pass + Adam update ----------
    def step(X_batch, Y_batch, t_step):
        nonlocal Wy, by, W, b, M, V
        B = len(X_batch)
        y_pred, cache, h_T, c_T = forward(X_batch)
        loss = np.mean((y_pred - Y_batch)**2)
        dy = (2/(B*K)) * (y_pred - Y_batch)  # loss归一化1/(B*K)，梯度需同步缩放

        dWy = dy.T @ h_T; dby = dy.sum(0)
        dh = dy @ Wy
        dc = np.zeros((B, hid))
        dW = {g: np.zeros_like(W[g]) for g in 'f i o c'.split()}
        db = {g: np.zeros_like(b[g]) for g in 'f i o c'.split()}

        for t in range(seq_len-1, -1, -1):
            h_prev, c_prev, i_g, f_g, o_g, ct_g, x_t = cache[t]
            c_cur = cache[t+1][1] if t < seq_len-1 else c_T  # next step's c_prev = this step's c
            h_cur = cache[t+1][0] if t < seq_len-1 else h_T

            do = dh * np.tanh(c_cur)
            dc = dc + dh * o_g * d_tanh(None, np.tanh(c_cur))
            di = dc * ct_g; dct = dc * i_g
            df = dc * c_prev          # c_prev = cache[t][1] = 本步遗忘门所用的 C_{t-1}

            do_g = do * d_sigmoid(o_g); di_g = di * d_sigmoid(i_g)
            df_g = df * d_sigmoid(f_g); dct_g = dct * d_tanh(None, ct_g)

            xh = np.hstack([x_t, h_prev])
            dW['o'] += do_g.T @ xh; db['o'] += do_g.sum(0)
            dW['i'] += di_g.T @ xh; db['i'] += di_g.sum(0)
            dW['f'] += df_g.T @ xh; db['f'] += df_g.sum(0)
            dW['c'] += dct_g.T @ xh; db['c'] += dct_g.sum(0)

            dh_xh = do_g@W['o']+di_g@W['i']+df_g@W['f']+dct_g@W['c']
            dh = dh_xh[:, n_feat:]
            dc = dc * f_g

        # Adam update
        for name, dval in [
            ('Wy', dWy), ('by', dby),
            ('W_f', dW['f']), ('b_f', db['f']), ('W_i', dW['i']), ('b_i', db['i']),
            ('W_o', dW['o']), ('b_o', db['o']), ('W_c', dW['c']), ('b_c', db['c']),
        ]:
            M[name] = 0.9*M[name] + 0.1*dval; V[name] = 0.999*V[name] + 0.001*dval**2
            m_hat = M[name]/(1-0.9**(t_step+1)); v_hat = V[name]/(1-0.999**(t_step+1))
            if name == 'Wy': Wy -= lr*m_hat/(np.sqrt(v_hat)+1e-8)
            elif name == 'by': by -= lr*m_hat/(np.sqrt(v_hat)+1e-8)
            elif name.startswith('W_'): W[name[2]] -= lr*m_hat/(np.sqrt(v_hat)+1e-8)
            else: b[name[2]] -= lr*m_hat/(np.sqrt(v_hat)+1e-8)
        return loss

    # ---------- training ----------
    t0 = time.time()
    n_batches = max(1, len(Xs_tr) // batch)
    best_val, wait = np.inf, 0
    log(f"  训练序列={len(Xs_tr)}, 批次数={n_batches}, {epochs}轮")

    for ep in range(epochs):
        idx = np.random.permutation(len(Xs_tr))
        losses = []
        for ib in range(n_batches):
            bi = idx[ib*batch:(ib+1)*batch]
            loss = step(Xs_tr[bi], Ys_tr[bi], ep*n_batches+ib)
            losses.append(loss)

        # validation
        yv, _, _, _ = forward(Xs_val); val_mse = float(np.mean((yv-Ys_val)**2))
        if val_mse < best_val: best_val = val_mse; wait = 0
        else: wait += 1
        if ep % 20 == 0:
            log(f"  ep {ep:3d}: train={np.mean(losses):.4e}  val={val_mse:.4e}")

    log(f"  训练完成 ({time.time()-t0:.1f}s), 最佳验证MSE={best_val:.4e}")

    # ---------- predict ----------
    Xs_te, _ = make_seqs(X_te_s, Y_te)
    Y_pred_m7, _, _, _ = forward(Xs_te)
    # pad first seq_len-1 predictions with M4
    pad = predictions[4][:seq_len-1]
    Y_pred_m7 = np.vstack([pad, Y_pred_m7])
    # sum-to-one
    s7 = Y_pred_m7.sum(1, keepdims=True); s7 = np.where(np.abs(s7)<1e-10, 1.0, s7)
    Y_pred_m7 = Y_pred_m7 / s7

    predictions[7] = Y_pred_m7
    m7_mse = compute_mse(Y_pred_m7, Y_te)
    m7_mae = compute_mae(Y_pred_m7, Y_te)
    m7_to  = compute_turnover(Y_pred_m7)
    log(f"  MSE: {m7_mse:.6e}  MAE: {m7_mae:.6e}  TO: {m7_to:.6f}")

    results[6]['MSE'] = m7_mse
    results[6]['MAE'] = m7_mae
    results[6]['Turnover'] = m7_to
    results[6]['FitTime_s'] = round(time.time()-t0, 1)

    dm_m7, pval_m7, log10p_m7 = dm_test(Y_pred_m7, Y_base, Y_te)
    if pval_m7 == 0.0 and not np.isnan(log10p_m7):
        p_fmt_m7 = "10^{" + f"{log10p_m7:.0f}" + "}"
    else:
        p_fmt_m7 = f"{pval_m7:.4e}"
    sig_m7 = '***' if (pval_m7 < 0.01 or (pval_m7 == 0.0 and not np.isnan(log10p_m7))) else ('**' if pval_m7 < 0.05 else ('*' if pval_m7 < 0.1 else ''))
    log(f"  Model 7 vs 1: DM={dm_m7:+.3f}  p={p_fmt_m7} {sig_m7}")
    results[6]['DM_stat'] = dm_m7
    results[6]['DM_pvalue'] = pval_m7
    results[6]['DM_log10_p'] = log10p_m7

    # ---- MCS 排名（所有模型完成后统一计算）----
    mse_dict = {r['Model']: r['MSE'] for r in results if not np.isnan(r['MSE'])}
    ranks = compute_mcs_rank(mse_dict)
    for r in results:
        r['MCS_rank'] = ranks.get(r['Model'], np.nan)

    # ---- 保存 ----
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "Table2_results.csv", index=False)
    for mid, Y_pred in predictions.items():
        np.save(out_dir / f"Y_pred_model{mid}.npy", Y_pred)

    # 保存拟合结果（系数/截距/scaler/列索引），供 DFL 等下游脚本加载
    import pickle
    model_dir = out_dir / "fitted_models"
    model_dir.mkdir(exist_ok=True)
    for mid, fitted in fitted_models.items():
        np.save(model_dir / f"coefs_model{mid}.npy", fitted['coefs'])
        np.save(model_dir / f"intercepts_model{mid}.npy", fitted['intercepts'])
        with open(model_dir / f"scaler_model{mid}.pkl", 'wb') as f:
            pickle.dump(fitted['scaler'], f)
        np.save(model_dir / f"feat_cols_model{mid}.npy", fitted['cols'])

    # ---- 打印 Table 2 ----
    log("\n" + "=" * 70)
    log("Table 2: OOS 预测精度对比")
    log("=" * 70)
    header = f"{'#':>2} {'Model':<20} {'MSE':>12} {'MAE':>12} {'DM':>8} {'p-val':>12} {'MCS':>4} {'Turnover':>10}"
    log(header)
    log("-" * 70)
    for r in results:
        def fmt(v, f, na='---'):
            return f.format(v) if not np.isnan(v) else na
        # p值: 若下溢为0则用log10格式
        if r.get('DM_pvalue') == 0.0 and not np.isnan(r.get('DM_log10_p', np.nan)):
            p_str = "10^{" + f"{r['DM_log10_p']:.0f}" + "}"
        else:
            p_str = fmt(r['DM_pvalue'], '{:.4e}')
        log(
            f"{r['Model']:>2} {r['Name']:<20} "
            f"{fmt(r['MSE'], '{:.4e}'):>12} {fmt(r['MAE'], '{:.4e}'):>12} "
            f"{fmt(r['DM_stat'], '{:+.3f}'):>8} {p_str:>12} "
            f"{fmt(r['MCS_rank'], '{:.0f}'):>4} {fmt(r['Turnover'], '{:.6f}'):>10}"
        )

    # ---- 验收检查 ----
    log("\n--- 验收检查 ---")
    m4_mse = results[3]['MSE']
    m3_mse = results[2]['MSE']
    m5_to  = results[4]['Turnover']
    m4_to  = results[3]['Turnover']
    m2_spa = results[1]['Sparsity']

    checks = [
        ("Model 4 MSE < Model 3 MSE (网络增量信息)",  m4_mse < m3_mse),
        ("Model 5 换手率 ≤ 1.5× Model 4",             m5_to <= 1.5 * m4_to),
        ("Model 2+ 稀疏度 ≥ 50%",                     m2_spa >= 0.5),
    ]
    for desc, ok in checks:
        log(f"  [{'✅' if ok else '❌'}] {desc}")

    log(f"\n输出: {out_dir / 'Table2_results.csv'}")
    log(f"      Y_pred_model1~5.npy")
    log("=" * 70)


if __name__ == "__main__":
    main()
