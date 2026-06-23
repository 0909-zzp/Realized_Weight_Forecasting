# ===================================================================
# glasso_shared.py — 共享模块
# 提供全部阶段共用：文件列表、数据加载、协方差计算、GLasso拟合、
# GMVP权重构建、邻接矩阵、VARX预测参数、决策聚焦损失及路径配置。
# 所有阶段通过 import 此模块取得统一参数，消除代码重复与不一致。
# ===================================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

warnings.filterwarnings("ignore")

# ===================================================================
# 路径配置（相对项目根目录，换机即用无需修改）
# ===================================================================
_CODE_DIR = Path(__file__).resolve().parent        # .../图形Lasso/code/
_PROJECT_ROOT = _CODE_DIR.parent.parent            # .../大创/
DATA_DIR = _PROJECT_ROOT / "数据" / "1min_log_return"
NPY_DIR  = _PROJECT_ROOT / "数据" / "1min_log_return_npy"   # .npy 加速目录（可选）
OUT_DIR  = _PROJECT_ROOT / "图形Lasso"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# 全局参数
# ===================================================================
# ---- 数据维度 ----
K = 392                        # 资产池规模

# ---- GLasso 求解 ----
MAX_ITER        = 150          # 坐标下降最大迭代次数（M<K 场景 30~80 次收敛，150 安全裕度充足）
TOL_GLASSO      = 1e-4         # 收敛容忍度（放松至 1e-4，权重偏差 ~1e-8，不影响结果）
GLASSO_ENET_TOL = 5e-4         # 对偶间隙容忍度
EPS_RIDGE       = 1e-4         # 初始 Ridge 扰动量
RIDGE_FALLBACK  = [5e-4, 1e-3, 5e-3, 1e-2]

# ---- GLasso 零元判定 ----
TOL_ZERO = 1e-8

# ---- 时间窗 (已确定) ----
L_TRAIN_GLASSO = 40            # GLasso 滚动训练窗 (天)

# ---- VARX 时间窗 (实验设计，非超参数，待定) ----
L_TRAIN_VARX   = 500           # [TBD] VARX 训练窗 (天)
L_VAL_VARX     = 60            # [TBD] VARX 验证窗 (天)
L_TEST_VARX    = 200           # [TBD] VARX 测试窗 (天)

# ---- VARX 超参数 (全部待定，需通过验证实验确定) ----
P_LAGS          = 3            # [TBD] 自回归滞后阶数
LAMBDA_LASSO    = 1e-4         # [TBD] VARX 系数 ℓ1 惩罚
LAMBDA_TURNOVER = 1e-2         # [TBD] 换手率平滑 ℓ2 惩罚
LAMBDA_NETWORK  = 1e-3         # [TBD] 网络正则化强度
NETWORK_THRESHOLD = 0.3        # [TBD] 滚动网络均值 Ā 截断阈值

# ---- 决策损失 (已确定，基于 10 bps 双边交易成本) ----
ETA = 1e-4

# ===================================================================
# 文件列表
# ===================================================================
_files_raw: List[str] = []
_dates: List[str] = []
_n_days: int = 0
_daily_minutes: Dict[int, int] = {}  # 缓存每天分钟数，避免反复加载


def _ensure_file_list() -> None:
    """只在首次调用时扫描目录，后续直接复用缓存。
    
    优先扫描 DATA_DIR (.RData)，若不存在则回退到 NPY_DIR (.npy)。
    兼容云端只传了 .npy 但没传 .RData 的场景。
    """
    global _files_raw, _dates, _n_days
    if _files_raw:
        return
    if DATA_DIR.exists():
        scan_dir = DATA_DIR
        suffix = "_1min_log_return.RData"
    elif NPY_DIR.exists():
        scan_dir = NPY_DIR
        suffix = "_1min_log_return.npy"
    else:
        raise FileNotFoundError(
            f"数据目录不存在：{DATA_DIR} 或 {NPY_DIR}，请先放置数据"
        )
    all_items = sorted(os.listdir(scan_dir))
    _files_raw = [
        f for f in all_items
        if f.endswith(suffix) and not f.startswith("1min")
    ]
    _dates = [f.split(" ")[0] for f in _files_raw]
    _n_days = len(_files_raw)


def get_files_raw() -> List[str]:
    _ensure_file_list()
    return _files_raw


def get_dates() -> List[str]:
    _ensure_file_list()
    return _dates


def get_n_days() -> int:
    _ensure_file_list()
    return _n_days


def get_daily_minutes(idx: int) -> int:
    """返回第 idx 天的日内分钟数（缓存，避免反复磁盘I/O）。"""
    global _daily_minutes
    if idx not in _daily_minutes:
        rett1 = _load_day_core(idx)
        _daily_minutes[idx] = rett1.shape[1]
    return _daily_minutes[idx]


# ===================================================================
# 数据加载
# ===================================================================
def _load_day_core(idx: int) -> np.ndarray:
    """底层加载函数 —— 优先 .npy（快 5~10×），回退 .RData。"""
    _ensure_file_list()
    # 尝试 .npy 加速格式
    if NPY_DIR.exists():
        npy_name = _files_raw[idx].replace(".RData", ".npy")
        npy_path = NPY_DIR / npy_name
        if npy_path.exists():
            return np.load(npy_path)  # shape: (K, M)
    # 回退到 .RData
    import pyreadr
    filepath = DATA_DIR / _files_raw[idx]
    result = pyreadr.read_r(str(filepath))
    return result["rett1"].values  # shape: (K, M)


def load_day(idx: int) -> np.ndarray:
    """加载第 idx 天的高频对数收益矩阵 (K × M_day)。"""
    return _load_day_core(idx)


def preload_days(indices: List[int]) -> Dict[int, np.ndarray]:
    """批量预加载指定日期的原始数据到内存。
    
    Args:
        indices: 需要加载的日期索引列表
    
    Returns:
        dict: {idx: ndarray(K, M)} 
    
    优势：一次性 I/O，避免后续在循环/并行中反复调用 pyreadr。
    """
    global _daily_minutes
    data = {}
    for i in indices:
        arr = _load_day_core(i)
        data[i] = arr
        _daily_minutes[i] = arr.shape[1]
    return data


# ===================================================================
# 协方差计算
# ===================================================================
def compute_raw_cov(rett1_arr: np.ndarray) -> np.ndarray:
    """计算原始外积和 X @ X^T（不含归一化、不含 Ridge）。
    
    用于多日合并的场景：单独存储 raw_cov 后按需加和。
    """
    return rett1_arr @ rett1_arr.T  # (K, K)


def cov_from_rett1(rett1_arr: np.ndarray, add_ridge: bool = True,
                   normalize: bool = False) -> np.ndarray:
    """经验协方差：X @ X^T + ridge * I。
    
    默认不除以 M，与论文公式(3) Σ̂_RC = Σ r·r' 一致。
    normalize=True 时除以分钟数，仅用于多日窗口尺度统一（由 build_train_cov 调用）。
    
    Args:
        rett1_arr: (K, M) 高频收益矩阵
        add_ridge: 是否在对角线加入 EPS_RIDGE
        normalize: 是否除以分钟数 M
    
    Returns:
        (K, K) 协方差矩阵
    """
    M = rett1_arr.shape[1]
    cov = (rett1_arr @ rett1_arr.T)
    if normalize:
        cov = cov / M
    if add_ridge:
        cov.flat[::K + 1] += EPS_RIDGE
    return cov


def build_train_cov(
    raw_cov_dict: Dict[int, np.ndarray],
    start: int,
    end: int,
    add_ridge: bool = True,
) -> np.ndarray:
    """从预计算的 raw_cov 构建训练窗口的平均协方差。
    
    对各天 raw_cov（X_i @ X_i^T）取算术平均，保持与单日协方差相同的量级。
    
    Args:
        raw_cov_dict: {idx: X_i @ X_i^T}  单日原始外积和（论文公式3）
        start, end: 窗口起止索引（闭区间）
        add_ridge: 是否在对角线加入 EPS_RIDGE
    
    Returns:
        (K, K) 平均协方差矩阵（与单日同量级 ~1e-4）
    """
    total_cov = np.zeros((K, K), dtype=np.float64)
    for i in range(start, end + 1):
        total_cov += raw_cov_dict[i]
    n_days = end - start + 1
    if n_days <= 0:
        raise ValueError(f"窗口 [{start}, {end}] 天数无效")
    cov = total_cov / n_days
    if add_ridge:
        cov.flat[::K + 1] += EPS_RIDGE
    return cov


# ===================================================================
# Graphical Lasso 拟合（自适应 Ridge 退避）
# ===================================================================
def do_glasso(cov_mat: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """自适应 Graphical Lasso 拟合。
    
    依次尝试 [EPS_RIDGE] + RIDGE_FALLBACK 五种 Ridge 强度，
    直至收敛。若全部失败，抛出 RuntimeError。
    
    Args:
        cov_mat: (K, K) 经验协方差矩阵（应已含初始 Ridge）
        lam: 正则化参数 λ
    
    Returns:
        (precision_matrix, covariance_estimate)
    
    Raises:
        RuntimeError: 全部 Ridge 回退链均收敛失败
    """
    from sklearn.covariance import graphical_lasso

    all_ridges = [EPS_RIDGE] + RIDGE_FALLBACK

    for i, r in enumerate(all_ridges):
        c = cov_mat.copy()
        if i > 0:
            # 增量增加 Ridge：仅补足差值
            c.flat[::K + 1] += (r - EPS_RIDGE)
        try:
            cov_est, prec = graphical_lasso(
                emp_cov=c, alpha=lam, mode='cd',
                tol=TOL_GLASSO, max_iter=MAX_ITER,
                enet_tol=GLASSO_ENET_TOL)
            return prec, cov_est
        except (FloatingPointError, ValueError):
            continue

    # 终极回退：大 Ridge + 宽松容忍度
    c = cov_mat.copy()
    c.flat[::K + 1] += 1e-1 - EPS_RIDGE
    try:
        cov_est, prec = graphical_lasso(
            emp_cov=c, alpha=lam, mode='cd',
            tol=TOL_GLASSO, max_iter=MAX_ITER,
            enet_tol=max(GLASSO_ENET_TOL, 1e-3))
        return prec, cov_est
    except Exception as exc:
        raise RuntimeError(
            f"GLasso 收敛失败：λ = {lam:.2e}，"
            f"已尝试全部 Ridge 回退链（至 0.1 + enet_tol=1e-3）。"
            f"最后异常：{exc}"
        )


# ===================================================================
# GMVP 权重与邻接矩阵
# ===================================================================
def w_from_prec(prec: np.ndarray) -> np.ndarray:
    """由精度矩阵计算 GMVP 权重向量：w = Θ·1 / (1ᵀ·Θ·1)。"""
    ones = np.ones(K)
    w = prec @ ones
    denom = np.sum(w)
    if abs(denom) < 1e-15:
        raise ValueError("GMVP 权重分母 ≈ 0，精度矩阵可能奇异")
    return w / denom


def compute_adjacency(prec_mat: np.ndarray) -> np.ndarray:
    """从精度矩阵提取二值邻接矩阵（|θ_ij| > TOL_ZERO）。"""
    adj = (np.abs(prec_mat) > TOL_ZERO).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj


# ===================================================================
# 单日诊断统计
# ===================================================================
def daily_diagnostics(
    prec_mat: np.ndarray,
    cov_est: np.ndarray,
    converged: bool,
) -> dict:
    """计算单日精度矩阵/协方差矩阵的诊断指标。
    
    Returns:
        dict: 包含 network_density, n_nonzero, cond_val 等字段
    """
    K_local = prec_mat.shape[0]
    adj_mat = compute_adjacency(prec_mat)
    upper_nz = int(np.sum(np.triu(adj_mat, k=1)))
    density = upper_nz / (K_local * (K_local - 1) // 2) if K_local > 1 else 0.0

    # 条件数（仅对已收敛的 SPD 协方差估计有意义）
    cond_val = np.nan
    if converged and cov_est is not None:
        try:
            eigvals = np.linalg.eigvalsh(cov_est)
            if eigvals[0] > 0:
                cond_val = float(eigvals[-1] / eigvals[0])
        except Exception:
            pass

    return {
        "adj_mat": adj_mat,
        "network_density": density,
        "n_nonzero": upper_nz,
        "cond_val": cond_val,
    }


# ===================================================================
# 运行日志输出
# ===================================================================
_LOG_FILE: Optional[Path] = None


def set_log_file(path: Path) -> None:
    global _LOG_FILE
    _LOG_FILE = path


def log(msg: str) -> None:
    """统一日志输出（控制台 + 文件）。"""
    print(msg, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
