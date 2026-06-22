# ===================================================================
# 图形 Lasso GMVP 完整流程（单文件整合版）
# 论文: Network-Regularized Decision-Focused Forecasts of
#       High-Dimensional Realized Minimum-Variance Portfolio Weights
#
# 使用: python 完整流程.py
# ===================================================================
import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 路径配置（相对项目根目录，换机即用）
# ================================================================
_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CODE_DIR.parent.parent
DATA_DIR = _PROJECT_ROOT / "数据" / "1min_log_return"
OUT_DIR  = _PROJECT_ROOT / "图形Lasso"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# 全局参数
# ================================================================
K = 392
MAX_ITER = 2000
TOL_GLASSO = 1e-5
EPS_RIDGE = 5e-5
TOL_ZERO = 1e-8
ETA = 1e-4
RIDGE_FALLBACK = [1e-4, 5e-4, 1e-3, 5e-3]

# ================================================================
# 文件列表（缓存）
# ================================================================
_files_raw: List[str] = []
_dates: List[str] = []
_n_days: int = 0
_daily_minutes: Dict[int, int] = {}

def _ensure_file_list():
    global _files_raw, _dates, _n_days
    if _files_raw:
        return
    all_items = sorted(os.listdir(DATA_DIR))
    _files_raw = [f for f in all_items
                  if f.endswith("_1min_log_return.RData") and not f.startswith("1min")]
    _dates = [f.split(" ")[0] for f in _files_raw]
    _n_days = len(_files_raw)

def get_files_raw(): _ensure_file_list(); return _files_raw
def get_dates(): _ensure_file_list(); return _dates
def get_n_days(): _ensure_file_list(); return _n_days

def get_daily_minutes(idx: int) -> int:
    global _daily_minutes
    if idx not in _daily_minutes:
        rett1 = _load_day_core(idx)
        _daily_minutes[idx] = rett1.shape[1]
    return _daily_minutes[idx]

# ================================================================
# 数据加载
# ================================================================
def _load_day_core(idx: int) -> np.ndarray:
    import pyreadr
    _ensure_file_list()
    filepath = DATA_DIR / _files_raw[idx]
    result = pyreadr.read_r(str(filepath))
    return result["rett1"].values

def load_day(idx: int) -> np.ndarray:
    return _load_day_core(idx)

def preload_days(indices: List[int]) -> Dict[int, np.ndarray]:
    global _daily_minutes
    data = {}
    for i in indices:
        arr = _load_day_core(i)
        data[i] = arr
        _daily_minutes[i] = arr.shape[1]
    return data

# ================================================================
# 协方差计算
# ================================================================
def compute_raw_cov(rett1_arr: np.ndarray) -> np.ndarray:
    return rett1_arr @ rett1_arr.T

def cov_from_rett1(rett1_arr: np.ndarray, add_ridge: bool = True,
                   normalize: bool = False) -> np.ndarray:
    cov = rett1_arr @ rett1_arr.T
    if normalize:
        cov = cov / rett1_arr.shape[1]
    if add_ridge:
        cov.flat[::K + 1] += EPS_RIDGE
    return cov

def build_train_cov(raw_cov_dict, start, end, add_ridge=True):
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

# ================================================================
# Graphical Lasso 拟合
# ================================================================
def do_glasso(cov_mat: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.covariance import graphical_lasso
    all_ridges = [EPS_RIDGE] + RIDGE_FALLBACK
    for i, r in enumerate(all_ridges):
        c = cov_mat.copy()
        if i > 0:
            c.flat[::K + 1] += (r - EPS_RIDGE)
        try:
            cov_est, prec = graphical_lasso(
                emp_cov=c, alpha=lam, mode='cd',
                tol=TOL_GLASSO, max_iter=MAX_ITER)
            return prec, cov_est
        except (FloatingPointError, ValueError):
            continue
    c = cov_mat.copy()
    c.flat[::K + 1] += 1e-1 - EPS_RIDGE
    cov_est, prec = graphical_lasso(
        emp_cov=c, alpha=lam, mode='cd', tol=TOL_GLASSO,
        max_iter=MAX_ITER, enet_tol=1e-3)
    return prec, cov_est

# ================================================================
# GMVP 权重与邻接矩阵
# ================================================================
def w_from_prec(prec: np.ndarray) -> np.ndarray:
    ones = np.ones(K)
    w = prec @ ones
    denom = np.sum(w)
    if abs(denom) < 1e-15:
        raise ValueError("GMVP 权重分母 ≈ 0")
    return w / denom

def compute_adjacency(prec_mat: np.ndarray) -> np.ndarray:
    adj = (np.abs(prec_mat) > TOL_ZERO).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj

def daily_diagnostics(prec_mat, cov_est, converged) -> dict:
    K_local = prec_mat.shape[0]
    adj_mat = compute_adjacency(prec_mat)
    upper_nz = int(np.sum(np.triu(adj_mat, k=1)))
    density = upper_nz / (K_local * (K_local - 1) // 2) if K_local > 1 else 0.0
    cond_val = np.nan
    if converged and cov_est is not None:
        try:
            eigvals = np.linalg.eigvalsh(cov_est)
            if eigvals[0] > 0:
                cond_val = float(eigvals[-1] / eigvals[0])
        except Exception:
            pass
    return {"adj_mat": adj_mat, "network_density": density,
            "n_nonzero": upper_nz, "cond_val": cond_val}

# ================================================================
# 日志
# ================================================================
_LOG_FILE: Optional[Path] = None

def set_log_file(path: Path):
    global _LOG_FILE
    _LOG_FILE = path

def log(msg: str):
    print(msg, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ================================================================
# 步骤1 配置
# ================================================================
BIC_SAMPLE_SIZE = 30
BIC_RANDOM_SEED = 42
BIC_LAMBDA_RANGE = (-10, -3)
BIC_N_LAMBDA = 30
L_TRAIN = 40
L_VAL = 60
FINE_MULTIPLIER = 1.0
FINE_N_LAMBDA = 11
N_WORKERS = max(1, mp.cpu_count() - 1)

# 步骤1 全局变量（用于步骤2）
lambda_opt = None
lambda_bic = None

# ================================================================
# 步骤1 Worker 函数
# ================================================================
def _worker_bic_one_day(args):
    day_idx, rett1_arr, lambda_candidates = args
    M_day = rett1_arr.shape[1]
    cov_day = cov_from_rett1(rett1_arr, add_ridge=True)
    n_lam = len(lambda_candidates)
    bic_vals = np.full(n_lam, np.nan)
    precisions = [None] * n_lam

    try:
        from sklearn.covariance import graphical_lasso_path as gl_path
        lam_desc = lambda_candidates[::-1]
        _, precs_desc, _ = gl_path(
            emp_cov=cov_day, alphas=lam_desc, mode='cd',
            tol=TOL_GLASSO, enet_tol=1e-3, max_iter=MAX_ITER, return_n_iter=True)
        for k, prec_k in enumerate(precs_desc):
            precisions[n_lam - 1 - k] = prec_k
    except Exception:
        pass

    for li, lam in enumerate(lambda_candidates):
        prec = precisions[li]
        if prec is None:
            try:
                prec, _ = do_glasso(cov_day, lam)
            except Exception:
                continue
        try:
            sign, logdet = np.linalg.slogdet(prec)
            if sign <= 0:
                continue
            loglik = 0.5 * (logdet - np.trace(prec @ cov_day))
            nz = np.sum(np.abs(np.triu(prec, k=1)) > TOL_ZERO)
            bic_vals[li] = -2.0 * loglik + nz * np.log(M_day)
        except Exception:
            continue
    return day_idx, bic_vals

def _worker_eval_lambda(args):
    lam, raw_cov_dict, daily_mins, validation_start, n_days, L_train = args
    total_var = 0.0; total_TO = 0.0; n_valid = 0
    prev_w = None
    for v in range(validation_start, n_days):
        train_start = v - L_train
        if train_start < 0:
            continue
        try:
            cov_train = build_train_cov(raw_cov_dict,
                                        train_start, v - 1, add_ridge=True)
        except ValueError:
            continue
        try:
            prec_train, _ = do_glasso(cov_train, lam)
            w_curr = w_from_prec(prec_train)
        except Exception:
            continue
        raw_cov_val = raw_cov_dict.get(v)
        if raw_cov_val is None:
            continue
        port_var = float(w_curr @ raw_cov_val @ w_curr)
        total_var += port_var; n_valid += 1
        if prev_w is not None:
            total_TO += float(np.sum(np.abs(w_curr - prev_w)))
        prev_w = w_curr
    if n_valid == 0:
        return (np.inf, np.nan, np.nan, 0)
    avg_var = total_var / n_valid
    avg_TO = total_TO / (n_valid - 1) if n_valid > 1 else 0.0
    return (lam, float(avg_var + ETA * avg_TO), avg_var, avg_TO, n_valid)

# ================================================================
# 步骤1 主程序
# ================================================================
def step1():
    global lambda_opt, lambda_bic
    set_log_file(OUT_DIR / "step1_log.txt")
    files_raw = get_files_raw()
    dates = get_dates()
    n_days = get_n_days()

    log("=" * 65)
    log("  步骤1: §4.1 λ_Ω 选择")
    log("=" * 65)
    log(f"交易日总数: {n_days}  资产数 K: {K}  并行核心: {N_WORKERS}")
    log(f"BIC 样本: {BIC_SAMPLE_SIZE}  训练窗口: {L_TRAIN}  验证窗口: {L_VAL}")
    log(f"Ridge ε: {EPS_RIDGE}  η: {ETA}")

    # ---- 阶段1: BIC 初筛 ----
    log("\n" + "-" * 50 + "\n阶段1: BIC 初筛\n" + "-" * 50)
    rng = np.random.default_rng(BIC_RANDOM_SEED)
    bic_population_end = max(1, n_days - L_VAL - L_TRAIN)
    actual_sample_size = max(min(BIC_SAMPLE_SIZE, bic_population_end), 10)
    sample_indices = sorted(rng.choice(bic_population_end, size=actual_sample_size, replace=False))
    log(f"BIC 样本日: {actual_sample_size} 天 (种子={BIC_RANDOM_SEED})")

    lambda_candidates = np.logspace(*BIC_LAMBDA_RANGE, BIC_N_LAMBDA)
    n_lam = len(lambda_candidates)
    log(f"λ 候选: {n_lam} ({lambda_candidates[0]:.2e}～{lambda_candidates[-1]:.2e})")

    t0 = time.time()
    sample_data = preload_days(sample_indices)
    log(f"预加载耗时: {time.time()-t0:.1f}s")

    t0 = time.time()
    bic_summary = np.full((n_lam, actual_sample_size), np.nan)
    sample_to_col = {si: ci for ci, si in enumerate(sample_indices)}
    tasks = [(si, sample_data[si], lambda_candidates) for si in sample_indices]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_bic_one_day, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            day_idx, bic_vals = future.result()
            bic_summary[:, sample_to_col[day_idx]] = bic_vals
            done += 1
            if done % 5 == 0 or done == actual_sample_size:
                elapsed_bic = time.time() - t0
                log(f"  BIC: {done}/{actual_sample_size}  "
                    f"刚完成: {dates[day_idx]}  耗时: {elapsed_bic:.0f}s")

    avg_bic = np.nanmean(bic_summary, axis=1)
    if not np.any(~np.isnan(avg_bic)):
        log("错误：无有效 BIC"); sys.exit(1)
    best_bic_idx = np.nanargmin(avg_bic)
    lambda_bic = lambda_candidates[best_bic_idx]
    if best_bic_idx == 0:
        log(f"⚠ BIC λ={lambda_bic:.2e} 位于下界")
    elif best_bic_idx == n_lam - 1:
        log(f"⚠ BIC λ={lambda_bic:.2e} 位于上界")
    log(f"\nBIC 最优 λ: {lambda_bic:.2e}  (BIC={avg_bic[best_bic_idx]:.1f}) 耗时: {time.time()-t0:.1f}s")

    # ---- 阶段2: 验证窗精选 ----
    log("\n" + "-" * 50 + "\n阶段2: 验证窗精选\n" + "-" * 50)
    validation_start = max(L_TRAIN, n_days - L_VAL)
    if validation_start >= n_days:
        log("错误：验证窗超出范围"); sys.exit(1)
    log(f"验证窗: {dates[validation_start]}～{dates[-1]}")

    lambda_fine = lambda_bic * np.logspace(-FINE_MULTIPLIER, FINE_MULTIPLIER, FINE_N_LAMBDA)
    for li, lam in enumerate(lambda_fine):
        marker = " ← BIC 最优" if abs(lam - lambda_bic) < 1e-15 else ""
        log(f"  λ[{li:2d}]={lam:.3e}{marker}")

    load_range_start = max(0, validation_start - L_TRAIN)
    all_indices = list(range(load_range_start, n_days))
    log(f"\n预计算 {len(all_indices)} 天协方差 ...")
    t_prep = time.time()
    raw_cov_dict: Dict[int, np.ndarray] = {}
    daily_mins: Dict[int, int] = {}
    for i in all_indices:
        rett1 = load_day(i)
        raw_cov_dict[i] = compute_raw_cov(rett1)
        daily_mins[i] = rett1.shape[1]
    log(f"预计算耗时: {time.time()-t_prep:.1f}s")

    t0 = time.time()
    results_map = {}
    tasks = [(lam, raw_cov_dict, daily_mins, validation_start, n_days, L_TRAIN)
             for lam in lambda_fine]
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, FINE_N_LAMBDA)) as executor:
        futures = {executor.submit(_worker_eval_lambda, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            lam, score, avg_var, avg_TO, n_valid = future.result()
            results_map[lam] = (score, avg_var, avg_TO, n_valid)
            done += 1
            log(f"  [{done}/{FINE_N_LAMBDA}] λ={lam:.2e} 得分={score:.4e} Var={avg_var:.4e} TO={avg_TO:.4f} 有效={n_valid}")

    valid_scores = np.array([results_map.get(lam, (np.inf,))[0] for lam in lambda_fine])
    best_idx = int(np.nanargmin(valid_scores))
    lambda_opt = lambda_fine[best_idx]
    best_score = results_map.get(lambda_opt, (np.nan,)*4)
    log(f"\n最优 λ_Ω: {lambda_opt:.2e}  得分: {best_score[0]:.4e}  耗时: {time.time()-t0:.1f}s")

    # 保存
    pd.DataFrame({"lambda": lambda_fine,
        "valid_score": [results_map.get(l, (np.nan,)*4)[0] for l in lambda_fine],
        "avg_variance": [results_map.get(l, (np.nan,)*4)[1] for l in lambda_fine],
        "avg_turnover": [results_map.get(l, (np.nan,)*4)[2] for l in lambda_fine],
        "n_valid_days": [results_map.get(l, (np.nan,)*4)[3] for l in lambda_fine],
        "is_optimal": [i == best_idx for i in range(FINE_N_LAMBDA)]
    }).to_csv(OUT_DIR / "lambda_selection_diagnostics.csv", index=False)

    with open(OUT_DIR / "lambda_opt.txt", "w", encoding="utf-8") as f:
        f.write(f"{lambda_opt}\n{lambda_bic}\n")

    bic_diag = pd.DataFrame({"lambda": lambda_candidates, "avg_bic": avg_bic,
        "is_bic_optimal": [i == best_bic_idx for i in range(n_lam)]})
    for ci, si in enumerate(sample_indices):
        bic_diag[f"BIC_day_{si}_{dates[si]}"] = bic_summary[:, ci]
    bic_diag.to_csv(OUT_DIR / "bic_diagnostics.csv", index=False)

    log(f"\n步骤1 完成。最优 λ = {lambda_opt:.2e}")

# ================================================================
# 步骤2 配置
# ================================================================
PROGRESS_INTERVAL = 100

# ================================================================
# 步骤2 单日处理函数
# ================================================================
def process_one_day(args):
    day_idx, fname, lam_opt = args
    date_str = fname.split(" ")[0]
    try:
        rett1_arr = load_day(day_idx)
        M_day = rett1_arr.shape[1]
        cov_raw = cov_from_rett1(rett1_arr, add_ridge=True)
        converged = True
        try:
            prec_mat, cov_est = do_glasso(cov_raw, lam_opt)
        except Exception:
            cov_fb = cov_raw.copy()
            cov_fb.flat[::K + 1] += 1e-2
            try:
                prec_mat, cov_est = do_glasso(cov_fb, lam_opt)
                converged = False
            except Exception:
                raise RuntimeError("GLasso 两级回退均失败")
        w_tilde = w_from_prec(prec_mat)
        raw_cov = rett1_arr @ rett1_arr.T
        rpv = float(w_tilde @ raw_cov @ w_tilde)
        diag = daily_diagnostics(prec_mat, cov_est, converged)
        net_degree = diag["adj_mat"].sum(axis=1).astype(np.int32)
        return {"day": day_idx, "date": date_str,
                "w_tilde": w_tilde.astype(np.float32), "rpv": rpv,
                "prec_mat": prec_mat.astype(np.float32), "adj_mat": diag["adj_mat"],
                "net_degree": net_degree, "network_density": diag["network_density"],
                "n_nonzero": diag["n_nonzero"], "cond_val": diag["cond_val"],
                "converged": converged, "M": M_day, "success": True, "error": None}
    except Exception as e:
        return {"day": day_idx, "date": date_str,
                "w_tilde": np.full(K, np.nan, dtype=np.float32), "rpv": np.nan,
                "prec_mat": None, "adj_mat": None,
                "net_degree": np.full(K, np.nan, dtype=np.float32),
                "network_density": np.nan, "n_nonzero": np.nan, "cond_val": np.nan,
                "converged": False, "M": 0, "success": False, "error": str(e)}

# ================================================================
# 步骤2 主程序
# ================================================================
def step2():
    set_log_file(OUT_DIR / "step2_log.txt")
    files_raw = get_files_raw()
    dates = get_dates()
    n_days = get_n_days()

    lambda_opt_path = OUT_DIR / "lambda_opt.txt"
    try:
        with open(lambda_opt_path, "r", encoding="utf-8") as f:
            lam_opt = float(f.readline().strip())
            lam_bic = float(f.readline().strip())
        log(f"读取 λ_Ω: {lam_opt:.2e}  (BIC: {lam_bic:.2e})")
    except FileNotFoundError:
        log("未找到 lambda_opt.txt — 先运行步骤1"); sys.exit(1)

    log(f"交易日: {n_days}  资产: {K}  并行: {N_WORKERS}  λ: {lam_opt:.2e}")

    tasks = [(i, fname, lam_opt) for i, fname in enumerate(files_raw)]
    results_dict: Dict[int, Dict] = {}
    failed_days = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_one_day, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as exc:
                day_idx = futures[future]
                log(f"  ⚠ 进程崩溃 {day_idx}({dates[day_idx]}): {exc}")
                res = {"day": day_idx, "date": dates[day_idx],
                       "w_tilde": np.full(K, np.nan, dtype=np.float32), "rpv": np.nan,
                       "prec_mat": None, "adj_mat": None,
                       "net_degree": np.full(K, np.nan, dtype=np.float32),
                       "network_density": np.nan, "n_nonzero": np.nan,
                       "cond_val": np.nan, "converged": False, "M": 0,
                       "success": False, "error": f"进程级异常: {exc}"}
            results_dict[res["day"]] = res
            if not res["success"]:
                failed_days.append(res)
            done += 1
            if done % PROGRESS_INTERVAL == 0 or done == n_days:
                elapsed = time.time() - t_start
                speed = done / elapsed if elapsed > 0 else 0
                eta_sec = (n_days - done) / speed if speed > 0 else 0
                log(f"  进度: {done}/{n_days} ({100*done/n_days:.1f}%) "
                    f"耗时: {elapsed:.0f}s  速度: {speed:.2f}天/s  "
                    f"ETA: {eta_sec:.0f}s  失败: {len(failed_days)}")

    results = [results_dict[i] for i in range(n_days)]
    elapsed = time.time() - t_start
    log(f"\n计算完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")

    if failed_days:
        log(f"\n⚠ {len(failed_days)} 天失败")
        pd.DataFrame([{"index": f["day"], "date": f["date"], "error": f["error"]}
                      for f in failed_days]).to_csv(OUT_DIR / "failed_days.csv", index=False)

    # 汇总
    reg_weights = np.array([r["w_tilde"] for r in results])
    daily_stats = pd.DataFrame({
        "date": [r["date"] for r in results],
        "rpv": [r["rpv"] for r in results],
        "network_density": [r["network_density"] for r in results],
        "n_nonzero": [r["n_nonzero"] for r in results],
        "cond_val": [r["cond_val"] for r in results],
        "converged": [r["converged"] for r in results],
        "M_minutes": [r["M"] for r in results],
        "success": [r["success"] for r in results],
    })

    TO = [np.nan]
    for t in range(1, n_days):
        if daily_stats["success"].iloc[t] and daily_stats["success"].iloc[t - 1]:
            TO.append(float(np.sum(np.abs(reg_weights[t] - reg_weights[t - 1]))))
        else:
            TO.append(np.nan)
    daily_stats["turnover"] = TO

    conv_mask = np.array([r["converged"] for r in results])
    conv_rate = np.mean(conv_mask) * 100
    log(f"收敛率: {conv_rate:.1f}%")

    # Table 1
    def build_row(vals, name):
        clean = vals[np.isfinite(vals)]
        if len(clean) == 0:
            return pd.DataFrame({"Variable":[name],"Mean":[np.nan],"Std":[np.nan],
                                 "P5":[np.nan],"Median":[np.nan],"P95":[np.nan],"Obs":[0]})
        return pd.DataFrame({"Variable":[name],"Mean":[np.mean(clean)],"Std":[np.std(clean)],
                             "P5":[np.percentile(clean,5)],"Median":[np.median(clean)],
                             "P95":[np.percentile(clean,95)],"Obs":[len(clean)]})

    rows = [
        build_row(np.nanmean(reg_weights, axis=1), "GMVP 权重 (截面均值)"),
        build_row(np.nanstd(reg_weights, axis=1), "GMVP 权重 (截面标准差)"),
        build_row(daily_stats["rpv"].dropna().values, "已实现组合方差 (RPV)"),
        build_row(daily_stats["n_nonzero"].dropna().values, "精度矩阵非零元数量"),
        build_row(daily_stats["network_density"].dropna().values, "网络密度"),
        build_row(np.nanmean(np.array([r["net_degree"] for r in results]), axis=1), "平均网络度"),
        build_row(pd.to_numeric(daily_stats["turnover"], errors="coerce").dropna().values, "换手率"),
        build_row(daily_stats["cond_val"].dropna().values, "条件数"),
        build_row(daily_stats["M_minutes"].dropna().values, "日内分钟数"),
        build_row(np.nanmax(reg_weights, axis=1) - np.nanmin(reg_weights, axis=1), "GMVP 权重极值"),
    ]
    table1 = pd.concat(rows, ignore_index=True)
    log("\n" + table1.to_string(index=False, float_format="%.4f"))

    neg_prop = np.nanmean((reg_weights < 0).mean(axis=1))
    log(f"平均负权重: {neg_prop*100:.2f}%")

    # 保存
    table1.to_csv(OUT_DIR / "Table1_Descriptive_Statistics.csv", index=False)
    weight_df = pd.DataFrame(reg_weights,
        columns=[f"Asset_{i+1}" for i in range(K)], index=dates)
    weight_df.to_csv(OUT_DIR / "GMVP_weights_daily.csv")
    daily_stats.to_csv(OUT_DIR / "Daily_Statistics.csv", index=False)

    last_prec = results[-1].get("prec_mat")
    if last_prec is not None:
        cols = [f"A_{i+1}" for i in range(K)]
        pd.DataFrame(last_prec, columns=cols, index=cols).to_csv(
            OUT_DIR / "Precision_Matrix_Last.csv")

    prec_dir = OUT_DIR / "daily_precision"
    adj_dir = OUT_DIR / "daily_adjacency"
    prec_dir.mkdir(parents=True, exist_ok=True)
    adj_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        date_str = r["date"]
        if r["prec_mat"] is not None:
            np.savez_compressed(prec_dir / f"prec_{date_str}.npz", precision=r["prec_mat"])
        if r["adj_mat"] is not None:
            np.savez_compressed(adj_dir / f"adj_{date_str}.npz", adjacency=r["adj_mat"])
        if (i + 1) % 500 == 0:
            log(f"  已保存 {i+1}/{n_days} ...")

    with open(OUT_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"交易日数: {n_days}  资产: {K}\n")
        f.write(f"日期范围: {dates[0]}～{dates[-1]}\n")
        f.write(f"λ_Ω (BIC): {lam_bic:.2e}  λ_Ω (精选): {lam_opt:.2e}\n")
        f.write(f"收敛率: {conv_rate:.1f}%  失败: {len(failed_days)}天\n")
        f.write(f"平均网络密度: {daily_stats['network_density'].mean():.4f}\n")
        f.write(f"平均非零: {daily_stats['n_nonzero'].mean():.1f}\n")
        f.write(f"平均RPV: {daily_stats['rpv'].mean():.4e}\n")
        f.write(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)\n")

    log(f"\n全部结果已保存至: {OUT_DIR}")
    log("=" * 65)
    log("  全流程完成")
    log("=" * 65)

# ================================================================
# 入口
# ================================================================
if __name__ == "__main__":
    log("=" * 65)
    log("  图形 Lasso GMVP 完整流程")
    log("=" * 65)
    step1()
    print("\n")
    step2()
