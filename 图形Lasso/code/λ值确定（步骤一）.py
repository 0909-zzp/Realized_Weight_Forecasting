# ===================================================================
# 步骤1: §4.1 λ_Ω 选择 (BIC 初筛 + 内部验证窗精选)
#
# ===================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ---- 导入共享模块 ----
from 共享模块 import (
    K, MAX_ITER, TOL_GLASSO, EPS_RIDGE, TOL_ZERO, ETA,
    RIDGE_FALLBACK,
    DATA_DIR, OUT_DIR,
    get_files_raw, get_dates, get_n_days, get_daily_minutes,
    load_day, preload_days,
    compute_raw_cov, cov_from_rett1, build_train_cov,
    do_glasso, w_from_prec, log, set_log_file,
)

# ===================================================================
# §4.1 阶段配置（所有硬编码参数集中于此并有注释）
# ===================================================================
BIC_SAMPLE_SIZE = 30         # BIC 初筛的样本日数量
BIC_RANDOM_SEED = 42         # 随机种子
BIC_LAMBDA_RANGE = (-10, -3) # BIC λ 搜索范围
BIC_N_LAMBDA = 30            # BIC λ 候选数量
L_TRAIN = 40                 # 滚动训练窗口交易日天数
L_VAL = 60                   # 内部验证窗天数
FINE_MULTIPLIER = 1.0        # 精细网格的半宽倍数
FINE_N_LAMBDA = 11           # 精细网格 λ 候选数量

# 并行核心数
N_WORKERS = max(1, mp.cpu_count() - 1)


# ===================================================================
# 工作进程顶层函数（必须在模块级别定义，Windows spawn 需要）
# ===================================================================
def _worker_bic_one_day(args: Tuple[int, np.ndarray, np.ndarray]) -> Tuple[int, np.ndarray]:
    """计算单日所有 λ 候选的 BIC 值。
    
    使用 sklearn 的 graphical_lasso_path 一次性计算整条正则化路径。
    暖启动（Warm Start）使 30 个 λ 的耗时接近单次独立拟合
    """
    day_idx, rett1_arr, lambda_candidates = args
    M_day = rett1_arr.shape[1]
    cov_day = cov_from_rett1(rett1_arr, add_ridge=True)

    n_lam = len(lambda_candidates)
    bic_vals = np.full(n_lam, np.nan)
    precisions = [None] * n_lam

    # ---- 尝试路径法（暖启动，极大加速） ----
    try:
        from sklearn.covariance import graphical_lasso_path as gl_path

        # gl_path 要求 λ 递减，且内部迭代次数受 max_iter 约束
        lam_desc = lambda_candidates[::-1]
        _, precs_desc, _ = gl_path(
            emp_cov=cov_day,
            alphas=lam_desc,
            mode='cd',
            tol=TOL_GLASSO,
            enet_tol=1e-3,
            max_iter=MAX_ITER,
            return_n_iter=True,
        )
        # gl_path 返回的顺序与 lam_desc 一致，需反转映射回原顺序
        for k, prec_k in enumerate(precs_desc):
            precisions[n_lam - 1 - k] = prec_k
    except Exception:
        pass  # 路径法失败则回退到逐 λ 独立拟合

    # ---- 逐 BIC 计算（优先用路径结果） ----
    for li, lam in enumerate(lambda_candidates):
        prec = precisions[li]
        if prec is None:
            # 路径法未覆盖此 λ → 回退独立拟合
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


def _worker_eval_lambda(args: Tuple[float, Dict[int, np.ndarray],
                                    Dict[int, int],
                                    int, int, int]) -> Tuple[float, float, float, int]:
    """验证窗内评估单个 λ 的得分（用于并行）。

    Args:
        lam: λ 值
        raw_cov_dict: {idx: X_i @ X_i^T} 原始外积（无归一化、无 Ridge）
        daily_mins: {idx: M_i}
        validation_start: 验证窗起始索引
        n_days: 总交易日数
        L_train: 训练窗口大小

    Returns:
        (score, avg_var, avg_TO, n_valid)
    """
    lam, raw_cov_dict, daily_mins, validation_start, n_days, L_train = args

    total_var = 0.0
    total_TO = 0.0
    n_valid = 0

    prev_w: Optional[np.ndarray] = None

    for v in range(validation_start, n_days):
        # ---- 训练窗: [v - L_train, v - 1] ----
        train_start = v - L_train
        if train_start < 0:
            continue

        try:
            cov_train = build_train_cov(
                raw_cov_dict,
                train_start, v - 1,
                add_ridge=True,
            )
        except ValueError:
            continue

        try:
            prec_train, _ = do_glasso(cov_train, lam)
            w_curr = w_from_prec(prec_train)
        except Exception:
            continue

        # ---- 验证: 使用 v 日的原始外积（不归一化）计算组合方差 ----
        raw_cov_val = raw_cov_dict.get(v)
        if raw_cov_val is None:
            continue

        port_var = float(w_curr @ raw_cov_val @ w_curr)
        total_var += port_var
        n_valid += 1

        # ---- 换手率: ||w_curr - w_prev||_1 ----
        if prev_w is not None:
            total_TO += float(np.sum(np.abs(w_curr - prev_w)))
        prev_w = w_curr

    if n_valid == 0:
        return (np.inf, np.nan, np.nan, 0)

    avg_var = total_var / n_valid
    avg_TO = total_TO / (n_valid - 1) if n_valid > 1 else 0.0
    score = float(avg_var + ETA * avg_TO)

    return (lam, score, avg_var, avg_TO, n_valid)


# ===================================================================
# 主程序
# ===================================================================
def main():
    # ---- 初始化 ----
    set_log_file(OUT_DIR / "step1_log.txt")

    files_raw = get_files_raw()
    dates = get_dates()
    n_days = get_n_days()

    log("=" * 65)
    log("  步骤1: §4.1 λ_Ω 选择（BIC 初筛 + 内部验证窗精选）")
    log("=" * 65)
    log(f"交易日总数: {n_days}")
    log(f"资产数 K:   {K}")
    log(f"并行核心数: {N_WORKERS}")
    log(f"BIC 样本日数: {BIC_SAMPLE_SIZE}")
    log(f"训练窗口:     {L_TRAIN} 天")
    log(f"验证窗口:     {L_VAL} 天")
    log(f"Ridge ε:      {EPS_RIDGE}")
    log(f"交易成本 η:   {ETA}")

    # ===================================================================
    # 阶段1: BIC 初筛（随机抽取 N 个样本日，并行计算）
    # ===================================================================
    log("\n" + "-" * 50)
    log("阶段1: BIC 初筛")
    log("-" * 50)

    rng = np.random.default_rng(BIC_RANDOM_SEED)

    # 从前 90% 的数据中随机抽取样本日（避免使用尾部的验证窗数据）
    bic_population_end = max(1, n_days - L_VAL - L_TRAIN)
    actual_sample_size = min(BIC_SAMPLE_SIZE, bic_population_end)
    actual_sample_size = max(actual_sample_size, 10)  # 至少 10 天

    sample_indices = sorted(
        rng.choice(bic_population_end, size=actual_sample_size, replace=False)
    )
    log(f"BIC 样本日: {actual_sample_size} 天 "
        f"(从日期 0～{bic_population_end - 1} 中随机抽取，种子={BIC_RANDOM_SEED})")

    lambda_candidates = np.logspace(BIC_LAMBDA_RANGE[0],
                                    BIC_LAMBDA_RANGE[1],
                                    BIC_N_LAMBDA)
    n_lam = len(lambda_candidates)
    log(f"λ 候选数: {n_lam} "
        f"(范围: {lambda_candidates[0]:.2e} ～ {lambda_candidates[-1]:.2e})")

    # 预加载 BIC 样本日数据
    t_load_start = time.time()
    sample_data = preload_days(sample_indices)
    log(f"数据预加载耗时: {time.time() - t_load_start:.1f} 秒")

    # 并行计算 BIC：每个样本日作为一个任务，内部遍历所有 λ
    t0 = time.time()
    bic_summary = np.full((n_lam, actual_sample_size), np.nan)
    sample_to_col = {si: ci for ci, si in enumerate(sample_indices)}

    tasks = [
        (si, sample_data[si], lambda_candidates)
        for si in sample_indices
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(_worker_bic_one_day, t): t[0]
            for t in tasks
        }
        done = 0
        for future in as_completed(futures):
            day_idx, bic_vals = future.result()
            col = sample_to_col[day_idx]
            bic_summary[:, col] = bic_vals
            done += 1
            if done % 10 == 0 or done == actual_sample_size:
                log(f"  BIC 进度: {done}/{actual_sample_size} 天")

    avg_bic = np.nanmean(bic_summary, axis=1)
    valid_bic = ~np.isnan(avg_bic)
    if not np.any(valid_bic):
        log("错误：所有 λ 候选均未生成有效 BIC，请扩大 λ 范围或降低 Ridge。")
        sys.exit(1)

    best_bic_idx = np.nanargmin(avg_bic)
    lambda_bic = lambda_candidates[best_bic_idx]

    # 边界检测：BIC 最优 λ 是否在网格边界
    if best_bic_idx == 0:
        log(f"\n⚠ 警告：BIC 最优 λ = {lambda_bic:.2e} 位于搜索下界，"
            f"真正的 λ_opt 可能更小，建议扩大 BIC_LAMBDA_RANGE 下界。")
    elif best_bic_idx == n_lam - 1:
        log(f"\n⚠ 警告：BIC 最优 λ = {lambda_bic:.2e} 位于搜索上界，"
            f"真正的 λ_opt 可能更大，建议扩大 BIC_LAMBDA_RANGE 上界。")

    log(f"\nBIC 最优 λ_Ω: {lambda_bic:.2e} "
        f"(BIC = {avg_bic[best_bic_idx]:.1f})")
    log(f"阶段1 耗时: {time.time() - t0:.1f} 秒")

    # ===================================================================
    # 阶段2: §4.1 内部验证窗精选（滚动训练窗口 + 并行 λ 评估）
    # ===================================================================
    log("\n" + "-" * 50)
    log("阶段2: §4.1 内部验证窗精选（滚动窗口训练）")
    log("-" * 50)

    validation_start = max(L_TRAIN, n_days - L_VAL)
    if validation_start >= n_days:
        log("错误：验证窗起始索引超出范围，请减少 L_TRAIN 或 L_VAL。")
        sys.exit(1)

    log(f"验证窗范围: 日期 {validation_start} ～ {n_days - 1} "
        f"({dates[validation_start]} ～ {dates[-1]})")

    # 精细 λ 候选：在 BIC 最优附近取对数等距网格
    lambda_fine = lambda_bic * np.logspace(-FINE_MULTIPLIER,
                                           FINE_MULTIPLIER,
                                           FINE_N_LAMBDA)
    log(f"精细 λ 候选数: {FINE_N_LAMBDA}")
    for li, lam in enumerate(lambda_fine):
        marker = " ← BIC 最优" if abs(lam - lambda_bic) < 1e-15 else ""
        log(f"  λ[{li:2d}] = {lam:.3e}{marker}")

    # ---- 预计算验证窗口所需的所有 raw_cov 和 daily_mins ----
    # 需要范围: [validation_start - L_train, n_days - 1] 即包含训练窗 + 验证窗
    load_range_start = max(0, validation_start - L_TRAIN)
    load_range_end = n_days - 1
    all_indices = list(range(load_range_start, load_range_end + 1))
    log(f"\n预计算 {len(all_indices)} 天的协方差矩阵 "
        f"(索引 {load_range_start}～{load_range_end}) ...")

    t_prep = time.time()
    raw_cov_dict: Dict[int, np.ndarray] = {}
    daily_mins: Dict[int, int] = {}
    for i in all_indices:
        rett1 = load_day(i)
        raw_cov_dict[i] = compute_raw_cov(rett1)
        daily_mins[i] = rett1.shape[1]
    log(f"协方差预计算耗时: {time.time() - t_prep:.1f} 秒")

    # ---- 并行评估每个精细 λ ----
    t0 = time.time()
    results_map: Dict[float, Tuple[float, float, float, int]] = {}

    tasks = [
        (lam, raw_cov_dict, daily_mins,
         validation_start, n_days, L_TRAIN)
        for lam in lambda_fine
    ]

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, FINE_N_LAMBDA)) as executor:
        futures = {
            executor.submit(_worker_eval_lambda, t): t[0]
            for t in tasks
        }
        done = 0
        for future in as_completed(futures):
            lam, score, avg_var, avg_TO, n_valid = future.result()
            results_map[lam] = (score, avg_var, avg_TO, n_valid)
            done += 1
            log(f"  [{done}/{FINE_N_LAMBDA}] "
                f"λ = {lam:.2e}  "
                f"得分 = {score:.4e}  "
                f"Var = {avg_var:.4e}  "
                f"TO = {avg_TO:.4f}  "
                f"有效天数 = {n_valid}")

    # ---- 选择最优 λ ----
    valid_scores = np.array([
        results_map.get(lam, (np.inf, np.nan, np.nan, 0))[0]
        for lam in lambda_fine
    ])

    best_idx = int(np.nanargmin(valid_scores))
    lambda_opt = lambda_fine[best_idx]
    best_score = results_map.get(lambda_opt, (np.nan, np.nan, np.nan, 0))

    log(f"\n§4.1 最优 λ_Ω: {lambda_opt:.2e}")
    log(f"  验证得分: {best_score[0]:.4e}")
    log(f"  组合方差: {best_score[1]:.4e}")
    log(f"  平均换手: {best_score[2]:.4f}")
    log(f"  有效天数: {best_score[3]}")
    log(f"阶段2 耗时: {time.time() - t0:.1f} 秒")

    # ===================================================================
    # 保存诊断与结果
    # ===================================================================
    log("\n" + "-" * 50)
    log("保存 λ 选择结果 ...")

    # 诊断 CSV
    diag = pd.DataFrame({
        "lambda": lambda_fine,
        "valid_score": [
            results_map.get(lam, (np.nan, np.nan, np.nan, 0))[0]
            for lam in lambda_fine
        ],
        "avg_variance": [
            results_map.get(lam, (np.nan, np.nan, np.nan, 0))[1]
            for lam in lambda_fine
        ],
        "avg_turnover": [
            results_map.get(lam, (np.nan, np.nan, np.nan, 0))[2]
            for lam in lambda_fine
        ],
        "n_valid_days": [
            results_map.get(lam, (np.nan, np.nan, np.nan, 0))[3]
            for lam in lambda_fine
        ],
        "is_optimal": [i == best_idx for i in range(FINE_N_LAMBDA)],
    })
    diag.to_csv(OUT_DIR / "lambda_selection_diagnostics.csv", index=False)

    # 最优 λ 供步骤2使用
    with open(OUT_DIR / "lambda_opt.txt", "w", encoding="utf-8") as f:
        f.write(f"{lambda_opt}\n")
        f.write(f"{lambda_bic}\n")

    # 完整 BIC 诊断
    bic_diag = pd.DataFrame({
        "lambda": lambda_candidates,
        "avg_bic": avg_bic,
        "is_bic_optimal": [i == best_bic_idx for i in range(n_lam)],
    })
    for ci, si in enumerate(sample_indices):
        bic_diag[f"BIC_day_{si}_{dates[si]}"] = bic_summary[:, ci]
    bic_diag.to_csv(OUT_DIR / "bic_diagnostics.csv", index=False)

    log(f"\n结果已保存至: {OUT_DIR}")
    log(f"  - lambda_selection_diagnostics.csv")
    log(f"  - bic_diagnostics.csv")
    log(f"  - lambda_opt.txt")
    log(f"  - step1_log.txt")
    log(f"\n最优 λ_Ω = {lambda_opt:.2e}")
    log("请运行步骤2: python glasso_step2_full.py")
    log("\n" + "=" * 65)
    log("  步骤1 完成")
    log("=" * 65)


if __name__ == "__main__":
    main()
