# ===================================================================
# 步骤2: 全量 GLasso 权重计算 + §5.1 描述性统计
#
# 论文公式:
#   (3)  Σ̂_RC_t = Σ r_{t,m} r_{t,m}'        已实现协方差
#   (4)  Θ̂_t     = argmin tr(Σ̂_RC Θ) - logdet(Θ) + λ_Ω Σ|Θ_ij|
#   (5)  w̃_t     = Θ̂·1 / 1ᵀ·Θ̂·1              GMVP 权重
#
# 单日独立 GLasso (不滚动)：每天用当日日内收益直接计算
# ===================================================================
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

from 共享模块 import (
    K, TOL_ZERO, EPS_RIDGE, OUT_DIR,
    get_dates, get_n_days,
    load_day, compute_raw_cov, do_glasso, w_from_prec,
    compute_adjacency,
    log, set_log_file,
)

# ===================================================================
# 阶段二专属配置
# ===================================================================
N_WORKERS = max(1, mp.cpu_count() - 1)


# ===================================================================
# 辅助函数
# ===================================================================
def _failed_result(day_idx: int, error_msg: str) -> Dict[str, Any]:
    """统一的失败结果占位字典。"""
    return {
        "day": day_idx, "w": np.full(K, np.nan, dtype=np.float64),
        "rpv": np.nan, "nz_edges": np.nan, "density": np.nan,
        "cond_val": np.nan, "M": 0,
        "degree": np.full(K, np.nan, dtype=np.float64),
        "success": False, "error": error_msg,
    }


# ===================================================================
# 单日处理 (模块级，Windows spawn 兼容)
# ===================================================================
def process_one_day(args: Tuple[int, float]) -> Dict[str, Any]:
    """论文公式(3)→(4)→(5)：单日 GLasso → GMVP 权重 → 邻接矩阵。

    Args:
        args: (day_idx, lambda_opt)
    """
    day_idx, lam = args

    try:
        # (3) 已实现协方差 Σ̂_RC = Σ r·r'
        rett = load_day(day_idx)
        M = rett.shape[1]
        raw_cov = compute_raw_cov(rett)              # X @ Xᵀ
        cov_ridge = raw_cov.copy()
        cov_ridge.flat[::K + 1] += EPS_RIDGE         # + ridge·I

        # (4) GLasso Θ̂
        prec, _ = do_glasso(cov_ridge, lam)

        # (5) GMVP 权重 w̃ = Θ·1 / 1ᵀ·Θ·1
        w = w_from_prec(prec)

        # 已实现组合方差 RPV = w̃ᵀ·Σ̂_RC·w̃
        rpv = float(w @ raw_cov @ w)

        # 邻接矩阵 A_{ij} = 1(|θ_{ij}| > TOL_ZERO)
        adj = compute_adjacency(prec)
        degree = adj.sum(axis=1).astype(np.int32)    # 每资产度
        nz_edges = int(degree.sum() / 2)             # 无向，取半
        density = nz_edges / (K * (K - 1) / 2)

        # 条件数（每5天计算一次，节省 ~10% 总耗时；其余天填 NaN，描述性统计自动忽略）
        if day_idx % 5 == 0:
            eigvals = np.linalg.eigvalsh(raw_cov)
            cond_val = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-15 else np.inf
        else:
            cond_val = np.nan

        return {
            "day": day_idx, "w": w, "rpv": rpv,
            "nz_edges": nz_edges, "density": density,
            "cond_val": cond_val, "M": M,
            "degree": degree, "success": True,
        }

    except Exception as e:
        return _failed_result(day_idx, str(e)[:120])


# ===================================================================
# 描述性统计
# ===================================================================
def desc_row(values: np.ndarray, name: str) -> dict:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return {"Variable": name, "Mean": np.nan, "Std": np.nan,
                "P5": np.nan, "Median": np.nan, "P95": np.nan, "Obs": 0}
    return {"Variable": name, "Mean": np.mean(clean), "Std": np.std(clean),
            "P5": np.percentile(clean, 5), "Median": np.median(clean),
            "P95": np.percentile(clean, 95), "Obs": len(clean)}


# ===================================================================
# 主程序
# ===================================================================
def main():
    set_log_file(OUT_DIR / "step2_log.txt")
    dates = get_dates()
    n_days = get_n_days()

    # 读取 λ
    lam_path = OUT_DIR / "lambda_opt.txt"
    try:
        with open(lam_path, "r", encoding="utf-8") as f:
            lam = float(f.readline().strip())
    except (FileNotFoundError, ValueError):
        log("lambda_opt.txt 缺失/无效，请先运行阶段一: python λΩ选择.py")
        sys.exit(1)

    log("=" * 65)
    log("  阶段二: 全量 GLasso 权重 + 描述性统计")
    log("=" * 65)
    log(f"交易日: {n_days}  K={K}  并行核数: {N_WORKERS}  λ_Ω={lam:.1e}")
    log(f"论文公式: (3)→(4)→(5)  单日独立 GLasso")

    # ---- 并行计算 ----
    log("\n并行计算中...")
    tasks = [(i, lam) for i in range(n_days)]
    results = {}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_day, t): t[0] for t in tasks}
        done = 0
        n_fail = 0

        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                day_idx = futures[fut]
                r = _failed_result(day_idx, f"进程崩溃: {str(e)[:80]}")

            results[r["day"]] = r
            if not r["success"]:
                n_fail += 1
            done += 1

            if done % 50 == 0 or done == n_days:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_days - done) if done > 0 else 0
                log(f"  {done}/{n_days} ({100*done/n_days:.0f}%)  "
                    f"耗时:{elapsed:.0f}s  ETA:{eta:.0f}s  失败:{n_fail}")

    elapsed = time.time() - t0
    log(f"\n完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)  失败: {n_fail}/{n_days}")

    # ---- 汇总（.get 兜底防止 KeyError，理论上不会漏但防御性编程）----
    ordered = [results.get(i, _failed_result(i, "未完成"))
               for i in range(n_days)]
    W = np.array([r["w"] for r in ordered])              # (n_days, K)

    daily = pd.DataFrame({
        "date":       dates,
        "rpv":        [r["rpv"] for r in ordered],
        "density":    [r["density"] for r in ordered],
        "nz_edges":   [r["nz_edges"] for r in ordered],
        "cond_val":   [r["cond_val"] for r in ordered],
        "M_minutes":  [r["M"] for r in ordered],
        "success":    [r["success"] for r in ordered],
    })

    # 换手率
    TO = [np.nan]
    for t in range(1, n_days):
        if daily["success"].iloc[t] and daily["success"].iloc[t - 1]:
            TO.append(float(np.sum(np.abs(W[t] - W[t - 1]))))
        else:
            TO.append(np.nan)
    daily["turnover"] = TO

    # ---- Table 1 ----
    log("\n" + "-" * 50)
    log("Table 1: 描述性统计")
    log("-" * 50)

    rows = [
        desc_row(np.nanmean(W, axis=1), "GMVP 权重 截面均值"),
        desc_row(np.nanstd(W, axis=1), "GMVP 权重 截面标准差"),
        desc_row(np.nanmax(W, axis=1) - np.nanmin(W, axis=1), "GMVP 权重 截面极差"),
        desc_row(daily["rpv"].values, "已实现组合方差 RPV"),
        desc_row(daily["nz_edges"].values, "精度矩阵非零边数"),
        desc_row(daily["density"].values, "网络密度"),
        desc_row(pd.to_numeric(daily["turnover"], errors="coerce").values,
                 "换手率 (Turnover)"),
        desc_row(daily["cond_val"].values, "条件数"),
        desc_row(daily["M_minutes"].values, "日内分钟数"),
    ]
    neg_ratio = np.mean((W < 0).mean(axis=1)) * 100
    rows.append({"Variable": "负权重比例 (%)", "Mean": neg_ratio,
                 "Std": np.nan, "P5": np.nan, "Median": np.nan,
                 "P95": np.nan, "Obs": n_days})

    t1 = pd.DataFrame(rows)
    log("\n" + t1.to_string(index=False, float_format="%.4f"))
    log(f"\nGLasso 收敛率: {(n_days-n_fail)/n_days*100:.1f}%")

    # ---- 保存 ----
    out_dir = OUT_DIR / "code" / "输出数据"
    out_dir.mkdir(parents=True, exist_ok=True)

    t1.to_csv(out_dir / "Table1_Descriptive_Statistics.csv", index=False)
    daily.to_csv(out_dir / "Daily_Statistics.csv", index=False)

    wt_cols = [f"A_{i+1}" for i in range(K)]
    pd.DataFrame(W, index=dates, columns=wt_cols).to_csv(
        out_dir / "reg_weights_2436.csv")

    # 最后一日精度矩阵（单独计算，避免存储 2436 个 prec 浪费内存）
    rett_last = load_day(n_days - 1)
    cov_last = compute_raw_cov(rett_last)
    cov_last.flat[::K + 1] += EPS_RIDGE
    prec_last, _ = do_glasso(cov_last, lam)
    pd.DataFrame(prec_last, columns=wt_cols, index=wt_cols).to_csv(
        out_dir / "最后一日的精度矩阵.csv")

    # 每资产平均网络度（阶段三特征工程需要，复用 process_one_day 中已存储的 degree）
    degrees = np.nanmean([r["degree"] for r in ordered
                          if r.get("degree") is not None], axis=0)
    pd.DataFrame({"asset": range(1, K+1), "mean_degree": degrees}).to_csv(
        out_dir / "per_asset_mean_degree.csv", index=False)

    # 失败日
    if n_fail > 0:
        fail_rows = [{"date": dates[r["day"]], "error": r.get("error", "")}
                     for r in ordered if not r["success"]]
        pd.DataFrame(fail_rows).to_csv(out_dir / "failed_days.csv", index=False)

    log(f"\n结果已保存至: {out_dir}")
    log("=" * 65)
    log("  阶段二 完成")
    log("=" * 65)


if __name__ == "__main__":
    main()
