# ===================================================================
# 固定时间窗：训练44天 | 验证5天 | 测试22天
# 训练窗不动，选出最优 λ，然后在测试集上评估
# ===================================================================
import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ---- 路径 ----
_CODE = Path(__file__).resolve().parent
_PROJ = _CODE.parent.parent
DATA_DIR = _PROJ / "数据" / "1min_log_return"
OUT_DIR  = _PROJ / "图形Lasso"

# ---- 参数 ----
K=392; MAX_ITER=800; TOL_GL=5e-5; ENET_TOL=5e-4; EPS_R=1e-4; ETA=1e-4
RIDGE_FB=[5e-4,1e-3,5e-3,1e-2]

# 时间窗大小（单位：交易日）
N_TRAIN = 44
N_VAL   = 5
N_TEST  = 22

# λ 扫描范围
LAMBDA_MIN_EXP = -5
LAMBDA_MAX_EXP = -2
N_LAMBDAS      = 8

# ---- 文件列表 ----
_files_raw = []; _dates = []; _n_days = 0
def _ensure():
    global _files_raw, _dates, _n_days
    if _files_raw: return
    _files_raw = sorted([f for f in os.listdir(DATA_DIR)
                         if f.endswith("_1min_log_return.RData") and not f.startswith("1min")])
    _dates = [f.split(" ")[0] for f in _files_raw]
    _n_days = len(_files_raw)

def get_dates(): _ensure(); return _dates
def get_n_days(): _ensure(); return _n_days

def load_day(idx):
    _ensure(); import pyreadr
    return pyreadr.read_r(str(DATA_DIR / _files_raw[idx]))["rett1"].values

def raw_cov(r): return r @ r.T

def do_glasso(cov_mat, lam):
    from sklearn.covariance import graphical_lasso
    for i, r in enumerate([EPS_R] + RIDGE_FB):
        c = cov_mat.copy()
        if i > 0: c.flat[::K+1] += (r - EPS_R)
        try:
            ce, pr = graphical_lasso(emp_cov=c, alpha=lam, mode='cd',
                                      tol=TOL_GL, max_iter=MAX_ITER, enet_tol=ENET_TOL)
            return pr, ce
        except (FloatingPointError, ValueError): continue
    c = cov_mat.copy(); c.flat[::K+1] += 1e-1 - EPS_R
    ce, pr = graphical_lasso(emp_cov=c, alpha=lam, mode='cd', tol=TOL_GL,
                              max_iter=MAX_ITER, enet_tol=max(ENET_TOL, 1e-3))
    return pr, ce

def w_from_prec(p): w=p @ np.ones(K); return w / np.sum(w)

_LOG = None
def set_log(path: Path): global _LOG; _LOG = path
def log(msg):
    print(msg, flush=True)
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as f: f.write(msg + "\n")

# ---- 主程序 ----
def main():
    set_log(OUT_DIR / "lambda_selection_log.txt")
    dates = get_dates(); n_days = get_n_days()

    # 用数据末尾来分配训练/验证/测试
    total_needed = N_TRAIN + N_VAL + N_TEST
    if n_days < total_needed:
        raise ValueError(f"数据不足：{n_days}天 < 需要{total_needed}天")

    # 训练：末尾往前取 total_needed 天，再分三段
    train_end   = n_days - N_TEST - N_VAL - 1
    train_start = train_end - N_TRAIN + 1
    val_start   = train_end + 1
    val_end     = val_start + N_VAL - 1
    test_start  = val_end + 1
    test_end    = test_start + N_TEST - 1

    log("=" * 65)
    log("  固定时间窗 λ 选择")
    log("=" * 65)
    log(f"训练: {dates[train_start]} ~ {dates[train_end]} ({N_TRAIN}天)")
    log(f"验证: {dates[val_start]} ~ {dates[val_end]} ({N_VAL}天)")
    log(f"测试: {dates[test_start]} ~ {dates[test_end]} ({N_TEST}天)")

    lambda_grid = np.logspace(LAMBDA_MIN_EXP, LAMBDA_MAX_EXP, N_LAMBDAS)
    log(f"λ 候选: {N_LAMBDAS} 个 ({lambda_grid[0]:.2e} ~ {lambda_grid[-1]:.2e})")

    # ---- 1. 构建训练协方差（44天合并） ----
    log("\n构建训练协方差 ...")
    train_sum = np.zeros((K, K), dtype=np.float64)
    for i in range(train_start, train_end + 1):
        train_sum += raw_cov(load_day(i))
    cov_train = train_sum / N_TRAIN
    cov_train.flat[::K+1] += EPS_R
    log(f"  训练协方差完成")

    # ---- 2. 加载验证日 raw_cov (5天) ----
    log("加载验证日 ...")
    val_raws = [raw_cov(load_day(i)) for i in range(val_start, val_end + 1)]

    # ---- 3. λ 扫瞄 → 验证得分 ----
    log("\n" + "-" * 50)
    log("λ 扫描（固定训练窗）")
    log("-" * 50)

    t0 = time.time()
    results = []
    for lam in lambda_grid:
        try:
            prec, _ = do_glasso(cov_train, lam)
            w = w_from_prec(prec)
        except Exception as e:
            log(f"  λ={lam:.2e}  GLasso 失败: {e}")
            results.append((lam, np.inf, np.nan, np.nan))
            continue

        # 在 5 天验证集上评估
        total_var, last_w = 0.0, None
        valid = 0
        for rv in val_raws:
            pv = float(w @ rv @ w)
            if np.isfinite(pv):
                total_var += pv
                valid += 1
                last_w = w   # 固定训练窗下权重不变，TO=0

        if valid == 0:
            results.append((lam, np.inf, np.nan, np.nan))
            log(f"  λ={lam:.2e}  无有效验证日")
        else:
            avg_var = total_var / valid
            # 固定窗下权重不变 → TO = 0
            score = avg_var
            results.append((lam, score, avg_var, 0.0))
            log(f"  λ={lam:.2e}  Score={score:.4e}  Var={avg_var:.4e}  valid={valid}")

    # ---- 4. 选最优 ----
    scores = [r[1] for r in results]
    best_idx = int(np.nanargmin(scores))
    lambda_opt = float(results[best_idx][0])

    log(f"\n最优 λ_Ω: {lambda_opt:.2e}  (验证得分: {results[best_idx][1]:.4e})")
    log(f"阶段耗时: {time.time()-t0:.1f}s")

    # ---- 5. 测试集评估 ----
    log("\n" + "-" * 50)
    log("测试集评估 (最优 λ)")
    log("-" * 50)

    prec_opt, _ = do_glasso(cov_train, lambda_opt)
    w_opt = w_from_prec(prec_opt)

    test_raws = [raw_cov(load_day(i)) for i in range(test_start, test_end + 1)]
    test_vars = [float(w_opt @ rv @ w_opt) for rv in test_raws]
    test_vars = [v for v in test_vars if np.isfinite(v)]

    if test_vars:
        test_avg = np.mean(test_vars)
        test_std = np.std(test_vars)
        log(f"  测试天数: {len(test_vars)}/{N_TEST}")
        log(f"  平均已实现方差: {test_avg:.4e}")
        log(f"  标准差: {test_std:.4e}")

        # 等权重benchmark
        w_eq = np.full(K, 1.0/K)
        eq_vars = [float(w_eq @ rv @ w_eq) for rv in test_raws]
        eq_avg = np.mean([v for v in eq_vars if np.isfinite(v)])
        log(f"  等权重benchmark: {eq_avg:.4e}")
    else:
        log(f"  测试集无有效日")

    # ---- 保存 ----
    pd.DataFrame(results, columns=["lambda","valid_score","avg_variance","avg_turnover"]
    ).to_csv(OUT_DIR / "lambda_selection_diagnostics.csv", index=False)

    with open(OUT_DIR / "lambda_opt.txt", "w", encoding="utf-8") as f:
        f.write(f"{lambda_opt}\n{lambda_opt}\n")

    # 保存测试结果
    with open(OUT_DIR / "test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"最优 λ: {lambda_opt:.2e}\n")
        if test_vars:
            f.write(f"测试平均方差: {test_avg:.4e} ± {test_std:.4e}\n")
            f.write(f"等权重方差: {eq_avg:.4e}\n")

    log(f"\n结果保存完成。")
    log(f"  - lambda_selection_diagnostics.csv")
    log(f"  - lambda_opt.txt")
    log(f"  - test_results.txt")

if __name__ == "__main__":
    main()
