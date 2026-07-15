"""阶段二: 纯权重计算 — BLAS单线程优化版
B1: Ridge in-place  B2: prec@ones一次  B3: eigvalsh每40天  B5: BLAS单线程
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, time, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from multiprocessing import Pool, cpu_count, freeze_support
from 共享模块 import K, MAX_ITER

RIDGE_CHAIN = [1e-4, 1e-3, 1e-2]
LAM = 3e-6  # 密度62%, OOS方差+5.6%
MAX_ITER = 100

ONES_VEC = np.ones(K, dtype=np.float64)

def work(args):
    i, fname, npydir, lam = args
    import numpy as np
    from sklearn.covariance import graphical_lasso
    try:
        rett = np.load(os.path.join(npydir, fname))
        raw_cov = rett @ rett.T
        prec = None; last_err = ""
        for ridge in RIDGE_CHAIN:
            raw_cov.flat[:K*(K+1):K+1] += ridge
            try:
                _, prec = graphical_lasso(raw_cov, alpha=lam, mode="cd",
                                           tol=1e-4, max_iter=MAX_ITER, enet_tol=5e-4)
                break
            except Exception as e:
                last_err = str(e)[:80]; raw_cov.flat[:K*(K+1):K+1] -= ridge
        if prec is None:
            raise RuntimeError("Ridge耗尽: %s" % last_err)
        p1 = prec @ ONES_VEC
        w = p1 / p1.sum()
        rpv = float(w @ raw_cov @ w)
        adj = (np.abs(prec) > 1e-8).astype(np.int8)
        np.fill_diagonal(adj, 0)
        deg = adj.sum(axis=1).astype(np.int32)
        nz = int(deg.sum() // 2)
        cond_val = np.nan
        if i % 40 == 0:
            eigs = np.linalg.eigvalsh(raw_cov)
            cond_val = float(eigs[-1] / eigs[0]) if eigs[0] > 1e-15 else np.inf
        return {"day": i, "w": w, "rpv": rpv, "nz_edges": nz,
                "density": nz / (K*(K-1)/2), "adj": adj,
                "cond_val": cond_val, "M": rett.shape[1], "success": True}
    except Exception as e:
        return {"day": i, "w": np.full(K, np.nan), "rpv": np.nan,
                "nz_edges": np.nan, "density": np.nan,
                "adj": np.zeros((K,K),dtype=np.int8),
                "cond_val": np.nan, "M": 0, "success": False, "error": str(e)[:120]}

def main():
    _cur = os.path.dirname(os.path.abspath(__file__))
    _project = os.path.dirname(_cur)
    _root = os.path.dirname(_project)
    npy_dir = os.path.join(_root, "数据", "1min_log_return_npy")
    if not os.path.exists(npy_dir):
        npy_dir = "/home/ubuntu/数据/1min_log_return_npy"

    all_items = sorted(os.listdir(npy_dir))
    files_list = [f for f in all_items if f.endswith("_1min_log_return.npy") and not f.startswith("1min")]
    dates = [f.split(" ")[0] for f in files_list]
    n_days = len(files_list)
    n_workers = max(1, cpu_count() - 1)

    print("交易日: %d  K=%d  workers=%d  lam=%.1e  max_iter=%d" % (n_days, K, n_workers, LAM, MAX_ITER))
    print("=" * 65)

    t0 = time.time()
    results = [None] * n_days
    done = 0; fails = 0
    tasks = [(i, f, npy_dir, LAM) for i, f in enumerate(files_list)]

    first_error = None
    with Pool(n_workers) as p:
        for r in p.imap_unordered(work, tasks):
            results[r["day"]] = r
            done += 1
            if not r["success"]:
                fails += 1
                if first_error is None:
                    first_error = r.get("error", "unknown")
            if done % 50 == 0 or done == n_days:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_days - done) if done > 0 else 0
                print("  %d/%d (%d%%)  %.0fs ETA:%.0fs  fail:%d" % (
                    done, n_days, 100*done//n_days, elapsed, eta, fails))
                if fails > 0 and first_error:
                    print("    [首错] %s" % first_error)

    elapsed = time.time() - t0
    print("完成! %.0fs (%.1fmin)  %.0fms/d  %d失败" % (elapsed, elapsed/60, elapsed/n_days*1000, fails))

    W = np.array([r["w"] for r in results])
    csv_dir = os.path.join(_project, "code", "输出数据")
    os.makedirs(csv_dir, exist_ok=True)

    wt_cols = ["A_%d" % (j+1) for j in range(K)]
    pd.DataFrame(W, index=dates, columns=wt_cols).to_csv(os.path.join(csv_dir, "reg_weights_2436.csv"))
    print("已保存: reg_weights_2436.csv")

    # 逐日邻接矩阵
    adj_dir = os.path.join(csv_dir, "adjacency")
    os.makedirs(adj_dir, exist_ok=True)
    adj_saved = 0
    for i, r in enumerate(results):
        if r["success"]:
            np.save(os.path.join(adj_dir, f"{dates[i]}.npy"), r["adj"])
            adj_saved += 1
    print(f"已保存: adjacency/ ({adj_saved}/{n_days}天)")

    # Daily_Statistics
    daily = pd.DataFrame({
        "date": dates,
        "rpv": [r["rpv"] for r in results],
        "density": [r["density"] for r in results],
        "nz_edges": [r["nz_edges"] for r in results],
        "M_minutes": [r["M"] for r in results],
        "success": [r["success"] for r in results],
    })
    TO = [np.nan]
    for t in range(1, n_days):
        if daily["success"].iloc[t] and daily["success"].iloc[t-1]:
            TO.append(float(np.sum(np.abs(W[t]-W[t-1]))))
        else:
            TO.append(np.nan)
    daily["turnover"] = TO
    daily.to_csv(os.path.join(csv_dir, "Daily_Statistics.csv"), index=False)
    print("已保存: Daily_Statistics.csv")

    if fails > 0:
        fail_rows = [{"date": dates[r["day"]], "error": r.get("error","")} for r in results if not r["success"]]
        pd.DataFrame(fail_rows).to_csv(os.path.join(csv_dir, "failed_days.csv"), index=False)

if __name__ == "__main__":
    freeze_support()
    main()
