"""阶段二: 纯计算Worker — 零项目依赖，Windows/Linux双平台"""
import sys, time, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from multiprocessing import Pool, cpu_count, freeze_support

K = 392
RIDGE_CHAIN = [1e-4, 1e-3, 1e-2]  # 精简化退避链
LAM = 5e-7
MAX_ITER = 100

def work(args):
    i, fname, npydir, k, lam = args
    import numpy as np
    from sklearn.covariance import graphical_lasso
    try:
        rett = np.load(os.path.join(npydir, fname))
        raw_cov = rett @ rett.T
        prec = None
        last_err = ""
        for ridge in RIDGE_CHAIN:
            cov_r = raw_cov.copy()
            cov_r.flat[::k + 1] += ridge
            try:
                _, prec = graphical_lasso(emp_cov=cov_r, alpha=lam, mode="cd",
                                           tol=1e-4, max_iter=MAX_ITER, enet_tol=5e-4)
                break
            except Exception as e:
                last_err = str(e)[:80]
        if prec is None:
            raise RuntimeError("Ridge耗尽: %s" % last_err)
        ones = np.ones(k)
        w = prec @ ones / np.sum(ones @ prec)
        rpv = float(w @ raw_cov @ w)
        adj = (np.abs(prec) > 1e-8).astype(np.int8)
        np.fill_diagonal(adj, 0)
        deg = adj.sum(axis=1).astype(np.int32)
        nz = int(deg.sum() // 2)
        cond_val = np.nan
        if i % 5 == 0:
            eigs = np.linalg.eigvalsh(raw_cov)
            cond_val = float(eigs[-1] / eigs[0]) if eigs[0] > 1e-15 else np.inf
        return {"day": i, "w": w, "rpv": rpv, "nz_edges": nz,
                "density": nz / (k * (k - 1) / 2),
                "cond_val": cond_val, "M": rett.shape[1], "degree": deg, "success": True}
    except Exception as e:
        return {"day": i, "w": np.full(k, np.nan), "rpv": np.nan, "nz_edges": np.nan,
                "density": np.nan, "cond_val": np.nan, "M": 0,
                "degree": np.full(k, np.nan), "success": False, "error": str(e)[:120]}

def main():
    # 路径自动解析
    _cur = os.path.dirname(os.path.abspath(__file__))
    _project = os.path.dirname(_cur)
    _root = os.path.dirname(_project)
    npy_dir = os.path.join(_root, "数据", "1min_log_return_npy")
    if not os.path.exists(npy_dir):
        npy_dir = "/home/ubuntu/数据/1min_log_return_npy"
    out_dir_root = _project

    all_items = sorted(os.listdir(npy_dir))
    files_list = [f for f in all_items if f.endswith("_1min_log_return.npy") and not f.startswith("1min")]
    dates = [f.split(" ")[0] for f in files_list]
    n_days = len(files_list)
    n_workers = max(1, cpu_count() - 1)

    print("交易日: %d  K=%d  workers=%d  lam=%.1e" % (n_days, K, n_workers, LAM))
    print("=" * 65)

    t0 = time.time()
    results = [None] * n_days
    done = 0
    fails = 0
    tasks = [(i, f, npy_dir, K, LAM) for i, f in enumerate(files_list)]

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
                print("  %d/%d (%d%%) %ds ETA:%ds fail:%d" % (
                    done, n_days, 100 * done // n_days, int(elapsed), int(eta), fails))
                if fails > 0 and first_error:
                    print("    [首错] %s" % first_error)

    elapsed = time.time() - t0
    print("完成! %.0fs (%.1fmin)" % (elapsed, elapsed / 60))

    W = np.array([r["w"] for r in results])
    csv_dir = os.path.join(out_dir_root, "code", "输出数据")
    os.makedirs(csv_dir, exist_ok=True)
    wt_cols = ["A_%d" % (j + 1) for j in range(K)]
    csv_path = os.path.join(csv_dir, "reg_weights_2436.csv")
    pd.DataFrame(W, index=dates, columns=wt_cols).to_csv(csv_path)
    print("已保存: %s" % csv_path)

if __name__ == "__main__":
    freeze_support()
    main()
