"""最小内存权重计算 — 单进程逐天 GLasso, 无 Pool, 无累积"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.covariance import graphical_lasso

sys.path.insert(0, str(Path(__file__).parent))
from 共享模块 import K

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-6
OUT_DIR = Path(__file__).parents[1] / "lambda_robustness" / f"lam_{LAM:.0e}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).parents[2]
npy_dir = ROOT / "数据" / "1min_log_return_npy"
files = sorted([f for f in npy_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()])
dates = [f.stem[:8] for f in files]

EPS = 1e-4
RIDGE_FALLBACK = [5e-4, 1e-3, 5e-3, 1e-2]
MAX_ITER = 100

def fit_one(raw_cov, lam, K):
    """自适应Ridge回退, 与共享模块.do_glasso一致"""
    all_ridges = [EPS] + RIDGE_FALLBACK
    for r in all_ridges:
        c = raw_cov.copy()
        c.flat[::K+1] += (r - EPS if r > EPS else 0)
        try:
            _, prec = graphical_lasso(c, alpha=lam, mode='cd', tol=1e-4, max_iter=MAX_ITER)
            return prec
        except:
            continue
    # 终极回退
    c = raw_cov.copy()
    c.flat[::K+1] += 1e-1
    try:
        _, prec = graphical_lasso(c, alpha=lam, mode='cd', tol=5e-4, max_iter=MAX_ITER)
        return prec
    except:
        return None

print(f"λ={LAM:.0e}  K={K}  days={len(files)}  sequential")
t0 = time.time()

weights_list = []
fails = 0
for i, fpath in enumerate(files):
    try:
        rett = np.load(str(fpath))
        raw = rett @ rett.T
        raw.flat[::K+1] += EPS
        prec = fit_one(raw, LAM, K)
        if prec is not None:
            w = prec @ np.ones(K)
            w = w / w.sum()
            weights_list.append(w)
        else:
            weights_list.append(np.full(K, np.nan))
            fails += 1
    except Exception:
        weights_list.append(np.full(K, np.nan))
        fails += 1
    if (i+1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(files)}  fails={fails}  {elapsed/60:.0f}min")

elapsed = time.time() - t0
print(f"完成: {elapsed/60:.0f}min  fails={fails}/{len(files)}")

# 保存
W = np.array(weights_list)
df = pd.DataFrame(W, columns=[f"A_{j}" for j in range(K)], index=dates)
df.to_csv(OUT_DIR / "reg_weights_2436.csv")
np.save(OUT_DIR / "weights.npy", W)
with open(OUT_DIR / "lambda.txt", 'w') as f:
    f.write(f"lam = {LAM}\n")
print(f"保存: {OUT_DIR}")
