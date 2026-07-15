"""不同 λ 的网络密度对比 — 5天采样"""
import os as _os, time, numpy as np, warnings
warnings.filterwarnings("ignore")
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

NPY = "d:/HuaweiMoveData/Users/27438/Desktop/大创/数据/1min_log_return_npy"
fs = sorted([f for f in _os.listdir(NPY) if f.endswith(".npy") and not f.startswith("1min")])
K = 392

from sklearn.covariance import graphical_lasso
LAMS = [5e-7, 1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5]
RIDGE = [1e-4, 1e-3, 1e-2]

idx = [0, len(fs)//3, 2*len(fs)//3, len(fs)//2, len(fs)-1]

print(f"{'λ':>10} {'密度%':>7} {'非零边':>8} {'度':>6} {'RPV':>10} {'ratio':>8}")
print("-" * 56)

for lam in LAMS:
    dens, nzs, degs, rpvs = [], [], [], []
    for i in idx:
        rett = np.load(_os.path.join(NPY, fs[i]))
        cov = rett @ rett.T
        prec = None
        for ridge in RIDGE:
            cov.flat[::K+1] += ridge
            try:
                _, prec = graphical_lasso(cov, alpha=lam, mode="cd",
                    tol=1e-4, max_iter=100, enet_tol=5e-4)
                break
            except: cov.flat[::K+1] -= ridge
        if prec is None: continue
        adj = (np.abs(prec) > 1e-8).astype(int); np.fill_diagonal(adj,0)
        nz = int(adj.sum()//2)
        w = prec@np.ones(K); w/=w.sum()
        dens.append(nz/(K*(K-1)/2)*100)
        nzs.append(nz)
        degs.append(nz*2/K)
        rpvs.append(w @ cov @ w)
    d_avg, nz_avg, deg_avg, rpv_avg = np.mean(dens), np.mean(nzs), np.mean(degs), np.mean(rpvs)
    rpv5e7 = rpvs[0] if LAMS[0]==lam else 0  # placeholder, just print raw
    print(f"{lam:10.1e} {d_avg:7.1f} {nz_avg:8.0f} {deg_avg:6.1f} {rpv_avg:10.2e}")

print("\n说明: RPV 越小越好 | 密度建议 30-70%")
