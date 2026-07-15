"""对失败日用更大 Ridge 重试"""
import os as _os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

NPY = "d:/HuaweiMoveData/Users/27438/Desktop/大创/数据/1min_log_return_npy"
DIR = r"d:\HuaweiMoveData\Users\27438\Desktop\大创\图形Lasso\code\输出数据"
d = pd.read_csv(DIR + "/Daily_Statistics.csv")
fails = d[~d["success"]]["date"].values
print(f"失败日: {len(fails)}天\n")

K, LAM = 392, 3e-6
from sklearn.covariance import graphical_lasso
ORD = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]  # 扩展退避链

for dt in fails:
    for f in _os.listdir(NPY):
        if f.startswith(str(dt)):
            rett = np.load(_os.path.join(NPY, f))
            cov = rett @ rett.T
            prec = None
            used_ridge = None
            for r in ORD:
                cov.flat[::K+1] += r
                try:
                    _, prec = graphical_lasso(cov, alpha=LAM, mode="cd",
                        tol=1e-4, max_iter=100, enet_tol=5e-4)
                    used_ridge = r; break
                except: cov.flat[::K+1] -= r
            if prec is not None:
                w = prec @ np.ones(K); w /= w.sum()
                rpv = float(w @ cov @ w)
                adj = (np.abs(prec) > 1e-8).astype(int); np.fill_diagonal(adj, 0)
                nz = int(adj.sum() // 2)
                print(f"{dt} ✅ Ridge={used_ridge:.0e}  RPV={rpv:.2e}  deg={nz*2/K:.0f}  dens={nz/(K*(K-1)/2)*100:.1f}%")
            else:
                print(f"{dt} ❌ Ridge耗尽")
            break
