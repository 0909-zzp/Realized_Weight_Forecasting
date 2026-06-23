"""阶段二: 纯串行权重计算 — 零 multiprocessing，Windows/Linux 通用"""
import sys, time, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

K = 392
RIDGE_CHAIN = [1e-4, 1e-3, 1e-2]
LAM = 5e-7
MAX_ITER = 100

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

print("交易日: %d  K=%d  模式:串行  lam=%.1e  ridge=%s" % (n_days, K, LAM, RIDGE_CHAIN))
print("=" * 65)

t0 = time.time()
W = np.full((n_days, K), np.nan)
fails = 0

for i, fname in enumerate(files_list):
    try:
        rett = np.load(os.path.join(npy_dir, fname))
        raw_cov = rett @ rett.T
        prec = None
        for ridge in RIDGE_CHAIN:
            cov_r = raw_cov.copy()
            cov_r.flat[::K + 1] += ridge
            try:
                from sklearn.covariance import graphical_lasso
                _, prec = graphical_lasso(emp_cov=cov_r, alpha=LAM, mode="cd",
                                           tol=1e-4, max_iter=MAX_ITER, enet_tol=5e-4)
                break
            except:
                continue
        if prec is None:
            raise RuntimeError("Ridge耗尽")
        ones = np.ones(K)
        W[i] = prec @ ones / np.sum(ones @ prec)
    except Exception as e:
        fails += 1

    if (i + 1) % 50 == 0 or i + 1 == n_days:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_days - i - 1)
        print("  %d/%d (%d%%) %ds ETA:%ds fail:%d" % (
            i + 1, n_days, 100 * (i + 1) // n_days, int(elapsed), int(eta), fails))

elapsed = time.time() - t0
print("完成! %.0fs (%.1fmin)" % (elapsed, elapsed / 60))

csv_dir = os.path.join(_project, "code", "输出数据")
os.makedirs(csv_dir, exist_ok=True)
wt_cols = ["A_%d" % (j + 1) for j in range(K)]
csv_path = os.path.join(csv_dir, "reg_weights_2436.csv")
pd.DataFrame(W, index=dates, columns=wt_cols).to_csv(csv_path)
print("已保存: %s" % csv_path)
