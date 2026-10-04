import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
CACHE = PROJ / "endtoend_project" / "cache"
FEAT = PROJ / "特征工程"
VARX = PROJ / "VARX"

K = 392
ETA = 1e-4
T0, T1 = 65, 265
P_LAGS = 20

Y_all = np.load(FEAT / "Y_targets.npy")
rets = np.load(CACHE / "simple_returns.npy")
covs = np.load(CACHE / "cov_cache.npy", mmap_mode="r")

n = len(Y_all)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
test_start = n_train + n_val
n_test = n - test_start

Y_test = Y_all[test_start:]
Y_train = Y_all[:n_train]
r_test = rets[test_start:]
Y_prev = Y_all[test_start - 1]

ell = K - 1
W_full = Y_all[:, :ell]

def norm(W):
    s = W.sum(axis=1, keepdims=True)
    s = np.where(np.abs(s) < 1e-12, 1.0, s)
    return W / s

def eval_window(Y_pred):
    W = Y_pred[T0:T1]
    R = T1 - T0 - 1
    port_ret = np.array([W[t] @ r_test[T0 + t + 1] for t in range(R)])
    to_vec = np.array([np.sum(np.abs(W[t + 1] - W[t])) for t in range(R)])
    rpv = np.array([W[t] @ (covs[test_start + T0 + t + 1] + 1e-4*np.eye(K)) @ W[t] for t in range(R)])
    mse = float(np.mean((W - Y_test[T0:T1]) ** 2))
    mae = float(np.mean(np.abs(W - Y_test[T0:T1])))
    rpv_ann = float(np.mean(rpv) * 252)
    to_mean = float(np.mean(to_vec))
    net = port_ret - ETA*to_vec
    sr = float(np.mean(net)/np.std(net, ddof=1)*np.sqrt(252)) if np.std(net) > 1e-15 else 0.0
    return dict(MSE=mse, MAE=mae, RPV=rpv_ann, TO=to_mean, SR=sr)

# ============ 1) G&G statistical-loss: per-equation Lasso VAR(20), lambda by BIC ============
from sklearn.linear_model import enet_path

B_file = VARX / "Y_pred_GG_lasso20_coefs.npy"
if B_file.exists():
    GB = np.load(B_file)
    g = GB[:, 0]
    Bmat = GB[:, 1:]
    print("loaded cached lasso VAR(20) coefs", GB.shape, flush=True)
else:
    p = P_LAGS
    Wtr = Y_train[:, :ell]
    n_samp = n_train - p
    X = np.hstack([Wtr[p-k:n_train-k, :] for k in range(1, p+1)])
    Y = Wtr[p:]
    Xm = X.mean(axis=0)
    Xs = X.std(axis=0); Xs[Xs == 0] = 1.0
    ym = Y.mean(axis=0)
    ys = Y.std(axis=0); ys[ys == 0] = 1.0
    Xs_ = ((X - Xm) / Xs).astype(np.float64)
    ys_ = ((Y - ym) / ys).astype(np.float64)
    del X, Y
    print(f"design {Xs_.shape}, starting parallel lasso fits ...", flush=True)

    def fit_one(i):
        y_i = ys_[:, i]
        alpha_max = float(np.max(np.abs(Xs_.T @ y_i)) / n_samp)
        alphas = alpha_max * np.geomspace(1.0, 1.0/20.0, 18)
        _, coefs, _ = enet_path(Xs_, y_i, l1_ratio=1.0, alphas=alphas)
        rss = np.sum((y_i[:, None] - Xs_ @ coefs)**2, axis=0)
        nnz = np.count_nonzero(coefs, axis=0)
        bic = n_samp * np.log(rss / n_samp) + nnz * np.log(n_samp)
        best = int(np.argmin(bic))
        beta_std = coefs[:, best]
        beta_orig = beta_std * (ys[i] / Xs)
        intercept = ym[i] - float(Xm @ beta_orig)
        return i, beta_orig, intercept

    from joblib import Parallel, delayed
    t0 = time.time()
    results = Parallel(n_jobs=4)(delayed(fit_one)(i) for i in range(ell))
    print(f"lasso fits done in {time.time()-t0:.1f}s", flush=True)

    g = np.zeros(ell)
    Bmat = np.zeros((ell, ell * p))
    for i, beta, icpt in results:
        g[i] = icpt
        Bmat[i, :] = beta
    np.save(B_file, np.concatenate([g[:, None], Bmat], axis=1))

def lasso20_forecast():
    p = P_LAGS
    Xtest = np.hstack([W_full[test_start-k:test_start-k+n_test, :] for k in range(1, p+1)])
    V = Xtest @ Bmat.T + g
    return norm(np.concatenate([V, 1 - V.sum(axis=1, keepdims=True)], axis=1))

Y_gg_lasso20 = lasso20_forecast()
np.save(VARX / "Y_pred_GG_lasso20.npy", Y_gg_lasso20)
m_lasso = eval_window(Y_gg_lasso20)
print("GG statistical-loss (Lasso VAR(20)):", m_lasso, flush=True)

# ============ 2) G&G economic-loss: scalar MHAR, M-type estimator ============
# v_t = gamma + phi1 w_{t-1} + phi2 w^{(w)}_{t-1} + phi3 w^{(m)}_{t-1}
# economic loss: min sum_t (w*_t - v*_t)' R~_t (w*_t - v*_t),  R~_t = R_{t-1}
# Work with zero-sum transforms: r_t = Itilde w_t^(ell); c_j,t = Itilde x_j,t^(ell).
# The last (k-th) element of any Itilde transform is minus the sum of the ell free elements.

theta_file = VARX / "Y_pred_GG_mhar_theta.npy"
if theta_file.exists():
    theta = np.load(theta_file)
    gamma_hat = theta[:ell]
    phi = theta[ell:]
    print("loaded cached MHAR theta", flush=True)
else:
    cw = np.cumsum(W_full, axis=0)
    Wweek = np.zeros_like(W_full); Wweek[5:] = (cw[5:] - cw[:-5]) / 5.0
    Wmonth = np.zeros_like(W_full); Wmonth[20:] = (cw[20:] - cw[:-20]) / 20.0

    ones_ell = np.ones(ell)
    A = np.zeros((ell, ell))
    Bv = np.zeros((ell, 3))
    D = np.zeros((3, 3))
    u = np.zeros(ell)
    v = np.zeros(3)

    t0 = time.time()
    for t in range(20, n_train):
        Rt = np.asarray(covs[t-1]) + 1e-4 * np.eye(K)
        R11 = Rt[:ell, :ell]
        r1k = Rt[:ell, K-1]
        rkk = Rt[K-1, K-1]

        # regressors: full zero-sum k-vectors (first ell free, last = -sum(free))
        cfree = [W_full[t-1], Wweek[t], Wmonth[t]]
        c1_all = np.empty((3, ell))
        ck_all = np.empty(3)
        for j in range(3):
            c1_all[j] = cfree[j]
            ck_all[j] = -float(cfree[j].sum())

        # response transform r_t = Itilde w_t^(ell)
        r1 = W_full[t]
        rk = -float(r1.sum())

        A += R11 - np.outer(r1k, ones_ell) - np.outer(ones_ell, r1k) + rkk * np.outer(ones_ell, ones_ell)

        Rc = np.empty((3, ell))
        rk_c = np.empty(3)
        for j in range(3):
            Rc[j] = R11 @ c1_all[j] + r1k * ck_all[j]
            rk_c[j] = float(r1k @ c1_all[j] + rkk * ck_all[j])
        Bv += Rc.T - np.outer(ones_ell, rk_c)

        for j in range(3):
            for kk in range(3):
                D[j, kk] += float(c1_all[j] @ Rc[kk] + ck_all[j] * rk_c[kk])

        Rr = R11 @ r1 + r1k * rk
        sr = float(r1k @ r1 + rkk * rk)
        u += Rr - ones_ell * sr
        for j in range(3):
            v[j] += float(c1_all[j] @ Rr + ck_all[j] * sr)

    print(f"MHAR economic-loss accumulation done in {time.time()-t0:.1f}s", flush=True)

    MM = np.zeros((ell+3, ell+3))
    MM[:ell, :ell] = A
    MM[:ell, ell:ell+3] = Bv
    MM[ell:ell+3, :ell] = Bv.T
    MM[ell:ell+3, ell:ell+3] = D
    Mr = np.concatenate([u, v])
    MM = 0.5 * (MM + MM.T)
    theta = np.linalg.solve(MM, Mr)
    gamma_hat = theta[:ell]
    phi = theta[ell:]
    np.save(theta_file, theta)

print("phi (w-1, weekly, monthly):", np.round(phi, 4), flush=True)

def mhar_forecast():
    cw = np.cumsum(W_full, axis=0)
    Wweek = np.zeros_like(W_full); Wweek[5:] = (cw[5:] - cw[:-5]) / 5.0
    Wmonth = np.zeros_like(W_full); Wmonth[20:] = (cw[20:] - cw[:-20]) / 20.0
    lag1 = W_full[test_start-1:test_start-1+n_test, :]
    lagw = Wweek[test_start:test_start+n_test, :]
    lagm = Wmonth[test_start:test_start+n_test, :]
    V = gamma_hat + phi[0]*lag1 + phi[1]*lagw + phi[2]*lagm
    return norm(np.concatenate([V, 1 - V.sum(axis=1, keepdims=True)], axis=1))

Y_gg_mhar = mhar_forecast()
np.save(VARX / "Y_pred_GG_mhar.npy", Y_gg_mhar)
m_mhar = eval_window(Y_gg_mhar)
print("GG economic-loss (scalar MHAR M-est):", m_mhar, flush=True)
