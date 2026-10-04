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

# ============ 1) G&G statistical-loss model: per-equation Lasso VAR(1) ============
from sklearn.linear_model import LassoLarsIC

coef_file = VARX / "Y_pred_GG_lasso_coefs.npy"
if coef_file.exists():
    GB = np.load(coef_file)
    g = GB[:, 0]
    B = GB[:, 1:]
    print("loaded cached lasso coefs", GB.shape, flush=True)
else:
    W = Y_train[:, :ell]
    Xtr = W[:-1]
    Ytr = W[1:]
    B = np.zeros((ell, ell))
    g = np.zeros(ell)
    t0 = time.time()
    for i in range(ell):
        m = LassoLarsIC(criterion="bic", fit_intercept=True, max_iter=100000)
        m.fit(Xtr, Ytr[:, i])
        B[i, :] = m.coef_
        g[i] = m.intercept_
        if (i+1) % 50 == 0:
            print(f"  lasso eq {i+1}/{ell} ({time.time()-t0:.1f}s)", flush=True)
    np.save(coef_file, np.concatenate([g[:, None], B], axis=1))

def lasso_forecast():
    lag_free = np.vstack([Y_prev[:ell], Y_test[:-1, :ell]])
    V = lag_free @ B.T + g
    return norm(np.concatenate([V, 1 - V.sum(axis=1, keepdims=True)], axis=1))

Y_gg_lasso = lasso_forecast()
np.save(VARX / "Y_pred_GG_lasso.npy", Y_gg_lasso)
m_lasso = eval_window(Y_gg_lasso)
print("GG statistical-loss (Lasso VAR(1)):", m_lasso, flush=True)

# ============ 2) G&G economic-loss model: scalar VAR(1) M-estimator ============
# Minimize sum_t (w*_t - v*_t)' R~_t (w*_t - v*_t),  R~_t = R_{t-1}
# with v*_t = e_k + Itilde @ gamma + phi * c_t,  c_t = Itilde @ w_{t-1}^(ell).
# Work with the zero-sum transform Itilde = [I_ell ; -ones_ell']:
#   r_t = Itilde @ w*_t^(ell)  (response),  c_t = Itilde @ w_{t-1}^(ell)  (regressor).

theta_file = VARX / "Y_pred_GG_econ_theta.npy"
if theta_file.exists():
    theta = np.load(theta_file)
    gamma_hat = theta[:ell]
    phi_hat = theta[ell]
    print("loaded cached econ theta", flush=True)
else:
    ones_ell = np.ones(ell)
    A = np.zeros((ell, ell))
    Bvec = np.zeros(ell)
    d = 0.0
    u = np.zeros(ell)
    v = 0.0
    t0 = time.time()
    for t in range(1, n_train):
        Rt = np.asarray(covs[t-1]) + 1e-4 * np.eye(K)
        R11 = Rt[:ell, :ell]
        r1k = Rt[:ell, K-1]
        rkk = Rt[K-1, K-1]
        c1 = Y_train[t-1, :ell]
        ck = Y_train[t-1, K-1] - 1.0
        r1 = Y_train[t, :ell]
        rk = Y_train[t, K-1] - 1.0

        A += R11 - np.outer(r1k, ones_ell) - np.outer(ones_ell, r1k) + rkk * np.outer(ones_ell, ones_ell)

        Rc = R11 @ c1 + r1k * ck
        sc = float(r1k @ c1 + rkk * ck)
        Bvec += Rc - ones_ell * sc

        d += float(c1 @ R11 @ c1 + 2 * ck * (r1k @ c1) + rkk * ck * ck)

        Rr = R11 @ r1 + r1k * rk
        sr = float(r1k @ r1 + rkk * rk)
        u += Rr - ones_ell * sr

        v += float(c1 @ (R11 @ r1 + r1k * rk) + ck * (r1k @ r1 + rkk * rk))

    print(f"economic loss accumulation done in {time.time()-t0:.1f}s", flush=True)

    MM = np.zeros((ell+1, ell+1))
    MM[:ell, :ell] = A
    MM[:ell, ell] = Bvec
    MM[ell, :ell] = Bvec
    MM[ell, ell] = d
    Mr = np.concatenate([u, [v]])
    MM = 0.5 * (MM + MM.T)
    theta = np.linalg.solve(MM, Mr)
    gamma_hat = theta[:ell]
    phi_hat = theta[ell]
    np.save(theta_file, theta)

print("phi_hat", phi_hat, "gamma_hat range", float(gamma_hat.min()), float(gamma_hat.max()), flush=True)

def econ_forecast():
    lag_free = np.vstack([Y_prev[:ell], Y_test[:-1, :ell]])
    V = gamma_hat + phi_hat * lag_free
    return norm(np.concatenate([V, 1 - V.sum(axis=1, keepdims=True)], axis=1))

Y_gg_econ = econ_forecast()
np.save(VARX / "Y_pred_GG_econ.npy", Y_gg_econ)
m_econ = eval_window(Y_gg_econ)
print("GG economic-loss (scalar VAR(1) M-est):", m_econ, flush=True)
