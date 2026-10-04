"""Table C.1 — Expanded benchmark comparison (appendix).

  * Exponential smoothing alpha selected on the VALIDATION block by MSE.
  * Partial-adjustment gamma selected on the VALIDATION block by matching
    a common turnover target (pre-specified).
  * Random forest predictions are cached to Y_pred_rf.npy.
  * Golosnoy & Gribisch (2022) rows (same scalar VAR(1) structure, different objective):
      - statistical-loss model  : scalar VAR(1) fitted by least squares (MSE)
      - economic-loss model     : scalar VAR(1) fitted by the M-type
                                  (economic GMVP loss) estimator with
                                  R~_t = R_{t-1}
    Predictions are cached to VARX/Y_pred_GG_var1_LS.npy and
    VARX/Y_pred_GG_econ.npy.

Common sample: 200-day window (test indices 65:265), 1 bp one-way cost.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
CACHE = PROJ / "endtoend_project" / "cache"
FEAT = PROJ / "特征工程"
VARX = PROJ / "VARX"
OUT = PROJ / "性能评估与可视化"

K = 392
ETA = 1e-4
T0, T1 = 65, 265
TARGET_TO = 0.10      # common validation turnover target for partial adjustment

valid = np.load(FEAT / "valid_indices.npy")
Y_all = np.load(FEAT / "Y_targets.npy")
X_all = np.load(FEAT / "X_features.npy")
rets = np.load(CACHE / "simple_returns.npy")
covs = np.load(CACHE / "cov_cache.npy", mmap_mode="r")

n = len(Y_all)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
test_start = n_train + n_val
n_test = n - test_start

Y_test = Y_all[test_start:]
Y_val = Y_all[n_train:test_start]
r_test = rets[test_start:]
r_val = rets[n_train:test_start]
Y_prev = Y_all[test_start - 1]
Y_vprev = Y_all[n_train - 1]

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

def norm(W):
    s = W.sum(axis=1, keepdims=True)
    s = np.where(np.abs(s) < 1e-12, 1.0, s)
    return W / s

def drift(w, r):
    w = np.asarray(w, float); r = np.asarray(r, float)
    denom = 1.0 + float(w @ r)
    return (w*(1.0+r)/denom) if abs(denom) > 1e-15 else w

rows = []

# 1 persistence
Y1 = np.empty_like(Y_test); Y1[0] = Y_prev; Y1[1:] = Y_test[:-1]
rows.append(("Last realized weight / graphical-lasso rule", eval_window(Y1)))

# 2 exponential smoothing, alpha selected on validation MSE
def es_path(alpha, Y_hist, y_prev):
    T = len(Y_hist); out = np.empty_like(Y_hist); f = y_prev.copy()
    for t in range(T):
        wlag = Y_hist[t-1] if t > 0 else y_prev
        f = (1-alpha)*wlag + alpha*f
        out[t] = f
    return out

def es_mse(alpha, Y_hist, y_prev):
    return float(np.mean((es_path(alpha, Y_hist, y_prev) - Y_hist)**2))

alphas = np.linspace(0.0, 0.95, 20)
best_alpha = min(alphas, key=lambda a: es_mse(a, Y_val, Y_vprev))
Y2 = es_path(best_alpha, Y_test, Y_prev)
rows.append((f"Exponential smoothing of weights (alpha={best_alpha:.2f})", eval_window(Y2)))

# 3-4 Golosnoy & Gribisch (2022) direct models (precomputed predictions)
Y_gg_lasso = np.load(VARX / "Y_pred_GG_var1_LS.npy")
rows.append(("Golosnoy and Gribisch statistical-loss model", eval_window(Y_gg_lasso)))
Y_gg_econ = np.load(VARX / "Y_pred_GG_econ.npy")
rows.append(("Golosnoy and Gribisch economic-loss model", eval_window(Y_gg_econ)))

# 5 covariance EWMA
lam = 0.94
S = (rets[:252].T @ rets[:252]) / 252 + 1e-4*np.eye(K)
Y5 = np.empty_like(Y_test)
for t in range(n_test):
    if t > 0:
        rp = rets[test_start + t - 1]
        S = lam*S + (1-lam)*np.outer(rp, rp)
    Ss = S + 1e-4*np.eye(K)
    try:
        w = np.linalg.solve(Ss, np.ones(K)); w = w/w.sum()
    except np.linalg.LinAlgError:
        w = np.ones(K)/K
    Y5[t] = w
rows.append(("Covariance EWMA (lambda=0.94)", eval_window(Y5)))

# partial adjustment, gamma selected on validation by common turnover target
def pred_val(model_id):
    coefs = np.load(VARX/f"fitted_models/coefs_model{model_id}.npy")
    inter = np.load(VARX/f"fitted_models/intercepts_model{model_id}.npy")
    cols = np.load(VARX/f"fitted_models/feat_cols_model{model_id}.npy")
    sc = joblib.load(VARX/f"fitted_models/scaler_model{model_id}.pkl")
    P = sc.transform(X_all[n_train:test_start][:, cols]) @ coefs.T + inter
    return norm(P)

def pa_path_and_turnover(base_v, r, y_prev, gamma):
    T = len(base_v); pos = y_prev.copy(); out = np.empty_like(base_v); to = np.empty(T)
    for t in range(T):
        pos_before = pos if t == 0 else drift(pos, r[t-1])
        pos_new = (1-gamma)*pos_before + gamma*base_v[t]
        pos_new = pos_new/pos_new.sum()
        out[t] = pos_new
        to[t] = 0.0 if t == 0 else float(np.sum(np.abs(pos_new - pos_before)))
        pos = pos_new
    return out, float(np.mean(to[1:]))

for mid, nm in [(3, "Partial-adjustment Sparse VARX"), (4, "Partial-adjustment Network VARX")]:
    v_val = pred_val(mid)
    v_test = np.load(VARX/f"Y_pred_model{mid}.npy")
    gammas = np.linspace(0.02, 1.0, 50)
    best_g, best_diff = None, np.inf
    for g in gammas:
        _, to_val = pa_path_and_turnover(v_val, r_val, Y_vprev, g)
        diff = abs(to_val - TARGET_TO)
        if diff < best_diff:
            best_diff, best_g = diff, g
    Ypa, _ = pa_path_and_turnover(v_test, r_test, Y_prev, best_g)
    rows.append((f"{nm} (gamma={best_g:.2f})", eval_window(Ypa)))

# 8 random forest (cached)
from sklearn.ensemble import RandomForestRegressor
rf_path = VARX/"Y_pred_rf.npy"
if rf_path.exists():
    Y_rf = np.load(rf_path)
else:
    print("Fitting random forest (100 trees, multi-output) ...", flush=True)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=42, min_samples_leaf=2)
    rf.fit(X_all[:n_train], Y_all[:n_train])
    Y_rf = norm(rf.predict(X_all[test_start:]))
    np.save(rf_path, Y_rf)
    print(f"  RF fit done in {time.time()-t0:.1f}s", flush=True)
rows.append(("Random forest", eval_window(Y_rf)))

# 9 Network VARX-S-DF
v6 = np.load(VARX/"Y_pred_model6_opt.npy")
rows.append(("Network VARX-S-DF", eval_window(v6)))

# print
print(f"{'Model':<46} {'MSE_w':>10} {'MAE_w':>10} {'RPV(ann)':>10} {'Turnover':>9} {'Net SR':>8}")
print("-"*100)
df_rows = []
for nm, m in rows:
    print(f"{nm:<46} {m['MSE']:>10.4e} {m['MAE']:>10.4e} {m['RPV']:>10.4e} {m['TO']:>9.4f} {m['SR']:>8.4f}")
    df_rows.append(dict(Model=nm, **m))

pd.DataFrame(df_rows).to_csv(OUT/"TableC1_partial.csv", index=False)
print("\nSaved: 性能评估与可视化/TableC1_partial.csv")
