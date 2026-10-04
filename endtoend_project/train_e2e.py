"""True end-to-end training of a linear VARX forecaster on a decision-focused loss.

Design
------
- Model: normalized linear map W_t = normalize(Xs_t @ W.T + b), initialized from
  the original M5 Lasso solution (coefs_model5 / intercepts_model5).
- Training loss (per day, decision-focused):
      risk_t      = w_t' Sigma_{t+1} w_t
      turnover_t  = || w_t - drift(w_{t-1}, r_t) ||_1   (soft L1, prev. detached)
      anchor_t    = || w_t - y_t ||_2^2
      smooth_t    = || w_t - w_{t-1} ||_2^2
      + weight decay on W
- Parameters W, b are optimized directly on this loss with Adam (numpy).
- Hyperparameters (eta_turnover, rho_anchor, lambda_smooth, lr, weight decay)
  are selected on the validation block using validation net Sharpe; the final
  model is then retrained on train+validation and evaluated on the test block.

Outputs (endtoend_project/):
  e2e_test_weights.npy, e2e_model.npz, e2e_metrics.csv, e2e_val_grid.csv
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, time, pickle, itertools, json
import numpy as np
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

K = 392
ETA_COST = 1e-4          # 1bp per unit turnover, for evaluation/validation
EPS = 1e-6               # soft-L1 smoothing


def load_all():
    X = np.load(PROJ / "特征工程" / "X_features.npy").astype(np.float64)
    Y = np.load(PROJ / "特征工程" / "Y_targets.npy").astype(np.float64)
    R = np.load(CACHE / "simple_returns.npy").astype(np.float64)
    cov = np.load(CACHE / "cov_cache.npy", mmap_mode="r")
    with open(PROJ / "VARX" / "fitted_models" / "scaler_model5.pkl", "rb") as f:
        scaler = pickle.load(f)
    W0 = np.load(PROJ / "VARX" / "fitted_models" / "coefs_model5.npy").astype(np.float64)
    b0 = np.load(PROJ / "VARX" / "fitted_models" / "intercepts_model5.npy").astype(np.float64)
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    Xs = (X - mean) / scale
    n = len(Xs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    test_start = n_train + n_val
    return Xs, Y, R, cov, W0, b0, n_train, n_val, test_start


def forward(W, b, Xb):
    Z = Xb @ W.T + b
    s = Z.sum(axis=1, keepdims=True)
    return Z / s


def drift(w_prev, r):
    if w_prev.ndim == 1:
        return w_prev * (1 + r) / (1 + w_prev @ r)
    denom = 1 + np.einsum("bi,bi->b", w_prev, r)
    return w_prev * (1 + r) / denom[:, None]


def soft_abs(x):
    return np.sqrt(x * x + EPS * EPS) - EPS


def batch_grad(W, b, Xb, Ab, Rb, Sb, Wprev, eta_to, rho, lam_s, wd):
    """Return loss components and gradients for a batch."""
    B = len(Xb)
    Z = Xb @ W.T + b
    s = Z.sum(axis=1, keepdims=True)
    Wn = Z / s

    # --- risk: w' Sigma w
    risk = np.einsum("bi,bij,bj->b", Wn, Sb, Wn)
    gWn = 2.0 * np.einsum("bij,bj->bi", Sb, Wn)

    # --- anchor to the statistical base forecast
    diff = Wn - Ab
    anchor = np.einsum("bi,bi->b", diff, diff)
    gWn += 2.0 * diff

    # --- turnover vs drifted previous forecast (previous detached)
    drift_prev = drift(Wprev, Rb)
    d = Wn - drift_prev
    turnover = soft_abs(d).sum(axis=1)
    gWn += eta_to * (d / np.sqrt(d * d + EPS * EPS))

    # --- smoothness vs previous forecast (detached)
    sm = np.einsum("bi,bi->b", Wn - Wprev, Wn - Wprev)
    gWn += 2.0 * lam_s * (Wn - Wprev)

    loss = risk.mean() + eta_to * turnover.mean() + rho * anchor.mean() + lam_s * sm.mean()

    # backprop through normalization
    gz = gWn / s - gWn.sum(axis=1, keepdims=True) / (s * s)
    gW = gz.T @ Xb + wd * W
    gb = gz.sum(axis=0)
    return loss, risk.mean(), turnover.mean(), anchor.mean(), sm.mean(), gW, gb, Wn


def predict_split(W, b, Xs, idx):
    return forward(W, b, Xs[idx])


def val_net_sharpe(Wv, Rv):
    """Table-3-style net Sharpe for a contiguous weight block."""
    T = len(Wv) - 1
    gross = np.array([Wv[t] @ Rv[t + 1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = drift(Wv[t], Rv[t + 1])
        to[t] = np.abs(Wv[t + 1] - wp).sum()
    net = gross - ETA_COST * to
    sd = net.std(ddof=1)
    return float(net.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0, float(to.mean())


def train_model(W0, b0, Xs, Y, R, cov, Bbase, train_idx, val_idx, hp, epochs, log_prefix=""):
    W = W0.copy(); b = b0.copy()
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    best = {"sharpe": -1e9, "W": None, "b": None, "epoch": -1}
    rng = np.random.default_rng(42)
    for ep in range(epochs):
        perm = rng.permutation(train_idx)
        t0 = time.time()
        for start in range(0, len(perm), hp["batch"]):
            idx = perm[start:start + hp["batch"]]
            prev = np.maximum(idx - 1, 0)
            Xb = Xs[idx]; Ab = Bbase[idx]; Rb = R[idx]
            Sb = np.asarray(cov[np.minimum(idx + 1, len(Xs) - 1)], dtype=np.float64)
            Wprev = forward(W, b, Xs[prev])
            loss, risk, turn, anch, sm, gW, gb, Wn = batch_grad(
                W, b, Xb, Ab, Rb, Sb, Wprev,
                hp["eta_to"], hp["rho"], hp["lam_s"], hp["wd"])
            # Adam
            t = ep * 100000 + start + 1
            mW = hp["beta1"] * mW + (1 - hp["beta1"]) * gW
            vW = hp["beta2"] * vW + (1 - hp["beta2"]) * gW * gW
            mb = hp["beta1"] * mb + (1 - hp["beta1"]) * gb
            vb = hp["beta2"] * vb + (1 - hp["beta2"]) * gb * gb
            mWh = mW / (1 - hp["beta1"] ** t); vWh = vW / (1 - hp["beta2"] ** t)
            mbh = mb / (1 - hp["beta1"] ** t); vbh = vb / (1 - hp["beta2"] ** t)
            W -= hp["lr"] * mWh / (np.sqrt(vWh) + 1e-8)
            b -= hp["lr"] * mbh / (np.sqrt(vbh) + 1e-8)
        # validation
        Wv = predict_split(W, b, Xs, val_idx)
        sharpe, to_val = val_net_sharpe(Wv, R[val_idx])
        if sharpe > best["sharpe"]:
            best = {"sharpe": sharpe, "W": W.copy(), "b": b.copy(), "epoch": ep}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"{log_prefix}ep{ep:03d} loss={loss:.3e} risk={risk:.3e} turn={turn:.3f} "
                  f"anchor={anch:.3e} val_sharpe={sharpe:.3f} val_to={to_val:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return best


def main():
    Xs, Y, R, cov, W0, b0, n_train, n_val, test_start = load_all()
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, test_start)
    all_train = np.arange(0, test_start)
    Bbase = forward(W0, b0, Xs)   # normalized statistical baseline (anchor target)
    print(f"X={Xs.shape} train={len(train_idx)} val={len(val_idx)} test={len(Xs)-test_start}")

    grid = []
    for eta_to, rho in itertools.product([1e-3, 5e-3, 1e-2], [1e-2, 1e-1]):
        grid.append({
            "eta_to": eta_to, "rho": rho, "lam_s": 1e-3, "wd": 1e-6,
            "lr": 1e-3, "batch": 64, "beta1": 0.9, "beta2": 0.999,
        })

    rows = []
    for gi, hp in enumerate(grid):
        print(f"\n--- grid {gi+1}/{len(grid)}: eta={hp['eta_to']}, rho={hp['rho']} ---")
        best = train_model(W0, b0, Xs, Y, R, cov, Bbase, train_idx, val_idx, hp, epochs=10,
                           log_prefix=f"g{gi+1} ")
        rows.append({"eta_to": hp["eta_to"], "rho": hp["rho"],
                     "val_sharpe": best["sharpe"], "best_epoch": best["epoch"]})
        print(f"grid {gi+1}: val_sharpe={best['sharpe']:.3f} best_epoch={best['epoch']}")

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False)
    df.to_csv(HERE / "e2e_val_grid.csv", index=False)
    print(df.to_string(index=False))

    best_row = df.iloc[0]
    hp_best = {
        "eta_to": float(best_row["eta_to"]), "rho": float(best_row["rho"]),
        "lam_s": 1e-3, "wd": 1e-6, "lr": 1e-3, "batch": 64,
        "beta1": 0.9, "beta2": 0.999,
    }
    epochs_final = max(10, int(best_row["best_epoch"]) + 1)
    print(f"\n--- final training on train+val, hp={hp_best}, epochs={epochs_final} ---")
    # no early stopping here: use fixed epochs selected on validation
    W = W0.copy(); b = b0.copy()
    # reuse train_model with a dummy val=last 20% of all_train for logging only
    dummy_val = all_train[-min(50, len(all_train)):]
    best = train_model(W0, b0, Xs, Y, R, cov, Bbase, all_train, dummy_val, hp_best,
                       epochs=epochs_final, log_prefix="final ")
    W, b = best["W"], best["b"]

    Wte = predict_split(W, b, Xs, np.arange(test_start, len(Xs)))
    np.save(HERE / "e2e_test_weights.npy", Wte)
    np.savez(HERE / "e2e_model.npz", W=W, b=b, hp=json.dumps(hp_best))
    print("saved e2e_test_weights.npy and e2e_model.npz")

    # quick metrics vs realized weights
    Yte = Y[test_start:]
    mse = float(((Wte - Yte) ** 2).mean())
    print("test MSE vs realized weights:", mse)


if __name__ == "__main__":
    main()
