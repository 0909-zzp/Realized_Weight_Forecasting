"""Stable end-to-end training (v2): box-constrained, lower LR, MSE-scaled anchor.

Improvements over v1:
  - project weights to |w_i| <= B and renormalize (same B as the post-hoc DFL);
  - anchor penalty is MSE-scaled (not sum-of-squares over 392 assets);
  - lr = 1e-4, gradient clipping, weight decay;
  - hyperparameters and stopping epoch chosen on the true 362-day validation block
    (validation net Sharpe), no in-sample dummy validation.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, pickle, itertools, json
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
K = 392
ETA_COST = 1e-4
EPS = 1e-6
BOX = 0.05


def load_all():
    X = np.load(PROJ / "特征工程" / "X_features.npy").astype(np.float64)
    Y = np.load(PROJ / "特征工程" / "Y_targets.npy").astype(np.float64)
    R = np.load(CACHE / "simple_returns.npy").astype(np.float64)
    cov = np.load(CACHE / "cov_cache.npy", mmap_mode="r")
    with open(PROJ / "VARX" / "fitted_models" / "scaler_model5.pkl", "rb") as f:
        scaler = pickle.load(f)
    W0 = np.load(PROJ / "VARX" / "fitted_models" / "coefs_model5.npy").astype(np.float64)
    b0 = np.load(PROJ / "VARX" / "fitted_models" / "intercepts_model5.npy").astype(np.float64)
    Xs = (X - scaler.mean_.astype(np.float64)) / scaler.scale_.astype(np.float64)
    n = len(Xs); n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train+n_val
    return Xs, Y, R, cov, W0, b0, n_train, n_val, test_start


def forward_box(W, b, Xb):
    Z = Xb @ W.T + b
    s = Z.sum(axis=1, keepdims=True)
    Wn = Z / s
    Wc = np.clip(Wn, -BOX, BOX)
    sc = Wc.sum(axis=1, keepdims=True)
    Wo = Wc / sc
    return Wo, (Z, s, Wn, Wc, sc)


def drift(w_prev, r):
    if w_prev.ndim == 1:
        return w_prev * (1 + r) / (1 + w_prev @ r)
    denom = 1 + np.einsum("bi,bi->b", w_prev, r)
    return w_prev * (1 + r) / denom[:, None]


def soft_abs(x):
    return np.sqrt(x*x + EPS*EPS) - EPS


def clip_grad(g, clip=1.0):
    n = np.linalg.norm(g)
    return g * (clip / (n + 1e-12)) if n > clip else g


def batch_loss_grad(W, b, Xb, Ab, Rb, Sb, Wprev, hp):
    B = len(Xb)
    Wo, cache = forward_box(W, b, Xb)
    Z, s, Wn, Wc, sc = cache
    risk = np.einsum("bi,bij,bj->b", Wo, Sb, Wo)
    gWo = 2.0 * np.einsum("bij,bj->bi", Sb, Wo)
    diff = Wo - Ab
    anchor = np.einsum("bi,bi->b", diff, diff) / K
    gWo += 2.0 * diff / K
    d = Wo - drift(Wprev, Rb)
    turnover = soft_abs(d).sum(axis=1)
    gWo += hp["eta_to"] * (d / np.sqrt(d*d + EPS*EPS))
    sm = np.einsum("bi,bi->b", Wo - Wprev, Wo - Wprev)
    gWo += 2.0 * hp["lam_s"] * (Wo - Wprev)
    loss = risk.mean() + hp["eta_to"]*turnover.mean() + hp["rho"]*anchor.mean() + hp["lam_s"]*sm.mean()
    # backprop through box + normalization
    dWc = gWo / sc - gWo.sum(axis=1, keepdims=True) / (sc*sc)
    dWn = dWc * (np.abs(Wn) < BOX)
    gz = dWn / s - dWn.sum(axis=1, keepdims=True) / (s*s)
    gW = gz.T @ Xb + hp["wd"] * W
    gb = gz.sum(axis=0)
    return loss, risk.mean(), turnover.mean(), anchor.mean(), sm.mean(), gW, gb, Wo


def val_stats(W, b, Xs, R, val_idx):
    Wv, _ = forward_box(W, b, Xs[val_idx])
    T = len(Wv) - 1
    gross = np.array([Wv[t] @ R[val_idx][t+1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = drift(Wv[t], R[val_idx][t+1])
        to[t] = np.abs(Wv[t+1] - wp).sum()
    net = gross - ETA_COST * to
    sd = net.std(ddof=1)
    sharpe = float(net.mean()/sd*np.sqrt(252)) if sd > 0 else 0.0
    return sharpe, float(to.mean())


def train(W0, b0, Xs, Y, R, cov, Bbase, train_idx, val_idx, hp, epochs, patience=8):
    W, b = W0.copy(), b0.copy()
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    best = {"sharpe": -1e9, "W": None, "b": None, "epoch": -1}
    wait = 0
    rng = np.random.default_rng(42)
    for ep in range(epochs):
        perm = rng.permutation(train_idx)
        t0 = time.time()
        for st in range(0, len(perm), hp["batch"]):
            idx = perm[st:st+hp["batch"]]
            prev = np.maximum(idx-1, 0)
            Wprev, _ = forward_box(W, b, Xs[prev])
            Sb = np.asarray(cov[np.minimum(idx+1, len(Xs)-1)], dtype=np.float64)
            loss, risk, turn, anch, sm, gW, gb, _ = batch_loss_grad(
                W, b, Xs[idx], Bbase[idx], R[idx], Sb, Wprev, hp)
            gW = clip_grad(gW, hp["clip"]); gb = clip_grad(gb, hp["clip"])
            t = ep*100000 + st + 1
            mW = hp["beta1"]*mW + (1-hp["beta1"])*gW; vW = hp["beta2"]*vW + (1-hp["beta2"])*gW*gW
            mb = hp["beta1"]*mb + (1-hp["beta1"])*gb; vb = hp["beta2"]*vb + (1-hp["beta2"])*gb*gb
            W -= hp["lr"]*(mW/(1-hp["beta1"]**t))/(np.sqrt(vW/(1-hp["beta2"]**t))+1e-8)
            b -= hp["lr"]*(mb/(1-hp["beta1"]**t))/(np.sqrt(vb/(1-hp["beta2"]**t))+1e-8)
        sharpe, to_val = val_stats(W, b, Xs, R, val_idx)
        if sharpe > best["sharpe"]:
            best = {"sharpe": sharpe, "W": W.copy(), "b": b.copy(), "epoch": ep}; wait = 0
        else:
            wait += 1
        if ep % 5 == 0 or ep == epochs-1:
            print(f"ep{ep:03d} loss={loss:.3e} risk={risk:.3e} turn={turn:.3f} "
                  f"anchor={anch:.3e} val_sharpe={sharpe:.3f} val_to={to_val:.3f} ({time.time()-t0:.1f}s)", flush=True)
        if wait >= patience:
            print("early stop at", ep); break
    return best


def main():
    Xs, Y, R, cov, W0, b0, n_train, n_val, test_start = load_all()
    train_idx = np.arange(0, n_train); val_idx = np.arange(n_train, test_start)
    Bbase, _ = forward_box(W0, b0, Xs)
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(Xs)-test_start}")

    grid = []
    for eta_to, rho in itertools.product([1e-3, 5e-3], [0.1, 1.0]):
        grid.append({"eta_to": eta_to, "rho": rho, "lam_s": 1e-3, "wd": 1e-6,
                     "lr": 1e-4, "batch": 64, "beta1": 0.9, "beta2": 0.999, "clip": 1.0})
    rows = []
    for gi, hp in enumerate(grid):
        print(f"\n--- v2 grid {gi+1}/{len(grid)}: eta={hp['eta_to']}, rho={hp['rho']} ---")
        best = train(W0, b0, Xs, Y, R, cov, Bbase, train_idx, val_idx, hp, epochs=40)
        rows.append({"eta_to": hp["eta_to"], "rho": hp["rho"], "val_sharpe": best["sharpe"],
                     "best_epoch": best["epoch"]})
        print("grid result:", rows[-1])
    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False)
    df.to_csv(HERE/"e2e_v2_val_grid.csv", index=False)
    print(df.to_string(index=False))

    best_row = df.iloc[0]
    hp_best = {"eta_to": float(best_row["eta_to"]), "rho": float(best_row["rho"]),
               "lam_s": 1e-3, "wd": 1e-6, "lr": 1e-4, "batch": 64,
               "beta1": 0.9, "beta2": 0.999, "clip": 1.0}
    epochs_final = int(best_row["best_epoch"]) + 1
    print(f"\n--- v2 final training on train+val, epochs={epochs_final}, hp={hp_best} ---")
    all_train = np.arange(0, test_start)
    W, b = W0.copy(), b0.copy()
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    rng = np.random.default_rng(42)
    for ep in range(epochs_final):
        perm = rng.permutation(all_train)
        for st in range(0, len(perm), hp_best["batch"]):
            idx = perm[st:st+hp_best["batch"]]; prev = np.maximum(idx-1, 0)
            Wprev, _ = forward_box(W, b, Xs[prev])
            Sb = np.asarray(cov[np.minimum(idx+1, len(Xs)-1)], dtype=np.float64)
            loss, risk, turn, anch, sm, gW, gb, _ = batch_loss_grad(
                W, b, Xs[idx], Bbase[idx], R[idx], Sb, Wprev, hp_best)
            gW = clip_grad(gW, hp_best["clip"]); gb = clip_grad(gb, hp_best["clip"])
            t = ep*100000 + st + 1
            mW = hp_best["beta1"]*mW + (1-hp_best["beta1"])*gW; vW = hp_best["beta2"]*vW + (1-hp_best["beta2"])*gW*gW
            mb = hp_best["beta1"]*mb + (1-hp_best["beta1"])*gb; vb = hp_best["beta2"]*vb + (1-hp_best["beta2"])*gb*gb
            W -= hp_best["lr"]*(mW/(1-hp_best["beta1"]**t))/(np.sqrt(vW/(1-hp_best["beta2"]**t))+1e-8)
            b -= hp_best["lr"]*(mb/(1-hp_best["beta1"]**t))/(np.sqrt(vb/(1-hp_best["beta2"]**t))+1e-8)
    Wte, _ = forward_box(W, b, Xs[np.arange(test_start, len(Xs))])
    np.save(HERE/"e2e_v2_test_weights.npy", Wte)
    np.savez(HERE/"e2e_v2_model.npz", W=W, b=b, hp=json.dumps(hp_best))
    Yte = Y[test_start:]
    print("v2 test MSE:", float(((Wte-Yte)**2).mean()))
    print("saved e2e_v2_test_weights.npy")


if __name__ == "__main__":
    main()
