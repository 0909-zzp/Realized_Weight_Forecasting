"""v8: strict full-parameter implementation of the paper's Eq.(17)-(18).

Objective (paper):
  L = (1/T) sum_t [ w_{t+1}' Sigma_{t+1} w_{t+1}
                  + eta * || w_{t+1} - w_t^+ ||_1 ]
    + rho * (1/T) sum_t || y_{t+1} - w_{t+1} ||_2^2
    + P(Theta)
with P = network-adaptive L1 (lambda1/2/3) + lambda_s * smoothness.
Model: full linear VARX map, w = z / (1'z), z = X W' + b.
No low-rank, no box, no extra L2, no M5 initialization/anchor.
Paper values: eta=1e-6, rho=1e-3, lambda1=3e-4, lambda2=1.3e-3,
lambda3=5e-4, lambda_s=3e-3. Full BPTT over sequences.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, json, math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
K = 392
ETA_COST = 1e-4
L_SEQ = 20
B_SEQ = 8
EPS = 1e-8

# paper values
ETA_TURN = 1e-6
RHO_ANCHOR = 1e-3
LAM1 = 3e-4
LAM2 = 1.3e-3
LAM3 = 5e-4
LAM_S = 3e-3


def load_all():
    X = np.load(PROJ/"特征工程"/"X_features.npy").astype(np.float32)
    Y = np.load(PROJ/"特征工程"/"Y_targets.npy").astype(np.float32)
    A_bar = np.load(PROJ/"特征工程"/"A_bar.npy")
    R = np.load(CACHE/"simple_returns.npy").astype(np.float32)
    cov = np.load(CACHE/"cov_cache.npy", mmap_mode="r")
    n = len(X); n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train+n_val
    mean = X[:n_train].mean(0); std = X[:n_train].std(0); std[std < 1e-8] = 1.0
    Xs = (X - mean)/std
    net = (A_bar[:n_train].mean(0) >= 0.7).astype(np.float32)
    np.fill_diagonal(net, 0.0)
    return Xs, Y, R, cov, net, n_train, n_val, test_start


class FullForecaster(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(K, F))   # neutral start: no signal
        self.b = nn.Parameter(torch.full((K,), 1.0/K))  # neutral start: equal weight

    def forward(self, X):
        z = X @ self.W.T + self.b
        # exact sum-to-one projection (Eq. 11 normalization, numerically stable)
        return z + (1.0 - z.sum(dim=-1, keepdim=True)) / K


def build_penalty(net):
    F = 3*K + 9
    P = torch.zeros(K, F)
    for l in range(3):
        conn = torch.from_numpy(net)
        P[:, l*K:(l+1)*K] = torch.where(conn > 0, LAM1, LAM2)
        P[:, l*K:(l+1)*K].fill_diagonal_(LAM1)
    P[:, 3*K:] = LAM3
    return P


def drift(w_prev, r):
    denom = 1.0 + (w_prev*r).sum(-1, keepdim=True)
    return w_prev*(1.0+r)/denom


def seq_loss(model, Xs_t, Y_t, R_t, cov, starts, penalty):
    B = len(starts)
    idx = torch.stack([torch.arange(s, s+L_SEQ) for s in starts])
    X_seq = Xs_t[idx]; Y_seq = Y_t[idx]; R_seq = R_t[idx]
    W_seq = model(X_seq.reshape(B*L_SEQ, -1)).reshape(B, L_SEQ, K)
    cov_idx = (idx+1).clamp(max=cov.shape[0]-1)
    Sig = torch.from_numpy(np.asarray(cov[cov_idx.numpy()], dtype=np.float32))
    risk = torch.einsum("blk,blkj,blj->bl", W_seq, Sig, W_seq)
    prev_idx = torch.clamp(idx[:,0]-1, min=0)
    w_prev0 = model(Xs_t[prev_idx]).detach()
    W_prev = torch.cat([w_prev0.unsqueeze(1), W_seq[:,:-1]], dim=1)
    drift_prev = drift(W_prev, R_seq)
    turn = (W_seq - drift_prev).abs().sum(dim=-1)          # exact L1
    smooth = ((W_seq - W_prev)**2).sum(dim=-1)
    anchor = ((W_seq - Y_seq)**2).sum(dim=-1)/K            # prediction accuracy term
    l1 = (penalty * model.W.abs()).sum()
    loss = (risk.mean() + ETA_TURN*turn.mean() + RHO_ANCHOR*anchor.mean()
            + l1 + LAM_S*smooth.mean())
    return loss, risk.mean(), turn.mean(), anchor.mean(), l1, smooth.mean()


@torch.no_grad()
def val_metrics(model, Xs_t, Y_t, R_t, val_idx):
    w = model(Xs_t[val_idx]).numpy(); Rv = R_t[val_idx].numpy()
    T = len(w)-1
    gross = np.array([w[t] @ Rv[t+1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = w[t]*(1+Rv[t+1])/(1+w[t]@Rv[t+1]); to[t] = np.abs(w[t+1]-wp).sum()
    net = gross - ETA_COST*to; sd = net.std(ddof=1)
    sharpe = float(net.mean()/sd*np.sqrt(252)) if sd > 0 else 0.0
    mse = float(((w - Y_t[val_idx].numpy())**2).mean())
    return sharpe, float(to.mean()), mse, float(np.abs(w).sum(1).mean())


def main():
    Xs, Y, R, cov, net, n_train, n_val, test_start = load_all()
    F = Xs.shape[1]
    Xs_t = torch.from_numpy(Xs); Y_t = torch.from_numpy(Y); R_t = torch.from_numpy(R)
    penalty = build_penalty(net)
    model = FullForecaster(F)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)  # no extra L2
    start_pool = np.arange(0, n_train-L_SEQ+1)
    best = {"score": -1e9, "state": None, "epoch": -1, "sharpe": None, "to": None, "mse": None}
    wait = 0; rng = np.random.default_rng(42)
    epochs, patience = 15, 5
    print(f"train={n_train} val={n_val} test={len(Xs)-test_start} F={F} params={K*F+K}")
    for ep in range(epochs):
        perm = rng.permutation(start_pool); t0 = time.time()
        for st in range(0, len(perm)-B_SEQ+1, B_SEQ):
            starts = perm[st:st+B_SEQ]
            loss, risk, turn, anch, l1, smooth = seq_loss(model, Xs_t, Y_t, R_t, cov, starts, penalty)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sharpe, to_val, mse_val, gross = val_metrics(model, Xs_t, Y_t, R_t, np.arange(n_train, test_start))
        score = sharpe - 5*max(0.0, to_val-0.5)
        if score > best["score"]:
            best.update({"score": score, "state": {k:v.detach().clone() for k,v in model.state_dict().items()},
                         "epoch": ep, "sharpe": sharpe, "to": to_val, "mse": mse_val}); wait = 0
        else: wait += 1
        if ep % 2 == 0 or ep == epochs-1:
            print(f"ep{ep:03d} loss={float(loss):.3e} risk={float(risk):.2e} turn={float(turn):.3f} "
                  f"anchor={float(anch):.2e} l1={float(l1):.3f} smooth={float(smooth):.2e} "
                  f"val_sharpe={sharpe:.3f} val_to={to_val:.3f} val_mse={mse_val:.2e} gross={gross:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if wait >= patience:
            print("early stop at", ep); break
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        Wte = model(torch.from_numpy(Xs[test_start:])).numpy()
    np.save(HERE/"e2e_v8b_projected_test_weights.npy", Wte)
    np.savez(HERE/"e2e_v8b_projected_model.npz", W=model.W.detach().numpy(), b=model.b.detach().numpy())
    print("v8 best epoch", best["epoch"], "val_sharpe", best["sharpe"], "val_mse", best["mse"])
    print("v8 test MSE:", float(((Wte-Y[test_start:])**2).mean()))
    print("saved e2e_v8b_projected_test_weights.npy")


if __name__ == "__main__":
    main()
