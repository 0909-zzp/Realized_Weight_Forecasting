"""End-to-end v6: from-scratch decision-focused training with full BPTT.

Key design (no M5 initialization, no M5 anchor):
  - Model: w = sum1_project(Xs @ (U V^T).T + b), with U (K x r), V (r x F),
    b initialized to 1/K (neutral equal-weight prior). Random U,V start.
  - Full BPTT: training samples are contiguous sequences of length L_SEQ;
    the turnover/smoothness terms use the previous *predicted* weight with
    gradients flowing through it, so the model learns the dynamic trade-off.
  - Decision loss: normalized risk (w' Sigma_{t+1} w) + turnover penalty
    + smoothness + network-adaptive L1 on W + L2 + optional soft box penalty.
  - Validation: true 362-day block, net Sharpe with a turnover constraint;
    early stopping and hyperparameter selection on validation only.
  - Final: retrain on train+validation for the selected number of epochs.

No original file is modified.
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
EPS = 1e-6
L_SEQ = 20
B_SEQ = 8


def load_all():
    X = np.load(PROJ/"特征工程"/"X_features.npy").astype(np.float32)
    Y = np.load(PROJ/"特征工程"/"Y_targets.npy").astype(np.float32)
    A_bar = np.load(PROJ/"特征工程"/"A_bar.npy")
    R = np.load(CACHE/"simple_returns.npy").astype(np.float32)
    cov = np.load(CACHE/"cov_cache.npy", mmap_mode="r")
    n = len(X); n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train+n_val
    mean = X[:n_train].mean(0); std = X[:n_train].std(0); std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    # network mask from training origins
    A_tr = A_bar[:n_train]
    net = (A_tr.mean(0) >= 0.7).astype(np.float32)
    np.fill_diagonal(net, 0.0)
    return Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start


class FullWForecaster(nn.Module):
    """Direct full parameterization W (K x F): the paper's coefficient matrix."""
    def __init__(self, F, K=392):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(K, F))
        self.b = nn.Parameter(torch.full((K,), 1.0/K))

    def weights(self, X):
        z = X @ self.W.T + self.b
        return z + (1.0 - z.sum(dim=-1, keepdim=True)) / self.W.shape[0]

    def forward(self, X):
        return self.weights(X)


class LowRankForecaster(nn.Module):
    def __init__(self, F, r, K=392, box=None):
        super().__init__()
        self.U = nn.Parameter(torch.randn(K, r) * 0.01)
        self.V = nn.Parameter(torch.randn(r, F) * 0.01)
        self.b = nn.Parameter(torch.full((K,), 1.0/K))
        self.box = box

    def weights(self, X):
        W = self.U @ self.V
        z = X @ W.T + self.b
        w = z + (1.0 - z.sum(dim=-1, keepdim=True)) / K
        return w

    def forward(self, X):
        """X: (N,F) -> weights (N,K)."""
        w = self.weights(X)
        if self.box is not None:
            w = torch.clamp(w, -self.box, self.box)
            w = w + (1.0 - w.sum(dim=-1, keepdim=True)) / K
        return w


def drift_torch(w_prev, r):
    denom = 1.0 + (w_prev * r).sum(dim=-1, keepdim=True)
    return w_prev * (1.0 + r) / denom


def soft_abs(x):
    return torch.sqrt(x*x + EPS*EPS) - EPS


def build_penalty(net, F, p_lags=3, lam1=1e-5, lam2=1e-4, lam3=5e-5):
    """Penalty matrix (K,F) for network-adaptive L1 on W."""
    P = torch.zeros(K, F)
    for l in range(p_lags):
        block = P[:, l*K:(l+1)*K]
        conn = torch.from_numpy(net)  # (K,K)
        block.copy_(torch.where(conn > 0, lam1, lam2))
        block.fill_diagonal_(lam1)
    P[:, p_lags*K:] = lam3
    return P


def seq_loss(model, Xs_t, Y_t, R_t, cov, starts, sc0, hp, penalty):
    """X/R are torch tensors; starts: (B,) start indices; returns loss and parts."""
    B = len(starts)
    idx = torch.stack([torch.arange(s, s+L_SEQ) for s in starts])  # (B,L)
    X_seq = Xs_t[idx]                    # (B,L,F)
    Y_seq = Y_t[idx]                     # (B,L,K)
    R_seq = R_t[idx]                     # (B,L,K)
    W_seq = model.weights(X_seq.reshape(B*L_SEQ, -1)).reshape(B, L_SEQ, K)
    # risk uses Sigma_{t+1}
    cov_idx = (idx + 1).clamp(max=cov.shape[0]-1)
    Sig = torch.from_numpy(np.asarray(cov[cov_idx.numpy()], dtype=np.float32))  # (B,L,K,K)
    risk = torch.einsum("blk,blkj,blj->bl", W_seq, Sig, W_seq)
    # previous weight: for t=0 use detached prediction at start-1 (or equal weight)
    prev_idx = torch.clamp(idx[:, 0]-1, min=0)
    w_prev0 = model.weights(Xs_t[prev_idx]).detach()
    W_prev = torch.cat([w_prev0.unsqueeze(1), W_seq[:, :-1]], dim=1)  # (B,L,K)
    drift_prev = drift_torch(W_prev, R_seq)
    turn = soft_abs(W_seq - drift_prev).sum(dim=-1)                   # (B,L)
    smooth = ((W_seq - W_prev)**2).sum(dim=-1)                        # (B,L)
    W = model.W if hasattr(model, 'W') else (model.U @ model.V)
    anchor = ((W_seq - Y_seq)**2).sum(dim=-1)          # prediction-accuracy term
    gross = W_seq.abs().sum(dim=-1)
    gross_pen = torch.relu(gross - hp.get("gross_cap", 1.5)).mean()
    l1 = (penalty * soft_abs(W)).sum()
    l2 = (W**2).sum()
    if hasattr(model, 'U'):
        l2 = l2 + 1e-6*(model.U**2).sum() + 1e-6*(model.V**2).sum()
    loss = (hp["a_risk"]*risk.mean()/sc0["risk0"] + hp["a_turn"]*turn.mean()/sc0["turn0"]
            + hp["a_smooth"]*smooth.mean()/sc0["smooth0"]
            + hp["a_anchor"]*anchor.mean()/sc0["anchor0"]
            + hp.get("a_gross",0.0)*gross_pen
            + hp["a_l1"]*l1 + hp["a_l2"]*l2)
    return loss, risk.mean(), turn.mean(), smooth.mean(), l1, l2


@torch.no_grad()
def val_metrics(model, Xs_t, Y_t, R_t, cov, val_idx, turnover_cap=0.5):
    w = model(Xs_t[val_idx]).numpy()
    Rv = R_t[val_idx].numpy()
    T = len(w)-1
    gross = np.array([w[t] @ Rv[t+1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = w[t]*(1+Rv[t+1])/(1+w[t]@Rv[t+1])
        to[t] = np.abs(w[t+1]-wp).sum()
    net = gross - ETA_COST*to
    sd = net.std(ddof=1)
    sharpe = float(net.mean()/sd*np.sqrt(252)) if sd > 0 else 0.0
    mse = float(((w - Y_t[val_idx].numpy())**2).mean())
    return sharpe, float(to.mean()), mse, float(np.abs(w).sum(1).mean())


def data_scales(Y, cov, idx):
    """Normalization scales from realized weights (stable for zero W init)."""
    Yb = np.asarray(Y[idx], dtype=np.float64)
    ii = np.minimum(np.asarray(idx)+1, len(Y)-1)
    sig = np.asarray(cov[ii], dtype=np.float64)
    risk0 = float(np.einsum("bi,bij,bj->b", Yb, sig, Yb).mean())
    if len(Yb) > 1:
        d = Yb[1:] - Yb[:-1]
        turn0 = float(np.abs(d).sum(axis=1).mean())
        smooth0 = float((d**2).sum(axis=1).mean())
    else:
        turn0, smooth0 = 1.0, 1.0
    anchor0 = float(((Yb - 1.0/K)**2).sum(axis=1).mean())
    return {"risk0": max(risk0,1e-8), "turn0": max(turn0,1e-8),
            "smooth0": max(smooth0,1e-8), "anchor0": max(anchor0,1e-8)}


def train(Xs, Y, R, cov, net, n_train, n_val, test_start, hp, epochs=30, patience=8, log=""):
    F = Xs.shape[1]
    Xs_t = torch.from_numpy(Xs); Y_t = torch.from_numpy(Y); R_t = torch.from_numpy(R)
    penalty = build_penalty(net, F, lam1=hp["lam1"], lam2=hp["lam2"], lam3=hp["lam3"])
    model = FullWForecaster(F) if hp.get("full", False) else LowRankForecaster(F, hp["rank"], box=hp.get("box"))
    if hp.get("W_init") is not None:
        with torch.no_grad():
            model.W.copy_(torch.from_numpy(hp["W_init"]).float())
            model.b.copy_(torch.from_numpy(hp["b_init"]).float())

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    start_pool = np.arange(0, n_train - L_SEQ + 1)
    # normalization scales from realized weights (stable for zero W init)
    sc0 = data_scales(Y, cov, np.arange(0, n_train))
    print(log, "scales:", {k: round(v, 8) for k, v in sc0.items()}, flush=True)

    best = {"score": -1e9, "state": None, "epoch": -1, "sharpe": None, "to": None, "mse": None}
    turnover_cap = hp.get("turnover_cap", 0.5)
    wait = 0
    rng = np.random.default_rng(42)
    total_steps = epochs * max(1, len(start_pool)//B_SEQ)
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(start_pool)
        t0 = time.time()
        for st in range(0, len(perm)-B_SEQ+1, B_SEQ):
            starts = perm[st:st+B_SEQ]
            # warmup + cosine lr
            frac = step / max(1, total_steps)
            lr = hp["lr"] * (min(1.0, frac/0.05) if frac < 0.05 else 0.5*(1+math.cos(math.pi*min(1.0, frac))))
            for g in opt.param_groups: g["lr"] = max(lr, hp["lr"]*0.05)
            loss, risk, turn, smooth, l1, l2 = seq_loss(
                model, Xs_t, Y_t, R_t, cov, starts, sc0, hp, penalty)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.get('clip',1.0))
            opt.step(); step += 1
        sharpe, to_val, mse_val, gross = val_metrics(model, Xs_t, Y_t, R_t, cov, np.arange(n_train, test_start))
        penalty_score = 10.0*max(0.0, to_val-0.10)
        gross_pen_score = 10.0*max(0.0, gross-1.5)
        mse_pen = 10.0*max(0.0, mse_val/3.0e-5 - 1.0)
        score = sharpe - penalty_score - gross_pen_score - mse_pen
        if score > best["score"]:
            best.update({"score": score, "state": {k: v.detach().clone() for k,v in model.state_dict().items()},
                         "epoch": ep, "sharpe": sharpe, "to": to_val, "mse": mse_val}); wait = 0
        else:
            wait += 1
        if ep % 5 == 0 or ep == epochs-1:
            print(f"{log}ep{ep:03d} loss={float(loss):.3f} risk={float(risk):.2e} turn={float(turn):.3f} "
                  f"smooth={float(smooth):.2e} l1={float(l1):.3f} val_sharpe={sharpe:.3f} "
                  f"val_to={to_val:.3f} val_mse={mse_val:.2e} gross={gross:.3f} lr={lr:.1e} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if wait >= patience:
            print(log, "early stop at", ep); break
    model.load_state_dict(best["state"])
    return model, best, sc0


def main():
    Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start = load_all()
    print(f"train={n_train} val={n_val} test={len(Xs)-test_start} L_SEQ={L_SEQ}")
    grid = []
    for a_turn in (1.0, 2.0, 4.0):
        for a_anchor in (1.0, 3.0):
            grid.append({"rank":8, "a_risk":1.0, "a_turn":a_turn, "a_smooth":0.3,
                         "a_anchor":a_anchor, "a_gross":1.0, "gross_cap":1.5,
                         "a_l1":1.0, "a_l2":1e-4, "lam1":1e-5, "lam2":1e-4, "lam3":5e-5,
                         "wd":1e-6, "lr":1e-3, "box":None})
    rows = []
    models = {}
    for gi, hp in enumerate(grid):
        print(f"\n--- v6 grid {gi+1}/{len(grid)}: rank={hp['rank']} turn={hp['a_turn']} box={hp['box']} ---")
        model, best, sc0 = train(Xs, Y, R, cov, net, n_train, n_val, test_start, hp,
                                 epochs=30, patience=8, log=f"g{gi+1} ")
        rows.append({"gi":gi, "rank":hp["rank"], "a_turn":hp["a_turn"], "a_anchor":hp["a_anchor"], "a_gross":hp["a_gross"], "box":hp["box"],
                     "val_sharpe":best["sharpe"], "val_turnover":best["to"],
                     "val_mse":best["mse"], "best_epoch":best["epoch"]})
        models[gi] = (model, hp, best["epoch"]+1)
        print("grid result:", rows[-1])
    df = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False)
    df.to_csv(HERE/"e2e_v6_val_grid.csv", index=False); print(df.to_string(index=False))

    best_row = df.iloc[0]
    best_gi = int(best_row["gi"])
    hp = grid[best_gi]
    epochs_final = int(best_row["best_epoch"]) + 1
    print(f"\n--- v6 final on train+val: grid {best_gi+1}, epochs={epochs_final} ---")
    model = train_final(Xs, Y, R, cov, net, test_start, hp, epochs_final)
    model.eval()
    with torch.no_grad():
        Wte = model(torch.from_numpy(Xs[test_start:])).numpy()
    np.save(HERE/"e2e_v6_test_weights.npy", Wte)
    np.savez(HERE/"e2e_v6_model.npz", U=model.U.detach().numpy(), V=model.V.detach().numpy(),
             b=model.b.detach().numpy(), hp=json.dumps(hp))
    print("v6 test MSE:", float(((Wte - Y[test_start:])**2).mean()))
    print("saved e2e_v6_test_weights.npy")


def train_final(Xs, Y, R, cov, net, n_origins, hp, epochs):
    """Fixed-epoch from-scratch training on origins [0, n_origins)."""
    F = Xs.shape[1]
    Xs_t = torch.from_numpy(Xs); Y_t = torch.from_numpy(Y); R_t = torch.from_numpy(R)
    penalty = build_penalty(net, F, lam1=hp["lam1"], lam2=hp["lam2"], lam3=hp["lam3"])
    model = FullWForecaster(F) if hp.get("full", False) else LowRankForecaster(F, hp["rank"], box=hp.get("box"))
    if hp.get("W_init") is not None:
        with torch.no_grad():
            model.W.copy_(torch.from_numpy(hp["W_init"]).float())
            model.b.copy_(torch.from_numpy(hp["b_init"]).float())

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    start_pool = np.arange(0, n_origins - L_SEQ + 1)
    sc0 = data_scales(Y, cov, np.arange(0, n_origins))
    print("final scales:", {k: round(v,8) for k,v in sc0.items()}, flush=True)
    rng = np.random.default_rng(42)
    total_steps = epochs * max(1, len(start_pool)//B_SEQ)
    step = 0
    for ep in range(epochs):
        perm = rng.permutation(start_pool)
        t0 = time.time()
        for st in range(0, len(perm)-B_SEQ+1, B_SEQ):
            starts = perm[st:st+B_SEQ]
            frac = step/max(1,total_steps)
            lr = hp["lr"]*(min(1.0, frac/0.05) if frac < 0.05 else 0.5*(1+math.cos(math.pi*min(1.0,frac))))
            for g in opt.param_groups: g["lr"] = max(lr, hp["lr"]*0.05)
            loss, risk, turn, smooth, l1, l2 = seq_loss(model, Xs_t, Y_t, R_t, cov, starts, sc0, hp, penalty)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.get('clip',1.0))
            opt.step(); step += 1
        print(f"final ep{ep:03d} loss={float(loss):.3f} risk={float(risk):.2e} turn={float(turn):.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return model


if __name__ == "__main__":
    main()
