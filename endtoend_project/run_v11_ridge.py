"""v11: direct full W + closed-form ridge initialization + decision fine-tuning.

No M5, no low-rank: the full 392x1185 coefficient matrix is initialized from a
ridge regression solved directly on the training data, then fine-tuned on the
decision-focused loss with a small learning rate and strong gradient clipping.
"""
import sys, time
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util, torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v10", str(HERE/"train_e2e_v10_full.py"))
v10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v10)
spec3 = importlib.util.spec_from_file_location("t3", str(HERE.parent/"性能评估与可视化"/"Table3_投资组合表现.py"))
t3 = importlib.util.module_from_spec(spec3); sys.modules["t3"]=t3; spec3.loader.exec_module(t3)

Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start = v10.load_all()
valid = np.load(HERE.parent/"特征工程"/"valid_indices.npy")
npy = sorted([f for f in (HERE.parent/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
files_full=[npy[i] for i in valid[test_start:]]; files_200=files_full[65:265]
Yte=Y[test_start:]

def ridge_init(X, Y, lam):
    F = X.shape[1]
    G = X.T.astype(np.float64) @ X.astype(np.float64) + lam*np.eye(F)
    W = np.linalg.solve(G, X.T.astype(np.float64) @ Y.astype(np.float64)).T
    b = Y.mean(0).astype(np.float64)
    return W.astype(np.float32), b.astype(np.float32)

def project(z):
    return z + (1.0 - z.sum(1, keepdims=True))/z.shape[1]

# choose ridge lambda on the validation block (no test leakage)
best_lam, best_mse, best_init = None, np.inf, None
for lam in (1e-2, 1e-1, 1.0, 10.0):
    W0, b0 = ridge_init(Xs[:n_train], Y[:n_train], lam)
    z = Xs[n_train:test_start] @ W0.T + b0
    w = project(z)
    mse = float(((w - Y[n_train:test_start])**2).mean())
    print(f"ridge lam={lam:g}: val MSE={mse:.4e}", flush=True)
    if mse < best_mse:
        best_lam, best_mse, best_init = lam, mse, (W0, b0)
print("selected ridge lam:", best_lam, "val MSE:", best_mse)
W0, b0 = best_init

hp = {"full":True, "a_risk":1.0, "a_turn":6.0, "a_smooth":0.3, "a_anchor":1.5,
      "a_gross":1.0, "gross_cap":1.5, "a_l1":1.0, "a_l2":1e-4,
      "lam1":1e-5, "lam2":1e-4, "lam3":5e-5, "wd":1e-6, "lr":1e-5,
      "clip":0.01, "box":None, "rank":8, "W_init":W0, "b_init":b0}

print("\n--- v11 fine-tuning on train (validation = 362 days) ---", flush=True)
torch.manual_seed(0); np.random.seed(0)
model, best, sc0 = v10.train(Xs,Y,R,cov,net,n_train,n_val,test_start,hp,
                             epochs=10, patience=5, log="v11 ")
print("v11 best:", {k:best[k] for k in ("epoch","sharpe","to","mse")}, flush=True)

# final: ridge init on train+val, fine-tune for the selected number of epochs
W0f, b0f = ridge_init(Xs[:test_start], Y[:test_start], best_lam)
hp_final = dict(hp); hp_final["W_init"]=W0f; hp_final["b_init"]=b0f
epochs_final = best["epoch"]+1
print(f"\n--- v11 final on train+val, epochs={epochs_final} ---", flush=True)
torch.manual_seed(0); np.random.seed(0)
model = v10.train_final(Xs,Y,R,cov,net,test_start,hp_final,epochs_final); model.eval()
with torch.no_grad():
    W = model(torch.from_numpy(Xs[test_start:])).numpy()
np.save(HERE/"e2e_v11_ridge_test_weights.npy", W)

def ev(W, files, sl):
    r=t3.compute_all({1:{"name":"v11","Y_pred":W[sl]}}, files)[1]
    return {"sharpe":r["sharpe_net"],"turnover":r["avg_turnover"],"vol":r["vol_annual"],
            "maxDD":r["max_dd"],"cum":r["cum_ret"],
            "MSE":float(((W[sl]-Yte[sl])**2).mean()),"gross":float(np.abs(W[sl]).sum(1).mean())}
m200=ev(W,files_200,slice(65,265)); mfull=ev(W,files_full,slice(0,363))
print("v11 200d:", {k:round(v,5) for k,v in m200.items()})
print("v11 full:", {k:round(v,5) for k,v in mfull.items()})
pd.DataFrame([{"window":"200d",**m200},{"window":"full363",**mfull}]).to_csv(HERE/"e2e_v11_metrics.csv", index=False)
