"""Cost-aware post-hoc DFL curve (validation-selected eta per cost)."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = PROJ/"论文"/"figures"; OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("vp", str(PROJ/"VARX"/"VAR及拓展（table2）.py"))
vp = importlib.util.module_from_spec(spec); sys.modules["vp"]=vp; spec.loader.exec_module(vp)

X = np.load(PROJ/"特征工程"/"X_features.npy")
Y = np.load(PROJ/"特征工程"/"Y_targets.npy")
A = np.load(PROJ/"特征工程"/"A_bar.npy")
valid = np.load(PROJ/"特征工程"/"valid_indices.npy")
n = len(X); n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train+n_val
npy = sorted([f for f in (PROJ/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
R_val = np.array([np.expm1(np.load(npy[i]).sum(axis=1)) for i in valid[n_train:test_start]])
R_test = np.array([np.expm1(np.load(npy[i]).sum(axis=1)) for i in valid[test_start:]])

print("fitting M5 on train ...", flush=True)
mask,_ = vp.build_network_mask(A[:n_train])
fitted = vp.fit_model(5, X[:n_train], Y[:n_train], mask, n_jobs=4)
M5_pred = vp.predict_model(X[n_train:], fitted)
Y_va_te = Y[n_train:]
print("M5 val+test predictions:", M5_pred.shape, flush=True)

def net_sharpe(W, R, cost):
    T = len(W)-1
    gross = np.array([W[t] @ R[t+1] for t in range(T)])
    to = np.array([np.abs(W[t+1]-W[t]*(1+R[t+1])/(1+W[t]@R[t+1])).sum() for t in range(T)])
    net = gross - cost*to
    sd = net.std(ddof=1)
    return (net.mean()/sd*np.sqrt(252) if sd>0 else 0.0), to.mean()

etas = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3]
costs_bp = np.arange(0.0, 20.01, 0.5)
sel_rows = []
weights_by_eta = {}
for eta in etas:
    t0=time.time()
    W = vp.compute_model6_drift_l1(M5_pred, Y_va_te, valid[:n_train],
                                   eta=eta, rho=1e-3, risk_mult=1.0,
                                   rolling_cov=True, cov_window=150)
    Wv, Wt = W[:n_val], W[n_val:]
    weights_by_eta[eta] = (Wv, Wt)
    print(f"eta={eta:g} solved ({time.time()-t0:.1f}s)", flush=True)
    for c in costs_bp:
        s_val,_ = net_sharpe(Wv, R_val, c/10000.0)
        sel_rows.append({"eta":eta,"cost_bp":c,"val_sharpe":s_val})

sel = pd.DataFrame(sel_rows)
best_eta = sel.loc[sel.groupby("cost_bp")["val_sharpe"].idxmax()][["cost_bp","eta"]]
curves = {"fixed_eta_1e-6": [], "cost_aware": []}
for c in costs_bp:
    _, Wt_fix = weights_by_eta[1e-6]
    s_fix,_ = net_sharpe(Wt_fix, R_test, c/10000.0)
    eta_best = float(best_eta.loc[best_eta.cost_bp==c,"eta"].iloc[0])
    _, Wt_best = weights_by_eta[eta_best]
    s_best,_ = net_sharpe(Wt_best, R_test, c/10000.0)
    curves["fixed_eta_1e-6"].append(s_fix); curves["cost_aware"].append(s_best)

plt.rcParams.update({"font.size":11,"savefig.dpi":300,"figure.constrained_layout.use":True})
fig,ax=plt.subplots(figsize=(9.2,4.6))
ax.plot(costs_bp,curves["fixed_eta_1e-6"],lw=2.2,label="Post-hoc DFL, fixed eta=1e-6")
ax.plot(costs_bp,curves["cost_aware"],lw=2.2,ls="--",label="Post-hoc DFL, cost-aware eta (validation-selected)")
ax.axhline(0,color="grey",lw=0.8)
for x in (1,5,10,20):
    ax.axvline(x,color="grey",ls=":",lw=0.8)
ax.set_xlabel("Transaction cost (bp per unit of turnover)")
ax.set_ylabel("Net Sharpe ratio (test block)")
ax.set_title("Post-hoc DFL: fixed vs cost-aware turnover penalty")
ax.grid(alpha=0.25); ax.legend(fontsize=9)
for ext in ("png","pdf"):
    fig.savefig(OUT/f"fig11_dfl_cost_aware.{ext}",bbox_inches="tight")
plt.close(fig)
print("saved:",OUT/"fig11_dfl_cost_aware.png")
print("\nselected eta by cost:")
print(best_eta[best_eta.cost_bp.isin([1,5,10,20])].to_string(index=False))
print("\nSharpe at key costs:")
for c in (1,5,10,20):
    i=int(np.where(costs_bp==c)[0][0])
    print(c,"bp: fixed",round(curves["fixed_eta_1e-6"][i],3),"cost-aware",round(curves["cost_aware"][i],3))
pd.DataFrame({"cost_bp":costs_bp,**curves}).to_csv(PROJ/"性能评估与可视化"/"DFL_cost_aware_curve.csv",index=False)
