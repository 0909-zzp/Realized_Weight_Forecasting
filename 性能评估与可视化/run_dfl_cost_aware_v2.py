"""Cost-aware post-hoc DFL v2: save weights and evaluate validation / 200d / full test."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, time, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"论文"/"figures"; OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("vp",str(PROJ/"VARX"/"VAR及拓展（table2）.py")); vp=importlib.util.module_from_spec(spec); sys.modules["vp"]=vp; spec.loader.exec_module(vp)

X=np.load(PROJ/"特征工程"/"X_features.npy"); Y=np.load(PROJ/"特征工程"/"Y_targets.npy")
A=np.load(PROJ/"特征工程"/"A_bar.npy"); valid=np.load(PROJ/"特征工程"/"valid_indices.npy")
n=len(X); n_train=int(n*0.70); n_val=int(n*0.15); test_start=n_train+n_val
npy=sorted([f for f in (PROJ/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
R_val=np.array([np.expm1(np.load(npy[i]).sum(axis=1)) for i in valid[n_train:test_start]])
R_test=np.array([np.expm1(np.load(npy[i]).sum(axis=1)) for i in valid[test_start:]])
Yte_full=Y[test_start:]

print("fit M5 ...",flush=True)
mask,_=vp.build_network_mask(A[:n_train]); fitted=vp.fit_model(5,X[:n_train],Y[:n_train],mask,n_jobs=4)
M5_pred=vp.predict_model(X[n_train:],fitted); Y_va_te=Y[n_train:]

def stats(W,R,cost):
    T=len(W)-1
    g=np.array([W[t]@R[t+1] for t in range(T)])
    to=np.array([np.abs(W[t+1]-W[t]*(1+R[t+1])/(1+W[t]@R[t+1])).sum() for t in range(T)])
    net=g-cost*to; sd=net.std(ddof=1); cum=np.cumprod(1+net)
    return dict(sharpe=net.mean()/sd*np.sqrt(252) if sd>0 else 0.0,turnover=to.mean(),
                vol=sd*np.sqrt(252),maxDD=(cum/np.maximum.accumulate(cum)-1).min(),cum=cum[-1]-1)

etas=[1e-6,1e-5,1e-4,5e-4]; Ws={}
for eta in etas:
    t0=time.time()
    W=vp.compute_model6_drift_l1(M5_pred,Y_va_te,valid[:n_train],eta=eta,rho=1e-3,
                                 risk_mult=1.0,rolling_cov=True,cov_window=150)
    Ws[eta]=(W[:n_val],W[n_val:])
    print(f"eta={eta:g} done {time.time()-t0:.1f}s",flush=True)
np.savez(PROJ/"性能评估与可视化"/"dfl_cost_aware_weights.npz",
         **{f"eta_{e}":Ws[e][1] for e in etas})

costs_bp=np.arange(0.0,20.01,0.5)
rows=[]
for eta in etas:
    Wv,Wt=Ws[eta]
    wt200=Wt[65:265]
    for c in (1,5,10,20):
        sv=stats(Wv,R_val,c/10000.0)
        s200=stats(wt200,R_test[65:265],c/10000.0)
        sfull=stats(Wt,R_test,c/10000.0)
        rows.append({"eta":eta,"cost_bp":c,"val_sharpe":sv["sharpe"],"test200_sharpe":s200["sharpe"],
                     "full_sharpe":sfull["sharpe"],"turnover200":s200["turnover"],
                     "gross200":float(np.abs(wt200).sum(1).mean()),"MSE200":float(((wt200-Yte_full[65:265])**2).mean())})
df=pd.DataFrame(rows); df.to_csv(PROJ/"性能评估与可视化"/"DFL_cost_aware_by_eta.csv",index=False)
print("\n=== by eta / cost ===")
print(df.round(4).to_string(index=False))

# cost-aware selection: maximize validation Sharpe per cost
sel=df.loc[df.groupby("cost_bp")["val_sharpe"].idxmax()][["cost_bp","eta","test200_sharpe","full_sharpe","turnover200"]]
print("\n=== selected by validation Sharpe ==="); print(sel.round(4).to_string(index=False))
# fixed eta=1e-6
fix=df[df.eta==1e-6][["cost_bp","test200_sharpe","full_sharpe","turnover200"]]
print("\n=== fixed eta=1e-6 ==="); print(fix.round(4).to_string(index=False))

# figure: 200d test Sharpe, fixed vs cost-aware
fig,ax=plt.subplots(figsize=(9.2,4.6))
ax.plot(df[df.eta==1e-6].cost_bp,df[df.eta==1e-6].test200_sharpe,lw=2.2,label="fixed eta=1e-6 (200d)")
# cost-aware curve on a fine grid: for each cost, choose eta maximizing validation Sharpe
fine=np.arange(0,20.01,0.5); ca=[]
for c in fine:
    cand=[]
    for eta in etas:
        Wv,Wt=Ws[eta]; s=stats(Wv,R_val,c/10000.0)["sharpe"]; cand.append((s,eta))
    eta=max(cand)[1]; ca.append(stats(Ws[eta][1][65:265],R_test[65:265],c/10000.0)["sharpe"])
ax.plot(fine,ca,lw=2.2,ls="--",label="cost-aware eta (validation-selected, 200d)")
ax.axhline(0,color="grey",lw=0.8)
for x in (1,5,10,20): ax.axvline(x,color="grey",ls=":",lw=0.8)
ax.set_xlabel("Transaction cost (bp per unit of turnover)"); ax.set_ylabel("Net Sharpe ratio (200d window)")
ax.set_title("Post-hoc DFL: fixed vs cost-aware turnover penalty"); ax.grid(alpha=0.25); ax.legend(fontsize=9)
for ext in ("png","pdf"): fig.savefig(OUT/f"fig12_dfl_cost_aware_200d.{ext}",bbox_inches="tight")
plt.close(fig); print("\nsaved:",OUT/"fig12_dfl_cost_aware_200d.png")
