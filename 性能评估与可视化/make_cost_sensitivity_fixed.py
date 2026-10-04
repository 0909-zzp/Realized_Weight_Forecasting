"""Cost sensitivity under fixed weights (no retraining): 200d and full 363d."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"论文"/"figures"; OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("t3",str(PROJ/"性能评估与可视化"/"Table3_投资组合表现.py")); t3=importlib.util.module_from_spec(spec); sys.modules["t3"]=t3; spec.loader.exec_module(t3)
models,test_files,n_days=t3.load_data()   # 200-day window
simple200=np.array([np.expm1(np.load(str(f)).sum(axis=1)) for f in test_files])

# full 363-day test block
valid=np.load(PROJ/"特征工程"/"valid_indices.npy"); n=len(np.load(PROJ/"特征工程"/"X_features.npy"))
n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
npy=sorted([f for f in (PROJ/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
full_files=[npy[i] for i in valid[test_start:]]
simple_full=np.array([np.expm1(np.load(f).sum(axis=1)) for f in full_files])
varx=PROJ/"VARX"

def series(Y,simple):
    T=len(Y)-1
    g=np.array([Y[t]@simple[t+1] for t in range(T)])
    to=np.array([np.abs(Y[t+1]-Y[t]*(1+simple[t+1])/(1+Y[t]@simple[t+1])).sum() for t in range(T)])
    return g,to

def sharpe(g,to,c):
    net=g-c*to; sd=net.std(ddof=1)
    return net.mean()/sd*np.sqrt(252) if sd>0 else 0.0

# 200d models from t3; full-test weights from npy
specs200={}
for mid,md in models.items():
    nm=md["name"]
    if nm in ("Sparse VARX","Network VARX","Network+Smooth","LSTM","Sample GMVP (W=20)") or "DFL" in nm:
        key="Network+Smooth + DFL" if "DFL" in nm else nm
        specs200[key]=md["Y_pred"]
specs200["Equal Weight"]=np.ones((n_days,392))/392

specsfull={}
for nm,fn in [("Sparse VARX","Y_pred_model3.npy"),("Network VARX","Y_pred_model4.npy"),
              ("Network+Smooth","Y_pred_model5.npy"),("LSTM","Y_pred_model7.npy"),
              ("Sample GMVP (W=20)","Y_pred_bench_sample_W20.npy"),
              ("Network+Smooth + DFL","Y_pred_model6_opt.npy")]:
    specsfull[nm]=np.load(varx/fn)[-363:]
specsfull["Equal Weight"]=np.ones((363,392))/392

costs=np.arange(0,20.001,0.5)
colors={"Network+Smooth + DFL":"tab:blue","Sparse VARX":"tab:green","Network+Smooth":"tab:orange",
        "Network VARX":"tab:red","Sample GMVP (W=20)":"tab:purple","LSTM":"tab:brown","Equal Weight":"black"}
rows=[]
for tag,specs,simple in [("200d",specs200,simple200),("full363",specsfull,simple_full)]:
    for name,Y in specs.items():
        g,to=series(Y,simple)
        for c in costs:
            rows.append({"window":tag,"model":name,"cost_bp":c,"sharpe":sharpe(g,to,c/10000.0),"turnover":to.mean()})
df=pd.DataFrame(rows); df.to_csv(PROJ/"性能评估与可视化"/"cost_sensitivity_fixed_weights.csv",index=False)

plt.rcParams.update({"font.size":11,"savefig.dpi":300,"figure.constrained_layout.use":True})
fig,axes=plt.subplots(1,2,figsize=(13.2,4.7),sharey=True)
for ax,tag,title in [(axes[0],"200d","200-day evaluation window"),(axes[1],"full363","Full 363-day test period")]:
    for name in specs200.keys():
        d=df[(df.window==tag)&(df.model==name)]
        ax.plot(d.cost_bp,d.sharpe,color=colors[name],lw=2.4 if "DFL" in name else 1.5,
                ls="--" if name=="Equal Weight" else "-",label=name)
    ax.axhline(0,color="grey",lw=0.8)
    for x in (1,5,10,20): ax.axvline(x,color="grey",ls=":",lw=0.7)
    ax.set_xlabel("Transaction cost (bp per unit of turnover)")
    ax.set_title(title); ax.grid(alpha=0.25)
axes[0].set_ylabel("Net Sharpe ratio")
axes[1].legend(fontsize=8,loc="upper right")
for ext in ("png","pdf"):
    fig.savefig(OUT/f"fig13_cost_sensitivity_fixed.{ext}",bbox_inches="tight")
plt.close(fig)
print("saved:",OUT/"fig13_cost_sensitivity_fixed.png")
for tag in ("200d","full363"):
    print(f"\n--- {tag} ---")
    piv=df[df.window==tag].pivot(index="cost_bp",columns="model",values="sharpe")
    print(piv.loc[[1,5,10,20]].round(3).to_string())
print("\nCSV:",PROJ/"性能评估与可视化"/"cost_sensitivity_fixed_weights.csv")
