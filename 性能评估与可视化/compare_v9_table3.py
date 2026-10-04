"""Compare v9 balanced end-to-end model with the paper's Table 3 models."""
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
OUT_FIG=PROJ/"论文"/"figures"; OUT_FIG.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("t3",str(PROJ/"性能评估与可视化"/"Table3_投资组合表现.py"))
t3=importlib.util.module_from_spec(spec); sys.modules["t3"]=t3; spec.loader.exec_module(t3)
models,test_files,n_days=t3.load_data()
simple=np.array([np.expm1(np.load(str(f)).sum(axis=1)) for f in test_files]); K=simple.shape[1]
Yte=np.load(PROJ/"特征工程"/"Y_targets.npy")[-363:][65:265]

def series(Y):
    T=len(Y)-1
    gross=np.array([Y[t]@simple[t+1] for t in range(T)])
    to=np.zeros(T)
    for t in range(T):
        wp=Y[t]*(1+simple[t+1])/(1+Y[t]@simple[t+1]); to[t]=np.abs(Y[t+1]-wp).sum()
    return gross,to

def metrics(g,to,cost):
    net=g-cost*to; cum=np.cumprod(1+net); sd=net.std(ddof=1)
    return dict(sharpe=net.mean()/sd*np.sqrt(252) if sd>0 else 0.0,vol=sd*np.sqrt(252),
                turnover=to.mean(),maxDD=(cum/np.maximum.accumulate(cum)-1).min(),cum=cum[-1]-1)

# assemble model weights
sel={}
for mid,md in models.items():
    nm=md["name"]
    if nm in ("Sparse VARX","Network VARX","Network+Smooth","LSTM","Sample GMVP (W=20)") or "DFL" in nm:
        key="Network+Smooth + DFL" if "DFL" in nm else nm
        sel[key]=md["Y_pred"]
sel["Equal Weight"]=np.ones((n_days,K))/K
sel["v9 balanced (end-to-end)"]=np.load(PROJ/"endtoend_project"/"e2e_v9_balanced_test_weights.npy")[65:265]

rows=[]
for name,Y in sel.items():
    g,to=series(Y)
    for c,lab in [(1e-4,"1bp"),(5e-4,"5bp"),(1e-3,"10bp"),(2e-3,"20bp")]:
        rows.append({"model":name,"cost":lab,**metrics(g,to,c),"MSE":float(((Y-Yte)**2).mean()),"gross":float(np.abs(Y).sum(1).mean())})
df=pd.DataFrame(rows)
print("=== 1bp full comparison ===")
tab=df[df.cost=="1bp"][["model","sharpe","turnover","vol","maxDD","cum","MSE","gross"]].copy()
print(tab.round(4).to_string(index=False))
print("\n=== net Sharpe by cost ===")
piv=df.pivot(index="model",columns="cost",values="sharpe")
print(piv.round(3).to_string())
df.to_csv(PROJ/"endtoend_project"/"Table3_vs_v9_cost_sensitivity.csv",index=False)

# figure
colors={"Network+Smooth + DFL":"tab:blue","v9 balanced (end-to-end)":"tab:cyan",
        "Sparse VARX":"tab:green","Network+Smooth":"tab:orange","Network VARX":"tab:red",
        "Sample GMVP (W=20)":"tab:purple","LSTM":"tab:brown","Equal Weight":"black"}
costs=np.arange(0,20.01,0.5)
plt.rcParams.update({"font.size":11,"savefig.dpi":300,"figure.constrained_layout.use":True})
fig,ax=plt.subplots(figsize=(9.6,4.9))
for name,Y in sel.items():
    g,to=series(Y)
    y=[metrics(g,to,c/10000.0)["sharpe"] for c in costs]
    ax.plot(costs,y,color=colors[name],lw=2.6 if "v9" in name or "DFL" in name else 1.5,
            ls="--" if name=="Equal Weight" else "-",label=name)
ax.axhline(0,color="grey",lw=0.8)
for x in (1,5,10,20):
    ax.axvline(x,color="grey",ls=":",lw=0.8); ax.text(x,ax.get_ylim()[1],f"{x}bp",ha="center",va="bottom",fontsize=8,color="grey")
ax.set_xlabel("Transaction cost (bp per unit of turnover)"); ax.set_ylabel("Net Sharpe ratio")
ax.set_title("Net Sharpe vs transaction cost: v9 vs Table 3 models")
ax.grid(alpha=0.25); ax.legend(ncol=2,fontsize=9,loc="upper right")
for ext in ("png","pdf"): fig.savefig(OUT_FIG/f"fig10_cost_sensitivity_v9.{ext}",bbox_inches="tight")
print("\nsaved figure:",OUT_FIG/"fig10_cost_sensitivity_v9.png")
