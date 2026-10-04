"""Cost-sensitivity curve for the main Table 3 models (200-day window)."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = PROJ / "论文" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("t3", str(PROJ/"性能评估与可视化"/"Table3_投资组合表现.py"))
t3 = importlib.util.module_from_spec(spec); sys.modules["t3"]=t3; spec.loader.exec_module(t3)
models, test_files, n_days = t3.load_data()
simple = np.array([np.expm1(np.load(str(f)).sum(axis=1)) for f in test_files])
K = simple.shape[1]

def series(Y):
    T = len(Y)-1
    gross = np.array([Y[t] @ simple[t+1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = Y[t]*(1+simple[t+1])/(1+Y[t]@simple[t+1])
        to[t] = np.abs(Y[t+1]-wp).sum()
    return gross, to

def sharpe(gross, to, cost):
    net = gross - cost*to
    sd = net.std(ddof=1)
    return net.mean()/sd*np.sqrt(252) if sd > 0 else 0.0

names = ["Sparse VARX", "Network+Smooth", "Network VARX",
         "Sample GMVP (W=20)", "LSTM"]
main = {}
for mid, md in models.items():
    nm = md["name"]
    if "DFL" in nm:
        main["Network+Smooth + DFL"] = series(md["Y_pred"])
    elif nm in names:
        main[nm] = series(md["Y_pred"])
main["Equal Weight"] = series(np.ones((n_days, K))/K)

costs_bp = np.arange(0.0, 20.01, 0.5)
colors = {"Network+Smooth + DFL":"tab:blue","Sparse VARX":"tab:green",
          "Network+Smooth":"tab:orange","Network VARX":"tab:red",
          "Sample GMVP (W=20)":"tab:purple","LSTM":"tab:brown","Equal Weight":"black"}
styles = {"Network+Smooth + DFL":"-","Equal Weight":"--"}
lw = {"Network+Smooth + DFL":2.6}

plt.rcParams.update({"font.size":11,"savefig.dpi":300,"figure.constrained_layout.use":True})
fig, ax = plt.subplots(figsize=(9.4,4.8))
for name,(g,to) in main.items():
    y=[sharpe(g,to,c/10000.0) for c in costs_bp]
    ax.plot(costs_bp,y,color=colors.get(name),ls=styles.get(name,"-"),
            lw=lw.get(name,1.6),label=name)
ymax=ax.get_ylim()[1]
for x in (1,5,10,20):
    ax.axvline(x,color="grey",ls=":",lw=0.8)
    ax.text(x,ymax,f"{x}bp",ha="center",va="bottom",fontsize=8,color="grey")
ax.axhline(0,color="grey",lw=0.8)
ax.set_xlabel("Transaction cost (bp per unit of turnover)")
ax.set_ylabel("Net Sharpe ratio")
ax.set_title("Net Sharpe vs transaction cost (200-day evaluation window)")
ax.grid(alpha=0.25)
ax.legend(ncol=2,fontsize=9,loc="upper right")
for ext in ("png","pdf"):
    fig.savefig(OUT/f"fig9_cost_sensitivity.{ext}",bbox_inches="tight")
plt.close(fig)
print("saved:",OUT/"fig9_cost_sensitivity.png")
for c in (1,5,10,20):
    print(c,"bp:",{name:round(sharpe(g,to,c/10000.0),3) for name,(g,to) in main.items()})
