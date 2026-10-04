"""Single-panel cost sensitivity for the 200-day window (fixed weights)."""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"论文"/"figures"; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(PROJ/"性能评估与可视化"/"cost_sensitivity_fixed_weights.csv")
d=df[df.window=="200d"]
colors={"Network+Smooth + DFL":"tab:blue","Sparse VARX":"tab:green","Network+Smooth":"tab:orange",
        "Network VARX":"tab:red","Sample GMVP (W=20)":"tab:purple","LSTM":"tab:brown","Equal Weight":"black"}
order=["Network+Smooth + DFL","Sparse VARX","Network+Smooth","Network VARX","Sample GMVP (W=20)","LSTM","Equal Weight"]
plt.rcParams.update({"font.size":11,"savefig.dpi":300,"figure.constrained_layout.use":True})
fig,ax=plt.subplots(figsize=(9.6,4.9))
for name in order:
    g=d[d.model==name]
    ax.plot(g.cost_bp,g.sharpe,color=colors[name],lw=2.7 if "DFL" in name else 1.5,
            ls="--" if name=="Equal Weight" else "-",label=name)
ax.axhline(0,color="grey",lw=0.8)
for x in (1,5,10,20):
    ax.axvline(x,color="grey",ls=":",lw=0.7)
    ax.text(x,ax.get_ylim()[1],f"{x}bp",ha="center",va="bottom",fontsize=8,color="grey")
ax.set_xlabel("Transaction cost (bp per unit of turnover)")
ax.set_ylabel("Net Sharpe ratio")
ax.set_title("Net Sharpe vs transaction cost (200-day evaluation window)")
ax.grid(alpha=0.25); ax.legend(ncol=2,fontsize=9,loc="upper right")
for ext in ("png","pdf"):
    fig.savefig(OUT/f"fig14_cost_sensitivity_200d.{ext}",bbox_inches="tight")
plt.close(fig)
print("saved:",OUT/"fig14_cost_sensitivity_200d.png")
piv=d.pivot(index="cost_bp",columns="model",values="sharpe")
print("\n", piv.loc[[1,5,10,20],order].round(3).to_string())
