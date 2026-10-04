import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"性能评估与可视化"
VARX=PROJ/"VARX"

valid=np.load(PROJ/"特征工程"/"valid_indices.npy")
n=len(np.load(PROJ/"特征工程"/"X_features.npy")); n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
Y=np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:][65:265]
preds={
 "VAR": np.load(VARX/"Y_pred_model1.npy")[-363:][65:265],
 "Sparse VAR": np.load(VARX/"Y_pred_model2.npy")[-363:][65:265],
 "Sparse VARX": np.load(VARX/"Y_pred_model3.npy")[-363:][65:265],
 "Network VARX": np.load(VARX/"Y_pred_model4_idio.npy")[-363:][65:265],
 "Network VARX-S": np.load(VARX/"Y_pred_model5_idio.npy")[-363:][65:265],
 "Network VARX-S-DF": np.load(VARX/"Y_pred_model6_opt_idio.npy")[-363:][65:265],
 "LSTM": np.load(VARX/"Y_pred_model7.npy")[-363:][65:265],
}
names=list(preds.keys())
K=Y.shape[1]
L_sq=np.column_stack([((preds[m]-Y)**2).mean(axis=1) for m in names])
L_ab=np.column_stack([np.abs(preds[m]-Y).mean(axis=1) for m in names])

panelA=[]
for i,m in enumerate(names):
    panelA.append({"model":m,"MSE_w":float(L_sq[:,i].mean()),"MAE_w":float(L_ab[:,i].mean())})

spec_mcs=importlib.util.spec_from_file_location("mcs",str(OUT/"MCS_完整程序.py")); mcs=importlib.util.module_from_spec(spec_mcs); sys.modules["mcs"]=mcs; spec_mcs.loader.exec_module(mcs)
mcs.MODEL_NAMES={i+1:names[i] for i in range(len(names))}
print("\n--- MCS squared ---")
mcs_sq=mcs.mcs_procedure(L_sq, n_boot=mcs.N_BOOTSTRAP, block_len=mcs.BLOCK_LEN, seed=mcs.SEED)
print("\n--- MCS absolute ---")
mcs_ab=mcs.mcs_procedure(L_ab, n_boot=mcs.N_BOOTSTRAP, block_len=mcs.BLOCK_LEN, seed=mcs.SEED)
for i,m in enumerate(names):
    panelA[i]["MCS_p_MSE"]=float(mcs_sq.loc[i,"MCS_pval"])
    panelA[i]["MCS_p_MAE"]=float(mcs_ab.loc[i,"MCS_pval"])
dfA=pd.DataFrame(panelA)
print("\n=== Panel A (idio) ===")
print(dfA[['model','MSE_w','MAE_w','MCS_p_MSE','MCS_p_MAE']].to_string(index=False))

def dm_pair(a,b,loss):
    la = ((a-Y)**2).mean(axis=1) if loss=="sq" else np.abs(a-Y).mean(axis=1)
    lb = ((b-Y)**2).mean(axis=1) if loss=="sq" else np.abs(b-Y).mean(axis=1)
    d=la-lb; T=len(d)
    if T<2 or d.var(ddof=1)<1e-15: return 0.0,1.0,np.nan
    max_lag=int(np.floor(4*(T/100)**(2/9)))
    dd=d-d.mean(); nw=np.dot(dd,dd)/T
    for lag in range(1,max_lag+1):
        w=1-lag/(max_lag+1); nw+=2*w*np.dot(dd[lag:],dd[:-lag])/T
    nw=nw/T if nw>1e-15 else d.var(ddof=1)/T
    t=d.mean()/np.sqrt(nw)
    logsf=sp_stats.norm.logsf(abs(t)); log10p=(np.log(2)+float(logsf))/np.log(10)
    return float(t),float(2*(1-sp_stats.norm.cdf(abs(t)))),float(log10p)

def dm_matrix(loss):
    M=len(names); rows=[]
    for i in range(M):
        for j in range(i):
            t,p,l10=dm_pair(preds[names[i]],preds[names[j]],loss)
            rows.append((names[i],names[j],t,p,l10))
    return rows

for loss,label in [("sq","Panel B squared"),("ab","Panel C absolute")]:
    rows=dm_matrix(loss)
    print(f"\n=== {label} (idio, row minus column) ===")
    for r in rows:
        print(f"  {r[0]:18s} vs {r[1]:18s} DM={r[2]:+.3f}  p={r[3]:.3e}  log10p={r[4]:.1f}")
