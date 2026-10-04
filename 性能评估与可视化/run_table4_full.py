"""Complete Table 4: Panel A (MSE/MAE/MCS), Panels B/C (pairwise DM)."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"性能评估与可视化"
VARX=PROJ/"VARX"

# --- load predictions aligned to the 200-day window ---
valid=np.load(PROJ/"特征工程"/"valid_indices.npy")
n=len(np.load(PROJ/"特征工程"/"X_features.npy")); n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
Y=np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:][65:265]
preds={
 "VAR": np.load(VARX/"Y_pred_model1.npy")[-363:][65:265],
 "Sparse VAR": np.load(VARX/"Y_pred_model2.npy")[-363:][65:265],
 "Sparse VARX": np.load(VARX/"Y_pred_model3.npy")[-363:][65:265],
 "Network VARX": np.load(VARX/"Y_pred_model4.npy")[-363:][65:265],
 "Network VARX-S": np.load(VARX/"Y_pred_model5.npy")[-363:][65:265],
 "Network VARX-S-DF": np.load(VARX/"Y_pred_model6_opt.npy")[-363:][65:265],
 "LSTM": np.load(VARX/"Y_pred_model7.npy")[-363:][65:265],
}
names=list(preds.keys())
K=Y.shape[1]
print("T =",Y.shape[0],"K =",K)

# daily per-asset-average losses
L_sq=np.column_stack([((preds[m]-Y)**2).mean(axis=1) for m in names])
L_ab=np.column_stack([np.abs(preds[m]-Y).mean(axis=1) for m in names])

# Panel A: average losses
panelA=[]
for i,m in enumerate(names):
    panelA.append({"model":m,"MSE_w":float(L_sq[:,i].mean()),"MAE_w":float(L_ab[:,i].mean())})

# --- MCS (squared and absolute) ---
spec_mcs=importlib.util.spec_from_file_location("mcs",str(OUT/"MCS_完整程序.py")); mcs=importlib.util.module_from_spec(spec_mcs); sys.modules["mcs"]=mcs; spec_mcs.loader.exec_module(mcs)
mcs.MODEL_NAMES={i+1:names[i] for i in range(len(names))}
print("\n--- MCS squared loss ---",flush=True)
mcs_sq=mcs.mcs_procedure(L_sq, n_boot=mcs.N_BOOTSTRAP, block_len=mcs.BLOCK_LEN, seed=mcs.SEED)
print("\n--- MCS absolute loss ---",flush=True)
mcs_ab=mcs.mcs_procedure(L_ab, n_boot=mcs.N_BOOTSTRAP, block_len=mcs.BLOCK_LEN, seed=mcs.SEED)
for i,m in enumerate(names):
    panelA[i]["MCS_p_MSE"]=float(mcs_sq.loc[i,"MCS_pval"])
    panelA[i]["MCS_in75_MSE"]=bool(mcs_sq.loc[i,"in_MCS_75"])
    panelA[i]["MCS_in90_MSE"]=bool(mcs_sq.loc[i,"in_MCS_90"])
    panelA[i]["MCS_rank_MSE"]=int(mcs_sq.loc[i,"MCS_rank"])
    panelA[i]["MCS_p_MAE"]=float(mcs_ab.loc[i,"MCS_pval"])
    panelA[i]["MCS_in75_MAE"]=bool(mcs_ab.loc[i,"in_MCS_75"])
    panelA[i]["MCS_in90_MAE"]=bool(mcs_ab.loc[i,"in_MCS_90"])
    panelA[i]["MCS_rank_MAE"]=int(mcs_ab.loc[i,"MCS_rank"])
dfA=pd.DataFrame(panelA); dfA.to_csv(OUT/"Table4_PanelA.csv",index=False)

# --- pairwise DM (same HAC as the paper's dm_test) ---
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
    M=len(names); mat=[[None]*M for _ in range(M)]
    for i in range(M):
        for j in range(i):
            t,p,l10=dm_pair(preds[names[i]],preds[names[j]],loss)
            mat[i][j]={"t":t,"p":p,"log10p":l10}
    return mat

dm_sq=dm_matrix("sq"); dm_ab=dm_matrix("ab")
def mat_to_df(mat):
    rows=[]
    for i in range(len(names)):
        for j in range(i):
            c=mat[i][j]; rows.append({"row":names[i],"col":names[j],"DM":c["t"],"p":c["p"],"log10p":c["log10p"]})
    return pd.DataFrame(rows)
dfB=mat_to_df(dm_sq); dfC=mat_to_df(dm_ab)
dfB.to_csv(OUT/"Table4_PanelB_DM_squared.csv",index=False)
dfC.to_csv(OUT/"Table4_PanelC_DM_absolute.csv",index=False)

print("\n=== Panel A ===")
print(dfA.to_string(index=False))
print("\n=== Panel B (squared loss, row minus column) ===")
print(dfB.to_string(index=False))
print("\n=== Panel C (absolute loss, row minus column) ===")
print(dfC.to_string(index=False))

# Panel A 的 LaTeX 由 run_table4_mcs_blocklen.py 负责(含块长敏感性与 HLN 成员标记);
# 此脚本只产出 Panel B/C, 不再写 Table4_LaTeX.tex, 以免覆盖成缺少方法说明的旧版式。
