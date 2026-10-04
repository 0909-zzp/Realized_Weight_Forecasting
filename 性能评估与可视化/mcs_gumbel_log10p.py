"""Per-model approximate MCS p-values via Gumbel tail extrapolation (log10 scale).

Corrected version:
  * HLN (2011) elimination rule: remove argmax_i mean relative loss
    (largest average loss), not argmax_i row-max t.
  * MCS p-value = cumulative max of round p-values over the rounds the
    model survived (survivors get p = 1, i.e. log10 p = 0).
  * Early stopping at alpha (never binds for this data).
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"性能评估与可视化"; VARX=PROJ/"VARX"
valid=np.load(PROJ/"特征工程"/"valid_indices.npy"); n=len(np.load(PROJ/"特征工程"/"X_features.npy"))
n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
Y=np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:][65:265]
names=["VAR","Sparse VAR","Sparse VARX","Network VARX","Network VARX-S","Network VARX-S-DF","LSTM"]
preds={"VAR":np.load(VARX/"Y_pred_model1.npy")[-363:][65:265],
       "Sparse VAR":np.load(VARX/"Y_pred_model2.npy")[-363:][65:265],
       "Sparse VARX":np.load(VARX/"Y_pred_model3.npy")[-363:][65:265],
       "Network VARX":np.load(VARX/"Y_pred_model4.npy")[-363:][65:265],
       "Network VARX-S":np.load(VARX/"Y_pred_model5.npy")[-363:][65:265],
       "Network VARX-S-DF":np.load(VARX/"Y_pred_model6_opt.npy")[-363:][65:265],
       "LSTM":np.load(VARX/"Y_pred_model7.npy")[-363:][65:265]}
L_sq=np.column_stack([((preds[m]-Y)**2).mean(axis=1) for m in names])
L_ab=np.column_stack([np.abs(preds[m]-Y).mean(axis=1) for m in names])

def block_bootstrap(T,B,block_len,seed=42):
    rng=np.random.default_rng(seed); idx=np.zeros((B,T),dtype=np.int64); nb=int(np.ceil(T/block_len))
    for b in range(B):
        s=[]
        for _ in range(nb):
            st=int(rng.integers(0,T-block_len+1)); s.extend(range(st,st+block_len))
        idx[b]=np.array(s[:T])
    return idx

def mcs_evt_log10(L,names,B=10000,block_len=5,alpha=0.10):
    T,M=L.shape; idx=block_bootstrap(T,B,block_len); surviving=list(range(M)); rows=[]; r=0
    running_max_log10p=-np.inf
    while len(surviving)>1:
        r+=1; m=len(surviving); Ls=L[:,surviving]
        # HLN relative losses: dbar_i = mean loss relative to current-set average
        dbar=Ls.mean(axis=0)-Ls.mean()
        d_mean=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i!=j: d_mean[i,j]=np.mean(Ls[:,i]-Ls[:,j])
        d_boot=np.zeros((B,m,m))
        for b in range(B):
            lm=Ls[idx[b],:].mean(axis=0); d_boot[b]=lm[:,None]-lm[None,:]
        d_var=np.mean((d_boot-d_mean[None,:,:])**2,axis=0); d_var=np.maximum(d_var,1e-20); d_std=np.sqrt(d_var)
        t_stat=d_mean/d_std; t_boot=(d_boot-d_mean[None,:,:])/d_std
        T_R=float(np.max(np.abs(t_stat))); T_R_boot=np.abs(t_boot).max(axis=(1,2))
        # Gumbel fit to the max statistic distribution
        loc,scale=sp_stats.gumbel_r.fit(T_R_boot)
        log10p=float(sp_stats.gumbel_r.logsf(T_R,loc=loc,scale=scale)/np.log(10))
        # HLN elimination rule: worst = largest average relative loss
        worst=int(np.argmax(dbar)); wg=surviving[worst]
        # HLN MCS p-value: cumulative max over rounds the model survived
        running_max_log10p=max(running_max_log10p,log10p)
        rows.append({"model":names[wg],"round":r,"T_R_obs":T_R,
                     "gumbel_loc":loc,"gumbel_scale":scale,
                     "log10_p":running_max_log10p,"p":float(10**running_max_log10p)})
        surviving.pop(worst)
        # early stop if the joint null cannot be rejected (does not bind here)
        if float(10**log10p) > alpha:
            break
    for s in surviving:
        rows.append({"model":names[s],"round":r+1,"T_R_obs":np.nan,
                     "gumbel_loc":np.nan,"gumbel_scale":np.nan,
                     "log10_p":0.0,"p":1.0})
    return pd.DataFrame(rows)

print("B=10000, squared loss ...",flush=True)
sq=mcs_evt_log10(L_sq,names); sq["loss"]="squared"
print("B=10000, absolute loss ...",flush=True)
ab=mcs_evt_log10(L_ab,names); ab["loss"]="absolute"
df=pd.concat([sq,ab]); df.to_csv(OUT/"MCS_Gumbel_log10p.csv",index=False)
for lab,d in [("squared",sq),("absolute",ab)]:
    print(f"\n=== {lab} loss, approximate log10 p (HLN elimination) ===")
    print(d[["model","round","T_R_obs","gumbel_scale","log10_p"]].to_string(index=False))
