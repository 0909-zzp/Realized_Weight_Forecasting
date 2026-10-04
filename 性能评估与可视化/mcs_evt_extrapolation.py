"""MCS p-value tail extrapolation via generalized Pareto distribution.

For each elimination round, fit a GPD to the upper tail of the bootstrap
max-statistic distribution and extrapolate the tail probability at the observed
statistic. Empirical p = #{T_R^b > T_R}/B; GPD p is an approximate non-zero p.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"性能评估与可视化"
VARX=PROJ/"VARX"

# losses (same as Table 4)
valid=np.load(PROJ/"特征工程"/"valid_indices.npy")
n=len(np.load(PROJ/"特征工程"/"X_features.npy")); n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
Y=np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:][65:265]
names=["VAR","Sparse VAR","Sparse VARX","Network VARX","Network VARX-S","Network VARX-S-DF","LSTM"]
preds={ "VAR":np.load(VARX/"Y_pred_model1.npy")[-363:][65:265],
        "Sparse VAR":np.load(VARX/"Y_pred_model2.npy")[-363:][65:265],
        "Sparse VARX":np.load(VARX/"Y_pred_model3.npy")[-363:][65:265],
        "Network VARX":np.load(VARX/"Y_pred_model4.npy")[-363:][65:265],
        "Network VARX-S":np.load(VARX/"Y_pred_model5.npy")[-363:][65:265],
        "Network VARX-S-DF":np.load(VARX/"Y_pred_model6_opt.npy")[-363:][65:265],
        "LSTM":np.load(VARX/"Y_pred_model7.npy")[-363:][65:265]}
L_sq=np.column_stack([((preds[m]-Y)**2).mean(axis=1) for m in names])
L_ab=np.column_stack([np.abs(preds[m]-Y).mean(axis=1) for m in names])

def block_bootstrap(T,B,block_len,seed=42):
    rng=np.random.default_rng(seed); idx=np.zeros((B,T),dtype=np.int64); nblk=int(np.ceil(T/block_len))
    for b in range(B):
        s=[]; 
        for _ in range(nblk):
            st=int(rng.integers(0,T-block_len+1)); s.extend(range(st,st+block_len))
        idx[b]=np.array(s[:T])
    return idx

def gpd_tail_p(T_R, T_R_boot):
    B=len(T_R_boot)
    emp=float(np.mean(T_R_boot>T_R))
    # threshold at 95th percentile
    u=float(np.quantile(T_R_boot,0.95))
    excess=T_R_boot[T_R_boot>u]-u
    Fu=float(np.mean(T_R_boot<=u))  # P(X<=u)
    # fit GPD to excesses with location fixed at 0
    try:
        c, loc, scale = sp_stats.genpareto.fit(excess, floc=0.0)
        # ensure finite support: if c<0, upper bound = -scale/c
        surv = 1 - sp_stats.genpareto.cdf(T_R-u, c, loc=0.0, scale=scale)
        p_gpd = float((1-Fu)*max(surv,1e-300))
    except Exception as e:
        # exponential fallback (c=0): scale = mean excess
        scale=float(excess.mean()) if len(excess)>0 else 1e-6
        surv=np.exp(-(T_R-u)/scale) if T_R>u else 1.0
        p_gpd=float((1-Fu)*surv)
    return emp, p_gpd, u, float(np.max(T_R_boot)), len(excess)

def mcs_evt(L,names,n_boot=10000,block_len=5,seed=42):
    T,M=L.shape
    idx=block_bootstrap(T,n_boot,block_len,seed)
    surviving=list(range(M)); rows=[]
    rnum=0
    while len(surviving)>1:
        rnum+=1; m=len(surviving); Ls=L[:,surviving]
        d_mean=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i!=j: d_mean[i,j]=np.mean(Ls[:,i]-Ls[:,j])
        d_boot=np.zeros((n_boot,m,m))
        for b in range(n_boot):
            lm=Ls[idx[b],:].mean(axis=0)
            d_boot[b]=lm[:,None]-lm[None,:]
        d_var=np.mean((d_boot-d_mean[None,:,:])**2,axis=0); d_var=np.maximum(d_var,1e-20); d_std=np.sqrt(d_var)
        t_stat=d_mean/d_std; t_boot=(d_boot-d_mean[None,:,:])/d_std
        T_R=float(np.max(np.abs(t_stat))); T_R_boot=np.abs(t_boot).max(axis=(1,2))
        emp,p_gpd,u,maxb,nexc=gpd_tail_p(T_R,T_R_boot)
        t_row=np.max(t_stat,axis=1); worst=int(np.argmax(t_row)); worst_global=surviving[worst]
        rows.append({"loss":"","model":names[worst_global],"round":rnum,"T_R_obs":T_R,
                     "empirical_p":emp,"GPD_p":p_gpd,"threshold_u":u,"bootstrap_max":maxb,"tail_excess_n":nexc})
        surviving.pop(worst)
    # survivor
    rows.append({"loss":"","model":names[surviving[0]],"round":rnum+1,"T_R_obs":np.nan,
                 "empirical_p":1.0,"GPD_p":1.0,"threshold_u":np.nan,"bootstrap_max":np.nan,"tail_excess_n":0})
    return pd.DataFrame(rows)

print("B=10000, squared loss ...",flush=True)
df_sq=mcs_evt(L_sq,names,n_boot=10000); df_sq["loss"]="squared"
print("B=10000, absolute loss ...",flush=True)
df_ab=mcs_evt(L_ab,names,n_boot=10000); df_ab["loss"]="absolute"
df=pd.concat([df_sq,df_ab]); df.to_csv(OUT/"MCS_EVT_extrapolation.csv",index=False)
for lab,d in [("squared",df_sq),("absolute",df_ab)]:
    print(f"\n=== {lab} loss ===")
    print(d[["model","round","T_R_obs","empirical_p","GPD_p","bootstrap_max"]].to_string(index=False))
