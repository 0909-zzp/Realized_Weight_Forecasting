"""Asset-subsample MCS robustness (squared loss).

For each draw of a random 50% or 70% asset subset, recompute daily squared
weight loss and run the standard MCS. Summarize membership frequency and rank.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"性能评估与可视化"; VARX=PROJ/"VARX"
valid=np.load(PROJ/"特征工程"/"valid_indices.npy"); n=len(np.load(PROJ/"特征工程"/"X_features.npy"))
n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
Y=np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:][65:265]  # (200,392)
names=["VAR","Sparse VAR","Sparse VARX","Network VARX","Network VARX-S","Network VARX-S-DF","LSTM"]
preds={"VAR":np.load(VARX/"Y_pred_model1.npy")[-363:][65:265],
       "Sparse VAR":np.load(VARX/"Y_pred_model2.npy")[-363:][65:265],
       "Sparse VARX":np.load(VARX/"Y_pred_model3.npy")[-363:][65:265],
       "Network VARX":np.load(VARX/"Y_pred_model4.npy")[-363:][65:265],
       "Network VARX-S":np.load(VARX/"Y_pred_model5.npy")[-363:][65:265],
       "Network VARX-S-DF":np.load(VARX/"Y_pred_model6_opt.npy")[-363:][65:265],
       "LSTM":np.load(VARX/"Y_pred_model7.npy")[-363:][65:265]}

spec=importlib.util.spec_from_file_location("mcs",str(OUT/"MCS_完整程序.py")); mcs=importlib.util.module_from_spec(spec); sys.modules["mcs"]=mcs; spec.loader.exec_module(mcs)
mcs.MODEL_NAMES={i+1:names[i] for i in range(len(names))}; mcs.log=lambda msg: None

def run_draws(frac, n_draws=100, B=2000):
    K=Y.shape[1]; size=int(frac*K); rows=[]
    for d in range(n_draws):
        rng=np.random.default_rng(1000+d)
        sel=rng.choice(K,size=size,replace=False)
        L=np.column_stack([((preds[m][:,sel]-Y[:,sel])**2).mean(axis=1) for m in names])
        df=mcs.mcs_procedure(L,n_boot=B,block_len=5,seed=42)
        for i,m in enumerate(names):
            rows.append({"frac":frac,"draw":d,"model":m,"in_MCS_90":bool(df.loc[i,"in_MCS_90"]),
                         "MCS_rank":int(df.loc[i,"MCS_rank"]),"raw_p":float(df.loc[i,"MCS_raw_pval"])})
        if (d+1)%20==0: print(f"frac={frac}: {d+1}/{n_draws}",flush=True)
    return pd.DataFrame(rows)

rows=pd.concat([run_draws(0.5,100), run_draws(0.7,100)])
rows.to_csv(OUT/"MCS_asset_subsample.csv",index=False)

summary=[]
for frac in (0.5,0.7):
    g=rows[rows.frac==frac]
    for m in names:
        gm=g[g.model==m]
        summary.append({"frac":frac,"model":m,
                        "in90_freq":gm.in_MCS_90.mean(),
                        "mean_rank":gm.MCS_rank.mean(),
                        "median_rank":gm.MCS_rank.median(),
                        "share_rank1":float((gm.MCS_rank==1).mean()),
                        "share_rawp_gt_010":float((gm.raw_p>0.10).mean())})
sdf=pd.DataFrame(summary); sdf.to_csv(OUT/"MCS_asset_subsample_summary.csv",index=False)
for frac in (0.5,0.7):
    print(f"\n=== asset subsample {int(frac*100)}%, 100 draws ===")
    print(sdf[sdf.frac==frac].round(4).to_string(index=False))
