"""Noise-sensitivity diagnostic for the MCS (squared loss).

Diagnostic only: add i.i.d. Gaussian noise to the realized-weight target and
re-run the standard MCS. Noise scale = lambda * std(realized weights).
This does NOT change the main results; it shows how much noise is required
before the model ranking becomes statistically indistinguishable.
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

sigma_w=float(Y.std())
levels=[0.0,0.1,0.25,0.5,1.0]
rows=[]
for lam in levels:
    if lam==0.0:
        Yn=Y.copy()
    else:
        rng=np.random.default_rng(1000+int(lam*100))
        Yn=Y + rng.normal(0, lam*sigma_w, size=Y.shape)
    L=np.column_stack([((preds[m]-Yn)**2).mean(axis=1) for m in names])
    df=mcs.mcs_procedure(L,n_boot=10000,block_len=5,seed=42)
    for i,m in enumerate(names):
        rows.append({"noise_lambda":lam,"model":m,"raw_p":float(df.loc[i,"MCS_raw_pval"]),
                     "MCS_rank":int(df.loc[i,"MCS_rank"]),"in_MCS_75":bool(df.loc[i,"in_MCS_75"]),
                     "in_MCS_90":bool(df.loc[i,"in_MCS_90"])})
    print(f"\n=== noise_lambda={lam} (sigma={lam*sigma_w:.5f}) ===")
    print(df[["Name","MCS_raw_pval","MCS_rank","in_MCS_75","in_MCS_90"]].to_string(index=False))
res=pd.DataFrame(rows); res.to_csv(OUT/"MCS_noise_sensitivity.csv",index=False)
# summary of which models in 90% set at each noise level
summary=res.pivot_table(index="noise_lambda",columns="model",values="in_MCS_90",aggfunc="max").astype(int)
print("\n=== 90% MCS membership by noise level ===")
print(summary.to_string())
summary.to_csv(OUT/"MCS_noise_sensitivity_summary.csv")
