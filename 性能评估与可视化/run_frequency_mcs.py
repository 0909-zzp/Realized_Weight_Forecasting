"""Frequency-aggregated MCS (weekly / monthly)."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path

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

spec=importlib.util.spec_from_file_location("mcs",str(OUT/"MCS_完整程序.py")); mcs=importlib.util.module_from_spec(spec); sys.modules["mcs"]=mcs; spec.loader.exec_module(mcs)
mcs.MODEL_NAMES={i+1:names[i] for i in range(len(names))}; mcs.log=lambda msg: None

def aggregate(L, freq):
    T=L.shape[0]
    nblk=T//freq
    L=L[:nblk*freq].reshape(nblk,freq,-1)
    return L.mean(axis=1)   # (nblk, M)

rows=[]
for loss_name, L in [("squared", np.column_stack([((preds[m]-Y)**2).mean(axis=1) for m in names])),
                     ("absolute", np.column_stack([np.abs(preds[m]-Y).mean(axis=1) for m in names]))]:
    for freq, block in [(1,5),(5,5),(20,2)]:
        Lagg=aggregate(L,freq)
        T=Lagg.shape[0]
        df=mcs.mcs_procedure(Lagg,n_boot=10000,block_len=min(block,T-1),seed=42)
        for i,m in enumerate(names):
            rows.append({"loss":loss_name,"freq":freq,"T":T,"model":m,
                         "raw_p":float(df.loc[i,"MCS_raw_pval"]),
                         "MCS_rank":int(df.loc[i,"MCS_rank"]),
                         "in_MCS_90":bool(df.loc[i,"in_MCS_90"])})
        print(f"\n=== {loss_name}, freq={freq} (T={T}) ===")
        print(df[["Name","MCS_raw_pval","Eliminated_round","MCS_rank","in_MCS_75","in_MCS_90"]].to_string(index=False))
res=pd.DataFrame(rows); res.to_csv(OUT/"MCS_frequency_aggregation.csv",index=False)
