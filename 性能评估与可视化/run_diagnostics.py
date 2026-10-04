"""Two diagnostic checks.

Check 1: refit VARX models M1-M5 on the training block and compare the
one-step-ahead predictions to the saved .npy files. Exact agreement confirms
that the saved predictions come from the declared no-look-ahead pipeline.

Check 2: run the squared-loss MCS under different block lengths and bootstrap
sizes to verify that membership and ranking do not flip.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, time, importlib.util
import numpy as np, pandas as pd
from pathlib import Path

PROJ=Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT=PROJ/"性能评估与可视化"
VARX=PROJ/"VARX"

spec=importlib.util.spec_from_file_location("vp",str(VARX/"VAR及拓展（table2）.py")); vp=importlib.util.module_from_spec(spec); sys.modules["vp"]=vp; spec.loader.exec_module(vp)

X=np.load(PROJ/"特征工程"/"X_features.npy"); Y=np.load(PROJ/"特征工程"/"Y_targets.npy"); A=np.load(PROJ/"特征工程"/"A_bar.npy")
n=len(X); n_train=int(n*.7); n_val=int(n*.15); test_start=n_train+n_val
mask,_=vp.build_network_mask(A[:n_train])

print("=== Check 1: refit M1-M5 and compare saved predictions ===")
for mid in range(1,6):
    t0=time.time()
    fitted=vp.fit_model(mid,X[:n_train],Y[:n_train],mask,n_jobs=4)
    pred=vp.predict_model(X[test_start:],fitted)
    saved=np.load(VARX/f"Y_pred_model{mid}.npy")[-363:]
    diff=np.abs(pred-saved).max()
    mse=float(((pred-Y[test_start:])**2).mean())
    print(f"M{mid}: max|pred-saved|={diff:.3e}  MSE={mse:.4e}  ({time.time()-t0:.1f}s)")

# M6/M7 not refit here: M6 is post-hoc DFL, M7 LSTM (heavy). Saved MSE matches Table 4.
print("\n=== Check 2: MCS sensitivity to block length and bootstrap size (squared loss) ===")
valid=np.load(PROJ/"特征工程"/"valid_indices.npy")
Yw=np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:][65:265]
names=["VAR","Sparse VAR","Sparse VARX","Network VARX","Network VARX-S","Network VARX-S-DF","LSTM"]
preds={"VAR":np.load(VARX/"Y_pred_model1.npy")[-363:][65:265],
       "Sparse VAR":np.load(VARX/"Y_pred_model2.npy")[-363:][65:265],
       "Sparse VARX":np.load(VARX/"Y_pred_model3.npy")[-363:][65:265],
       "Network VARX":np.load(VARX/"Y_pred_model4.npy")[-363:][65:265],
       "Network VARX-S":np.load(VARX/"Y_pred_model5.npy")[-363:][65:265],
       "Network VARX-S-DF":np.load(VARX/"Y_pred_model6_opt.npy")[-363:][65:265],
       "LSTM":np.load(VARX/"Y_pred_model7.npy")[-363:][65:265]}
L=np.column_stack([((preds[m]-Yw)**2).mean(axis=1) for m in names])

spec_m=importlib.util.spec_from_file_location("mcs",str(OUT/"MCS_完整程序.py")); mcs=importlib.util.module_from_spec(spec_m); sys.modules["mcs"]=mcs; spec_m.loader.exec_module(mcs)
mcs.MODEL_NAMES={i+1:names[i] for i in range(len(names))}
mcs.log=lambda msg: None  # suppress verbose logging

rows=[]
for block in (1,5,10):
    for B in (2000,5000,10000):
        df=mcs.mcs_procedure(L,n_boot=B,block_len=block,seed=42)
        df["block"]=block; df["B"]=B
        rows.append(df)
res=pd.concat(rows)
keep=res[["block","B","Name","MCS_raw_pval","Eliminated_round","MCS_rank","in_MCS_75","in_MCS_90"]]
keep.to_csv(OUT/"MCS_sensitivity_block_boot.csv",index=False)
print(keep.to_string(index=False))
print("\nmembership stability: models in 90% set by setting:")
for (b,B),g in keep.groupby(["block","B"]):
    print(f"  block={b}, B={B}: {g.loc[g.in_MCS_90,'Name'].tolist()}")
