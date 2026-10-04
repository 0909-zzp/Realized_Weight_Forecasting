import os as _os
_os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, time, importlib.util, numpy as np
from pathlib import Path
root = Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
sys.path.insert(0, str(root/'图形Lasso'/'code'))
spec = importlib.util.spec_from_file_location('vp', str(root/'VARX'/'VAR及拓展（table2）.py'))
vp = importlib.util.module_from_spec(spec); sys.modules['vp']=vp; spec.loader.exec_module(vp)

X = np.load(root/'特征工程'/'X_features.npy')
Y = np.load(root/'特征工程'/'Y_targets.npy')
A_idio = np.load(root/'特征工程'/'A_bar_idio.npy')
n=len(X); ntr=int(0.7*n); nval=int(0.15*n); ts=ntr+nval
Xtr,Ytr=X[:ntr],Y[:ntr]; Xte,Yte=X[ts:],Y[ts:]
THR=0.117
mask=(A_idio[:ntr].mean(0)>=THR).astype(float); np.fill_diagonal(mask,0)
valid_indices=np.load(root/'特征工程'/'valid_indices.npy')
train_day_indices=valid_indices[:ntr]

def dfl(Ybase):
    return vp.compute_model6_drift_l1(Ybase, Yte, train_day_indices, eta=1e-6, rho=1e-3, risk_mult=1.0, rolling_cov=True, cov_window=150)

# --- no exog: model5 with blocks=['lagged'], idio mask ---
old5=dict(vp.MODELS[5]); vp.MODELS[5]['blocks']=['lagged']
try:
    fit_noexog=vp.fit_model(5, Xtr, Ytr, mask, n_jobs=4)
    Y_noexog_base=vp.predict_model(Xte, fit_noexog)
finally:
    vp.MODELS[5].clear(); vp.MODELS[5].update(old5)
Y_noexog=dfl(Y_noexog_base)
np.save(root/'VARX'/'Y_pred_ablate_noexog_idio.npy', Y_noexog)
print('no-exog idio done')

# --- no smoothness: DFL on model4_idio (network, no smoothing) ---
Y_m4_idio=np.load(root/'VARX'/'Y_pred_model4_idio.npy')
Y_nosmooth=dfl(Y_m4_idio)
np.save(root/'VARX'/'Y_pred_ablate_nosmooth_idio.npy', Y_nosmooth)
print('no-smoothness idio done')

# --- no network: Sparse VARX + smooth + DFL (unchanged by idio, but recompute for completeness) ---
old3=dict(vp.MODELS[3]); vp.MODELS[3]['self_free']=True; vp.MODELS[3]['smooth']=True; vp.MODELS[3]['network']=False; vp.MODELS[3]['lasso_lambda']=vp.LAMBDA_LASSO
try:
    fit_nonnet=vp.fit_model(3, Xtr, Ytr, None, n_jobs=4)
    Y_m3s=vp.predict_model(Xte, fit_nonnet)
finally:
    vp.MODELS[3].clear(); vp.MODELS[3].update(old3)
Y_nonnet=dfl(Y_m3s)
np.save(root/'VARX'/'Y_pred_ablate_nonnet_idio.npy', Y_nonnet)
print('no-network idio done')
