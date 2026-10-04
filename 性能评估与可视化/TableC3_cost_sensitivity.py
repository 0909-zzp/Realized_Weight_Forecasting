import os, warnings, importlib.util, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
P=Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
sys.path.insert(0, str(P/'图形Lasso'/'code'))
from 共享模块 import K, LAMBDA_LASSO

spec=importlib.util.spec_from_file_location('vp', str(P/'VARX'/'VAR及拓展（table2）.py'))
vp=importlib.util.module_from_spec(spec); sys.modules['vp']=vp; spec.loader.exec_module(vp)

X=np.load(P/'特征工程'/'X_features.npy'); Y=np.load(P/'特征工程'/'Y_targets.npy'); A=np.load(P/'特征工程'/'A_bar.npy')
valid=np.load(P/'特征工程'/'valid_indices.npy')
n=len(X); ntr=int(n*.7); nval=int(n*.15); test_start=ntr+nval; n_test=n-test_start
Xtr,Ytr,Atr=X[:ntr],Y[:ntr],A[:ntr]
Xte,Yte=X[test_start:],Y[test_start:]
train_day_indices=valid[:ntr]

# full-test simple returns (363 days)
npy=sorted([f for f in (P/'数据'/'1min_log_return_npy').iterdir() if f.suffix=='.npy' and f.name[0].isdigit()])
test_indices=valid[test_start:]
R_full=np.array([np.expm1(np.load(npy[i]).sum(axis=1)) for i in test_indices])
print('R_full', R_full.shape)

def drift(w,r):
    w=np.asarray(w,float); r=np.asarray(r,float)
    d=1.0+float(w@r)
    return (w*(1+r)/d) if abs(d)>1e-15 else w

# ---- reconstruct partial-adjustment benchmark (gamma=0.38 toward Sparse VARX) ----
def pa_path(base_v, r, y_prev, gamma):
    T=len(base_v); pos=y_prev.copy(); out=np.empty_like(base_v)
    for t in range(T):
        pos_before = pos if t==0 else drift(pos, r[t-1])
        pos_new=(1-gamma)*pos_before + gamma*base_v[t]
        pos_new=pos_new/pos_new.sum()
        out[t]=pos_new; pos=pos_new
    return out

Y_sparse_full=np.load(P/'VARX'/'Y_pred_model3.npy')[-n_test:]
Y_prev=Y[test_start-1]
Y_pa_full=pa_path(Y_sparse_full, R_full, Y_prev, 0.38)
print('partial-adjustment done')

# ---- reconstruct Sparse VARX-S-DF: M3 + smooth + DFL ----
old3=dict(vp.MODELS[3])
vp.MODELS[3]['self_free']=True
vp.MODELS[3]['smooth']=True
vp.MODELS[3]['network']=False
vp.MODELS[3]['lasso_lambda']=LAMBDA_LASSO
try:
    fitted_nonnet=vp.fit_model(3, Xtr, Ytr, None, n_jobs=4)
    Y_m3_smooth=vp.predict_model(Xte, fitted_nonnet)
finally:
    vp.MODELS[3].clear(); vp.MODELS[3].update(old3)
print('M3+smooth fitted')

t0=time.time()
Y_sdf_full=vp.compute_model6_drift_l1(Y_m3_smooth, Yte, train_day_indices,
    eta=1e-6, rho=1e-3, risk_mult=1.0, rolling_cov=True, cov_window=150)
print('Sparse VARX-S-DF done', round(time.time()-t0,1),'s')

# ---- collect 7 paths (full 363) ----
paths={
 'Equal weight': np.ones((n_test,K))/K,
 'Sample GMVP': np.load(P/'VARX'/'Y_pred_bench_sample_W20.npy')[-n_test:],
 'Sparse VARX': Y_sparse_full,
 'Network VARX': np.load(P/'VARX'/'Y_pred_model4.npy')[-n_test:],
 'Partial-adjustment benchmark': Y_pa_full,
 'Sparse VARX-S-DF': Y_sdf_full,
 'Network VARX-S-DF': np.load(P/'VARX'/'Y_pred_model6_opt.npy')[-n_test:],
}

# ---- evaluate on 200-day window [65:265] ----
T0,T1=65,265
R_win=R_full[T0:T1]  # 200 days
def series(Y_win, R):
    T=len(Y_win)-1
    g=np.array([Y_win[t]@R[t+1] for t in range(T)])
    to=np.array([np.abs(Y_win[t+1]-Y_win[t]*(1+R[t+1])/(1+Y_win[t]@R[t+1])).sum() for t in range(T)])
    return g,to

costs_bp=[0.0,0.5,1.0,5.0,10.0]
rows=[]
for name,W in paths.items():
    W_win=W[T0:T1]
    g,to=series(W_win, R_win)
    row={'Model':name,'Turnover':float(to.mean())}
    for c in costs_bp:
        net=g-(c/10000.0)*to
        sd=net.std(ddof=1)
        sr=float(net.mean()/sd*np.sqrt(252)) if sd>1e-15 else 0.0
        row[f'{c:g}bp']=sr
    rows.append(row)

df=pd.DataFrame(rows)
df.to_csv(P/'性能评估与可视化'/'TableC3_cost_sensitivity.csv', index=False)
print()
print(df.round(4).to_string(index=False))
print('\nSaved: 性能评估与可视化/TableC3_cost_sensitivity.csv')
