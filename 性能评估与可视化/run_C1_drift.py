import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, warnings, importlib.util
import numpy as np, pandas as pd, joblib
from pathlib import Path
warnings.filterwarnings("ignore")

root = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
sys.path.insert(0, str(root/'图形Lasso'/'code'))
spec=importlib.util.spec_from_file_location('vp', str(root/'VARX'/'VAR及拓展（table2）.py'))
vp=importlib.util.module_from_spec(spec); sys.modules['vp']=vp; spec.loader.exec_module(vp)

K=392; ETA=1e-4; T0,T1=65,265; TARGET_TO=0.10
CACHE=root/'endtoend_project'/'cache'; FEAT=root/'特征工程'; VARX=root/'VARX'
valid=np.load(FEAT/'valid_indices.npy'); Y_all=np.load(FEAT/'Y_targets.npy'); X_all=np.load(FEAT/'X_features.npy')
rets=np.load(CACHE/'simple_returns.npy'); covs=np.load(CACHE/'cov_cache.npy',mmap_mode='r')
n=len(Y_all); n_train=int(0.70*n); n_val=int(0.15*n); test_start=n_train+n_val; n_test=n-test_start
Y_test=Y_all[test_start:]; Y_val=Y_all[n_train:test_start]; r_test=rets[test_start:]; r_val=rets[n_train:test_start]
Y_prev=Y_all[test_start-1]; Y_vprev=Y_all[n_train-1]
rf=np.load(root/'数据'/'rf_daily_test.npy')[T0:T1]

def drift(w,r):
    w=np.asarray(w,float); r=np.asarray(r,float); denom=1.0+float(w@r); return (w*(1.0+r)/denom) if abs(denom)>1e-15 else w

def eval_window(Y_pred):
    W=Y_pred[T0:T1]; R=T1-T0-1
    port_ret=np.array([W[t]@r_test[T0+t+1] for t in range(R)])
    # drift-adjusted turnover
    to_vec=np.array([np.sum(np.abs(W[t+1]-drift(W[t], r_test[T0+t+1]))) for t in range(R)])
    rpv=np.array([W[t]@(covs[test_start+T0+t+1]+1e-4*np.eye(K))@W[t] for t in range(R)])
    mse=float(np.mean((W-Y_test[T0:T1])**2)); mae=float(np.mean(np.abs(W-Y_test[T0:T1])))
    rpv_ann=float(np.mean(rpv)*252); to_mean=float(np.mean(to_vec))
    net=port_ret-ETA*to_vec
    rf_a=rf[1:] if len(rf)==len(net)+1 else rf[:len(net)]
    exc=net-rf_a
    sr=float(exc.mean()/exc.std(ddof=1)*np.sqrt(252)) if exc.std(ddof=1)>1e-15 else 0.0
    return dict(MSE=mse,MAE=mae,RPV=rpv_ann,TO=to_mean,SR=sr)

def norm(W):
    s=W.sum(axis=1,keepdims=True); s=np.where(np.abs(s)<1e-12,1.0,s); return W/s

rows=[]
# 1 persistence
Y1=np.empty_like(Y_test); Y1[0]=Y_prev; Y1[1:]=Y_test[:-1]
rows.append(("Last realized weight / graphical-lasso rule", eval_window(Y1)))
# 2 exponential smoothing
def es_path(alpha,Y_hist,y_prev):
    T=len(Y_hist); out=np.empty_like(Y_hist); f=y_prev.copy()
    for t in range(T):
        wlag=Y_hist[t-1] if t>0 else y_prev; f=(1-alpha)*wlag+alpha*f; out[t]=f
    return out
def es_mse(alpha,Y_hist,y_prev): return float(np.mean((es_path(alpha,Y_hist,y_prev)-Y_hist)**2))
alphas=np.linspace(0.0,0.95,20); best_alpha=min(alphas,key=lambda a: es_mse(a,Y_val,Y_vprev))
Y2=es_path(best_alpha,Y_test,Y_prev)
rows.append((f"Exponential smoothing of weights (alpha={best_alpha:.2f})", eval_window(Y2)))
# 3-4 G&G
rows.append(("Golosnoy and Gribisch statistical-loss model", eval_window(np.load(VARX/"Y_pred_GG_var1_LS.npy"))))
rows.append(("Golosnoy and Gribisch economic-loss model", eval_window(np.load(VARX/"Y_pred_GG_econ.npy"))))
# 5 covariance EWMA
lam=0.94; S=(rets[:252].T@rets[:252])/252+1e-4*np.eye(K); Y5=np.empty_like(Y_test)
for t in range(n_test):
    if t>0:
        rp=rets[test_start+t-1]; S=lam*S+(1-lam)*np.outer(rp,rp)
    Ss=S+1e-4*np.eye(K)
    try: w=np.linalg.solve(Ss,np.ones(K)); w=w/w.sum()
    except np.linalg.LinAlgError: w=np.ones(K)/K
    Y5[t]=w
rows.append(("Covariance EWMA (lambda=0.94)", eval_window(Y5)))
# partial adjustment
def pred_val(model_id, coefs=None, inter=None, cols=None, sc=None):
    if coefs is None:
        coefs=np.load(VARX/f"fitted_models/coefs_model{model_id}.npy"); inter=np.load(VARX/f"fitted_models/intercepts_model{model_id}.npy"); cols=np.load(VARX/f"fitted_models/feat_cols_model{model_id}.npy"); sc=joblib.load(VARX/f"fitted_models/scaler_model{model_id}.pkl")
    P=sc.transform(X_all[n_train:test_start][:,cols])@coefs.T+inter
    return norm(P)
def pa_path_and_to(base_v,r,y_prev,gamma):
    T=len(base_v); pos=y_prev.copy(); out=np.empty_like(base_v); to=np.empty(T)
    for t in range(T):
        pos_before=pos if t==0 else drift(pos,r[t-1])
        pos_new=(1-gamma)*pos_before+gamma*base_v[t]; pos_new=pos_new/pos_new.sum()
        out[t]=pos_new; to[t]=0.0 if t==0 else float(np.sum(np.abs(pos_new-pos_before))); pos=pos_new
    return out,float(np.mean(to[1:]))
# PA Sparse VARX (mid=3, dense, unchanged)
v_val=pred_val(3); v_test=np.load(VARX/"Y_pred_model3.npy")
gammas=np.linspace(0.02,1.0,50); best_g=None; best_diff=np.inf
for g in gammas:
    _,to_val=pa_path_and_to(v_val,r_val,Y_vprev,g); diff=abs(to_val-TARGET_TO)
    if diff<best_diff: best_diff,best_g=diff,g
Ypa,_=pa_path_and_to(v_test,r_test,Y_prev,best_g)
rows.append((f"Partial-adjustment Sparse VARX (gamma={best_g:.2f})", eval_window(Ypa)))
# PA Network VARX (mid=4, idio)
A_idio=np.load(FEAT/'A_bar_idio.npy'); ntr=int(0.7*len(A_idio))
mask=(A_idio[:ntr].mean(0)>=0.117).astype(float); np.fill_diagonal(mask,0)
f4=vp.fit_model(4, X_all[:n_train], Y_all[:n_train], mask, n_jobs=4)
coefs=f4['coefs']; inter=f4['intercepts']; cols=f4['cols']; sc=f4['scaler']
v_val4=norm(sc.transform(X_all[n_train:test_start][:,cols])@coefs.T+inter)
v_test4=np.load(VARX/"Y_pred_model4_idio.npy")[-n_test:]
best_g=None; best_diff=np.inf
for g in gammas:
    _,to_val=pa_path_and_to(v_val4,r_val,Y_vprev,g); diff=abs(to_val-TARGET_TO)
    if diff<best_diff: best_diff,best_g=diff,g
Ypa4,_=pa_path_and_to(v_test4,r_test,Y_prev,best_g)
rows.append((f"Partial-adjustment Network VARX (gamma={best_g:.2f})", eval_window(Ypa4)))
# 8 random forest
rows.append(("Random forest", eval_window(np.load(VARX/"Y_pred_rf.npy"))))
# 9 DF idio
rows.append(("Network VARX-S-DF", eval_window(np.load(VARX/"Y_pred_model6_opt_idio.npy"))))

print(f"{'Model':<48} {'MSE_w':>10} {'MAE_w':>10} {'RPV':>9} {'Turnover':>9} {'Net SR':>8}")
for nm,m in rows:
    print(f"{nm:<48} {m['MSE']*1e5:>10.4f} {m['MAE']*1e3:>10.4f} {m['RPV']*1e3:>9.4f} {m['TO']:>9.4f} {m['SR']:>8.4f}")
