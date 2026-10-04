import os, sys, time, warnings, importlib.util
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'图形Lasso'/'code'))
sys.path.insert(0, str(ROOT/'特征工程'))
fspec=importlib.util.spec_from_file_location('featmod3', ROOT/'特征工程'/'特征工程.py'); feat=importlib.util.module_from_spec(fspec); sys.modules['featmod3']=feat; fspec.loader.exec_module(feat)
vspec=importlib.util.spec_from_file_location('varxmod3', ROOT/'VARX'/'VAR及拓展（table2）.py'); vp=importlib.util.module_from_spec(vspec); sys.modules['varxmod3']=vp; vspec.loader.exec_module(vp)
K=vp.K; NPY=ROOT/'数据'/'1min_log_return_npy'; files=sorted([f for f in NPY.iterdir() if f.suffix=='.npy' and f.name[0].isdigit()]); dates=[f.stem[:8] for f in files]; T=len(files)
print('load exog', flush=True); t=time.time(); exog=feat.load_exogenous(dates,str(NPY),smooth_window=0); exog_arr=exog.fillna(method='ffill').fillna(0).values; print('exog',exog_arr.shape,round(time.time()-t,1), flush=True)
EPS=1e-4; RIDGE=[1e-4,5e-4,1e-3,5e-3,1e-2]

def fit_day(f,lam):
    from sklearn.covariance import graphical_lasso
    rett=np.load(f); raw=rett@rett.T; raw.flat[::K+1]+=EPS
    for r in RIDGE:
        c=raw.copy()
        if r>EPS: c.flat[::K+1]+=(r-EPS)
        try:
            _,prec=graphical_lasso(c,alpha=lam,mode='cd',tol=1e-4,max_iter=100,enet_tol=5e-4)
            return prec
        except Exception: continue
    return None

def gen(lam):
    print('gen precision', lam, flush=True); t=time.time(); res=Parallel(n_jobs=8,backend='threading')(delayed(fit_day)(str(f),lam) for f in files)
    W=np.full((T,K),np.nan); adj=[None]*T
    for i,prec in enumerate(res):
        if prec is None: continue
        w=prec@np.ones(K); W[i]=w/w.sum(); a=(np.abs(prec)>1e-8).astype(np.int8); np.fill_diagonal(a,0); adj[i]=a
    print('gen done', lam, 'sec', round(time.time()-t,1), 'missing', int(np.isnan(W).any(axis=1).sum()), flush=True)
    return W,adj

def save(tag,W,adj):
    arr=np.array([a if a is not None else np.zeros((K,K),dtype=np.int8) for a in adj],dtype=np.int8)
    np.savez_compressed(ROOT/'table5_alt'/f'{tag}_target.npz', weights=W, adj=arr)

def load(tag):
    z=np.load(ROOT/'table5_alt'/f'{tag}_target.npz'); return z['weights'],[z['adj'][i] for i in range(z['adj'].shape[0])]

def run(tag,W,adj,lam):
    out=ROOT/'table5_alt'/tag; out.mkdir(parents=True,exist_ok=True)
    vm=~np.isnan(W).any(axis=1); t=time.time()
    X,Y,Abar,vi,names=feat.build_feature_matrix(W,adj,exog_arr,vm,p_lags=vp.P_LAGS,include_net_topo=False)
    print(tag,'features',X.shape,'sec',round(time.time()-t,1), flush=True)
    np.save(out/'X_features.npy',X); np.save(out/'Y_targets.npy',Y); np.save(out/'A_bar.npy',Abar); np.save(out/'valid_indices.npy',np.array(vi))
    n=len(X); tr=int(n*.7); va=int(n*.15); ts=tr+va; Xtr=X[:tr]; Ytr=Y[:tr]; Atr=Abar[:tr]
    mask,dens=vp.build_network_mask(Atr); print(tag,'dens',dens, flush=True)
    valid_days=vi[:tr]; Yte=Y[ts:]; nt=n-ts
    for mid,tm in [(3,'S'),(4,'N'),(5,'M5')]:
        t=time.time(); fit=vp.fit_model(mid,Xtr,Ytr,mask if mid in (4,5) else None,n_jobs=8); pred=vp.predict_model(X[ts:],fit); np.save(out/f'Y_pred_{tm}.npy',pred); print(tag,tm,round(time.time()-t,1), flush=True)
    base=np.load(out/'Y_pred_M5.npy'); t=time.time()
    dfl=vp.compute_model6_drift_l1(base,Yte,valid_days,eta=1e-6,rho=1e-3,risk_mult=1.0,rolling_cov=True,cov_window=150)
    np.save(out/'Y_pred_DF.npy',dfl); print(tag,'DF',round(time.time()-t,1), flush=True)
    # correct alignment: pred[65:265] vs target[65:265] on the same test index
    a,b=65,265; yy=Y[ts:][a:b]
    def mm(p): return float(((p-yy)**2).mean()), float(np.abs(p-yy).mean())
    S=np.load(out/'Y_pred_S.npy')[-nt:][a:b]; N=np.load(out/'Y_pred_N.npy')[-nt:][a:b]; DF=np.load(out/'Y_pred_DF.npy')[-nt:][a:b]
    sm,sa=mm(S); nm,na=mm(N); dm,da=mm(DF)
    row={'lam':lam,'mse_S':sm,'mae_S':sa,'mse_N':nm,'mae_N':na,'mse_DF':dm,'mae_DF':da,'ratio_N_mse':nm/sm,'ratio_N_mae':na/sa,'ratio_DF_mse':dm/sm,'ratio_DF_mae':da/sa}
    print(tag,'RESULT',row, flush=True); (out/'result.txt').write_text(str(row),encoding='utf-8')
    del X,Y,Abar,W,adj,Xtr,Ytr,Atr

if __name__=='__main__':
    for lam in [5e-5, 1e-5]:
        tag=f'precision_lam{lam:.0e}'
        pf=ROOT/'table5_alt'/f'{tag}_target.npz'
        if pf.exists(): W,adj=load(tag)
        else: W,adj=gen(lam); save(tag,W,adj)
        run(tag,W,adj,lam)
    print('MODERATE_LAMBDA_DONE', flush=True)
