import os, sys, time, warnings, importlib.util
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'图形Lasso'/'code'))
sys.path.insert(0, str(ROOT/'特征工程'))

# feature module
fspec = importlib.util.spec_from_file_location('featmod', ROOT/'特征工程'/'特征工程.py')
feat = importlib.util.module_from_spec(fspec); sys.modules['featmod']=feat; fspec.loader.exec_module(feat)
# varx module
vspec = importlib.util.spec_from_file_location('varxmod', ROOT/'VARX'/'VAR及拓展（table2）.py')
vp = importlib.util.module_from_spec(vspec); sys.modules['varxmod']=vp; vspec.loader.exec_module(vp)

K = vp.K
NPY_DIR = ROOT/'数据'/'1min_log_return_npy'
files = sorted([f for f in NPY_DIR.iterdir() if f.suffix == '.npy' and f.name[0].isdigit()])
dates = [f.stem[:8] for f in files]
T = len(files)
print('days', T, 'K', K, flush=True)

# shared exogenous
print('loading exogenous', flush=True)
t0=time.time()
exog_df = feat.load_exogenous(dates, str(NPY_DIR), smooth_window=0)
exog_arr = exog_df.fillna(method='ffill').fillna(0).values
print('exog done', exog_arr.shape, 'sec', round(time.time()-t0,1), flush=True)

EPS=1e-4
RIDGE_CHAIN=[1e-4,5e-4,1e-3,5e-3,1e-2]
COV_WINDOW=20
PRECISION_LAM=1e-3

def fit_glasso_day(f, lam):
    from sklearn.covariance import graphical_lasso
    rett=np.load(f); raw=rett@rett.T; raw.flat[::K+1]+=EPS
    for r in RIDGE_CHAIN:
        c=raw.copy()
        if r>EPS: c.flat[::K+1]+=(r-EPS)
        try:
            _,prec=graphical_lasso(c,alpha=lam,mode='cd',tol=1e-4,max_iter=100,enet_tol=5e-4)
            return prec
        except Exception:
            continue
    return None

def precision_targets():
    print('generating precision target lam=1e-3', flush=True)
    t0=time.time()
    results=Parallel(n_jobs=8, backend='threading')(delayed(fit_glasso_day)(str(f), PRECISION_LAM) for f in files)
    W=np.full((T,K), np.nan)
    adj_list=[None]*T
    for i,prec in enumerate(results):
        if prec is None: continue
        w=prec@np.ones(K); W[i]=w/w.sum()
        adj=(np.abs(prec)>1e-8).astype(np.int8); np.fill_diagonal(adj,0); adj_list[i]=adj
    print('precision target done', 'sec', round(time.time()-t0,1), 'missing', int(np.isnan(W).any(axis=1).sum()), flush=True)
    return W, adj_list

def covariance_targets():
    print('generating covariance target W=20', flush=True)
    t0=time.time()
    W=np.full((T,K), np.nan)
    # rolling covariance accumulator
    acc=None
    for i,f in enumerate(files):
        rett=np.load(f); raw=rett@rett.T
        raw.flat[::K+1]+=EPS
        if acc is None:
            acc=raw.copy()
        else:
            acc=acc+raw
        n_used=min(COV_WINDOW, i+1)
        if i>=COV_WINDOW:
            old=np.load(files[i-COV_WINDOW]); oldraw=old@old.T; oldraw.flat[::K+1]+=EPS; acc=acc-oldraw
        cov=acc/n_used
        cov.flat[::K+1]+=1e-6
        try:
            inv=np.linalg.inv(cov)
            w=inv@np.ones(K); W[i]=w/w.sum()
        except Exception:
            W[i]=np.nan
    # no sparse adjacency; use baseline adjacency from existing dir for fixed network mask
    adjdir=ROOT/'图形Lasso'/'code'/'输出数据'/'adjacency'
    adj_list=[None]*T
    for i,d in enumerate(dates):
        p=adjdir/f'{d}.npy'
        if p.exists(): adj_list[i]=np.load(p)
    print('covariance target done', 'sec', round(time.time()-t0,1), 'missing', int(np.isnan(W).any(axis=1).sum()), flush=True)
    return W, adj_list

def save_target(tag, W, adj_list):
    arr=np.array([a if a is not None else np.zeros((K,K),dtype=np.int8) for a in adj_list], dtype=np.int8)
    np.savez_compressed(ROOT/'table5_alt'/f'{tag}_target.npz', weights=W, adj=arr)

def load_target(tag):
    z=np.load(ROOT/'table5_alt'/f'{tag}_target.npz')
    return z['weights'], [z['adj'][i] for i in range(z['adj'].shape[0])]

def run_target(tag, W, adj_list):
    outdir=ROOT/'table5_alt'/tag
    outdir.mkdir(parents=True, exist_ok=True)
    valid_mask=~np.isnan(W).any(axis=1)
    print(tag,'building features', flush=True)
    t0=time.time()
    X,Y,A_bar,valid_idx,names=feat.build_feature_matrix(W, adj_list, exog_arr, valid_mask, p_lags=vp.P_LAGS, include_net_topo=False)
    print(tag,'features done', X.shape, Y.shape, A_bar.shape, 'sec', round(time.time()-t0,1), flush=True)
    np.save(outdir/'X_features.npy', X); np.save(outdir/'Y_targets.npy', Y); np.save(outdir/'A_bar.npy', A_bar)
    np.save(outdir/'valid_indices.npy', np.array(valid_idx))
    # split exactly as baseline
    n=len(X); tr=int(n*.7); va=int(n*.15); ts=tr+va
    Xtr=X[:tr]; Ytr=Y[:tr]; Atr=A_bar[:tr]
    mask,dens=vp.build_network_mask(Atr)
    print(tag,'mask density',dens,'train/test',tr,ts,'n',n,flush=True)
    valid_days=valid_idx[:tr]
    Yte=Y[ts:]
    for mid,tagm in [(3,'S'),(4,'N'),(5,'M5')]:
        t0=time.time(); fit=vp.fit_model(mid, Xtr, Ytr, mask if mid in (4,5) else None, n_jobs=8); pred=vp.predict_model(X[ts:], fit); np.save(outdir/f'Y_pred_{tagm}.npy', pred); print(tag,tagm,'train+pred sec',round(time.time()-t0,1),flush=True)
    base=np.load(outdir/'Y_pred_M5.npy')
    t0=time.time(); dfl=vp.compute_model6_drift_l1(base, Yte, valid_days, eta=1e-6, rho=1e-3, risk_mult=1.0, rolling_cov=True, cov_window=150); np.save(outdir/'Y_pred_DF.npy', dfl); print(tag,'DF sec',round(time.time()-t0,1),flush=True)
    # metrics reported 200-day
    a,b=65,265
    yy=Y[ts:][a:b]
    def mm(p): return float(((p-yy)**2).mean()), float(np.abs(p-yy).mean())
    S=np.load(outdir/'Y_pred_S.npy')[-363:][a:b]; N=np.load(outdir/'Y_pred_N.npy')[-363:][a:b]; DF=np.load(outdir/'Y_pred_DF.npy')[-363:][a:b]
    sm,sa=mm(S); nm,na=mm(N); dm,da=mm(DF)
    row={'tag':tag,'mse_S':sm,'mae_S':sa,'mse_N':nm,'mae_N':na,'mse_DF':dm,'mae_DF':da,'ratio_N_mse':nm/sm,'ratio_N_mae':na/sa,'ratio_DF_mse':dm/sm,'ratio_DF_mae':da/sa}
    print(tag,'RESULT',row,flush=True)
    with open(outdir/'result.txt','w',encoding='utf-8') as f: f.write(str(row))
    del X,Y,A_bar,W,adj_list,Xtr,Ytr,Atr

if __name__ == '__main__':
    pt=ROOT/'table5_alt'/'precision_lam1e-3_target.npz'
    if pt.exists(): Wp,Ap=load_target('precision_lam1e-3')
    else: Wp,Ap=precision_targets(); save_target('precision_lam1e-3',Wp,Ap)
    run_target('precision_lam1e-3', Wp, Ap)
    del Wp, Ap
    ct=ROOT/'table5_alt'/'covariance_W20_sample_target.npz'
    if ct.exists(): Wc,Ac=load_target('covariance_W20_sample')
    else: Wc,Ac=covariance_targets(); save_target('covariance_W20_sample',Wc,Ac)
    run_target('covariance_W20_sample', Wc, Ac)
    print('ALL_PANELC_TARGETS_DONE', flush=True)
