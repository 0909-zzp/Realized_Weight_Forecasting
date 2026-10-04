import os, sys, time, warnings, importlib.util
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
from sklearn.covariance import ShrunkCovariance
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'图形Lasso'/'code')); sys.path.insert(0,str(ROOT/'特征工程'))
fspec=importlib.util.spec_from_file_location('featmod4',ROOT/'特征工程'/'特征工程.py'); feat=importlib.util.module_from_spec(fspec); sys.modules['featmod4']=feat; fspec.loader.exec_module(feat)
vspec=importlib.util.spec_from_file_location('varxmod4',ROOT/'VARX'/'VAR及拓展（table2）.py'); vp=importlib.util.module_from_spec(vspec); sys.modules['varxmod4']=vp; vspec.loader.exec_module(vp)
K=vp.K; NPY=ROOT/'数据'/'1min_log_return_npy'; files=sorted([f for f in NPY.iterdir() if f.suffix=='.npy' and f.name[0].isdigit()]); dates=[f.stem[:8] for f in files]; T=len(files)
print('load exog',flush=True); exog=feat.load_exogenous(dates,str(NPY),smooth_window=0); exog_arr=exog.fillna(method='ffill').fillna(0).values
EPS=1e-4; SHRINK=0.1; tag='covariance_shrink0.1'; out=ROOT/'table5_alt'/tag; out.mkdir(parents=True,exist_ok=True)
print('gen shrink target',flush=True); t=time.time(); W=np.full((T,K),np.nan)
for i,f in enumerate(files):
 rett=np.load(f); raw=rett@rett.T; raw.flat[::K+1]+=EPS
 cov=ShrunkCovariance(shrinkage=SHRINK).fit(rett.T).covariance_
 w=np.linalg.solve(cov+1e-6*np.eye(K),np.ones(K)); W[i]=w/w.sum()
print('shrink target done',round(time.time()-t,1),'missing',int(np.isnan(W).any(axis=1).sum()),flush=True)
adjdir=ROOT/'图形Lasso'/'code'/'输出数据'/'adjacency'; adj=[None]*T
for i,d in enumerate(dates):
 p=adjdir/f'{d}.npy'
 if p.exists(): adj[i]=np.load(p)
# save target npz
np.savez_compressed(ROOT/'table5_alt'/f'{tag}_target.npz',weights=W,adj=np.array([a if a is not None else np.zeros((K,K),dtype=np.int8) for a in adj],dtype=np.int8))
vm=~np.isnan(W).any(axis=1); t=time.time(); X,Y,Abar,vi,names=feat.build_feature_matrix(W,adj,exog_arr,vm,p_lags=vp.P_LAGS,include_net_topo=False); print('features',X.shape,'sec',round(time.time()-t,1),flush=True)
np.save(out/'X_features.npy',X); np.save(out/'Y_targets.npy',Y); np.save(out/'A_bar.npy',Abar); np.save(out/'valid_indices.npy',np.array(vi))
n=len(X); tr=int(n*.7); va=int(n*.15); ts=tr+va; Xtr=X[:tr]; Ytr=Y[:tr]; Atr=Abar[:tr]; mask,dens=vp.build_network_mask(Atr); valid_days=vi[:tr]; Yte=Y[ts:]
print('dens',dens,flush=True)
for mid,tm in [(3,'S'),(4,'N'),(5,'M5')]:
 t=time.time(); fit=vp.fit_model(mid,Xtr,Ytr,mask if mid in (4,5) else None,n_jobs=8); pred=vp.predict_model(X[ts:],fit); np.save(out/f'Y_pred_{tm}.npy',pred); print(tm,round(time.time()-t,1),flush=True)
base=np.load(out/'Y_pred_M5.npy'); t=time.time(); dfl=vp.compute_model6_drift_l1(base,Yte,valid_days,eta=1e-6,rho=1e-3,risk_mult=1.0,rolling_cov=True,cov_window=150); np.save(out/'Y_pred_DF.npy',dfl); print('DF',round(time.time()-t,1),flush=True)
a,b=65,265; yy=Y[ts:][a:b]
def mm(p): return float(((p-yy)**2).mean()),float(np.abs(p-yy).mean())
S=np.load(out/'Y_pred_S.npy')[-363:][a:b]; N=np.load(out/'Y_pred_N.npy')[-363:][a:b]; DF=np.load(out/'Y_pred_DF.npy')[-363:][a:b]
sm,sa=mm(S); nm,na=mm(N); dm,da=mm(DF); row={'tag':tag,'mse_S':sm,'mse_N':nm,'mse_DF':dm,'ratio_N_mse':nm/sm,'ratio_N_mae':na/sa,'ratio_DF_mse':dm/sm,'ratio_DF_mae':da/sa}; print('RESULT',row,flush=True); (out/'result.txt').write_text(str(row),encoding='utf-8')
print('SHRINK_DONE',flush=True)
