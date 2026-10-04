import os, sys, time, warnings, importlib.util
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'图形Lasso'/'code')); sys.path.insert(0,str(ROOT/'性能评估与可视化'))
spec=importlib.util.spec_from_file_location('roll2',ROOT/'性能评估与可视化'/'dfl_oos_rolling_sigma.py'); roll=importlib.util.module_from_spec(spec); sys.modules['roll2']=roll; spec.loader.exec_module(roll)
spec=importlib.util.spec_from_file_location('tune2',ROOT/'性能评估与可视化'/'dfl_oos_tuning.py'); tune=importlib.util.module_from_spec(spec); sys.modules['tune2']=tune; spec.loader.exec_module(tune)
vspec=importlib.util.spec_from_file_location('varxmod5',ROOT/'VARX'/'VAR及拓展（table2）.py'); vp=importlib.util.module_from_spec(vspec); sys.modules['varxmod5']=vp; vspec.loader.exec_module(vp)
K=vp.K; NPY=ROOT/'数据'/'1min_log_return_npy'; all_files=sorted([f for f in NPY.iterdir() if f.suffix=='.npy' and f.name[0].isdigit()])
tag='covariance_shrink0.1'; d=ROOT/'table5_alt'/tag
X=np.load(d/'X_features.npy'); Y=np.load(d/'Y_targets.npy'); Abar=np.load(d/'A_bar.npy'); vi=np.load(d/'valid_indices.npy')
n=len(X); tr=int(n*.7); va=int(n*.15); ts=tr+va; Xtr=X[:tr]; Ytr=Y[:tr]; Atr=Abar[:tr]; mask,dens=vp.build_network_mask(Atr)
fit=vp.fit_model(5,Xtr,Ytr,mask,n_jobs=8); Yb=vp.predict_model(X[tr:],fit); Yb_val=Yb[:va]; Yb_test=Yb[va:]; Yval=Y[tr:tr+va]; Ytest=Y[ts:]
rets=lambda inds: np.array([np.expm1(np.load(all_files[i]).sum(axis=1)) for i in inds]); rv=rets(vi[tr:tr+va]); rt=rets(vi[ts:])
sv=roll.build_sigmas(vi[tr:tr+va],150,all_files); st=roll.build_sigmas(vi[ts:],150,all_files)
best=None
for rho in [1e-3,1e-2,1e-1,1.0,10.0]:
 mv=tune.precompute_minvs(sv,rho,1.0); yd=tune.dfl_drift_l1_rolling(Yb_val,rv,Yval[0],sv,mv,1e-6,rho,1.0); mse=float(((yd-Yval)**2).mean()); print('rho',rho,'val',mse,flush=True)
 if best is None or mse<best[0]: best=(mse,rho)
rho=best[1]; mv=tune.precompute_minvs(st,rho,1.0); yd=tune.dfl_drift_l1_rolling(Yb_test,rt,Ytest[0],st,mv,1e-6,rho,1.0); np.save(d/'Y_pred_DF_retuned.npy',yd)
a,b=65,265; yy=Ytest[a:b]; S=np.load(d/'Y_pred_S.npy')[-363:][a:b]; N=np.load(d/'Y_pred_N.npy')[-363:][a:b]; D=yd[a:b]
def mm(p): return float(((p-yy)**2).mean()),float(np.abs(p-yy).mean())
sm,sa=mm(S); nm,na=mm(N); dm,da=mm(D); row={'tag':tag,'best_rho':rho,'val_mse':best[0],'ratio_N_mse':nm/sm,'ratio_N_mae':na/sa,'ratio_DF_mse':dm/sm,'ratio_DF_mae':da/sa,'mse_S':sm,'mse_N':nm,'mse_DF':dm}; (d/'result_retuned.txt').write_text(str(row),encoding='utf-8'); print('RETUNED',row,flush=True)
