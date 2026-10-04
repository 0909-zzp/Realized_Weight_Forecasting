import os, sys, time, warnings, importlib.util
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'图形Lasso'/'code')); sys.path.insert(0,str(ROOT/'性能评估与可视化')); sys.path.insert(0,str(ROOT/'特征工程'))
# import dfl modules
spec=importlib.util.spec_from_file_location('roll',ROOT/'性能评估与可视化'/'dfl_oos_rolling_sigma.py'); roll=importlib.util.module_from_spec(spec); sys.modules['roll']=roll; spec.loader.exec_module(roll)
spec=importlib.util.spec_from_file_location('tune',ROOT/'性能评估与可视化'/'dfl_oos_tuning.py'); tune=importlib.util.module_from_spec(spec); sys.modules['tune']=tune; spec.loader.exec_module(tune)
# varx module
vspec=importlib.util.spec_from_file_location('varxmod3',ROOT/'VARX'/'VAR及拓展（table2）.py'); vp=importlib.util.module_from_spec(vspec); sys.modules['varxmod3']=vp; vspec.loader.exec_module(vp)
K=vp.K; NPY=ROOT/'数据'/'1min_log_return_npy'; all_files=sorted([f for f in NPY.iterdir() if f.suffix=='.npy' and f.name[0].isdigit()])
WIN=150; ETA=1e-6; RISK=1.0; RHO_GRID=[1e-3,1e-2,1e-1,1.0,10.0]

def simple_rets(day_indices):
    return np.array([np.expm1(np.load(all_files[i]).sum(axis=1)) for i in day_indices])

def run_tag(tag):
    d=ROOT/'table5_alt'/tag
    X=np.load(d/'X_features.npy'); Y=np.load(d/'Y_targets.npy'); Abar=np.load(d/'A_bar.npy'); vi=np.load(d/'valid_indices.npy')
    n=len(X); tr=int(n*.7); va=int(n*.15); ts=tr+va
    Xtr=X[:tr]; Ytr=Y[:tr]; Atr=Abar[:tr]
    mask,dens=vp.build_network_mask(Atr)
    t=time.time(); fit=vp.fit_model(5,Xtr,Ytr,mask,n_jobs=8); Yb=vp.predict_model(X[tr:],fit); print(tag,'M5 train/pred',round(time.time()-t,1),'dens',dens,flush=True)
    Yb_val=Yb[:va]; Yb_test=Yb[va:]
    Yval=Y[tr:tr+va]; Ytest=Y[ts:]
    div_val=vi[tr:tr+va]; div_test=vi[ts:]
    rets_val=simple_rets(div_val); rets_test=simple_rets(div_test)
    sig_val=roll.build_sigmas(div_val,WIN,all_files); sig_test=roll.build_sigmas(div_test,WIN,all_files)
    best=None
    for rho in RHO_GRID:
        t=time.time(); mv=tune.precompute_minvs(sig_val,rho,RISK); yd=tune.dfl_drift_l1_rolling(Yb_val,rets_val,Yval[0],sig_val,mv,ETA,rho,RISK); mse=float(((yd-Yval)**2).mean()); print(tag,'rho',rho,'val_mse',mse,'sec',round(time.time()-t,1),flush=True)
        if best is None or mse<best[0]: best=(mse,rho)
    rho=best[1]
    mv=tune.precompute_minvs(sig_test,rho,RISK); yd=tune.dfl_drift_l1_rolling(Yb_test,rets_test,Ytest[0],sig_test,mv,ETA,rho,RISK)
    np.save(d/'Y_pred_DF_retuned.npy',yd)
    a,b=65,265; yy=Ytest[a:b]; S=np.load(d/'Y_pred_S.npy')[-363:][a:b]; N=np.load(d/'Y_pred_N.npy')[-363:][a:b]; D=yd[a:b]
    def mm(p): return float(((p-yy)**2).mean()),float(np.abs(p-yy).mean())
    sm,sa=mm(S); nm,na=mm(N); dm,da=mm(D)
    row={'tag':tag,'best_rho':rho,'val_mse':best[0],'ratio_N_mse':nm/sm,'ratio_N_mae':na/sa,'ratio_DF_mse':dm/sm,'ratio_DF_mae':da/sa,'mse_S':sm,'mse_N':nm,'mse_DF':dm}
    (d/'result_retuned.txt').write_text(str(row),encoding='utf-8'); print(tag,'RETUNED',row,flush=True)
    del X,Y,Abar,Xtr,Ytr,Atr,sig_val,sig_test

if __name__=='__main__':
    for tag in ['precision_lam5e-04','precision_lam7e-04','precision_lam1e-3','covariance_W20_sample']:
        run_tag(tag)
    print('RETUNE_DONE',flush=True)
