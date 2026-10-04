import os, warnings, importlib.util, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import networkx as nx
P=Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
OUT=P/'性能评估与可视化'
spec=importlib.util.spec_from_file_location('vm2', str(P/'VARX'/'VAR及拓展（table2）.py')); vm=importlib.util.module_from_spec(spec); sys.modules['vm2']=vm; spec.loader.exec_module(vm)
X=np.load(P/'特征工程'/'X_features.npy'); Y=np.load(P/'特征工程'/'Y_targets.npy')
A_idio=np.load(P/'特征工程'/'A_bar_idio.npy')
sector=pd.read_csv(P/'数据'/'sector_labels.csv')['sector'].tolist()
K=392; n=len(X); ntr=int(0.7*n); nval=int(0.15*n); ts=ntr+nval
Xtr,Ytr=X[:ntr],Y[:ntr]; Xte=X[ts:]; Yte=Y[ts:]
Y_true=Yte[65:265]; Y_sparse=np.load(P/'VARX'/'Y_pred_model3.npy')[65:265]
THR=0.117

def build_mask(Atr):
    m=(Atr.mean(0)>=THR).astype(float); np.fill_diagonal(m,0); return m

real_mask=build_mask(A_idio[:ntr])
real_dens=real_mask.sum()/(K*(K-1))
print(f'因子分解网络: 密度={real_dens:.4f} 平均度={real_mask.sum(1).mean():.1f} 边数={int(real_mask.sum()/2)}')

def fit_eval(mask):
    fitted=vm.fit_model(4, Xtr, Ytr, mask, n_jobs=4)
    Yp=vm.predict_model(Xte, fitted); Ype=Yp[65:265]
    mse=vm.compute_mse(Ype, Y_true); mae=vm.compute_mae(Ype, Y_true)
    dm,p,lp=vm.dm_test(Ype, Y_sparse, Y_true)
    return mse,mae,dm,p

def density_matched(ref, rng):
    E=int(ref.sum()//2); iu=np.triu_indices(K,k=1); npair=len(iu[0])
    ch=rng.choice(npair,size=E,replace=False)
    m=np.zeros((K,K)); m[iu[0][ch],iu[1][ch]]=1; m[iu[1][ch],iu[0][ch]]=1; return m

def degree_preserving(ref, rng):
    G=nx.from_numpy_array(ref); seed=int(rng.integers(0,2**31-1))
    G2=nx.double_edge_swap(G, nswap=G.number_of_edges()*5, max_tries=G.number_of_edges()*100, seed=seed)
    return nx.to_numpy_array(G2)

# sector mask
sarr=np.array(sector); sector_mask=(sarr[:,None]==sarr[None,:]).astype(float); np.fill_diagonal(sector_mask,0)

rows=[]
mse0=vm.compute_mse(Y_sparse, Y_true); mae0=vm.compute_mae(Y_sparse, Y_true)
rows.append(dict(network='Uniform cross-asset penalty', density=np.nan, MSE=mse0, MAE=mae0, DM=0.0, DM_p=np.nan))
# realized (factor-decomposed)
mseR,maeR,dmR,pR=fit_eval(real_mask)
rows.append(dict(network='Realized forecasting network', density=real_dens, MSE=mseR, MAE=maeR, DM=dmR, DM_p=pR))
print(f'realized: MSE={mseR:.4e} MAE={maeR:.4e} DM={dmR:.3f}', flush=True)
# fixed initial-training
mask_fixed=build_mask(A_idio[:500]); dens_fixed=mask_fixed.sum()/(K*(K-1))
mseF,maeF,dmF,pF=fit_eval(mask_fixed)
rows.append(dict(network='Fixed initial-training network', density=dens_fixed, MSE=mseF, MAE=maeF, DM=dmF, DM_p=pF))
print(f'fixed: density={dens_fixed:.4f} MSE={mseF:.4e} DM={dmF:.3f}', flush=True)
# sector
mseS,maeS,dmS,pS=fit_eval(sector_mask)
rows.append(dict(network='Sector network', density=sector_mask.sum()/(K*(K-1)), MSE=mseS, MAE=maeS, DM=dmS, DM_p=pS))
print(f'sector: MSE={mseS:.4e} DM={dmS:.3f}', flush=True)
# random draws
for kind,gen in [('Density-matched randomized network',density_matched),('Degree-preserving randomized network',degree_preserving)]:
    rng=np.random.default_rng(42); mse_list=[]; mae_list=[]; dm_list=[]; sig=0
    for d in range(20):
        m=gen(real_mask,rng); mse_d,mae_d,dm_d,p_d=fit_eval(m)
        mse_list.append(mse_d); mae_list.append(mae_d); dm_list.append(dm_d); sig+=int(p_d<0.05)
        if (d+1)%5==0: print(f'  {kind} {d+1}/20 done', flush=True)
    rows.append(dict(network=kind, density=real_dens, MSE=float(np.mean(mse_list)), MSE_sd=float(np.std(mse_list,ddof=1)), MAE=float(np.mean(mae_list)), MAE_sd=float(np.std(mae_list,ddof=1)), DM=float(np.mean(dm_list)), DM_sd=float(np.std(dm_list,ddof=1)), DM_sig_frac=sig/20))

df=pd.DataFrame(rows)
df.to_csv(OUT/'TableC2_network_controls.csv', index=False)
print('\n'+df.to_string())
print('\nSaved: TableC2_network_controls.csv')
