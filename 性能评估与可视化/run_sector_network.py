import os, warnings, importlib.util, numpy as np, pandas as pd
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
warnings.filterwarnings('ignore')
from pathlib import Path
P=Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
spec=importlib.util.spec_from_file_location('vm', P/'VARX'/'VAR及拓展（table2）.py')
vm=importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)

sector_csv = P/'数据'/'sector_labels.csv'
df = pd.read_csv(sector_csv)
print('sector csv rows', len(df), 'cols', list(df.columns))
sectors = df['sector'].tolist()
K=392
assert len(sectors)==K
# build sector mask: same-sector connected (exclude self)
sarr = np.array(sectors)
mask = (sarr[:,None]==sarr[None,:]).astype(float)
np.fill_diagonal(mask,0)
dens = mask.sum()/(K*(K-1))
print('sector mask density', round(dens,4), 'sectors', sorted(set(sectors)))

X=np.load(P/'特征工程'/'X_features.npy'); Y=np.load(P/'特征工程'/'Y_targets.npy')
n=len(X); ntr=int(0.7*n); nval=int(0.15*n); test_start=ntr+nval
Xtr,Ytr = X[:ntr], Y[:ntr]
Xte,Yte = X[test_start:], Y[test_start:]
Y_true = Yte[65:265]
Y_sparse = np.load(P/'VARX'/'Y_pred_model3.npy')[65:265]

fitted = vm.fit_model(4, Xtr, Ytr, mask, n_jobs=4)
Yp = vm.predict_model(Xte, fitted)
Ype = Yp[65:265]
mse = vm.compute_mse(Ype, Y_true); mae = vm.compute_mae(Ype, Y_true)
dm,p,lp = vm.dm_test(Ype, Y_sparse, Y_true)
print(f'Sector network: density={dens:.4f} MSE={mse:.6e} MAE={mae:.6e} DM={dm:+.3f} p={p:.3e}')
# save sector prediction
np.save(P/'VARX'/'Y_pred_model4_sector.npy', Yp)
# update CSV
csv=P/'性能评估与可视化'/'TableC2_network_controls.csv'
d = pd.read_csv(csv)
# remove existing sector row if any
d = d[d['network']!='Sector network']
d = pd.concat([d, pd.DataFrame([dict(network='Sector network', density=dens, MSE=mse, MAE=mae, DM=dm, DM_p=p)])], ignore_index=True)
d.to_csv(csv, index=False)
print(d.to_string())
