import sys, time, importlib.util, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
spec = importlib.util.spec_from_file_location('table2', r'D:\HuaweiMoveData\Users\27438\Desktop\大创\VARX\VAR及拓展（table2）.py')
m = importlib.util.module_from_spec(spec)
sys.modules['table2'] = m
spec.loader.exec_module(m)

base = r'D:\HuaweiMoveData\Users\27438\Desktop\大创'
data = m.load_data()
X, Y, Abar = data['X'], data['Y'], data['A_bar']
sp = m.split_data(X, Y, Abar)
Xtr, Ytr, Atr = sp['train']
Xte, Yte, Ate = sp['test']
mask, dens = m.build_network_mask(Atr)
print('density', dens, 'train', Xtr.shape, 'test', Xte.shape, flush=True)
t0 = time.time()
fitted = m.fit_model(5, Xtr, Ytr, mask, n_jobs=4)
print('fit time', time.time()-t0, flush=True)
P = m.predict_model(Xte, fitted)
Yte_eval = Yte[65:265]; Pe = P[65:265]
mse = float(np.mean((Pe-Yte_eval)**2)); mae = float(np.mean(np.abs(Pe-Yte_eval)))
print(f'REFIT model5 MSE={mse:.6e} MAE={mae:.6e}', flush=True)
np.save(r'D:\HuaweiMoveData\Users\27438\Desktop\大创\VARX\Y_pred_model5_refit.npy', P)
print('saved refit predictions')
