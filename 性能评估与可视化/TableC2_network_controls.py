import os, time, warnings, importlib.util, numpy as np, pandas as pd
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
from pathlib import Path
import networkx as nx

P = Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
OUT = P / '性能评估与可视化'

spec = importlib.util.spec_from_file_location('vm', P/'VARX'/'VAR及拓展（table2）.py')
vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)

X = np.load(P/'特征工程'/'X_features.npy')
Y = np.load(P/'特征工程'/'Y_targets.npy')
A = np.load(P/'特征工程'/'A_bar.npy')
K = X.shape[1] - 9  # lagged features + 9 exog -> K = P_LAGS*K actually; use shared K
K = 392
n = len(X); ntr = int(0.7*n); nval = int(0.15*n); test_start = ntr + nval
Xtr, Ytr, Atr = X[:ntr], Y[:ntr], A[:ntr]
Xte, Yte = X[test_start:], Y[test_start:]
Y_true = Yte[65:265]

Y_sparse = np.load(P/'VARX'/'Y_pred_model3.npy')          # uniform penalty (Sparse VARX)
Y_sparse_eval = Y_sparse[65:265]

real_mask, real_dens = vm.build_network_mask(Atr)

def fit_eval(mask):
    fitted = vm.fit_model(4, Xtr, Ytr, mask, n_jobs=4)
    Yp = vm.predict_model(Xte, fitted)
    Ype = Yp[65:265]
    mse = vm.compute_mse(Ype, Y_true)
    mae = vm.compute_mae(Ype, Y_true)
    dm, p, lp = vm.dm_test(Ype, Y_sparse_eval, Y_true)
    return mse, mae, dm, p

# ---- random mask generators ----
def density_matched_mask(ref_mask, rng):
    E = int(ref_mask.sum() // 2)
    iu = np.triu_indices(K, k=1)
    npair = len(iu[0])
    chosen = rng.choice(npair, size=E, replace=False)
    m = np.zeros((K, K))
    m[iu[0][chosen], iu[1][chosen]] = 1.0
    m[iu[1][chosen], iu[0][chosen]] = 1.0
    return m

def degree_preserving_mask(ref_mask, rng, nswap=None, max_tries=200):
    G = nx.from_numpy_array(ref_mask)
    E = G.number_of_edges()
    nswap = nswap if nswap is not None else E * 5
    # convert rng to seed for networkx
    seed = int(rng.integers(0, 2**31 - 1))
    G2 = nx.double_edge_swap(G, nswap=nswap, max_tries=max_tries*nswap, seed=seed)
    return nx.to_numpy_array(G2)

# ---- evaluate fixed rows ----
rows = []

def row(name, density, mse, mae, dm, p):
    return dict(network=name, density=density, MSE=mse, MAE=mae, DM=dm, DM_p=p)

# uniform (no network)
mse0 = vm.compute_mse(Y_sparse_eval, Y_true)
mae0 = vm.compute_mae(Y_sparse_eval, Y_true)
rows.append(row('Uniform cross-asset penalty', np.nan, mse0, mae0, 0.0, np.nan))

# realized network
mseR = vm.compute_mse(np.load(P/'VARX'/'Y_pred_model4.npy')[65:265], Y_true)
maeR = vm.compute_mae(np.load(P/'VARX'/'Y_pred_model4.npy')[65:265], Y_true)
dmR, pR, _ = vm.dm_test(np.load(P/'VARX'/'Y_pred_model4.npy')[65:265], Y_sparse_eval, Y_true)
rows.append(row('Realized forecasting network', real_dens, mseR, maeR, dmR, pR))

# fixed initial-training network: first 500 training days
mask_fixed, dens_fixed = vm.build_network_mask(Atr[:500])
mseF, maeF, dmF, pF = fit_eval(mask_fixed)
rows.append(row('Fixed initial-training network', dens_fixed, mseF, maeF, dmF, pF))
print('fixed rows done', flush=True)

# ---- random draws ----
N_DRAW = 20
for kind, gen in [('Density-matched randomized network', density_matched_mask),
                  ('Degree-preserving randomized network', degree_preserving_mask)]:
    rng = np.random.default_rng(42)
    mse_list, mae_list, dm_list, sig = [], [], [], 0
    for d in range(N_DRAW):
        if kind.startswith('Density'):
            m = gen(real_mask, rng)
        else:
            m = gen(real_mask, rng)
        mse_d, mae_d, dm_d, p_d = fit_eval(m)
        mse_list.append(mse_d); mae_list.append(mae_d); dm_list.append(dm_d)
        sig += int(p_d < 0.05)
        print(f'  {kind} draw {d+1}/{N_DRAW}: MSE={mse_d:.4e} MAE={mae_d:.4e} DM={dm_d:+.3f} p={p_d:.3f}', flush=True)
    rows.append(dict(network=kind, density=real_dens,
                     MSE=float(np.mean(mse_list)), MSE_sd=float(np.std(mse_list, ddof=1)),
                     MAE=float(np.mean(mae_list)), MAE_sd=float(np.std(mae_list, ddof=1)),
                     DM=float(np.mean(dm_list)), DM_sd=float(np.std(dm_list, ddof=1)),
                     DM_sig_frac=sig/N_DRAW))

df = pd.DataFrame(rows)
df.to_csv(OUT/'TableC2_network_controls.csv', index=False)
print()
print(df.to_string())
print('\nSaved:', OUT/'TableC2_network_controls.csv')
