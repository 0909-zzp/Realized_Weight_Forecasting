import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, importlib.util, time
import numpy as np
from pathlib import Path

ROOT = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
sys.path.insert(0, str(ROOT / "图形Lasso" / "code"))

# load the VARX module
spec = importlib.util.spec_from_file_location("vp", str(ROOT / "VARX" / "VAR及拓展（table2）.py"))
vp = importlib.util.module_from_spec(spec); sys.modules["vp"] = vp
spec.loader.exec_module(vp)

X = np.load(ROOT / "特征工程" / "X_features.npy")
Y = np.load(ROOT / "特征工程" / "Y_targets.npy")
A_idio = np.load(ROOT / "特征工程" / "A_bar_idio.npy")

n = len(X); ntr = int(0.7 * n); nval = int(0.15 * n); ts = ntr + nval
print("n =", n, "ntr =", ntr, "nval =", nval, "test =", n - ts)

Xtr, Ytr = X[:ntr], Y[:ntr]
Xte, Yte = X[ts:], Y[ts:]

# idio network mask: training-window mean >= 0.117, zero diagonal
THR = 0.117
mask = (A_idio[:ntr].mean(0) >= THR).astype(float)
np.fill_diagonal(mask, 0)
density = mask.sum() / (vp.K * (vp.K - 1))
deg = mask.sum(1).mean()
print(f"idio mask: density={density:.4f}  mean degree={deg:.1f}")

valid_indices = np.load(ROOT / "特征工程" / "valid_indices.npy")
train_day_indices = valid_indices[:ntr]

for mid in (4, 5):
    t0 = time.time()
    fitted = vp.fit_model(mid, Xtr, Ytr, mask, n_jobs=4)
    Yp = vp.predict_model(Xte, fitted)
    out = ROOT / "VARX" / f"Y_pred_model{mid}_idio.npy"
    np.save(out, Yp)
    # quick eval on 200-day window
    Ype = Yp[65:265]
    mse = float(np.mean((Ype - Yte[65:265]) ** 2))
    mae = float(np.mean(np.abs(Ype - Yte[65:265])))
    print(f"model{mid} fitted in {time.time()-t0:.0f}s  MSE={mse:.6e}  MAE={mae:.6e}  saved {out.name}")

# DFL on idio model5
Y_m5 = np.load(ROOT / "VARX" / "Y_pred_model5_idio.npy")
t0 = time.time()
Y_dfl = vp.compute_model6_drift_l1(
    Y_m5, Yte, train_day_indices,
    eta=1e-6, rho=1e-3, risk_mult=1.0, rolling_cov=True, cov_window=150,
)
out = ROOT / "VARX" / "Y_pred_model6_opt_idio.npy"
np.save(out, Y_dfl)
print(f"model6 DFL done in {time.time()-t0:.0f}s  saved {out.name}")
