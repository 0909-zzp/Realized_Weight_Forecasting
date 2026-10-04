"""v5: low-rank end-to-end with an effective step size (lr=1e-3)."""
import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("v4", str(Path(__file__).resolve().parent/"train_e2e_v4.py"))
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)

Xs, Y, R, cov, z0, n_train, n_val, test_start = v4.load_all()
train_idx = np.arange(0, n_train); val_idx = np.arange(n_train, test_start)
Wb = z0[train_idx]; sig = np.asarray(cov[np.minimum(train_idx+1, len(Xs)-1)], dtype=np.float64)
sc0 = {"risk0": float(np.einsum("bi,bij,bj->b", Wb, sig, Wb).mean()),
       "anchor_scale": float(((Wb-Y[train_idx])**2).mean()),
       "turn0": float(np.mean(np.abs(Wb[1:]-Wb[:-1]).sum(axis=1))),
       "smooth0": float(np.mean(((Wb[1:]-Wb[:-1])**2).sum(axis=1)))}
print("M5 val baseline:", v4.val_stats(np.zeros((v4.K,v4.RANK)), np.zeros((v4.RANK,Xs.shape[1])), Xs, z0, Y, R, val_idx))
hp = {"alpha_risk":1.0,"alpha_turn":0.3,"alpha_smooth":0.1,"anchor_start":5.0,"anchor_end":0.5,
      "lr":1e-3,"batch":64,"beta1":0.9,"beta2":0.999,"clip":0.1,"v_init":1e-3}
t0=time.time()
best = v4.train(Xs, Y, R, cov, z0, z0, train_idx, val_idx, hp, sc0, epochs=30, patience=8)
print("v5 best:", {k: best[k] for k in ("val" if False else "sharpe","to","mse","epoch")}, "time", round(time.time()-t0,1))
epochs_final = best["epoch"]+1
all_train = np.arange(0, test_start)
bf = v4.train(Xs, Y, R, cov, z0, z0, all_train, all_train[-50:], hp, sc0, epochs=epochs_final, patience=10**9)
U, V = bf["U_final"], bf["V_final"]
Wte, _ = v4.forward(z0[test_start:], Xs[test_start:], U, V)
np.save(Path(__file__).resolve().parent/"e2e_v5_test_weights.npy", Wte)
print("v5 test MSE:", float(((Wte-Y[test_start:])**2).mean()))
