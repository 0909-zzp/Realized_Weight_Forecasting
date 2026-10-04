"""Evaluate v6 (from-scratch end-to-end) at 1 epoch and 10 epochs."""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v6", str(HERE/"train_e2e_v6.py"))
v6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v6)

Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start = v6.load_all()
hp = {"rank":8,"a_risk":1.0,"a_turn":1.0,"a_smooth":0.3,"a_l1":1.0,"a_l2":1e-4,
      "lam1":1e-5,"lam2":1e-4,"lam3":5e-5,"wd":1e-6,"lr":1e-3,"box":None}
for ep in (1, 10):
    t0=time.time()
    model = v6.train_final(Xs, Y, R, cov, net, test_start, hp, ep)
    model.eval()
    import torch
    with torch.no_grad():
        Wte = model(torch.from_numpy(Xs[test_start:])).numpy()
    np.save(HERE/f"e2e_v6_ep{ep}_test_weights.npy", Wte)
    print(f"v6 ep{ep}: test MSE={float(((Wte-Y[test_start:])**2).mean()):.4e} time={time.time()-t0:.1f}s")
