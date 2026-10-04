"""Evaluate v7 (end-to-end + prediction-accuracy anchor) at 1/5/10 epochs."""
import sys, time
import numpy as np
from pathlib import Path
import importlib.util, torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v7",str(HERE/"train_e2e_v7.py"))
v7=importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)
Xs,Y,R,cov,net,mean,std,n_train,n_val,test_start=v7.load_all()
hp={"rank":8,"a_risk":1.0,"a_turn":1.0,"a_smooth":0.3,"a_anchor":1.0,"a_l1":1.0,"a_l2":1e-4,
    "lam1":1e-5,"lam2":1e-4,"lam3":5e-5,"wd":1e-6,"lr":1e-3,"box":None}
for ep in (1,5,10):
    torch.manual_seed(0); np.random.seed(0)
    t0=time.time(); model=v7.train_final(Xs,Y,R,cov,net,test_start,hp,ep); model.eval()
    with torch.no_grad():
        W=model(torch.from_numpy(Xs[test_start:])).numpy()
    np.save(HERE/f"e2e_v7_ep{ep}_test_weights.npy",W)
    print(f"v7 ep{ep}: test MSE={float(((W-Y[test_start:])**2).mean()):.4e} time={time.time()-t0:.1f}s")
