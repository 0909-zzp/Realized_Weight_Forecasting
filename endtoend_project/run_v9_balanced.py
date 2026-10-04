"""Train the v9 balanced configuration selected by the constraint-aware score."""
import sys, time
import numpy as np
from pathlib import Path
import importlib.util, torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v9",str(HERE/"train_e2e_v9.py"))
v9=importlib.util.module_from_spec(spec); spec.loader.exec_module(v9)
Xs,Y,R,cov,net,mean,std,n_train,n_val,test_start=v9.load_all()
hp={"rank":8,"a_risk":1.0,"a_turn":4.0,"a_smooth":0.3,"a_anchor":1.0,"a_gross":1.0,"gross_cap":1.5,
    "a_l1":1.0,"a_l2":1e-4,"lam1":1e-5,"lam2":1e-4,"lam3":5e-5,"wd":1e-6,"lr":1e-3,"box":None}
torch.manual_seed(0); np.random.seed(0)
t0=time.time()
model=v9.train_final(Xs,Y,R,cov,net,test_start,hp,1)
model.eval()
with torch.no_grad():
    W=model(torch.from_numpy(Xs[test_start:])).numpy()
np.save(HERE/"e2e_v9_balanced_test_weights.npy",W)
print(f"v9 balanced: test MSE={float(((W-Y[test_start:])**2).mean()):.4e} "
      f"gross={float(np.abs(W).sum(1).mean()):.3f} time={time.time()-t0:.1f}s")
