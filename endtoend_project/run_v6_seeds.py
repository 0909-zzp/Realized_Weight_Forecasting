"""Seed robustness for v6 from-scratch end-to-end (1 epoch)."""
import sys, time
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v6", str(HERE/"train_e2e_v6.py"))
v6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v6)
spec3 = importlib.util.spec_from_file_location("t3", str(HERE.parent/"性能评估与可视化"/"Table3_投资组合表现.py"))
t3 = importlib.util.module_from_spec(spec3); sys.modules["t3"]=t3; spec3.loader.exec_module(t3)

Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start = v6.load_all()
valid = np.load(HERE.parent/"特征工程"/"valid_indices.npy")
npy = sorted([f for f in (HERE.parent/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
files_full=[npy[i] for i in valid[test_start:]]; files_200=files_full[65:265]
Yte=Y[test_start:]
hp = {"rank":8,"a_risk":1.0,"a_turn":1.0,"a_smooth":0.3,"a_l1":1.0,"a_l2":1e-4,
      "lam1":1e-5,"lam2":1e-4,"lam3":5e-5,"wd":1e-6,"lr":1e-3,"box":None}
rows=[]
for seed in (0,1,2,3,4):
    torch.manual_seed(seed); np.random.seed(seed)
    model = v6.train_final(Xs, Y, R, cov, net, test_start, hp, 1)
    model.eval()
    with torch.no_grad():
        W = model(torch.from_numpy(Xs[test_start:])).numpy()
    for tag, files, sl in [("full_363",files_full,slice(0,363)),("window_200",files_200,slice(65,265))]:
        models={1:{"name":"v6","Y_pred":W[sl]}}
        res=t3.compute_all(models, files)[1]
        rows.append({"seed":seed,"window":tag,"sharpe":res["sharpe_net"],"turnover":res["avg_turnover"],
                     "vol":res["vol_annual"],"maxDD":res["max_dd"],"cum":res["cum_ret"],
                     "MSE":float(((W[sl]-Yte[sl])**2).mean()),"gross":float(np.abs(W[sl]).sum(1).mean())})
    print("seed",seed,"done",flush=True)
df=pd.DataFrame(rows); df.to_csv(HERE/"e2e_v6_seeds.csv",index=False)
for tag in ("window_200","full_363"):
    d=df[df.window==tag]
    print(f"\n--- {tag} (5 seeds) ---")
    print(d[["seed","sharpe","turnover","vol","maxDD","cum","MSE","gross"]].to_string(index=False))
    print("mean:", d[["sharpe","turnover","vol","maxDD","cum","MSE","gross"]].mean().round(4).to_dict())
    print("std :", d[["sharpe","turnover","vol","maxDD","cum","MSE","gross"]].std().round(4).to_dict())
