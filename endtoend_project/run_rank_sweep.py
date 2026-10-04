"""Rank sweep under the v9 objective and constraints.

ranks: 4, 8, 16, 32, 64, 392 (392 = full-rank linear map).
Same protocol for every rank: 12 epochs, validation early stopping with the
constraint-aware score, then retrain on train+val for the selected epochs and
evaluate the test period.
"""
import sys, time, json
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util, torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v9", str(HERE/"train_e2e_v9.py"))
v9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v9)
spec3 = importlib.util.spec_from_file_location("t3", str(HERE.parent/"性能评估与可视化"/"Table3_投资组合表现.py"))
t3 = importlib.util.module_from_spec(spec3); sys.modules["t3"]=t3; spec3.loader.exec_module(t3)

Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start = v9.load_all()
valid = np.load(HERE.parent/"特征工程"/"valid_indices.npy")
npy = sorted([f for f in (HERE.parent/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
files_full=[npy[i] for i in valid[test_start:]]; files_200=files_full[65:265]
Yte=Y[test_start:]

base = {"a_risk":1.0,"a_turn":4.0,"a_smooth":0.3,"a_anchor":1.0,"a_gross":1.0,"gross_cap":1.5,
        "a_l1":1.0,"a_l2":1e-4,"lam1":1e-5,"lam2":1e-4,"lam3":5e-5,"wd":1e-6,"lr":1e-3,"box":None}

def eval_weights(W, files, sl):
    r = t3.compute_all({1:{"name":"m","Y_pred":W[sl]}}, files)[1]
    return dict(sharpe=r["sharpe_net"], turnover=r["avg_turnover"], vol=r["vol_annual"],
                maxDD=r["max_dd"], cum=r["cum_ret"],
                MSE=float(((W[sl]-Yte[sl])**2).mean()), gross=float(np.abs(W[sl]).sum(1).mean()))

rows=[]
for rank in (4,8,16,32,64,392):
    hp = dict(base); hp["rank"]=rank
    print(f"\n===== rank {rank} =====", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    model, best, sc0 = v9.train(Xs,Y,R,cov,net,n_train,n_val,test_start,hp,epochs=12,patience=5,log=f"r{rank} ")
    print(f"rank {rank} best epoch={best['epoch']} val_sharpe={best['sharpe']:.3f} "
          f"val_to={best['to']:.3f} val_mse={best['mse']:.2e}", flush=True)
    epochs_final = best["epoch"]+1
    torch.manual_seed(0); np.random.seed(0)
    model = v9.train_final(Xs,Y,R,cov,net,test_start,hp,epochs_final); model.eval()
    with torch.no_grad():
        W = model(torch.from_numpy(Xs[test_start:])).numpy()
    np.save(HERE/f"e2e_rank{rank}_test_weights.npy", W)
    m200 = eval_weights(W, files_200, slice(65,265))
    mfull = eval_weights(W, files_full, slice(0,363))
    rows.append({"rank":rank, "best_epoch":best["epoch"], "val_sharpe":best["sharpe"],
                 "val_turnover":best["to"], "val_mse":best["mse"],
                 **{f"200_{k}":v for k,v in m200.items()},
                 **{f"full_{k}":v for k,v in mfull.items()}})
    print("rank",rank,"200d:",{k:round(v,4) for k,v in m200.items()}, flush=True)
    print("rank",rank,"full:",{k:round(v,4) for k,v in mfull.items()}, flush=True)
df=pd.DataFrame(rows); df.to_csv(HERE/"e2e_rank_sweep.csv",index=False)
print("\n===== rank sweep summary =====")
cols=["rank","best_epoch","val_sharpe","val_turnover","val_mse","200_sharpe","200_turnover","200_MSE","200_gross","full_sharpe","full_turnover","full_MSE","full_gross"]
print(df[cols].round(4).to_string(index=False))
