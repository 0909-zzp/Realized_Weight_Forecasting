"""v10: direct full-W end-to-end with nested time-series validation."""
import sys, time, json
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util, torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v10", str(HERE/"train_e2e_v10_full.py"))
v10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v10)
spec3 = importlib.util.spec_from_file_location("t3", str(HERE.parent/"性能评估与可视化"/"Table3_投资组合表现.py"))
t3 = importlib.util.module_from_spec(spec3); sys.modules["t3"]=t3; spec3.loader.exec_module(t3)

Xs, Y, R, cov, net, mean, std, n_train, n_val, test_start = v10.load_all()
valid = np.load(HERE.parent/"特征工程"/"valid_indices.npy")
npy = sorted([f for f in (HERE.parent/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
files_full=[npy[i] for i in valid[test_start:]]; files_200=files_full[65:265]
Yte=Y[test_start:]

base = {"full":True, "a_risk":1.0, "a_smooth":0.3, "a_gross":1.0, "gross_cap":1.5,
        "a_l1":1.0, "a_l2":1e-4, "lam1":1e-5, "lam2":1e-4, "lam3":5e-5,
        "wd":1e-6, "lr":1e-5, "clip":0.01, "box":None, "rank":8}
configs = [
    dict(base, a_turn=4.0, a_anchor=1.0),
    dict(base, a_turn=6.0, a_anchor=1.5),
    dict(base, a_turn=6.0, a_anchor=2.0),
]
folds = [(1152,1452), (1452,1752), (1752,2052)]   # (train_end, val_end)

def score(sharpe, to, mse):
    return sharpe - 10*max(0.0, to-0.10) - 10*max(0.0, mse/3.0e-5-1.0)

rows=[]
for ci, hp in enumerate(configs):
    print(f"\n===== v10 config {ci+1}/{len(configs)}: turn={hp['a_turn']} anchor={hp['a_anchor']} =====")
    fold_scores=[]; fold_epochs=[]
    for fi, (tr_end, va_end) in enumerate(folds):
        torch.manual_seed(0); np.random.seed(0)
        model, best, sc0 = v10.train(Xs,Y,R,cov,net,tr_end,va_end-tr_end,va_end,hp,
                                     epochs=4, patience=2, log=f"c{ci+1}f{fi+1} ")
        s = score(best["sharpe"], best["to"], best["mse"])
        fold_scores.append(s); fold_epochs.append(best["epoch"])
        rows.append({"config":ci+1,"a_turn":hp["a_turn"],"a_anchor":hp["a_anchor"],
                     "fold":fi+1,"train_end":tr_end,"val_end":va_end,
                     "val_sharpe":best["sharpe"],"val_turnover":best["to"],
                     "val_mse":best["mse"],"score":s,"best_epoch":best["epoch"]})
        print(f"  fold{fi+1}: val_sharpe={best['sharpe']:.3f} to={best['to']:.3f} "
              f"mse={best['mse']:.2e} score={s:.3f}", flush=True)
    rows.append({"config":ci+1,"a_turn":hp["a_turn"],"a_anchor":hp["a_anchor"],"fold":"avg",
                 "val_sharpe":np.mean([r["val_sharpe"] for r in rows[-3:]]),
                 "val_turnover":np.mean([r["val_turnover"] for r in rows[-3:]]),
                 "val_mse":np.mean([r["val_mse"] for r in rows[-3:]]),
                 "score":float(np.mean(fold_scores)),
                 "best_epoch":int(np.median(fold_epochs))})
    print("config avg score:", float(np.mean(fold_scores)), flush=True)

df=pd.DataFrame(rows); df.to_csv(HERE/"e2e_v10_nested_folds.csv", index=False)
avg=df[df.fold=="avg"].sort_values("score", ascending=False)
print("\n=== nested CV average ===")
print(avg.to_string(index=False))
best_cfg_i = int(avg.iloc[0]["config"])-1
hp = configs[best_cfg_i]
epochs_final = max(1, int(avg.iloc[0]["best_epoch"])+1)
print(f"\n=== final full-W training on train+val: config {best_cfg_i+1}, epochs={epochs_final} ===")
torch.manual_seed(0); np.random.seed(0)
model = v10.train_final(Xs,Y,R,cov,net,test_start,hp,epochs_final); model.eval()
with torch.no_grad():
    W = model(torch.from_numpy(Xs[test_start:])).numpy()
np.save(HERE/"e2e_v10_test_weights.npy", W)
print("v10 test MSE:", float(((W-Yte)**2).mean()), "gross:", float(np.abs(W).sum(1).mean()))

def eval_w(W, files, sl):
    r=t3.compute_all({1:{"name":"v10","Y_pred":W[sl]}}, files)[1]
    return {"sharpe":r["sharpe_net"],"turnover":r["avg_turnover"],"vol":r["vol_annual"],
            "maxDD":r["max_dd"],"cum":r["cum_ret"],
            "MSE":float(((W[sl]-Yte[sl])**2).mean()),"gross":float(np.abs(W[sl]).sum(1).mean())}
m200=eval_w(W,files_200,slice(65,265)); mfull=eval_w(W,files_full,slice(0,363))
print("v10 200d:", {k:round(v,5) for k,v in m200.items()})
print("v10 full:", {k:round(v,5) for k,v in mfull.items()})
pd.DataFrame([{"window":"200d",**m200},{"window":"full363",**mfull}]).to_csv(HERE/"e2e_v10_final_metrics.csv",index=False)
