"""v12: full W warm-started from a rank-392 (full-rank) factored solution,
with adaptive turnover penalty during decision-focused fine-tuning."""
import sys, time, copy
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util, torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v10",str(HERE/"train_e2e_v10_full.py")); v10=importlib.util.module_from_spec(spec); spec.loader.exec_module(v10)
spec3=importlib.util.spec_from_file_location("t3",str(HERE.parent/"性能评估与可视化"/"Table3_投资组合表现.py")); t3=importlib.util.module_from_spec(spec3); sys.modules["t3"]=t3; spec3.loader.exec_module(t3)

Xs,Y,R,cov,net,mean,std,n_train,n_val,test_start=v10.load_all()
valid=np.load(HERE.parent/"特征工程"/"valid_indices.npy")
npy=sorted([f for f in (HERE.parent/"数据"/"1min_log_return_npy").iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
files_full=[npy[i] for i in valid[test_start:]]; files_200=files_full[65:265]; Yte=Y[test_start:]

# --- step 1: rank-392 (full-rank factored) solution on train as warm start ---
hp_r = {"full":False,"rank":392,"a_risk":1.0,"a_turn":4.0,"a_smooth":0.3,"a_anchor":1.0,
        "a_gross":1.0,"gross_cap":1.5,"a_l1":1.0,"a_l2":1e-4,"lam1":1e-5,"lam2":1e-4,"lam3":5e-5,
        "wd":1e-6,"lr":1e-3,"clip":0.1,"box":None}
print("--- warm start: rank-392 factored on train ---", flush=True)
torch.manual_seed(0); np.random.seed(0)
m_r, best_r, _ = v10.train(Xs,Y,R,cov,net,n_train,n_val,test_start,hp_r,epochs=2,patience=2,log="ws ")
W_init = (m_r.U @ m_r.V).detach().numpy().copy()
b_init = m_r.b.detach().numpy().copy()
z_ws = Xs[test_start:] @ W_init.T + b_init
w_ws = z_ws + (1.0 - z_ws.sum(1, keepdims=True))/392
print("warm-start MSE on test:", float(((w_ws - Yte)**2).mean()), flush=True)

# --- step 2: direct full W fine-tuning with adaptive turnover penalty ---
F=Xs.shape[1]
Xs_t=torch.from_numpy(Xs); Y_t=torch.from_numpy(Y); R_t=torch.from_numpy(R)
penalty=v10.build_penalty(net, F, lam1=1e-5, lam2=1e-4, lam3=5e-5)
model=v10.FullWForecaster(F)
with torch.no_grad():
    model.W.copy_(torch.from_numpy(W_init).float()); model.b.copy_(torch.from_numpy(b_init).float())
opt=torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
sc0=v10.data_scales(Y,cov,np.arange(0,n_train))
start_pool=np.arange(0,n_train-v10.L_SEQ+1)
a_turn=6.0
best={"score":-1e9,"state":None,"epoch":-1,"sharpe":None,"to":None,"mse":None}
rng=np.random.default_rng(42)
for ep in range(8):
    hp=dict(hp_r); hp.update({"full":True,"a_turn":a_turn,"lr":1e-5,"clip":0.01})
    perm=rng.permutation(start_pool); t0=time.time()
    for st in range(0,len(perm)-v10.B_SEQ+1,v10.B_SEQ):
        starts=perm[st:st+v10.B_SEQ]
        loss,risk,turn,anch,l1,smooth=v10.seq_loss(model,Xs_t,Y_t,R_t,cov,starts,sc0,hp,penalty)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip"]); opt.step()
    sharpe,to_val,mse_val,gross=v10.val_metrics(model,Xs_t,Y_t,R_t,cov,np.arange(n_train,test_start))
    score=sharpe-10*max(0.0,to_val-0.10)-10*max(0.0,mse_val/3e-5-1.0)-10*max(0.0,gross-1.5)
    if score>best["score"]:
        best.update({"score":score,"state":{k:v.detach().clone() for k,v in model.state_dict().items()},
                     "epoch":ep,"sharpe":sharpe,"to":to_val,"mse":mse_val}); 
    print(f"ep{ep} a_turn={a_turn:.2f} loss={float(loss):.3f} val_sharpe={sharpe:.3f} "
          f"val_to={to_val:.3f} val_mse={mse_val:.2e} gross={gross:.3f} score={score:.3f} ({time.time()-t0:.1f}s)", flush=True)
    if to_val>0.10: a_turn=min(a_turn*1.5,60.0)
    elif to_val<0.05: a_turn=max(a_turn*0.8,1.0)

model.load_state_dict(best["state"]); model.eval()
with torch.no_grad(): W=model(torch.from_numpy(Xs[test_start:])).numpy()
np.save(HERE/"e2e_v12_adaptive_test_weights.npy",W)
print("v12 best epoch",best["epoch"],"val",best["sharpe"],best["to"],best["mse"])
def ev(W,files,sl):
    r=t3.compute_all({1:{"name":"v12","Y_pred":W[sl]}},files)[1]
    return {"sharpe":r["sharpe_net"],"turnover":r["avg_turnover"],"vol":r["vol_annual"],"maxDD":r["max_dd"],"cum":r["cum_ret"],"MSE":float(((W[sl]-Yte[sl])**2).mean()),"gross":float(np.abs(W[sl]).sum(1).mean())}
m200=ev(W,files_200,slice(65,265)); mfull=ev(W,files_full,slice(0,363))
print("v12 200d:",{k:round(v,5) for k,v in m200.items()})
print("v12 full:",{k:round(v,5) for k,v in mfull.items()})
pd.DataFrame([{"window":"200d",**m200},{"window":"full363",**mfull}]).to_csv(HERE/"e2e_v12_metrics.csv",index=False)
