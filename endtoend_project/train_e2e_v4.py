"""End-to-end v4: low-rank residual directly in the M5 output space.

z = z0 + (Xs @ V.T) @ U.T,   w = z   (no renormalization, no box)
z0 is the M5 forecast (already sums to one exactly), so the model starts at M5.
Only U (K x r) and V (r x F) are trained on the decision-focused loss.
Loss terms are normalized by their M5-initialization values; the anchor to M5
decays over training.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
K = 392
ETA_COST = 1e-4
EPS = 1e-6
RANK = 8


def load_all():
    import importlib.util
    X = np.load(PROJ/"特征工程"/"X_features.npy").astype(np.float64)
    Y = np.load(PROJ/"特征工程"/"Y_targets.npy").astype(np.float64)
    A_bar = np.load(PROJ/"特征工程"/"A_bar.npy")
    R = np.load(CACHE/"simple_returns.npy").astype(np.float64)
    cov = np.load(CACHE/"cov_cache.npy", mmap_mode="r")
    n = len(X); n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train+n_val
    # Recompute the final M5 (current lambda_s=3e-3) inside this project, read-only.
    spec = importlib.util.spec_from_file_location("vp", str(PROJ/"VARX"/"VARX".replace("VARX","VAR及拓展（table2）.py")))
    spec = importlib.util.spec_from_file_location("vp", str(PROJ/"VARX"/"VAR及拓展（table2）.py"))
    vp = importlib.util.module_from_spec(spec); sys.modules["vp"] = vp; spec.loader.exec_module(vp)
    mask, _ = vp.build_network_mask(A_bar[:n_train])
    fitted = vp.fit_model(5, X[:n_train], Y[:n_train], mask, n_jobs=4)
    z0 = vp.predict_model(X, fitted)
    scaler = fitted["scaler"]
    Xs = (X - scaler.mean_.astype(np.float64))/scaler.scale_.astype(np.float64)
    return Xs, Y, R, cov, z0, n_train, n_val, test_start


def forward(z0b, Xb, U, V):
    H = Xb @ V.T
    return z0b + H @ U.T, H


def drift(w_prev, r):
    if w_prev.ndim == 1:
        return w_prev*(1+r)/(1+w_prev@r)
    denom = 1 + np.einsum("bi,bi->b", w_prev, r)
    return w_prev*(1+r)/denom[:, None]


def soft_abs(x):
    return np.sqrt(x*x+EPS*EPS)-EPS


def clip_grad(g, clip=0.1):
    n = np.linalg.norm(g)
    return g*(clip/(n+1e-12)) if n > clip else g


def batch_loss_grad(z0b, Xb, Ab, Rb, Sb, Wprev, U, V, sc0, hp, alpha_anchor):
    Wo, H = forward(z0b, Xb, U, V)
    risk = np.einsum("bi,bij,bj->b", Wo, Sb, Wo)
    gW = 2.0*np.einsum("bij,bj->bi", Sb, Wo)*hp["alpha_risk"]/sc0["risk0"]
    diff = Wo - Ab
    anchor = np.einsum("bi,bi->b", diff, diff)
    gW += 2.0*diff*alpha_anchor/sc0["anchor_scale"]
    d = Wo - drift(Wprev, Rb)
    turnover = soft_abs(d).sum(axis=1)
    gW += hp["alpha_turn"]*(d/np.sqrt(d*d+EPS*EPS))/sc0["turn0"]
    sm = np.einsum("bi,bi->b", Wo-Wprev, Wo-Wprev)
    gW += 2.0*hp["alpha_smooth"]*(Wo-Wprev)/sc0["smooth0"]
    loss = (hp["alpha_risk"]*risk.mean()/sc0["risk0"] + hp["alpha_turn"]*turnover.mean()/sc0["turn0"]
            + alpha_anchor*anchor.mean()/sc0["anchor_scale"] + hp["alpha_smooth"]*sm.mean()/sc0["smooth0"])
    gH = gW @ U
    gU = gW.T @ H
    gV = gH.T @ Xb
    return loss, risk.mean(), turnover.mean(), anchor.mean(), sm.mean(), gU, gV


def val_stats(U, V, Xs, z0, Y, R, val_idx):
    Wv, _ = forward(z0[val_idx], Xs[val_idx], U, V)
    Rv = R[val_idx]
    T = len(Wv)-1
    gross = np.array([Wv[t] @ Rv[t+1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        to[t] = np.abs(Wv[t+1]-drift(Wv[t], Rv[t+1])).sum()
    net = gross - ETA_COST*to
    sd = net.std(ddof=1)
    sharpe = float(net.mean()/sd*np.sqrt(252)) if sd > 0 else 0.0
    mse = float(((Wv - Y[val_idx])**2).mean())
    return sharpe, float(to.mean()), mse


def train(Xs, Y, R, cov, z0, Bbase, train_idx, val_idx, hp, sc0, epochs=40, patience=10):
    U = np.zeros((K, RANK)); V = hp.get("v_init", 1e-4)*np.random.default_rng(0).normal(size=(RANK, Xs.shape[1]))
    mU=np.zeros_like(U); vU=np.zeros_like(U); mV=np.zeros_like(V); vV=np.zeros_like(V)
    best={"score":-1e9,"U":None,"V":None,"epoch":-1,"sharpe":None,"to":None,"mse":None}
    wait=0; rng=np.random.default_rng(42)
    for ep in range(epochs):
        alpha_anchor = hp["anchor_start"]*(hp["anchor_end"]/hp["anchor_start"])**(ep/max(1,epochs-1))
        perm = rng.permutation(train_idx); t0=time.time()
        for st in range(0, len(perm), hp["batch"]):
            idx=perm[st:st+hp["batch"]]; prev=np.maximum(idx-1,0)
            Wprev,_ = forward(z0[prev], Xs[prev], U, V)
            Sb=np.asarray(cov[np.minimum(idx+1,len(Xs)-1)],dtype=np.float64)
            loss,risk,turn,anch,sm,gU,gV = batch_loss_grad(
                z0[idx], Xs[idx], Bbase[idx], R[idx], Sb, Wprev, U, V, sc0, hp, alpha_anchor)
            gU=clip_grad(gU,hp["clip"]); gV=clip_grad(gV,hp["clip"])
            t=ep*100000+st+1
            mU=hp["beta1"]*mU+(1-hp["beta1"])*gU; vU=hp["beta2"]*vU+(1-hp["beta2"])*gU*gU
            mV=hp["beta1"]*mV+(1-hp["beta1"])*gV; vV=hp["beta2"]*vV+(1-hp["beta2"])*gV*gV
            U-=hp["lr"]*(mU/(1-hp["beta1"]**t))/(np.sqrt(vU/(1-hp["beta2"]**t))+1e-8)
            V-=hp["lr"]*(mV/(1-hp["beta1"]**t))/(np.sqrt(vV/(1-hp["beta2"]**t))+1e-8)
        sharpe,to_val,mse_val = val_stats(U,V,Xs,z0,Y,R,val_idx)
        penalty = 10*max(0.0,to_val-0.35)+5*max(0.0,mse_val/2.46e-5-1.2)
        score = sharpe-penalty
        if score>best["score"]:
            best.update({"score":score,"U":U.copy(),"V":V.copy(),"epoch":ep,
                         "sharpe":sharpe,"to":to_val,"mse":mse_val}); wait=0
        else: wait+=1
        if ep%5==0 or ep==epochs-1:
            print(f"ep{ep:03d} loss={loss:.3f} risk={risk:.2e} turn={turn:.3f} anchor={anch:.2e} "
                  f"val_sharpe={sharpe:.3f} val_to={to_val:.3f} val_mse={mse_val:.2e} "
                  f"a_anch={alpha_anchor:.2f} ({time.time()-t0:.1f}s)", flush=True)
        if wait>=patience:
            print("early stop at",ep); break
    best["U_final"],best["V_final"]=U.copy(),V.copy()
    return best


def main():
    Xs,Y,R,cov,z0,n_train,n_val,test_start = load_all()
    train_idx=np.arange(0,n_train); val_idx=np.arange(n_train,test_start)
    Bbase = z0.copy()   # M5 forecast is exactly the anchor
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(Xs)-test_start} rank={RANK}")
    Wb=Bbase[train_idx]; sig=np.asarray(cov[np.minimum(train_idx+1,len(Xs)-1)],dtype=np.float64)
    sc0={"risk0":float(np.einsum("bi,bij,bj->b",Wb,sig,Wb).mean()),
         "anchor_scale":float(((Wb-Y[train_idx])**2).mean()),
         "turn0":float(np.mean(np.abs(Wb[1:]-Wb[:-1]).sum(axis=1))),
         "smooth0":float(np.mean(((Wb[1:]-Wb[:-1])**2).sum(axis=1)))}
    print("scales:", {k:round(v,8) for k,v in sc0.items()})
    # validation baseline (M5)
    print("M5 val baseline:", val_stats(np.zeros((K,RANK)), np.zeros((RANK,Xs.shape[1])), Xs, z0, Y, R, val_idx))

    grid=[
        {"alpha_risk":0.3,"alpha_turn":0.1,"alpha_smooth":0.1,"anchor_start":10.0,"anchor_end":1.0},
        {"alpha_risk":1.0,"alpha_turn":0.1,"alpha_smooth":0.1,"anchor_start":10.0,"anchor_end":1.0},
        {"alpha_risk":0.3,"alpha_turn":0.3,"alpha_smooth":0.1,"anchor_start":10.0,"anchor_end":1.0},
        {"alpha_risk":1.0,"alpha_turn":0.3,"alpha_smooth":0.1,"anchor_start":5.0,"anchor_end":0.5},
    ]
    for hp in grid:
        hp.update({"lr":1e-4,"batch":64,"beta1":0.9,"beta2":0.999,"clip":0.1})
    rows=[]
    for gi,hp in enumerate(grid):
        print(f"\n--- v4 grid {gi+1}/{len(grid)}: risk={hp['alpha_risk']} turn={hp['alpha_turn']} ---")
        best=train(Xs,Y,R,cov,z0,Bbase,train_idx,val_idx,hp,sc0,epochs=40,patience=10)
        rows.append({"alpha_risk":hp["alpha_risk"],"alpha_turn":hp["alpha_turn"],
                     "anchor_start":hp["anchor_start"],"anchor_end":hp["anchor_end"],
                     "val_sharpe":best["sharpe"],"val_turnover":best["to"],
                     "val_mse":best["mse"],"best_epoch":best["epoch"]})
        print("grid result:",rows[-1])
    df=pd.DataFrame(rows).sort_values("val_sharpe",ascending=False)
    df.to_csv(HERE/"e2e_v4_val_grid.csv",index=False); print(df.to_string(index=False))
    br=df.iloc[0]
    hp_best={"alpha_risk":float(br["alpha_risk"]),"alpha_turn":float(br["alpha_turn"]),
             "alpha_smooth":0.1,"anchor_start":float(br["anchor_start"]),
             "anchor_end":float(br["anchor_end"]),"lr":1e-4,"batch":64,
             "beta1":0.9,"beta2":0.999,"clip":0.1}
    epochs_final=int(br["best_epoch"])+1
    print(f"\n--- v4 final on train+val, epochs={epochs_final}, hp={hp_best} ---")
    all_train=np.arange(0,test_start)
    bf=train(Xs,Y,R,cov,z0,Bbase,all_train,all_train[-50:],hp_best,sc0,epochs=epochs_final,patience=10**9)
    U,V=bf["U_final"],bf["V_final"]
    Wte,_=forward(z0[test_start:],Xs[test_start:],U,V)
    np.save(HERE/"e2e_v4_test_weights.npy",Wte)
    np.savez(HERE/"e2e_v4_model.npz",U=U,V=V,hp=json.dumps(hp_best))
    print("v4 test MSE:",float(((Wte-Y[test_start:])**2).mean()))
    print("saved e2e_v4_test_weights.npy")


if __name__=="__main__":
    main()
