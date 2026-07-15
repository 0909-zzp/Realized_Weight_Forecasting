"""补测小 λ，固定窗（复用 _λ的选择.py 的参数）。"""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path; import os, pyreadr

_CODE=Path(__file__).resolve().parent; _PROJ=_CODE.parent.parent
DATA_DIR=_PROJ/"数据"/"1min_log_return"
K=392; MAX_ITER=800; TOL_GL=5e-5; ENET_TOL=5e-4; EPS_R=1e-4
RF=[5e-4,1e-3,5e-3,1e-2]; N_T=44; N_V=5; N_TEST=22

fs=sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_1min_log_return.RData") and not f.startswith("1min")])
ds=[f.split(" ")[0] for f in fs]; nd=len(fs)
te=nd-N_TEST-N_V-1; ts=te-N_T+1
vs=te+1; ve=vs+N_V-1
print(f"训练:{ds[ts]}~{ds[te]}  验证:{ds[vs]}~{ds[ve]}")

tc=np.zeros((K,K),dtype=np.float64)
for i in range(ts,te+1):
    r=pyreadr.read_r(str(DATA_DIR/fs[i]))["rett1"].values; tc+=r@r.T
tc/=N_T; tc.flat[::K+1]+=EPS_R

val_raws=[pyreadr.read_r(str(DATA_DIR/fs[i]))["rett1"].values for i in range(vs,ve+1)]
val_raws=[rv@rv.T for rv in val_raws]

from sklearn.covariance import graphical_lasso
def gl(cov,lam):
    for i,r in enumerate([EPS_R]+RF):
        c=cov.copy()
        if i>0: c.flat[::K+1]+=(r-EPS_R)
        try:
            ce,pr=graphical_lasso(emp_cov=c,alpha=lam,mode='cd',tol=TOL_GL,max_iter=MAX_ITER,enet_tol=ENET_TOL)
            return pr,ce
        except: continue
    c=cov.copy(); c.flat[::K+1]+=1e-1-EPS_R
    ce,pr=graphical_lasso(emp_cov=c,alpha=lam,mode='cd',tol=TOL_GL,max_iter=MAX_ITER,enet_tol=max(ENET_TOL,1e-3))
    return pr,ce
def w_(p): w=p@np.ones(K); return w/np.sum(w)

lams=[2e-6,3e-6,5e-6,7e-6,1e-5]
print(f"\n测试 λ:"); t0=time.time()
for lam in lams:
    try:
        pr,_=gl(tc,lam); ww=w_(pr)
        pvs=[float(ww@rv@ww) for rv in val_raws]; pvs=[v for v in pvs if np.isfinite(v)]
        if pvs:
            print(f"  λ={lam:.1e}  Var={np.mean(pvs):.4e} ({len(pvs)}/5)")
        else:
            print(f"  λ={lam:.1e}  无有效日")
    except Exception as e:
        print(f"  λ={lam:.1e}  FAIL: {type(e).__name__}")
print(f"耗时: {time.time()-t0:.0f}s")
