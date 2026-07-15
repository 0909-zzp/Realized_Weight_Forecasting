"""试更小 λ：3e-7, 1e-7，固定窗。"""
import sys,warnings,numpy as np;warnings.filterwarnings("ignore");sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path; import os,pyreadr,time
_CODE=Path(__file__).resolve().parent;_PROJ=_CODE.parent.parent
DATA=_PROJ/"数据"/"1min_log_return"
K=392;MAX_ITER=800;TOL_GL=5e-5;ENET_TOL=5e-4;EPS_R=1e-4;RF=[5e-4,1e-3,5e-3,1e-2]
fs=sorted([f for f in os.listdir(DATA) if f.endswith("_1min_log_return.RData") and not f.startswith("1min")])
ds=[f.split(" ")[0] for f in fs];nd=len(fs)
te=nd-22-5-1;ts=te-44+1;vs=te+1;ve=vs+5-1
tc=np.zeros((K,K),dtype=np.float64)
for i in range(ts,te+1):r=pyreadr.read_r(str(DATA/fs[i]))["rett1"].values;tc+=r@r.T
tc/=44;tc.flat[::K+1]+=EPS_R
vr=[pyreadr.read_r(str(DATA/fs[i]))["rett1"].values for i in range(vs,ve+1)]
vr=[v@v.T for v in vr]
from sklearn.covariance import graphical_lasso
print(f"训练:{ds[ts]}~{ds[te]}  验证:{ds[vs]}~{ds[ve]}\n")
for lam in [3e-7,1e-7]:
    t0=time.time();ok=False
    for i,r in enumerate([EPS_R]+RF):
        c=tc.copy()
        if i>0:c.flat[::K+1]+=(r-EPS_R)
        try:
            ce,pr=graphical_lasso(emp_cov=c,alpha=lam,mode='cd',tol=TOL_GL,max_iter=MAX_ITER,enet_tol=ENET_TOL)
            w=pr@np.ones(K);w=w/np.sum(w)
            pvs=[float(w@rv@w) for rv in vr];pvs=[v for v in pvs if np.isfinite(v)]
            if pvs:
                print(f"  λ={lam:.1e}  Var={np.mean(pvs):.4e} ({len(pvs)}/5) 耗时:{time.time()-t0:.0f}s  ridge={r:.0e}")
                ok=True;break
        except Exception as e:
            print(f"  λ={lam:.1e}  ridge={r:.0e}  {type(e).__name__}: {str(e)[:50]}")
    if not ok:print(f"  λ={lam:.1e}  ALL FAILED")
