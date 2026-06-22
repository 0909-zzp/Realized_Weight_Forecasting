"""快速验证：大λ + 更小λ（补测）。"""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

_CODE_DIR = Path(__file__).resolve().parent
_PROJ  = _CODE_DIR.parent.parent
DATA_DIR = _PROJ / "数据" / "1min_log_return"
OUT_DIR  = _PROJ / "图形Lasso"

K=392; MAX_ITER=800; TOL_GL=5e-5; ENET_TOL=5e-4; EPS_R=1e-4; ETA=1e-4
RF=[5e-4,1e-3,5e-3,1e-2]; L_T=40; L_V=60

# Worker 函数 (模块级)
def _load(fs, idx):
    import pyreadr
    return pyreadr.read_r(str(DATA_DIR/fs[idx]))["rett1"].values

def _cr(r): return r@r.T
def _build(raw,s,e):
    tc=np.zeros((K,K),dtype=np.float64)
    for i in range(s,e+1): tc+=raw[i]
    tc/=(e-s+1); tc.flat[::K+1]+=EPS_R; return tc

def _gl(cov,lam):
    from sklearn.covariance import graphical_lasso
    for i,r in enumerate([EPS_R]+RF):
        c=cov.copy()
        if i>0: c.flat[::K+1]+=(r-EPS_R)
        try:
            ce,pr=graphical_lasso(emp_cov=c,alpha=lam,mode='cd',tol=TOL_GL,max_iter=MAX_ITER,enet_tol=ENET_TOL)
            return pr,ce
        except (FloatingPointError,ValueError): continue
    c=cov.copy(); c.flat[::K+1]+=1e-1-EPS_R
    ce,pr=graphical_lasso(emp_cov=c,alpha=lam,mode='cd',tol=TOL_GL,max_iter=MAX_ITER,enet_tol=max(ENET_TOL,1e-3))
    return pr,ce

def _w(prec): w=prec@np.ones(K); return w/np.sum(w)

def _worker(args):
    v,ct,rv,lams=args; res=[]
    for lam in lams:
        try:
            pr,_=_gl(ct,lam); w=_w(pr); res.append((float(w@rv@w),w))
        except Exception: res.append((np.nan,None))
    return v,res

if __name__ == '__main__':
    import os
    fs=sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_1min_log_return.RData") and not f.startswith("1min")])
    ds=[f.split(" ")[0] for f in fs]; nd=len(fs)

    lams=np.array([5e-6,1.39e-3,3.73e-3,1e-2]); N_L=4
    N_W=max(1,mp.cpu_count()-1)
    ve=nd-31; vs=max(L_T,ve-L_V)
    print(f"验证窗: {ds[vs]} ~ {ds[ve-1]}")
    print(f"测试 λ: {', '.join(f'{l:.2e}' for l in lams)}")

    ls=max(0,vs-L_T)
    raw={}
    for i in range(ls,ve): raw[i]=_cr(_load(fs,i))

    vd=list(range(vs,ve)); tasks=[]
    for v in vd:
        ct=_build(raw,v-L_T,v-1)
        tasks.append((v,ct,raw[v],lams))

    t0=time.time()
    tv=[0.0]*N_L; tt=[0.0]*N_L; nv=[0]*N_L; rbd={}
    with ProcessPoolExecutor(max_workers=N_W) as ex:
        fut={ex.submit(_worker,t):t[0] for t in tasks}
        done=0; nt=len(tasks)
        for f in as_completed(fut):
            v,res=f.result(); rbd[v]=res; done+=1
            e=time.time()-t0; s=done/e if e>0 else 0
            print(f"  {done}/{nt} 耗时:{e:.0f}s  ETA:{((nt-done)/s):.0f}s  日期:{ds[v]}")

    pw=[None]*N_L
    for v in sorted(vd):
        rl=rbd.get(v,[])
        for li in range(N_L):
            if li<len(rl): pv,w=rl[li]
            else: pv,w=np.nan,None
            if np.isfinite(pv) and w is not None:
                tv[li]+=pv; nv[li]+=1
                if pw[li] is not None: tt[li]+=float(np.sum(np.abs(w-pw[li])))
                pw[li]=w

    print("\n结果：")
    for li,lam in enumerate(lams):
        n=nv[li]
        if n>0:
            av=tv[li]/n; at=tt[li]/(n-1) if n>1 else 0
            print(f"  λ={lam:.2e}  Var={av:.4e}  TO={at:.4f}  Score={float(av+ETA*at):.4e}  n={n}")
        else:
            print(f"  λ={lam:.2e}  ALL FAILED n=0")
