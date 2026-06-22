"""测试 λ=5e-7 的 22 天 OOS 表现。"""
import sys,warnings,numpy as np;warnings.filterwarnings("ignore");sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path; import os,pyreadr
_CODE=Path(__file__).resolve().parent;_PROJ=_CODE.parent.parent
DATA=_PROJ/"数据"/"1min_log_return"
K=392;MAX_ITER=800;TOL_GL=5e-5;ENET_TOL=5e-4;EPS_R=1e-4;RF=[5e-4,1e-3,5e-3,1e-2]
lam=5e-7
fs=sorted([f for f in os.listdir(DATA) if f.endswith("_1min_log_return.RData") and not f.startswith("1min")])
ds=[f.split(" ")[0] for f in fs];nd=len(fs)
te=nd-22-5-1;ts=te-44+1;vs=te+1;ve=vs+5-1;tss=ve+1;tse=tss+22-1
print(f"训练:{ds[ts]}~{ds[te]}  测试:{ds[tss]}~{ds[tse]}  λ={lam:.1e}")

tc=np.zeros((K,K),dtype=np.float64)
for i in range(ts,te+1):r=pyreadr.read_r(str(DATA/fs[i]))["rett1"].values;tc+=r@r.T
tc/=44;tc.flat[::K+1]+=EPS_R

from sklearn.covariance import graphical_lasso
for i,r in enumerate([EPS_R]+RF):
    c=tc.copy()
    if i>0:c.flat[::K+1]+=(r-EPS_R)
    try:
        ce,pr=graphical_lasso(emp_cov=c,alpha=lam,mode='cd',tol=TOL_GL,max_iter=MAX_ITER,enet_tol=ENET_TOL)
        w=pr@np.ones(K);w=w/np.sum(w);break
    except:continue
print(f"GLasso 完成，权重范围:[{w.min():.4f},{w.max():.4f}]")

tr=[pyreadr.read_r(str(DATA/fs[i]))["rett1"].values for i in range(tss,tse+1)]
tr=[rv@rv.T for rv in tr]
pvs=[float(w@rv@w) for rv in tr];pvs=[v for v in pvs if np.isfinite(v)]
print(f"\nGLasso({lam:.1e}) 测试:")
print(f"  天数: {len(pvs)}/22")
print(f"  平均方差: {np.mean(pvs):.4e}")
print(f"  标准差: {np.std(pvs):.4e}")

we=np.full(K,1./K)
ev=[float(we@rv@we) for rv in tr];ev=[v for v in ev if np.isfinite(v)]
print(f"\n等权重 测试:")
print(f"  平均方差: {np.mean(ev):.4e}")
print(f"  GLasso 超额: {(np.mean(pvs)/np.mean(ev)-1)*100:.1f}%")
