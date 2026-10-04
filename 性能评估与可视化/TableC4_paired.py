import os, warnings, importlib.util, sys, time
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
P=Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
sys.path.insert(0, str(P/'图形Lasso'/'code'))
from 共享模块 import K, LAMBDA_LASSO
spec=importlib.util.spec_from_file_location('vp', str(P/'VARX'/'VAR及拓展（table2）.py'))
vp=importlib.util.module_from_spec(spec); sys.modules['vp']=vp; spec.loader.exec_module(vp)

X=np.load(P/'特征工程'/'X_features.npy'); Y=np.load(P/'特征工程'/'Y_targets.npy'); A=np.load(P/'特征工程'/'A_bar.npy')
valid=np.load(P/'特征工程'/'valid_indices.npy'); n=len(X); ntr=int(n*.7); nval=int(n*.15); ts=ntr+nval
covs=np.load(P/'endtoend_project'/'cache'/'cov_cache.npy', mmap_mode='r')
npy=sorted([f for f in (P/'数据'/'1min_log_return_npy').iterdir() if f.suffix=='.npy' and f.name[0].isdigit()])
R=np.array([np.expm1(np.load(npy[i]).sum(axis=1)) for i in valid[ts:]])

# reconstruct partial adjustment + Sparse VARX-S-DF (save for reuse)
def drift(w,r):
    w=np.asarray(w,float); r=np.asarray(r,float); d=1+float(w@r); return (w*(1+r)/d) if abs(d)>1e-15 else w
def pa_path(bv,r,yp,g):
    T=len(bv); pos=yp.copy(); out=np.empty_like(bv)
    for t in range(T):
        pb=pos if t==0 else drift(pos,r[t-1]); pn=(1-g)*pb+g*bv[t]; pn=pn/pn.sum(); out[t]=pn; pos=pn
    return out
Ysp=np.load(P/'VARX'/'Y_pred_model3.npy')[-363:]
pa_file=P/'VARX'/'Y_pred_pa_sparse.npy'
if pa_file.exists(): Ypa=np.load(pa_file)
else: Ypa=pa_path(Ysp, R, Y[ts-1], 0.38); np.save(pa_file, Ypa)
sdf_file=P/'VARX'/'Y_pred_svarx_sdf.npy'
if sdf_file.exists(): Ysdf=np.load(sdf_file)
else:
    old=dict(vp.MODELS[3]); vp.MODELS[3]['self_free']=True; vp.MODELS[3]['smooth']=True; vp.MODELS[3]['network']=False; vp.MODELS[3]['lasso_lambda']=LAMBDA_LASSO
    fitted=vp.fit_model(3, X[:ntr], Y[:ntr], None, n_jobs=4)
    Ym3s=vp.predict_model(X[ts:], fitted)
    vp.MODELS[3].clear(); vp.MODELS[3].update(old)
    Ysdf=vp.compute_model6_drift_l1(Ym3s, Y[ts:], valid[:ntr], eta=1e-6, rho=1e-3, risk_mult=1.0, rolling_cov=True, cov_window=150)
    np.save(sdf_file, Ysdf)

models={
 'DF': np.load(P/'VARX'/'Y_pred_model6_opt.npy')[-363:],
 'Network VARX-S': np.load(P/'VARX'/'Y_pred_model5.npy')[-363:],
 'Sparse VARX-S-DF': Ysdf,
 'Partial adjustment': Ypa,
 'Sample GMVP': np.load(P/'VARX'/'Y_pred_bench_sample_W20.npy')[-363:],
}
T0,T1=65,265
Rw=R[T0:T1]
def daily_series(W):
    Ww=W[T0:T1]; T=len(Ww)-1
    gross=np.array([Ww[t]@Rw[t+1] for t in range(T)])
    to=np.array([np.abs(Ww[t+1]-Ww[t]*(1+Rw[t+1])/(1+Ww[t]@Rw[t+1])).sum() for t in range(T)])
    rpv=np.array([Ww[t]@(covs[ts+T0+t+1]+1e-4*np.eye(K))@Ww[t] for t in range(T)])
    net=gross-1e-4*to
    return net, rpv, to

series={m:daily_series(W) for m,W in models.items()}

def sharpe(s):
    return float(np.mean(s)/np.std(s,ddof=1)*np.sqrt(252)) if np.std(s)>1e-15 else 0.0

# point estimates
pairs=[('DF','Network VARX-S',['sharpe_diff','rpv_ratio','to_diff']),
       ('DF','Sparse VARX-S-DF',['sharpe_diff','rpv_ratio']),
       ('DF','Partial adjustment',['sharpe_diff','rpv_ratio']),
       ('DF','Sample GMVP',['sharpe_diff','rpv_ratio'])]

def block_bootstrap(net_a, net_b, stat, L=5, B=2000, seed=42):
    n=len(net_a); rng=np.random.default_rng(seed); nb=int(np.ceil(n/L))
    est=np.empty(B)
    for b in range(B):
        st=rng.integers(0,n-L+1,size=nb)
        idx=np.concatenate([np.arange(s,s+L) for s in st])[:n]
        est[b]=stat(net_a[idx], net_b[idx])
    return est

rows=[]
for a,b,ms in pairs:
    na,rpva,toa=series[a]; nb,rpvb,tob=series[b]
    for msr in ms:
        if msr=='sharpe_diff':
            pt=sharpe(na)-sharpe(nb)
            st=lambda x,y: sharpe(x)-sharpe(y)
            dist=block_bootstrap(na,nb,st)
            mname='Net Sharpe difference'
        elif msr=='rpv_ratio':
            pt=float(np.mean(rpva)/np.mean(rpvb))
            st=lambda x,y: float(np.mean(x)/np.mean(y))
            dist=block_bootstrap(rpva,rpvb,st)
            mname='RPV ratio'
        else:
            pt=float(np.mean(toa)-np.mean(tob))
            st=lambda x,y: float(np.mean(x)-np.mean(y))
            dist=block_bootstrap(toa,tob,st)
            mname='Turnover difference'
        lo,hi=np.percentile(dist,[2.5,97.5])
        rows.append(dict(comparison=f'{a} versus {b}', measure=mname, estimate=pt, lo=lo, hi=hi))

df=pd.DataFrame(rows)
df.to_csv(P/'性能评估与可视化'/'TableC4_paired.csv', index=False)
print()
for r in rows:
    print(f"{r['comparison']:<28} {r['measure']:<22} {r['estimate']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
print('\nSaved: 性能评估与可视化/TableC4_paired.csv')
