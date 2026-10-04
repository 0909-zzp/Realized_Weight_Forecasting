import os, time, warnings, numpy as np
os.environ['OPENBLAS_NUM_THREADS']='1'
warnings.filterwarnings('ignore')
from pathlib import Path
P=Path(r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
VARX=P/'VARX'
Y=np.load('特征工程/Y_targets.npy')
rets=np.load('endtoend_project/cache/simple_returns.npy')
covs=np.load('endtoend_project/cache/cov_cache.npy', mmap_mode='r')
K=392; ell=K-1; ETA=1e-4; T0,T1=65,265
n=len(Y); n_train=int(0.7*n); n_val=int(0.15*n); test_start=n_train+n_val; n_test=n-test_start
Y_test=Y[test_start:]; Y_train=Y[:n_train]; r_test=rets[test_start:]
W_full=Y[:,:ell]

def norm(W):
    s=W.sum(1,keepdims=True); s=np.where(np.abs(s)<1e-12,1.0,s); return W/s

def ev(P_):
    W=P_[T0:T1]; R=T1-T0-1
    pr=np.array([W[t]@r_test[T0+t+1] for t in range(R)])
    to=np.array([np.abs(W[t+1]-W[t]).sum() for t in range(R)])
    rpv=np.array([W[t]@(covs[test_start+T0+t+1]+1e-4*np.eye(K))@W[t] for t in range(R)])
    mse=float(np.mean((W-Y_test[T0:T1])**2)); mae=float(np.mean(np.abs(W-Y_test[T0:T1])))
    rpv_ann=float(np.mean(rpv)*252); to_mean=float(np.mean(to))
    net=pr-ETA*to; sr=float(np.mean(net)/np.std(net,ddof=1)*np.sqrt(252)) if np.std(net)>1e-15 else 0.0
    return dict(MSE=mse,MAE=mae,RPV=rpv_ann,TO=to_mean,SR=sr)

rows=[]
cached=[
 ('Lasso VAR(1)','Y_pred_GG_lasso.npy'),
 ('Lasso VAR(20)','Y_pred_GG_lasso20.npy'),
 ('scalar VAR(1) M~','Y_pred_GG_econ.npy'),
 ('scalar MHAR M~','Y_pred_GG_mhar.npy'),
]
for nm,fn in cached:
    rows.append((nm, ev(np.load(VARX/fn))))

cw=np.cumsum(W_full,axis=0)
Ww=np.zeros_like(W_full); Ww[5:]=(cw[5:]-cw[:-5])/5
Wm=np.zeros_like(W_full); Wm[20:]=(cw[20:]-cw[:-20])/20
ones=np.ones(ell)

def fit_econ_mhat(nreg):
    A=np.zeros((ell,ell)); Bv=np.zeros((ell,nreg)); D=np.zeros((nreg,nreg)); u=np.zeros(ell); v=np.zeros(nreg)
    for t in range(20,n_train):
        Rt=np.asarray(covs[t])+1e-4*np.eye(K)   # M^ : current R_t
        R11=Rt[:ell,:ell]; r1k=Rt[:ell,K-1]; rkk=Rt[K-1,K-1]
        cfree=[W_full[t-1]] if nreg==1 else [W_full[t-1], Ww[t], Wm[t]]
        c1a=np.empty((nreg,ell)); cka=np.empty(nreg)
        for j in range(nreg):
            c1a[j]=cfree[j]; cka[j]=-float(cfree[j].sum())
        r1=W_full[t]; rk=-float(r1.sum())
        A+=R11-np.outer(r1k,ones)-np.outer(ones,r1k)+rkk*np.outer(ones,ones)
        Rc=np.empty((nreg,ell)); rkc=np.empty(nreg)
        for j in range(nreg):
            Rc[j]=R11@c1a[j]+r1k*cka[j]; rkc[j]=float(r1k@c1a[j]+rkk*cka[j])
        Bv+=Rc.T-np.outer(ones,rkc)
        for j in range(nreg):
            for kk in range(nreg):
                D[j,kk]+=float(c1a[j]@Rc[kk]+cka[j]*rkc[kk])
        Rr=R11@r1+r1k*rk; sr=float(r1k@r1+rkk*rk); u+=Rr-ones*sr
        for j in range(nreg): v[j]+=float(c1a[j]@Rr+cka[j]*sr)
    MM=np.zeros((ell+nreg,ell+nreg)); MM[:ell,:ell]=A; MM[:ell,ell:]=Bv; MM[ell:,:ell]=Bv.T; MM[ell:,ell:]=D
    Mr=np.concatenate([u,v]); MM=0.5*(MM+MM.T); th=np.linalg.solve(MM,Mr)
    return th[:ell], th[ell:]

def econ_pred(gamma,phi,nreg):
    cw_=np.cumsum(W_full,axis=0); Ww_=np.zeros_like(W_full); Ww_[5:]=(cw_[5:]-cw_[:-5])/5
    Wm_=np.zeros_like(W_full); Wm_[20:]=(cw_[20:]-cw_[:-20])/20
    lag1=W_full[test_start-1:test_start-1+n_test]
    if nreg==1:
        V=gamma+phi[0]*lag1
    else:
        V=gamma+phi[0]*lag1+phi[1]*Ww_[test_start:test_start+n_test]+phi[2]*Wm_[test_start:test_start+n_test]
    return norm(np.concatenate([V,1-V.sum(1,keepdims=True)],1))

g1,p1=fit_econ_mhat(1)
rows.append(('scalar VAR(1) M^', ev(econ_pred(g1,p1,1))))
g3,p3=fit_econ_mhat(3)
rows.append(('scalar MHAR M^', ev(econ_pred(g3,p3,3))))
print('phi M^ VAR(1):', np.round(p1,4), ' phi M^ MHAR:', np.round(p3,4), flush=True)

# Post-Lasso VAR(20): OLS refit on Lasso-selected support
GB=np.load(VARX/'Y_pred_GG_lasso20_coefs.npy'); g_l=GB[:,0]; B_l=GB[:,1:]
p=20; Wtr=Y_train[:,:ell]
X=np.hstack([Wtr[p-k:n_train-k,:] for k in range(1,p+1)]); Yt=Wtr[p:]
g_post=np.zeros(ell); B_post=np.zeros_like(B_l)
for i in range(ell):
    sup=np.nonzero(B_l[i,:])[0]
    if sup.size==0:
        g_post[i]=Yt[:,i].mean(); continue
    Xs=np.column_stack([np.ones(len(Yt)), X[:,sup]])
    coef,_,_,_=np.linalg.lstsq(Xs,Yt[:,i],rcond=None)
    g_post[i]=coef[0]; B_post[i,sup]=coef[1:]
Xtest=np.hstack([W_full[test_start-k:test_start-k+n_test,:] for k in range(1,p+1)])
V=Xtest@B_post.T+g_post
rows.append(('Post-Lasso VAR(20)', ev(norm(np.concatenate([V,1-V.sum(1,keepdims=True)],1)))))

print()
print('%-24s %10s %10s %10s %8s %8s' % ('variant','MSE','MAE','RPV','TO','SR'))
for nm,m in rows:
    print('%-24s %10.4e %10.4e %10.4e %8.4f %8.4f' % (nm,m['MSE'],m['MAE'],m['RPV'],m['TO'],m['SR']))
