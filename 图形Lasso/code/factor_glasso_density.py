"""因子分解 GLasso + BIC λ选择 — 降低网络密度
基于 Brownlees, Nualart & Sun (2018, JAE)
BIC(λ) = n × [-log det(K) + tr(K×S)] + log(n) × #{非零边}
"""
import sys, numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parents[1] / '图形Lasso' / 'code'))
from 共享模块 import K, load_day, compute_raw_cov, EPS_RIDGE
from sklearn.covariance import graphical_lasso

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 500  # 默认500天测试

ROOT = Path(__file__).parents[2]
npy_dir = ROOT / '数据' / '1min_log_return_npy'
files = sorted([f for f in npy_dir.iterdir() if f.suffix == '.npy' and f.name[0].isdigit()])
if N_DAYS > 0: files = files[-N_DAYS:]

RIDGE_CHAIN = [1e-4, 1e-3, 5e-3, 1e-2]
LAM_CANDIDATES = [5e-7, 1e-6, 3e-6, 5e-6, 1e-5, 5e-5, 1e-4]

print(f'因子+BIC GLasso: {len(files)}天  λ候选={len(LAM_CANDIDATES)}个')
import time; t0 = time.time()

dens_raw, dens_idio = [], []
deg_raw, deg_idio = [], []
lam_selected = []
fails_idio = 0

for i, fpath in enumerate(files):
    rett = np.load(str(fpath))
    daily_r = rett.sum(axis=1)
    
    # ── 原始 GLasso (固定 λ=3e-6) ──
    raw = compute_raw_cov(rett); raw.flat[::K+1] += EPS_RIDGE
    for r in RIDGE_CHAIN:
        c = raw.copy()
        if r > EPS_RIDGE: c.flat[::K+1] += (r - EPS_RIDGE)
        try:
            _, prec = graphical_lasso(c, alpha=3e-6, mode='cd', tol=1e-4, max_iter=100)
            adj = (np.abs(prec) > 1e-8).astype(int); np.fill_diagonal(adj, 0)
            dens_raw.append(adj.sum() / (K * (K - 1)))
            deg_raw.append(adj.sum(axis=1).mean())
            break
        except: continue
    
    # ── 因子分解 ──
    mkt_r = float(daily_r.mean())
    betas = np.zeros(K)
    if np.var(daily_r) > 1e-15:
        for j in range(K):
            cij = np.cov(daily_r[j], np.array([mkt_r]), ddof=1)[0,1]
            betas[j] = cij / np.var(mkt_r) if np.var(mkt_r) > 1e-15 else 0
    resid = daily_r - betas * mkt_r
    raw_I = np.outer(resid, resid)
    
    # ── BIC λ 选择 ──
    best_lam, best_bic = None, np.inf
    best_adj = None
    for lam in LAM_CANDIDATES:
        for r in RIDGE_CHAIN:
            c = raw_I.copy(); c.flat[::K+1] += r
            try:
                _, prec_I = graphical_lasso(c, alpha=lam, mode='cd', tol=1e-4, max_iter=100)
                adj_I = (np.abs(prec_I) > 1e-8).astype(int); np.fill_diagonal(adj_I, 0)
                nz = int(adj_I.sum() / 2)  # 非零边数
                # BIC (Yuan & Lin, 2007)
                try:
                    log_det = np.linalg.slogdet(prec_I)[1]
                    bic = K * (-log_det + np.trace(prec_I @ c / K)) + np.log(K) * nz
                except:
                    bic = np.inf
                if bic < best_bic:
                    best_bic = bic; best_lam = lam; best_adj = adj_I.copy()
                break
            except: continue
    
    if best_adj is not None:
        dens_idio.append(best_adj.sum() / (K * (K - 1)))
        deg_idio.append(best_adj.sum(axis=1).mean())
        lam_selected.append(best_lam)
    else:
        fails_idio += 1
    
    if (i+1) % 200 == 0:
        t = time.time() - t0
        print(f'  {i+1}/{len(files)}  raw_d={np.mean(dens_raw):.1%} idio_d={np.mean(dens_idio):.1%} '
              f'raw_deg={np.mean(deg_raw):.0f} idio_deg={np.mean(deg_idio):.0f} lam_mode={max(set(lam_selected),key=lam_selected.count):.0e}  {t/60:.0f}min')

t = time.time() - t0
print(f'\n=== 结果 ({t/60:.0f}min) ===')
print(f'原始 GLasso(λ=3e-6):')
print(f'  密度: {np.mean(dens_raw):.1%} ± {np.std(dens_raw):.1%}')
print(f'  mean degree: {np.mean(deg_raw):.0f} ± {np.std(deg_raw):.0f}')
print(f'因子分解 + BIC:')
print(f'  密度: {np.mean(dens_idio):.1%} ± {np.std(dens_idio):.1%}')
print(f'  mean degree: {np.mean(deg_idio):.0f} ± {np.std(deg_idio):.0f}')
print(f'  λ (mode): {max(set(lam_selected),key=lam_selected.count):.1e}')
print(f'  降幅: {(1-np.mean(deg_idio)/np.mean(deg_raw))*100:.0f}%')
print(f'  失败: {fails_idio}/{len(files)}')

