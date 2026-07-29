"""因子分解 + BIC GLasso — 降低网络密度测试
Brownlees, Nualart & Sun (2018, JAE) 方法
"""
import sys, numpy as np, time
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parents[1] / '图形Lasso' / 'code'))
from 共享模块 import K, load_day, compute_raw_cov, EPS_RIDGE
from sklearn.covariance import graphical_lasso

# 参数
N_DAYS = 100  # 0=全量, >0=抽样
LAM_CAND = [1e-6, 5e-6, 1e-5, 5e-5]              # BIC λ 候选
RIDGE_CHAIN = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]     # 保证 SPD

# 数据
ROOT = Path(__file__).parents[2]
npy_dir = ROOT / '数据' / '1min_log_return_npy'
all_files = sorted([f for f in npy_dir.iterdir() if f.suffix == '.npy' and f.name[0].isdigit()])
rng = np.random.default_rng(42)
if N_DAYS > 0 and N_DAYS < len(all_files):
    files = rng.choice(all_files, N_DAYS, replace=False)
else:
    files = all_files

print(f'因子分解+BIC GLasso: N={len(files)}天  λ候选={LAM_CAND}')
t0 = time.time()

# ── 存储 ──
deg_raw, deg_idio = [], []
lam_sel = []
fails = 0

for i, fpath in enumerate(files):
    rett = np.load(str(fpath))          # (K, M)
    M = rett.shape[1]
    daily_r = rett.sum(axis=1)          # (K,) 日收益

    # ── 1. 原始 GLasso(λ=3e-6) ──
    raw_cov = compute_raw_cov(rett)
    raw_cov.flat[::K+1] += EPS_RIDGE
    for r_base in RIDGE_CHAIN:
        c_raw = raw_cov.copy()
        if r_base > EPS_RIDGE:
            c_raw.flat[::K+1] += (r_base - EPS_RIDGE)
        try:
            _, prec_raw = graphical_lasso(c_raw, alpha=3e-6, mode='cd', tol=1e-4, max_iter=100)
            adj = (np.abs(prec_raw) > 1e-8).astype(int)
            np.fill_diagonal(adj, 0)
            deg_raw.append(adj.sum(axis=1).mean())
            break
        except:
            continue

    # ── 2. 因子分解 (日内数据, 同频) ──
    # 等权市场组合的日内收益序列 (M,)
    mkt_intraday = rett.mean(axis=0)
    var_mkt = np.var(mkt_intraday)
    if var_mkt < 1e-15:
        fails += 1; continue

    # β: 从日内收益直接估计 (同频)
    betas = np.array([
        np.cov(rett[i], mkt_intraday, ddof=1)[0,1] / var_mkt
        for i in range(K)
    ])

    # 系统性协方差: Σ_sys = σ²_m · ββᵀ
    sigma2_m = var_mkt * M   # 日内已实现方差 (不除M)
    cov_sys = sigma2_m * np.outer(betas, betas)

    # 异质性协方差 = 原始 - 系统性
    cov_I = raw_cov - cov_sys
    cov_I.flat[::K+1] += EPS_RIDGE  # 补 Ridge 防退化

    # ── 3. BIC 选 λ ──
    best_lam, best_bic = None, np.inf
    best_adj = None
    for lam in LAM_CAND:
        for r_base in RIDGE_CHAIN:
            c = cov_I.copy()
            c.flat[::K+1] += max(0, r_base - EPS_RIDGE)
            try:
                _, prec = graphical_lasso(c, alpha=lam, mode='cd', tol=5e-4, max_iter=150)
                adj_lam = (np.abs(prec) > 1e-8).astype(int)
                np.fill_diagonal(adj_lam, 0)
                nz = adj_lam.sum() / 2
                ld = np.linalg.slogdet(prec)[1]
                bic = K * (-ld + np.trace(prec @ c / K)) + np.log(K) * nz
                if bic < best_bic:
                    best_bic = bic; best_lam = lam
                    best_adj = adj_lam.copy()
                break
            except:
                continue

    if best_adj is not None:
        deg_idio.append(best_adj.sum(axis=1).mean())
        lam_sel.append(best_lam)
    else:
        fails += 1

    if (i+1) % 10 == 0:
        t_elapsed = time.time() - t0
        if len(deg_idio) > 0:
            print(f'  {i+1}/{len(files)}  原始deg={np.mean(deg_raw):.0f}  异质deg={np.mean(deg_idio):.0f}  '
                  f'BIC_lam={max(set(lam_sel),key=lam_sel.count):.1e}  fails={fails}  {t_elapsed/60:.0f}min')

# ── 结果 ──
t_total = time.time() - t0
print(f'\n{"="*55}')
print(f'结果 ({t_total/60:.0f}min, {len(deg_idio)}/{N_DAYS} 成功)')
print(f'{"="*55}')
if len(deg_raw) > 0:
    print(f'原始 GLasso(λ=3e-6):')
    print(f'  mean degree: {np.mean(deg_raw):.0f} ± {np.std(deg_raw):.0f}')
    print(f'  密度: {np.mean(deg_raw)/391*100:.1f}%')
if len(deg_idio) > 0:
    print(f'因子分解 + BIC:')
    print(f'  mean degree: {np.mean(deg_idio):.0f} ± {np.std(deg_idio):.0f}')
    print(f'  密度: {np.mean(deg_idio)/391*100:.1f}%')
    print(f'  降幅: {100*(1-np.mean(deg_idio)/np.mean(deg_raw)):.0f}%')
    print(f'  BIC λ (mode): {max(set(lam_sel),key=lam_sel.count):.1e}')
