"""因子分解 + GLasso — 全量多进程并行版
基于 Brownlees, Nualart & Sun (2018, JAE)
固定 λ=5e-5 (100天采样 BIC mode, 非逐日BIC), 免逐日搜索
cov_I = raw_cov - cov_sys, 其中 cov_sys = σ²_m·ββᵀ (CAPM因子分解)
8进程并行 → 全量2436天 ≈ 3-4小时
支持断点续跑: python factor_bic_full.py --resume
"""
import sys, os, gc, time, argparse, numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings; warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 路径与导入
# ═══════════════════════════════════════════════════════════════
_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))

# ═══════════════════════════════════════════════════════════════
# 参数 (与 factor_bic_test.py 和 共享模块.py 一致)
# ═══════════════════════════════════════════════════════════════
K = 392
LAM_RAW   = 3e-6    # 原始 GLasso λ
LAM_IDIO  = 5e-5    # 异质 GLasso λ (100天 BIC mode)
EPS_RIDGE = 1e-4
RIDGE_CHAIN = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
N_WORKERS = 8        # 并行进程数
BATCH_SIZE = 32      # 每批提交天数 (4×workers)
CHECKPOINT_EVERY = 100

OUT_DIR = _PROJECT_ROOT / '图形Lasso'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT_DIR / 'factor_bic_checkpoint.npz'
LOG_PATH  = OUT_DIR / 'factor_bic_full_stdout.log'
NPY_DIR   = _PROJECT_ROOT / '数据' / '1min_log_return_npy'

# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════
def log(msg: str):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ═══════════════════════════════════════════════════════════════
# Worker 函数 (每个子进程独立运行)
# ═══════════════════════════════════════════════════════════════
def _process_one_day(args):
    """处理单天数据, 返回 (idx, raw_deg, idio_deg, raw_ok, idio_ok, elapsed).
    
    子进程中导入 sklearn, 避免主进程序列化开销.
    """
    # 限制 BLAS 线程数, 6 workers × 2 threads = 12 cores
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    
    from sklearn.covariance import graphical_lasso
    
    fpath_str, idx = args
    t0 = time.time()
    
    try:
        rett = np.load(fpath_str)  # (K, M)
    except Exception:
        return idx, np.nan, np.nan, False, False, time.time() - t0
    
    M = rett.shape[1]
    raw_cov = rett @ rett.T
    raw_cov.flat[::K+1] += EPS_RIDGE
    
    # ── 原始 GLasso ──
    raw_deg = np.nan
    raw_ok = False
    for r_base in RIDGE_CHAIN:
        c = raw_cov.copy()
        if r_base > EPS_RIDGE:
            c.flat[::K+1] += (r_base - EPS_RIDGE)
        try:
            _, prec = graphical_lasso(c, alpha=LAM_RAW, mode='cd',
                                       tol=1e-4, max_iter=100)
            adj = (np.abs(prec) > 1e-8).astype(int)
            np.fill_diagonal(adj, 0)
            raw_deg = float(adj.sum(axis=1).mean())
            raw_ok = True
            break
        except (FloatingPointError, ValueError):
            continue
    
    # ── 因子分解 (同频 CAPM) ──
    idio_deg = np.nan
    idio_ok = False
    
    mkt_intraday = rett.mean(axis=0)
    var_mkt = np.var(mkt_intraday)
    if var_mkt >= 1e-15:
        # 向量化 β 估计
        mkt_demean = mkt_intraday - mkt_intraday.mean()
        rets_demean = rett - rett.mean(axis=1, keepdims=True)
        cov_with_mkt = rets_demean @ mkt_demean / (M - 1)
        betas = cov_with_mkt / var_mkt
        
        sigma2_m = var_mkt * M
        cov_sys = sigma2_m * np.outer(betas, betas)
        cov_I = raw_cov - cov_sys
        cov_I.flat[::K+1] += EPS_RIDGE
        
        # ── 异质 GLasso (固定 λ) ──
        for r_base in RIDGE_CHAIN:
            c = cov_I.copy()
            if r_base > EPS_RIDGE:
                c.flat[::K+1] += (r_base - EPS_RIDGE)
            try:
                _, prec = graphical_lasso(c, alpha=LAM_IDIO, mode='cd',
                                           tol=5e-4, max_iter=150)
                adj = (np.abs(prec) > 1e-8).astype(int)
                np.fill_diagonal(adj, 0)
                idio_deg = float(adj.sum(axis=1).mean())
                idio_ok = True
                break
            except (FloatingPointError, ValueError):
                continue
    
    return idx, raw_deg, idio_deg, raw_ok, idio_ok, time.time() - t0


# ═══════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='从断点续跑')
    parser.add_argument('--n_days', type=int, default=0, help='测试天数 (0=全量)')
    parser.add_argument('--workers', type=int, default=N_WORKERS, help='并行进程数')
    args = parser.parse_args()
    
    n_workers = min(args.workers, N_WORKERS)
    
    # ── 数据文件列表 ──
    all_files = sorted([
        str(f) for f in NPY_DIR.iterdir()
        if f.suffix == '.npy' and f.name[0].isdigit()
    ])
    total_days = len(all_files)
    
    if args.n_days > 0:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_days, min(args.n_days, total_days), replace=False)
        all_files = [all_files[i] for i in indices]
        total_days = len(all_files)
    
    log(f'{"="*55}')
    log(f'因子分解 + GLasso 全量并行版')
    log(f'总天数: {total_days}  |  K={K}  |  Workers={n_workers}')
    log(f'原始 λ={LAM_RAW:.0e}  |  异质 λ={LAM_IDIO:.0e}')
    log(f'Checkpoint: 每 {CHECKPOINT_EVERY} 天 → {CKPT_PATH}')
    log(f'{"="*55}')
    
    # ── 断点续跑 ──
    start_idx = 0
    raw_degs = [np.nan] * total_days
    idio_degs = [np.nan] * total_days
    raw_oks = [False] * total_days
    idio_oks = [False] * total_days
    
    if args.resume and CKPT_PATH.exists():
        ckpt = np.load(CKPT_PATH, allow_pickle=True)
        start_idx = int(ckpt['last_idx']) + 1
        saved_raw = ckpt['raw_degs']
        saved_idio = ckpt['idio_degs']
        saved_raw_ok = ckpt['raw_oks']
        saved_idio_ok = ckpt['idio_oks']
        for i in range(min(start_idx, total_days)):
            if i < len(saved_raw):
                raw_degs[i] = saved_raw[i]
                idio_degs[i] = saved_idio[i]
                raw_oks[i] = bool(saved_raw_ok[i])
                idio_oks[i] = bool(saved_idio_ok[i])
        log(f'[RESUME] 从第 {start_idx} 天继续 '
            f'(已完成 {start_idx} 天)')
    
    # ── 主循环: 分批并行处理 ──
    t_all = time.time()
    pending = list(range(start_idx, total_days))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        while pending:
            # 取一批
            batch = pending[:BATCH_SIZE]
            pending = pending[BATCH_SIZE:]
            
            # 提交任务
            futures = {
                executor.submit(_process_one_day, (all_files[i], i)): i
                for i in batch
            }
            
            # 收集结果 (无序)
            for future in as_completed(futures):
                idx, raw_d, idio_d, raw_ok, idio_ok, elapsed = future.result()
                raw_degs[idx] = raw_d
                idio_degs[idx] = idio_d
                raw_oks[idx] = raw_ok
                idio_oks[idx] = idio_ok
            
            # ── 进度 ──
            done_count = total_days - len(pending)
            raw_valid = [d for d in raw_degs[:done_count] if not np.isnan(d)]
            idio_valid = [d for d in idio_degs[:done_count] if not np.isnan(d)]
            fails_raw = sum(1 for ok in raw_oks[:done_count] if not ok)
            fails_idio = sum(1 for ok in idio_oks[:done_count] if not ok)
            
            elapsed = time.time() - t_all
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total_days - done_count) / rate if rate > 0 else 0
            
            if raw_valid and idio_valid:
                log(f'  [{done_count}/{total_days}]  '
                    f'raw_deg={np.mean(raw_valid):.0f}±{np.std(raw_valid):.0f}  '
                    f'idio_deg={np.mean(idio_valid):.0f}±{np.std(idio_valid):.0f}  '
                    f'fail(raw={fails_raw},idio={fails_idio})  '
                    f'ETA={eta/60:.0f}min  [{elapsed/60:.0f}min]')
            else:
                log(f'  [{done_count}/{total_days}]  processing...  '
                    f'ETA={eta/60:.0f}min  [{elapsed/60:.0f}min]')
            
            # ── Checkpoint ──
            if done_count > 0 and done_count % CHECKPOINT_EVERY == 0:
                np.savez(CKPT_PATH,
                         last_idx=done_count - 1,
                         raw_degs=np.array(raw_degs[:done_count]),
                         idio_degs=np.array(idio_degs[:done_count]),
                         raw_oks=np.array(raw_oks[:done_count]),
                         idio_oks=np.array(idio_oks[:done_count]))
                gc.collect()
    
    # ── 清理 checkpoint ──
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
    
    # ── 最终结果 ──
    raw_valid = [d for d in raw_degs if not np.isnan(d)]
    idio_valid = [d for d in idio_degs if not np.isnan(d)]
    fails_raw = sum(1 for ok in raw_oks if not ok)
    fails_idio = sum(1 for ok in idio_oks if not ok)
    
    t_total = time.time() - t_all
    log(f'\n{"="*55}')
    log(f'全量完成 ({t_total/60:.0f}min)')
    log(f'{"="*55}')
    log(f'成功: raw={len(raw_valid)}天  idio={len(idio_valid)}天')
    log(f'失败: raw={fails_raw}  idio={fails_idio}')
    
    if raw_valid:
        d_raw = np.array(raw_valid)
        log(f'\n原始 GLasso (λ={LAM_RAW:.0e}):')
        log(f'  mean degree: {d_raw.mean():.1f} ± {d_raw.std():.1f}')
        log(f'  密度: {d_raw.mean()/(K-1)*100:.1f}%')
        log(f'  中位数: {np.median(d_raw):.1f}')
        log(f'  范围: [{d_raw.min():.1f}, {d_raw.max():.1f}]')
    
    if idio_valid:
        d_idio = np.array(idio_valid)
        log(f'\n因子分解 + GLasso (λ={LAM_IDIO:.0e}):')
        log(f'  mean degree: {d_idio.mean():.1f} ± {d_idio.std():.1f}')
        log(f'  密度: {d_idio.mean()/(K-1)*100:.1f}%')
        log(f'  中位数: {np.median(d_idio):.1f}')
        log(f'  范围: [{d_idio.min():.1f}, {d_idio.max():.1f}]')
        
        if raw_valid:
            reduction = (1 - d_idio.mean() / d_raw.mean()) * 100
            log(f'\n  降幅: {reduction:.0f}%  ({d_raw.mean():.0f}→{d_idio.mean():.0f})')
    
    # ── 保存最终结果 ──
    result_path = OUT_DIR / 'factor_bic_full_results.npz'
    np.savez(result_path,
             raw_degs=np.array(raw_degs),
             idio_degs=np.array(idio_degs),
             raw_oks=np.array(raw_oks),
             idio_oks=np.array(idio_oks),
             lam_raw=LAM_RAW,
             lam_idio=LAM_IDIO,
             n_days=total_days,
             n_workers=n_workers)
    log(f'\n结果已保存: {result_path}')


if __name__ == '__main__':
    main()
