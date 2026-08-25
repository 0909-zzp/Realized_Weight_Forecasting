"""构建因子分解版 A_bar — 直接对齐原始2415样本
Step 1: 并行生成 factor-decomposed adjacency 存到 adjacency_idio/
Step 2: 读原始adjacency获取 valid_mask → 对齐样本 → 保存 A_bar_idio.npy
"""
import sys, os, gc, time, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings; warnings.filterwarnings('ignore')

_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))

K = 392
LAM_IDIO = 5e-5
EPS_RIDGE = 1e-4
RIDGE_CHAIN = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
ROLLING_WINDOW = 20
P_LAGS = 3
N_WORKERS = 4
BATCH_SIZE = 32

NPY_DIR = _PROJECT_ROOT / '数据' / '1min_log_return_npy'
ADJ_OUT = _PROJECT_ROOT / '图形Lasso' / 'code' / '输出数据' / 'adjacency_idio'
ADJ_OUT.mkdir(parents=True, exist_ok=True)
OUT_PATH = _PROJECT_ROOT / '特征工程' / 'A_bar_idio.npy'

def _process_day(args):
    """返回 (idx, success, adj_matrix)"""
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    from sklearn.covariance import graphical_lasso
    
    fpath_str, idx = args
    try:
        rett = np.load(fpath_str)
    except:
        return idx, False, None
    
    M = rett.shape[1]
    raw_cov = rett @ rett.T
    raw_cov.flat[::K+1] += EPS_RIDGE
    
    mkt = rett.mean(axis=0)
    var_mkt = np.var(mkt)
    if var_mkt < 1e-15:
        return idx, False, None
    
    mkt_dm = mkt - mkt.mean()
    rets_dm = rett - rett.mean(axis=1, keepdims=True)
    betas = (rets_dm @ mkt_dm / (M-1)) / var_mkt
    cov_sys = (var_mkt * M) * np.outer(betas, betas)
    cov_I = raw_cov - cov_sys
    cov_I.flat[::K+1] += EPS_RIDGE
    
    for r_base in RIDGE_CHAIN:
        c = cov_I.copy()
        if r_base > EPS_RIDGE:
            c.flat[::K+1] += (r_base - EPS_RIDGE)
        try:
            _, prec = graphical_lasso(c, alpha=LAM_IDIO, mode='cd',
                                       tol=5e-4, max_iter=100)
            adj = (np.abs(prec) > 1e-8).astype(np.float32)
            np.fill_diagonal(adj, 0)
            return idx, True, adj
        except (FloatingPointError, ValueError):
            continue
    
    return idx, False, None

def compute_rolling_mean(adj_list, t, window=ROLLING_WINDOW):
    start = max(0, t - window + 1)
    end = t + 1
    mats = [m for m in adj_list[start:end] if m is not None]
    if not mats:
        return np.zeros((K, K), dtype=np.float32)
    return np.mean(np.stack(mats), axis=0)

def main():
    all_files = sorted([str(f) for f in NPY_DIR.iterdir()
                        if f.suffix == '.npy' and f.name[0].isdigit()])
    total_days = len(all_files)
    print(f'=== Phase 1: Factor-decomposed adjacency ({total_days} days, {N_WORKERS} workers) ===')
    
    adj_list = [None] * total_days
    t0 = time.time()
    pending = list(range(total_days))
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        while pending:
            batch = pending[:BATCH_SIZE]
            pending = pending[BATCH_SIZE:]
            futures = {executor.submit(_process_day, (all_files[i], i)): i for i in batch}
            for future in as_completed(futures):
                idx, ok, adj = future.result()
                adj_list[idx] = adj if ok else None
            
            done = total_days - len(pending)
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_days - done) / rate if rate > 0 else 0
                n_ok = sum(1 for a in adj_list[:done] if a is not None)
                print(f'  [{done}/{total_days}] OK={n_ok}  ETA={eta/60:.0f}min')
    
    n_ok = sum(1 for a in adj_list if a is not None)
    print(f'Phase 1 done: {n_ok}/{total_days} OK ({time.time()-t0:.0f}s)')
    
    # Save individual adjacency files
    print('Saving adjacency files...')
    for i, adj in enumerate(adj_list):
        if adj is not None:
            fname = Path(all_files[i]).name.replace('_1min_log_return.npy', '.npy')
            np.save(str(ADJ_OUT / fname), adj)
    print(f'Saved {n_ok} files to {ADJ_OUT}')
    
    # === Phase 2: Build A_bar matching original 2415 samples ===
    print('\n=== Phase 2: Building A_bar_idio ===')
    orig_adj_dir = _PROJECT_ROOT / '图形Lasso' / 'code' / '输出数据' / 'adjacency'
    orig_files = sorted(orig_adj_dir.glob('*.npy'))
    orig_dates = [f.stem for f in orig_files]
    
    # Build valid_mask: day i is valid if its date has an original adjacency file
    all_dates = [Path(f).name[:8] for f in all_files]
    valid_mask = np.array([d in orig_dates for d in all_dates], dtype=bool)
    print(f'Original valid days: {valid_mask.sum()}/{total_days}')
    
    # Compute rolling averages for all days
    a_cache = [None] * total_days
    for i in range(total_days):
        if valid_mask[i]:
            a_cache[i] = compute_rolling_mean(adj_list, i)
    
    # Build A_bar matching original sample logic:
    # For t in range(P_LAGS, T): if valid_mask[t] and all lags valid → sample
    A_bar_list = []
    sample_day_map = []  # track which day each sample corresponds to
    for t in range(P_LAGS, total_days):
        if not valid_mask[t]:
            continue
        lag_ok = all(valid_mask[t - lag] for lag in range(1, P_LAGS + 1))
        if not lag_ok:
            continue
        a_val = a_cache[t-1] if a_cache[t-1] is not None else np.zeros((K, K))
        A_bar_list.append(a_val)
        sample_day_map.append(t)
    
    A_bar_idio = np.array(A_bar_list, dtype=np.float64)
    n_samples = A_bar_idio.shape[0]
    
    # Verify alignment with original
    orig_A = np.load(str(_PROJECT_ROOT / '特征工程' / 'A_bar.npy'))
    print(f'Original A_bar: {orig_A.shape[0]} samples')
    print(f'New A_bar_idio: {n_samples} samples')
    
    if n_samples == orig_A.shape[0]:
        print('✓ Sample count matches!')
    else:
        print(f'⚠ Mismatch: {n_samples} vs {orig_A.shape[0]}. Truncating/padding...')
        if n_samples > orig_A.shape[0]:
            A_bar_idio = A_bar_idio[:orig_A.shape[0]]
        else:
            pad = np.zeros((orig_A.shape[0] - n_samples, K, K))
            A_bar_idio = np.concatenate([A_bar_idio, pad], axis=0)
    
    print(f'A_bar_idio: shape={A_bar_idio.shape}, mean={A_bar_idio.mean():.4f}, '
          f'min={A_bar_idio.min():.4f}, max={A_bar_idio.max():.4f}')
    np.save(str(OUT_PATH), A_bar_idio)
    print(f'Saved: {OUT_PATH}')

if __name__ == '__main__':
    main()
