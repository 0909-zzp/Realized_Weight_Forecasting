import os, sys, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
NPY_DIR = ROOT / '数据' / '1min_log_return_npy'
OUT_BASE = ROOT / 'table5_alt'
K = 392
EPS = 1e-4
RIDGE_CHAIN = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
BASELINE_LAM = 1e-6
ALT_LAM = 1e-5
COV_WINDOW = 20

def do_glasso(cov, lam):
    from sklearn.covariance import graphical_lasso
    for r in RIDGE_CHAIN:
        c = cov.copy()
        c.flat[::K+1] += (r - EPS) if r > EPS else 0.0
        try:
            _, prec = graphical_lasso(c, alpha=lam, mode='cd', tol=1e-4,
                                      max_iter=100, enet_tol=5e-4)
            return prec
        except Exception:
            continue
    c = cov.copy()
    c.flat[::K+1] += 1e-1
    try:
        _, prec = graphical_lasso(c, alpha=lam, mode='cd', tol=5e-4,
                                  max_iter=100, enet_tol=5e-4)
        return prec
    except Exception:
        return None

def precision_worker(args):
    i, fpath, lam = args
    rett = np.load(fpath)
    raw = rett @ rett.T
    raw.flat[::K+1] += EPS
    prec = do_glasso(raw, lam)
    if prec is None:
        return i, np.full(K, np.nan), np.zeros((K, K), dtype=np.int8), False
    w = prec @ np.ones(K)
    w = w / w.sum()
    adj = (np.abs(prec) > 1e-8).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return i, w, adj, True

def covariance_worker(args):
    i, fpaths, lam = args
    acc = np.zeros((K, K), dtype=np.float64)
    for fpath in fpaths:
        rett = np.load(fpath)
        raw = rett @ rett.T
        raw.flat[::K+1] += EPS
        acc += raw
    cov = acc / len(fpaths)
    prec = do_glasso(cov, lam)
    if prec is None:
        return i, np.full(K, np.nan), np.zeros((K, K), dtype=np.int8), False
    w = prec @ np.ones(K)
    w = w / w.sum()
    adj = (np.abs(prec) > 1e-8).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return i, w, adj, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['precision','covariance'], required=True)
    ap.add_argument('--lam', type=float, default=ALT_LAM)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    files = sorted([f for f in NPY_DIR.iterdir() if f.suffix == '.npy' and f.name[0].isdigit()])
    if args.limit:
        files = files[:args.limit]
    dates = [f.stem[:8] for f in files]
    tag = f'{args.mode}_lam{args.lam:.0e}'
    outdir = OUT_BASE / tag
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'mode={args.mode} lam={args.lam:.0e} days={len(files)} workers={args.workers}', flush=True)
    tasks = []
    for i, f in enumerate(files):
        if args.mode == 'precision':
            tasks.append((i, str(f), args.lam))
        else:
            lo = max(0, i - COV_WINDOW + 1)
            tasks.append((i, [str(files[j]) for j in range(lo, i+1)], args.lam))
    W = np.full((len(files), K), np.nan)
    ADJ = np.zeros((len(files), K, K), dtype=np.int8)
    OK = [False] * len(files)
    t0 = time.time()
    done = 0
    worker = precision_worker if args.mode == 'precision' else covariance_worker
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, task): task[0] for task in tasks}
        for fut in as_completed(futs):
            i, w, adj, ok = fut.result()
            W[i] = w; ADJ[i] = adj; OK[i] = ok
            done += 1
            if done % 200 == 0 or done == len(files):
                print(f'  {done}/{len(files)} fails={sum(not x for x in OK[:done])} elapsed={time.time()-t0:.0f}s', flush=True)
    np.save(outdir / 'weights.npy', W)
    for i in range(len(files)):
        np.save(outdir / 'adjacency' / f'{dates[i]}.npy', ADJ[i]) if False else None
    adjdir = outdir / 'adjacency'; adjdir.mkdir(exist_ok=True)
    for i in range(len(files)):
        np.save(adjdir / f'{dates[i]}.npy', ADJ[i])
    pd = __import__('pandas')
    pd.DataFrame(W, index=dates, columns=[f'A_{j}' for j in range(K)]).to_csv(outdir / 'reg_weights_2436.csv')
    pd.DataFrame({'date': dates, 'success': OK}).to_csv(outdir / 'Daily_Statistics.csv', index=False)
    print('DONE', outdir, 'fails', sum(not x for x in OK), 'elapsed', time.time()-t0, flush=True)

if __name__ == '__main__':
    main()
