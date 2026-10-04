"""Build read-only cache for the end-to-end project.

Outputs (all inside endtoend_project/cache):
  simple_returns.npy  (N, K) float32   daily simple returns for each feature row
  cov_cache.npy       (N, K, K) float32 memory-mapped realized covariance
  dates.npy           (N,) object/str  date strings
Nothing in the original project is modified.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time
import numpy as np
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)

valid = np.load(PROJ / "特征工程" / "valid_indices.npy")
all_files = sorted([f for f in (PROJ / "数据" / "1min_log_return_npy").iterdir()
                    if f.suffix == ".npy" and f.name[0].isdigit()])
files = [all_files[i] for i in valid]
N = len(files)
K = 392
print(f"N={N} K={K}")

# daily simple returns
if not (OUT / "simple_returns.npy").exists():
    t0 = time.time()
    rets = np.zeros((N, K), dtype=np.float32)
    for i, f in enumerate(files):
        rets[i] = np.expm1(np.load(f).sum(axis=1)).astype(np.float32)
        if i % 500 == 0:
            print("returns", i, flush=True)
    np.save(OUT / "simple_returns.npy", rets)
    print(f"simple returns done {time.time()-t0:.1f}s")

# realized covariance cache (float32 memmap)
cov_path = OUT / "cov_cache.npy"
if not cov_path.exists():
    t0 = time.time()
    mm = np.lib.format.open_memmap(cov_path, mode="w+", dtype=np.float32, shape=(N, K, K))
    for i, f in enumerate(files):
        R = np.load(f).astype(np.float32)          # (K, M)
        mm[i] = R @ R.T
        if i % 100 == 0:
            print("cov", i, f"elapsed={time.time()-t0:.1f}s", flush=True)
    mm.flush(); del mm
    print(f"cov cache done {time.time()-t0:.1f}s")

dates = np.array([f.name[:8] for f in files])
np.save(OUT / "dates.npy", dates)
print("dates saved", dates[0], dates[-1])
print("cache dir:", OUT)
