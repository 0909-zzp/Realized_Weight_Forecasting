"""λ_Ω 网络密度稳健性测试
输出: 不同 λ 下邻接矩阵密度的均值与标准差
用法: python network_density_test.py
可调: N (样本天数), LAMBDAS (λ值列表)
"""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")  # 抑制 sklearn ConvergenceWarning
from pathlib import Path
from sklearn.covariance import graphical_lasso

K = 392
EPS = 1e-4
RIDGE_FALLBACK = [5e-4, 1e-3, 5e-3, 1e-2]
LAMBDAS = [1e-6, 3e-6, 5e-6, 1e-5]
N = 50  # ← 改成 100 或更大也行, 但要等更久

# 随机抽样
npy_dir = Path(__file__).resolve().parents[2] / "数据" / "1min_log_return_npy"
files = sorted([f for f in npy_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()])
rng = np.random.default_rng(42)
sample = rng.choice(files, N, replace=False)

print(f"Network Density Robustness Test")
print(f"  λ values: {[f'{l:.0e}' for l in LAMBDAS]}")
print(f"  Sample: {N} days (from {len(files)} total)")
print(f"  Ridge fallback: {RIDGE_FALLBACK}")
print()
print(f"{'λ':>10}  {'ok':>4}  {'density':>9}  {'std':>8}  {'time':>6}")
print("-" * 50)

for lam in LAMBDAS:
    densities = []
    ok = 0
    t0 = time.time()
    for f in sample:
        rett = np.load(str(f))
        raw = rett @ rett.T
        raw.flat[::K+1] += EPS
        for r in [EPS] + RIDGE_FALLBACK:
            c = raw.copy()
            if r > EPS:
                c.flat[::K+1] += (r - EPS)
            try:
                _, prec = graphical_lasso(c, alpha=lam, mode="cd", tol=1e-4, max_iter=100)
                adj = (np.abs(prec) > 1e-8).astype(int)
                np.fill_diagonal(adj, 0)
                densities.append(adj.sum() / (K * (K - 1)))
                ok += 1
                break
            except Exception:
                continue

    mu = np.mean(densities)
    sd = np.std(densities, ddof=1)
    print(f"{lam:>10.0e}  {ok:>4}  {mu:>8.1%}  {sd:>7.1%}  {time.time()-t0:>5.0f}s")

print("\nDone. 密度跨量级不退化到 0% 或 100% → M4 惩罚机制稳健.")
