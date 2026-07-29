"""GMVP 基准权重计算: Sample / Shrinkage / GLasso
用于 Table 3 补充基准模型, 统一使用 Table3 脚本的评估框架.
"""
import sys, numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[1] / '图形Lasso' / 'code'))
from 共享模块 import K, load_day, compute_raw_cov, EPS_RIDGE
from sklearn.covariance import LedoitWolf, graphical_lasso

# 参数
W = 40           # 滚动窗口天数
LAM_GL = 3e-6    # GLasso 正则化参数

# 路径
feat = Path(__file__).parents[1] / '特征工程'
varx = Path(__file__).parents[1] / 'VARX'
valid_idx = np.load(feat / 'valid_indices.npy')
n = len(np.load(feat / 'Y_targets.npy'))
nt = int(0.7 * n)
nv = int(0.15 * n)
ntotal = nt + nv

# 全测试期 + 前面W天缓冲
test_days_all = valid_idx[ntotal : ntotal + 363]
buf_days = valid_idx[ntotal - W : ntotal]

# 日收益(用于Shrinkage)
daily_r = np.zeros((363, K))
for i, didx in enumerate(test_days_all):
    daily_r[i] = load_day(didx).sum(axis=1)

print(f'计算 GMVP 基准权重 (W={W}天, λ_GL={LAM_GL:.0e})')
print(f'  测试期: {363}天, 缓冲: {W}天')

w_sample = np.zeros((363, K))
w_shrink = np.zeros((363, K))
w_glasso = np.zeros((363, K))

for t in range(363):
    # 滚动窗口日内协方差
    cov_sum = np.zeros((K, K))
    for j in range(W):
        tidx = t - W + 1 + j
        if tidx < 0:
            didx = buf_days[tidx]
        else:
            didx = test_days_all[tidx]
        rett = load_day(didx)
        raw = compute_raw_cov(rett)
        raw.flat[::K+1] += EPS_RIDGE
        cov_sum += raw
    cov = cov_sum / W

    # --- Sample GMVP ---
    try:
        w = np.linalg.solve(cov, np.ones(K)); w = w / w.sum()
    except:
        w = np.ones(K) / K
    w_sample[t] = w

    # --- Shrinkage GMVP (Ledoit-Wolf on daily returns) ---
    R = np.zeros((W, K))
    for j in range(W):
        tidx = t - W + 1 + j
        if tidx < 0:
            R[j] = load_day(buf_days[tidx]).sum(axis=1)
        else:
            R[j] = daily_r[tidx]
    try:
        lw = LedoitWolf().fit(R)
        cov_sh = lw.covariance_ + EPS_RIDGE * np.eye(K)
        w = np.linalg.solve(cov_sh, np.ones(K)); w = w / w.sum()
    except:
        w = np.ones(K) / K
    w_shrink[t] = w

    # --- GLasso GMVP ---
    try:
        _, prec = graphical_lasso(cov, alpha=LAM_GL, mode='cd', tol=1e-4, max_iter=100)
        w = prec @ np.ones(K); w = w / w.sum()
    except:
        w = np.ones(K) / K
    w_glasso[t] = w

    if t % 100 == 0:
        print(f'  {t}/363')

# 保存
np.save(varx / 'Y_pred_bench_sample.npy', w_sample)
np.save(varx / 'Y_pred_bench_shrink.npy', w_shrink)
np.save(varx / 'Y_pred_bench_glasso.npy', w_glasso)
print(f'已保存: Y_pred_bench_sample/shrink/glasso.npy')
print('完成')
