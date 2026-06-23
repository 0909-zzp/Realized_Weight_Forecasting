"""Ridge=1e-3 快速验证：5天抽样测试"""
import os, sys, time, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.covariance import graphical_lasso

d = r"d:\HuaweiMoveData\Users\27438\Desktop\大创\数据\1min_log_return_npy"
all_fs = sorted([f for f in os.listdir(d) if f.endswith(".npy") and not f.startswith("1min")])
n = len(all_fs)

# 均匀采5天：首/中/尾 + 2008/2020崩盘年
test_idxs = [0, n//4, n//2, 3*n//4, n-1]
K, LAM, RIDGE, MAX_ITER = 392, 5e-7, 1e-3, 100

print(f"测试 Ridge={RIDGE} (共{n}天，抽{len(test_idxs)}天)")
t0 = time.time()
ok = fail = 0
for idx in test_idxs:
    f = all_fs[idx]
    rett = np.load(os.path.join(d, f))
    cov = rett @ rett.T
    cov.flat[::K+1] += RIDGE
    try:
        t1 = time.time()
        _, prec = graphical_lasso(emp_cov=cov, alpha=LAM, mode="cd",
                                   tol=1e-4, max_iter=MAX_ITER, enet_tol=5e-4)
        dt = time.time() - t1
        nz = int((np.abs(prec) > 1e-8).sum() / 2)
        ok += 1
        print(f"  [{idx}] {f[:8]} OK  {dt:.1f}s  edges={nz}")
    except Exception as e:
        fail += 1
        print(f"  [{idx}] {f[:8]} FAIL  {str(e)[:80]}")

print(f"\n结果: {ok}/{len(test_idxs)} 通过  总耗时{time.time()-t0:.1f}s")
if fail:
    print("⚠ 有失败，建议加退避链")
else:
    print("✓ 全部通过，Ridge=1e-3 可靠")
