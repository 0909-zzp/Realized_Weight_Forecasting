"""快速诊断：测试 graphical_lasso 是否正常。"""
import numpy as np
K = 392

# 构造合法协方差矩阵
rng = np.random.default_rng(42)
X = rng.normal(0, 0.01, (K, 400))
cov = X @ X.T + np.eye(K) * 5e-5

print(f"cov shape={cov.shape}, diag range=[{np.min(np.diag(cov)):.2e}, {np.max(np.diag(cov)):.2e}]")

try:
    from sklearn.covariance import graphical_lasso
    print("✓ graphical_lasso 可导入")
except ImportError as e:
    print(f"✗ 导入失败: {e}"); exit(1)

for lam in [1e-2, 1e-4, 1e-6, 1e-8]:
    try:
        cov_est, prec = graphical_lasso(emp_cov=cov, alpha=lam, mode='cd', max_iter=2000, tol=1e-5)
        nz = np.sum(np.abs(np.triu(prec, k=1)) > 1e-8)
        print(f"✓ λ={lam:.0e}: prec shape={prec.shape}, non-zero edges={nz}")
    except Exception as e:
        print(f"✗ λ={lam:.0e}: {type(e).__name__}: {e}")

# 测试 cov_init
print("\n--- 测试 cov_init ---")
try:
    cov_est1, _ = graphical_lasso(emp_cov=cov, alpha=1e-2, mode='cd', max_iter=2000, tol=1e-5)
    cov_est2, prec2 = graphical_lasso(emp_cov=cov, alpha=1e-4, mode='cd', max_iter=2000, tol=1e-5,
                                       cov_init=cov_est1)
    print(f"✓ cov_init 可用, prec shape={prec2.shape}")
except TypeError as e:
    print(f"✗ cov_init 参数不支持: {e}")
except Exception as e:
    print(f"✗ 其他错误: {type(e).__name__}: {e}")
