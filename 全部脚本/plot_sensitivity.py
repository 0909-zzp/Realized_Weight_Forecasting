"""可视化 M4 参数敏感性 — 120 组网格搜索结果"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

df = pd.read_csv(Path(__file__).parents[1] / 'VARX' / 'tuning_summary.csv')
m4 = df[df['model'] == 'M4'].copy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
plt.rcParams.update({'font.size': 11})

# Panel 1: MSE vs lambda1, colored by tau
ax = axes[0]
for tau in sorted(m4['NETWORK_THRESHOLD'].unique()):
    sub = m4[m4['NETWORK_THRESHOLD'] == tau]
    means = sub.groupby('LAMBDA_LASSO')['mean_val_mse'].mean()
    ax.plot([np.log10(v) for v in means.index], means.values * 1e5, 
            'o-', label=f'tau={tau:.2f}', markersize=6)
ax.set_xlabel('log10(lambda1)')
ax.set_ylabel('Validation MSE (x1e-5)')
ax.set_title('M4: MSE vs lambda1')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 2: MSE vs tau, colored by lambda1
ax = axes[1]
for lam in sorted(m4['LAMBDA_LASSO'].unique()):
    sub = m4[m4['LAMBDA_LASSO'] == lam]
    means = sub.groupby('NETWORK_THRESHOLD')['mean_val_mse'].mean()
    ax.plot(means.index, means.values * 1e5, 's-', 
            label=f'lam1={lam:.0e}', markersize=6)
ax.set_xlabel('tau (threshold)')
ax.set_ylabel('Validation MSE (x1e-5)')
ax.set_title('M4: MSE vs tau')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: MSE vs lambda_net, colored by tau
ax = axes[2]
for tau in sorted(m4['NETWORK_THRESHOLD'].unique()):
    sub = m4[m4['NETWORK_THRESHOLD'] == tau]
    means = sub.groupby('LAMBDA_NETWORK')['mean_val_mse'].mean()
    ax.plot([np.log10(v) for v in means.index], means.values * 1e5, 
            'D-', label=f'tau={tau:.2f}', markersize=6)
ax.set_xlabel('log10(lambda_net)')
ax.set_ylabel('Validation MSE (x1e-5)')
ax.set_title('M4: MSE vs lambda_net')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / 'M4_sensitivity.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
