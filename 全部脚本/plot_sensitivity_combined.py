"""M4 参数敏感性 — 综合可视化 (热力图 + 排序曲线)"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path(__file__).parents[1] / 'VARX' / 'tuning_summary.csv')
m4 = df[df['model'] == 'M4'].copy()
m4['mse_scaled'] = m4['mean_val_mse'] * 1e5
best = m4.loc[m4['mean_val_mse'].idxmin()]
sorted_mse = m4.sort_values('mean_val_mse')['mse_scaled'].values
m3a_best = 1.8075  # M3a best validation MSE (×1e-5) from final_params.json

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.rcParams.update({'font.size': 10})

# ====== LEFT: 热力图 λ₁ × τ (固定 λ_net = 1e-3, last_day = False) ======
ax = axes[0]
pivot = m4[(m4['LAMBDA_NETWORK']==1e-3) & (m4['USE_LAST_DAY']==False)]
pivot_table = pivot.pivot_table(
    values='mse_scaled', index='LAMBDA_LASSO', columns='NETWORK_THRESHOLD', aggfunc='mean'
)
im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn_r',
               extent=[0.7, 0.9, np.log10(5e-4), np.log10(1e-4)])
ax.set_xticks([0.7, 0.8, 0.85, 0.9])
ax.set_yticks([np.log10(5e-4), np.log10(3e-4), np.log10(1e-4)])
ax.set_yticklabels(['5e-4', '3e-4', '1e-4'])
ax.set_xlabel('tau (threshold)')
ax.set_ylabel('lambda1')
ax.set_title('MSE (x1e-5) vs lambda1 & tau\n(fixed: lambda_net=1e-3)')
plt.colorbar(im, ax=ax, shrink=0.8)

# Mark optimum
opt_tau = float(best['NETWORK_THRESHOLD'])
opt_lam = float(best['LAMBDA_LASSO'])
ax.plot(opt_tau, np.log10(opt_lam), 'k*', markersize=15, markeredgewidth=1.5, 
        markeredgecolor='white')

# ====== RIGHT: 排序折线, 颜色区分 λ₁ 组, 附 M3a 参照线 ======
ax = axes[1]
m4_sorted = m4.sort_values('mean_val_mse').reset_index(drop=True)
for lam, c, lb in [(1e-4, '#e74c3c', 'lambda1=1e-4 (weak)'), 
                      (5e-4, '#2980b9', 'lambda1=5e-4 (strong)'),
                      (3e-4, '#27ae60', 'lambda1=3e-4 (optimal)')]:
    mask = m4_sorted['LAMBDA_LASSO'] == lam
    vals = m4_sorted.loc[mask, 'mse_scaled'].values
    # Connect points within same lambda group
    idx = np.where(mask)[0]
    ax.plot(idx + 1, vals, color=c, linewidth=1.2, alpha=0.8, label=lb)
    ax.scatter(idx + 1, vals, color=c, s=12, alpha=0.6, edgecolors='none')

ax.axhline(y=m3a_best, color='orange', linestyle=':', linewidth=1.5, 
           label=f'M3a best: {m3a_best:.2f}')
ax.set_xlabel('Parameter combination (sorted by MSE)')
ax.set_ylabel('Validation MSE (\u00d71e-5)')
ax.set_title('M4: 120 grid-search combinations')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / 'M4_sensitivity_combined.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
