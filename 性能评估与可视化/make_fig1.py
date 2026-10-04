import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]
out_dir = root / "论文" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

K = 392
TAU = 0.117

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

# ---- data sources ----
w = pd.read_csv(root / "图形Lasso" / "code" / "输出数据" / "reg_weights_2436.csv", index_col=0).values.astype(float)
A = np.load(root / "特征工程" / "A_bar_idio.npy")   # (2415, 392, 392)

# ---- (a) realized-weight autocorrelation (5-95% band across assets) ----
T = w.shape[0]
maxlag = 300
wc = w - np.nanmean(w, axis=0)
pct = {p: [] for p in (5, 50, 95)}
for lag in range(1, maxlag + 1):
    a = wc[:T-lag]; b = wc[lag:]
    num = np.nansum(a * b, axis=0)
    den = np.sqrt(np.nansum(a**2, axis=0)) * np.sqrt(np.nansum(b**2, axis=0))
    acf = num / np.where(den > 0, den, np.nan)
    acf = acf[np.isfinite(acf)]
    pct[5].append(np.percentile(acf, 5))
    pct[50].append(np.percentile(acf, 50))
    pct[95].append(np.percentile(acf, 95))
lags = np.arange(1, maxlag + 1)

# ---- (b) daily realized-weight turnover ----
diff = np.abs(np.diff(w, axis=0)).sum(axis=1)
to_valid = diff[np.isfinite(diff)]

# ---- (c) daily network density (raw off-diagonal mean of A_bar_idio) ----
iu = np.triu_indices(K, 1)
density = A[:, iu[0], iu[1]].mean(axis=1)

# ---- (d) edge persistence ----
M = A > TAU
E = M[:, iu[0], iu[1]]
persist = (E[:-1] & E[1:]).sum(axis=1) / np.maximum(E[:-1].sum(axis=1), 1)

fig, axs = plt.subplots(2, 2, figsize=(10, 7.2))
ax_a, ax_b, ax_c, ax_d = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

# (a)
ax_a.fill_between(lags, pct[5], pct[95], alpha=0.25, color="tab:blue", label="5\u201395% band")
ax_a.plot(lags, pct[50], color="tab:blue", lw=1.2, label="Median")
ax_a.set_xlabel("Lag")
ax_a.set_ylabel("Weight autocorrelation")
ax_a.set_title("(a) Realized-weight autocorrelation")
ax_a.set_ylim(min(pct[5]) - 0.04, max(pct[95]) + 0.04)
ax_a.legend(loc="upper right", frameon=False, fontsize=8)

# (b)
x_b = np.arange(len(to_valid))
ax_b.plot(x_b, to_valid, lw=0.6, color="tab:red")
ax_b.axhline(to_valid.mean(), color="black", ls="--", lw=1.0, label=f"Mean = {to_valid.mean():.4f}")
ax_b.set_xlabel("Trading day")
ax_b.set_ylabel("Turnover")
ax_b.set_title("(b) Daily realized-weight turnover")
ax_b.legend(loc="upper right", frameon=False, fontsize=8)

# (c)
x_c = np.arange(len(density))
ax_c.plot(x_c, density, lw=0.6, color="tab:green")
ax_c.axhline(density.mean(), color="black", ls="--", lw=1.0, label=f"Mean = {density.mean():.4f}")
ax_c.set_xlabel("Trading day")
ax_c.set_ylabel("Density")
ax_c.set_title("(c) Daily network density")
ax_c.legend(loc="upper right", frameon=False, fontsize=8)

# (d)
x_d = np.arange(len(persist))
ax_d.plot(x_d, persist, lw=0.6, color="tab:purple")
ax_d.axhline(persist.mean(), color="black", ls="--", lw=1.0, label=f"Mean = {persist.mean():.4f}")
ax_d.set_xlabel("Trading day")
ax_d.set_ylabel("Persistence")
ax_d.set_title(f"(d) Edge persistence, \u03c4 = {TAU}")
ax_d.legend(loc="upper right", frameon=False, fontsize=8)

for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.grid(False)

for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"fig1_weights_turnover_idio.{ext}", bbox_inches="tight")
plt.close(fig)

print("turnover mean =", to_valid.mean())
print("density mean =", density.mean())
print("persistence mean =", persist.mean())
print("saved ->", out_dir / "fig1_weights_turnover_idio")
