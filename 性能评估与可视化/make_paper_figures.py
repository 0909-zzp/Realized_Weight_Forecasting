"""Generate paper figures (priority high): weight dynamics, network degree,
cumulative net returns, lambda_s sensitivity.

Outputs to 论文/figures/*.png (300 dpi) and *.pdf.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

root = Path(__file__).parents[1]
out_dir = root / "论文" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(root / "图形Lasso" / "code"))
from 共享模块 import K, ETA

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_dir / name)

# ============================================================
# Figure 1: realized weight dynamics and turnover
# ============================================================
def fig_weights_turnover():
    ds = pd.read_csv(root / "图形Lasso" / "code" / "输出数据" / "Daily_Statistics.csv")
    w = pd.read_csv(root / "图形Lasso" / "code" / "输出数据" / "reg_weights_2436.csv", index_col=0)
    n = min(len(ds), len(w))
    ds, w = ds.iloc[:n], w.iloc[:n]
    dates = pd.to_datetime(ds["date"].astype(str), format="%Y%m%d")
    wv = w.values

    # 代表资产: 全样本平均 |w| 最大的前3只
    order = np.argsort(-np.abs(wv).mean(axis=0))[:3]
    med = np.median(wv, axis=1)
    p5 = np.percentile(wv, 5, axis=1)
    p95 = np.percentile(wv, 95, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.2), sharex=True)
    ax = axes[0]
    for i, idx in enumerate(order):
        ax.plot(dates, wv[:, idx], lw=0.8, label=f"Representative asset {idx+1}")
    ax.set_ylabel("Realized weight")
    ax.set_title("Realized GMVP weights and turnover")
    ax.legend(ncol=3, loc="upper right")

    ax = axes[1]
    ax.fill_between(dates, p5, p95, alpha=0.25, label="P5-P95 band")
    ax.plot(dates, med, color="black", lw=0.8, label="Median")
    ax.set_ylabel("Realized weight")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(dates, ds["turnover"], lw=0.6, color="tab:red")
    ax.axhline(ds["turnover"].mean(), color="black", ls="--", lw=1,
               label=f"Mean = {ds['turnover'].mean():.2f}")
    ax.set_ylabel("Daily turnover")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")

    for a in axes:
        a.grid(alpha=0.25)
    save(fig, "fig1_weights_turnover")

# ============================================================
# Figure 2: network degree distributions
# ============================================================
def fig_network_degree():
    npz = np.load(root / "图形Lasso" / "factor_bic_full_results.npz")
    raw = npz["raw_degs"][npz["raw_oks"]]
    idio = npz["idio_degs"][npz["idio_oks"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, data, color, title in [
        (axes[0], raw, "tab:blue", "Raw GLasso network"),
        (axes[1], idio, "tab:green", "Factor-decomposed network"),
    ]:
        ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="white")
        mean, med = data.mean(), np.median(data)
        ax.axvline(mean, color="black", ls="--", lw=1.2, label=f"Mean = {mean:.1f}")
        ax.axvline(med, color="black", ls=":", lw=1.2, label=f"Median = {med:.1f}")
        ax.set_xlabel("Network degree")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Days")
    fig.suptitle("Daily degree distribution of the realized asset network")
    save(fig, "fig2_network_degree")

# ============================================================
# helper: net daily returns on the 200-day window (same as Table 3)
# ============================================================
def load_window_net_returns():
    npy_dir = root / "数据" / "1min_log_return_npy"
    all_files = sorted([f for f in npy_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()])
    valid = np.load(root / "特征工程" / "valid_indices.npy")
    n_total = valid.shape[0]
    n_test = n_total - int(n_total * 0.7) - int(n_total * 0.15)
    win_idx = valid[-n_test:][65:265]
    files = [all_files[i] for i in win_idx]
    simple = np.array([np.expm1(np.load(f).sum(axis=1)) for f in files])
    dates = [f.name[:8] for f in files]
    return simple, dates

def net_returns(Y, simple):
    T = len(Y) - 1
    pr = np.array([Y[t] @ simple[t + 1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = Y[t]; rn = simple[t + 1]
        wd = wp * (1 + rn) / (1 + wp @ rn)
        to[t] = np.abs(Y[t + 1] - wd).sum()
    return pr - ETA * to

# ============================================================
# Figure 3: cumulative net returns
# ============================================================
def fig_cumulative_returns():
    simple, dates = load_window_net_returns()
    pd_dates = pd.to_datetime(dates[1:], format="%Y%m%d")
    varx = root / "VARX"

    specs = [
        ("Network+Smooth+DFL", varx / "Y_pred_model6_opt.npy", "tab:blue"),
        ("Network VARX", varx / "Y_pred_model4.npy", "tab:orange"),
        ("Sparse VARX", varx / "Y_pred_model3.npy", "tab:green"),
        ("Sample GMVP", varx / "Y_pred_bench_sample_W20.npy", "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for name, fpath, color in specs:
        Y = np.load(fpath)[-363:][65:265].astype(float)
        nr = net_returns(Y, simple)
        cum = np.cumprod(1 + nr) - 1
        ax.plot(pd_dates, cum * 100, lw=1.6, color=color,
                label=f"{name} ({cum[-1]*100:.1f}%)")

    Y_eq = np.ones((len(simple), K)) / K
    nr_eq = net_returns(Y_eq, simple)
    cum_eq = np.cumprod(1 + nr_eq) - 1
    ax.plot(pd_dates, cum_eq * 100, lw=1.6, color="black", ls="--",
            label=f"Equal Weight ({cum_eq[-1]*100:.1f}%)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return (%)")
    ax.set_title("Out-of-sample cumulative net returns (200-day window)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    save(fig, "fig3_cumulative_returns")

# ============================================================
# Figure 4: lambda_s sensitivity
# ============================================================
def fig_lambda_s():
    df = pd.read_csv(root / "性能评估与可视化" / "smooth_lambda_grid.csv")
    lam = df["lambda_s"].values
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    ax.semilogx(lam, df["base_turnover"], "o-", label="Base model turnover")
    ax.semilogx(lam, df["dfl_turnover"], "s-", label="DFL turnover")
    ax.set_xlabel("Smoothing penalty $\\lambda_s$")
    ax.set_ylabel("Average daily turnover")
    ax.legend(); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.semilogx(lam, df["sharpe"], "o-", color="tab:red")
    ax.axvline(3e-3, color="black", ls="--", lw=1, label="Selected $\\lambda_s = 3\\times 10^{-3}$")
    ax.set_xlabel("Smoothing penalty $\\lambda_s$")
    ax.set_ylabel("Net Sharpe ratio")
    ax.legend(); ax.grid(alpha=0.25)
    fig.suptitle("Turnover-smoothing sensitivity")
    save(fig, "fig4_lambda_s_sensitivity")

if __name__ == "__main__":
    fig_weights_turnover()
    fig_network_degree()
    fig_cumulative_returns()
    fig_lambda_s()
    print("figures ->", out_dir)
