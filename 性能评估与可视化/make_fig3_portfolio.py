import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

root = Path(__file__).parents[1]
out_dir = root / "论文" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

K = 392
ETA = 1e-4  # one-way cost (1 bp)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

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

def daily_series(Y, simple):
    """Return net returns, daily turnover."""
    T = len(Y) - 1
    pr = np.array([Y[t] @ simple[t + 1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = Y[t]; rn = simple[t + 1]
        wd = wp * (1 + rn) / (1 + wp @ rn)
        to[t] = np.abs(Y[t + 1] - wd).sum()
    return pr - ETA * to, to

def main():
    simple, dates = load_window_net_returns()
    pd_dates = pd.to_datetime(dates[1:], format="%Y%m%d")  # aligned to net-return series

    varx = root / "VARX"
    specs = [
        ("Network VARX-S-DF", varx / "Y_pred_model6_opt.npy", "tab:blue"),
        ("Network VARX",       varx / "Y_pred_model4.npy", "tab:orange"),
        ("Sparse VARX",        varx / "Y_pred_model3.npy", "tab:green"),
        ("Sample GMVP",        varx / "Y_pred_bench_sample_W20.npy", "tab:red"),
        ("Equal weight",       None, "black"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 8.4), sharex=True)

    for name, fpath, color in specs:
        if fpath is None:
            Y = np.ones((len(simple), K)) / K
        else:
            Y = np.load(fpath)[-363:][65:265].astype(float)
        nr, to = daily_series(Y, simple)
        wealth = np.cumprod(1 + nr)
        dd = wealth / np.maximum.accumulate(wealth) - 1
        roll = pd.Series(to).rolling(20, min_periods=1).mean().values

        axes[0].plot(pd_dates, wealth, lw=1.6, color=color,
                     label=name)
        axes[1].plot(pd_dates, dd, lw=1.4, color=color, label=name)
        axes[2].plot(pd_dates, roll, lw=1.4, color=color, label=name)

    axes[0].set_ylabel("Cumulative net wealth")
    axes[1].set_ylabel("Drawdown")
    axes[2].set_ylabel("Rolling 20-day turnover")
    axes[2].set_xlabel("Date")

    axes[0].set_title("Out-of-sample portfolio performance")
    axes[0].legend(ncol=2, loc="upper left", frameon=False, fontsize=8)
    axes[1].legend(ncol=2, loc="upper right", frameon=False, fontsize=8)
    axes[2].legend(ncol=2, loc="upper right", frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[2].get_xticklabels(), rotation=0, ha="center")

    fig.align_ylabels(axes)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_portfolio_over_time.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out_dir / "fig3_portfolio_over_time")

if __name__ == "__main__":
    main()

