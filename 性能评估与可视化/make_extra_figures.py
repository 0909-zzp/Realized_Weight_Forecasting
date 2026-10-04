"""Generate extra paper figures: weight persistence (ACF), ablation contribution,
market-state bars, drawdown curves. Outputs to 论文/figures.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.cwd()
out_dir = root / "论文" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("mk", root / "性能评估与可视化" / "make_paper_figures.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_dir / name)

def acf(x, max_lag=30):
    x = x - x.mean()
    v = np.dot(x, x)
    out = []
    for k in range(max_lag + 1):
        out.append(np.dot(x[k:], x[:-k]) / v if k > 0 else 1.0)
    return np.array(out)

def fig_weight_acf():
    w = pd.read_csv(root / "图形Lasso" / "code" / "输出数据" / "reg_weights_2436.csv", index_col=0)
    wv = w.ffill(axis=0).bfill(axis=0).values  # 少量缺失日前后填充, 保持连续时序
    order = np.argsort(-np.abs(wv).mean(axis=0))[:3]
    lags = np.arange(31)
    avg = np.mean([acf(wv[:, i]) for i in range(wv.shape[1])], axis=0)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, idx in enumerate(order):
        ax.plot(lags, acf(wv[:, idx]), lw=1.4, label=f"Representative asset {idx+1}")
    ax.plot(lags, avg, "k--", lw=1.6, label="Cross-sectional average")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Persistence of realized GMVP weights")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "fig5_weight_acf")

EN_LABELS = {
    "网络VARX + 平滑 + DFL": "Network VARX + Smooth + DFL",
    "网络VARX + 平滑（无外生）+ DFL": "Network VARX + Smooth (no exog.) + DFL",
    "稀疏VARX + 平滑（无网络）+ DFL": "Sparse VARX + Smooth (no network) + DFL",
    "网络VARX + DFL": "Network VARX + DFL",
    "网络VARX + 平滑": "Network VARX + Smooth",
}

def fig_ablation():
    df = pd.read_csv(root / "消融分析" / "Table4_完整.csv")
    labels = [EN_LABELS.get(x, x) for x in df["设定"].tolist()]
    sharpe = df["Sharpe_net"].tolist()
    to = df["Turnover"].tolist()
    colors = ["tab:blue" if "DFL" in l and "无" not in l and "平滑" in l else "tab:grey" for l in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].barh(labels, sharpe, color=colors)
    axes[0].set_xlabel("Net Sharpe ratio")
    axes[0].set_title("Ablation: net Sharpe")
    axes[0].invert_yaxis()
    axes[1].barh(labels, to, color="tab:orange", alpha=0.85)
    axes[1].set_xlabel("Average daily turnover")
    axes[1].set_title("Ablation: turnover")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    save(fig, "fig6_ablation")

def fig_market_state():
    df = pd.read_csv(root / "性能评估与可视化" / "table5_market_state.csv")
    models = df["model"].unique()
    low = {m: df[(df.model == m) & (df.state == "low_vol")]["mean_daily"].iloc[0] * 100
           for m in models}
    high = {m: df[(df.model == m) & (df.state == "high_vol")]["mean_daily"].iloc[0] * 100
            for m in models}
    names = list(models)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.bar(x - 0.2, [low[m] for m in names], 0.4, label="Low-volatility state")
    ax.bar(x + 0.2, [high[m] for m in names], 0.4, label="High-volatility state")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("Daily net return (%)")
    ax.set_title("Net returns by VIX-based volatility state")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig7_market_state")

def fig_drawdown():
    simple, dates = mk.load_window_net_returns()
    pd_dates = pd.to_datetime(dates[1:], format="%Y%m%d")
    varx = root / "VARX"
    specs = [
        ("Network+Smooth+DFL", varx / "Y_pred_model6_opt.npy", "tab:blue"),
        ("Network VARX", varx / "Y_pred_model4.npy", "tab:orange"),
        ("Sparse VARX", varx / "Y_pred_model3.npy", "tab:green"),
        ("Sample GMVP", varx / "Y_pred_bench_sample_W20.npy", "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for name, fpath, color in specs:
        Y = np.load(fpath)[-363:][65:265].astype(float)
        nr = mk.net_returns(Y, simple)
        cum = np.cumprod(1 + nr)
        dd = cum / np.maximum.accumulate(cum) - 1
        ax.plot(pd_dates, dd * 100, lw=1.3, color=color, label=name)
    Y_eq = np.ones((len(simple), 392)) / 392
    nr = mk.net_returns(Y_eq, simple)
    cum = np.cumprod(1 + nr)
    dd = cum / np.maximum.accumulate(cum) - 1
    ax.plot(pd_dates, dd * 100, "k--", lw=1.3, label="Equal Weight")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Out-of-sample drawdown (200-day window)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    save(fig, "fig8_drawdown")

if __name__ == "__main__":
    fig_weight_acf()
    fig_ablation()
    fig_market_state()
    fig_drawdown()
    print("extra figures ->", out_dir)
