"""Experiment: drawdown-control overlays on the DFL portfolio.

This script does NOT modify any Table3 output. It evaluates overlay strategies
on top of the existing DFL weights (Y_pred_model6_opt.npy):

  1. base            : the current DFL portfolio (reproduction check)
  2. vol_scale       : Moreira-Muir style volatility targeting
                       exposure = clamp(target_vol / realized_vol, 0, max_scale)
  3. dd_brake        : drawdown-aware deleveraging
                       exposure declines linearly once drawdown < -dd_target
  4. dd_brake_smooth : same rule with one-day smoothing to reduce whipsaw
  5. vol+dd          : combination of volatility targeting and drawdown brake

Metrics follow Table3 conventions: 1bp cost is applied to the exact risky
trading volume, which includes both weight rebalancing and exposure changes.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log, set_log_file, compute_raw_cov, EPS_RIDGE


START_DATE = "20181219"
END_DATE = "20191008"
MAIN_DFL_START = 60
WINDOW = 200
DFL_FILE = Path(__file__).parents[1] / "VARX" / "Y_pred_model6_opt.npy"
OUT_CSV = Path(__file__).with_name("drawdown_control_experiment.csv")
ROBUST_CSV = Path(__file__).with_name("drawdown_control_robustness.csv")
PLOT_PNG = Path(__file__).with_name("drawdown_control_compare.png")


def load_all_days():
    """Load the full 363-day DFL test window once, then slice per window."""
    root = Path(__file__).parents[1]
    npy_dir = root / "数据" / "1min_log_return_npy"
    all_files = sorted(
        [f for f in npy_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()]
    )
    all_dates = [f.name[:8] for f in all_files]
    si = all_dates.index(START_DATE)
    d0 = si - MAIN_DFL_START
    files = all_files[d0 : d0 + len(np.load(DFL_FILE))]

    simple_rets = np.zeros((len(files), K), dtype=np.float64)
    covs = []
    for i, fpath in enumerate(files):
        rett = np.load(str(fpath))
        simple_rets[i] = np.expm1(rett.sum(axis=1))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs.append(raw)
    Y = np.load(DFL_FILE).astype(np.float64)
    return simple_rets, covs, Y


def base_series(simple_rets, covs, Y):
    T = len(Y) - 1
    ret = np.array([Y[t] @ simple_rets[t + 1] for t in range(T)])
    to_vec = np.zeros(T)
    rpv_vec = np.zeros(T)
    for t in range(T):
        w_prev = Y[t]
        r_next = simple_rets[t + 1]
        w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
        to_vec[t] = np.sum(np.abs(Y[t + 1] - w_drifted))
        rpv_vec[t] = Y[t] @ covs[t + 1] @ Y[t]
    return ret, to_vec, rpv_vec


def metrics(name, ret, to_eff, rpv_eff, mean_scale):
    net = ret - ETA * to_eff
    vol = float(np.std(ret, ddof=1)) * np.sqrt(252)
    sr = (
        float(np.mean(net) / np.std(net, ddof=1)) * np.sqrt(252)
        if np.std(net) > 1e-15
        else 0.0
    )
    cum = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak))
    return {
        "name": name,
        "vol_annual": vol,
        "rpv_annual": float(np.mean(rpv_eff)) * 252,
        "turnover_eff": float(np.mean(to_eff)),
        "sharpe_net": sr,
        "max_dd": dd,
        "cum_ret": float(np.prod(1 + ret) - 1),
        "mean_scale": mean_scale,
    }


def vol_scale_series(base_ret, win, target_vol, max_scale):
    T = len(base_ret)
    scale = np.ones(T + 1)
    for t in range(1, T + 1):
        lookback = min(win, t)
        if lookback >= 2:
            rv = np.std(base_ret[t - lookback : t], ddof=1) * np.sqrt(252)
            scale[t] = float(np.clip(target_vol / rv, 0.0, max_scale))
    return scale


def _brake_rule(dd, dd_target):
    if dd < -dd_target:
        return float(np.clip(1.0 - (-dd - dd_target) / dd_target, 0.0, 1.0))
    return 1.0


def dd_brake_scale(base_ret, dd_target, alpha=1.0):
    T = len(base_ret)
    scale = np.ones(T + 1)
    nav, peak, dd = 1.0, 1.0, 0.0
    prev_scale = 1.0
    for t in range(T):
        target = _brake_rule(dd, dd_target)
        scale[t] = alpha * target + (1.0 - alpha) * prev_scale
        r = scale[t] * base_ret[t]
        nav *= 1.0 + r
        peak = max(peak, nav)
        dd = nav / peak - 1.0
        prev_scale = scale[t]
    return scale


def apply_scale(simple_rets, Y, base_ret, rpv_vec, scale):
    """Apply exposure scale and compute exact risky turnover including scale changes."""
    T = len(base_ret)
    ret = scale[:T] * base_ret
    to_eff = np.zeros(T)
    for t in range(T):
        w_prev = Y[t]
        r_next = simple_rets[t + 1]
        w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
        w_ret = w_prev @ r_next
        sleeve_frac = scale[t] * (1.0 + w_ret) / (1.0 + scale[t] * w_ret)
        to_eff[t] = np.sum(np.abs(scale[t + 1] * Y[t + 1] - sleeve_frac * w_drifted))
    rpv_eff = scale[:T] ** 2 * rpv_vec
    return ret, to_eff, rpv_eff


def evaluate(simple_rets, covs, Y, variant):
    base_ret, to_vec, rpv_vec = base_series(simple_rets, covs, Y)
    kind, params = variant[0], variant[1]
    if kind == "base":
        scale = np.ones(len(base_ret) + 1)
    elif kind == "vol":
        scale = vol_scale_series(base_ret, *params)
    elif kind == "dd":
        scale = dd_brake_scale(base_ret, *params)
    elif kind == "vol+dd":
        vol_scale = vol_scale_series(base_ret, *params[0])
        dd_scale = dd_brake_scale(base_ret, *params[1])
        scale = vol_scale * dd_scale
    else:
        raise ValueError(kind)
    ret, to_eff, rpv_eff = apply_scale(simple_rets, Y, base_ret, rpv_vec, scale)
    return metrics(variant[2], ret, to_eff, rpv_eff, float(scale[: len(base_ret)].mean()))


def main():
    set_log_file(Path(__file__).with_name("drawdown_control_experiment_log.txt"))
    log("=" * 70)
    log("Drawdown-control experiment on DFL (does not touch Table3 files)")
    log("=" * 70)

    simple_all, covs_all, Y_all = load_all_days()
    si = MAIN_DFL_START
    simple_rets = simple_all[si : si + WINDOW]
    covs = covs_all[si : si + WINDOW]
    Y = Y_all[si : si + WINDOW]
    log(f"main window: {START_DATE}~{END_DATE}, T={len(Y) - 1}")

    variants = [("base", (), "DFL base")]
    for win, target in [
        (5, 0.07), (5, 0.08), (10, 0.07), (10, 0.08), (10, 0.09),
        (20, 0.07), (20, 0.08), (20, 0.09), (60, 0.08), (60, 0.09),
    ]:
        for max_scale in (1.0, 1.2):
            variants.append(
                (
                    "vol",
                    (win, target, max_scale),
                    f"vol(win={win},tgt={target:.2f},max={max_scale:.1f})",
                )
            )
    for dd_target in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        variants.append(("dd", (dd_target,), f"dd_brake(target={dd_target:.3f})"))
    for dd_target in (0.02, 0.03):
        variants.append(
            ("dd", (dd_target, 0.5), f"dd_smooth(target={dd_target:.2f},alpha=0.5)")
        )
    for win, target in ((10, 0.08), (20, 0.08)):
        for dd_target in (0.03, 0.04):
            variants.append(
                (
                    "vol+dd",
                    ((win, target, 1.0), (dd_target,)),
                    f"vol+dd(win={win},tgt={target:.2f},dd={dd_target:.2f})",
                )
            )

    rows = [evaluate(simple_rets, covs, Y, v) for v in variants]
    df = pd.DataFrame(rows).sort_values("max_dd")
    df.to_csv(OUT_CSV, index=False)

    header = (
        f"{'name':<40} {'vol':>8} {'RPV_ann':>10} {'TO_eff':>8} "
        f"{'Sharpe':>8} {'MaxDD':>8} {'CumRet':>8} {'scale':>6}"
    )
    log("\n" + header)
    log("-" * len(header))
    for _, r in df.iterrows():
        log(
            f"{r['name']:<40} {r['vol_annual']:>8.4f} {r['rpv_annual']:>10.6f} "
            f"{r['turnover_eff']:>8.4f} {r['sharpe_net']:>8.4f} "
            f"{r['max_dd']:>8.4f} {r['cum_ret']:>8.4f} {r['mean_scale']:>6.3f}"
        )
    log(f"\nSaved: {OUT_CSV}")

    robustness_starts = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    robust_variants = [
        ("base", (), "base"),
        ("dd", (0.02,), "dd02"),
        ("dd", (0.03,), "dd03"),
        ("dd", (0.02, 0.5), "dd02s"),
        ("vol", (10, 0.08, 1.0), "vol080"),
        ("vol", (20, 0.09, 1.0), "vol090"),
        ("vol+dd", ((20, 0.08, 1.0), (0.03,)), "vol+dd03"),
    ]
    rows = []
    for dstart in robustness_starts:
        s = simple_all[dstart : dstart + WINDOW]
        c = covs_all[dstart : dstart + WINDOW]
        y = Y_all[dstart : dstart + WINDOW]
        for v in robust_variants:
            r = evaluate(s, c, y, v)
            r["dstart"] = dstart
            rows.append(r)
    robust = pd.DataFrame(rows)
    robust.to_csv(ROBUST_CSV, index=False)

    log("\nRobustness across 200-day windows (dstart = index in DFL test set):")
    pivot = robust.pivot_table(
        index="dstart",
        columns="name",
        values=["max_dd", "sharpe_net", "cum_ret"],
    )
    log("\nMaxDD by window:")
    log(pivot["max_dd"].to_string())
    log("\nNet Sharpe by window:")
    log(pivot["sharpe_net"].to_string())
    log(f"\nSaved: {ROBUST_CSV}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for name, color in [
            ("DFL base", "#1f77b4"),
            ("dd_brake(target=0.020)", "#d62728"),
            ("dd_smooth(target=0.02,alpha=0.5)", "#2ca02c"),
        ]:
            v = next(x for x in variants if x[2] == name)
            r = evaluate(simple_rets, covs, Y, v)
            scale = None
            kind, params = v[0], v[1]
            if kind == "dd":
                scale = dd_brake_scale(
                    base_series(simple_rets, covs, Y)[0], *params
                )
            elif kind == "base":
                scale = np.ones(len(simple_rets))
            base_ret, _, _ = base_series(simple_rets, covs, Y)
            ret = scale[: len(base_ret)] * base_ret
            cum = np.cumprod(1 + ret)
            axes[0].plot(cum, label=name, color=color)
            peak = np.maximum.accumulate(cum)
            axes[1].plot((cum - peak) / peak, color=color)
        axes[0].set_title("Cumulative NAV")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].set_title("Drawdown")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=140)
        log(f"Saved: {PLOT_PNG}")
    except Exception as exc:
        log(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()
