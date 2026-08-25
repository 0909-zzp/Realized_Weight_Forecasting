"""Validation-tuned overlays applied fresh at the Table3 window start.

Unlike a continuous live simulation, Table3 evaluates every strategy as if it
starts fully invested at the window start. This script follows that convention:
the overlay threshold is tuned on the 362-day validation window, then the
overlay is applied from a fresh start on the holdout window.
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
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log, set_log_file, compute_raw_cov, EPS_RIDGE

import dfl_oos_tuning as tune


ETA_CHOSEN = 1e-7
RHO_CHOSEN = 1e-4
RISK_CHOSEN = 1.0
EVAL_START, EVAL_STOP = 60, 260
OUT_CSV = Path(__file__).with_name("dfl_oos_overlay.csv")


def brake_scale_fresh(base_ret, dd_target, alpha=1.0):
    T = len(base_ret)
    scale = np.ones(T + 1)
    nav, peak, dd = 1.0, 1.0, 0.0
    prev_scale = 1.0
    for t in range(T):
        target = 1.0
        if dd < -dd_target:
            target = float(np.clip(1.0 - (-dd - dd_target) / dd_target, 0.0, 1.0))
        scale[t] = alpha * target + (1.0 - alpha) * prev_scale
        r = scale[t] * base_ret[t]
        nav *= 1.0 + r
        peak = max(peak, nav)
        dd = nav / peak - 1.0
        prev_scale = scale[t]
    return scale


def vol_scale_fresh(base_ret, win, target_vol, max_scale=1.0):
    T = len(base_ret)
    scale = np.ones(T + 1)
    for t in range(1, T + 1):
        lookback = min(win, t)
        if lookback >= 2:
            rv = np.std(base_ret[t - lookback : t], ddof=1) * np.sqrt(252)
            scale[t] = float(np.clip(target_vol / rv, 0.0, max_scale))
    return scale


def apply_overlay(simple_rets, Y, base_ret, scale):
    T = len(base_ret)
    ret = scale[:T] * base_ret
    to_eff = np.zeros(T)
    for t in range(T):
        w_prev = Y[t]
        r_next = simple_rets[t + 1]
        w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
        w_ret = w_prev @ r_next
        sleeve = scale[t] * (1.0 + w_ret) / (1.0 + scale[t] * w_ret)
        to_eff[t] = np.sum(np.abs(scale[t + 1] * Y[t + 1] - sleeve * w_drifted))
    return ret, to_eff


def metrics(ret, to_eff):
    net = ret - ETA * to_eff
    sr = (
        float(np.mean(net) / np.std(net, ddof=1)) * np.sqrt(252)
        if np.std(net) > 1e-15
        else 0.0
    )
    cum = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak))
    return {
        "sharpe_net": sr,
        "turnover": float(np.mean(to_eff)),
        "max_dd": dd,
        "cum_ret": float(np.prod(1 + ret) - 1),
    }


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_overlay_log.txt"))
    log("=" * 70)
    log("Validation-tuned overlays, fresh start on Table3 window")
    log("=" * 70)

    (
        Y_pred_val,
        Y_val,
        Y_pred_test,
        Y_test,
        train_day_indices,
        simple_rets_val,
        simple_rets_test,
        all_files,
        test_indices,
    ) = tune.load_parts()
    Sigma = tune.rolling_sigma(train_day_indices)

    Y_dfl_val = tune.dfl_drift_l1(
        Y_pred_val, simple_rets_val, Y_val[0], Sigma,
        ETA_CHOSEN, RHO_CHOSEN, RISK_CHOSEN,
    )
    Y_dfl_test = tune.dfl_drift_l1(
        Y_pred_test[:260], simple_rets_test, Y_test[0], Sigma,
        ETA_CHOSEN, RHO_CHOSEN, RISK_CHOSEN,
    )

    base_val_ret = np.array(
        [Y_dfl_val[t] @ simple_rets_val[t + 1] for t in range(len(Y_dfl_val) - 1)]
    )
    Y_eval = Y_dfl_test[EVAL_START:EVAL_STOP]
    simple_eval = simple_rets_test[EVAL_START:EVAL_STOP]
    base_eval_ret = np.array(
        [Y_eval[t] @ simple_eval[t + 1] for t in range(len(Y_eval) - 1)]
    )

    base_m = metrics(base_eval_ret, np.zeros(len(base_eval_ret)))
    log(f"\nDFL base holdout: sharpe={base_m['sharpe_net']:+.3f} "
        f"maxdd={base_m['max_dd']:.3f} cum={base_m['cum_ret']:+.3f}")

    rows = []
    for dd_target in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        for alpha in (1.0, 0.5):
            scale_val = brake_scale_fresh(base_val_ret, dd_target, alpha)
            ret_val, to_val = apply_overlay(
                simple_rets_val, Y_dfl_val, base_val_ret, scale_val
            )
            m_val = metrics(ret_val, to_val)
            scale_eval = brake_scale_fresh(base_eval_ret, dd_target, alpha)
            ret_eval, to_eval = apply_overlay(
                simple_eval, Y_eval, base_eval_ret, scale_eval
            )
            m_eval = metrics(ret_eval, to_eval)
            rows.append(
                {
                    "kind": "brake",
                    "dd_target": dd_target,
                    "alpha": alpha,
                    "val_sharpe": m_val["sharpe_net"],
                    "val_maxdd": m_val["max_dd"],
                    "val_turnover": m_val["turnover"],
                    "holdout_sharpe": m_eval["sharpe_net"],
                    "holdout_maxdd": m_eval["max_dd"],
                    "holdout_turnover": m_eval["turnover"],
                    "holdout_cum_ret": m_eval["cum_ret"],
                }
            )

    for win in (10, 20, 60):
        for target_vol in (0.05, 0.06, 0.07, 0.08, 0.09):
            scale_val = vol_scale_fresh(base_val_ret, win, target_vol)
            ret_val, to_val = apply_overlay(
                simple_rets_val, Y_dfl_val, base_val_ret, scale_val
            )
            m_val = metrics(ret_val, to_val)
            scale_eval = vol_scale_fresh(base_eval_ret, win, target_vol)
            ret_eval, to_eval = apply_overlay(
                simple_eval, Y_eval, base_eval_ret, scale_eval
            )
            m_eval = metrics(ret_eval, to_eval)
            rows.append(
                {
                    "kind": "vol",
                    "dd_target": target_vol,
                    "alpha": float(win),
                    "val_sharpe": m_val["sharpe_net"],
                    "val_maxdd": m_val["max_dd"],
                    "val_turnover": m_val["turnover"],
                    "holdout_sharpe": m_eval["sharpe_net"],
                    "holdout_maxdd": m_eval["max_dd"],
                    "holdout_turnover": m_eval["turnover"],
                    "holdout_cum_ret": m_eval["cum_ret"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    log("\nBrake tuned by validation Sharpe, then holdout:")
    for kind in ("brake", "vol"):
        sub = df[df["kind"] == kind].copy()
        if sub.empty:
            continue
        best = sub.loc[sub["val_sharpe"].idxmax()]
        log(
            f"  {kind}: param={best['dd_target']:.3f}/{best['alpha']:.1f} "
            f"-> holdout sharpe={best['holdout_sharpe']:+.3f} "
            f"maxdd={best['holdout_maxdd']:.3f} to={best['holdout_turnover']:.3f}"
        )

    log("\nBrake tuned by validation maxdd, then holdout:")
    for kind in ("brake", "vol"):
        sub = df[df["kind"] == kind].copy()
        if sub.empty:
            continue
        best = sub.loc[sub["val_maxdd"].idxmax()]
        log(
            f"  {kind}: param={best['dd_target']:.3f}/{best['alpha']:.1f} "
            f"-> holdout sharpe={best['holdout_sharpe']:+.3f} "
            f"maxdd={best['holdout_maxdd']:.3f} to={best['holdout_turnover']:.3f}"
        )

    log(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
