"""Daily rolling covariance for DFL, tuned on validation only.

Each DFL day uses a covariance estimated from the previous `win` trading days
(rather than a fixed covariance from the training end). Window length, eta and
rho are selected on the 362-day validation window, then evaluated once on the
Table3 holdout. No overlay and no cash.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "图形Lasso" / "code"))
from 共享模块 import K, log, set_log_file, compute_raw_cov, EPS_RIDGE

import dfl_oos_tuning as tune


EVAL_START, EVAL_STOP = 60, 260
CSV_VAL = Path(__file__).with_name("dfl_oos_rolling_sigma_tuning.csv")
CSV_HOLDOUT = Path(__file__).with_name("dfl_oos_rolling_sigma_holdout.csv")


def build_sigmas(day_indices, win, all_files):
    """Rolling covariance for each day, using the previous `win` days."""
    start = day_indices[0] - win
    end = day_indices[-1]
    covs = []
    for i in range(start, end + 1):
        rett = np.load(str(all_files[i]))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs.append(raw)

    T = len(day_indices)
    sigmas = np.empty((T, K, K), dtype=np.float64)
    s = np.sum(covs[0:win], axis=0) / win
    sigmas[0] = s
    for t in range(1, T):
        s = s + (covs[win + t - 1] - covs[t - 1]) / win
        sigmas[t] = s
    return sigmas


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_rolling_sigma_log.txt"))
    log("=" * 70)
    log("DFL daily rolling covariance tuning (no overlay)")
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

    feat_dir = ROOT / "特征工程"
    valid_indices = np.load(feat_dir / "valid_indices.npy")
    n = len(valid_indices)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    val_indices = valid_indices[n_train : n_train + n_val]

    win_grid = [20, 40, 60]
    eta_grid = [1e-7, 1e-6, 1e-5]
    rho_grid = [1e-4, 1e-3, 1e-2]
    risk_mult, box = 1.0, 0.05

    rows = []
    t0 = time.time()
    for win in win_grid:
        sigmas_val = build_sigmas(val_indices, win, all_files)
        for rho in rho_grid:
            minvs_val = tune.precompute_minvs(sigmas_val, rho, risk_mult)
            for eta in eta_grid:
                Y_dfl_val = tune.dfl_drift_l1_rolling(
                    Y_pred_val, simple_rets_val, Y_val[0],
                    sigmas_val, minvs_val, eta, rho, risk_mult, box,
                )
                m = tune.portfolio_metrics(Y_dfl_val, simple_rets_val)
                m.update({"win": win, "eta": eta, "rho": rho})
                rows.append(m)
                log(
                    f"win={win} eta={eta:.0e} rho={rho:.0e}  "
                    f"sharpe={m['sharpe_net']:+.3f} to={m['turnover']:.3f} "
                    f"maxdd={m['max_dd']:.3f} calmar={m['calmar']:.2f}"
                )
            del minvs_val
        del sigmas_val
    log(f"validation grid done ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_VAL, index=False)
    usable = df[(df["turnover"] >= 0.01) & (df["turnover"] <= 0.50)].copy()
    log(f"total {len(df)} rows, usable TO: {len(usable)}")

    def pick(rule, floor=0.0):
        if rule == "min_maxdd":
            sub = usable[usable["sharpe_net"] >= floor]
            if sub.empty:
                return None
            return sub.loc[sub["max_dd"].idxmax()]
        if rule == "calmar":
            if usable.empty:
                return None
            return usable.loc[usable["calmar"].idxmax()]
        if rule == "sharpe":
            if usable.empty:
                return None
            return usable.loc[usable["sharpe_net"].idxmax()]
        raise ValueError(rule)

    candidates = []
    for rule, floor, label in [
        ("min_maxdd", 0.70, "min_maxdd_sharpe070"),
        ("calmar", 0.0, "best_calmar"),
        ("sharpe", 0.0, "best_sharpe"),
    ]:
        row = pick(rule, floor)
        if row is None:
            continue
        candidates.append(
            (label, int(row["win"]), float(row["eta"]), float(row["rho"]))
        )
        log(
            f"\nValidation pick {label}: win={row['win']} eta={row['eta']:.0e} "
            f"rho={row['rho']:.0e} sharpe={row['sharpe_net']:+.3f} "
            f"to={row['turnover']:.3f} maxdd={row['max_dd']:.3f}"
        )

    covs_test = []
    for i in test_indices[:260]:
        rett = np.load(str(all_files[i]))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs_test.append(raw)

    final_rows = []
    for label, win, eta, rho in candidates:
        sigmas_test = build_sigmas(test_indices[:260], win, all_files)
        minvs_test = tune.precompute_minvs(sigmas_test, rho, risk_mult)
        Y_dfl_test = tune.dfl_drift_l1_rolling(
            Y_pred_test[:260], simple_rets_test, Y_test[0],
            sigmas_test, minvs_test, eta, rho, risk_mult, box,
        )
        Y_eval = Y_dfl_test[EVAL_START:EVAL_STOP]
        m = tune.portfolio_metrics(
            Y_eval,
            simple_rets_test[EVAL_START:EVAL_STOP],
            covs_test[EVAL_START:EVAL_STOP],
        )
        m.update({"label": label, "win": win, "eta": eta, "rho": rho})
        final_rows.append(m)
        log(
            f"\nholdout {label}: win={win} eta={eta:.0e} rho={rho:.0e} "
            f"vol={m['vol_annual']:.4f} rpv={m.get('rpv_annual', np.nan):.6f} "
            f"to={m['turnover']:.3f} sharpe={m['sharpe_net']:+.3f} "
            f"maxdd={m['max_dd']:.3f} cum={m['cum_ret']:+.3f}"
        )
        del minvs_test, sigmas_test

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(CSV_HOLDOUT, index=False)
    log(f"\nSaved: {CSV_VAL}")
    log(f"Saved: {CSV_HOLDOUT}")


if __name__ == "__main__":
    main()
