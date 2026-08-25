"""Extend the DFL rolling-covariance window grid on validation only.

Windows are extended to both shorter (10) and longer (80, 100, 120, 150) than
the current 60. eta=1e-6 and rho=1e-3 are fixed from the earlier validation
selection; window length is selected on validation, then evaluated once on the
Table3 holdout.
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
import dfl_oos_rolling_sigma as roll


EVAL_START, EVAL_STOP = 60, 260
ETA, RHO, RISK, BOX = 1e-6, 1e-3, 1.0, 0.05
CSV_VAL = Path(__file__).with_name("dfl_oos_rolling_window_ext_tuning.csv")
CSV_HOLDOUT = Path(__file__).with_name("dfl_oos_rolling_window_ext_holdout.csv")


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_rolling_window_ext_log.txt"))
    log("=" * 70)
    log("DFL rolling-covariance window extension (validation selection)")
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

    old_csv = Path(__file__).with_name("dfl_oos_rolling_sigma_tuning.csv")
    if old_csv.exists():
        old = pd.read_csv(old_csv)
        old = old[(old["eta"] == ETA) & (old["rho"] == RHO)]
        done = set(old["win"].astype(int))
    else:
        old = pd.DataFrame()
        done = set()

    win_grid = [10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400]
    rows = []
    t0 = time.time()
    for win in win_grid:
        if win in done:
            continue
        sigmas_val = roll.build_sigmas(val_indices, win, all_files)
        minvs_val = tune.precompute_minvs(sigmas_val, RHO, RISK)
        Y_dfl_val = tune.dfl_drift_l1_rolling(
            Y_pred_val, simple_rets_val, Y_val[0],
            sigmas_val, minvs_val, ETA, RHO, RISK, BOX,
        )
        m = tune.portfolio_metrics(Y_dfl_val, simple_rets_val)
        m.update({"win": win, "eta": ETA, "rho": RHO})
        rows.append(m)
        log(
            f"win={win}  sharpe={m['sharpe_net']:+.3f} "
            f"to={m['turnover']:.3f} maxdd={m['max_dd']:.3f} "
            f"calmar={m['calmar']:.2f}"
        )
        del minvs_val, sigmas_val
    log(f"new validation runs done ({time.time()-t0:.1f}s)")

    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df = df[df["eta"] == ETA].drop_duplicates("win").sort_values("win")
    df.to_csv(CSV_VAL, index=False)
    usable = df[(df["turnover"] >= 0.01) & (df["turnover"] <= 0.50)].copy()
    log("\nValidation by window (eta=1e-6, rho=1e-3):")
    for _, r in df.iterrows():
        log(
            f"  win={r['win']:>3}  sharpe={r['sharpe_net']:+.3f} "
            f"to={r['turnover']:.3f} maxdd={r['max_dd']:.3f} "
            f"calmar={r['calmar']:.2f}"
        )

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
        candidates.append((label, int(row["win"])))
        log(
            f"\nValidation pick {label}: win={row['win']} "
            f"sharpe={row['sharpe_net']:+.3f} to={row['turnover']:.3f} "
            f"maxdd={row['max_dd']:.3f}"
        )

    covs_test = []
    for i in test_indices[:260]:
        rett = np.load(str(all_files[i]))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs_test.append(raw)

    final_rows = []
    for label, win in candidates:
        sigmas_test = roll.build_sigmas(test_indices[:260], win, all_files)
        minvs_test = tune.precompute_minvs(sigmas_test, RHO, RISK)
        Y_dfl_test = tune.dfl_drift_l1_rolling(
            Y_pred_test[:260], simple_rets_test, Y_test[0],
            sigmas_test, minvs_test, ETA, RHO, RISK, BOX,
        )
        Y_eval = Y_dfl_test[EVAL_START:EVAL_STOP]
        m = tune.portfolio_metrics(
            Y_eval,
            simple_rets_test[EVAL_START:EVAL_STOP],
            covs_test[EVAL_START:EVAL_STOP],
        )
        m.update({"label": label, "win": win, "eta": ETA, "rho": RHO})
        final_rows.append(m)
        log(
            f"\nholdout {label}: win={win} vol={m['vol_annual']:.4f} "
            f"rpv={m.get('rpv_annual', np.nan):.6f} to={m['turnover']:.3f} "
            f"sharpe={m['sharpe_net']:+.3f} maxdd={m['max_dd']:.3f} "
            f"cum={m['cum_ret']:+.3f}"
        )
        del minvs_test, sigmas_test

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(CSV_HOLDOUT, index=False)
    log(f"\nSaved: {CSV_VAL}")
    log(f"Saved: {CSV_HOLDOUT}")


if __name__ == "__main__":
    main()
