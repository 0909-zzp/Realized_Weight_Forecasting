"""Tune DFL covariance window and box constraint on validation only.

No overlay and no cash. The covariance window controls how many training days
are used for the fixed DFL covariance; the box constrains single-asset weights.
Selection happens on the 362-day validation window, then the chosen configs are
evaluated once on the Table3 holdout.
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
CSV_VAL = Path(__file__).with_name("dfl_oos_window_box_tuning.csv")
CSV_HOLDOUT = Path(__file__).with_name("dfl_oos_window_box_holdout.csv")


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_window_box_tuning_log.txt"))
    log("=" * 70)
    log("DFL covariance-window and box tuning (no overlay)")
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

    win_grid = [20, 40, 60]
    box_grid = [0.02, 0.03, 0.05, 0.08]
    eta_grid = [1e-7]
    rho_grid = [1e-4, 1e-3, 1e-2]
    risk_mult = 1.0

    rows = []
    t0 = time.time()
    for win in win_grid:
        Sigma = tune.rolling_sigma(train_day_indices, win)
        for box in box_grid:
            for eta in eta_grid:
                for rho in rho_grid:
                    Y_dfl_val = tune.dfl_drift_l1(
                        Y_pred_val, simple_rets_val, Y_val[0], Sigma,
                        eta, rho, risk_mult, box,
                    )
                    m = tune.portfolio_metrics(Y_dfl_val, simple_rets_val)
                    m.update(
                        {"win": win, "box": box, "eta": eta, "rho": rho,
                         "risk_mult": risk_mult}
                    )
                    rows.append(m)
                    log(
                        f"win={win} box={box:.2f} eta={eta:.0e} rho={rho:.0e}  "
                        f"sharpe={m['sharpe_net']:+.3f} to={m['turnover']:.3f} "
                        f"maxdd={m['max_dd']:.3f} calmar={m['calmar']:.2f}"
                    )
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
            (label, int(row["win"]), float(row["box"]),
             float(row["eta"]), float(row["rho"]))
        )
        log(
            f"\nValidation pick {label}: win={row['win']} box={row['box']:.2f} "
            f"eta={row['eta']:.0e} rho={row['rho']:.0e} "
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
    for label, win, box, eta, rho in candidates:
        Sigma = tune.rolling_sigma(train_day_indices, win)
        Y_dfl_test = tune.dfl_drift_l1(
            Y_pred_test[:260], simple_rets_test, Y_test[0], Sigma,
            eta, rho, risk_mult, box,
        )
        Y_eval = Y_dfl_test[EVAL_START:EVAL_STOP]
        m = tune.portfolio_metrics(
            Y_eval,
            simple_rets_test[EVAL_START:EVAL_STOP],
            covs_test[EVAL_START:EVAL_STOP],
        )
        m.update(
            {"label": label, "win": win, "box": box, "eta": eta, "rho": rho,
             "risk_mult": risk_mult}
        )
        final_rows.append(m)
        log(
            f"\nholdout {label}: win={win} box={box:.2f} eta={eta:.0e} "
            f"rho={rho:.0e} vol={m['vol_annual']:.4f} "
            f"rpv={m.get('rpv_annual', np.nan):.6f} to={m['turnover']:.3f} "
            f"sharpe={m['sharpe_net']:+.3f} maxdd={m['max_dd']:.3f} "
            f"cum={m['cum_ret']:+.3f}"
        )

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(CSV_HOLDOUT, index=False)
    log(f"\nSaved: {CSV_VAL}")
    log(f"Saved: {CSV_HOLDOUT}")


if __name__ == "__main__":
    main()
