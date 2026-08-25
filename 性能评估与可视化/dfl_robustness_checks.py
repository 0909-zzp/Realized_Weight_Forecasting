"""Robustness checks for the DFL configuration.

1. Parameter-neighborhood sensitivity: win x eta x rho grid around the chosen
   config, all evaluated on the validation set and then on the Table3 holdout.
2. Cross-window robustness: the chosen DFL config (eta=1e-6, rho=1e-3,
   rolling Sigma window=150) evaluated on all 200-day windows in the test set,
   compared against Network VARX (M4).
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
RISK, BOX = 1.0, 0.05
CSV_SENS = Path(__file__).with_name("dfl_param_sensitivity.csv")
CSV_WINDOW = Path(__file__).with_name("dfl_cross_window.csv")


def main():
    set_log_file(Path(__file__).with_name("dfl_robustness_checks_log.txt"))
    log("=" * 70)
    log("DFL robustness: parameter neighborhood + cross-window stability")
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

    covs_test = []
    for i in test_indices[:260]:
        rett = np.load(str(all_files[i]))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs_test.append(raw)

    # ---- 1. Parameter-neighborhood sensitivity ----
    win_grid = [100, 150, 200]
    eta_grid = [1e-6, 3e-6]
    rho_grid = [1e-3, 3e-3]
    rows = []
    t0 = time.time()
    for win in win_grid:
        sigmas_val = roll.build_sigmas(val_indices, win, all_files)
        sigmas_test = roll.build_sigmas(test_indices[:260], win, all_files)
        for rho in rho_grid:
            minvs_val = tune.precompute_minvs(sigmas_val, rho, RISK)
            minvs_test = tune.precompute_minvs(sigmas_test, rho, RISK)
            for eta in eta_grid:
                Y_val = tune.dfl_drift_l1_rolling(
                    Y_pred_val, simple_rets_val, Y_val[0],
                    sigmas_val, minvs_val, eta, rho, RISK, BOX,
                )
                m_val = tune.portfolio_metrics(Y_val, simple_rets_val)

                Y_te = tune.dfl_drift_l1_rolling(
                    Y_pred_test[:260], simple_rets_test, Y_test[0],
                    sigmas_test, minvs_test, eta, rho, RISK, BOX,
                )
                Y_eval = Y_te[EVAL_START:EVAL_STOP]
                m_te = tune.portfolio_metrics(
                    Y_eval,
                    simple_rets_test[EVAL_START:EVAL_STOP],
                    covs_test[EVAL_START:EVAL_STOP],
                )
                row = {
                    "win": win, "eta": eta, "rho": rho,
                    "val_sharpe": m_val["sharpe_net"],
                    "val_maxdd": m_val["max_dd"],
                    "val_calmar": m_val["calmar"],
                    "holdout_sharpe": m_te["sharpe_net"],
                    "holdout_maxdd": m_te["max_dd"],
                    "holdout_turnover": m_te["turnover"],
                    "holdout_cum_ret": m_te["cum_ret"],
                }
                rows.append(row)
                log(
                    f"win={win} eta={eta:.0e} rho={rho:.0e}  "
                    f"val sharpe={m_val['sharpe_net']:+.3f} "
                    f"maxdd={m_val['max_dd']:.3f} | "
                    f"holdout sharpe={m_te['sharpe_net']:+.3f} "
                    f"maxdd={m_te['max_dd']:.3f} to={m_te['turnover']:.3f}"
                )
            del minvs_test, minvs_val
        del sigmas_test, sigmas_val
    log(f"parameter sensitivity done ({time.time()-t0:.1f}s)")
    sens = pd.DataFrame(rows).sort_values(["win", "rho", "eta"])
    sens.to_csv(CSV_SENS, index=False)

    # ---- 2. Cross-window robustness for the chosen config ----
    Y_dfl = np.load(ROOT / "VARX" / "Y_pred_model6_opt.npy")
    Y_m4 = np.load(ROOT / "VARX" / "Y_pred_model4.npy")
    simple_rets_full = np.array(
        [np.expm1(np.load(str(all_files[i])).sum(axis=1))
         for i in test_indices]
    )

    wrows = []
    for dstart in range(0, 161, 20):
        for label, Y in [("DFL(Σ150)", Y_dfl), ("Network VARX", Y_m4)]:
            Yw = Y[dstart : dstart + 200]
            sw = simple_rets_full[dstart : dstart + 200]
            covw = []
            for i in test_indices[dstart : dstart + 200]:
                rett = np.load(str(all_files[i]))
                raw = compute_raw_cov(rett)
                raw.flat[:: K + 1] += EPS_RIDGE
                covw.append(raw)
            m = tune.portfolio_metrics(Yw, sw, covw)
            m.update({"dstart": dstart, "model": label})
            wrows.append(m)
            log(
                f"window {dstart:>3} {label:<14} sharpe={m['sharpe_net']:+.3f} "
                f"maxdd={m['max_dd']:.3f} to={m['turnover']:.3f} "
                f"cum={m['cum_ret']:+.3f}"
            )
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(CSV_WINDOW, index=False)

    log(f"\nSaved: {CSV_SENS}")
    log(f"Saved: {CSV_WINDOW}")


if __name__ == "__main__":
    main()
