"""Reduce DFL drawdown by tuning risk_mult / eta / rho on validation only.

No overlay and no cash: this is pure DFL hyperparameter tuning. The risk
multiplier scales the covariance term in the DFL objective, pushing the
solution toward the static minimum-variance portfolio.
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
from 共享模块 import K, ETA, log, set_log_file, compute_raw_cov, EPS_RIDGE

import dfl_oos_tuning as tune


EVAL_START, EVAL_STOP = 60, 260
CSV_VAL = Path(__file__).with_name("dfl_oos_drawdown_tuning.csv")
CSV_HOLDOUT = Path(__file__).with_name("dfl_oos_drawdown_holdout.csv")


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_drawdown_tuning_log.txt"))
    log("=" * 70)
    log("Pure DFL hyperparameter tuning for lower drawdown (no overlay)")
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

    # Reuse already-computed validation rows (risk 1,2,5,10) if available.
    if CSV_VAL.exists():
        old = pd.read_csv(CSV_VAL)
        done = set(zip(old["eta"], old["rho"], old["risk_mult"]))
    else:
        old = pd.DataFrame()
        done = set()

    eta_grid = [1e-7, 1e-6, 1e-5]
    rho_grid = [1e-4, 1e-3, 1e-2]
    risk_grid = [10.0, 20.0, 50.0, 100.0]
    rows = []
    t0 = time.time()
    for eta in eta_grid:
        for rho in rho_grid:
            for risk_mult in risk_grid:
                if (eta, rho, risk_mult) in done:
                    continue
                Y_dfl_val = tune.dfl_drift_l1(
                    Y_pred_val, simple_rets_val, Y_val[0], Sigma,
                    eta, rho, risk_mult,
                )
                m = tune.portfolio_metrics(Y_dfl_val, simple_rets_val)
                m.update({"eta": eta, "rho": rho, "risk_mult": risk_mult})
                rows.append(m)
                log(
                    f"eta={eta:.0e} rho={rho:.0e} risk={risk_mult:.0f}  "
                    f"sharpe={m['sharpe_net']:+.3f} to={m['turnover']:.3f} "
                    f"maxdd={m['max_dd']:.3f} calmar={m['calmar']:.2f}"
                )
    log(f"new validation runs done ({time.time()-t0:.1f}s)")

    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df = df.drop_duplicates(["eta", "rho", "risk_mult"]).sort_values(
        ["risk_mult", "eta", "rho"]
    )
    df.to_csv(CSV_VAL, index=False)
    usable = df[(df["turnover"] >= 0.01) & (df["turnover"] <= 0.50)].copy()
    log(f"validation grid total: {len(df)} rows, usable TO: {len(usable)}")

    def pick(rule, floor):
        if rule == "min_maxdd":
            sub = usable[usable["sharpe_net"] >= floor]
            if sub.empty:
                return None
            return sub.loc[sub["max_dd"].idxmax()]
        if rule == "calmar":
            if usable.empty:
                return None
            return usable.loc[usable["calmar"].idxmax()]
        raise ValueError(rule)

    candidates = []
    for rule, floor, label in [
        ("min_maxdd", 0.70, "min_maxdd_sharpe070"),
        ("min_maxdd", 1.00, "min_maxdd_sharpe100"),
        ("calmar", 0.0, "best_calmar"),
    ]:
        row = pick(rule, floor)
        if row is None:
            log(f"no candidate for {label}")
            continue
        candidates.append(
            (label, float(row["eta"]), float(row["rho"]), float(row["risk_mult"]))
        )
        log(
            f"\nValidation pick {label}: eta={row['eta']:.0e} rho={row['rho']:.0e} "
            f"risk={row['risk_mult']:.0f} sharpe={row['sharpe_net']:+.3f} "
            f"to={row['turnover']:.3f} maxdd={row['max_dd']:.3f}"
        )

    covs_test = []
    for i in test_indices[:260]:
        rett = np.load(str(all_files[i]))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs_test.append(raw)

    final_rows = []
    for label, eta, rho, risk_mult in candidates:
        Y_dfl_test = tune.dfl_drift_l1(
            Y_pred_test[:260], simple_rets_test, Y_test[0], Sigma,
            eta, rho, risk_mult,
        )
        Y_eval = Y_dfl_test[EVAL_START:EVAL_STOP]
        m = tune.portfolio_metrics(
            Y_eval,
            simple_rets_test[EVAL_START:EVAL_STOP],
            covs_test[EVAL_START:EVAL_STOP],
        )
        m.update(
            {"label": label, "eta": eta, "rho": rho, "risk_mult": risk_mult}
        )
        final_rows.append(m)
        log(
            f"\nholdout {label}: eta={eta:.0e} rho={rho:.0e} risk={risk_mult:.0f} "
            f"vol={m['vol_annual']:.4f} rpv={m.get('rpv_annual', np.nan):.6f} "
            f"to={m['turnover']:.3f} sharpe={m['sharpe_net']:+.3f} "
            f"maxdd={m['max_dd']:.3f} cum={m['cum_ret']:+.3f}"
        )

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(CSV_HOLDOUT, index=False)
    log(f"\nSaved: {CSV_VAL}")
    log(f"Saved: {CSV_HOLDOUT}")


if __name__ == "__main__":
    main()
