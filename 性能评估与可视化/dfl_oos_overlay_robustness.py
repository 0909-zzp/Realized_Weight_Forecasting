"""Robustness of validation-tuned fresh-start overlays across test windows."""
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
from 共享模块 import K, ETA, log, set_log_file

import dfl_oos_tuning as tune
import dfl_oos_overlay as overlay


ETA_CHOSEN = 1e-7
RHO_CHOSEN = 1e-4
RISK_CHOSEN = 1.0
OUT_CSV = Path(__file__).with_name("dfl_oos_overlay_robustness.csv")


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_overlay_robustness_log.txt"))
    log("=" * 70)
    log("Fresh-start overlay robustness across 200-day test windows")
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

    # Full 363-day test returns for all sliding windows.
    data_dir = next(d for d in ROOT.rglob("1min_log_return_npy") if d.is_dir())
    all_files = sorted(
        [f for f in data_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()]
    )
    simple_rets_full = np.array(
        [np.expm1(np.load(all_files[i]).sum(axis=1)) for i in test_indices]
    )
    Y_dfl_full = tune.dfl_drift_l1(
        Y_pred_test, simple_rets_full, Y_test[0], Sigma,
        ETA_CHOSEN, RHO_CHOSEN, RISK_CHOSEN,
    )

    rows = []
    for dstart in range(0, 161, 20):
        Y = Y_dfl_full[dstart : dstart + 200]
        simple = simple_rets_full[dstart : dstart + 200]
        base_ret = np.array(
            [Y[t] @ simple[t + 1] for t in range(len(Y) - 1)]
        )

        to_vec = np.zeros(len(base_ret))
        for t in range(len(base_ret)):
            w_prev = Y[t]
            r_next = simple[t + 1]
            w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
            to_vec[t] = np.sum(np.abs(Y[t + 1] - w_drifted))
        m = overlay.metrics(base_ret - ETA * to_vec, np.zeros(len(base_ret)))
        base = {"dstart": dstart, "kind": "base", "param": 0.0,
                "sharpe": m["sharpe_net"], "maxdd": m["max_dd"],
                "turnover": float(np.mean(to_vec)), "cum": m["cum_ret"]}
        rows.append(base)

        for dd_target, alpha, label in [
            (0.015, 1.0, "brake015"),
            (0.015, 0.5, "brake015s"),
            (0.02, 1.0, "brake020"),
            (0.02, 0.5, "brake020s"),
        ]:
            scale = overlay.brake_scale_fresh(base_ret, dd_target, alpha)
            ret, to_eff = overlay.apply_overlay(simple, Y, base_ret, scale)
            m = overlay.metrics(ret, to_eff)
            rows.append(
                {"dstart": dstart, "kind": label, "param": dd_target,
                 "sharpe": m["sharpe_net"], "maxdd": m["max_dd"],
                 "turnover": m["turnover"], "cum": m["cum_ret"]}
            )

        scale = overlay.vol_scale_fresh(base_ret, 60, 0.05)
        ret, to_eff = overlay.apply_overlay(simple, Y, base_ret, scale)
        m = overlay.metrics(ret, to_eff)
        rows.append(
            {"dstart": dstart, "kind": "vol005_60", "param": 0.05,
             "sharpe": m["sharpe_net"], "maxdd": m["max_dd"],
             "turnover": m["turnover"], "cum": m["cum_ret"]}
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    pivot = df.pivot_table(
        index="dstart", columns="kind", values=["maxdd", "sharpe", "cum"]
    )
    log("\nMaxDD by window:")
    log(pivot["maxdd"].to_string())
    log("\nNet Sharpe by window:")
    log(pivot["sharpe"].to_string())
    log(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
