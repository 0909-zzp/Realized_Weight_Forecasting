"""Walk-forward DFL parameter re-selection on the test period.

VARX predictions are fixed (trained before the test set), but DFL parameters
(rolling covariance window and eta) are re-selected every `BLOCK` days using
only the preceding 362 days. The resulting continuous out-of-sample path is
then compared with the fixed-parameter configuration.
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
from 共享模块 import log, set_log_file

import dfl_oos_tuning as tune
import dfl_oos_rolling_sigma as roll


EVAL_START, EVAL_STOP = 60, 260
RHO, RISK, BOX = 1e-3, 1.0, 0.05
BLOCK = 30
SEL_LEN = 362
OUT_CSV = Path(__file__).with_name("dfl_walkforward_results.csv")


def main():
    set_log_file(Path(__file__).with_name("dfl_walkforward_log.txt"))
    log("=" * 70)
    log("Walk-forward DFL parameter re-selection (test period)")
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

    data_dir = next(d for d in ROOT.rglob("1min_log_return_npy") if d.is_dir())
    files = sorted(
        [f for f in data_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()]
    )
    simple_rets_test_full = np.array(
        [np.expm1(np.load(str(files[i])).sum(axis=1)) for i in test_indices]
    )

    P = np.vstack([Y_pred_val, Y_pred_test])          # (725, K)
    A = np.concatenate([Y_val, Y_test])               # (725, K)
    S = np.concatenate([simple_rets_val, simple_rets_test_full])  # (725, K)
    DI = np.concatenate([val_indices, test_indices])  # (725,)
    test_start = len(Y_pred_val)                      # 362

    Y_dfl_test = np.zeros_like(Y_pred_test)
    w_prev = Y_test[0]
    chosen = []
    t0 = time.time()
    s = test_start
    k = 0
    while s < test_start + len(Y_pred_test):
        block_stop = min(s + BLOCK, test_start + len(Y_pred_test))
        sel_start = s - SEL_LEN

        # Build one sigma series covering selection window + block.
        day_window = DI[sel_start:block_stop]
        best = None
        for win in (100, 150, 200):
            sigmas = roll.build_sigmas(day_window, win, all_files)
            sig_sel = sigmas[:SEL_LEN]
            minvs_sel = tune.precompute_minvs(sig_sel, RHO, RISK)
            for eta in (1e-6, 3e-6):
                Y_sel = tune.dfl_drift_l1_rolling(
                    P[sel_start : sel_start + SEL_LEN],
                    S[sel_start : sel_start + SEL_LEN],
                    A[sel_start],
                    sig_sel,
                    minvs_sel,
                    eta,
                    RHO,
                    RISK,
                    BOX,
                )
                m = tune.portfolio_metrics(
                    Y_sel, S[sel_start : sel_start + SEL_LEN]
                )
                if 0.01 <= m["turnover"] <= 0.50:
                    score = m["calmar"]
                else:
                    score = -np.inf
                if best is None or score > best[0]:
                    best = (score, win, eta, sigmas)
            del minvs_sel
        score, win, eta, sigmas = best

        sig_block = sigmas[SEL_LEN:]
        minvs_block = tune.precompute_minvs(sig_block, RHO, RISK)
        Y_block = tune.dfl_drift_l1_rolling(
            P[s:block_stop],
            S[s:block_stop],
            w_prev,
            sig_block,
            minvs_block,
            eta,
            RHO,
            RISK,
            BOX,
        )
        Y_dfl_test[s - test_start : block_stop - test_start] = Y_block
        w_prev = Y_block[-1]
        chosen.append(
            {"block": k, "start": s - test_start, "stop": block_stop - test_start,
             "win": win, "eta": eta, "sel_calmar": score}
        )
        log(
            f"block {k:>2} test[{s-test_start:>3}:{block_stop-test_start:>3}] "
            f"win={win} eta={eta:.0e} sel_calmar={score:.2f} "
            f"({time.time()-t0:.0f}s)"
        )
        s = block_stop
        k += 1

    log(f"\nwalk-forward generation done ({time.time()-t0:.1f}s)")

    # Metrics: full 363-day test and Table3 window.
    m_full = tune.portfolio_metrics(
        Y_dfl_test, simple_rets_test_full
    )
    m_eval = tune.portfolio_metrics(
        Y_dfl_test[EVAL_START:EVAL_STOP],
        simple_rets_test_full[EVAL_START:EVAL_STOP],
    )
    log(f"\nwalk-forward full 363d: sharpe={m_full['sharpe_net']:+.3f} "
        f"maxdd={m_full['max_dd']:.3f} to={m_full['turnover']:.3f} "
        f"cum={m_full['cum_ret']:+.3f}")
    log(f"walk-forward Table3 window: sharpe={m_eval['sharpe_net']:+.3f} "
        f"maxdd={m_eval['max_dd']:.3f} to={m_eval['turnover']:.3f} "
        f"cum={m_eval['cum_ret']:+.3f}")

    # Fixed-parameter comparison (eta=1e-6, rho=1e-3, win=150).
    Y_fixed = np.load(ROOT / "VARX" / "Y_pred_model6_opt.npy")
    m_fix_full = tune.portfolio_metrics(Y_fixed, simple_rets_test_full)
    m_fix_eval = tune.portfolio_metrics(
        Y_fixed[EVAL_START:EVAL_STOP],
        simple_rets_test_full[EVAL_START:EVAL_STOP],
    )
    log(f"\nfixed full 363d: sharpe={m_fix_full['sharpe_net']:+.3f} "
        f"maxdd={m_fix_full['max_dd']:.3f} to={m_fix_full['turnover']:.3f}")
    log(f"fixed Table3 window: sharpe={m_fix_eval['sharpe_net']:+.3f} "
        f"maxdd={m_fix_eval['max_dd']:.3f} to={m_fix_eval['turnover']:.3f}")

    rows = []
    for label, Y, mf, me in [
        ("walkforward", Y_dfl_test, m_full, m_eval),
        ("fixed", Y_fixed, m_fix_full, m_fix_eval),
    ]:
        rows.append(
            {
                "config": label,
                "full_sharpe": mf["sharpe_net"],
                "full_maxdd": mf["max_dd"],
                "full_turnover": mf["turnover"],
                "full_cum": mf["cum_ret"],
                "table3_sharpe": me["sharpe_net"],
                "table3_maxdd": me["max_dd"],
                "table3_turnover": me["turnover"],
                "table3_cum": me["cum_ret"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    pd.DataFrame(chosen).to_csv(
        Path(__file__).with_name("dfl_walkforward_blocks.csv"), index=False
    )
    log(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
