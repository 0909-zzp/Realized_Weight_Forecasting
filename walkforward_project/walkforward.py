"""Walk-forward project (read-only on original code/data).

Rolling re-estimation over the original 363-day test origins:
  - refit Network VARX + Smooth (M5) every REFIT_EVERY origins using the
    preceding TRAIN_L origins, with fixed hyperparameters from the original
    configuration (hyperparameters are NOT re-selected on each window);
  - apply the same daily drift-aware L1 DFL post-processing;
  - evaluate M5 and M5+DFL over the full test files with Table-3-style net
    returns, plus equal weight.

Nothing in the original project is modified; all outputs are written here.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

warnings = None
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = Path(__file__).resolve().parent

# ---- import original helpers (read-only) ----
varx_path = PROJ / "VARX" / "VAR及拓展（table2）.py"
spec = importlib.util.spec_from_file_location("vp", str(varx_path))
vp = importlib.util.module_from_spec(spec)
sys.modules["vp"] = vp
spec.loader.exec_module(vp)

t3_path = PROJ / "性能评估与可视化" / "Table3_投资组合表现.py"
spec3 = importlib.util.spec_from_file_location("t3", str(t3_path))
t3 = importlib.util.module_from_spec(spec3)
sys.modules["t3"] = t3
spec3.loader.exec_module(t3)

sys.path.insert(0, str(PROJ / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log as shared_log

# ---- config (fixed, no per-window tuning) ----
TRAIN_L = 500          # origins used for each refit (original L_TRAIN_VARX target)
REFIT_EVERY = 20       # refit every ~20 origins (about monthly)
DFL_ETA = 1e-6
DFL_RHO = 1e-3
COV_WINDOW = 150
MODEL_ID = 5           # Network VARX + Smooth


def load_data():
    feat = PROJ / "特征工程"
    X = np.load(feat / "X_features.npy")
    Y = np.load(feat / "Y_targets.npy")
    A_bar = np.load(feat / "A_bar.npy")
    valid = np.load(feat / "valid_indices.npy")
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    test_start = n_train + n_val
    return X, Y, A_bar, valid, n_train, test_start


def test_file_list(valid, test_start):
    npy_dir = PROJ / "数据" / "1min_log_return_npy"
    all_files = sorted([f for f in npy_dir.iterdir()
                        if f.suffix == ".npy" and f.name[0].isdigit()])
    idx = valid[test_start:]
    return [all_files[i] for i in idx]


def walkforward_base(X, Y, A_bar, test_start):
    """Refit every REFIT_EVERY origins; return (363,K) raw forecasts."""
    n_test = len(Y) - test_start
    W = np.zeros((n_test, K), dtype=np.float64)
    n_refit = 0
    t0 = time.time()
    for start in range(0, n_test, REFIT_EVERY):
        end = min(start + REFIT_EVERY, n_test)
        lo = max(0, test_start + start - TRAIN_L)
        hi = test_start + start
        A_win = A_bar[lo:hi]
        net_mask, _ = vp.build_network_mask(A_win)
        fitted = vp.fit_model(MODEL_ID, X[lo:hi], Y[lo:hi], net_mask, n_jobs=4)
        rows = X[test_start + start: test_start + end]
        W[start:end] = vp.predict_model(rows, fitted)
        n_refit += 1
        print(f"refit block {start:4d}-{end-1:4d}: fit={time.time()-t0:.1f}s", flush=True)
    print(f"total refits: {n_refit} in {time.time()-t0:.1f}s")
    return W


def evaluate(weights_dict, test_files, out_name):
    models = {k: {"name": v[0], "Y_pred": v[1]} for k, v in weights_dict.items()}
    res = t3.compute_all(models, test_files)
    rows = []
    for k in list(weights_dict.keys()):
        r = res[k]
        rows.append({
            "model": weights_dict[k][0],
            "avg_ret": r["avg_ret"], "vol_annual": r["vol_annual"],
            "rpv_annual": r["rpv_annual"], "avg_turnover": r["avg_turnover"],
            "sharpe_net": r["sharpe_net"], "max_dd": r["max_dd"],
            "cum_ret": r["cum_ret"],
        })
    rows.append({
        "model": "Equal Weight",
        "avg_ret": res["eq"]["avg_ret"], "vol_annual": res["eq"]["vol_annual"],
        "rpv_annual": res["eq"]["rpv_annual"], "avg_turnover": res["eq"]["avg_turnover"],
        "sharpe_net": res["eq"]["sharpe_net"], "max_dd": res["eq"]["max_dd"],
        "cum_ret": res["eq"]["cum_ret"],
    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / out_name, index=False)
    print(df.to_string(index=False))
    return df


def main():
    X, Y, A_bar, valid, n_train, test_start = load_data()
    test_files = test_file_list(valid, test_start)
    print(f"X={X.shape} test_start={test_start} test_origins={len(Y)-test_start} files={len(test_files)}")

    # calibration: original one-fit M5 forecasts evaluated the same way
    Y_m5_orig = np.load(PROJ / "VARX" / "Y_pred_model5.npy")[-363:]
    Y_m6_orig = np.load(PROJ / "VARX" / "Y_pred_model6_opt.npy")[-363:]
    print("\n--- calibration (original one-fit forecasts, full test) ---")
    evaluate({"m5": ("Network VARX + Smooth (one-fit, original)", Y_m5_orig),
              "m5dfl": ("Network VARX + Smooth + DFL (one-fit, original)", Y_m6_orig)},
             test_files, "calibration_full_test.csv")

    # walk-forward base forecasts
    print("\n--- walk-forward refitting M5 ---")
    W_base = walkforward_base(X, Y, A_bar, test_start)
    np.save(OUT / "W_base_walkforward.npy", W_base)

    # DFL post-processing (same daily drift-aware L1 solve as original)
    print("\n--- DFL post-processing ---")
    Y_te = Y[test_start:]
    W_dfl = vp.compute_model6_drift_l1(
        W_base, Y_te, valid[:n_train],
        eta=DFL_ETA, rho=DFL_RHO, risk_mult=1.0,
        rolling_cov=True, cov_window=COV_WINDOW,
    )
    np.save(OUT / "W_dfl_walkforward.npy", W_dfl)

    print("\n--- walk-forward results, full test ---")
    evaluate({"m5": ("Network VARX + Smooth (walk-forward)", W_base),
              "m5dfl": ("Network VARX + Smooth + DFL (walk-forward)", W_dfl)},
             test_files, "walkforward_full_test.csv")

    print("\nfiles written to:", OUT)


if __name__ == "__main__":
    main()
