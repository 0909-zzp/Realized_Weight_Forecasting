"""Evaluate the end-to-end trained weights against original benchmarks."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("t3", str(PROJ / "性能评估与可视化" / "Table3_投资组合表现.py"))
t3 = importlib.util.module_from_spec(spec); sys.modules["t3"] = t3; spec.loader.exec_module(t3)

# test files (full 363 origins)
valid = np.load(PROJ / "特征工程" / "valid_indices.npy")
n = len(np.load(PROJ / "特征工程" / "X_features.npy"))
n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train+n_val
npy_dir = PROJ / "数据" / "1min_log_return_npy"
all_files = sorted([f for f in npy_dir.iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
test_files_full = [all_files[i] for i in valid[test_start:]]
test_files_200 = test_files_full[65:265]

W_e2e = np.load(HERE/"e2e_test_weights.npy")
varx = PROJ/"VARX"
W_m5 = np.load(varx/"Y_pred_model5.npy")[-363:]
W_dfl = np.load(varx/"Y_pred_model6_opt.npy")[-363:]
Y_te = np.load(PROJ/"特征工程"/"Y_targets.npy")[test_start:]

def evaluate(files, tag):
    models = {
        "e2e": {"name": "End-to-end DFL", "Y_pred": W_e2e[-(len(W_e2e)-(len(files)-len(files))) :] if False else None},
    }
    # slice weights to match file window
    if len(files) == 363:
        sl = slice(0, 363)
    else:
        sl = slice(65, 265)
    models = {
        1: {"name": "End-to-end DFL", "Y_pred": W_e2e[sl]},
        2: {"name": "Network VARX + Smooth (M5)", "Y_pred": W_m5[sl]},
        3: {"name": "Network VARX + Smooth + DFL (post-hoc)", "Y_pred": W_dfl[sl]},
    }
    res = t3.compute_all(models, files)
    rows = []
    for k, md in models.items():
        r = res[k]
        rows.append({"window": tag, "model": md["name"], "avg_ret": r["avg_ret"],
                     "vol_annual": r["vol_annual"], "rpv_annual": r["rpv_annual"],
                     "avg_turnover": r["avg_turnover"], "sharpe_net": r["sharpe_net"],
                     "max_dd": r["max_dd"], "cum_ret": r["cum_ret"]})
    rows.append({"window": tag, "model": "Equal Weight", "avg_ret": res["eq"]["avg_ret"],
                 "vol_annual": res["eq"]["vol_annual"], "rpv_annual": res["eq"]["rpv_annual"],
                 "avg_turnover": res["eq"]["avg_turnover"], "sharpe_net": res["eq"]["sharpe_net"],
                 "max_dd": res["eq"]["max_dd"], "cum_ret": res["eq"]["cum_ret"]})
    df = pd.DataFrame(rows)
    print(f"\n--- {tag} ---")
    print(df.to_string(index=False))
    return df

df1 = evaluate(test_files_full, "full_363")
df2 = evaluate(test_files_200, "window_200")

# MSE vs realized weights
for name, W in [("End-to-end DFL", W_e2e), ("M5", W_m5), ("Post-hoc DFL", W_dfl)]:
    print(name, "full MSE", round(float(((W-Y_te)**2).mean()), 10),
          "200d MSE", round(float(((W[65:265]-Y_te[65:265])**2).mean()), 10))

pd.concat([df1, df2]).to_csv(HERE/"e2e_metrics.csv", index=False)
print("\nsaved e2e_metrics.csv")
