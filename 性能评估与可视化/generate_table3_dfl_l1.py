"""生成 Table3 使用的 drift-aware L1 DFL 权重。

输出: VARX/Y_pred_model6_opt.npy
配置: η=1e-6, ρ=1e-3, 每日滚动协方差 cov_window=150 (M5 base)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import log, set_log_file


ROOT = Path(__file__).parents[1]
VARX_DIR = ROOT / "VARX"
FEAT_DIR = ROOT / "特征工程"
DFL_ETA = 1e-6
DFL_RHO = 1e-3
DFL_RISK = 1.0
DFL_COV_WINDOW = 150
DFL_ROLLING_COV = True


def _load_varx():
    path = VARX_DIR / "VAR及拓展（table2）.py"
    spec = importlib.util.spec_from_file_location("vp", path)
    vp = importlib.util.module_from_spec(spec)
    sys.modules["vp"] = vp
    spec.loader.exec_module(vp)
    return vp


def main():
    set_log_file(Path(__file__).parent / "generate_table3_dfl_l1_log.txt")
    vp = _load_varx()

    data = vp.load_data()
    splits = vp.split_data(data["X"], data["Y"], data["A_bar"])
    Y_te = splits["test"][1]
    n_train = splits["train"][0].shape[0]
    valid_indices = np.load(FEAT_DIR / "valid_indices.npy")
    train_day_indices = valid_indices[:n_train]

    Y_pred_m5 = np.load(VARX_DIR / "Y_pred_model5.npy")[-len(Y_te):]
    log(f"DFL drift-aware L1: eta={DFL_ETA:g}, rho={DFL_RHO:g}, "
        f"risk={DFL_RISK:g}, rolling={DFL_ROLLING_COV}, "
        f"cov_window={DFL_COV_WINDOW}, T={len(Y_te)}")

    Y_dfl = vp.compute_model6_drift_l1(
        Y_pred_m5, Y_te, train_day_indices,
        eta=DFL_ETA, rho=DFL_RHO, risk_mult=DFL_RISK,
        rolling_cov=DFL_ROLLING_COV, cov_window=DFL_COV_WINDOW,
    )

    out_path = VARX_DIR / "Y_pred_model6_opt.npy"
    np.save(out_path, Y_dfl)
    log(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
