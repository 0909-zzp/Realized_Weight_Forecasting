"""Table 5 — Market State Analysis (市场状态分析)

口径与论文§5.5一致:
  市场状态   : 用 VIX 分位数划分高波动/低波动两个状态
               (VIX >= 中位数 -> high_vol, 否则 low_vol)
  评估窗口   : 200天 [65:265] = 2018-12-28 ~ 2019-10-15
  组合收益   : 与Table3相同的逐日净收益 (扣1bp双边交易成本), 第1天用于建仓,
               从第2天起共199天; 状态与199天收益按日期对齐
  状态内指标 : mean_daily / vol_ann / sharpe_ann / turnover 只在状态子样本内计算
  high-low   : 高波动状态均值 - 低波动状态均值, Newey-West HAC (maxlags=5) t检验

产出: table5_market_state.csv
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log, set_log_file

TSTART, TEND = 65, 265
VIX_CSV = Path(__file__).parents[1] / "数据" / "vix_daily.csv"
NPY_DIR = Path(__file__).parents[1] / "数据" / "1min_log_return_npy"
VARX_DIR = Path(__file__).parents[1] / "VARX"
FEAT_DIR = Path(__file__).parents[1] / "特征工程"
OUT_CSV = Path(__file__).parent / "table5_market_state.csv"

MODELS = {
    "NetworkVARX+Smooth+DFL": VARX_DIR / "Y_pred_model6_opt.npy",
    "NetworkVARX": VARX_DIR / "Y_pred_model4.npy",
    "NetworkVARX+Smooth": VARX_DIR / "Y_pred_model5.npy",
    "SparseVARX": VARX_DIR / "Y_pred_model3.npy",
    "SampleGMVP": VARX_DIR / "Y_pred_bench_sample_W20.npy",
    "EqualWeight": None,
}


def load_market_state() -> np.ndarray:
    """VIX 中位数划分高/低波动状态, 与199天组合收益对齐。"""
    valid = np.load(FEAT_DIR / "valid_indices.npy")
    n_total = valid.shape[0]
    n_test = n_total - int(n_total * 0.7) - int(n_total * 0.15)
    win_idx = valid[-n_test:][TSTART:TEND]

    all_files = sorted([f for f in NPY_DIR.iterdir()
                        if f.suffix == ".npy" and f.name[0].isdigit()])
    dates = np.array([int(f.name[:8]) for f in all_files])
    win_dates = dates[win_idx]

    vix = pd.read_csv(VIX_CSV)
    dmap = pd.Series(vix["vix"].values, index=vix["date"].astype(int))
    vix_win = dmap.reindex(win_dates).values.astype(float)[1:]  # 199天
    if np.isnan(vix_win).any():
        raise ValueError(f"VIX缺失{int(np.isnan(vix_win).sum())}天, 需先补数")

    med = float(np.median(vix_win))
    state = np.where(vix_win >= med, "high_vol", "low_vol")
    counts = {s: int((state == s).sum()) for s in ("high_vol", "low_vol")}
    log(f"VIX中位数: {med:.2f}  状态天数: {counts} (合计{sum(counts.values())})")
    return state


def load_daily_returns() -> tuple:
    valid = np.load(FEAT_DIR / "valid_indices.npy")
    n_total = valid.shape[0]
    n_test = n_total - int(n_total * 0.7) - int(n_total * 0.15)
    win_idx = valid[-n_test:][TSTART:TEND]

    all_files = sorted([f for f in NPY_DIR.iterdir()
                        if f.suffix == ".npy" and f.name[0].isdigit()])
    files = [all_files[i] for i in win_idx]
    simple_rets = np.array([np.expm1(np.load(f).sum(axis=1)) for f in files])
    return simple_rets, files


def model_net_returns(Y: np.ndarray, simple_rets: np.ndarray) -> tuple:
    T = len(Y) - 1
    port_ret = np.array([Y[t] @ simple_rets[t + 1] for t in range(T)])
    to_vec = np.zeros(T, dtype=np.float64)
    for t in range(T):
        w_prev = Y[t]
        r_next = simple_rets[t + 1]
        w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
        to_vec[t] = np.abs(Y[t + 1] - w_drifted).sum()
    net_ret = port_ret - ETA * to_vec
    return net_ret, to_vec


def hac_t_high_low(y: np.ndarray, state: np.ndarray) -> tuple:
    import statsmodels.api as sm
    from scipy import stats as sp_stats

    hi = (state == "high_vol").astype(float)
    lo = (state == "low_vol").astype(float)
    X = np.column_stack([np.ones(len(y)), hi, lo])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    b = res.params[1] - res.params[2]
    se = np.sqrt(res.cov_params()[1, 1] + res.cov_params()[2, 2]
                 - 2 * res.cov_params()[1, 2])
    t = b / se if se > 0 else 0.0
    p = 2 * (1 - sp_stats.norm.cdf(abs(t)))
    return float(t), float(p)


def main():
    set_log_file(Path(__file__).parent / "table5_market_state_log.txt")
    log("=" * 70)
    log("Table 5 — Market State Analysis (VIX分位, 论文§5.5)")
    log("=" * 70)

    state = load_market_state()
    simple_rets, files = load_daily_returns()
    log(f"窗口: {files[0].name[:8]} ~ {files[-1].name[:8]} ({len(files)}天)")

    rows = []
    for name, fpath in MODELS.items():
        if fpath is None:
            Y = np.ones((len(files), K)) / K
        else:
            Y = np.load(fpath)[-363:][TSTART:TEND].astype(np.float64)
        net_ret, to_vec = model_net_returns(Y, simple_rets)

        for s in ("low_vol", "high_vol"):
            idx = state == s
            days = int(idx.sum())
            mean_daily = float(net_ret[idx].mean())
            vol_daily = float(net_ret[idx].std(ddof=1))
            vol_ann = vol_daily * np.sqrt(252)
            sharpe_ann = mean_daily / vol_daily * np.sqrt(252) if vol_daily > 1e-15 else 0.0
            turnover = float(to_vec[idx].mean())
            rows.append([name, s, days, mean_daily, vol_ann, sharpe_ann, turnover, np.nan, np.nan])

        hi_idx = state == "high_vol"
        lo_idx = state == "low_vol"
        diff = float(net_ret[hi_idx].mean() - net_ret[lo_idx].mean())
        t, p = hac_t_high_low(net_ret, state)
        rows.append([name, "high-low", int(lo_idx.sum()), diff, np.nan, np.nan, np.nan, t, p])
        log(f"{name}: high-low diff={diff:.6f}  HAC t={t:.3f}  p={p:.4f}")

    df = pd.DataFrame(rows, columns=[
        "model", "state", "days", "mean_daily", "vol_ann", "sharpe_ann",
        "turnover", "hac_t", "hac_p",
    ])
    df.to_csv(OUT_CSV, index=False)
    log(f"保存: {OUT_CSV}")
    log("=" * 70)


if __name__ == "__main__":
    main()
