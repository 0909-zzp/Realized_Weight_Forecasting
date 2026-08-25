"""Table 3 — 样本外投资组合表现 (200天窗口, GMVP W=20, DFL drift-aware L1)

指标：
  波动率   — 年化组合日收益标准差
  RPV      — 已实现组合方差 (主指标)
  Turnover — 日换手率 (经济换手, 排除被动漂移)
  净夏普   — 扣除交易成本后的年化夏普比率
  最大回撤 — 累计收益的最大跌幅

配置 (与HANDOFF一致):
  窗口: 2018-12-28 ~ 2019-10-15 (200天)
  GMVP基准: W=20天滚动窗口
  DFL: drift-aware L1, η=1e-6, ρ=1e-3 (OOS验证期选参)
       + 每日滚动协方差 cov_window=150
  GMVP基准权重文件: Y_pred_bench_*_W20.npy
  DFL预测: Y_pred_model6_opt.npy (drift-aware L1, η=1e-6)
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log, set_log_file, load_day, compute_raw_cov, EPS_RIDGE

# HANDOFF 配置
START_DATE = '20181228'  # 2018-12-28
END_DATE   = '20191015'  # 2019-10-15
GMVP_W = 20              # GMVP 滚动窗口
DFL_ETA = 1e-6            # drift-aware L1 换手惩罚 (OOS验证期选择)
DFL_RHO = 1e-3


# ===================================================================
# 加载数据
# ===================================================================
def load_data():
    root = Path(__file__).parents[1]
    varx_dir    = root / "VARX"
    feat_dir    = root / "特征工程"
    npy_dir     = root / "数据" / "1min_log_return_npy"

    # --- 200天窗口 ---
    all_files = sorted([f for f in npy_dir.iterdir()
                        if f.suffix == '.npy' and f.name[0].isdigit()])
    all_dates = [f.name[:8] for f in all_files]
    si = all_dates.index(START_DATE)
    ei = all_dates.index(END_DATE)
    test_files = all_files[si:ei+1]
    n_days = len(test_files)
    log(f"200天窗口: {START_DATE}~{END_DATE} ({n_days}天)")

    # --- VARX 模型预测 (test set, 取200天窗口 [60:260]) ---
    # 200天窗口在363天测试集中位于索引 [60:260]
    TSTART, TEND = 65, 265
    models = {}
    for mid, name in [(1,'VAR'),(2,'Sparse VAR'),(3,'Sparse VARX'),
                      (4,'Network VARX'),(5,'Network+Smooth')]:
        f = varx_dir / f"Y_pred_model{mid}.npy"
        if f.exists():
            models[mid] = {'name': name, 'Y_pred': np.load(f)[TSTART:TEND]}

    # DFL drift-aware L1 版本
    f_dfl = varx_dir / "Y_pred_model6_opt.npy"
    if f_dfl.exists():
        models[6] = {
            'name': 'Network+Smooth + DFL (L1 drift, η=1e-6, 滚动Σ150d)',
            'Y_pred': np.load(f_dfl)[TSTART:TEND],
        }

    # LSTM
    f_lstm = varx_dir / "Y_pred_model7.npy"
    if f_lstm.exists():
        models[7] = {'name': 'LSTM', 'Y_pred': np.load(f_lstm)[TSTART:TEND]}

    # --- GMVP 基准 (W=20) ---
    for mid_offset, name, fname in [
        (10, 'Sample GMVP (W=20)', 'Y_pred_bench_sample_W20.npy'),
        (11, 'Shrinkage GMVP (W=20)', 'Y_pred_bench_shrink_W20.npy'),
        (12, 'GLasso GMVP (W=20)', 'Y_pred_bench_glasso_W20.npy'),
    ]:
        f = varx_dir / fname
        if f.exists():
            models[mid_offset] = {'name': name, 'Y_pred': np.load(f)[TSTART:TEND]}

    return models, test_files, n_days


# ===================================================================
# 指标计算
# ===================================================================
def compute_all(models, test_files, eta=ETA):
    """遍历测试日, 计算每个模型的逐日组合收益与风险指标。"""
    n_days = len(test_files)
    results = {}

    # 预加载个股日度简单收益 (log→simple) + 已实现协方差
    log(f"加载 {n_days} 天日内收益...")
    t0 = time.time()
    simple_rets = np.zeros((n_days, K), dtype=np.float64)
    covs = [None] * n_days
    for i, fpath in enumerate(test_files):
        rett = np.load(str(fpath))
        log_ret = rett.sum(axis=1)  # log return, (K,)
        simple_rets[i] = np.expm1(log_ret)  # → simple return for compounding
        raw = compute_raw_cov(rett)
        raw.flat[::K+1] += EPS_RIDGE
        covs[i] = raw
        if i % 100 == 0 and i > 0:
            log(f"  {i}/{n_days}...")
    log(f"  {n_days} 天完成 ({time.time()-t0:.1f}s)")

    # 等权重基准 (与模型对齐: Y[t]→rets[t+1], 取rets[1:])
    w_eq = np.ones(K) / K
    eq_ret = simple_rets[1:] @ w_eq
    eq_rpv = np.array([w_eq @ covs[t+1] @ w_eq for t in range(n_days-1)])
    eq_drifted = w_eq * (1 + simple_rets[1:]) / (1 + simple_rets[1:] @ w_eq)[:, None]
    eq_to = np.abs(w_eq - eq_drifted).sum(axis=1)
    results['eq'] = _metrics(len(eq_ret), eq_ret, eq_to, eq_rpv, eta, '等权重')

    # 各模型
    for mid, md in models.items():
        Y = md['Y_pred'].astype(np.float64)
        T = min(len(Y)-1, n_days-1)
        port_ret = np.array([Y[t] @ simple_rets[t+1] for t in range(T)])
        # 经济换手率: 目标权重 vs 被动漂移权重 (排除价格变动的影响)
        to_vec = np.zeros(T, dtype=np.float64)
        for t in range(T):
            w_prev = Y[t]                    # t时刻的持仓权重
            r_next = simple_rets[t+1]         # t→t+1的个股简单收益
            w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
            to_vec[t] = np.sum(np.abs(Y[t+1] - w_drifted))
        rpv_vec = np.array([Y[t] @ covs[t+1] @ Y[t] for t in range(T)])

        results[mid] = _metrics(T, port_ret, to_vec, rpv_vec, eta, md['name'])

    return results


def _metrics(T, port_ret, to_vec, rpv_vec, eta, name):
    """单策略指标汇总。"""
    # 扣除交易成本的净收益
    net_ret = port_ret - eta * to_vec
    avg_ret    = float(np.mean(net_ret))
    vol_daily  = float(np.std(net_ret, ddof=1))
    vol_annual = vol_daily * np.sqrt(252)
    rpv_mean   = float(np.mean(rpv_vec))
    rpv_annual = rpv_mean * 252
    avg_to     = float(np.mean(to_vec))
    sr_annual = float(np.mean(net_ret)/np.std(net_ret, ddof=1)*np.sqrt(252)) if np.std(net_ret)>1e-15 else 0
    # 最大回撤
    cum = np.cumprod(1 + net_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))
    # 累计收益
    cum_ret = float(np.prod(1 + net_ret) - 1)

    return {
        'name': name,
        'avg_ret': avg_ret, 'vol_annual': vol_annual,
        'rpv_annual': rpv_annual, 'rpv_daily': rpv_mean,
        'avg_turnover': avg_to,
        'sharpe_net': sr_annual, 'max_dd': max_dd,
        'cum_ret': cum_ret,
    }


# ===================================================================
# 主入口
# ===================================================================
def main():
    out_dir = Path(__file__).parent
    set_log_file(out_dir / "table3_log.txt")
    log("=" * 70)
    log("Table 3 — 样本外投资组合表现")
    log("=" * 70)

    models, test_files, n_days = load_data()
    log(f"测试窗口: {START_DATE}~{END_DATE} ({n_days}天)")

    results = compute_all(models, test_files)

    # 打印表格
    header = f"{'Model':<18} {'波动率':>10} {'RPV(年)':>12} {'Turnover':>10} {'夏普':>8} {'最大回撤':>10}"
    log("\n" + header)
    log("-" * 76)
    rows = []
    for key in ['eq'] + list(models.keys()):
        r = results[key]
        log(f"{r['name']:<18} "
            f"{r['vol_annual']:>10.4f} "
            f"{r['rpv_annual']:>12.6e} "
            f"{r['avg_turnover']:>10.4f} "
            f"{r['sharpe_net']:>8.4f} "
            f"{r['max_dd']:>10.4f}")
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "Table3_results.csv", index=False)
    log(f"\n保存: Table3_results.csv")


if __name__ == "__main__":
    main()
