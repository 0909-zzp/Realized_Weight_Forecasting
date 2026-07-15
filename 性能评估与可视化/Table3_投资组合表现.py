"""Table 3 — 样本外投资组合表现

指标：
  波动率   — 年化组合日收益标准差
  RPV      — 已实现组合方差 (主指标)
  Turnover — 日换手率
  净夏普   — 扣除交易成本后的年化夏普比率
  最大回撤 — 累计收益的最大跌幅

基准对比：
  等权重 (1/K)、VAR、Lasso 族、DFL、LSTM
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


# ===================================================================
# 加载数据
# ===================================================================
def load_data():
    root = Path(__file__).parents[1]
    varx_dir    = root / "VARX"
    feat_dir    = root / "特征工程"

    # 模型预测权重
    models = {}
    for mid, name in [(1,'VAR'),(2,'Sparse VAR'),(3,'Sparse VARX'),
                      (4,'Network VARX'),(5,'+Smooth'),(6,'DFL VARX'),(7,'LSTM')]:
        f = varx_dir / f"Y_pred_model{mid}.npy"
        if f.exists():
            models[mid] = {'name': name, 'Y_pred': np.load(f)}

    # 测试日索引
    valid_idx = np.load(feat_dir / "valid_indices.npy")
    n = len(np.load(feat_dir / "Y_targets.npy"))
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    test_days = valid_idx[n_train + n_val:]

    return models, test_days, n_train, n_val


# ===================================================================
# 指标计算
# ===================================================================
def compute_all(models, test_days, eta=ETA):
    """遍历测试日，计算每个模型的逐日组合收益与风险指标。"""
    n_days = len(test_days)
    results = {}

    # 预加载个股日度收益 + 已实现协方差
    log(f"加载 {n_days} 天日内收益...")
    t0 = time.time()
    daily_rets = np.zeros((n_days, K), dtype=np.float64)
    covs = [None] * n_days
    for i, didx in enumerate(test_days):
        rett = load_day(didx)
        daily_rets[i] = rett.sum(axis=1)
        raw = compute_raw_cov(rett)
        raw.flat[::K+1] += EPS_RIDGE
        covs[i] = raw
        if i % 100 == 0 and i > 0:
            log(f"  {i}/{n_days}...")
    log(f"  {n_days} 天完成 ({time.time()-t0:.1f}s)")

    # 等权重基准
    w_eq = np.ones(K) / K
    eq_ret = daily_rets @ w_eq
    eq_rpv = np.array([w_eq @ covs[t] @ w_eq for t in range(n_days)])
    results['eq'] = _metrics(len(eq_ret), eq_ret, np.zeros(len(eq_ret)), eq_rpv, eta, '等权重')

    # 各模型
    for mid, md in models.items():
        Y = md['Y_pred'].astype(np.float64)
        T = min(len(Y)-1, n_days-1)
        port_ret = np.array([Y[t] @ daily_rets[t+1] for t in range(T)])
        to_vec   = np.array([np.sum(np.abs(Y[t+1] - Y[t])) for t in range(T)])
        rpv_vec  = np.array([Y[t] @ covs[t+1] @ Y[t] for t in range(T)])
        results[mid] = _metrics(T, port_ret, to_vec, rpv_vec, eta, md['name'])

    return results


def _metrics(T, port_ret, to_vec, rpv_vec, eta, name):
    """单策略指标汇总。"""
    avg_ret    = float(np.mean(port_ret))
    vol_daily  = float(np.std(port_ret, ddof=1))
    vol_annual = vol_daily * np.sqrt(252)
    rpv_mean   = float(np.mean(rpv_vec))
    rpv_annual = rpv_mean * 252
    avg_to     = float(np.mean(to_vec))
    # 扣除交易成本的净收益
    net_ret = port_ret - eta * to_vec
    sr_annual = float(np.mean(net_ret)/np.std(net_ret, ddof=1)*np.sqrt(252)) if np.std(net_ret)>1e-15 else 0
    # 最大回撤
    cum = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))
    # 累计收益
    cum_ret = float(np.prod(1 + port_ret) - 1)

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

    models, test_days, n_train, n_val = load_data()
    log(f"测试日: {len(test_days)} 天 (索引 {test_days[0]}~{test_days[-1]})")

    results = compute_all(models, test_days)

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
