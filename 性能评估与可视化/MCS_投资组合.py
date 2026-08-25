"""投资组合收益版 MCS（Hansen, Lunde & Nason, 2011）。

严格使用 HANDOFF.md 的 Table 3 评估窗口：2018-12-19 至 2019-10-08，
共 200 个交易日。策略为等权重、Sparse VARX、Network VARX 和
DFL VARX（eta=0 的 Y_pred_model6_opt）。损失定义为扣除 1bp 交易成本后的
日组合收益的相反数，即 loss_t = -(r_t - ETA * turnover_t)。

这是投资表现的探索性检验，不替代 Table 2 的 MSE-MCS。
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "图形Lasso" / "code"))
from 共享模块 import K, ETA, load_day  # noqa: E402

START_DAY = 2115  # 2018-12-19
END_DAY = 2314    # 2019-10-08
N_BOOTSTRAP = 5000
SEED = 42
BLOCK_LENGTHS = (1, 5, 10)


def moving_block_indices(T, B, block_len, rng):
    """生成 B 次移动块 bootstrap 索引。"""
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T - block_len + 1, size=(B, n_blocks))
    offsets = np.arange(block_len)
    return (starts[:, :, None] + offsets).reshape(B, -1)[:, :T]


def mcs(L, block_len):
    """TR 统计量的逐步 MCS，返回每个模型的 p 值与淘汰轮次。"""
    T, M = L.shape
    rng = np.random.default_rng(SEED)
    idx = moving_block_indices(T, N_BOOTSTRAP, block_len, rng)
    active = list(range(M))
    pvals = np.zeros(M)
    rounds = np.full(M, -1, dtype=int)
    round_no = 0

    while len(active) > 1:
        round_no += 1
        X = L[:, active]
        # d[i,j] = loss_i - loss_j；正值表示 i 更差。
        d = X[:, :, None] - X[:, None, :]
        dbar = d.mean(axis=0)
        sd = d.std(axis=0, ddof=1)
        t_obs = np.divide(dbar, sd / np.sqrt(T), out=np.zeros_like(dbar), where=sd > 1e-15)
        tr_obs = float(np.abs(t_obs).max())

        # 在 H0 下对差分序列中心化后重抽样，向量化计算 bootstrap TR。
        d_boot_mean = d[idx].mean(axis=1) - dbar
        t_boot = np.divide(d_boot_mean, sd[None, :, :] / np.sqrt(T),
                           out=np.zeros_like(d_boot_mean), where=sd[None, :, :] > 1e-15)
        tr_boot = np.abs(t_boot).max(axis=(1, 2))
        pval = float((tr_boot >= tr_obs).mean())

        worst_local = int(np.argmax(t_obs.max(axis=1)))
        worst_global = active.pop(worst_local)
        pvals[worst_global] = pval
        rounds[worst_global] = round_no

    pvals[active[0]] = 1.0
    rounds[active[0]] = round_no + 1
    return pvals, rounds


def load_returns(full_test=False, all_table2=False):
    """按 Table 3 的 t 权重、t+1 实现收益时序，构造 200 天净收益。"""
    valid = np.load(ROOT / "特征工程" / "valid_indices.npy")
    test_start = int(np.where(valid == 2055)[0][0])
    if full_test:
        # 与原 Table 3 的 t 权重、t+1 收益口径一致：363 组预测产生 362 个实现收益。
        start_pos, end_pos = test_start + 1, len(valid) - 1
    else:
        start_pos = int(np.where(valid == START_DAY)[0][0])
        end_pos = int(np.where(valid == END_DAY)[0][0])
    if not full_test and end_pos - start_pos + 1 != 200:
        raise RuntimeError("handoff 指定窗口不是 200 天，请检查 valid_indices.npy")

    realized = np.vstack([load_day(int(d)).sum(axis=1) for d in valid[start_pos:end_pos + 1]])
    rows = np.arange(start_pos - test_start, end_pos - test_start + 1)
    if all_table2:
        # 与 Table 2 的 M1--M7 一一对应；M6 保持原始 L2 预测，不以 eta=0 版本替代。
        specs = [
            ("M1 VAR (OLS)", "Y_pred_model1.npy"),
            ("M2 Sparse VAR", "Y_pred_model2.npy"),
            ("M3 Sparse VARX", "Y_pred_model3.npy"),
            ("M4 Network VARX", "Y_pred_model4.npy"),
            ("M5 Network+Smooth", "Y_pred_model5.npy"),
            ("M6 DFL VARX (L2)", "Y_pred_model6.npy"),
            ("M7 LSTM", "Y_pred_model7.npy"),
        ]
    else:
        specs = [
            ("Equal weight", None),
            ("Sparse VARX", "Y_pred_model3.npy"),
            ("Network VARX", "Y_pred_model4.npy"),
            ("DFL VARX (eta=0)", "Y_pred_model6_opt.npy"),
        ]

    net_returns, turnovers = {}, {}
    for name, filename in specs:
        if filename is None:
            gross = realized @ (np.ones(K) / K)
            turnover = np.zeros(len(realized))
        else:
            y = np.load(ROOT / "VARX" / filename).astype(np.float64)
            w = y[rows]
            gross = (w * realized).sum(axis=1)
            drifted = w * (1.0 + realized) / (1.0 + gross)[:, None]
            turnover = np.abs(y[rows + 1] - drifted).sum(axis=1)
        net_returns[name] = gross - ETA * turnover
        turnovers[name] = turnover
    return pd.DataFrame(net_returns), pd.DataFrame(turnovers), len(realized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-test", action="store_true",
                        help="使用完整测试期（362 个可实现组合收益）作为附录稳健性检验")
    parser.add_argument("--all-table2", action="store_true",
                        help="仅比较 Table 2 的 M1--M7，并使用其原始预测文件")
    args = parser.parse_args()
    net, turnover, n_days = load_returns(full_test=args.full_test, all_table2=args.all_table2)
    loss = -net.to_numpy()
    summary = pd.DataFrame({
        "Model": net.columns,
        "Mean_net_daily_return": net.mean().to_numpy(),
        "Net_Sharpe_annual": (net.mean() / net.std(ddof=1) * np.sqrt(252)).to_numpy(),
        "Mean_turnover": turnover.mean().to_numpy(),
    })

    rows = []
    for block_len in BLOCK_LENGTHS:
        pvals, rounds = mcs(loss, block_len)
        for i, name in enumerate(net.columns):
            rows.append({"Block_length": block_len, "Model": name,
                         "MCS_pval": pvals[i], "Eliminated_round": rounds[i],
                         "In_MCS_75": pvals[i] > 0.25, "In_MCS_90": pvals[i] > 0.10})
    results = pd.DataFrame(rows).merge(summary, on="Model", how="left")
    # 与现有 MSE-MCS 一致：p-value 越大代表越优，rank=1 最优。
    results["Investment_MCS_rank"] = results.groupby("Block_length")["MCS_pval"].rank(
        ascending=False, method="min"
    ).astype(int)

    out = Path(__file__).parent
    suffix = "_extended" if args.full_test else ""
    if args.all_table2:
        suffix += "_Table2all"
    results.to_csv(out / f"MCS_投资组合_results{suffix}.csv", index=False, encoding="utf-8-sig")
    net.corr().to_csv(out / f"MCS_投资组合_net_return_correlation{suffix}.csv", encoding="utf-8-sig")
    net.to_csv(out / f"MCS_投资组合_net_returns{suffix}.csv", index=False, encoding="utf-8-sig")

    label = "完整测试期（362 天）" if args.full_test else "200 天（2018-12-19 至 2019-10-08）"
    if args.all_table2:
        label += "，Table 2 全部 M1--M7"
    print(f"投资组合 MCS：{label}，损失 = -净日收益")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMCS 结果（block=5）：")
    print(results[results["Block_length"] == 5][["Model", "MCS_pval", "Investment_MCS_rank", "Eliminated_round", "In_MCS_75", "In_MCS_90"]].to_string(index=False))
    print(f"\n已保存至：{out}")


if __name__ == "__main__":
    main()
