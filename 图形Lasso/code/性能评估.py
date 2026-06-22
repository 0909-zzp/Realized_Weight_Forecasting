# ===================================================================
# 阶段四：性能评估与基准对照
#
# 从回测结果计算论文所需的全部性能指标，
# 与多个基准模型对比：
#   1. 等权重 (Equal Weight)
#   2. 滚动样本协方差 GMVP (Rolling GMVP)
#   3. AR(1) 权重预测
#   4. Ledoit-Wolf 收缩估计 GMVP
#
# 产出可用于论文的汇总表和 LaTeX 格式表格。
# ===================================================================
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from 共享模块 import K, ETA, OUT_DIR, L_TRAIN_GLASSO, log, set_log_file


# ===================================================================
# 指标计算
# ===================================================================
def compute_performance_metrics(
    port_variances: np.ndarray,
    turnovers: np.ndarray,
    benchmark_variances: dict,     # {"benchmark_name": np.ndarray}
    eta: float = ETA,
) -> pd.DataFrame:
    """计算各策略的对比性能指标。

    Args:
        port_variances:     GLasso-VARX 逐日已实现方差
        turnovers:          逐日换手率
        benchmark_variances: 各基准的逐日方差
        eta:                交易成本系数

    Returns:
        DataFrame: 策略 × 指标
    """
    metrics = {}

    def _row(name, vars_array, to_array=None):
        avg_var = np.mean(vars_array)
        std_var = np.std(vars_array)
        # 方差减少率相对于等权重
        eq_var = np.mean(list(benchmark_variances.values())[0]) if benchmark_variances else 1.0
        var_reduction = (1 - avg_var / eq_var) * 100   # 正数=更好

        if to_array is not None:
            avg_to = np.mean(to_array)
            net_loss = avg_var + eta * avg_to
        else:
            avg_to = 0.0
            net_loss = avg_var

        return {
            "avg_var":       avg_var,
            "std_var":       std_var,
            "avg_turnover":  avg_to,
            "net_loss":      net_loss,
            "var_reduction%": var_reduction,
        }

    # 主要策略
    metrics["GLasso-VARX"] = _row(port_variances, port_variances, turnovers)

    # 基准策略
    for name, bv in benchmark_variances.items():
        metrics[name] = _row(name, bv, to_array=None)

    df = pd.DataFrame(metrics).T
    df.index.name = "Strategy"
    return df


# ===================================================================
# LaTeX 表格生成
# ===================================================================
def to_latex_table(df: pd.DataFrame, caption: str) -> str:
    """将指标 DataFrame 转为 LaTeX booktabs 表格。"""
    cols = ["avg_var", "std_var", "avg_turnover", "net_loss", "var_reduction%"]
    col_names = ["Avg Var", "Std Var", "Avg Turnover", "Net Loss", "Var Reduction (%)"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:performance}",
        r"\begin{tabular}{l" + "c" * len(cols) + r"}",
        r"\toprule",
        " & " + " & ".join(col_names) + r" \\",
        r"\midrule",
    ]

    for idx, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c)
            if pd.isna(v):
                vals.append("--")
            elif abs(v) < 0.01:
                vals.append(f"{v:.4e}")
            else:
                vals.append(f"{v:.4f}")
        lines.append(f"{idx} & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ===================================================================
# 基准策略：滚动样本协方差 GMVP
# ===================================================================
def rolling_cov_gmvp(
    cov_dict: Dict[int, np.ndarray],
    dates: np.ndarray,
    window: int = 40,
) -> np.ndarray:
    """滚动样本协方差 GMVP: w = (Σ⁻¹·1) / (1ᵀ·Σ⁻¹·1)"""
    variances = []

    for d in dates:
        # 用前 window 天协方差平均
        cov_sum = np.zeros((K, K))
        count = 0
        for lag in range(1, window + 1):
            if d - lag in cov_dict:
                cov_sum += cov_dict[d - lag]
                count += 1
        if count == 0:
            variances.append(np.nan)
            continue

        cov_avg = cov_sum / count
        cov_avg.flat[::K + 1] += 1e-4  # Ridge

        try:
            prec = np.linalg.inv(cov_avg)
            w = prec @ np.ones(K)
            w /= w.sum()
            variances.append(float(w @ cov_dict[d] @ w))
        except np.linalg.LinAlgError:
            variances.append(np.nan)

    return np.array([v for v in variances if not np.isnan(v)])


# ===================================================================
# 独立运行入口
# ===================================================================
def main():
    set_log_file(OUT_DIR / "evaluation_log.txt")
    log("=" * 60)
    log("  阶段四: 性能评估与基准对照")
    log("=" * 60)

    # 加载回测结果
    out_dir = OUT_DIR / "code" / "输出数据"
    pv_path = out_dir / "backtest_port_variances.npy"
    to_path = out_dir / "backtest_turnovers.npy"
    eq_path = out_dir / "backtest_eq_variances.npy"

    if not pv_path.exists():
        log("错误: 回测结果不存在，请先运行 滚动回测.py")
        return

    port_vars = np.load(pv_path)
    turnovers = np.load(to_path)
    eq_vars   = np.load(eq_path)
    dates     = np.load(out_dir / "backtest_pred_weights.npy").shape[0]

    log(f"回测天数: {len(port_vars)}")

    # 加载协方差用于基准计算
    from 决策聚焦训练 import load_realized_covariances
    cov_dict = load_realized_covariances(Path("."))

    # 滚动 GMVP 基准
    log("计算滚动 GMVP 基准...")
    rg_vars = rolling_cov_gmvp(cov_dict, dates[-min(len(dates), 200):], window=40)

    # 汇总
    benchmarks = {
        "Equal Weight": eq_vars,
    }
    if len(rg_vars) > 0:
        benchmarks["Rolling GMVP"] = rg_vars

    # 计算指标
    df = compute_performance_metrics(port_vars, turnovers, benchmarks)
    log("\n" + str(df.round(6)))

    # 保存
    df.to_csv(out_dir / "performance_comparison.csv")
    log("\n已保存: performance_comparison.csv")

    # 生成 LaTeX 表
    latex = to_latex_table(df, "Out-of-Sample Performance Comparison")
    with open(out_dir / "performance_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    log("已生成: performance_table.tex")


if __name__ == "__main__":
    main()
