"""MCS (Model Confidence Set) — Hansen, Lunde & Nason (2011)

完整 bootstrap 实现, 替代 Table 2 中的简化 MSE 排名。

算法:
  1. 初始化 M = 全部模型
  2. 计算逐对损失差分 t-statistics, T_R = max |t_{ij}|
  3. Block bootstrap (块长度~5天) 获取 T_R 的零分布
  4. 若 H0 拒绝 (p < α), 剔除最差模型, 回到步骤 2
  5. 输出 MCS p-values (越大越好, >α 的模型进入置信集)

参考文献:
  Hansen, P. R., Lunde, A., & Nason, J. M. (2011).
  The model confidence set. Econometrica, 79(2), 453-497.

输出:
  MCS_results.csv   — 每个模型的 MCS p-value (75% 和 90% 置信水平)
  mcs_log.txt       — 运行日志
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ===================================================================
# 参数
# ===================================================================
N_BOOTSTRAP = 2000       # bootstrap 重复次数
BLOCK_LEN   = 5          # block bootstrap 块长度 (交易日, 约1周)
SEED        = 42         # 随机种子 (可复现)
ALPHA_LEVELS = [0.25, 0.10]  # MCS 置信水平: 75% 和 90%

MODEL_NAMES = {
    1: 'VAR (OLS)',
    2: 'Sparse VAR',
    3: 'Sparse VARX',
    4: 'Network VARX',
    5: 'Network+Smooth',
    6: 'DFL VARX (L2)',
    7: 'LSTM',
}


def load_loss_series() -> Tuple[np.ndarray, np.ndarray, int]:
    """加载各模型预测和真值, 计算逐日 MSE 损失序列。

    Returns:
        L:     (T, M) 损失矩阵, L[t,m] = MSE of model m at day t
        Y_te:  (T, K) 真实 GMVP 权重 (备用)
        n_test: 测试集天数
    """
    proj = Path(__file__).parents[1]
    pred_dir = proj / "VARX"
    Y_te = np.load(proj / "特征工程" / "Y_targets.npy")

    # 测试集从 Table 2 划分: 70/15/15 → test 是最后15% (余数363天)
    n_total = Y_te.shape[0]
    n_test = n_total - int(n_total * 0.7) - int(n_total * 0.15)  # 363, 与Table2一致
    Y_te = Y_te[-n_test:]  # 取测试集部分

    # 加载各模型预测
    L_list = []
    for mid in range(1, 8):
        Y_pred = np.load(pred_dir / f"Y_pred_model{mid}.npy")
        Y_pred = Y_pred[-n_test:]  # 对齐测试集
        # 逐日 MSE (对 K=392 维取平均)
        daily_mse = ((Y_pred - Y_te) ** 2).mean(axis=1)  # (T,)
        L_list.append(daily_mse)

    L = np.column_stack(L_list)  # (T, M)
    return L, Y_te, n_test


def block_bootstrap(T: int, B: int, block_len: int, rng: np.random.Generator
                    ) -> np.ndarray:
    """生成 B 个 block bootstrap 索引矩阵。

    移动块 bootstrap (Moving Block Bootstrap):
      1. 从 [0, T-block_len] 中随机抽 block_len 长度的连续块
      2. 重复 ceil(T/block_len) 次
      3. 拼接后取前 T 个

    Returns:
        idx_matrix: (B, T) 每行为一次 bootstrap 重采样索引
    """
    n_blocks = int(np.ceil(T / block_len))
    idx_matrix = np.zeros((B, T), dtype=np.int64)

    for b in range(B):
        resampled = []
        for _ in range(n_blocks):
            # 随机选起始位置
            start = rng.integers(0, T - block_len + 1)
            resampled.extend(range(start, start + block_len))
        idx_matrix[b] = np.array(resampled[:T])

    return idx_matrix


def mcs_procedure(L: np.ndarray, n_boot: int = N_BOOTSTRAP,
                  block_len: int = BLOCK_LEN,
                  seed: int = SEED
                  ) -> pd.DataFrame:
    """Hansen-Lunde-Nason (2011) MCS 主程序。

    Args:
        L: (T, M) 损失矩阵, L[t,m] = MSE of model m at day t

    Returns:
        DataFrame: model, MCS_pval, MCS_raw_pval, in_MCS_75, in_MCS_90
    """
    T, M = L.shape
    rng = np.random.default_rng(seed)
    log(f"MCS: T={T}天, M={M}模型, bootstrap={n_boot}次, block_len={block_len}")

    # 预生成所有 bootstrap 索引 (所有迭代共用)
    boot_idx = block_bootstrap(T, n_boot, block_len, rng)
    log(f"  Bootstrap 索引矩阵: ({n_boot} × {T}), 耗时忽略不计")

    # 初始化: 所有模型都在集合中, p-values 记录淘汰时的 p 值
    surviving = list(range(M))           # 当前存活模型索引 (0-based)
    model_pvals = np.zeros(M)            # 单调调整后的 MCS p-value
    model_raw_pvals = np.zeros(M)        # 淘汰时集合检验的原始 p-value
    eliminated_round = np.full(M, -1)    # 淘汰轮次

    round_num = 0
    while len(surviving) > 1:
        round_num += 1
        m_current = len(surviving)
        log(f"\n  Round {round_num}: {m_current} models surviving "
            f"→ {[MODEL_NAMES[s+1] for s in surviving]}")

        # ---- Step 1: 对当前存活集合计算损失差分 ----
        L_sub = L[:, surviving]                            # (T, m)
        d_mean = np.zeros((m_current, m_current))          # 均值差分

        for i in range(m_current):
            for j in range(m_current):
                if i == j:
                    continue
                d_mean[i, j] = np.mean(L_sub[:, i] - L_sub[:, j])

        # ---- Step 2: Bootstrap 重采样均值差分 ----
        # HLN 用 bootstrap 均值的方差标准化 t 统计量，而不是样本内方差。
        d_boot_mean = np.zeros((n_boot, m_current, m_current))
        for b in range(n_boot):
            loss_mean = L_sub[boot_idx[b], :].mean(axis=0)  # (m,)
            d_boot_mean[b] = loss_mean[:, None] - loss_mean[None, :]

        d_boot_var = np.mean(
            (d_boot_mean - d_mean[None, :, :]) ** 2, axis=0
        )
        d_boot_var = np.maximum(d_boot_var, 1e-20)
        d_boot_std = np.sqrt(d_boot_var)

        # 观测与 bootstrap t-statistics
        t_stat = d_mean / d_boot_std
        t_boot = (d_boot_mean - d_mean[None, :, :]) / d_boot_std

        # 检验统计量: T_R = max_{i,j} |t_{ij}|
        T_R_obs = np.max(np.abs(t_stat))
        T_R_boot = np.abs(t_boot).max(axis=(1, 2))

        # ---- Step 3: Bootstrap p-value ----
        p_val = np.mean(T_R_boot > T_R_obs)
        log(f"    T_R={T_R_obs:.3f}  p={p_val:.6f}  "
            f"({np.sum(T_R_boot > T_R_obs)}/{n_boot} bootstrap > T_R)")

        # ---- Step 4: 决策 ----
        # 对于每个 alpha, 若 p < alpha 则拒绝 H0 (集合不全是等优的)
        # 若 p >= alpha, 当前集合即 MCS

        # 淘汰最差模型 (不论 p 值如何, 只要不止一个模型就删)
        # 淘汰准则: argmax_i sup_{j} t_{ij} (i 相对所有 j 最差的那个)
        # 即 t_{i,•} = max_j t_{ij} 中最大的 i
        t_max_row = np.max(t_stat, axis=1)                 # (m,) 每行最大t
        worst_local = int(np.argmax(t_max_row))            # 行索引 (0..m-1)
        worst_global = surviving[worst_local]              # 全局模型索引

        # 记录此模型在当前集合被淘汰时的原始 p-value
        model_pvals[worst_global] = p_val
        model_raw_pvals[worst_global] = p_val
        eliminated_round[worst_global] = round_num

        log(f"    淘汰: {MODEL_NAMES[worst_global+1]} (loc={worst_local}, "
            f"t_max={t_max_row[worst_local]:.3f})")

        # 从存活集合移除
        surviving.pop(worst_local)

    # 最后一个幸存模型: p-value = 1.0 (总在 MCS 中)
    if surviving:
        last = surviving[0]
        model_pvals[last] = 1.0
        model_raw_pvals[last] = 1.0
        eliminated_round[last] = round_num + 1
        log(f"\n  最终幸存: {MODEL_NAMES[last+1]} (p=1.0)")

    # Hansen et al. (2011): 序贯 p 值单调调整
    # 只有展示用的 MCS p-value 做单调调整，原始 p-value 保留用于 membership/rank
    order = np.argsort(eliminated_round)  # 从最早淘汰到最后幸存
    running_max = 0.0
    for idx in order:
        running_max = max(running_max, model_raw_pvals[idx])
        model_pvals[idx] = running_max

    # ---- 输出: 对每个 α 判断是否在 MCS 中 ----
    results = []
    for mid in range(M):
        results.append({
            'Model': mid + 1,
            'Name': MODEL_NAMES[mid + 1],
            'MCS_pval': round(model_pvals[mid], 6),
            'MCS_raw_pval': model_raw_pvals[mid],
            'Eliminated_round': eliminated_round[mid],
        })
    df = pd.DataFrame(results)

    # MCS rank: 原始 p 值降序；p 值相同时按淘汰轮次降序（越晚淘汰越优）
    rank_order = df.sort_values(
        ['MCS_raw_pval', 'Eliminated_round'],
        ascending=[False, False],
    ).index
    rank_map = {
        model: i + 1
        for i, model in enumerate(df.loc[rank_order, 'Model'])
    }
    df['MCS_rank'] = df['Model'].map(rank_map).astype(int)

    # 为每个 alpha 添加列
    for alpha in ALPHA_LEVELS:
        col_name = f'in_MCS_{int((1-alpha)*100)}'
        df[col_name] = df['MCS_raw_pval'] > alpha
        df[f'MCS_pval_{(1-alpha):.0%}'] = df['MCS_pval'].apply(
            lambda p: f"{p:.4f}"
        )

    return df


def log(msg: str) -> None:
    """控制台+文件日志。"""
    print(msg, flush=True)
    log_file = Path(__file__).parent / "mcs_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    log("=" * 72)
    log("  MCS (Model Confidence Set) — Hansen, Lunde & Nason (2011)")
    log(f"  Bootstrap: {N_BOOTSTRAP}次  Block长度: {BLOCK_LEN}天")
    log("=" * 72)

    # ---- 加载数据 ----
    L, Y_te, n_test = load_loss_series()
    T, M = L.shape
    log(f"\n数据: T={T}测试天, M={M}模型")
    log(f"模型列表: {[MODEL_NAMES[i+1] for i in range(M)]}")

    # 打印各模型 MSE
    log(f"\n各模型测试集 MSE:")
    for mid in range(M):
        avg_mse = np.mean(L[:, mid])
        log(f"  M{mid+1} {MODEL_NAMES[mid+1]:<20} {avg_mse:.4e}")

    # ---- 运行 MCS ----
    t0 = time.time()
    results = mcs_procedure(L, N_BOOTSTRAP, BLOCK_LEN, SEED)
    elapsed = time.time() - t0
    log(f"\n总耗时: {elapsed:.1f}s")

    # ---- 输出结果 ----
    log(f"\n{'='*72}")
    log(f"MCS 结果")
    log(f"{'='*72}")
    for alpha in ALPHA_LEVELS:
        pct = int((1 - alpha) * 100)
        in_set = results[results[f'in_MCS_{pct}']]
        log(f"\n{pct}% 置信集 ({len(in_set)}/{M} 模型):")
        for _, row in in_set.iterrows():
            log(f"  M{int(row['Model'])} {row['Name']:<20} p={row['MCS_pval']:.4f}")

    log(f"\n完整 p-values:")
    for _, row in results.iterrows():
        log(f"  M{int(row['Model'])} {row['Name']:<20} "
            f"p(adj)={row['MCS_pval']:.4f} p(raw)={row['MCS_raw_pval']:.4f}  "
            f"淘汰轮次={int(row['Eliminated_round'])}")

    # ---- 保存 ----
    out_path = Path(__file__).parent / "MCS_results.csv"
    results.to_csv(out_path, index=False)
    log(f"\n结果已保存: {out_path}")

    # ---- Table 2 风格输出 ----
    log(f"\n{'='*72}")
    log(f"Table 2 MCS 列 (MCS rank = raw p 降序, 同 p 按淘汰轮次降序)")
    log(f"{'='*72}")
    log(f"{'#':>2} {'Model':<20} {'MSE':>12} {'MCS p-val':>12} {'MCS rank':>10} {'75%MCS':>8} {'90%MCS':>8}")
    log("-" * 78)
    for _, row in results.iterrows():
        avg_mse = np.mean(L[:, int(row['Model'])-1])
        mcs75 = '✓' if row[f'in_MCS_75'] else '✗'
        mcs90 = '✓' if row[f'in_MCS_90'] else '✗'
        log(f"{int(row['Model']):>2} {row['Name']:<20} "
            f"{avg_mse:>12.4e} {row['MCS_pval']:>12.4f} {int(row['MCS_rank']):>10} "
            f"{mcs75:>8} {mcs90:>8}")

    log(f"\n{'='*72}")
    log("MCS 完成")
    log("=" * 72)


if __name__ == "__main__":
    main()
