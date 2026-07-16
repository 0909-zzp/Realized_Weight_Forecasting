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

    # 测试集从 Table 2 划分: 70/15/15 → test 是最后15%
    n_total = Y_te.shape[0]
    n_test = int(n_total * 0.15)
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
        DataFrame: model, MCS_pval_75, MCS_pval_90, in_MCS_75, in_MCS_90
    """
    T, M = L.shape
    rng = np.random.default_rng(seed)
    log(f"MCS: T={T}天, M={M}模型, bootstrap={n_boot}次, block_len={block_len}")

    # 预生成所有 bootstrap 索引 (所有迭代共用)
    boot_idx = block_bootstrap(T, n_boot, block_len, rng)
    log(f"  Bootstrap 索引矩阵: ({n_boot} × {T}), 耗时忽略不计")

    # 初始化: 所有模型都在集合中, p-values 记录淘汰时的 p 值
    surviving = list(range(M))           # 当前存活模型索引 (0-based)
    model_pvals = np.zeros(M)            # 每个模型被淘汰时的集合 p-value
    eliminated_round = np.full(M, -1)    # 淘汰轮次

    round_num = 0
    while len(surviving) > 1:
        round_num += 1
        m_current = len(surviving)
        log(f"\n  Round {round_num}: {m_current} models surviving "
            f"→ {[MODEL_NAMES[s+1] for s in surviving]}")

        # ---- Step 1: 对当前存活集合计算损失差分 t-statistics ----
        L_sub = L[:, surviving]                            # (T, m)
        d_mean = np.zeros((m_current, m_current))          # 均值差分
        d_var  = np.zeros((m_current, m_current))          # 方差
        t_stat = np.zeros((m_current, m_current))          # t-statistics

        for i in range(m_current):
            for j in range(m_current):
                if i == j:
                    continue
                dij = L_sub[:, i] - L_sub[:, j]            # (T,)
                d_mean[i, j] = np.mean(dij)
                v = np.var(dij, ddof=1)
                d_var[i, j] = v
                if v > 1e-20:
                    t_stat[i, j] = d_mean[i, j] / np.sqrt(v / T)

        # 检验统计量: T_R = max_{i,j} |t_{ij}|
        T_R_obs = np.max(np.abs(t_stat))

        # ---- Step 2: Bootstrap 零分布 ----
        # H0: E[d_{ij,t}] = 0  → 中心化: d*_{ij,t} = d_{ij,t} - d̄_{ij}
        # 对每对 (i,j), 我们只需要 bootstrap T_R, 不需要逐个保存所有差分
        T_R_boot = np.zeros(n_boot)

        for b in range(n_boot):
            idx = boot_idx[b]
            L_boot = L_sub[idx, :]                          # (T, m) 重采样
            t_boot = np.zeros((m_current, m_current))

            for i in range(m_current):
                for j in range(m_current):
                    if i == j:
                        continue
                    dij_b = L_boot[:, i] - L_boot[:, j]     # (T,)
                    # 中心化: 原差分均值假设为零
                    dij_centered = dij_b - d_mean[i, j]     # H0: mean=0
                    dm = np.mean(dij_centered)
                    vv = np.var(dij_centered, ddof=1)
                    if vv > 1e-20:
                        t_boot[i, j] = dm / np.sqrt(vv / T)

            T_R_boot[b] = np.max(np.abs(t_boot))

        # ---- Step 3: Bootstrap p-value ----
        p_val = np.mean(T_R_boot >= T_R_obs)
        log(f"    T_R={T_R_obs:.3f}  p={p_val:.6f}  "
            f"({np.sum(T_R_boot >= T_R_obs)}/{n_boot} bootstrap >= T_R)")

        # ---- Step 4: 决策 ----
        # 对于每个 alpha, 若 p < alpha 则拒绝 H0 (集合不全是等优的)
        # 若 p >= alpha, 当前集合即 MCS

        # 淘汰最差模型 (不论 p 值如何, 只要不止一个模型就删)
        # 淘汰准则: argmax_i sup_{j} t_{ij} (i 相对所有 j 最差的那个)
        # 即 t_{i,•} = max_j t_{ij} 中最大的 i
        t_max_row = np.max(t_stat, axis=1)                 # (m,) 每行最大t
        worst_local = int(np.argmax(t_max_row))            # 行索引 (0..m-1)
        worst_global = surviving[worst_local]              # 全局模型索引

        # 记录此模型在当前集合的 p-value
        model_pvals[worst_global] = p_val
        eliminated_round[worst_global] = round_num

        log(f"    淘汰: {MODEL_NAMES[worst_global+1]} (loc={worst_local}, "
            f"t_max={t_max_row[worst_local]:.3f})")

        # 从存活集合移除
        surviving.pop(worst_local)

    # 最后一个幸存模型: p-value = 1.0 (总在 MCS 中)
    if surviving:
        last = surviving[0]
        model_pvals[last] = 1.0
        eliminated_round[last] = round_num + 1
        log(f"\n  最终幸存: {MODEL_NAMES[last+1]} (p=1.0)")

    # ---- 输出: 对每个 α 判断是否在 MCS 中 ----
    results = []
    for mid in range(M):
        pv = model_pvals[mid]
        results.append({
            'Model': mid + 1,
            'Name': MODEL_NAMES[mid + 1],
            'MCS_pval': round(pv, 6),
            'Eliminated_round': eliminated_round[mid],
        })
    df = pd.DataFrame(results)

    # MCS rank: 按 p-value 降序 (p越大=越优, rank=1最优)
    df['MCS_rank'] = df['MCS_pval'].rank(ascending=False, method='min').astype(int)

    # 为每个 alpha 添加列
    for alpha in ALPHA_LEVELS:
        col_name = f'in_MCS_{int((1-alpha)*100)}'
        df[col_name] = df['MCS_pval'] > alpha
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
        log(f"  M{int(row['Model'])} {row['Name']:<20} p={row['MCS_pval']:.4f}  "
            f"淘汰轮次={int(row['Eliminated_round'])}")

    # ---- 保存 ----
    out_path = Path(__file__).parent / "MCS_results.csv"
    results.to_csv(out_path, index=False)
    log(f"\n结果已保存: {out_path}")

    # ---- Table 2 风格输出 ----
    log(f"\n{'='*72}")
    log(f"Table 2 MCS 列 (MCS rank = p-value降序)")
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
