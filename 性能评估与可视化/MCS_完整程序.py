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
BLOCK_LEN   = 5          # block bootstrap 块长度 (交易日, 约1周) - 默认, Table 2 沿用
# 敏感性网格: 对齐 Hansen et al. (2011) 附录中 p6/p9/p12 多块长报告的做法
BLOCK_LENGTHS = [5, 10, 20]
SEED        = 42         # 随机种子 (可复现)
ALPHA_LEVELS = [0.25, 0.10]  # MCS 置信水平: 75% 和 90%

MODEL_NAMES = {
    1: 'VAR (OLS)',
    2: 'Sparse VAR',
    3: 'Sparse VARX',
    4: 'Network VARX',
    5: 'Network+Smooth',
    6: 'Network VARX + Smooth + DFL',
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

    # 与Table2/3/4统一: 200天评估窗口 [65:265] = 2018-12-28 ~ 2019-10-15
    TSTART, TEND = 65, 265
    Y_te = Y_te[TSTART:TEND]

    # 加载各模型预测
    L_list = []
    for mid in range(1, 8):
        Y_pred = np.load(pred_dir / f"Y_pred_model{mid}.npy")
        Y_pred = Y_pred[-n_test:][TSTART:TEND]  # 对齐测试集并截取200天窗口
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


def _round_statistics(L_sub: np.ndarray, boot_idx: np.ndarray, n_boot: int
                      ) -> Tuple[float, float, int, float, np.ndarray]:
    """单轮 EPA 检验 (等价预测能力), 返回 (T_R, p_hat, 最差局部索引, 其 t 值, 每行最大 t)。

    p_hat 做 +1 平滑: (1 + #{T_R* > T_R}) / (1 + B), 因此可报告的最小水平是
    1/(B+1) 而不是 0 —— 0 只是 bootstrap 分辨率的下限, 不是一个测得出来的数。
    """
    d_bar = L_sub.mean(axis=0)
    D_obs = d_bar[:, None] - d_bar[None, :]              # (m, m) 均值差分
    boot_mean = L_sub[boot_idx, :].mean(axis=1)          # (B, m) 重采样均值
    D_boot = boot_mean[:, :, None] - boot_mean[:, None, :]

    d_boot_std = np.sqrt(np.maximum(
        ((D_boot - D_obs) ** 2).mean(axis=0), 1e-20))
    t_stat = D_obs / d_boot_std
    t_boot = (D_boot - D_obs) / d_boot_std

    T_R_obs = float(np.max(np.abs(t_stat)))
    T_R_boot = np.abs(t_boot).max(axis=(1, 2))
    n_exceed = int(np.count_nonzero(T_R_boot > T_R_obs))
    p_hat = (1.0 + n_exceed) / (1.0 + n_boot)

    t_max_row = np.max(t_stat, axis=1)
    worst = int(np.argmax(t_max_row))
    return T_R_obs, float(p_hat), worst, float(t_max_row[worst]), t_max_row


def mcs_path(L: np.ndarray, n_boot: int = N_BOOTSTRAP,
             block_len: int = BLOCK_LEN, seed: int = SEED,
             verbose: bool = True) -> Tuple[list, int]:
    """走完 HLN 的嵌套淘汰路径, 返回 (每轮记录, 最终幸存模型)。

    淘汰顺序只由检验统计量决定, 与 α 无关 —— α 仅决定在第几轮停下。所以这条
    路径算一次就能读出任意置信水平下的 MCS, 不必为每个 α 重复 bootstrap。
    """
    T, M = L.shape
    rng = np.random.default_rng(seed)
    boot_idx = block_bootstrap(T, n_boot, block_len, rng)
    if verbose:
        log(f"MCS: T={T}天, M={M}模型, bootstrap={n_boot}次, "
            f"block_len={block_len}, seed={seed}")

    surviving = list(range(M))
    rounds = []
    while len(surviving) > 1:
        before = list(surviving)
        T_R, p_hat, worst, t_worst, _ = _round_statistics(L[:, before], boot_idx, n_boot)
        elim = int(surviving.pop(worst))
        rounds.append({
            'round': len(rounds) + 1,
            'models_before': before,
            'models_after': list(surviving),
            'T_R': T_R,
            'p_hat': p_hat,
            'eliminated': elim,
            't_worst': t_worst,
        })
        if verbose:
            log(f"\n  Round {len(rounds)}: {len(before)} models surviving "
                f"→ {[MODEL_NAMES[s + 1] for s in before]}")
            log(f"    T_R={T_R:.3f}  p_hat={p_hat:.6f}  "
                f"淘汰 {MODEL_NAMES[elim + 1]} (t={t_worst:.3f})")

    last = int(surviving[0])
    if verbose:
        log(f"\n  路径结束: 唯一幸存 {MODEL_NAMES[last + 1]}")
    return rounds, last


def mcs_member_sets(rounds: list, M: int, alphas=ALPHA_LEVELS) -> dict:
    """按 HLN 停止规则读集合: 首轮 p_hat >= α 时不再拒绝, 当时在场的模型即 MCS。

    Returns: {α: (成员索引集合, 停止轮次)}。若始终拒绝到只剩一个模型, 集合为该
    唯一幸存者, 停止轮次记为 len(rounds)+1。
    """
    out = {}
    for a in alphas:
        S = list(range(M))
        stop = len(rounds) + 1
        for r in rounds:
            if r['p_hat'] >= a:
                stop = r['round']
                break
            S = r['models_after']
        out[a] = (set(int(x) for x in S), stop)
    return out


def mcs_procedure(L: np.ndarray, n_boot: int = N_BOOTSTRAP,
                  block_len: int = BLOCK_LEN, seed: int = SEED,
                  alphas=ALPHA_LEVELS, verbose: bool = True) -> pd.DataFrame:
    """Hansen-Lunde-Nason (2011) MCS —— α 停止规则版。

    Args:
        L: (T, M) 损失矩阵, L[t,m] = 模型 m 在第 t 天的损失

    Returns:
        DataFrame: Model, Name, MCS_pval, MCS_raw_pval, Eliminated_round,
                   MCS_rank, in_MCS_{75,90}, stop_round_{75,90}
    """
    T, M = L.shape
    rounds, last = mcs_path(L, n_boot=n_boot, block_len=block_len,
                            seed=seed, verbose=verbose)
    member_sets = mcs_member_sets(rounds, M, alphas)

    model_raw_pvals = np.zeros(M)          # 淘汰那一轮的 p_hat
    eliminated_round = np.full(M, -1)
    for r in rounds:
        model_raw_pvals[r['eliminated']] = r['p_hat']
        eliminated_round[r['eliminated']] = r['round']
    model_raw_pvals[last] = 1.0
    eliminated_round[last] = len(rounds) + 1

    # step-wise 展示用 p 值: 后淘汰者不低于先淘汰者 (单调调整)
    model_pvals = model_raw_pvals.copy()
    order = [r['eliminated'] for r in rounds] + [last]
    running_max = 0.0
    for idx in order:
        running_max = max(running_max, model_raw_pvals[idx])
        model_pvals[idx] = running_max

    results = []
    for mid in range(M):
        results.append({
            'Model': mid + 1,
            'Name': MODEL_NAMES[mid + 1],
            'MCS_pval': round(float(model_pvals[mid]), 6),
            'MCS_raw_pval': float(model_raw_pvals[mid]),
            'Eliminated_round': int(eliminated_round[mid]),
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

    # 成员资格来自停止规则下的集合, 而不是 p_hat 与 α 的直接比较
    for alpha in alphas:
        members, stop = member_sets[alpha]
        pct = int((1 - alpha) * 100)
        df[f'in_MCS_{pct}'] = [mid in members for mid in range(M)]
        df[f'stop_round_{pct}'] = stop
        df[f'MCS_pval_{(1 - alpha):.0%}'] = df['MCS_pval'].apply(
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
