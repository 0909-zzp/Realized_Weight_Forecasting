"""VARX 参数调优 — rolling folds + 消融证据链。

设计原则：
  1. 无稀疏约束：各模型按 MSE 目标自由选择最优参数
     （论文中如实讨论模型自发选择的稀疏度与特征工程的关系）
  2. M2/M3/M4/M5 分层搜索，测试集不参与选择
  3. M3a = M3 + 自身滞后近似零惩罚，用于剥离自环豁免贡献
  4. M4 必须相对 M3a 提供网络惩罚净增量（消融证据链）
  5. M5 按经济目标选 λ_s：在 MSE 允许小幅退化下优先降低换手

输出：
  tuning_folds.csv          每折每组参数结果
  tuning_summary.csv        聚合后的搜索结果
  ablation_self_lag.csv     M3 vs M3a 消融
  tau_robustness.csv        M4 τ 稳健性
  final_params.json         推荐参数
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))

import 共享模块
from 共享模块 import K, log, set_log_file
import VAR及拓展 as vp
from VAR及拓展 import compute_mse, compute_turnover


# ----------------------------- 搜索空间 -----------------------------
L1_M2_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
L1_M3_GRID = [5e-5, 1e-4, 3e-4, 5e-4]
L3_GRID = [5e-5, 1e-4, 5e-4, 1e-3]
TAU_GRID = [0.7, 0.8, 0.85, 0.9]
LNET_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LS_GRID = [0, 1e-4, 1e-3, 5e-3, 1e-2]
WINDOW_GRID = [False, True]       # False=全训练期均值, True=仅最后一天快照

M5_MSE_TOL = 0.10  # M5 在 MSE 容忍范围内优先降换手


def make_folds(n: int, test_ratio: float = 0.15) -> List[Dict[str, int]]:
    """构建 3 个 expanding-window 验证折，最后 15% 留作最终测试集。"""
    test_len = int(n * test_ratio)
    tuning_end = n - test_len
    val_len = min(240, max(160, tuning_end // 8))
    train_ends = [1200, 1450, 1690]
    folds = []
    for i, train_end in enumerate(train_ends, start=1):
        if train_end + val_len > tuning_end:
            continue
        folds.append({
            "fold": i,
            "train_start": 0,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": train_end + val_len,
        })
    if not folds:
        raise ValueError(f"样本量不足，无法构建 rolling folds: n={n}, tuning_end={tuning_end}")
    return folds


def slice_fold(X, Y, A_bar, fold: Dict[str, int]):
    tr = slice(fold["train_start"], fold["train_end"])
    va = slice(fold["val_start"], fold["val_end"])
    return X[tr], Y[tr], A_bar[tr], X[va], Y[va], A_bar[va]


def _set_global(name: str, value, backup: Dict):
    if name not in backup:
        backup[name] = getattr(共享模块, name)
    setattr(共享模块, name, value)
    setattr(vp, name, value)


def run_one(model_id: int, params: Dict, X_tr, Y_tr, A_tr, X_val, Y_val) -> Dict:
    """临时覆盖全局参数和模型配置，运行一次验证集评估。"""
    backup_globals = {}
    cfg = vp.MODELS[model_id]
    backup_cfg = cfg.copy()
    old_lam_s = vp.LAMBDA_S

    try:
        for key in ("LAMBDA_LASSO", "LAMBDA_EXOG", "LAMBDA_NETWORK",
                    "NETWORK_THRESHOLD", "LAMBDA_TURNOVER"):
            if key in params:
                _set_global(key, params[key], backup_globals)

        if "lasso_lambda" in params:
            cfg["lasso_lambda"] = params["lasso_lambda"]
        elif "LAMBDA_LASSO" in params and cfg["sparse"]:
            cfg["lasso_lambda"] = params["LAMBDA_LASSO"]
        if "self_free" in params:
            cfg["self_free"] = params["self_free"]
        if "LAMBDA_TURNOVER" in params:
            vp.LAMBDA_S = params["LAMBDA_TURNOVER"]

        if cfg["network"]:
            use_last_day = bool(params.get("USE_LAST_DAY", False))
            net_mask, density = vp.build_network_mask(
                A_tr, threshold=params.get("NETWORK_THRESHOLD", 共享模块.NETWORK_THRESHOLD),
                use_last_day=use_last_day)
        else:
            net_mask, density = None, np.nan

        t0 = time.time()
        fitted = vp.fit_model(model_id, X_tr, Y_tr, net_mask, n_jobs=4)
        y_pred = vp.predict_model(X_val, fitted)
        elapsed = time.time() - t0

        # 经济换手率: |w_pred_t - w_actual_{t-1}|，传入 Y_val 上一期
        y_actual_prev = Y_val if Y_val is not None else None
        # 精确计算: t=0 用 Y_tr 最后一天, t=1.. 用 Y_val[0..n-2]
        n_val = len(Y_val)
        y_prev_econ = np.vstack([Y_tr[-1:], Y_val[:n_val - 1]]) if n_val > 1 else Y_tr[-1:]

        return {
            "MSE": compute_mse(y_pred, Y_val),
            "Turnover_econ": compute_turnover(y_pred, y_prev_econ),
            "Turnover_seq": compute_turnover(y_pred),
            "Sparsity": float(np.mean(np.abs(fitted["coefs"]) < 1e-8)) if cfg["sparse"] else 0.0,
            "n_alive": int(np.sum(np.abs(fitted["coefs"]) >= 1e-8)) if cfg["sparse"] else 0,
            "Density": float(density) if not np.isnan(density) else np.nan,
            "FitTime_s": elapsed,
        }
    finally:
        for key, value in backup_globals.items():
            setattr(共享模块, key, value)
            setattr(vp, key, value)
        vp.MODELS[model_id].clear()
        vp.MODELS[model_id].update(backup_cfg)
        vp.LAMBDA_S = old_lam_s


def run_across_folds(model_label: str, model_id: int, params: Dict,
                     X, Y, A_bar, folds: List[Dict[str, int]]) -> List[Dict]:
    rows = []
    for fold in folds:
        X_tr, Y_tr, A_tr, X_val, Y_val, _ = slice_fold(X, Y, A_bar, fold)
        result = run_one(model_id, params, X_tr, Y_tr, A_tr, X_val, Y_val)
        row = {
            "model": model_label,
            "model_id": model_id,
            "fold": fold["fold"],
            **params,
            **result,
        }
        rows.append(row)
        log(f"    fold={fold['fold']} MSE={result['MSE']:.4e} "
            f"TO={result['Turnover_econ']:.4f} alive={result.get('n_alive', 'N/A')} "
            f"sparse={result['Sparsity']:.1%} "
            f"density={result['Density'] if not np.isnan(result['Density']) else np.nan:.3f}")
    return rows


def summarize(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    param_cols = [
        c for c in ["model", "model_id", "LAMBDA_LASSO", "LAMBDA_EXOG",
                    "LAMBDA_NETWORK", "NETWORK_THRESHOLD", "LAMBDA_TURNOVER",
                    "lasso_lambda", "self_free", "USE_LAST_DAY"]
        if c in df.columns
    ]
    agg = df.groupby(param_cols, dropna=False).agg(
        mean_val_mse=("MSE", "mean"),
        std_val_mse=("MSE", "std"),
        mean_turnover_econ=("Turnover_econ", "mean"),
        mean_turnover_seq=("Turnover_seq", "mean"),
        mean_sparsity=("Sparsity", "mean"),
        mean_n_alive=("n_alive", "mean"),
        mean_density=("Density", "mean"),
        mean_fit_time_s=("FitTime_s", "mean"),
    ).reset_index()
    return agg


def pick_best(df: pd.DataFrame, sparsity_cap: float = None) -> pd.Series:
    """按 MSE（次优换手率）排序选最优，可选稀疏度上限过滤。

    当前搜索无稀疏约束（sparsity_cap=None），如需后验过滤可传入。
    """
    pool = df
    if sparsity_cap is not None:
        viable = df[df["mean_sparsity"] <= sparsity_cap]
        if len(viable):
            pool = viable
    return pool.sort_values(["mean_val_mse", "mean_turnover_econ"]).iloc[0]


def main():
    out_dir = Path(__file__).parent
    set_log_file(out_dir / "grid_search_log.txt")

    log("=" * 72)
    log("VARX 参数调优 — rolling folds + M3a/M4/M5 消融")
    log("=" * 72)

    data = vp.load_data()
    X, Y, A_bar = data["X"], data["Y"], data["A_bar"]
    folds = make_folds(len(X))
    log(f"数据: X={X.shape} Y={Y.shape} A_bar={A_bar.shape}")
    for f in folds:
        log(f"fold {f['fold']}: train=[{f['train_start']},{f['train_end']}) "
            f"val=[{f['val_start']},{f['val_end']})")

    all_rows = []

    # ----------------------------- M2 -----------------------------
    log("\n" + "=" * 72)
    log("M2 Sparse VAR: 搜索 λ1_M2")
    m2_rows = []
    for lam in L1_M2_GRID:
        params = {"LAMBDA_LASSO": lam, "lasso_lambda": lam, "LAMBDA_NETWORK": 0.0}
        log(f"  λ1_M2={lam:.0e}")
        m2_rows += run_across_folds("M2", 2, params, X, Y, A_bar, folds)
    all_rows += m2_rows
    m2_summary = summarize(m2_rows)
    m2_best = pick_best(m2_summary)  # 无稀疏约束，模型按 MSE 自由选择
    log(f"  最优 M2: λ1={m2_best['LAMBDA_LASSO']:.0e}, "
        f"MSE={m2_best['mean_val_mse']:.4e}, sparse={m2_best['mean_sparsity']:.1%}")

    # ----------------------------- M3 -----------------------------
    log("\n" + "=" * 72)
    log("M3 Sparse VARX: 搜索 λ1 × λ3")
    m3_rows = []
    for lam, exo in product(L1_M3_GRID, L3_GRID):
        params = {"LAMBDA_LASSO": lam, "lasso_lambda": lam,
                  "LAMBDA_EXOG": exo, "LAMBDA_NETWORK": 0.0, "self_free": False}
        log(f"  λ1={lam:.0e} λ3={exo:.0e}")
        m3_rows += run_across_folds("M3", 3, params, X, Y, A_bar, folds)
    all_rows += m3_rows
    m3_summary = summarize(m3_rows)
    m3_best = pick_best(m3_summary)  # 无稀疏约束，模型按 MSE 自由选择
    log(f"  最优 M3: λ1={m3_best['LAMBDA_LASSO']:.0e}, λ3={m3_best['LAMBDA_EXOG']:.0e}, "
        f"MSE={m3_best['mean_val_mse']:.4e}, sparse={m3_best['mean_sparsity']:.1%}")

    # ----------------------------- M3a -----------------------------
    log("\n" + "=" * 72)
    log("M3a 消融: Sparse VARX + self-lag unpenalized")
    base_l1 = float(m3_best["LAMBDA_LASSO"])
    base_l3 = float(m3_best["LAMBDA_EXOG"])
    m3a_params = {
        "LAMBDA_LASSO": base_l1, "lasso_lambda": base_l1,
        "LAMBDA_EXOG": base_l3, "LAMBDA_NETWORK": 0.0, "self_free": True,
    }
    m3a_rows = run_across_folds("M3a", 3, m3a_params, X, Y, A_bar, folds)
    all_rows += m3a_rows
    m3a_best = summarize(m3a_rows).iloc[0]
    log(f"  M3 -> M3a: {m3_best['mean_val_mse']:.4e} -> {m3a_best['mean_val_mse']:.4e}")

    # ----------------------------- M4 -----------------------------
    log("\n" + "=" * 72)
    log("M4 Network VARX: 搜索 λ1 × τ × λ_net × last_day，并要求优于 M3a")
    log(f"  M3 最优 λ₁={base_l1:.0e}, 稀疏度={float(m3_best['mean_sparsity']):.1%}")
    log(f"  last_day: {WINDOW_GRID} (False=全训练期均值, True=仅最后一天)")
    L1_M4_GRID = [1e-4, 3e-4, 5e-4]   # M4 允许用不同于 M3 的 λ₁
    m4_rows = []
    for l1_m4, tau, lnet, last in product(L1_M4_GRID, TAU_GRID, LNET_GRID, WINDOW_GRID):
        params = {
            "LAMBDA_LASSO": l1_m4, "lasso_lambda": l1_m4,
            "LAMBDA_EXOG": base_l3, "LAMBDA_NETWORK": lnet,
            "NETWORK_THRESHOLD": tau, "USE_LAST_DAY": last,
            "self_free": True,
        }
        log(f"  λ1={l1_m4:.0e} τ={tau} λ_net={lnet:.0e} last_day={last}")
        m4_rows += run_across_folds("M4", 4, params, X, Y, A_bar, folds)
    all_rows += m4_rows
    m4_summary = summarize(m4_rows)
    # 消融条件: M4 需优于 M3a 且 λ_net > 0（网络惩罚有实质贡献）
    viable_m4 = m4_summary[
        (m4_summary["mean_val_mse"] < float(m3a_best["mean_val_mse"])) &
        (m4_summary["LAMBDA_NETWORK"] > 0)
    ]
    if len(viable_m4):
        m4_best = viable_m4.sort_values(["mean_val_mse", "mean_turnover_econ"]).iloc[0]
        m4_reason = "M4 beats M3a on mean validation MSE"
    else:
        m4_best = m4_summary.sort_values(["mean_val_mse", "mean_turnover_econ"]).iloc[0]
        m4_reason = "WARNING: no M4 candidate beats M3a; picked lowest MSE"
    log(f"  最优 M4: λ1={m4_best.get('LAMBDA_LASSO', m4_best.get('lasso_lambda', base_l1)):.0e}, "
        f"τ={m4_best['NETWORK_THRESHOLD']}, λ_net={m4_best['LAMBDA_NETWORK']:.0e}, "
        f"last_day={m4_best.get('USE_LAST_DAY', False)}, "
        f"MSE={m4_best['mean_val_mse']:.4e}, sparse={m4_best['mean_sparsity']:.1%}, "
        f"alive={m4_best['mean_n_alive']:.0f}, density={m4_best['mean_density']:.1%}")
    log(f"  选择原因: {m4_reason}")

    # ----------------------------- M5 -----------------------------
    log("\n" + "=" * 72)
    log("M5 Network+Smooth: 搜索 λ_s，按 MSE 容忍下的低换手选择")
    m5_rows = []
    # M5 继承 M4 的全部参数（含 λ₁ 和 USE_LAST_DAY）
    m4_l1 = float(m4_best.get("LAMBDA_LASSO", m4_best.get("lasso_lambda", base_l1)))
    m4_last = bool(m4_best.get("USE_LAST_DAY", False))
    for ls in LS_GRID:
        params = {
            "LAMBDA_LASSO": m4_l1, "lasso_lambda": m4_l1,
            "LAMBDA_EXOG": base_l3,
            "LAMBDA_NETWORK": float(m4_best["LAMBDA_NETWORK"]),
            "NETWORK_THRESHOLD": float(m4_best["NETWORK_THRESHOLD"]),
            "USE_LAST_DAY": m4_last,
            "LAMBDA_TURNOVER": ls,
            "self_free": True,
        }
        log(f"  λ_s={ls:.0e}")
        m5_rows += run_across_folds("M5", 5, params, X, Y, A_bar, folds)
    all_rows += m5_rows
    m5_summary = summarize(m5_rows)
    mse_cap = (1 + M5_MSE_TOL) * float(m4_best["mean_val_mse"])
    viable_m5 = m5_summary[
        (m5_summary["mean_val_mse"] <= mse_cap) &
        (m5_summary["mean_turnover_econ"] <= float(m4_best["mean_turnover_econ"]))
    ]  # M5 经济目标：MSE容忍内选最低换手
    if len(viable_m5):
        m5_best = viable_m5.sort_values(["mean_turnover_econ", "mean_val_mse"]).iloc[0]
        m5_reason = f"MSE <= M4 + {M5_MSE_TOL:.0%}, turnover <= M4; picked lowest turnover"
    else:
        m5_best = m5_summary.sort_values(["mean_val_mse", "mean_turnover_econ"]).iloc[0]
        m5_reason = "WARNING: no smooth candidate met turnover/MSE constraints; picked lowest MSE"
    log(f"  最优 M5: λ_s={m5_best['LAMBDA_TURNOVER']:.0e}, "
        f"MSE={m5_best['mean_val_mse']:.4e}, turnover={m5_best['mean_turnover_econ']:.4f}")
    log(f"  选择原因: {m5_reason}")

    # ----------------------------- 输出 -----------------------------
    all_df = pd.DataFrame(all_rows)
    summary_df = summarize(all_rows)
    all_df.to_csv(out_dir / "tuning_folds.csv", index=False)
    summary_df.to_csv(out_dir / "tuning_summary.csv", index=False)

    ablation = pd.DataFrame([
        {"model": "M3", "mean_val_mse": m3_best["mean_val_mse"],
         "mean_turnover_econ": m3_best["mean_turnover_econ"],
         "mean_sparsity": m3_best["mean_sparsity"],
         "mean_n_alive": m3_best["mean_n_alive"]},
        {"model": "M3a", "mean_val_mse": m3a_best["mean_val_mse"],
         "mean_turnover_econ": m3a_best["mean_turnover_econ"],
         "mean_sparsity": m3a_best["mean_sparsity"],
         "mean_n_alive": m3a_best["mean_n_alive"]},
        {"model": "M4", "mean_val_mse": m4_best["mean_val_mse"],
         "mean_turnover_econ": m4_best["mean_turnover_econ"],
         "mean_sparsity": m4_best["mean_sparsity"],
         "mean_n_alive": m4_best["mean_n_alive"]},
    ])
    ablation.to_csv(out_dir / "ablation_self_lag.csv", index=False)

    tau_robustness = m4_summary.sort_values(["NETWORK_THRESHOLD", "LAMBDA_NETWORK"])
    tau_robustness.to_csv(out_dir / "tau_robustness.csv", index=False)

    final_params = {
        "P_LAGS": int(共享模块.P_LAGS),
        "A_BAR_ROLLING_WINDOW": int(getattr(共享模块, "Ā_ROLLING_WINDOW")),
        "LAMBDA_LASSO_M2": float(m2_best["LAMBDA_LASSO"]),
        "LAMBDA_LASSO_M3": base_l1,
        "LAMBDA_LASSO_M4": m4_l1,
        "LAMBDA_EXOG": base_l3,
        "LAMBDA_NETWORK": float(m4_best["LAMBDA_NETWORK"]),
        "NETWORK_THRESHOLD": float(m4_best["NETWORK_THRESHOLD"]),
        "USE_LAST_DAY": m4_last,
        "LAMBDA_TURNOVER": float(m5_best["LAMBDA_TURNOVER"]),
        "m4_selected_reason": m4_reason,
        "m5_selected_reason": m5_reason,
        "mean_val_mse": {
            "M2": float(m2_best["mean_val_mse"]),
            "M3": float(m3_best["mean_val_mse"]),
            "M3a": float(m3a_best["mean_val_mse"]),
            "M4": float(m4_best["mean_val_mse"]),
            "M5": float(m5_best["mean_val_mse"]),
        },
        "mean_turnover_econ": {
            "M4": float(m4_best["mean_turnover_econ"]),
            "M5": float(m5_best["mean_turnover_econ"]),
        },
        "mean_sparsity": {
            "M2": float(m2_best["mean_sparsity"]),
            "M3": float(m3_best["mean_sparsity"]),
            "M4": float(m4_best["mean_sparsity"]),
        },
        "mean_n_alive": {
            "M3": float(m3_best["mean_n_alive"]),
            "M4": float(m4_best["mean_n_alive"]),
        },
        "mean_network_density": float(m4_best["mean_density"]),
    }
    with open(out_dir / "final_params.json", "w", encoding="utf-8") as f:
        json.dump(final_params, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 72)
    log("最终推荐参数")
    log("=" * 72)
    for key, value in final_params.items():
        if isinstance(value, dict):
            log(f"{key}: {value}")
        elif isinstance(value, float):
            log(f"{key}: {value:.6g}")
        else:
            log(f"{key}: {value}")
    log("\n输出: tuning_folds.csv, tuning_summary.csv, ablation_self_lag.csv, "
        "tau_robustness.csv, final_params.json")


if __name__ == "__main__":
    main()
