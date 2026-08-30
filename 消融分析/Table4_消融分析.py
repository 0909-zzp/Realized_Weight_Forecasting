"""Table 4 — 组件消融: 严格单因素减法

设计 (每次只移除一个组件):
  完整模型 = 网络VARX+平滑+DFL (M5 + drift-aware L1 DFL)
  无外部   = M5无外生+DFL → 移除exog特征块 (保留网络+平滑)
  无网络   = 稀疏VARX+平滑+DFL → 移除网络差异化惩罚 (保留平滑)
  无平滑   = M4+DFL      → 移除换手率平滑
  无DFL    = M5          → 移除决策聚焦调优

评估口径与Table2/3/MCS统一: 200天窗口 [65:265] = 2018-12-28 ~ 2019-10-15,
DFL 使用最终版本: drift-aware L1, η=1e-6, ρ=1e-3, 每日滚动协方差150天。

产出:
  Table4_完整.csv — MSE_w / RPV / Turnover / 净夏普
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ---- 导入 table2 的函数 ----
_varx_path = Path(__file__).parents[1] / "VARX" / "VAR及拓展（table2）.py"
_spec = importlib.util.spec_from_file_location("vp", _varx_path)
vp = importlib.util.module_from_spec(_spec)
sys.modules["vp"] = vp
_spec.loader.exec_module(vp)

# ---- 导入共享模块 ----
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log as shared_log, set_log_file, load_day, compute_raw_cov, EPS_RIDGE, LAMBDA_LASSO

# ---- 导入 Table3 的函数 ----
_t3_path = Path(__file__).parents[1] / "性能评估与可视化" / "Table3_投资组合表现.py"
_spec3 = importlib.util.spec_from_file_location("t3", _t3_path)
t3 = importlib.util.module_from_spec(_spec3)
sys.modules["t3"] = t3
_spec3.loader.exec_module(t3)

log = shared_log


def main():
    out_dir = Path(__file__).parent
    set_log_file(out_dir / "table4_full_log.txt")

    log("=" * 72)
    log("  Table 4 — DFL基线消融 (完整→减法)")
    log("=" * 72)

    # ================================================================
    # 1. 加载数据
    # ================================================================
    data = vp.load_data()
    X, Y, A_bar = data['X'], data['Y'], data['A_bar']
    splits = vp.split_data(X, Y, A_bar)
    X_tr, Y_tr, A_tr = splits['train']
    X_te, Y_te, A_te = splits['test']
    n_train, n_test = X_tr.shape[0], X_te.shape[0]

    # 200天评估窗口: 测试集索引 [65:265] (2018-12-28 ~ 2019-10-15), 与Table2/3/MCS统一
    TSTART, TEND = 65, 265

    # 训练日索引 (DFL 需要)
    feat_dir = Path(__file__).parents[1] / "特征工程"
    valid_indices = np.load(feat_dir / "valid_indices.npy")
    train_day_indices = valid_indices[:n_train]

    # 200天评估用Y_te切片
    Y_te_200 = Y_te[TSTART:TEND]

    # 网络掩码
    net_mask, density = vp.build_network_mask(A_tr)

    log(f"数据: train={n_train}  test={n_test}")
    log(f"网络: density={density:.1%}")

    # ================================================================
    # 2. 生成严格单因素消融基模型预测 (不在标准模型中, 需单独拟合)
    # ================================================================
    # 无外生: M5 去掉 exog 特征块 (保留网络+平滑)
    log(f"\n--- 生成 无外生 基模型 (M5 仅滞后块, λ₁={LAMBDA_LASSO:.0e}) ---")
    old_cfg5 = dict(vp.MODELS[5])
    vp.MODELS[5]['blocks'] = ['lagged']
    try:
        fitted_noexog = vp.fit_model(5, X_tr, Y_tr, net_mask, n_jobs=4)
        Y_pred_m5_noexog = vp.predict_model(X_te, fitted_noexog)
        m_noexog_mse = vp.compute_mse(Y_pred_m5_noexog[TSTART:TEND], Y_te_200)
        log(f"  无外生基模型 MSE: {m_noexog_mse:.4e}")
    finally:
        vp.MODELS[5].clear()
        vp.MODELS[5].update(old_cfg5)

    # 无网络: M3 + 平滑, 去掉网络差异化惩罚 (保留外生+平滑)
    log(f"\n--- 生成 无网络 基模型 (稀疏VARX+平滑, λ₁={LAMBDA_LASSO:.0e}) ---")
    old_cfg3 = dict(vp.MODELS[3])
    vp.MODELS[3]['self_free'] = True
    vp.MODELS[3]['smooth'] = True
    vp.MODELS[3]['network'] = False
    vp.MODELS[3]['lasso_lambda'] = LAMBDA_LASSO
    try:
        fitted_nonnet = vp.fit_model(3, X_tr, Y_tr, None, n_jobs=4)
        Y_pred_m3a_smooth = vp.predict_model(X_te, fitted_nonnet)
        m_nonnet_mse = vp.compute_mse(Y_pred_m3a_smooth[TSTART:TEND], Y_te_200)
        log(f"  无网络基模型 MSE: {m_nonnet_mse:.4e}")
    finally:
        vp.MODELS[3].clear()
        vp.MODELS[3].update(old_cfg3)

    # 加载已有预测
    Y_pred_m4 = np.load(Path(__file__).parents[1] / "VARX" / "Y_pred_model4.npy")
    Y_pred_m5 = np.load(Path(__file__).parents[1] / "VARX" / "Y_pred_model5.npy")

    # 取测试集 (全量363天, 用于DFL; 评估时截取200天窗口)
    Y_pred_m4 = Y_pred_m4[-n_test:]
    Y_pred_m5 = Y_pred_m5[-n_test:]

    # ================================================================
    # 3. DFL 应用到各基模型
    # ================================================================
    rho_dfl = getattr(vp, 'RHO_DFL', 1e-3)
    dfl_eta = 1e-6  # drift-aware L1 换手惩罚 (OOS验证期选择)
    log(f"\n--- DFL 后处理 (η={dfl_eta}, ρ={rho_dfl}) ---")

    # 无外生 + DFL
    t0 = time.time()
    Y_pred_noexog_dfl = vp.compute_model6_drift_l1(
        Y_pred_m5_noexog, Y_te, train_day_indices,
        eta=dfl_eta, rho=rho_dfl, risk_mult=1.0,
        rolling_cov=True, cov_window=150,
    )
    log(f"  无外生+DFL 完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_noexog_dfl[TSTART:TEND], Y_te_200):.4e}")

    # 无网络 + DFL
    t0 = time.time()
    Y_pred_nonnet_dfl = vp.compute_model6_drift_l1(
        Y_pred_m3a_smooth, Y_te, train_day_indices,
        eta=dfl_eta, rho=rho_dfl, risk_mult=1.0,
        rolling_cov=True, cov_window=150,
    )
    log(f"  无网络+DFL 完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_nonnet_dfl[TSTART:TEND], Y_te_200):.4e}")

    # M4 + DFL
    t0 = time.time()
    Y_pred_m4_dfl = vp.compute_model6_drift_l1(
        Y_pred_m4, Y_te, train_day_indices,
        eta=dfl_eta, rho=rho_dfl, risk_mult=1.0,
        rolling_cov=True, cov_window=150,
    )
    log(f"  M4+DFL  完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_m4_dfl[TSTART:TEND], Y_te_200):.4e}")

    # M5 + DFL
    t0 = time.time()
    Y_pred_m5_dfl = vp.compute_model6_drift_l1(
        Y_pred_m5, Y_te, train_day_indices,
        eta=dfl_eta, rho=rho_dfl, risk_mult=1.0,
        rolling_cov=True, cov_window=150,
    )
    log(f"  M5+DFL  完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_m5_dfl[TSTART:TEND], Y_te_200):.4e}")

    # ================================================================
    # 4. 投资组合评估 (复用 Table 3)
    # ================================================================
    log(f"\n--- 投资组合表现评估 ---")

    test_indices = valid_indices[n_train + int(0.15 * len(X)):]
    # 截取200天窗口
    test_indices = test_indices[TSTART:TEND]
    # 转换为文件路径以匹配新版 Table3 接口
    npy_dir = Path(__file__).parents[1] / "数据" / "1min_log_return_npy"
    all_files = sorted([str(f) for f in npy_dir.iterdir()
                        if f.suffix == '.npy' and f.name[0].isdigit()])
    test_files = [all_files[i] for i in test_indices]

    models = {
        2:  {'name': '网络VARX + 平滑（无外生）+ DFL', 'Y_pred': Y_pred_m5_noexog[TSTART:TEND]},
        3:  {'name': '稀疏VARX + 平滑（无网络）+ DFL', 'Y_pred': Y_pred_m3a_smooth[TSTART:TEND]},
        4:  {'name': '网络VARX + DFL',                 'Y_pred': Y_pred_m4[TSTART:TEND]},
        5:  {'name': '网络VARX + 平滑',                'Y_pred': Y_pred_m5[TSTART:TEND]},
        6:  {'name': '网络VARX + 平滑 + DFL',          'Y_pred': Y_pred_m5_dfl[TSTART:TEND]},
        7:  {'name': '网络VARX + 平滑（无外生）+ DFL', 'Y_pred': Y_pred_noexog_dfl[TSTART:TEND]},
        8:  {'name': '稀疏VARX + 平滑（无网络）+ DFL', 'Y_pred': Y_pred_nonnet_dfl[TSTART:TEND]},
        9:  {'name': '网络VARX + DFL',                 'Y_pred': Y_pred_m4_dfl[TSTART:TEND]},
    }
    # 按需要的顺序排列
    eval_order = {
        '网络VARX + 平滑 + DFL': 6,          # M5+DFL
        '网络VARX + 平滑（无外生）+ DFL': 7,  # M5无外生+DFL
        '稀疏VARX + 平滑（无网络）+ DFL': 8,  # 稀疏VARX+平滑+DFL
        '网络VARX + DFL': 9,                 # M4+DFL
        '网络VARX + 平滑': 5,                # M5
    }

    # 只评估需要的模型
    needed = set(eval_order.values())
    models_subset = {k: v for k, v in models.items() if k in needed}

    results = t3.compute_all(models_subset, test_files)

    # ================================================================
    # 5. 输出 Table 4
    # ================================================================
    # MSE
    mse_dict = {
        5: vp.compute_mse(Y_pred_m5[TSTART:TEND], Y_te_200),
        6: vp.compute_mse(Y_pred_m5_dfl[TSTART:TEND], Y_te_200),
        7: vp.compute_mse(Y_pred_noexog_dfl[TSTART:TEND], Y_te_200),
        8: vp.compute_mse(Y_pred_nonnet_dfl[TSTART:TEND], Y_te_200),
        9: vp.compute_mse(Y_pred_m4_dfl[TSTART:TEND], Y_te_200),
    }

    log(f"\n{'='*80}")
    log(f"Table 4 — 消融分析: DFL基线 + 减法")
    log(f"{'='*80}")
    header = f"{'设定':<16} {'MSE_w':>14} {'RPV(年)':>12} {'换手率':>10} {'净夏普':>10}"
    log(header)
    log("-" * 68)

    rows = []
    for label, mid in eval_order.items():
        r = results[mid]
        mse = mse_dict[mid]
        log(f"{label:<16} "
            f"{mse:>14.4e} "
            f"{r['rpv_annual']:>12.6f} "
            f"{r['avg_turnover']:>10.4f} "
            f"{r['sharpe_net']:>10.4f}")
        rows.append({
            '设定': label,
            'MSE_w': f"{mse:.4e}",
            'RPV_annual': round(r['rpv_annual'], 6),
            'Turnover': round(r['avg_turnover'], 4),
            'Sharpe_net': round(r['sharpe_net'], 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "Table4_完整.csv", index=False)
    log(f"\n保存: {out_dir / 'Table4_完整.csv'}")
    log("=" * 72)


if __name__ == "__main__":
    main()
