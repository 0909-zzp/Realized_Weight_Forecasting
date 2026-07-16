"""Table 4 — 消融分析: DFL基线 + 逐组件减法

设计:
  完整模型 = M5 (Network+Smooth) + DFL → 所有组件全开
  无外部   = M2 (Sparse VAR) + DFL      → 移除外生变量
  无网络   = M3a (Self-free VARX) + DFL → 移除网络惩罚差异化
  无平滑   = M4 + DFL = M6              → 移除换手率平滑
  无DFL    = M5                          → 移除决策聚焦调优

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
from 共享模块 import K, ETA, log as shared_log, set_log_file, load_day, compute_raw_cov, EPS_RIDGE

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

    # 训练日索引 (DFL 需要)
    feat_dir = Path(__file__).parents[1] / "特征工程"
    valid_indices = np.load(feat_dir / "valid_indices.npy")
    train_day_indices = valid_indices[:n_train]

    # 网络掩码
    net_mask, density = vp.build_network_mask(A_tr)

    log(f"数据: train={n_train}  test={n_test}")
    log(f"网络: density={density:.1%}")

    # ================================================================
    # 2. 生成 M3a 预测 (不在标准模型中，需单独拟合)
    # ================================================================
    log(f"\n--- 生成 M3a 预测 ---")
    old_cfg = dict(vp.MODELS[3])
    vp.MODELS[3]['self_free'] = True
    try:
        fitted_m3a = vp.fit_model(3, X_tr, Y_tr, None, n_jobs=4)
        Y_pred_m3a = vp.predict_model(X_te, fitted_m3a)
        m3a_mse = vp.compute_mse(Y_pred_m3a, Y_te)
        log(f"  M3a MSE: {m3a_mse:.4e}")
    finally:
        vp.MODELS[3].clear()
        vp.MODELS[3].update(old_cfg)

    # 加载已有预测
    Y_pred_m2 = np.load(Path(__file__).parents[1] / "VARX" / "Y_pred_model2.npy")
    Y_pred_m4 = np.load(Path(__file__).parents[1] / "VARX" / "Y_pred_model4.npy")
    Y_pred_m5 = np.load(Path(__file__).parents[1] / "VARX" / "Y_pred_model5.npy")

    Y_pred_m2 = Y_pred_m2[-n_test:]
    Y_pred_m4 = Y_pred_m4[-n_test:]
    Y_pred_m5 = Y_pred_m5[-n_test:]

    # ================================================================
    # 3. DFL 应用到各基模型
    # ================================================================
    rho_dfl = getattr(vp, 'RHO_DFL', 1e-3)
    log(f"\n--- DFL 后处理 (η={ETA}, ρ={rho_dfl}) ---")

    # M2 + DFL
    t0 = time.time()
    Y_pred_m2_dfl = vp.compute_model6(Y_pred_m2, Y_te, train_day_indices, eta=ETA)
    log(f"  M2+DFL  完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_m2_dfl, Y_te):.4e}")

    # M3a + DFL
    t0 = time.time()
    Y_pred_m3a_dfl = vp.compute_model6(Y_pred_m3a, Y_te, train_day_indices, eta=ETA)
    log(f"  M3a+DFL 完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_m3a_dfl, Y_te):.4e}")

    # M4 + DFL (已有的 M6)
    t0 = time.time()
    Y_pred_m4_dfl = vp.compute_model6(Y_pred_m4, Y_te, train_day_indices, eta=ETA)
    log(f"  M4+DFL  完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_m4_dfl, Y_te):.4e}")

    # M5 + DFL
    t0 = time.time()
    Y_pred_m5_dfl = vp.compute_model6(Y_pred_m5, Y_te, train_day_indices, eta=ETA)
    log(f"  M5+DFL  完成 ({time.time()-t0:.1f}s)  MSE={vp.compute_mse(Y_pred_m5_dfl, Y_te):.4e}")

    # ================================================================
    # 4. 投资组合评估 (复用 Table 3)
    # ================================================================
    log(f"\n--- 投资组合表现评估 ---")

    test_days = valid_indices[n_train + int(0.15 * len(X)):]

    models = {
        2:  {'name': 'M2 (无外生)',     'Y_pred': Y_pred_m2},
        3:  {'name': 'M3a (无网络)',     'Y_pred': Y_pred_m3a},
        4:  {'name': 'M4 (无平滑)',      'Y_pred': Y_pred_m4},
        5:  {'name': 'M5 (无DFL)',       'Y_pred': Y_pred_m5},
        6:  {'name': 'M5+DFL (完整)',     'Y_pred': Y_pred_m5_dfl},
        7:  {'name': 'M2+DFL',           'Y_pred': Y_pred_m2_dfl},
        8:  {'name': 'M3a+DFL',          'Y_pred': Y_pred_m3a_dfl},
        9:  {'name': 'M4+DFL',           'Y_pred': Y_pred_m4_dfl},
    }
    # 按需要的顺序排列
    eval_order = {
        '完整': 6,   # M5+DFL
        '无外部': 7,  # M2+DFL
        '无网络': 8,  # M3a+DFL
        '无平滑': 9,  # M4+DFL
        '无DFL': 5,   # M5
    }

    # 只评估需要的模型
    needed = set(eval_order.values())
    models_subset = {k: v for k, v in models.items() if k in needed}

    results = t3.compute_all(models_subset, test_days)

    # ================================================================
    # 5. 输出 Table 4
    # ================================================================
    # MSE
    mse_dict = {
        5: vp.compute_mse(Y_pred_m5, Y_te),
        6: vp.compute_mse(Y_pred_m5_dfl, Y_te),
        7: vp.compute_mse(Y_pred_m2_dfl, Y_te),
        8: vp.compute_mse(Y_pred_m3a_dfl, Y_te),
        9: vp.compute_mse(Y_pred_m4_dfl, Y_te),
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
