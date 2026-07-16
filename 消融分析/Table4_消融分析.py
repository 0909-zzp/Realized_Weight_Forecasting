"""Table 4: 消融分析 — 拆解 Network VARX (M4) 各组件边际贡献

消融链路 (论文 §5.3):
  M2 (Sparse VAR)
    → +外生变量 → M3 (Sparse VARX)
    → +自环豁免 → M3a (Sparse VARX + self_free)
    → +网络惩罚 → M4 (Network VARX)

每步量化对应组件的边际 MSE 改善, 配合 pairwise DM 检验验证统计显著性。

输出:
  Table4_results.csv    测试集指标 + 边际贡献
  table4_log.txt        运行日志
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from copy import deepcopy

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# 导入 Table 2 的共享函数和模型定义
# 文件名含中文+括号，无法直接 import，用 importlib 加载
import importlib.util

_varx_path = Path(__file__).parent / "VAR及拓展（table2）.py"
_spec = importlib.util.spec_from_file_location("VAR及拓展", _varx_path)
vp = importlib.util.module_from_spec(_spec)
sys.modules["VAR及拓展"] = vp
_spec.loader.exec_module(vp)

# 从 vp 模块取出所有需要的符号
load_data           = vp.load_data
split_data          = vp.split_data
build_network_mask  = vp.build_network_mask
fit_model           = vp.fit_model
predict_model       = vp.predict_model
compute_mse         = vp.compute_mse
compute_mae         = vp.compute_mae
dm_test             = vp.dm_test
compute_turnover    = vp.compute_turnover
MODELS              = vp.MODELS
FEAT_BLOCKS         = vp.FEAT_BLOCKS
K                   = vp.K
P_LAGS              = vp.P_LAGS

# 共享模块在标准路径下，可以直接 import
sys.path.insert(0, str(Path(__file__).parents[1] / "图形Lasso" / "code"))
from 共享模块 import (
    LAMBDA_LASSO, LAMBDA_LASSO_M2, LAMBDA_NETWORK,
    LAMBDA_EXOG, NETWORK_THRESHOLD,
    log, set_log_file,
)


def run_ablation():
    """主流程: 拟合 M2/M3/M3a/M4 → 测试集评估 → 边际拆解。"""
    out_dir = Path(__file__).parent
    set_log_file(out_dir / "table4_log.txt")

    log("=" * 72)
    log("  Table 4: 消融分析 — Network VARX 组件拆解")
    log("=" * 72)
    log(f"  消融链: M2 → M3 → M3a → M4")
    log(f"  K={K}  P_LAGS={P_LAGS}")
    log(f"  λ₁(M2)={LAMBDA_LASSO_M2:.0e}  λ₁(M3/M3a/M4)={LAMBDA_LASSO:.0e}")
    log(f"  λ₃={LAMBDA_EXOG:.0e}  λ_net={LAMBDA_NETWORK:.0e}  τ={NETWORK_THRESHOLD}")

    # ================================================================
    # 1. 加载数据 & 划分
    # ================================================================
    data = load_data()
    X, Y, A_bar = data['X'], data['Y'], data['A_bar']
    log(f"\n数据: X={X.shape}  Y={Y.shape}  A_bar={A_bar.shape}")

    splits = split_data(X, Y, A_bar)
    X_tr, Y_tr, A_tr = splits['train']
    X_te, Y_te, A_te = splits['test']
    n_train = X_tr.shape[0]
    n_test  = X_te.shape[0]
    log(f"划分: 训练={n_train}  测试={n_test}")

    # ================================================================
    # 2. 网络掩码 (M4 需要)
    # ================================================================
    net_mask, density = build_network_mask(A_tr, threshold=NETWORK_THRESHOLD)
    avg_degree = net_mask.sum(axis=1).mean()
    log(f"网络掩码: 密度={density:.1%}  平均度={avg_degree:.0f}  (τ={NETWORK_THRESHOLD})")

    # ================================================================
    # 3. 拟合 M1 基准 (用于 DM 检验)
    # ================================================================
    log(f"\n--- M1 (VAR, 基准) ---")
    t0 = time.time()
    fitted_m1 = fit_model(1, X_tr, Y_tr, None, n_jobs=4)
    Y_pred_m1 = predict_model(X_te, fitted_m1)
    m1_mse = compute_mse(Y_pred_m1, Y_te)
    log(f"  M1 基准 MSE: {m1_mse:.4e}  (耗时 {time.time()-t0:.1f}s)")

    # ================================================================
    # 4. 拟合消融链各模型
    # ================================================================
    models_info = {
        'M2':  {'id': 2, 'desc': 'Sparse VAR'},
        'M3':  {'id': 3, 'desc': 'Sparse VARX'},
        'M3a': {'id': 3, 'desc': '+ Self-lag exempt', 'override': {'self_free': True}},
        'M4':  {'id': 4, 'desc': 'Network VARX'},
    }

    predictions = {}
    results = {}

    for label, info in models_info.items():
        model_id = info['id']
        desc = info['desc']

        log(f"\n--- {label}: {desc} ---")

        # M3a: 临时修改 MODELS[3] 的 self_free
        restore_cfg = None
        if 'override' in info:
            restore_cfg = dict(MODELS[model_id])  # 备份
            for k, v in info['override'].items():
                MODELS[model_id][k] = v
            log(f"  临时覆写: {info['override']}")

        try:
            t0 = time.time()
            fitted = fit_model(model_id, X_tr, Y_tr, net_mask, n_jobs=4)
            t_fit = time.time() - t0

            Y_pred = predict_model(X_te, fitted)
            predictions[label] = Y_pred

            mse  = compute_mse(Y_pred, Y_te)
            mae  = compute_mae(Y_pred, Y_te)
            sparsity = float(np.mean(np.abs(fitted['coefs']) < 1e-8))
            turnover = compute_turnover(Y_pred)

            dm, pval, log10p = dm_test(Y_pred, Y_pred_m1, Y_te)

            results[label] = {
                'Model': label,
                'Description': desc,
                'MSE': mse,
                'MAE': mae,
                'Sparsity': sparsity,
                'Turnover': turnover,
                'DM_vs_M1': dm,
                'DM_pval': pval,
                'DM_log10p': log10p,
                'FitTime_s': round(t_fit, 1),
            }

            log(f"  拟合: {t_fit:.1f}s")
            log(f"  MSE: {mse:.6e}  MAE: {mae:.6e}")
            log(f"  稀疏度: {sparsity:.1%}  换手率: {turnover:.6f}")
            if pval == 0.0 and not np.isnan(log10p):
                log(f"  DM vs M1: {dm:+.3f}  p=10^{{{log10p:.0f}}}")
            else:
                log(f"  DM vs M1: {dm:+.3f}  p={pval:.4e}")

        finally:
            if restore_cfg is not None:
                MODELS[model_id].clear()
                MODELS[model_id].update(restore_cfg)

    # ================================================================
    # 5. Pairwise DM 检验 (相邻模型间)
    # ================================================================
    log(f"\n--- Pairwise DM 检验 (相邻模型间) ---")
    ablation_steps = [
        ('M2',  'M3',  '+ Exogenous variables (外生变量)'),
        ('M3',  'M3a', '+ Self-lag exemption (自环豁免)'),
        ('M3a', 'M4',  '+ Network penalty (网络惩罚)'),
    ]

    pairwise_results = []
    for m_a, m_b, interpretation in ablation_steps:
        Y_a = predictions[m_a]
        Y_b = predictions[m_b]
        dm, pval, log10p = dm_test(Y_b, Y_a, Y_te)

        mse_a = results[m_a]['MSE']
        mse_b = results[m_b]['MSE']
        delta_mse = mse_b - mse_a  # 负 = 改善
        rel_improve = (mse_a - mse_b) / mse_a * 100  # 正 = 改善%

        pairwise_results.append({
            'Step': f'{m_a}→{m_b}',
            'Component': interpretation,
            'MSE_before': mse_a,
            'MSE_after': mse_b,
            'ΔMSE': delta_mse,
            'RelImprove%': rel_improve,
            'DM_pairwise': dm,
            'DM_pval': pval,
            'DM_log10p': log10p,
        })

        if pval == 0.0 and not np.isnan(log10p):
            p_str = f"10^{{{log10p:.0f}}}"
        else:
            p_str = f"{pval:.4e}"
        sig = '✓' if delta_mse < 0 else '✗'
        log(f"  {m_a}→{m_b}: ΔMSE={delta_mse:+.4e} ({rel_improve:+.2f}%)  "
            f"DM={dm:+.3f}  p={p_str}  [{sig}] {interpretation}")

    # 总体改善
    mse_m2 = results['M2']['MSE']
    mse_m4 = results['M4']['MSE']
    total_delta = mse_m4 - mse_m2
    total_rel = (mse_m2 - mse_m4) / mse_m2 * 100
    dm_total, pval_t, log10p_t = dm_test(predictions['M4'], predictions['M2'], Y_te)

    # ================================================================
    # 6. 输出 Panel A: 各模型单项指标
    # ================================================================
    log(f"\n{'='*80}")
    log(f"Panel A: 测试集预测精度 (各模型单项)")
    log(f"{'='*80}")
    header_a = f"{'Model':<6} {'Description':<24} {'MSE':>12} {'MAE':>12} {'DM vs M1':>10} {'p-val':>14} {'Sparsity':>10} {'Turnover':>10}"
    log(header_a)
    log("-" * 80)
    for label in ['M2', 'M3', 'M3a', 'M4']:
        r = results[label]
        def fmt(v, f, na='---'):
            return f.format(v) if not np.isnan(v) else na
        if r['DM_pval'] == 0.0 and not np.isnan(r['DM_log10p']):
            p_str = "10^{" + f"{r['DM_log10p']:.0f}" + "}"
        else:
            p_str = fmt(r['DM_pval'], '{:.4e}')
        log(
            f"{r['Model']:<6} {r['Description']:<24} "
            f"{fmt(r['MSE'], '{:.4e}'):>12} {fmt(r['MAE'], '{:.4e}'):>12} "
            f"{fmt(r['DM_vs_M1'], '{:+.3f}'):>10} {p_str:>14} "
            f"{fmt(r['Sparsity'], '{:.1%}'):>10} {fmt(r['Turnover'], '{:.6f}'):>10}"
        )

    # ================================================================
    # 7. 输出 Panel B: 边际贡献拆解
    # ================================================================
    log(f"\n{'='*80}")
    log(f"Panel B: 边际贡献拆解 (组件消融)")
    log(f"{'='*80}")
    header_b = f"{'Step':<10} {'Component':<32} {'MSE_before':>12} {'MSE_after':>12} {'ΔMSE':>12} {'Rel%':>8} {'DM':>8} {'p-val':>14}"
    log(header_b)
    log("-" * 80)
    for pr in pairwise_results:
        if pr['DM_pval'] == 0.0 and not np.isnan(pr['DM_log10p']):
            p_str = "10^{" + f"{pr['DM_log10p']:.0f}" + "}"
        else:
            p_str = f"{pr['DM_pval']:.4e}" if not np.isnan(pr['DM_pval']) else '---'
        log(
            f"{pr['Step']:<10} {pr['Component']:<32} "
            f"{pr['MSE_before']:>12.4e} {pr['MSE_after']:>12.4e} "
            f"{pr['ΔMSE']:>+12.4e} {pr['RelImprove%']:>+7.2f}% "
            f"{pr['DM_pairwise']:>+8.3f} {p_str:>14}"
        )
    # 合计行
    if pval_t == 0.0 and not np.isnan(log10p_t):
        p_str_t = "10^{" + f"{log10p_t:.0f}" + "}"
    else:
        p_str_t = f"{pval_t:.4e}"
    log("-" * 80)
    log(
        f"{'M2→M4':<10} {'TOTAL (all components)':<32} "
        f"{mse_m2:>12.4e} {mse_m4:>12.4e} "
        f"{total_delta:>+12.4e} {total_rel:>+7.2f}% "
        f"{dm_total:>+8.3f} {p_str_t:>14}"
    )

    # ================================================================
    # 8. 保存结果
    # ================================================================
    # Panel A
    df_a = pd.DataFrame([results[l] for l in ['M2', 'M3', 'M3a', 'M4']])
    # Panel B
    df_b = pd.DataFrame(pairwise_results)
    # 合并保存
    df_all = pd.concat([
        pd.DataFrame([{'Table': 'Panel A: Individual Metrics'}]), df_a,
        pd.DataFrame([{}]),
        pd.DataFrame([{'Table': 'Panel B: Marginal Contributions'}]), df_b,
    ], ignore_index=True)
    df_all.to_csv(out_dir / "Table4_results.csv", index=False)
    log(f"\n结果已保存: {out_dir / 'Table4_results.csv'}")

    # ================================================================
    # 9. 验收检查
    # ================================================================
    log(f"\n--- 验收检查 ---")
    checks = [
        ("M2→M3: 外生变量改善 MSE", pairwise_results[0]['ΔMSE'] < 0),
        ("M3→M3a: 自环豁免改善 MSE", pairwise_results[1]['ΔMSE'] < 0),
        ("M3a→M4: 网络惩罚改善 MSE", pairwise_results[2]['ΔMSE'] < 0),
        ("M4 优于 M2 (总体改善)", mse_m4 < mse_m2),
    ]
    all_ok = True
    for desc, ok in checks:
        log(f"  [{'✅' if ok else '❌'}] {desc}")
        if not ok:
            all_ok = False

    if all_ok:
        log(f"\n  ✅ 全部验收通过: 各组件均提供正向贡献，消融证据链完整。")
    else:
        log(f"\n  ⚠️ 部分验收未通过，请检查参数或数据。")

    log(f"\n{'='*72}")
    log(f"Table 4 消融分析完成")
    log(f"{'='*72}")


if __name__ == "__main__":
    run_ablation()
