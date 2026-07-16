# 项目文件导览

> 图形 Lasso 高维已实现 GMVP 权重预测
> 最后更新: 2026-07-15

---

## 核心模块

| 文件 | 用途 |
|---|---|
| `图形Lasso/code/共享模块.py` | 参数唯一源 + 公共函数 |
| `图形Lasso/code/纯权重计算.py` | 阶段二: 逐日 GLasso → GMVP 权重 + 邻接 |
| `特征工程/特征工程.py` | 阶段三-A: X/Y/A_bar 构建 |
| `VARX/VAR及拓展（table2）.py` | 阶段三-B: M1~M7 拟合 + Table 2 (含DFL L2) |
| `VARX/网格搜索.py` | 阶段三-B: 分层消融参数搜索 |
| `性能评估与可视化/Table3_投资组合表现.py` | 阶段四: 组合表现评估 |

## VARX/ 目录

```
VARX/
├── VAR及拓展（table2）.py    # 主脚本 (7模型)
├── 网格搜索.py                # 参数搜索
├── 网格搜索规范.md            # 实验设计文档
├── Table2_results.csv         # ★ Table 2
├── Y_pred_model1~7.npy        # 预测序列 (7.7MB)
├── final_params.json          # 最优参数
├── tuning_folds.csv           # 搜索逐折
├── tuning_summary.csv         # 搜索汇总
├── ablation_self_lag.csv      # 消融
├── tau_robustness.csv         # tau 稳健性
└── fitted_models/             # 模型参数 (18MB)
    ├── coefs_model1~5.npy
    ├── intercepts_model1~5.npy
    ├── scaler_model1~5.pkl
    └── feat_cols_model1~5.npy
```

## 特征工程/ 输出

| 文件 | 维度 |
|---|---|
| `X_features.npy` | ~2400 x 1185 |
| `Y_targets.npy` | ~2400 x 392 |
| `A_bar.npy` | ~2400 x 392 x 392 |
| `valid_indices.npy` | 有效日全局索引 |

## 数据层

```
数据/
├── 1min_log_return/       2436 .RData
├── 1min_log_return_npy/   .npy 缓存
├── vix_daily.csv
├── term_spread.csv
├── credit_spread.csv
└── dxy_close.csv
```

## 已确定参数 (2026-07-15)

| 参数 | 值 |
|---|:---:|
| K | 392 |
| lambda_Omega | 3e-6 |
| P_LAGS | 3 |
| LAMBDA_LASSO | 3e-4 |
| LAMBDA_LASSO_M2 | 5e-4 |
| LAMBDA_NETWORK | 1e-3 |
| LAMBDA_EXOG | 5e-4 |
| LAMBDA_TURNOVER | 1e-3 |
| NETWORK_THRESHOLD | 0.7 |
| ETA | 1e-4 |
| RHO_DFL | 1e-3 |

## Table 2 (最终, MCS正式)

| # | Model | MSE | DM | MCS p-val | 90%MCS |
|:---:|---:|---:|---:|---:|:---:|
| 4 | Network VARX | 2.12e-5 | -41.27 | 1.0000 | ✅ |
| 5 | +Smooth | 2.19e-5 | -41.78 | 0.1245 | ✅ |
| 7 | LSTM | 6.48e-3 | +1.64 | 0.2645 | ✅ |
| 3 | Sparse VARX | 2.25e-5 | -40.28 | 0.0995 | ❌ |
| 2 | Sparse VAR | 2.29e-5 | -40.01 | 0.0980 | ❌ |
| 6 | DFL VARX (L2) | 2.28e-5 | -40.26 | 0.0895 | ❌ |
| 1 | VAR | 9.76e-5 | - | 0.0625 | ❌ |

> MCS: Hansen-Lunde-Nason (2011) bootstrap, 2000次, 块长5天. M4唯一p=1.0.

## Table 3 要点

M4 风险最低 (RPV=0.0101)。DFL 夏普最优 (-0.051 vs M4 -0.342)，回撤最小 (-12.6%)。
DFL 采用 L2 闭式解: w* = (Sigma + (eta+rho)I)^(-1) . (eta.w_prev + rho.w_stat).

## 消融分析/ (Table 4)  ✅ 2026-07-16

```
消融分析/
├── Table4_消融分析.py        # 脚本
├── Table4_results.csv         # ★ Table 4
└── table4_log.txt             # 运行日志
```

### Table 4 边际贡献

| Step | ΔMSE | 相对改善 | DM pairwise | 解释 |
|:---:|---:|---:|---:|---|
| M2→M3 | −3.54e-7 | +1.55% | −20.77 | +外生变量 |
| M3→M3a | −1.27e-6 | +5.64% | −22.10 | +自环豁免 (~73%总贡献) |
| M3a→M4 | −1.17e-7 | +0.55% | −11.95 | +网络惩罚 |
| M2→M4 | −1.74e-6 | +7.61% | −23.16 | 全部组件合计 |

> 自环豁免贡献最大，网络惩罚虽最小但高度显著 (p≈10⁻³²)。消融证据链完整。

## 待办

- [x] Table 2
- [x] Table 3
- [x] Table 4 (消融分析) ✅ 2026-07-16
- [x] MCS 完整程序 ✅ 2026-07-16
- [ ] 图 1-4 论文可视化
