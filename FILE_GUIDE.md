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

## 已确定参数 (2026-07-18, P0/P1/P2验证)

| 参数 | 值 |
|---|:---:|
| K | 392 |
| lambda_Omega | 3e-6 |
| P_LAGS | 3 |
| LAMBDA_LASSO (M3/M4/M5) | 3e-4 |
| LAMBDA_LASSO_M2 | 5e-4 |
| LAMBDA_LASSO_M3a | 4.5e-4 |
| LAMBDA_NETWORK | 1e-3 |
| LAMBDA_EXOG | 5e-4 |
| LAMBDA_TURNOVER | 1e-3 |
| NETWORK_THRESHOLD | 0.7 |
| ETA | 1e-4 |
| RHO_DFL | 1e-3 |

## Table 2 (最终, MCS正式)

| # | Model | MSE | DM | MCS p-val | MCS rank | 90%MCS |
|:---:|---:|---:|---:|---:|:---:|:---:|
| 4 | Network VARX | 2.12e-5 | -41.27 | 1.0000 | 1 | ✅ |
| 7 | LSTM | 6.48e-3 | +1.64 | 0.2645 | 2 | ✅ |
| 5 | +Smooth | 2.19e-5 | -41.78 | 0.1245 | 3 | ✅ |
| 3 | Sparse VARX | 2.25e-5 | -40.28 | 0.0995 | 4 | ❌ |
| 2 | Sparse VAR | 2.29e-5 | -40.01 | 0.0980 | 5 | ❌ |
| 6 | DFL VARX (L2) | 2.28e-5 | -40.26 | 0.0895 | 6 | ❌ |
| 1 | VAR | 9.76e-5 | - | 0.0625 | 7 | ❌ |

> MCS: Hansen-Lunde-Nason (2011) bootstrap, 2000次, 块长5天. M4唯一p=1.0.

## Table 3 要点

Sample GMVP RPV 最低 (0.00573)。DFL 采用 drift-aware L1 换手约束
(η=1e-6, ρ=1e-3, M5基模型, OOS验证期选参) + 每日滚动协方差150天，换手 0.0685，
净夏普 +2.00，最大回撤 -3.79%，为主动模型中最优。
生成脚本: `性能评估与可视化/generate_table3_dfl_l1.py`。

## 消融分析/ (Table 4) ✅ 2026-07-16

```
消融分析/
├── Table4_消融分析.py        # 主脚本 (DFL基线减法)
├── Table4_完整.csv         # ★ Table 4
└── Table4_加法消融_旧.py      # 旧版 (加法链, 已替换)
```

### Table 4 — DFL基线减法

| 设定 | MSE_w | RPV(年) | 换手率 | 净夏普 |
|---|---:|---:|---:|---:|
| 网络VARX + 平滑 + DFL | 2.64×10⁻⁵ | 0.0056 | 0.0685 | +1.9989 |
| 网络VARX + 平滑（无外生）+ DFL | 2.59×10⁻⁵ | 0.0056 | 0.0695 | +1.9824 |
| 稀疏VARX + 平滑（无网络）+ DFL | 2.63×10⁻⁵ | 0.0056 | 0.0698 | +2.0055 |
| 网络VARX + DFL | 2.54×10⁻⁵ | 0.0056 | 0.1026 | +1.9024 |
| 网络VARX + 平滑 | 2.46×10⁻⁵ | 0.0063 | 0.2720 | +1.6099 |

> DFL贡献最大 (夏普+0.3890); 平滑项贡献 (+0.0965), 网络项贡献可忽略.

## 待办

- [x] Table 2
- [x] Table 3
- [x] Table 4 (消融分析) ✅ 2026-07-16
- [x] MCS 完整程序 ✅ 2026-07-16
- [ ] 图 1-4 论文可视化
