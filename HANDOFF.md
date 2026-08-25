# 项目交接 — 最终版

> 最后更新: 2026-08-10 (代码审计修复后重跑)
> 本文件供新对话快速衔接，涵盖所有关键决策和最终结果

---

## 三张表最终数据

### Table 2：样本外预测（363天）

| Model | MSE | DM (HAC) | MSE-MCS p | MSE-MCS rk | 投资-MCS p | 投资-MCS rk |
|---|---|---|---|---|---|---|
| VAR | 9.76e-05 | — | 0.000 | 2 | 0.484 | 3 |
| Sparse VAR | 2.29e-05 | -22.96 | 0.000 | 2 | 0.527 | 2 |
| Sparse VARX | 2.25e-05 | -23.08 | 0.000 | 2 | 0.476 | 4 |
| **Network VARX (M4)** | **2.12e-05** | **-23.53** | **1.000** | **1** | 0.377 | 7 |
| Network+Smooth (M5) | 2.19e-05 | -23.69 | 0.000 | 2 | 0.452 | 6 |
| DFL VARX (M6) | 2.28e-05 | -22.77 | 0.000 | 2 | **1.000** | **1** |
| LSTM (M7) | 2.96e-05 | -21.96 | 0.000 | 2 | 0.470 | 5 |

> MSE-MCS: 损失=MSE, block bootstrap 2000次, 块长5天, 序贯p值单调调整。
> 投资-MCS: 损失=-净日收益, block bootstrap 5000次, 块长5天 (MCS_投资组合.py)。
> M4在预测精度上唯一最优(p=1.0), DFL在投资表现上排名第1(p=1.0)。
> LSTM已修复: softmax梯度修正, 验证集拼接序列(无真值填充泄露), MSE更新为2.96e-05。

### Table 3：投资组合表现（200天: 2018-12-28~2019-10-15, GMVP W=20, 简单收益）

| Model | Vol | Turnover | 夏普 |
|---|---|---|---|
| Equal weight | 0.123 | 0.01 | +1.08 |
| VAR (M1) | 0.121 | 3.23 | +0.94 |
| Sparse VAR (M2) | 0.091 | 0.25 | +1.70 |
| Sparse VARX (M3) | 0.091 | 0.30 | +1.67 |
| Network VARX (M4) | 0.089 | 0.37 | **+1.59** |
| Network+Smooth (M5) | 0.081 | 0.27 | +1.61 |
| **Network+Smooth + DFL (L1 drift, η=1e-6, 滚动Σ150d)** | **0.075** | **0.07** | **+2.00** |
| LSTM (M7) | 0.110 | 0.31 | +1.14 |
| Sample GMVP (W=20) | 0.073 | 0.34 | +1.60 |
| Shrinkage GMVP (W=20) | 0.096 | 0.37 | +1.09 |
| GLasso GMVP (W=20) | 0.105 | 0.70 | +1.60 |

> 外生变量axis bug修复 + 日期对齐 + 简单收益转换后重新计算。
> DFL参数: M5基模型, drift-aware L1, η=1e-6, ρ=1e-3, λ_s=3e-3, 每日滚动协方差150天
> (OOS验证期 2017-04-11~2018-09-20 选参), 无回撤刹车。
> GMVP基准使用W=20天滚动窗口（与Ā_ROLLING_WINDOW=20一致）。

### Table 4：组件消融（200天窗口，η=1e-6, 滚动Σ150）

| Ablation | MSE_w | Turnover | 夏普 |
|---|---|---|---|
| 网络VARX+平滑+DFL | 2.64e-05 | 0.07 | +2.00 |
| 网络VARX+平滑（无外生）+DFL | 2.59e-05 | 0.07 | +1.98 |
| 稀疏VARX+平滑（无网络）+DFL | 2.63e-05 | 0.07 | +2.01 |
| 网络VARX+DFL | 2.54e-05 | 0.10 | +1.90 |
| 网络VARX+平滑 | 2.46e-05 | 0.27 | +1.61 |

> +DFL将夏普从+1.61提升至+1.98~2.01。"无外部"(M2+DFL)夏普最高(+2.01),
> 说明最简单的基模型+DFL即可达到最优投资表现; 网络和外生变量的信息
> 已被DFL的锚定机制所覆盖。注: "无外部"同时改变λ₁/self_free, 非严格单因素消融。

---

## Table3 DFL 最优配置（重要发现）

**参数在评估窗口前的验证期选择，避免在 Table3 窗口上调参。**

DFL 使用真实漂移后持仓的 L1 换手约束：

| 参数 | 值 | 说明 |
|---|---|---|
| η | **1e-6** | drift-aware L1——OOS验证期按夏普选参 |
| ρ | 1e-3 | 锚定强度——OOS验证期按夏普选参 |
| Sigma窗口 | 150 | 每日滚动协方差 (每日用前150天重新估计) |
| DFL_BOX | 0.05 | 盒约束不变 |

```
DFL目标函数: min ½wᵀΣw + η·||w−w_drifted_{t-1}||₁ + ½ρ||w−w_stat||²
                                        ↑
                        w_drifted = 上一期持仓按收益漂移后的真实持仓
                        直接惩罚实际再平衡量，而非过期的目标权重
```

> Table3 结果: 回撤 -3.79%, 净夏普 +2.00, 换手 0.0685, 累计收益 +12.4% (均为净收益口径)。

---

## Table 1 — 因子分解

| | 原始 GLasso | 因子分解+BIC | 降幅 |
|---|---|---|---|
| Mean degree | 244 ± 52 | **12 ± 22** | 95% |
| 中位数 | 227 | **8** | — |
| 密度 | 60% | **3.1%** | — |

- 脚本: `图形Lasso/code/factor_bic_full.py` (全量2436天, 4进程并行, 3.6h)
- BIC选 λ=5e-5, 2436/2436天成功
- Table 1 主行报244, 下行报12(因子分解后), 正文引用Brownlees et al.(2018)

---

## 网络密度稳健性（已验证）

切换为因子分解网络(77边, 密度20%)后:
- M4权重相关性 99.87%
- Table 2 MSE差异 <0.4%
- Table 3 夏普差异 <0.02
- Table 4 结论不变
- **论文用236边即可，审稿人追问时用此实验堵**

---

## 投资MCS（已放弃）

试了8种方案全部失败——日收益相关性0.97+, 信噪比太低。
文献中投资MCS需10年+数据, 200~725天不够。
论文策略: Table 2用MSE版MCS, Table 3用夏普排序, Table 4用累积收益差。

---

## 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| K | 392 | 资产数 |
| λ_Ω | 1e-6 | GLasso λ (密度62%, 度244) |
| P_LAGS | 3 | 自回归滞后 |
| λ₁ | 3e-4 | 连接资产L1 (M3/M4/M5) |
| λ₁_M2 | 5e-4 | M2独立最优 |
| λ₃ | 5e-4 | 外生变量L1 |
| λ_net | 1e-3 | 未连接额外L1 |
| τ | 0.7 | 网络阈值 |
| Ā_ROLLING_WINDOW | 20 | 网络滚动窗口 |
| GMVP W | 20 | GMVP基准滚动窗口 |
| DFL η | 1e-6 | drift-aware L1 换手惩罚 (OOS验证期) |
| DFL ρ | 1e-3 | 锚定强度 (OOS验证期) |
| DFL Sigma窗口 | 150 | 每日滚动协方差 (每日前150天) |
| ETA | 1e-4 | 交易成本(1bp) |

---

## 关键脚本

| 脚本 | 功能 |
|---|---|
| `图形Lasso/code/共享模块.py` | 参数唯一源 |
| `图形Lasso/code/纯权重计算.py` | 日频GLasso+GMVP权重 |
| `图形Lasso/code/factor_bic_full.py` | 因子分解+BIC全量并行版 |
| `图形Lasso/code/build_Abar_idio.py` | 构建因子分解版A_bar |
| `特征工程/特征工程.py` | X/Y/A_bar构建 |
| `VARX/VAR及拓展（table2）.py` | Table 2主脚本(7模型+DFL+DM+MCS) |
| `性能评估与可视化/Table3_投资组合表现.py` | 组合评估 |
| `性能评估与可视化/generate_table3_dfl_l1.py` | 生成drift-aware L1版DFL权重 |
| `性能评估与可视化/compute_gmvp_benchmarks.py` | GMVP基准生成(改W=20) |
| `性能评估与可视化/MCS_完整程序.py` | 正式MCS(需重跑以更新LSTM) |
| `消融分析/Table4_消融分析.py` | 消融分析 |

---

## 重要文件路径

| 文件 | 说明 |
|---|---|
| `特征工程/A_bar.npy` | 原始A_bar (244边, 密度62%) |
| `特征工程/A_bar_orig.npy` | 原始备份 |
| `特征工程/A_bar_idio.npy` | 因子分解版A_bar (77边) |
| `VARX/Y_pred_model6_opt.npy` | DFL(drift-aware L1, η=1e-6)预测 |
| `VARX/Y_pred_bench_sample_W20.npy` | Sample GMVP W=20 |
| `VARX/Y_pred_bench_shrink_W20.npy` | Shrinkage GMVP W=20 |
| `VARX/Y_pred_bench_glasso_W20.npy` | GLasso GMVP W=20 |
| `VARX/Y_pred_model{4,5,6}_idio.npy` | 因子分解版预测(实验用) |
| `图形Lasso/factor_bic_full_results.npz` | 因子分解逐日degree |

---

## MCS 重跑提醒

> MCS已完成重跑, M4 p=1.0, 其余p=0。投资版MCS通过MCS_投资组合.py生成。
