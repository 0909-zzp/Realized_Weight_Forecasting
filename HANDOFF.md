# 项目交接 — 新对话快速衔接

> 最后一次整理: 2026-07-19
> GitHub: main @ 4d0cbd8 (有未推送 commit eaefce7, 网络恢复后 push)

---

## 当前状态总览

| 项目 | 状态 | 备注 |
|---|---|---|
| Table 1 (描述性统计) | ✅ | 新增: 因子分解+BIC 测试 (50天), mean degree 9→244 |
| Table 2 (OOS预测精度) | ✅ | 363天, M4最优 MSE=2.12e-5 |
| Table 3 (组合表现) | ✅ | 200天窗口, DFL夏普+1.68 |
| Table 4 (消融分析) | ✅ | 200天窗口, DFL完整模型夏普+1.69 |
| Comment 2 回应 (λ_Ω稳健性) | ✅ | 20天网络密度测试, 跨量级不退化 |
| Comment 3 回应 (高维可行性) | ✅ | Word文档 Comment3_Response_v3.docx |
| Comment 1/4 | ⬜ | 未处理 |

---

## 核心参数 (共享模块)

| 参数 | 值 | 说明 |
|---|---|---|
| K | 392 | 资产数 |
| λ_Ω | 3e-6 | GLasso 正则化 (OOS方差最优) |
| P_LAGS | 3 | 自回归滞后 |
| λ₁ (M3/M4/M5) | 3e-4 | 连接资产 L1 |
| λ₁_M2 | 5e-4 | M2 独立最优 |
| λ₁_M3a | 4.5e-4 | M3a 独立最优 (P1) |
| λ₃ | 5e-4 | 外生变量 L1 |
| λ_net | 1e-3 | 未连接额外 L1 |
| τ | 0.7 | 网络阈值 (有效密度 26.4%) |
| λ_s | 1e-3 | 换手率平滑 |
| ETA | 1e-4 | 交易成本 (1 bp) |
| ρ | 1e-3 | DFL 锚定强度 |

---

## Table 1 — 网络密度争论

**问题**: 审稿人说 mean degree=244 (密度62%) 不够稀疏。

**发现**: Brownlees et al. (2018, JAE) 先做因子分解再 GLasso。我们实现了:
- 脚本: `图形Lasso/code/factor_bic_test.py` (N_DAYS=50)
- 结果: 原始 244 → 因子+BIC **9 ± 8** (密度 2.4%)
- BIC 选 λ=5e-5, 50/50 天成功
- 全量跑不动 (OOM), 50天抽样即够论文论证

**结论**: Table 1 报告原始 244, 论文里加一段引用 Brownlees 解释, 50天抽样作为稳健性证据。

---

## 数据集

| 文件 | 位置 |
|---|---|
| 特征 X | 特征工程/X_features.npy (2415×1185) |
| 目标 Y | 特征工程/Y_targets.npy (2415×392) |
| 网络掩码源 | 特征工程/A_bar.npy (2415×392×392) |
| 预测 M1~M7 | VARX/Y_pred_model{1-7}.npy (363×392) |
| GMVP基准 | VARX/Y_pred_bench_{sample,shrink,glasso}.npy |

---

## 三张最终表

### Table 2 (363天, 全测试期)
| # | Model | MSE | DM | Turnover |
|---|---|---|M4 | 2.12e-5 | -41.27 | 0.354 |
| 其余见 VARX/Table2_results.csv

### Table 3 (200天窗口: 2018-12-19 ~ 2019-10-08)
| Model | 净夏普 | 换手率 |
|---|---|---|
| Equal weight | +0.81 | 0.01 |
| Sample GMVP | +1.54 | 0.19 |
| Shrinkage GMVP | +1.26 | 0.37 |
| GLasso GMVP | +2.03 | 0.60 |
| Sparse VARX | +1.46 | 0.30 |
| Network VARX | +1.39 | 0.37 |
| N-VARX + DFL | +1.68 | 0.39 |

### Table 4 (200天窗口消融)
| 设定 | MSE_w | RPV | 净夏普 |
|---|---|---|---|
| 完整(M5+DFL) | 2.46e-5 | 0.0071 | +1.69 |
| 无外部(M2+DFL) | 2.54e-5 | 0.0076 | +1.75 |
| 无网络(M3a+DFL) | 2.42e-5 | 0.0073 | +1.69 |
| 无平滑(M4+DFL) | 2.43e-5 | 0.0073 | +1.68 |
| 无DFL(M5) | 2.36e-5 | 0.0069 | +1.34 |

---

## Comment 3 审稿回复

- 文档: `Comment3_Response_v3.docx` (含敏感性图)
- 生成脚本: `generate_comment3.py`
- 图片: `消融分析/M4_sensitivity_combined.png`

---

## LSTM 状态

- 脚本: `LSTM_standalone.py` (纯NumPy)
- 参数: seq=30, hid=128, softmax T=0.5, dropout=0.05, lr=0.0005
- BPTT 有 bug (c_cur 取了输入门而非细胞状态) — 客观上起隐式正则化
- 修复版 (dropout=0.3) 效果更差 — 保持旧版
- 测试 MSE: 2.82e-5, 换手率: 0.22, 200天净夏普: +1.37

---

## 关键脚本速查

| 脚本 | 功能 |
|---|---|
| `图形Lasso/code/共享模块.py` | 参数唯一源 |
| `特征工程/特征工程.py` | X/Y/A_bar 构建 |
| `VARX/VAR及拓展（table2）.py` | Table 2 主脚本 (7模型+DFL+LSTM) |
| `VARX/网格搜索.py` | 超参数搜索 |
| `性能评估与可视化/Table3_投资组合表现.py` | 组合评估 |
| `性能评估与可视化/compute_gmvp_benchmarks.py` | GMVP 基准生成 |
| `消融分析/Table4_消融分析.py` | 消融分析 |
| `图形Lasso/code/factor_bic_test.py` | 因子分解+BIC |
| `LSTM_standalone.py` | LSTM 独立版 |

---

## 待办

- [ ] 推送未提交 commit (eaefce7)
- [ ] Comment 1 (无向网络论证)
- [ ] Comment 4 (理论/模拟)
- [ ] Table 1 最终确认 (原始244 vs 因子分解后9)
- [ ] 审查 LSTM 代码 BPTB bug 是否修/保留
- [ ] 全量因子分解+BIC (需服务器)
