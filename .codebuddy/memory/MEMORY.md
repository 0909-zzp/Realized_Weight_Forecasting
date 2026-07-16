# 大创项目长期记忆

## 🚀 快速衔接 (新对话从这里开始)

**项目**: 图形 Lasso 高维已实现 GMVP 权重预测 (S&P 500, K=392, 2008-2020)

**已完成**: 
- ✅ 阶段一-二: GLasso λ=3e-6, 2425天权重+邻接矩阵
- ✅ 阶段三: 特征工程 (X: 1185维), VARX 7模型 (M1-M7), 网格搜索
- ✅ Table 2 最终: M4 N-V rank1 (MSE=2.12e-5), DFL rank5 (2.28e-5)
- ✅ Table 3 最终: DFL 夏普最优 (-0.056) vs M4 (-0.342)
- ✅ Table 4 消融: 自环豁免 73%贡献, 网络惩罚显著, 证据链完整
- ✅ 参数全部确认 (见下方), 文档已更新

**待完成**:
- [x] Table 4: 消融分析 (M4→M3a→M3→M2 拆解各组件贡献) ✅ 2026-07-16
- [ ] 图 1-4: 论文可视化 (λ曲线/权重热力图/网络密度/累计回测)
- [ ] MCS 完整程序 (Hansen 2011 bootstrap)

**核心文件**:
- 参数唯一源: `图形Lasso/code/共享模块.py`
- Table 2 主脚本: `VARX/VAR及拓展（table2）.py`
- Table 3 脚本: `性能评估与可视化/Table3_投资组合表现.py`
- Table 4 脚本: `消融分析/Table4_消融分析.py`
- 网格搜索: `VARX/网格搜索.py`
- 文档: `readme/README.md`, `FILE_GUIDE.md`, `README.md`(GitHub)

**DFL 实现**: L2 闭式解 `w* = (Σ̂+(η+ρ)I)⁻¹·(η·w_prev+ρ·w_stat)`, 网络差异化 η, 盒约束±5%

**GitHub**: `https://github.com/0909-zzp/Realized_Weight_Forecasting`

## 版本记录
- **2026-07-16**: Table 4 消融分析完成
  - 消融链: M2→M3→M3a→M4, 所有 pairwise DM 高度显著
  - 自环豁免贡献最大 (73%), 网络惩罚虽小但显著 (p≈10⁻³²)
  - M2 λ₁ 改回 5e-4 (LAMBDA_LASSO_M2), 与 final_params.json 一致
  - 脚本: `消融分析/Table4_消融分析.py`, 结果: `消融分析/Table4_results.csv`
- **2026-07-15**: DFL 改为 L2 闭式解，Table 2/3 锁定
  - DFL: min w^T·Σ·w + η·‖w−w_prev‖² + ρ·‖w−w_stat‖² → 闭式解
  - Table 2: M4 rank1 (MSE=2.12e-5), DFL rank5 (MSE=2.28e-5, TO=0.357)
  - Table 3: DFL 夏普最佳 (−0.051 vs M4 −0.342), 回撤最小 (−12.6%)
  - λ_s 同步: 5e-3→1e-3 → M5 表现改善 (MSE 2.63→2.19e-5)
  - 文件整理: VARX/下删冗余日志, 模型参数移入 fitted_models/
- **2026-07-14**: Table 2 全模型跑通, Table 3 初版, LSTM BPTT 梯度修正
- **2026-07-07**: 代码审查14问题, P0~P2修复
- **2026-07-03**: 网格搜索完成, VARX参数初选
- **2026-07-01**: VARX代码审计, Bug修复
- **2026-06-28**: 特征工程, Table 2首版
- **2026-06-21**: 项目初始化

## 项目架构（2026-06-28更新）
- 四阶段流水线：L1共享模块 → L2-A λ选择 → L2-B 权重计算 → L3-A 特征工程 → L3-B VARX预测 → L3-C DFL训练 → L4 评估/可视化
- README位置：`大创/.codebuddy/README.md`（非图形Lasso/下）
- 参数唯一源：`图形Lasso/code/共享模块.py`

## 特征工程关键约束
1. **Ā 不进入 X**：滚动网络均值作为正则化掩码源 `M=1(Ā≥threshold)`，存入 `A_bar.npy` (T,K,K)，不展平入 X（避免153K维爆炸）
2. **VIX CSV 加载**：pandas read_csv 默认将数字列解析为 int64 index，必须显式 `dtype={0: str}` + `index.astype(str)` 才能用字符串查找
3. **PSI 计算**：必须用分位数分箱 + 比例（非density），≤0.1阈值源自信用评分（1-2年），对跨10年金融数据放宽至 ≤0.5
4. 外生变量全部 `shift(1)` 防泄露

## VARX预测 — 代码审计结论（2026-07-01）
1. **Bug #1（已修复）**：`fit_model` else分支用 `np.full(n_feat, LAMBDA_LASSO)` 跳过build_penalty_vector，M3未应用λ₃。修复为统一调用 `build_penalty_vector(i, False/True, mask)`。
2. **Bug #2（已修复）**：M4/M5自环系数 B_{ℓ,ii} 被λ₂惩罚，论文公式(13) λ₂项显式含j≠i。修复：自环→1e-12。
3. **设计原则**：R1网格搜索必须加稀疏度约束(≤92%)。仅评估M3可纯贪心MSE；对比M3 vs M4必须约束稀疏度，否则λ₁过大(≥5e-4)时交叉项全零→M4无靶子→消融实验退化。
4. **网络惩罚有效**：Bug修复+稀疏度约束后，M4在τ=0.6~0.9区间一致优于M3（改善3%~12%），最佳组合：λ₁=1e-4, λ₃=5e-4, τ=0.9, λ_net=5e-3, λ_s=1e-3。

## 已确定参数

### 阶段一-二（GLasso）
| 参数 | 值 |
|---|---|
| K | 392 |
| λ | 3e-6 |
| MAX_ITER | 150 |
| EPS_RIDGE | 1e-4 |
| L_TRAIN_GLASSO | 40 |

### 阶段三（VARX，最终 2026-07-15）
| 参数 | 值 | 说明 |
|---|---|---|
| P_LAGS | 3 | 固定 |
| λ₁ (LAMBDA_LASSO) | 3e-04 | M3/M4/M5共用 |
| λ₁_M2 | 5e-04 | M2独立搜索 |
| λ₃ (LAMBDA_EXOG) | 5e-04 | 外生变量 |
| τ (NETWORK_THRESHOLD) | 0.7 | 密度~26% |
| λ_s (LAMBDA_TURNOVER) | 1e-03 | M5换手率平滑 |
| ETA | 1e-04 | DFL交易成本 |
| RHO_DFL | 1e-3 | DFL锚定 |

### Table 2 最终 (2026-07-16 重跑, M2 λ₁=5e-4)
| # | Model | MSE | DM | MCS |
|:---:|---:|---:|---:|
| 4 | Network VARX | 2.12e-5 | −41.27 | 1 |
| 5 | +Smooth | 2.19e-5 | −41.78 | 2 |
| 3 | Sparse VARX | 2.25e-5 | −40.28 | 3 |
| 6 | DFL VARX (L2) | 2.28e-5 | −40.26 | 4 |
| 2 | Sparse VAR | 2.29e-5 | −40.01 | 5 |
| 1 | VAR | 9.76e-5 | — | 6 |
| 7 | LSTM | 6.48e-3 | +1.64 | 7 |

### DFL 最终方案 (L2 闭式解)
- 目标: min w^T·Σ·w + η·‖w−w_prev‖² + ρ·‖w−w_stat‖²
- 闭式: w* = (Σ̂+(η+ρ)I)⁻¹·(η·w_prev+ρ·w_stat), 归一化, 盒±5%
- Σ̂ 为训练期末40天滚动协方差
- TO=0.357 与 M4(0.354) 持平, 夏普=−0.051 最优

### Table 2 vs Table 4 分工
- **Table 2** (§5.2): 7模型水平对比, MSE/MAE/DM/MCS
- **Table 4** (§5.3): 拆M4→M3a→M3→M2 消融各组件贡献

### Table 4 消融结果 (2026-07-16)
| Step | ΔMSE | 相对改善 | DM pairwise | 解释 |
|:---:|---:|---:|---:|---|
| M2→M3 | −3.54e-7 | +1.55% | −20.77 | +外生变量 |
| M3→M3a | −1.27e-6 | +5.64% | −22.10 | +自环豁免 (最大贡献) |
| M3a→M4 | −1.17e-7 | +0.55% | −11.95 | +网络惩罚 |
| **M2→M4** | **−1.74e-6** | **+7.61%** | **−23.16** | **全部组件合计** |

**Panel A** (测试集单项):
| Model | MSE | MAE | Sparsity | Turnover |
|---:|---:|---:|---:|---:|
| M2 Sparse VAR | 2.290e-5 | 3.568e-3 | 99.0% | 0.240 |
| M3 Sparse VARX | 2.254e-5 | 3.542e-3 | 97.2% | 0.283 |
| M3a +Self-lag | 2.127e-5 | 3.446e-3 | 98.2% | 0.349 |
| M4 Network VARX | 2.116e-5 | 3.436e-3 | 98.9% | 0.354 |

**关键发现**: 自环豁免贡献最大 (73%总改善), 网络惩罚虽最小但高度显著 (p≈10⁻³²). 所有 pairwise DM 均显著, 消融证据链完整.
