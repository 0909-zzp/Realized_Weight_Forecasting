# 图形 Lasso 高维已实现 GMVP 权重预测

> 暨南大学大学生创新创业训练计划 · 基于机器学习的高维投资组合统计建模

---

## 项目简介

### 问题定义

给定 S&P 500 成分股（K = 392）的 1 分钟日内高频收益率（2008-2020，2436 个交易日），
需完成两项核心任务：

| 任务 | 目标 | 对应阶段 |
|------|------|:--:|
| **构造** | 用 Graphical Lasso 从高频收益中稳定估计每日 GMVP 权重与资产条件依赖网络 | 一、二 |
| **预测** | 用网络正则化 VARX 模型预测未来 GMVP 权重，决策聚焦损失直接最小化组合风险+交易成本 | 三、四 |

### 项目产出

- 全部 2436 个交易日的 GMVP 权重序列 (N × K 矩阵)
- 逐日资产网络 (边集、密度、中心性)
- VARX 模型拟合结果与滚动 OOS 回测指标
- 多基准模型对比（等权重 / 滚动 GMVP / 收缩估计）

### 论文对应

| 论文章节 | 内容 | 对应模块 |
|:--:|------|------|
| §3.1 | GLasso 精度矩阵估计 + GMVP 权重 | `λΩ选择.py` + `权重计算+描述性分析（步骤二）.py` |
| §3.2 | 网络正则化 VARX 预测模型 | `特征工程.py` + `VARX预测.py` |
| §3.3 | 决策聚焦损失函数 | `决策聚焦训练.py` |
| §4.1 | λ_Ω 选择与交易成本校准 | `λΩ选择.py` (当前 λ_Ω = 5e-7) |
| §5.1 | 权重描述性统计 | `权重计算+描述性分析（步骤二）.py` |
| §5.2 | OOS 预测表现与基准对照 | `VARX预测.py` + `性能评估.py` |

### 技术路径

```
高频收益 ──GLasso──→ 稀疏精度矩阵 ──→ GMVP 权重 + 资产网络
                                            │
                    ┌───────────────────────┘
                    ▼
            特征工程 (滞后权重 + 外生变量 + 网络拓扑)
                    │
                    ▼
            网络正则化 VARX 预测
                    │
                    ▼
            决策聚焦损失: min Var(w̃_pred) + η·Turnover
```

### 当前数据结果

| 指标 | 值 |
|------|:--:|
| 最优 λ_Ω | 5e-7 |
| 测试集 OOS (λ=5e-7 vs 等权重) | -41.9% 组合方差 |
| 全量权重矩阵 | 2436 × 392（计算中） |
| 权重计算方式 | 纯权重计算.py（独立脚本，零项目依赖） |
| 并行策略 | multiprocessing.Pool + imap_unordered（11 worker） |
| 数据读取 | .npy 快读（已转换 2436 天） |
| R 版对照 | code/R.R（超参数已对齐，串行，可用作备份） |

---

## 目录

- [1. 项目架构](#1-项目架构)
- [2. 模块说明](#2-模块说明)
- [3. 环境配置](#3-环境配置)
- [4. 全流程操作](#4-全流程操作)
- [5. 参数配置](#5-参数配置)
- [6. 开发规范与代码约束](#6-开发规范与代码约束)
- [7. 输出文件](#7-输出文件)

---

## 1. 项目架构

### 1.1 四阶段流水线

```
                          ┌───────────────────────┐
                          │  L0 数据层             │
                          │  1min_log_return/      │
                          │  2436 .RData 文件       │
                          │  K=392 × M≈390        │
                          └───────────┬───────────┘
                                      │
┌─────────────────────────────────────┼─────────────────────────────┐
│                           共享模块.py (L1)                         │
│   参数 · 文件索引 · 数据加载 · 协方差 · GLasso · 权重 · 诊断 · 日志  │
└──────────┬──────────────┬──────────────┬──────────────┬────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
    │  阶段一    │  │  阶段二    │  │  阶段三    │  │  阶段四    │
    │  L2-A     │  │  L2-B     │  │  L3       │  │  L4       │
    │           │  │           │  │           │  │           │
    │ λΩ选择.py │→│ 权重计算   │→│ 特征工程.py│→│ 性能评估.py│
    │           │  │ +描述性    │  │ VARX预测  │  │ 可视化.py  │
    │ 固定窗    │  │ 统计.py   │  │ 决策聚焦   │  │           │
    │ 44+5+22  │  │           │  │ 训练.py   │  │           │
    │           │  │ 滚动窗40天 │  │ 滚动回测   │  │           │
    └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
          │              │              │              │
          │ lambda_opt   │              │              │
          │    .txt      │              │              │
          └──────────────┘              │              │
                                        │              │
          权重矩阵 + 网络 + 换手率 ─────┘              │
                                        │              │
               预测权重 + 回测指标 ──────────────────┘
```

### 1.2 论文公式 ↔ 代码映射

| 论文 | 含义 | 代码 |
|:--:|------|------|
| (3) Σ̂_RC | 已实现协方差 | `compute_raw_cov()` |
| (4) Ω̂ | GLasso 精度矩阵 | `do_glasso()` |
| (5) w̃ | GMVP 权重 | `w_from_prec()` |
| (6) A | 资产网络 | `compute_adjacency()` |
| (7) Ā | 滚动网络均值 | `rolling_network_mean()` (特征工程.py) |
| §3.2 | VARX 模型 | `fit_varx_asset()` (VARX预测.py) |
| §3.3 | 决策聚焦损失 | `decision_focused_loss()` (决策聚焦训练.py) |

---

## 2. 模块说明

### 2.1 `code/共享模块.py` — 基础设施层

所有模块的唯一 import 来源。集中管理全部参数、路径和公共函数。

**参数**：

```python
# 数据
K = 392                         # 资产数

# GLasso
MAX_ITER = 150                  # 坐标下降最大迭代（优化后，原500）
TOL_GLASSO = 1e-4               # 收敛容差
GLASSO_ENET_TOL = 5e-4          # 对偶间隙容差
EPS_RIDGE = 1e-4                # Ridge 扰动
RIDGE_FALLBACK = [5e-4, 1e-3, 5e-3, 1e-2]
TOL_ZERO = 1e-8                 # 零元阈值

# 时间窗
L_TRAIN_GLASSO = 40             # GLasso 滚动训练窗
L_TRAIN_VARX   = 500            # VARX 训练窗
L_VAL_VARX     = 60             # VARX 验证窗
L_TEST_VARX    = 200            # VARX 测试窗

# VARX 模型
P_LAGS          = 3             # 滞后阶数
LAMBDA_LASSO    = 1e-4          # ℓ1 惩罚
LAMBDA_TURNOVER = 1e-2          # ℓ2 平滑惩罚
LAMBDA_NETWORK  = 1e-3          # 网络正则化强度
NETWORK_THRESHOLD = 0.3         # 网络边截断阈值

# 决策损失
ETA = 1e-4                      # 交易成本系数
```

### 2.2 阶段一：`code/λΩ选择.py`

固定时间窗（训练44天+验证5天+测试22天）扫描候选 λ，选最小化验证集组合方差的最优 λ。

**产出**：`lambda_opt.txt`, `lambda_selection_diagnostics.csv`, `test_results.txt`

### 2.3 阶段二：`code/权重计算+描述性分析（步骤二）.py` / `code/纯权重计算.py`

**权重计算方案**：

| 脚本 | 依赖 | 适用 | 状态 |
|------|------|------|:--:|
| `权重计算+描述性分析（步骤二）.py` | 共享模块 | 全量统计+诊断 | 需修复并行问题 |
| **`纯权重计算.py`** | **零项目依赖** | **快速权重输出** | ✅ 主力 |

`纯权重计算.py` 特点：
- 不 import 共享模块，参数独立嵌入（对齐共享模块）
- 用 `multiprocessing.Pool.imap_unordered` 替代 `ProcessPoolExecutor`（Windows spawn 兼容）
- Ridge 退避链：`[1e-4, 1e-3, 1e-2]`（精简化三步）
- `if __name__ == '__main__': main()` + `freeze_support()`（Windows 安全）

用选定的 λ_Ω 对全部 2436 个交易日逐日计算单日 GLasso（论文公式 (3)→(4)→(5)），得到每日 GMVP 权重与资产网络。

**产出**：`reg_weights_2436.csv`

### 2.4 阶段三-A：`code/特征工程.py`

从阶段二输出的权重矩阵和网络中提取 VARX 预测特征。

**功能**：
- 构建滞后权重特征 (P_LAGS × K 维)
- 接入外生变量 (VIX/行业价差等)
- 计算网络拓扑特征 (度中心性/PageRank/聚类系数)
- 构建滚动网络均值 (论文公式7)

**产出**：`X_features.npy`, `Y_targets.npy`

### 2.5 阶段三-B：`code/VARX预测.py`

网络正则化 VARX 模型定义、拟合、滚动 OOS 回测。

**功能**：
- `build_network_mask()` — 根据滚动网络均值构建二值约束掩码
- `fit_varx_asset()` — 逐资产 Lasso 拟合（支持网络掩码引导的稀疏化）
- `predict_varx()` — 模型预测
- `main()` 内嵌扩展窗口滚动回测（训练→预测→评分→窗口前移）

**产出**：`varx_coefs.npy`, `backtest_port_variances.npy`, `backtest_turnovers.npy`

### 2.6 阶段三-C：`code/决策聚焦训练.py`

将交易成本嵌入训练损失，替代传统 MSE。

**损失函数**：
```
L = w̃_predᵀ·Σ̂_realized·w̃_pred + η·‖w̃_pred - w̃_{t-1}‖₁
```

**功能**：
- `decision_focused_loss()` — 单期损失
- `evaluate_predictions()` — 全序列评估

### 2.7 阶段四：`code/性能评估.py`

多基准性能对比与 LaTeX 表格生成。

**基准**：
1. 等权重 (Equal Weight)
2. 滚动样本协方差 GMVP (Rolling GMVP)

**产出**：`performance_comparison.csv`, `performance_table.tex`

### 2.8 阶段四：`code/可视化.py`

论文图表生成。

**图表**：
1. λ-得分选择曲线
2. GMVP 权重全样本热力图
3. 网络密度 + 换手率双轴时序
4. 累计 OOS 回测表现对照

### 2.10 辅助脚本

| 脚本 | 用途 |
|------|------|
| `纯权重计算.py` | **主力**：独立权重计算（零项目依赖，Windows/Linux双平台） |
| `_spot_test.py` | Ridge 值快速验证（抽样 5 天）|
| `_convert_npy.py` | .RData → .npy 批量转换（I/O 加速 5~10×） |
| `快速λ选择.py` | 跳过 λ 扫描，直接使用已知 λ 评估 |
| `_quick_test.py` | 环境诊断：验证 graphical_lasso 可运行 |
| `_check_cond.py` | 协方差条件数检查 |
| `_check_big_lambda.py` | 大 λ 值失败诊断 |
| `_test_small_lambda.py` | λ 细化搜索 (2e-6~1e-5) |
| `_test_deeper.py` | λ 深度探索 (1e-7~5e-7) |
| `R.R` | R 语言备份版权重计算（超参与 Python 对齐，串行） |

### 2.11 云端脚本（试用阶段）

| 脚本 | 平台 | 用途 |
|------|------|------|
| `_run_pool.py` | Ubuntu 云端 | multiprocessing.Pool 并行版 |
| `_run_serial.py` | Ubuntu 云端 | 串行 nohup 后台版 |
| `_run_clean.py` | 通用 | 零项目依赖并行版（前期迭代） |

### 2.12 已弃用模块

| 文件 | 原用途 | 替代 |
|------|------|------|
| `λ值确定（步骤一）.py` | BIC+精搜选 λ | `λΩ选择.py` |
| `完整流程.py` | 旧流程入口 | 分别运行各阶段脚本 |
| `仅验证窗精选.py` | 滚动验证窗 | `λΩ选择.py` |

---

## 3. 环境配置

### 3.1 系统要求

| 事项 | 要求 |
|------|------|
| Python | 3.9+ (实测 3.13) |
| 内存 | ≥ 8 GB (推荐 16 GB) |
| 磁盘 | ≥ 2 GB |

### 3.2 安装

```bash
cd 大创/图形Lasso
pip install numpy pandas scikit-learn pyreadr matplotlib
```

### 3.3 依赖清单

| 包 | 最低版本 | 用途 |
|---|:--:|------|
| numpy | 1.21+ | 矩阵运算 |
| pandas | 1.3+ | DataFrame, CSV |
| scikit-learn | 1.0+ | graphical_lasso, Lasso |
| pyreadr | 0.4+ | .RData 读取 |
| matplotlib | 3.5+ | 可视化 (可选) |

### 3.4 数据

```
大创/
└── 数据/
    └── 1min_log_return/
        ├── 2008 01 02_1min_log_return.RData
        ├── ...
        └── (共 2436 个文件)
```

路径由 `共享模块.py` 自动解析，无需手动配置。

---

## 4. 全流程操作

### 4.1 数据预检（每次运行前必做）

**原则**：运行任何阶段前，先验证数据格式与代码假设一致。避免 30-60 分钟计算后发现数据不匹配。

**检查清单**：

| # | 检查项 | 命令/方法 | 期望 |
|:--:|------|------|------|
| 1 | 文件数量 | `ls 数据/1min_log_return/ | grep "_1min_log_return.RData" | wc -l` | 2436 |
| 2 | `rett1` 变量存在 | `pyreadr.read_r(file)["rett1"]` | 无 KeyError |
| 3 | 数据维度 (K×M) | `rett1.shape` | K=392, M=390 附近 |
| 4 | 数据类型 | `rett1.dtype` | float64 |
| 5 | 缺失值 | `np.isnan(rett1).sum()` | 0 |
| 6 | 无穷值 | `np.isinf(rett1).sum()` | 0 |
| 7 | 数值范围 | `rett1.min(), rett1.max()` | 对数收益合理范围 (~±10%) |

**快速诊断脚本模板**：

```python
# 数据预检 (运行前粘贴到 Python 终端)
import os, numpy as np, pyreadr
DATA = "d:/.../大创/数据/1min_log_return"
files = sorted([f for f in os.listdir(DATA) if f.endswith("_1min_log_return.RData") and not f.startswith("1min")])
print(f"交易日: {len(files)}")

# 抽样 5 个交易日检查
for i in [0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]:
    rett = pyreadr.read_r(os.path.join(DATA, files[i]))["rett1"].values
    print(f"[{i}] shape={rett.shape} dtype={rett.dtype} "
          f"range=[{rett.min():.4f},{rett.max():.4f}] "
          f"NaN={np.isnan(rett).sum()} Inf={np.isinf(rett).sum()}")
```

**已知特征**（2026-06-21 验证通过）：

| 属性 | 值 | 备注 |
|------|:--:|------|
| K (资产数) | 392 | 与共享模块一致 |
| M (分钟数) | 390 | 抽样无波动 |
| dtype | float64 | GLasso 要求的精度 |
| NaN / Inf | 0 | 数据干净 |
| 全零列 | 每天约 1 列 | 不影响计算，Ridge 退避链覆盖 |
| M < K | 390 < 392 | 原始协方差奇异，Ridge 必不可少 |

### 4.2 快速开始（已有 λ_opt）

```bash
cd 大创/图形Lasso

# 步骤 0: 数据预检（见 §4.1）
# 步骤 1: 环境检查
python code/_quick_test.py

# 步骤 2: 全量权重计算 (30-60 min)
python code/权重计算+描述性分析（步骤二）.py

# 步骤 3: 特征工程
python code/特征工程.py

# 步骤 4: VARX 拟合
python code/VARX预测.py

# 步骤 5: 决策聚焦评估
python code/决策聚焦训练.py

# 步骤 6: 性能评估 + 可视化
python code/性能评估.py
python code/可视化.py
```

### 4.3 从头开始（需重选 λ）

```bash
cd 大创/图形Lasso

# 步骤 0: 数据预检（见 §4.1）
# 阶段一: 选 λ (10-15 min)
python code/λΩ选择.py

# 阶段二: 全量权重 (30-60 min)
python code/权重计算+描述性分析（步骤二）.py

# 阶段三: VARX 预测流水线
python code/特征工程.py
python code/VARX预测.py          # 含滚动回测
python code/决策聚焦训练.py

# 阶段四: 评估与可视化
python code/性能评估.py
python code/可视化.py
```

### 4.4 常见问题

| 问题 | 解决 |
|------|------|
| **Windows multiprocessing 卡住** | 用 `纯权重计算.py`（`Pool.imap_unordered` + `freeze_support`）替代原版 |
| **全部天数失败 (Non SPD result)** | Ridge 不够：用退避链 `[1e-4, 1e-3, 1e-2]` 从小到大试 |
| **云端 8vCPU 实例 Pool 卡死** | 竞价实例受限，换 AutoDL 或本地跑 `纯权重计算.py` |
| **ProcessPoolExecutor 无进度** | Windows spawn + 子进程重复扫目录：改用 `Pool.imap_unordered` |
| GLasso 卡住 | 协方差条件数过高 (崩盘日)，退避链自动升级 Ridge |
| FloatingPointError | λ 太小，增大 EPS_RIDGE 或提高 λ 下界 |
| 内存不足 | 减少 N_WORKERS |
| spawn RuntimeError | 确保 `if __name__ == '__main__':` + `freeze_support()` |
| 所有 λ 得分 inf | sklearn 版本问题，检查 graphical_lasso 无 cov_init 参数 |

---

## 5. 参数配置

所有参数在 `code/共享模块.py` 中集中定义。修改后同步更新本 README。

### 5.1 已确定参数 ✓（阶段一-二验证通过）

| 参数 | 值 | 说明 |
|------|:--:|------|
| K | 392 | 资产数 |
| MAX_ITER | 150 | GLasso 最大迭代（优化后，原 500，M<K 场景 30~80 次即收敛） |
| TOL_GLASSO | 1e-4 | 收敛容差 |
| EPS_RIDGE | 1e-4 | Ridge 扰动 |
| λ_Ω | 5e-7 | 选定正则化参数 |
| L_TRAIN_GLASSO | 40 | 滚动训练天数 |
| ETA | 1e-4 | 交易成本系数 (10 bps) |

### 5.2 VARX 超参数 ⚠ 全部待定

> 当前值为初始占位，需通过阶段三验证实验确定。

| 参数 | 占位值 | 说明 | 选参方式 |
|------|:--:|------|------|
| P_LAGS | 3 | 自回归阶数 | AIC/BIC 或验证集 |
| LAMBDA_LASSO | 1e-4 | ℓ1 正则化 | 交叉验证 |
| LAMBDA_TURNOVER | 1e-2 | ℓ2 平滑 | 验证集调优 |
| LAMBDA_NETWORK | 1e-3 | 网络正则化 | 验证集调优 |
| NETWORK_THRESHOLD | 0.3 | 边截断 | 验证集调优 |

### 5.3 VARX 实验设计参数 ⚠ 待定（非超参数）

| 参数 | 占位值 | 说明 |
|------|:--:|------|
| L_TRAIN_VARX | 500 | 训练窗 |
| L_VAL_VARX | 60 | 验证窗 |
| L_TEST_VARX | 200 | 测试窗 |

---

## 6. 开发规范与代码约束

### 6.1 通用规范

| 规则 | 说明 |
|------|------|
| **论文为准** | 任何实现细节与论文公式冲突时，以论文为准。代码注释需标注对应的公式编号 |
| **数据预检** | 运行任何阶段前，先验证数据格式与代码假设一致（维度K×M、dtype、NaN/Inf、数值范围）。抽样 5 个文件即可，避免 30+ 分钟计算后发现不匹配 |
| UTF-8 输出 | 每个脚本首行 `sys.stdout.reconfigure(encoding='utf-8')` |
| Windows spawn | 并行入口必须 `if __name__ == '__main__': main()`，工作函数必须模块级 |
| 类型注解 | 公共 API 函数需标注参数和返回值类型 |
| float64 全程 | GLasso 输入必须是 `float64`，`float32` 导致 100% 失败 |
| 无硬编码 | 所有魔数从 `共享模块.py` 导入，禁止脚本内写死数值 |
| 纯函数优先 | 协方差/GLasso/权重函数无副作用，方便测试与复用 |

### 6.2 GLasso 调用规范

所有 GLasso 拟合必须通过 `共享模块.do_glasso()` 调用，禁止直接调用 `sklearn.covariance.graphical_lasso`：

```python
# ✅ 正确
from 共享模块 import do_glasso
prec, cov_est = do_glasso(cov_mat, lam)

# ❌ 错误
from sklearn.covariance import graphical_lasso
prec, cov_est = graphical_lasso(emp_cov=c, alpha=lam, cov_init=...)  # cov_init 不兼容
```

### 6.3 协方差计算规范

- **不除以分钟数 M**：协方差 = `X @ Xᵀ + ridge·I`，与论文公式(3)一致
- 多日合并：各天 `raw_cov` 求算术平均，保持单日与多日同量级
- `EPS_RIDGE` 统一从 `共享模块` 导入

### 6.4 命名规范

| 类别 | 规范 | 示例 |
|------|------|------|
| 主线脚本 | 中文描述性名称 | `λΩ选择.py`, `特征工程.py` |
| 辅助/诊断 | `_` 前缀 + 英文 | `_check_cond.py`, `_test_small_lambda.py` |
| 输出文件 | 英文小写 + 下划线 | `lambda_opt.txt`, `backtest_port_variances.npy` |
| 弃用脚本 | 文件头部标注 `# [已弃用]` 及替代方案 | — |

### 6.5 模块依赖规则

```
L0 数据 ──→ L1 共享模块 ──→ L2 构造层 ──→ L3 预测层 ──→ L4 评估层
```

| 规则 | 说明 |
|------|------|
| 单向依赖 | 上层 import 下层，反之禁止 |
| L1 零项目依赖 | `共享模块.py` 不 import 任何项目内其他 `.py` 文件 |
| 文件耦合 | 阶段间通过 `.txt` / `.csv` / `.npy` 传递数据，不直接 import |
| 参数唯一源 | 所有可调参数在 `共享模块.py` 中有且仅有一处定义 |

### 6.6 参数修改规程

```
1. 在 共享模块.py 中修改参数值
2. 更新 README §5 对应的参数表条目
3. 重新运行受影响的阶段脚本
4. 对比新旧输出，确认无退化
5. 若参数从 [TBD] 变为已确定，摘除标记并注明验证方式
```

### 6.7 弃用代码标记

已弃用的脚本在文件头部标注：

```python
# ===================================================================
# [已弃用] 原因：BIC 对协方差尺度敏感
# 替代脚本：code/λΩ选择.py
# 保留仅供参考，不再维护。
# ===================================================================
```

### 6.8 日志规范

```python
from 共享模块 import set_log_file, log

set_log_file(OUT_DIR / "模块名_log.txt")

log("=" * 65)
log("  阶段标题")
log("=" * 65)
log(f"参数: 交易日={n_days}  K={K}  并行={N_WORKERS}")
log(f"进度: {done}/{total} ({pct:.0f}%)  耗时:{elapsed:.0f}s  ETA:{eta:.0f}s")
```

---

## 7. 输出文件

### 阶段一-二（构造层）

| 文件 | 位置 | 来源 |
|------|------|:--:|
| `lambda_opt.txt` | `图形Lasso/` | λΩ选择.py |
| `lambda_selection_diagnostics.csv` | `图形Lasso/` | λΩ选择.py |
| `test_results.txt` | `图形Lasso/` | λΩ选择.py |
| `reg_weights_2436.csv` | `code/输出数据/` | 纯权重计算.py |
| `reg_weights_2436_R.csv` | `code/输出数据/` | R.R（备份验证） |
| `reg_cond_2436_R.csv` | `code/输出数据/` | R.R |
| `Table1_Descriptive_Statistics.csv` | `code/输出数据/` | 步骤二 |
| `Daily_Statistics.csv` | `code/输出数据/` | 步骤二 |
| `最后一日的精度矩阵.csv` | `code/输出数据/` | 步骤二 |
| `per_asset_mean_degree.csv` | `code/输出数据/` | 步骤二 |
| `failed_days.csv` | `code/输出数据/` | 步骤二 (如果有失败日) |

### 阶段三（预测层）

| 文件 | 位置 |
|------|------|
| `X_features.npy` | `code/输出数据/` |
| `Y_targets.npy` | `code/输出数据/` |
| `varx_coefs.npy` | `code/输出数据/` |
| `backtest_port_variances.npy` | `code/输出数据/` |
| `backtest_turnovers.npy` | `code/输出数据/` |

### 阶段四（评估层）

| 文件 | 位置 |
|------|------|
| `performance_comparison.csv` | `code/输出数据/` |
| `performance_table.tex` | `code/输出数据/` |
| `图1_lambda_selection.png` ~ `图4_累计回测表现.png` | `图片/` |
