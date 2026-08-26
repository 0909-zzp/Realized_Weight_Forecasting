# Realized Weight Forecasting with Network-Regularized VARX

> Undergraduate Innovation Training Program · Jinan University  
> High-Dimensional Portfolio Modeling via Graphical Lasso & Decision-Focused Learning

---

## Overview

This project predicts **daily realized GMVP (Global Minimum Variance Portfolio) weights** for **392 S&P 500 stocks** using **1-minute intraday returns** (2008–2020, 2,436 trading days).

We combine three innovations:

1. **Graphical Lasso** (Friedman et al., 2008) stabilizes high-dimensional covariance estimation and constructs a sparse asset network.
2. **Network-regularized VARX** (Guo & Minca, 2022) predicts weights using lagged weights, macro variables, and network-guided sparsity.
3. **Decision-Focused Loss (DFL)** directly minimizes portfolio risk + turnover costs, not statistical prediction error.

---

## Project Structure

```
Realized_Weight_Forecasting/
│
├── 图形Lasso/code/               # Stage 1-2: GLasso + GMVP weights
│   ├── 共享模块.py                 # Central parameter source + shared utilities
│   ├── 纯权重计算.py               # Daily GLasso → weight + adjacency matrix
│   ├── λ的选择/                   # λ_Ω selection (optimal λ=3e-6)
│   └── 输出数据/                   # Output: reg_weights.csv, adjacency/*.npy
│
├── 特征工程/                       # Stage 3-A: Feature engineering
│   └── 特征工程.py                 # X(lagged+exog) + Y(target) + A_bar(network mask source)
│
├── VARX/                           # Stage 3-B: Model fitting + grid search
│   ├── VAR及拓展（table2）.py      # Main script — 7 models (FA Lasso + DFL + LSTM)
│   ├── 网格搜索.py                  # Hierarchical ablation parameter search
│   ├── 网格搜索规范.md              # Search experiment design document
│   ├── Table2_results.csv          # ★ Final Table 2
│   ├── final_params.json           # Confirmed optimal parameters
│   └── fitted_models/              # Trained model coefficients (regeneratable)
│
├── 性能评估与可视化/                 # Stage 4: Evaluation
│   ├── Table3_投资组合表现.py       # Portfolio performance (volatility, RPV, Sharpe, drawdown)
│   └── Table3_results.csv          # ★ Final Table 3
│
├── 消融分析/                        # Table 4: Ablation analysis
│   ├── Table4_消融分析.py            # Ablation script (M2→M3→M3a→M4)
│   └── Table4_results.csv           # ★ Final Table 4
│
├── 数据/                            # Data (excluded from git — see .gitignore)
│
├── readme/README.md                 # Full project manual (Chinese)
├── FILE_GUIDE.md                    # File directory guide
├── .codebuddy/memory/MEMORY.md      # Project memory (version history)
└── .gitignore
```

---

## Key Parameters (Grid Search Confirmed, 2026-07-18)

| Parameter | Value | Description |
|---|---|---|
| K | 392 | Asset count |
| λ_Ω | 3e-6 | GLasso regularization (density 62%) |
| P_LAGS | 3 | VARX lag order |
| λ₁ (LAMBDA_LASSO) | 3e-4 | Connected-asset lag L1 penalty (M3/M4/M5) |
| λ₁_M2 | 5e-4 | M2 independent λ₁ (P0: verified non-boundary) |
| λ₁_M3a | 4.5e-4 | M3a independent λ₁ (P1: self-lag exemption optimum) |
| λ₃ (LAMBDA_EXOG) | 5e-4 | Exogenous variable L1 (P2: optimal for full pipeline) |
| τ (NETWORK_THRESHOLD) | 0.7 | Network threshold (density ~26%) |
| λ_net (LAMBDA_NETWORK) | 1e-3 | Unconnected-asset additional L1 |
| λ_s (LAMBDA_TURNOVER) | 1e-3 | Turnover smoothing L2 |
| η (ETA) | 1e-4 | DFL transaction cost (10 bps) |
| ρ (RHO_DFL) | 1e-3 | DFL anchor strength |

---

## Results

### Table 2 — Out-of-Sample Prediction Accuracy (200d: 2018-12-28~2019-10-15)

| Model | MSE | DM Stat | MCS p-val | MCS Rank | 90% MCS |
|---|---|---|---|---|---|
| **Network VARX** | **2.30×10⁻⁵** | −17.34 | **1.0000** | 1 | ✅ |
| LSTM | 3.23×10⁻⁵ | −16.21 | 0.0000 | 6 | ❌ |
| Network+Smooth | 2.46×10⁻⁵ | −17.11 | 0.0000 | 5 | ❌ |
| Sparse VARX | 2.43×10⁻⁵ | −17.09 | 0.0000 | 3 | ❌ |
| Sparse VAR | 2.47×10⁻⁵ | −17.01 | 0.0000 | 4 | ❌ |
| Network VARX + Smooth + DFL | 2.64×10⁻⁵ | −16.30 | 0.0000 | 2 | ❌ |
| VAR (OLS) | 1.06×10⁻⁴ | — | 0.0000 | 7 | ❌ |

> MCS: Hansen-Lunde-Nason (2011) block bootstrap, 2,000 replications, block length 5 days.
> M4 is the only model with p=1.0 and the only model in the 75% and 90% MCS; all other models are eliminated at p≈0.

### Table 3 — Portfolio Performance

| Model | Volatility | RPV(ann) | Turnover | Sharpe | Max DD |
|---|---|---|---|---|---|
| Equal-Weight | 0.123 | 0.01263 | 0.010 | +1.08 | −8.9% |
| VAR | 0.121 | 0.01323 | 3.229 | +0.94 | −6.6% |
| Sparse VAR | 2.47×10⁻⁵ | −17.01 | 0.0000 | 4 | ❌ |
| Sparse VARX | 2.43×10⁻⁵ | −17.09 | 0.0000 | 3 | ❌ |
| Network VARX | 0.089 | 0.00723 | 0.368 | +1.59 | −5.7% |
| Network+Smooth | 2.46×10⁻⁵ | −17.11 | 0.0000 | 5 | ❌ |
| **Network+Smooth + DFL (L1 drift, η=1e-6, 滚动Σ150d)** | 0.075 | 0.00563 | 0.069 | **+2.00** | **−3.79%** |
| LSTM | 3.23×10⁻⁵ | −16.21 | 0.0000 | 6 | ❌ |
| Sample GMVP (W=20) | 0.073 | 0.00573 | 0.341 | +1.60 | −4.4% |
| Shrinkage GMVP (W=20) | 0.096 | 0.00911 | 0.371 | +1.09 | −7.5% |
| GLasso GMVP (W=20) | 0.105 | 0.00953 | 0.697 | +1.60 | −5.7% |

> DFL uses a drift-aware L1 turnover penalty with OOS-validated parameters (η=1e-6, ρ=1e-3) and a daily rolling 150-day covariance; it achieves the best net Sharpe among active models.

### Table 4 — Ablation Analysis (DFL baseline, top-down subtraction)

| Configuration | MSE_w | RPV(ann) | Turnover | Net Sharpe |
|---|---:|---:|---:|---:|
| Full model (M5+DFL) | 2.64×10⁻⁵ | 0.0056 | 0.0685 | +1.9989 |
| − Exogenous vars (M2+DFL) | 2.59×10⁻⁵ | 0.0056 | 0.0695 | +1.9824 |
| − Network penalty (M3a+DFL) | 2.63×10⁻⁵ | 0.0056 | 0.0698 | +2.0055 |
| − Smooth penalty (M4+DFL) | 2.54×10⁻⁵ | 0.0056 | 0.1026 | +1.9024 |
| − DFL (M5 only) | 2.46×10⁻⁵ | 0.0063 | 0.2720 | +1.6099 |

> DFL contributes the largest Sharpe gain (+0.3890); the smoothness penalty contributes (+0.0965), while the network penalty is negligible. MSE_w rises as DFL trades accuracy for utility.

---

## Setup

```bash
pip install numpy pandas scikit-learn pyreadr matplotlib scipy
```

Data: S&P 500 1-min intraday log returns (2008–2020) — 2,436 `.RData` files under `数据/1min_log_return/`.

## Reproduction

```bash
# Stage 2: GLasso daily GMVP weights
python 图形Lasso/code/纯权重计算.py

# Stage 3-A: Feature engineering
python 特征工程/特征工程.py

# Stage 3-B: Grid search + model fitting
python VARX/网格搜索.py
python "VARX/VAR及拓展（table2）.py"

# Stage 4: Portfolio evaluation
python 性能评估与可视化/Table3_投资组合表现.py
```

## Citation

If you use this code, please cite:

- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432–441.
- Guo, W., & Minca, A. (2022). Large vector autoregressive exogenous factor model with network regularization. *Journal of Network Theory in Finance*, 8(1), 1–25.
- Golosnoy, V., & Gribisch, B. (2022). Modeling and forecasting realized portfolio weights. *Journal of Banking & Finance*, 138, 106404.

## License

This project is part of the Jinan University Undergraduate Innovation Training Program. All rights reserved.
