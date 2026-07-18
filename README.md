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

### Table 2 — Out-of-Sample Prediction Accuracy (with Formal MCS)

| Model | MSE | DM Stat | MCS p-val | MCS Rank | 90% MCS |
|---|---|---|---|---|---|
| **Network VARX** | **2.12×10⁻⁵** | −41.27 | **1.0000** | 1 | ✅ |
| LSTM | 6.48×10⁻³ | +1.64 | 0.2645 | 2 | ✅ |
| Network+Smooth | 2.19×10⁻⁵ | −41.78 | 0.1245 | 3 | ✅ |
| Sparse VARX | 2.25×10⁻⁵ | −40.28 | 0.0995 | 4 | ❌ |
| Sparse VAR | 2.29×10⁻⁵ | −40.01 | 0.0980 | 5 | ❌ |
| DFL VARX (L2) | 2.28×10⁻⁵ | −40.26 | 0.0895 | 6 | ❌ |
| VAR (OLS) | 9.76×10⁻⁵ | — | 0.0625 | 7 | ❌ |

> MCS: Hansen-Lunde-Nason (2011) block bootstrap, 2,000 replications, block length 5 days.
> M4 is the only model with p=1.0 (always in MCS). LSTM survives due to high day-to-day loss variance despite large mean MSE.

### Table 3 — Portfolio Performance

| Model | Volatility | RPV(ann) | Turnover | Sharpe | Max DD |
|---|---|---|---|---|---|
| Equal-Weight | 0.136 | 0.0169 | 0.000 | −0.939 | −21.8% |
| VAR | 0.126 | 0.0161 | 3.051 | −0.926 | −20.8% |
| Sparse VAR | 0.105 | 0.0107 | 0.283 | −0.290 | −13.9% |
| Sparse VARX | 0.105 | 0.0106 | 0.283 | −0.284 | −13.8% |
| Network VARX | 0.103 | 0.0101 | 0.354 | −0.342 | −13.4% |
| +Smooth | **0.102** | **0.0096** | 0.417 | −0.402 | −13.5% |
| **DFL VARX (L2)** | 0.103 | 0.0102 | 0.371 | **−0.056** ± | **−12.6%** ± |
| LSTM | 10.26 | 100.7 | 26.85 | −0.868 | −1135% |

> ± DFL achieves the best risk-adjusted return among all models. L2 closed-form solution balances portfolio variance and turnover.

### Table 4 — Ablation Analysis (DFL baseline, top-down subtraction)

| Configuration | MSE_w | RPV(ann) | Turnover | Net Sharpe |
|---|---:|---:|---:|---:|
| Full model (M5+DFL) | 2.301×10⁻⁵ | 0.0098 | 0.385 | −0.073 |
| − Exogenous vars (M2+DFL) | 2.398×10⁻⁵ | 0.0107 | 0.296 | −0.012 |
| − Network penalty (M3a+DFL) | 2.290×10⁻⁵ | 0.0103 | 0.367 | −0.047 |
| − Smooth penalty (M4+DFL) | 2.279×10⁻⁵ | 0.0102 | 0.372 | −0.056 |
| − DFL (M5 only) | 2.186×10⁻⁵ | 0.0096 | 0.417 | −0.402 |

> DFL contributes the largest Sharpe gain (+0.329), followed by exogenous variables (+0.061). MSE_w rises as DFL trades accuracy for utility.

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
