## 最小实验：因子分解网络 → Table 3 指标

### 步骤
1. 生成因子分解adjacency → 计算A_bar_idio.npy（并行，约70分钟）
2. 只重训M4(Network VARX)和M6(DFL)（约20分钟）
3. 跑Table 3投资组合评估（约5分钟）
4. 输出对比表：236边 vs 12边的夏普/换手/波动

### 不改原文件
- 新A_bar另存为 `A_bar_idio.npy`
- 新预测另存为 `Y_pred_model4_idio.npy` 等
- 原有结果完全保留