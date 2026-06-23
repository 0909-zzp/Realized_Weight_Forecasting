# ===================================================================
# R 版本：GLasso GMVP 权重计算（阶段二备份方案）
# 路径已修正为当前项目结构，可直接运行
# ===================================================================

# ---- 参数（与 Python 共享模块完全对齐） ----
data_dir <- "d:/HuaweiMoveData/Users/27438/Desktop/大创/数据/1min_log_return"
lambda <- 5e-7           # λ_Ω（与 Python 一致）
eps_ridge <- 1e-4         # EPS_RIDGE（与 Python 一致）
MAX_ITER <- 150           # maxit（与 Python 优化后一致，原5000）
TOL <- 1e-4               # thr（与 Python TOL_GLASSO 一致，原1e-6）
K <- 392

# ---- 输出目录 ----
out_dir <- "d:/HuaweiMoveData/Users/27438/Desktop/大创/图形Lasso/code/输出数据"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---- 文件列表 ----
files <- list.files(data_dir, pattern = "_1min_log_return.RData$", full.names = TRUE)
files <- sort(files)
n_days <- min(2436, length(files))       # 修正：2426 → 2436
cat("将处理", n_days, "个交易日\n")

# ---- 加载包 ----
if (!require(glasso)) install.packages("glasso")
library(glasso)

# ---- 存储 ----
reg_weights <- matrix(NA, n_days, K)
cond_vals <- numeric(n_days)
prec_last <- NULL

# ---- 主循环 ----
for (day in 1:n_days) {
  load(files[day])
  if (!exists("rett1")) stop(paste("Day", day, ": rett1 不存在"))
  
  returns <- t(rett1)                           # (M=390, K=392)
  cov_raw <- crossprod(returns)                 # X^T X，论文公式(3)
  cov_ridge <- cov_raw
  diag(cov_ridge) <- diag(cov_ridge) + eps_ridge # Ridge 退避
  
  gl <- glasso(cov_ridge, rho = lambda, maxit = MAX_ITER, thr = TOL)
  reg_cov <- gl$w                              # 平滑后的协方差估计
  ones <- rep(1, K)
  w <- solve(reg_cov) %*% ones                 # Θ·1
  w <- w / sum(w)                              # 归一化 = GMVP 权重，论文公式(5)
  
  reg_weights[day, ] <- w
  cond_vals[day] <- kappa(cov_raw, exact = TRUE)  # 基于原始协方差（对齐Python）
  prec_last <- gl$wi                             # 始终保留，循环结束即为最后一天
  
  rm(rett1, returns, cov_raw, cov_ridge, gl, reg_cov, w)
  gc()
  
  if (day %% 50 == 0) cat("  ", day, "/", n_days, "\n")
}

# ---- 保存 ----
write.csv(reg_weights, file.path(out_dir, "reg_weights_2436_R.csv"), row.names = FALSE)
write.csv(data.frame(cond = cond_vals), file.path(out_dir, "reg_cond_2436_R.csv"), row.names = FALSE)
write.csv(prec_last, file.path(out_dir, "prec_last_2436_R.csv"), row.names = FALSE)

cat("处理完成！结果保存在:", out_dir, "\n")
