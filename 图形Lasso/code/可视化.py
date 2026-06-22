# ===================================================================
# 阶段四：可视化
#
# 生成论文所需的全部图表：
#   1. λ-得分选择曲线
#   2. GMVP 权重热力图（全样本）
#   3. 网络密度与换手率时序
#   4. 组合方差累计时序
#   5. 回测表现对照图
# ===================================================================
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')               # 非交互后端，适合服务器/无GUI环境
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from 共享模块 import K, ETA, OUT_DIR, log, set_log_file

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
})


# ===================================================================
# 图1: λ-得分选择曲线
# ===================================================================
def plot_lambda_selection(diagnostics_path: Path, out_path: Path):
    """画 λ-验证得分曲线，标注最优。"""
    df = pd.read_csv(diagnostics_path)
    if 'lambda' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['lambda'], df['valid_score'], 'o-', color='#2E86AB', markersize=6, linewidth=2)

    best_idx = df['valid_score'].idxmin()
    ax.plot(df['lambda'].iloc[best_idx], df['valid_score'].iloc[best_idx],
            'ro', markersize=10, label=f"Optimal λ={df['lambda'].iloc[best_idx]:.1e}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('lambda')
    ax.set_ylabel('Validation Score')
    ax.set_title('Lambda Selection Curve')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    fig.savefig(out_path)
    plt.close(fig)


# ===================================================================
# 图2: GMVP 权重热力图
# ===================================================================
def plot_weight_heatmap(weights_path: Path, out_path: Path, n_days: int = 500):
    """画全样本权重热力图（采样显示）。"""
    df = pd.read_csv(weights_path, index_col=0)

    step = max(1, len(df) // n_days)
    w_sub = df.iloc[::step].values

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(w_sub.T, aspect='auto', cmap='RdBu_r', vmin=-0.02, vmax=0.02)
    ax.set_xlabel('Trading Day (sampled)')
    ax.set_ylabel('Asset')
    ax.set_title('GMVP Weights Heatmap')
    plt.colorbar(im, ax=ax, label='Weight')
    fig.savefig(out_path)
    plt.close(fig)


# ===================================================================
# 图3: 网络密度 + 换手率双轴时序
# ===================================================================
def plot_network_turnover(daily_path: Path, out_path: Path):
    """画网络密度和换手率的双轴时序图。"""
    df = pd.read_csv(daily_path)
    if 'date' not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = range(len(df))
    color1 = '#2E86AB'
    ax1.plot(x, df.get('network_density', [0]*len(df)), color=color1, linewidth=1.0)
    ax1.set_ylabel('Network Density', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.plot(x, df.get('turnover', [0]*len(df)), color=color2, linewidth=1.0, alpha=0.7)
    ax2.set_ylabel('Turnover', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    n_ticks = min(5, len(df))
    tick_positions = np.linspace(0, len(df) - 1, n_ticks, dtype=int)
    tick_labels = [str(df['date'].iloc[i]) for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)

    ax1.set_title('Network Density and Turnover Over Time')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===================================================================
# 图4: 回测表现对照
# ===================================================================
def plot_backtest_comparison(out_dir: Path):
    """画回测 GLasso-VARX vs 等权重累计方差曲线。"""
    pv_path = out_dir / "code" / "输出数据" / "backtest_port_variances.npy"
    eq_path = out_dir / "code" / "输出数据" / "backtest_eq_variances.npy"

    if not pv_path.exists() or not eq_path.exists():
        return

    gl_vars = np.load(pv_path)
    eq_vars = np.load(eq_path)
    min_len = min(len(gl_vars), len(eq_vars))

    gl_cum = np.cumsum(gl_vars[:min_len])
    eq_cum = np.cumsum(eq_vars[:min_len])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gl_cum, color='#2E86AB', linewidth=1.5, label='GLasso-VARX')
    ax.plot(eq_cum, color='#A23B72', linewidth=1.5, linestyle='--', label='Equal Weight')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative Realized Variance')
    ax.set_title('Cumulative Out-of-Sample Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "图4_累计回测表现.png")
    plt.close(fig)


# ===================================================================
# 独立运行入口
# ===================================================================
def main():
    set_log_file(OUT_DIR / "visualization_log.txt")
    log("=" * 60)
    log("  阶段四: 可视化")
    log("=" * 60)

    fig_dir = OUT_DIR / "图片"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 图1: λ-得分曲线
    diag_path = OUT_DIR / "lambda_selection_diagnostics.csv"
    if diag_path.exists():
        log("绘制 λ 选择曲线...")
        plot_lambda_selection(diag_path, fig_dir / "图1_lambda_selection.png")

    # 图2: 权重热力图
    weights_path = OUT_DIR / "code" / "输出数据" / "reg_weights_2436.csv"
    if weights_path.exists():
        log("绘制权重热力图...")
        plot_weight_heatmap(weights_path, fig_dir / "图2_weight_heatmap.png")

    # 图3: 网络密度+换手率
    daily_path = OUT_DIR / "code" / "输出数据" / "每日的描述性数据.csv"
    if daily_path.exists():
        log("绘制网络密度+换手率...")
        plot_network_turnover(daily_path, fig_dir / "图3_network_turnover.png")

    # 图4: 回测对照
    log("绘制回测对照...")
    plot_backtest_comparison(OUT_DIR)

    log(f"\n图片已保存至: {fig_dir}")


if __name__ == "__main__":
    main()
