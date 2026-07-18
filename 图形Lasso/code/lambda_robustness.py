"""λ_Ω 稳健性分析 — Comment 2 回应
对多个 λ_Ω 值重跑权重计算→特征工程→Table2, 检查模型排名稳定性。
"""
import os, sys, shutil, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 大创/
LAMBDAS = [1e-6, 1e-5]  # 极端值: 最密/最稀, 加上已有3e-6即可对比
OUT_DIR = ROOT / "lambda_robustness"
OUT_DIR.mkdir(exist_ok=True)

WEIGHT_SCRIPT = Path(__file__).parent / "weights_minimal.py"  # 单进程,防OOM
FEAT_SCRIPT   = ROOT / "特征工程" / "特征工程.py"
TABLE2_SCRIPT = ROOT / "VARX" / "VAR及拓展（table2）.py"

def run_weights(lam_val, idx):
    """运行最小权重计算（单进程, 命令行参数传入 λ）。"""
    out_sub = OUT_DIR / f"lam_{idx}_{lam_val:.0e}"
    out_sub.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  λ_Ω = {lam_val:.0e}  — 权重计算 (minimal)")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(WEIGHT_SCRIPT), str(lam_val)],
                          cwd=str(WEIGHT_SCRIPT.parent),
                          capture_output=False, timeout=14400)
    elapsed = time.time() - t0
    print(f"  完成: {elapsed/60:.0f}min")
    return result.returncode == 0

def run_features(lam_val, idx):
    """用当前权重重新跑特征工程。"""
    out_sub = OUT_DIR / f"lam_{idx}_{lam_val:.0e}"
    
    # 复制权重文件到特征工程能读的位置
    src_weights = out_sub / "reg_weights_2436.csv"
    dst_weights = ROOT / "特征工程" / "X_features.npy"  # 只复制，特征工程会读csv
    
    # 特征工程从 图形Lasso/code/输出数据/ 读取权重
    # 所以需要临时替换
    orig_weight_dir = WEIGHT_SCRIPT.parent / "输出数据"
    orig_weight_file = orig_weight_dir / "reg_weights_2436.csv"
    
    # 备份原权重
    if orig_weight_file.exists():
        shutil.move(str(orig_weight_file), str(orig_weight_file) + ".bak")
    
    try:
        shutil.copy2(str(src_weights), str(orig_weight_file))
        
        print(f"\n{'='*60}")
        print(f"  λ_Ω = {lam_val:.0e}  — 特征工程")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, str(FEAT_SCRIPT)],
                              cwd=str(FEAT_SCRIPT.parent),
                              capture_output=False, timeout=600)
        
        # 复制特征输出
        feat_out = ROOT / "特征工程"
        for f in feat_out.glob("*.npy"):
            shutil.copy2(f, out_sub / f.name)
        for f in feat_out.glob("*.csv"):
            shutil.copy2(f, out_sub / f.name)
        for f in feat_out.glob("*.txt"):
            shutil.copy2(f, out_sub / f.name)
        return True
    except subprocess.TimeoutExpired:
        print(f"  ✗ 超时")
        return False
    finally:
        if (str(orig_weight_file) + ".bak" and 
            Path(str(orig_weight_file) + ".bak").exists()):
            shutil.move(str(orig_weight_file) + ".bak", str(orig_weight_file))

def run_table2(lam_val, idx):
    """用当前特征跑 Table 2。"""
    out_sub = OUT_DIR / f"lam_{idx}_{lam_val:.0e}"
    
    print(f"\n{'='*60}")
    print(f"  λ_Ω = {lam_val:.0e}  — Table 2")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(TABLE2_SCRIPT)],
                          cwd=str(TABLE2_SCRIPT.parent),
                          capture_output=False, timeout=1800)
    
    # 复制 Table 2 输出
    varx_dir = ROOT / "VARX"
    for f in varx_dir.glob("Table2_results.csv"):
        shutil.copy2(f, out_sub / f.name)
    for f in varx_dir.glob("Y_pred_model*.npy"):
        shutil.copy2(f, out_sub / f.name)
    return result.returncode == 0

def main():
    print("λ_Ω 稳健性分析 — Comment 2")
    print(f"λ 候选: {[f'{l:.0e}' for l in LAMBDAS]}")
    print(f"输出: {OUT_DIR}")
    
    results_summary = []
    
    for i, lam in enumerate(LAMBDAS):
        idx = i + 1
        print(f"\n{'#'*70}")
        print(f"### λ_Ω[{idx}/{len(LAMBDAS)}] = {lam:.0e}")
        print(f"{'#'*70}")
        
        ok1 = run_weights(lam, idx)
        if not ok1:
            print(f"  ✗ 权重计算失败，跳过后续")
            continue
        
        ok2 = run_features(lam, idx)
        if not ok2:
            print(f"  ✗ 特征工程失败，跳过后续")
            continue
        
        ok3 = run_table2(lam, idx)
        
        # 读取结果
        try:
            import pandas as pd
            df = pd.read_csv(OUT_DIR / f"lam_{idx}_{lam:.0e}" / "Table2_results.csv")
            m4_mse = df[df['Model']==4]['MSE'].values[0]
            results_summary.append({
                'lambda': lam,
                'M4_MSE': m4_mse,
            })
            print(f"  M4 MSE = {m4_mse:.4e}")
        except:
            pass
    
    # 汇总
    print(f"\n{'='*60}")
    print("稳健性汇总")
    print(f"{'='*60}")
    print(f"{'λ_Ω':>10}  {'M4_MSE':>12}")
    print("-"*30)
    for r in results_summary:
        print(f"{r['lambda']:>10.0e}  {r['M4_MSE']:>12.4e}")
    
    import pandas as pd
    pd.DataFrame(results_summary).to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    print(f"\n保存: {OUT_DIR / 'robustness_summary.csv'}")

if __name__ == "__main__":
    main()
