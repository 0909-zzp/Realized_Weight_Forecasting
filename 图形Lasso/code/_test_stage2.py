"""单日测试 process_one_day。"""
import sys; sys.path.insert(0, r"d:/HuaweiMoveData/Users/27438/Desktop/大创/图形Lasso/code")
sys.stdout.reconfigure(encoding="utf-8")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "step2", r"d:/HuaweiMoveData/Users/27438/Desktop/大创/图形Lasso/code/权重计算+描述性分析（步骤二）.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

r = mod.process_one_day((0, 2436, 5e-7))
print(f"success={r['success']} rpv={r.get('rpv', 'N/A'):.4e} density={r.get('density', 'N/A'):.4f} nz_edges={r.get('nz_edges', 'N/A')}")
w = r["w"]
print(f"w range=[{w.min():.4f},{w.max():.4f}] M={r['M']} sum(w)={w.sum():.6f}")
