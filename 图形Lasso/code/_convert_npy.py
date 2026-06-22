# ===================================================================
# _convert_npy.py — 将 .RData 批量转换为 .npy（加速 I/O）
#
# 用法：python code/_convert_npy.py
#
# 输出：数据/1min_log_return_npy/ （2436个 .npy 文件，约 2.5GB）
# 完成后 共享模块.py 自动优先读取 .npy，无需手动切换。
# ===================================================================
import os, sys, time
import numpy as np
import pyreadr
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# 路径
_CODE = Path(__file__).resolve().parent
_PROJ = _CODE.parent.parent
SRC_DIR = _PROJ / "数据" / "1min_log_return"
DST_DIR = _PROJ / "数据" / "1min_log_return_npy"
DST_DIR.mkdir(parents=True, exist_ok=True)

# 文件列表（与共享模块过滤逻辑一致）
all_items = sorted(os.listdir(SRC_DIR))
rdata_files = [f for f in all_items
               if f.endswith("_1min_log_return.RData") and not f.startswith("1min")]

print(f"源文件: {len(rdata_files)} 个 .RData")
print(f"目标目录: {DST_DIR}")
print(f"预计占用: {len(rdata_files) * 392 * 390 * 8 / 1024 / 1024:.0f} MB")
print("=" * 50)

t0 = time.time()
for i, f in enumerate(rdata_files):
    src_path = SRC_DIR / f
    # .npy 文件名：去掉 .RData 后缀
    dst_name = f.replace(".RData", ".npy")
    dst_path = DST_DIR / dst_name

    # 跳过已转换
    if dst_path.exists():
        continue

    try:
        result = pyreadr.read_r(str(src_path))
        rett = result["rett1"].values
        np.save(dst_path, rett)
    except Exception as e:
        print(f"  ✗ [{i}] {f}: {e}")
        continue

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(rdata_files) - i - 1)
        print(f"  {i+1}/{len(rdata_files)}  "
              f"耗时:{elapsed:.0f}s  ETA:{eta:.0f}s")

elapsed = time.time() - t0
n_done = len(list(DST_DIR.glob("*.npy")))
print(f"\n完成: {n_done}/{len(rdata_files)} 个文件")
print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"目标目录: {DST_DIR}")
print(f"\n共享模块.py 将自动检测并优先使用 .npy 格式。")
