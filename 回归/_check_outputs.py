import pandas as pd
import os

files = [
    'panel_data.csv','desc_statistics.csv','correlation_matrix.csv',
    'dfi_yearly.csv','region_desc.csv','vif_table.csv','unit_root.csv',
    'full_regression_table.csv','first_stage_iv.csv','iv_stage2.csv',
    'heterogeneity_full.csv','subgroup_regression.csv',
    'results_summary.csv','results_full_table.csv'
]

print("=== 文件生成状态 ===")
for f in files:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"  {f:40s} {status}")

print()
rs = pd.read_csv('results_summary.csv', encoding='utf-8-sig')
print("=== 核心系数汇总 ===")
print(rs.to_string(index=False))

print()
print("=== 诊断检验 ===")
vif = pd.read_csv('vif_table.csv', encoding='utf-8-sig')
print(vif.to_string(index=False))
print()
ur = pd.read_csv('unit_root.csv', encoding='utf-8-sig')
print(ur.to_string(index=False))

print()
print("=== 分区域子样本 ===")
sg = pd.read_csv('subgroup_regression.csv', encoding='utf-8-sig')
print(sg.to_string(index=False))
