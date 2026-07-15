import pandas as pd
import numpy as np

desktop = 'D:/HuaweiMoveData/Users/27438/Desktop'
df = pd.read_excel(f'{desktop}/笔记报表_补充后_monthly.xlsx', sheet_name=0)

# 活跃率相关列
rate_col_all = '去重后小红星站外活跃率（30日归因）'
uv_col_all = '去重后小红星站外活跃行为（30日归因）'
rate_col_rpt = '小红星站外活跃率(去重后30日归因)'
uv_col_rpt = '小红星站外活跃行为UV(去重后30日归因)'
readuv_col = '小红星任务期阅读UV'

for label, rate_col, uv_col, cost_col in [
    ('全部数据版', rate_col_all, uv_col_all, '去重后小红星站外活跃成本（30日归因）'),
    ('笔记报表版', rate_col_rpt, uv_col_rpt, '小红星站外活跃成本(去重后30日归因)'),
]:
    df_sub = df[[rate_col, uv_col, readuv_col, '展现量']].copy()
    for c in [rate_col, uv_col, readuv_col]:
        df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')
    df_sub = df_sub.dropna()
    n = len(df_sub)
    if n == 0:
        print(f'{label}: 无有效数据')
        continue
    rate_act = df_sub[uv_col] / df_sub[readuv_col]
    rate_imp = df_sub[uv_col] / df_sub['展现量']
    corr_readuv = rate_act.corr(df_sub[rate_col])
    corr_impress = rate_imp.corr(df_sub[rate_col])
    print(f'{label} (n={n}):')
    print(f'  活跃率 = 活跃UV/阅读UV  相关度: {corr_readuv:.4f}')
    print(f'  活跃率 = 活跃UV/展现量   相关度: {corr_impress:.4f}')
    print(f'  活跃率 mean={df_sub[rate_col].mean():.4f}, min={df_sub[rate_col].min():.4f}, max={df_sub[rate_col].max():.4f}')
    print(f'  阅读UV mean={df_sub[readuv_col].mean():.0f}, 活跃UV mean={df_sub[uv_col].mean():.0f}')
    print()

# 统计非空行
for c in [rate_col_all, rate_col_rpt]:
    n = pd.to_numeric(df[c], errors='coerce').notna().sum()
    print(f'{c}: {n}/{len(df)} 非空')
