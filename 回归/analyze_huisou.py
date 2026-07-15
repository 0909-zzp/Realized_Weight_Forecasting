import pandas as pd

df = pd.read_excel('D:/HuaweiMoveData/Users/27438/Desktop/笔记报表_补充后_v3.xlsx')

print('=== 列名列表 ===')
for i, c in enumerate(df.columns):
    print(f'  {i+1}. {c}')

print(f'\n总行数: {len(df)}')

print('\n=== 时间列分布 ===')
if '时间' in df.columns:
    time_counts = df['时间'].value_counts()
    print(time_counts.to_string())

print('\n=== 回搜率相关指标统计 ===')
rel_cols = ['时间', '展现量', '点击量', '回搜率', '回搜数量', '回搜成本', '点击率', '互动量', '消费']
for c in rel_cols:
    if c in df.columns:
        s = df[c].dropna()
        if s.dtype == 'object':
            print(f'{c}: {len(s)}非空, {s.nunique()}种取值')
        else:
            print(f'{c}: {len(s)}非空, min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}, std={s.std():.6f}')

print('\n=== 概念解释 ===')
print('回搜率 = 回搜数量 / 展现量 （每条笔记的回搜率）')
if '回搜数量' in df.columns and '展现量' in df.columns:
    df_check = df[['回搜数量', '展现量']].dropna()
    df_check['反算回搜率'] = df_check['回搜数量'] / df_check['展现量']
    print(f'  验证: 反算回搜率 vs 原始回搜率 相关性 = {df_check["反算回搜率"].corr(df.loc[df_check.index, "回搜率"]):.6f}')
