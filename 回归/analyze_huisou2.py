import pandas as pd
import numpy as np

df = pd.read_excel('D:/HuaweiMoveData/Users/27438/Desktop/笔记报表_补充后_v3.xlsx')

# 筛选有回搜率的行
df_hr = df[df['回搜率'].notna()].copy()

# 按时间分组，对比不同加权方式的综合回搜率
print('=== 各时间段综合回搜率：不加权 vs 不同权重对比 ===\n')

# 限定至少有3条笔记的时间段
grouped = df_hr.groupby('时间').filter(lambda g: len(g) >= 3)
groups = grouped.groupby('时间')

results = []
for name, g in groups:
    n = len(g)
    # 不加权（简单平均）
    simple_avg = g['回搜率'].mean()
    # 展现量加权
    imp_weighted = np.average(g['回搜率'], weights=g['展现量'])
    # 消费加权
    spend_weighted = np.average(g['回搜率'], weights=g['消费'].clip(lower=0.01))
    # 点击量加权
    click_weighted = np.average(g['回搜率'], weights=g['点击量'].clip(lower=0.01))
    # 回搜数量加权（相当于 总回搜/总展现）
    total_rate = g['回搜数量'].sum() / g['展现量'].sum()
    # 总消费（万元）
    total_spend = g['消费'].sum()
    results.append({
        '时间': name, '笔记数': n, '总消费': total_spend,
        '简单平均': simple_avg, '展现量加权': imp_weighted,
        '消费加权': spend_weighted, '点击量加权': click_weighted,
        '总回搜/总展现': total_rate
    })

result_df = pd.DataFrame(results).sort_values('时间')
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
print(result_df.to_string(index=False))

print('\n\n=== 各加权方式的相关性矩阵 ===')
corr_cols = ['简单平均', '展现量加权', '消费加权', '点击量加权', '总回搜/总展现']
corr = result_df[corr_cols].corr()
print(corr.to_string())

print('\n\n=== 结论 ===')
print('1. "总回搜/总展现" 等价于以展现量为权重，是标准做法')
print('2. 展现量加权 ≈ 总回搜/总展现（理论上精确相等）')
print('3. 不加权简单平均会严重偏误：小展现量笔记的回搜率噪声大，却获得与爆款笔记同等的权重')
print('4. 推荐权重：展现量（最直接）或 消费（反映投入权重）')
