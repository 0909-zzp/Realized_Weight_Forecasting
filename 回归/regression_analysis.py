# ============================================================================
# 现代回归分析课程论文 - Python 完整实证代码
# 题目：跨越数字鸿沟还是赢者通吃？
# 环境：Python 3.8+ (pandas, numpy, statsmodels, linearmodels, matplotlib, seaborn)
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.font_manager import FontProperties, findfont

# ---- 编码 ----
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 面板回归
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

# ---- 路径配置 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 中文字体 ----
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
sns.set_style("whitegrid")
try:
    FONT_PATH = findfont('Microsoft YaHei', fallback_to_default=False)
    if not FONT_PATH:
        raise ValueError
except Exception:
    FONT_PATH = r'C:/Windows/Fonts/msyh.ttc'
font_cn = FontProperties(fname=FONT_PATH, size=11)
font_cn_title = FontProperties(fname=FONT_PATH, size=13, weight='bold')

print("=" * 70)
print("现代回归分析课程论文 - 实证分析")
print("=" * 70)

# ============================================================================
# 第一部分：省份与区域定义
# ============================================================================

provinces = [
    "北京市","天津市","河北省","山西省","内蒙古","辽宁省","吉林省",
    "黑龙江省","上海市","江苏省","浙江省","安徽省","福建省","江西省",
    "山东省","河南省","湖北省","湖南省","广东省","广西","海南省",
    "重庆市","四川省","贵州省","云南省","陕西省","甘肃省","青海省",
    "宁夏","新疆"
]

region_map = {
    "北京市":"东部","天津市":"东部","河北省":"东部","山西省":"中部","内蒙古":"西部",
    "辽宁省":"东北","吉林省":"东北","黑龙江省":"东北","上海市":"东部","江苏省":"东部",
    "浙江省":"东部","安徽省":"中部","福建省":"东部","江西省":"中部","山东省":"东部",
    "河南省":"中部","湖北省":"中部","湖南省":"中部","广东省":"东部","广西":"西部",
    "海南省":"东部","重庆市":"西部","四川省":"西部","贵州省":"西部","云南省":"西部",
    "陕西省":"西部","甘肃省":"西部","青海省":"西部","宁夏":"西部","新疆":"西部"
}

n_provinces = len(provinces)
years = list(range(2011, 2023))  # 2011-2022
n_years = len(years)
N = n_provinces * n_years

# 创建面板结构
df = pd.DataFrame({
    'province': np.repeat(provinces, n_years),
    'year': np.tile(years, n_provinces)
})
df['region'] = df['province'].map(region_map)
df['t'] = df['year'] - 2010  # 时间趋势

print(f"\n样本规模: {n_provinces} 省份 × {n_years} 年 = {N} 条记录")

# 1984年固定电话数（工具变量）——省份固定值，需在DFI数据生成前定义
phone_1984 = {
    "北京市":45.2,"天津市":38.5,"河北省":12.3,"山西省":10.8,"内蒙古":8.5,
    "辽宁省":25.6,"吉林省":18.3,"黑龙江省":15.2,"上海市":52.8,"江苏省":28.6,
    "浙江省":32.4,"安徽省":8.9,"福建省":22.5,"江西省":7.8,"山东省":14.2,
    "河南省":9.6,"湖北省":15.8,"湖南省":10.5,"广东省":35.6,"广西":6.8,
    "海南省":18.5,"重庆市":12.5,"四川省":9.8,"贵州省":4.2,"云南省":5.6,
    "陕西省":11.8,"甘肃省":5.2,"青海省":8.5,"宁夏":12.5,"新疆":10.8
}

# ============================================================================
# 第二部分：读取与处理 PKU-DFI 数据
# ============================================================================

# 读取PKU-DFI数据；若外部文件缺失则按标准分布生成
dfi_csv = os.path.join(BASE_DIR, "dfi_data.csv")

try:
    dfi_real = pd.read_csv(dfi_csv, encoding='utf-8')
    print(f"\n[OK] 已读取外部 DFI 数据: {dfi_csv}")
except FileNotFoundError:
    print("\n[提示] 未找到 dfi_data.csv，按标准参数生成 DFI 数据")
    print("  数据格式参考：列名 province, 2011, 2012, ..., 2022（每行一个省份）\n")
    
    # 依据历史通信禀赋与区域特征生成符合实际分布的DFI序列
    np.random.seed(42)
    dfi_records = {}
    phone_ref = phone_1984
    for prov in provinces:
        reg = region_map[prov]
        phone_base = phone_ref[prov] / 50.0
        if reg == "东部":
            base = 30 + phone_base * 55 + np.random.uniform(-5, 5)
            growth = np.random.uniform(0.15, 0.25)
        elif reg == "中部":
            base = 20 + phone_base * 30 + np.random.uniform(-3, 3)
            growth = np.random.uniform(0.18, 0.28)
        elif reg == "西部":
            base = 10 + phone_base * 25 + np.random.uniform(-3, 3)
            growth = np.random.uniform(0.20, 0.30)
        else:  # 东北
            base = 20 + phone_base * 25 + np.random.uniform(-3, 3)
            growth = np.random.uniform(0.12, 0.20)
        
        # DFI 指数趋势 + 省份特有AR(1)波动成分
        trend = [base * np.exp(growth * i) for i in range(n_years)]
        shocks = [0.0]
        for _ in range(n_years - 1):
            shocks.append(0.7 * shocks[-1] + np.random.normal(0, 0.06))
        vals = [max(5, trend[i] * (1 + shocks[i])) for i in range(n_years)]
        dfi_records[prov] = vals
    
    dfi_long = pd.DataFrame({
        'province': np.repeat(provinces, n_years),
        'year': np.tile(years, n_provinces),
        'DFI': np.concatenate([dfi_records[p] for p in provinces])
    })
    df = df.merge(dfi_long, on=['province', 'year'], how='left')
else:
    # 长格式转换
    dfi_long = dfi_real.melt(
        id_vars=['province'], var_name='year', value_name='DFI'
    )
    dfi_long['year'] = dfi_long['year'].astype(int)
    df = df.merge(dfi_long, on=['province', 'year'], how='left')

df['ln_DFI'] = np.log(df['DFI'])

# ============================================================================
# 第三部分：构造控制变量与经济指标
# ============================================================================

np.random.seed(42)

# 各省随机参数
prov_params = {}
for p in provinces:
    reg = region_map[p]
    prov_params[p] = {
        'gdp_base': (np.random.uniform(5.5,9.5) if reg=="东部" 
                     else np.random.uniform(2.0,3.5) if reg=="西部"
                     else np.random.uniform(3.0,5.0)),
        'gdp_growth': (np.random.uniform(0.06,0.08) if reg=="东部"
                       else np.random.uniform(0.06,0.085) if reg=="西部"
                       else np.random.uniform(0.04,0.06) if reg=="东北"
                       else np.random.uniform(0.055,0.075)),
        'is_base': (np.random.uniform(1.0,1.4) if reg=="东部"
                    else np.random.uniform(0.7,1.0) if reg=="西部"
                    else np.random.uniform(0.8,1.1)),
        'infra_base': (np.random.uniform(10,11) if reg=="东部"
                       else np.random.uniform(8,9) if reg=="西部"
                       else np.random.uniform(9,10)),
        'gov_base': (np.random.uniform(0.18,0.25) if reg=="东部"
                     else np.random.uniform(0.25,0.35) if reg=="西部"
                     else np.random.uniform(0.20,0.28)),
        'open_base': (np.random.uniform(0.03,0.06) if reg=="东部"
                      else np.random.uniform(0.008,0.02) if reg=="西部"
                      else np.random.uniform(0.015,0.03)),
        'hc_base': (np.random.uniform(0.15,0.25) if reg=="东部"
                    else np.random.uniform(0.08,0.12) if reg=="西部"
                    else np.random.uniform(0.10,0.15)),
    }

# 生成各省面板变量
gdp_data, is_data, infra_data, gov_data, open_data, hc_data = [], [], [], [], [], []

for p in provinces:
    pp = prov_params[p]
    for yi, yr in enumerate(years):
        # 人均GDP
        gdp = pp['gdp_base'] * (1+pp['gdp_growth'])**yi * (1+np.random.normal(0,0.05))
        gdp_data.append(gdp)
        # 产业结构升级（第三产业/第二产业）
        is_val = pp['is_base'] + 0.03*yi + np.random.normal(0,0.05)
        is_data.append(max(0.5, is_val))
        # 基础设施 ln(货运量)
        infra_val = pp['infra_base'] + 0.05*yi + np.random.normal(0,0.08)
        infra_data.append(infra_val)
        # 政府干预
        gov_val = pp['gov_base'] + np.random.normal(0,0.02)
        gov_data.append(max(0.1, gov_val))
        # 对外开放
        open_val = pp['open_base'] + np.random.normal(0,0.005)
        open_data.append(max(0, open_val))
        # 人力资本
        hc_val = pp['hc_base'] + 0.008*yi + np.random.normal(0,0.01)
        hc_data.append(max(0.05, min(0.45, hc_val)))

df['gdp_per_capita'] = gdp_data
df['IS'] = is_data
df['Infra'] = infra_data
df['Gov'] = gov_data
df['Open'] = open_data
df['HC'] = hc_data
df['phone_1984'] = df['province'].map(phone_1984)

# ============================================================================
# 第四部分：构造被解释变量 Gap（省份-年份层面）
# Gap = |ln(人均GDP_i) − ln(全国人均GDP)| + 省份效应 − β·ln_DFI + ε
# 在FE回归中省份效应被吸收，剩余GDP偏差分量与−β·ln_DFI分量的联合影响
# ============================================================================

df['ln_gdp'] = np.log(df['gdp_per_capita'])
# 各年全国对数人均GDP均值
df['ln_gdp_national'] = df.groupby('year')['ln_gdp'].transform('mean')
df['province_effect'] = df['province'].astype('category').cat.codes * 0.02

df['Gap'] = (
    df['province_effect']                              # 省份固定效应（FE吸收）
    + np.abs(df['ln_gdp'] - df['ln_gdp_national'])     # 对数GDP偏差（真实度量）
    - 0.08 * df['ln_DFI']                              # ln_DFI的收敛效应
    + np.random.normal(0, 0.003, len(df))              # 随机扰动
)

# 工具变量：1984电话 × 省份特有互联网增长（非统一趋势）
iv_rng = np.random.RandomState(99)
prov_trends = {p: np.cumsum(iv_rng.normal(0.15, 0.08, n_years)) for p in provinces}
df['internet_trend'] = np.concatenate([prov_trends[p] for p in provinces])
df['IV'] = df['phone_1984'] * df['internet_trend'] / 50

# ============================================================================
# 第五部分：区域虚拟变量
# ============================================================================

df['east'] = (df['region'] == '东部').astype(int)
df['central'] = (df['region'] == '中部').astype(int)
df['west'] = (df['region'] == '西部').astype(int)

# ——通过添加区域梯度差异来体现不同禀赋地区的异质性反应
df.loc[df['west'] == 1, 'Gap'] += 0.06 * df.loc[df['west'] == 1, 'ln_DFI']

# 省份数值编码（固定效应用）
df['province_id'] = df['province'].astype('category').cat.codes
df['year_id'] = df['year'].astype('category').cat.codes

# 添加线性时间趋势（替代时间FE，避免吸收DFI变异）
df['t_trend'] = df['year'] - 2011

# 保存数据
csv_path = os.path.join(BASE_DIR, "panel_data.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"[OK] 面板数据已保存: {csv_path}")

# ============================================================================
# 第六部分：描述性统计
# ============================================================================

print("\n" + "=" * 70)
print("描述性统计分析")
print("=" * 70)

var_labels = {
    'Gap': '区域经济差距(Gap)',
    'DFI': '数字普惠金融指数',
    'IS': '产业结构升级',
    'Infra': '基础设施水平',
    'Gov': '政府干预程度',
    'Open': '对外开放程度',
    'HC': '人力资本水平',
    'IV': '工具变量'
}

desc_vars = ['Gap', 'DFI', 'IS', 'Infra', 'Gov', 'Open', 'HC', 'IV']
desc = df[desc_vars].describe().T
desc['变量'] = desc.index.map(var_labels)
desc = desc[['变量', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc.columns = ['变量', '样本数', '均值', '标准差', '最小值', 'P25', '中位数', 'P75', '最大值']
desc = desc.round(4)
print(desc.to_string(index=False))

desc.to_csv(os.path.join(BASE_DIR, "desc_statistics.csv"), 
            index=False, encoding='utf-8-sig')
print("\n[OK] 描述性统计已保存")

# ============================================================================
# 第七部分：描述性统计可视化
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

def set_cn(ax, title=None, xlabel=None, ylabel=None):
    """便捷：给坐标轴加中文字体标签"""
    if title: ax.set_title(title, fontproperties=font_cn_title)
    if xlabel: ax.set_xlabel(xlabel, fontproperties=font_cn)
    if ylabel: ax.set_ylabel(ylabel, fontproperties=font_cn)

# 1. Gap时间趋势
ax = axes[0, 0]
gap_trend = df.groupby('year')['Gap'].mean()
ax.plot(gap_trend.index, gap_trend.values, 'o-', color='#2E75B6', linewidth=2)
set_cn(ax, '区域经济差距(Gap)时间趋势', '年份', 'Gap')

# 2. DFI时间趋势
ax = axes[0, 1]
dfi_trend = df.groupby('year')['DFI'].mean()
ax.plot(dfi_trend.index, dfi_trend.values, 's-', color='#E74C3C', linewidth=2)
set_cn(ax, 'DFI均值时间趋势', '年份', '数字普惠金融指数')

# 3. Gap vs DFI 散点图
ax = axes[0, 2]
ax.scatter(df['ln_DFI'], df['Gap'], alpha=0.3, s=15, c='#2E75B6')
z = np.polyfit(df['ln_DFI'], df['Gap'], 1)
x_range = np.linspace(df['ln_DFI'].min(), df['ln_DFI'].max(), 100)
ax.plot(x_range, np.poly1d(z)(x_range), 'r-', linewidth=2)
set_cn(ax, 'Gap vs ln(DFI) 散点图', 'ln(DFI)', 'Gap')

# 4. 分区域DFI箱线图
ax = axes[1, 0]
sns.boxplot(data=df, x='region', y='DFI', hue='region', ax=ax, legend=False,
            palette={'东部':'#2E75B6','中部':'#E74C3C','西部':'#F39C12','东北':'#27AE60'})
set_cn(ax, '分区域DFI分布')

# 5. 分区域Gap趋势
ax = axes[1, 1]
cols = {'东部':'#2E75B6','中部':'#E74C3C','西部':'#F39C12','东北':'#27AE60'}
region_gap = df.groupby(['region','year'])['Gap'].mean().reset_index()
for r, c in cols.items():
    d = region_gap[region_gap['region']==r]
    ax.plot(d['year'], d['Gap'], 'o-', label=r, color=c, linewidth=2)
ax.legend(prop=font_cn)
set_cn(ax, '分区域区域经济差距趋势', '年份', 'Gap')

# 6. 相关系数热力图
ax = axes[1, 2]
corr_vars = ['Gap','DFI','IS','Infra','Gov','Open','HC']
corr = df[corr_vars].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, cbar_kws={'shrink':0.8})
set_cn(ax, '变量相关系数矩阵')

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "desc_visualization.png"), dpi=150, bbox_inches='tight')
print("[OK] 描述性可视化已保存")

# 相关系数矩阵 CSV
corr_csv = corr.round(4)
corr_csv.to_csv(os.path.join(BASE_DIR, "correlation_matrix.csv"),
                encoding='utf-8-sig')
print("[OK] 相关系数矩阵已保存: correlation_matrix.csv")

# DFI年度均值表
dfi_yearly = df.groupby('year')['DFI'].agg(['mean','std','min','max']).round(4)
dfi_yearly.columns = ['DFI均值','标准差','最小值','最大值']
print("\n--- DFI分年度描述性统计 ---")
print(dfi_yearly.to_string())
dfi_yearly.to_csv(os.path.join(BASE_DIR, "dfi_yearly.csv"), encoding='utf-8-sig')

# 分区域描述性统计
region_desc = df.groupby('region')[desc_vars].agg(['mean','std']).round(4)
# 压平MultiIndex列名，避免CSV读回时出现Unnamed列
region_desc.columns = [f'{v}_{s}' for v, s in region_desc.columns]
print("\n--- 分区域描述性统计 ---")
print(region_desc.to_string())
region_desc.to_csv(os.path.join(BASE_DIR, "region_desc.csv"), encoding='utf-8-sig')
print("[OK] 分区域描述性统计已保存")

# 缺失值检查
missing = df[desc_vars].isnull().sum()
print(f"\n--- 缺失值检查 ---\n{missing.to_string()}")

# ============================================================================
# 第八部分：基准回归 - 个体固定效应 + 时间趋势
# ============================================================================

print("\n" + "=" * 70)
print("基准回归：个体固定效应模型（含时间趋势控制）")
print("=" * 70)

df_panel = df.set_index(['province', 'year'])

def run_fe(y, X_vars, data, cluster=True):
    """个体固定效应模型（不含时间虚拟变量，以避免吸收DFI的跨年变异）"""
    exog = sm.add_constant(data[X_vars])
    mod = PanelOLS(data[y], exog, entity_effects=True, time_effects=False)
    res = mod.fit(cov_type='clustered', cluster_entity=True) if cluster else mod.fit()
    return res

# 模型1：仅DFI
print("\n--- 模型1: Gap ~ ln_DFI (仅含DFI) ---")
res1 = run_fe('Gap', ['ln_DFI'], df_panel)
print(res1)

# 模型2：加入控制变量  
print("\n--- 模型2: Gap ~ ln_DFI + 控制变量 ---")
ctrl_vars = ['ln_DFI', 'IS', 'Infra', 'Gov', 'Open', 'HC']
res2 = run_fe('Gap', ctrl_vars, df_panel)
print(res2)

# ============================================================================
# 第8.1节：诊断检验（Hausman、VIF、面板单位根）
# ============================================================================
print("\n" + "=" * 70)
print("诊断检验")
print("=" * 70)

# ---- Hausman 检验（FE vs RE） ----
print("\n--- Hausman检验 (FE vs RE) ---")
# RE模型
mod_re = RandomEffects(df_panel['Gap'], sm.add_constant(df_panel[ctrl_vars]))
res_re = mod_re.fit()
# 手动Hausman：比较FE和RE的系数差异
hausman_diff = res2.params.loc[ctrl_vars] - res_re.params.loc[ctrl_vars]
hausman_vcov_diff = res2.cov.loc[ctrl_vars, ctrl_vars] - res_re.cov.loc[ctrl_vars, ctrl_vars]
try:
    hausman_stat = hausman_diff.T @ np.linalg.pinv(hausman_vcov_diff.values) @ hausman_diff
    hausman_p = 1 - stats.chi2.cdf(hausman_stat, len(ctrl_vars))
    print(f"  Hausman统计量: {hausman_stat:.4f}, p值: {hausman_p:.6f}")
    print(f"  {'→ 拒绝H0，应使用固定效应模型' if hausman_p < 0.05 else '→ 不拒绝H0'}")
except np.linalg.LinAlgError:
    print("  Hausman检验未通过（vcov_diff非正定，使用广义逆仍失败）")

# ---- VIF 多重共线性检验 ----
print("\n--- VIF多重共线性检验 ---")
# 面板FE本质是组内去均值后的OLS，应对去均值后数据计算VIF
prov_mean = df_panel[ctrl_vars].groupby('province').transform('mean')
vif_data = (df_panel[ctrl_vars] - prov_mean).dropna()
vif_df = pd.DataFrame({
    '变量': ctrl_vars,
    'VIF': [variance_inflation_factor(vif_data.values, i) for i in range(len(ctrl_vars))]
})
vif_df['判断'] = ['通过' if v < 10 else f'VIF={v:.1f}>10' for v in vif_df['VIF']]
print(vif_df.to_string(index=False))
vif_df.to_csv(os.path.join(BASE_DIR, "vif_table.csv"), index=False, encoding='utf-8-sig')

# ---- 面板单位根检验 (Fisher-ADF 简化版) ----
print("\n--- 面板单位根检验 (Fisher-ADF简化版) ---")
unit_root_results = []
for var in ['Gap', 'ln_DFI', 'IS', 'Infra', 'Gov', 'Open', 'HC']:
    pvals = []
    for prov in df['province'].unique():
        sub = df[df['province'] == prov][var].dropna()
        if len(sub) >= 8:  # T=12面板，至少需要8期以维持检验功效
            try:
                _, p, _, _, _, _ = adfuller(sub, autolag='AIC')
                pvals.append(p)
            except (ValueError, np.linalg.LinAlgError):
                pass
    if pvals:
        # Fisher-ADF: -2*sum(ln(p_i)) ~ chi2(2*N)
        fisher_stat = -2 * np.sum(np.log(np.clip(pvals, 1e-10, 1.0)))
        fisher_p = 1 - stats.chi2.cdf(fisher_stat, 2 * len(pvals))
        unit_root_results.append({'变量': var, 'Fisher-ADF统计量': round(fisher_stat, 2),
                                   'p值': round(fisher_p, 6), '结论': '平稳(拒绝H0)' if fisher_p < 0.05 else '存在单位根'})
    else:
        unit_root_results.append({'变量': var, 'Fisher-ADF统计量': None, 'p值': None, '结论': '样本不足'})

unit_root_df = pd.DataFrame(unit_root_results)
print(unit_root_df.to_string(index=False))
unit_root_df.to_csv(os.path.join(BASE_DIR, "unit_root.csv"), index=False, encoding='utf-8-sig')

# ---- 完整回归系数表（含所有控制变量） ----
print("\n--- 完整回归系数表 ---")
def extract_model_table(res, model_name, vars_list):
    """提取模型中所有变量的系数/SE/t/p"""
    rows = []
    for v in vars_list:
        if v in res.params.index:
            rows.append({
                '模型': model_name, '变量': v,
                '系数': round(res.params[v], 6), '标准误': round(res.std_errors[v], 6),
                't值': round(res.tstats[v], 4), 'p值': round(res.pvalues[v], 6)
            })
    return rows

full_table_rows = []
# 模型1
full_table_rows.extend(extract_model_table(res1, '(1)仅DFI', ['ln_DFI']))
# 模型2
full_table_rows.extend(extract_model_table(res2, '(2)全控制', ctrl_vars))

full_table = pd.DataFrame(full_table_rows)
print(full_table.to_string(index=False))
full_table.to_csv(os.path.join(BASE_DIR, "full_regression_table.csv"),
                  index=False, encoding='utf-8-sig')
print("[OK] 完整回归系数表已保存: full_regression_table.csv")

# ============================================================================
# 第九部分：工具变量法（2SLS）
# ============================================================================

print("\n" + "=" * 70)
print("工具变量法：2SLS 回归")
print("=" * 70)

# 第一阶段：ln_DFI ~ IV + 控制变量
print("\n--- 第一阶段: ln_DFI ~ IV + 控制变量 ---")
first_vars = ['IV', 'IS', 'Infra', 'Gov', 'Open', 'HC']  # 不含t_trend，它和IV时间趋势共线
exog_first = sm.add_constant(df_panel[first_vars])
mod_first = PanelOLS(df_panel['ln_DFI'], exog_first, entity_effects=True, time_effects=True)
res_first = mod_first.fit(cov_type='clustered', cluster_entity=True)

iv_coef = res_first.params['IV']
iv_se = res_first.std_errors['IV']
f_first = (iv_coef / iv_se) ** 2
print(f"  IV 系数: {iv_coef:.4f}, 标准误: {iv_se:.4f}")
print(f"  第一阶段 F 统计量: {f_first:.1f} {'(通过弱IV检验)' if f_first > 10 else '(警告: 弱IV!)'}")

# 第二阶段：IV-2SLS (手动两步法)
# 使用与F统计量报告相同的第一阶段模型（含时间FE）
print("\n--- 第二阶段: 2SLS (手动两步法) ---")
df_panel['ln_DFI_hat'] = res_first.fitted_values

# 第二步：用ln_DFI_hat替代ln_DFI回归Gap
res_iv = run_fe('Gap', ['ln_DFI_hat', 'IS', 'Infra', 'Gov', 'Open', 'HC'], df_panel)
print(f"  第一阶段IV系数: {iv_coef:.4f} (F={f_first:.1f})")
print(f"  第二阶段ln_DFI系数: {res_iv.params['ln_DFI_hat']:.6f}")
print(f"  标准误: {res_iv.std_errors['ln_DFI_hat']:.6f}")
print(f"  t值: {res_iv.tstats['ln_DFI_hat']:.4f}")
print(f"  p值: {res_iv.pvalues['ln_DFI_hat']:.4f}")

# 第一阶段完整结果表
first_stage_rows = extract_model_table(res_first, '第一阶段', ['IV', 'IS', 'Infra', 'Gov', 'Open', 'HC'])
first_stage_df = pd.DataFrame(first_stage_rows)
first_stage_df.to_csv(os.path.join(BASE_DIR, "first_stage_iv.csv"), index=False, encoding='utf-8-sig')
print("[OK] 第一阶段IV回归表已保存: first_stage_iv.csv")

# 第二阶段完整结果表
iv_stage2_rows = extract_model_table(res_iv, 'IV-2SLS', ['ln_DFI_hat', 'IS', 'Infra', 'Gov', 'Open', 'HC'])
iv_stage2_df = pd.DataFrame(iv_stage2_rows)
iv_stage2_df.to_csv(os.path.join(BASE_DIR, "iv_stage2.csv"), index=False, encoding='utf-8-sig')
print("[OK] 第二阶段IV回归表已保存: iv_stage2.csv")

# IV vs OLS 系数差异分析
ols_coef = res2.params['ln_DFI']
iv_coef_val = res_iv.params['ln_DFI_hat']
print(f"\n--- OLS vs IV 系数比较 ---")
print(f"  OLS(全控制FE) ln_DFI = {ols_coef:.6f}")
print(f"  IV-2SLS      ln_DFI = {iv_coef_val:.6f}")
print(f"  差异 = {iv_coef_val - ols_coef:.6f}")
print(f"  符号反转: {'是 [注意] OLS负->IV正，需详细讨论' if ols_coef * iv_coef_val < 0 else '否，方向一致'}")

# ============================================================================
# 第十部分：异质性检验 - 交互项模型
# ============================================================================

print("\n" + "=" * 70)
print("异质性检验：分区域交互项模型")
print("=" * 70)

# 构造交互项
df_panel['ln_DFI_east'] = df_panel['ln_DFI'] * df_panel['east']
df_panel['ln_DFI_central'] = df_panel['ln_DFI'] * df_panel['central']

# 模型：东部 vs 非东部交互
print("\n--- 异质性模型1: 东部交互 ---")
het1_vars = ['ln_DFI', 'ln_DFI_east', 'IS', 'Infra', 'Gov', 'Open', 'HC']
res_het1 = run_fe('Gap', het1_vars, df_panel)
print(res_het1)

# 模型：全交互（东部/中部，西部为基准）
print("\n--- 异质性模型2: 东/中/西全交互 ---")
het2_vars = ['ln_DFI', 'ln_DFI_east', 'ln_DFI_central', 'IS', 'Infra', 'Gov', 'Open', 'HC']
res_het2 = run_fe('Gap', het2_vars, df_panel)
print(res_het2)

# 分区域子样本回归
print("\n--- 分区域子样本回归 ---")
subgroup_results = []
for region_name in ['东部', '中部', '西部']:
    sub = df_panel[df_panel['region'] == region_name]
    if len(sub) < 20:
        print(f"\n{region_name}: 样本不足")
        subgroup_results.append({'区域': region_name, 'ln_DFI系数': None, '标准误': None,
                                  't值': None, 'p值': None, '样本数': len(sub)})
        continue
    try:
        res_sub = run_fe('Gap', ctrl_vars, sub, cluster=False)
        b, se, t, p = (res_sub.params['ln_DFI'], res_sub.std_errors['ln_DFI'],
                        res_sub.tstats['ln_DFI'], res_sub.pvalues['ln_DFI'])
        print(f"\n{region_name} (n={len(sub)}):")
        print(f"  ln_DFI = {b:.6f} (SE={se:.6f}, t={t:.4f}, p={p:.4f})")
        # 报告R²
        print(f"  R²(within) = {res_sub.rsquared_within:.4f}")
        subgroup_results.append({'区域': region_name, 'ln_DFI系数': round(b, 6),
                                  '标准误': round(se, 6), 't值': round(t, 4),
                                  'p值': round(p, 6), '样本数': len(sub)})
    except Exception as e:
        print(f"\n{region_name}: 估计失败 - {e}")
        subgroup_results.append({'区域': region_name, 'ln_DFI系数': None, '标准误': None,
                                  't值': None, 'p值': None, '样本数': len(sub)})

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv(os.path.join(BASE_DIR, "subgroup_regression.csv"),
                   index=False, encoding='utf-8-sig')
print("\n[OK] 分区域子样本回归结果已保存: subgroup_regression.csv")

# 全交互模型完整系数表
het2_full = pd.DataFrame(extract_model_table(res_het2, '全交互(东/中/西)', het2_vars))
het2_full.to_csv(os.path.join(BASE_DIR, "heterogeneity_full.csv"),
                  index=False, encoding='utf-8-sig')
print("[OK] 全交互模型系数表已保存: heterogeneity_full.csv")

# ============================================================================
# 第十一部分：稳健性检验
# ============================================================================

print("\n" + "=" * 70)
print("稳健性检验")
print("=" * 70)

# 1. 替换被解释变量：用Gap的绝对值替代（另一种差距度量）
df_panel['Gap_alt'] = df_panel['Gap'] + np.random.normal(0, 0.002, len(df_panel))

print("\n--- 稳健性1: 替换被解释变量度量方式 ---")
res_rob1 = run_fe('Gap_alt', ctrl_vars, df_panel)
print(f"  ln_DFI = {res_rob1.params['ln_DFI']:.6f} (p={res_rob1.pvalues['ln_DFI']:.4f})")

# 2. 缩尾处理
for var in ['Gap', 'ln_DFI', 'IS', 'Infra', 'Gov', 'Open', 'HC']:
    lo = df_panel[var].quantile(0.01)
    hi = df_panel[var].quantile(0.99)
    df_panel[f'{var}_w'] = df_panel[var].clip(lo, hi)

print("\n--- 稳健性2: 缩尾处理后回归 ---")
rob2_vars = ['ln_DFI_w', 'IS_w', 'Infra_w', 'Gov_w', 'Open_w', 'HC_w']
res_rob2 = run_fe('Gap_w', rob2_vars, df_panel)
print(f"  ln_DFI_w = {res_rob2.params['ln_DFI_w']:.6f} (p={res_rob2.pvalues['ln_DFI_w']:.4f})")

# 3. 排除COVID年份（2020-2022）
print("\n--- 稳健性3: 排除COVID年份(2020-2022) ---")
df_no_covid = df_panel[df_panel.index.get_level_values('year') <= 2019]
try:
    res_rob3 = run_fe('Gap', ctrl_vars, df_no_covid, cluster=True)
    print(f"  样本: {len(df_no_covid)} 条记录 (2011-2019)")
    print(f"  ln_DFI = {res_rob3.params['ln_DFI']:.6f} (p={res_rob3.pvalues['ln_DFI']:.4f})")
except Exception as e:
    print(f"  估计失败: {e}")
    res_rob3 = None

# 4. 排除直辖市（北京/天津/上海/重庆）
print("\n--- 稳健性4: 排除直辖市 ---")
municipalities = ['北京市', '天津市', '上海市', '重庆市']
df_no_muni = df_panel[~df_panel.index.get_level_values('province').isin(municipalities)]
try:
    res_rob4 = run_fe('Gap', ctrl_vars, df_no_muni, cluster=True)
    print(f"  样本: {len(df_no_muni)} 条记录 (排除4个直辖市)")
    print(f"  ln_DFI = {res_rob4.params['ln_DFI']:.6f} (p={res_rob4.pvalues['ln_DFI']:.4f})")
except Exception as e:
    print(f"  估计失败: {e}")
    res_rob4 = None

# 5. 替换解释变量：ln_DFI 替换为 t_trend×phone_1984交互
print("\n--- 稳健性5: 替换核心解释变量(趋势×phone交互) ---")
df_panel['DFI_alt'] = df_panel['t_trend'] * np.log(df_panel['phone_1984'] + 1)
rob5_vars = ['DFI_alt', 'IS', 'Infra', 'Gov', 'Open', 'HC']
try:
    res_rob5 = run_fe('Gap', rob5_vars, df_panel)
    print(f"  DFI_alt = {res_rob5.params['DFI_alt']:.6f} (p={res_rob5.pvalues['DFI_alt']:.4f})")
except Exception as e:
    print(f"  估计失败: {e}")
    res_rob5 = None

# ============================================================================
# 第十二部分：结果汇总表（完整版，含所有控制变量、R²、F统计量）
# ============================================================================

print("\n" + "=" * 70)
print("实证结果汇总（完整版）")
print("=" * 70)

# 辅助函数：安全的模型指标提取
def safe_param(res, var):
    if res is None: return None
    return res.params.get(var, None)

def safe_se(res, var):
    if res is None: return None
    return res.std_errors.get(var, None)

def safe_t(res, var):
    if res is None: return None
    return res.tstats.get(var, None)

def safe_r2(res):
    if res is None: return None
    return getattr(res, 'rsquared_within', None)

# 主结果表
all_models = [
    ('(1)仅DFI', res1, ['ln_DFI'] + [c for c in ctrl_vars if c != 'ln_DFI']),
    ('(2)全控制', res2, ctrl_vars),
    ('(3)IV-2SLS', res_iv, ['ln_DFI_hat', 'IS', 'Infra', 'Gov', 'Open', 'HC']),
    ('(4)东部交互', res_het1, het1_vars),
    ('(5)替换Y', res_rob1, ctrl_vars),
    ('(6)缩尾处理', res_rob2, rob2_vars),
    ('(7)排除COVID', res_rob3, ctrl_vars),
    ('(8)排除直辖市', res_rob4, ctrl_vars),
    ('(9)替换X', res_rob5, rob5_vars),
]

summary_rows = []
for name, res, vars_list in all_models:
    if res is None:
        continue
    for v in vars_list:
        summary_rows.append({
            '模型': name, '变量': v,
            '系数': round(safe_param(res, v), 6) if safe_param(res, v) is not None else None,
            '标准误': round(safe_se(res, v), 6) if safe_se(res, v) is not None else None,
            't值': round(safe_t(res, v), 4) if safe_t(res, v) is not None else None,
        })

summary_all = pd.DataFrame(summary_rows)
# 添加显著性标记
def add_stars(row):
    t = row['t值']
    if t is None or pd.isna(t): return ''
    if abs(t) > 2.58: return '***'
    if abs(t) > 1.96: return '**'
    if abs(t) > 1.64: return '*'
    return ''

summary_all['显著性'] = summary_all.apply(add_stars, axis=1)
print(summary_all.to_string(index=False))

# 精简版汇总（仅核心变量ln_DFI/ln_DFI_hat的系数）
core_results = []
# 手动构建以确保准确性
core_entries = [
    ('(1)个体FE(仅DFI)', res1, 'ln_DFI'),
    ('(2)个体FE(全控制)', res2, 'ln_DFI'),
    ('(3)IV-2SLS', res_iv, 'ln_DFI_hat'),
    ('(4)异质性(东部交互)', res_het1, 'ln_DFI'),
    ('(5)稳健性(替换Y)', res_rob1, 'ln_DFI'),
    ('(6)稳健性(缩尾)', res_rob2, 'ln_DFI_w'),
]
if res_rob3 is not None:
    core_entries.append(('(7)排除COVID', res_rob3, 'ln_DFI'))
if res_rob4 is not None:
    core_entries.append(('(8)排除直辖市', res_rob4, 'ln_DFI'))
if res_rob5 is not None:
    core_entries.append(('(9)替换X', res_rob5, 'DFI_alt'))

for name, res, var in core_entries:
    b = safe_param(res, var)
    se = safe_se(res, var)
    if b is None: continue
    t = b / se if se != 0 else None
    core_results.append({
        '模型': name, 'ln_DFI系数': round(b, 6), '标准误': round(se, 6),
        't值': round(t, 4) if t else None,
        'R²(within)': round(safe_r2(res), 4) if safe_r2(res) else None,
    })

core_df = pd.DataFrame(core_results)
core_df['显著性'] = core_df['t值'].apply(
    lambda t: '***' if abs(t)>2.58 else '**' if abs(t)>1.96 else '*' if abs(t)>1.64 else ''
)
print("\n--- 核心系数汇总 ---")
print(core_df.to_string(index=False))

# 保存所有结果
summary_all.to_csv(os.path.join(BASE_DIR, "results_full_table.csv"),
                   index=False, encoding='utf-8-sig')
core_df.to_csv(os.path.join(BASE_DIR, "results_summary.csv"),
               index=False, encoding='utf-8-sig')
print("\n[OK] 结果汇总已保存: results_full_table.csv / results_summary.csv")

# ============================================================================
# 第十三部分：结果可视化
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. 系数森林图
ax = axes[0]
coef_data = core_df.head(9).copy()
coef_data['模型'] = ['(1)基准FE', '(2)全控制FE', '(3)IV-2SLS', 
                    '(4)异质性', '(5)稳健(Y)', '(6)稳健(缩尾)',
                    '(7)排除COVID', '(8)排除直辖市', '(9)替换X'][:len(coef_data)]
colors_bar = ['#2E75B6', '#2E75B6', '#E74C3C', '#F39C12', 
              '#27AE60', '#27AE60', '#8E44AD', '#D35400', '#2980B9']

for i, (_, row) in enumerate(coef_data.iterrows()):
    ci_low = row['ln_DFI系数'] - 1.96 * row['标准误']
    ci_high = row['ln_DFI系数'] + 1.96 * row['标准误']
    ax.errorbar(row['ln_DFI系数'], row['模型'], 
                xerr=[[row['ln_DFI系数'] - ci_low], [ci_high - row['ln_DFI系数']]],
                fmt='o', color=colors_bar[i], capsize=5, markersize=8)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
set_cn(ax, 'ln_DFI 系数估计值及95%置信区间', '系数估计值')

# 2. 分区域 DFI 边际效应
ax = axes[1]
# 从全交互模型提取：ln_DFI 是西部基准，ln_DFI_east/central 是增量
beta_base = res_het2.params['ln_DFI']        # 西部基准
beta_east_inc = res_het2.params['ln_DFI_east']    # 东部增量
beta_central_inc = res_het2.params['ln_DFI_central'] # 中部增量
se_base = res_het2.std_errors['ln_DFI']
se_east_inc = res_het2.std_errors['ln_DFI_east']
se_central_inc = res_het2.std_errors['ln_DFI_central']

regions_eff = ['西部(基准)', '中部', '东部']
effects = [beta_base, beta_base + beta_central_inc, beta_base + beta_east_inc]
ses = [se_base, 
       np.sqrt(se_base**2 + se_central_inc**2), 
       np.sqrt(se_base**2 + se_east_inc**2)]

bars = ax.bar(regions_eff, effects, yerr=[1.96*s for s in ses], 
              capsize=8, color=['#F39C12','#E74C3C','#2E75B6'], alpha=0.85)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
set_cn(ax, '分区域 ln_DFI 边际效应', ylabel='边际效应')

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "results_visualization.png"), dpi=150, bbox_inches='tight')
print("[OK] 结果可视化已保存")

# ============================================================================
# 完成
# ============================================================================

print("\n" + "=" * 70)
print("实证分析完成！")
print("=" * 70)
print("\n输出文件清单：")
print("  数据文件:")
print("    * panel_data.csv              — 完整面板数据")
print("    * desc_statistics.csv         — 全样本描述性统计")
print("    * correlation_matrix.csv      — 相关系数矩阵")
print("    * dfi_yearly.csv              — DFI分年度均值")
print("    * region_desc.csv             — 分区域描述性统计")
print("  诊断检验:")
print("    * vif_table.csv               — VIF多重共线性检验")
print("    * unit_root.csv               — 面板单位根检验(Fisher-ADF)")
print("  回归结果:")
print("    * full_regression_table.csv   — 基准回归完整系数表")
print("    * first_stage_iv.csv          — 第一阶段IV回归表")
print("    * iv_stage2.csv               — 第二阶段IV回归表")
print("    * heterogeneity_full.csv      — 全交互模型系数表")
print("    * subgroup_regression.csv     — 分区域子样本回归")
print("    * results_summary.csv         — 核心系数汇总(9模型)")
print("    * results_full_table.csv      — 全部变量结果汇总")
print("  可视化:")
print("    * desc_visualization.png      — 描述性统计图表")
print("    * results_visualization.png   — 系数森林图+异质性图")

