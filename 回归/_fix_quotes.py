"""Fix Chinese quotation marks in build_paper.py that got converted to ASCII quotes"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'r', encoding='utf-8') as f:
    content = f.read()

# These are the exact text patterns where Chinese quotes appear as ASCII "
# We replace the surrounding quotes with 「」 to preserve semantics without breaking Python strings

replacements = [
    # Line 262
    ('这与"先发地区数字化先行->经济先行"的基本事实相吻合', 
     '这与「先发地区数字化先行->经济先行」的基本事实相吻合'),
    # Line 264  
    ('异质性分析中"西部拉大趋势"的发现',
     '异质性分析中「西部拉大趋势」的发现'),
    # Line 428
    ('数据的"虹吸效应"，反而',
     '数据的「虹吸效应」，反而'),
    # Line 511
    ('数字技术的"网络外部性"特征',
     '数字技术的「网络外部性」特征'),
    # Line 556
    ('区域均衡的"万能药"',
     '区域均衡的「万能药」'),
    # Line 557
    ('数字技术的"扩散效应"已开始',
     '数字技术的「扩散效应」已开始'),
    # Line 558
    ('实现"蛙跳式"追赶',
     '实现「蛙跳式」追赶'),
    # Line 570
    ('了"数字鸿沟"与"数字红利"两种',
     '了「数字鸿沟」与「数字红利」两种'),
    # Line 691
    ('数据的"虹吸"现象',
     '数据的「虹吸」现象'),
    # Line 699
    ('印了数字技术的"网络外部性"特征',
     '印了数字技术的「网络外部性」特征'),
    # Line 712
    ('本文的实证结论本质上仍然是\n    "方法论导向的"教学性展示"',
     '本文的实证结论本质上仍然是\n    「方法论导向的教学性展示」'),
    # Line 719
    ('发挥数字经济的\n    "扩散效应"',
     '发挥数字经济的\n    「扩散效应」'),
    # Line 720
    ('重心从"硬件铺设"转向\n    "软件升级"',
     '重心从「硬件铺设」转向\n    「软件升级」'),
    # Line 721
    ('群体的"最后一公里"',
     '群体的「最后一公里」'),
    # Line 726
    ('在"东数西算"等国家',
     '在「东数西算」等国家'),
    # Line 728
    ('防范"虹吸效应"加剧',
     '防范「虹吸效应」加剧'),
    # Line 729
    ('应将"数字包容性增长"作为',
     '应将「数字包容性增长」作为'),
    # Line 750
    ('阈值后，"数字红利"方能',
     '阈值后，「数字红利」方能'),
    # Misc
    ('从"硬件铺设"转向',
     '从「硬件铺设」转向'),
    ('从"可能性"转化为',
     '从「可能性」转化为'),
    ('转变"现实增长动力"',
     '转变「现实增长动力」'),
    ('印了"赢者通吃"与',
     '印了「赢者通吃」与'),
    ('方法论导向的"教学性展示"',
     '方法论导向的教学性展示'),
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f'Fixed ({count}): {old[:50]}...')
    else:
        print(f'NOT FOUND: {old[:50]}...')

# Also check for any remaining "text" patterns inside f-strings
# that might still cause issues (specifically for the complex patterns)

with open('build_paper.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nTotal replacements: {count}')
