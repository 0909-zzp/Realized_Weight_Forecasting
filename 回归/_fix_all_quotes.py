"""Comprehensive fix: find ALL Chinese-style double quotes in build_paper.py"""
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Strategy: Find all ' "text" ' patterns where the text contains Chinese characters
# These are Chinese quotation marks that got converted to ASCII
# We replace them with 「text」

# Pattern: ASCII quote followed by 2+ Chinese chars followed by ASCII quote
# But only inside Python strings (after non-quote-start-of-string contexts)

# Simpler approach: manually list all remaining patterns
remaining = [
    # Line 426
    ('所捕捉的"历史通信禀赋->数字金融发展"这一', '所捕捉的「历史通信禀赋->数字金融发展」这一'),
    # From reading the file context, other patterns
    ('工具变量利用的是"', '工具变量利用的是「'),
    ('"这一局部变异来源', '」这一局部变异来源'),
    ('能力上"污染"了', '能力上「污染」了'),
    ('反而加剧区域\n    "差距', '反而加剧区域\n    差距'),  # fix missing quote closure
    ('并将\n    "数字红包"从', '并将\n    「数字红包」从'),
    ('数字经济的"扩散效应"与', '数字经济的「扩散效应」与'),
    ('数字经济的"极化风险"', '数字经济的「极化风险」'),
    ('"数字门槛"问题', '「数字门槛」问题'),
    ('了"数字经济的"收敛效应"', '了数字经济的「收敛效应」'),
    ('"普惠效应"或"收敛效应"', '「普惠效应」或「收敛效应」'),
    ('依然成立的"稳健性"', '依然成立的稳健性'),
    ('的\"恶化效应\"', '的「恶化效应」'),
    ('以"数字驱动"为核心', '以「数字驱动」为核心'),
]

count = 0
for old, new in remaining:
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f'Fixed ({count}): {old[:60]}')

# Also do a regex-based fix for generic patterns:
# Find " followed by CJK character sequence followed by " 
# But be careful not to break actual Python syntax

# Pattern: ASCII " immediately preceded by a CJK char and followed by a CJK char
# This indicates the quotes are Chinese quotation marks
cjk = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
# Look for CJK"text"CJK pattern
pattern = f'({cjk})"((?:{cjk}|[\u4e00-\u9fff，。！？、：；“”「」（）—…-])+)"({cjk})'
matches = list(re.finditer(pattern, content))
print(f'\nRegex matches found: {len(matches)}')
for m in matches[:20]:
    print(f'  "{m.group(2)[:40]}"')

# Replace these with CJK「text」CJK
content = re.sub(pattern, r'\1「\2」\3', content)

with open('build_paper.py', 'w', encoding='utf-8') as f:
    f.write(content)
print(f'\nManual fixes: {count}, Regex fixes applied')
