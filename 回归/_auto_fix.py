"""
Auto-fix: Find ALL remaining Chinese double quotes converted to ASCII " in build_paper.py.
Strategy: For each line, detect if it's part of a string context and has broken quotes.
"""
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'rb') as f:
    raw = f.read()

# Decode to text for processing
content = raw.decode('utf-8')
lines = content.split('\n')

# Our goal: find strings where " appears inside, indicating Chinese quotes that were converted
# Pattern: inside add_para() or similar string contexts, find " followed by Chinese chars followed by "

# Step 1: Identify segments that are Python string literals
# Step 2: Within those, find "ChineseText" patterns where the quotes are NOT Python delimiters
# Step 3: Replace with 「ChineseText」

cjk_char = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
cjk_word = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef->]'

# Pattern: inside a string (between Python opening " and closing "),
# find: CJK char + " + text (may contain CJK, ->, etc) + " + CJK char
# The tricky part: we can't just regex the whole file because " is used for Python strings too
# So we need to be more precise

# Better approach: Process line by line, and for lines that are part of add_para/f-string contexts,
# replace ALL non-delimiter ASCII quotes with Chinese brackets

# Simplest approach: find " that is preceded by a CJK char (and not at the start of a string)
# This means the " should be a Chinese left/right quote

# Pattern: CJK + " + non-quote characters + " + CJK
pattern = f'({cjk_char})"((?:{cjk_word}|[，。！？、：；（）%\\d\\.\\-+])+)"({cjk_char})'

def replace_func(m):
    left_cjk = m.group(1)
    text = m.group(2)
    right_cjk = m.group(3)
    return f'{left_cjk}\u300c{text}\u300d{right_cjk}'

# Also handle: CJK"text" at string boundary
# And: "text"CJK (where text starts after a non-Python-delimiter context)

# Do multiple passes
for _pass in range(5):
    old_len = len(content)
    content = re.sub(pattern, replace_func, content)
    if len(content) == old_len:
        break
    print(f'Pass {_pass+1}: {old_len} -> {len(content)} chars')

# Additional fix: handle patterns like: 的"text"的 → 的「text」的
# But only within string contexts (between Python quotes that are properly matched)

with open('build_paper.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Auto-fix complete')
