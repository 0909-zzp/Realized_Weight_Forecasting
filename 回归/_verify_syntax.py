"""Verify Python syntax and find ALL broken lines in build_paper.py"""
import py_compile
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    py_compile.compile('build_paper.py', doraise=True)
    print('Syntax OK!')
except py_compile.PyCompileError as e:
    print(f'Syntax error: {e}')

# Also find all lines where " appears inside a string context and breaks things
# Strategy: find all " that are not part of Python string delimiters
with open('build_paper.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('\n=== Scanning for potential issues ===')
in_string = False
string_delim = None

for i, line in enumerate(lines, 1):
    # Count quote characters
    stripped = line.strip()
    if stripped.startswith('#') or stripped.startswith('"""'):
        continue
    
    # Simple heuristic: find lines with add_para that have odd quote counts
    if 'add_para(' in line:
        # Check next few lines
        for j in range(5):
            if i + j < len(lines):
                nl = lines[i + j]
                # Count " in this line (ignoring escaped)
                quotes = nl.count('"') - nl.count('\\"')
                if quotes % 2 != 0 and not nl.strip().startswith('#'):
                    print(f'  Suspicious line {i+j+1} ({quotes} quotes): {nl.rstrip()[:100]}')
