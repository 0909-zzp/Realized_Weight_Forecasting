"""Merge all multi-line add_para blocks into single lines"""
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Strategy: Find all add_para( ... ) blocks with multi-line strings
# and replace them with single-line versions

def flatten_add_para(match):
    """Merge a multi-line add_para block into single line"""
    block = match.group(0)
    lines = block.split('\n')
    # Extract the first line (add_para()
    first_line = lines[0]
    # Collect all string content
    parts = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == ')':
            break
        if stripped.startswith('f"') and stripped.endswith('"'):
            # Extract content between f" ... "
            parts.append(stripped[2:-1])
        elif stripped.startswith('"') and stripped.endswith('"'):
            parts.append(stripped[1:-1])
    
    if not parts:
        return block
    
    # Determine if any part came from an f-string
    has_f = any(l.strip().startswith('f"') for l in lines if l.strip())
    prefix = 'f' if has_f else ''
    
    combined = prefix + '"' + ''.join(parts) + '"'
    return first_line + '\n    ' + combined + '\n)'

# Apply to all multi-line add_para blocks
# Pattern: add_para( followed by one or more lines of indented strings, ending with )
pattern = r'add_para\(\s*\n(?:\s*(?:f)?\"[^\"]*\"\s*\n)+\s*\)'
new_text = re.sub(pattern, flatten_add_para, text)

with open('build_paper.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Multi-line add_para blocks merged into single lines')
