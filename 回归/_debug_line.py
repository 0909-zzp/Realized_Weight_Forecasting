import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'rb') as f:
    raw = f.read()

lines = raw.split(b'\n')
# Check around line 260-270
for line_no in range(259, 270):
    line = lines[line_no]
    print(f'\nLine {line_no+1}:')
    for i in range(min(80, len(line))):
        b = line[i]
        c = chr(b) if 32 <= b <= 126 else f'[0x{b:02X}]'
        print(c, end='')
    print()
