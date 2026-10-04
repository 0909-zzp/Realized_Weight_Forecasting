# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
k = sys.argv[1]
path = r'D:/HuaweiMoveData/Users/27438/Desktop/大创/omml_all.txt'
with open(path, encoding='utf-8') as f:
    text = f.read()
marker = f'===== 式({k}) ====='
start = text.index(marker) + len(marker)
# find next marker
nxt = text.find('===== 式(', start)
block = text[start:nxt].strip() if nxt != -1 else text[start:].strip()
print(block)
