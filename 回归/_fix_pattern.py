import re, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'r', encoding='utf-8') as f:
    content = f.read()

cjk = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
tchars = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\uff0c\uff1b\uff1a\u2014\u3001\u3002\uff01\uff1f\uff08\uff09%\.\d\->]'
pattern = f'({cjk})\"((?:{tchars})+)\"({cjk})'

LQ = '\u300c'
RQ = '\u300d'

for _pass in range(10):
    old_len = len(content)
    new_content = re.sub(pattern, r'\1' + LQ + r'\2' + RQ + r'\3', content)
    if len(new_content) == old_len:
        break
    content = new_content
    print(f'Pass {_pass+1}: applied')

with open('build_paper.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
