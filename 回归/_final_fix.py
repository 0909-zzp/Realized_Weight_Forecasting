"""
Final fix: 
1. Remove invalid \' inside f-strings (they were incorrectly escaped during merging)
2. Manually merge remaining multi-line add_para blocks
"""
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('build_paper.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Fix escaped single quotes in f-strings
# In f"..." strings, we don't need to escape single quotes
# Pattern: \' inside an f-string should just be '
# But ONLY fix those that look incorrectly escaped

# First, fix the specific known broken line
old_line = r"""f"第一，数字普惠金融的发展在基准模型中显著缩小了省际经济差距，ln_DFI的FE估计系数约为{b2:.4f}，在10%水平上显著。从经济显著性来看，ln_DFI每提高一个标准差，区域差距约下降{abs(get_desc_val(\'区域经济差距\')[\'标准差\']*b2/desc.loc[0,\'标准差\']):.2f}个标准差。该结论在替换被解释变量度量方式、缩尾处理、排除疫情冲击、排除直辖市以及替换解释变量等一系列稳健性检验后始终成立，展现出较强的实证可靠性。\""""

new_line = """f"第一，数字普惠金融的发展在基准模型中显著缩小了省际经济差距，ln_DFI的FE估计系数约为{b2:.4f}，在10%水平上显著。从经济显著性来看，ln_DFI每提高一个标准差，区域差距约下降{abs(get_desc_val('区域经济差距')['标准差']*b2/desc.loc[0,'标准差']):.2f}个标准差。该结论在替换被解释变量度量方式、缩尾处理、排除疫情冲击、排除直辖市以及替换解释变量等一系列稳健性检验后始终成立，展现出较强的实证可靠性。\""""

if old_line in content:
    content = content.replace(old_line, new_line)
    print('Fixed line 1: get_desc_val escape issue')
else:
    print('Line 1 not found for exact match, trying regex...')
    # Try regex-based fix: replace \' inside f-strings with '
    # Be careful: only fix inside f"..." strings, not regular "..." strings
    def fix_escapes(m):
        inner = m.group(1)
        # Replace \' with ' inside f-string content
        inner = inner.replace("\\'", "'")
        return 'f"' + inner + '"'
    
    # Pattern: f"text with \' escapes"
    content = re.sub(r'f"((?:(?!f").)*?\\[\\\'].*?)"', fix_escapes, content, flags=re.DOTALL)

# Step 2: Fix any remaining \' escapes in add_para blocks
content = content.replace("\\'区域经济差距\\'", "'区域经济差距'")
content = content.replace("\\'标准差\\'", "'标准差'")
print('Fixed all escaped single quotes')

# Step 3: Merge remaining multi-line add_para (lines 1053-1065 block)
# Find the unmerged block
old_block = """add_para(

    f\"第三，数字经济的收敛效应存在显著的区域梯度差异。交互项模型和分样本回归一致\"

    f\"表明：东部地区数字经济的弥合效应最强（系数约为{east_b:.4f}），中部次之\"

    f\"（约为{central_b:.4f}），西部最弱（约为{west_b:.4f}）。这一\\\"东部>中部>西部\\\"\"

    f\"的梯度格局印证了数字技术的「网络外部性」特征——在数字基础设施、人力资本储备\"

    f\"和产业生态更为完善的地区，数字化能够更高效地转化为区域追赶的引擎。\"

)"""

# Build the merged version
new_block = """add_para(
    f\"第三，数字经济的收敛效应存在显著的区域梯度差异。交互项模型和分样本回归一致表明：东部地区数字经济的弥合效应最强（系数约为{east_b:.4f}），中部次之（约为{central_b:.4f}），西部最弱（约为{west_b:.4f}）。这一\\\"东部>中部>西部\\\"的梯度格局印证了数字技术的「网络外部性」特征——在数字基础设施、人力资本储备和产业生态更为完善的地区，数字化能够更高效地转化为区域追赶的引擎。\"
)"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print('Merged final multi-line block')
else:
    print('Multi-line block not found for exact match')
    # Try a more flexible merge
    # Find all remaining multi-line add_para blocks
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'add_para(':
            # Check if next lines are strings
            j = i + 1
            string_lines = []
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped == ')':
                    break
                if stripped and (stripped.startswith('f"') or stripped.startswith('"')):
                    string_lines.append(j)
                elif stripped == '':
                    pass  # skip empty lines
                else:
                    break  # unexpected content
                j += 1
            
            if len(string_lines) > 1 and j < len(lines) and lines[j].strip() == ')':
                # Found a multi-line block that wasn't merged
                # Extract content from all strings
                prefix = 'f' if lines[string_lines[0]].strip().startswith('f"') else ''
                all_text = ''
                for sl in string_lines:
                    s = lines[sl].strip()
                    # Remove f"..." or "..." wrapper, handle escaped quotes
                    if s.startswith('f"'):
                        inner = s[2:-1]
                    else:
                        inner = s[1:-1]
                    all_text += inner
                
                # Build merged line
                new_line = f'    {prefix}"{all_text}"'
                # Replace the block
                lines[i] = 'add_para('
                lines[i+1:j+1] = [new_line, ')']
                print(f'Merged block at line {i+1}')
            
            i = j + 1
        else:
            i += 1
    
    content = '\n'.join(lines)

with open('build_paper.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Final fix complete')
