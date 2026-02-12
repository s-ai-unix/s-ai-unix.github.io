#!/usr/bin/env python3
"""修复 Markdown 文件中的数学公式格式"""

import re
import sys

def fix_math_formulas(file_path):
    """修复文件中的数学公式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 修复块公式中的格式问题
    # 1. 修复类似 $...$ $...$ 的格式为正确的公式
    # 例如: $f_t$ = $\sigma$(...) -> $$f_t = \sigma(...)$$

    # 这种情况比较复杂，我们先处理一些简单模式

    # 2. 修复列表中的公式，将每个$...$保持不变，但修复多个$连接的情况
    # 例如: $\bar{X}$ = $\frac{1}{n}$... -> $\bar{X} = \frac{1}{n}...$

    # 修复模式：公式中多个 $...$ 连接
    # 匹配类似 $a$ $op$ $b$ 的模式，转换为 $a op b$

    # 修复数学符号之间的空格和 $
    content = re.sub(r'\$\$(\w+)\$\$', r'$$\1$$', content)  # 移除多余的$

    # 更复杂的模式：在行内公式中，多个独立的 $...$ 应该合并
    # 但这需要更仔细的处理，避免误伤代码块

    lines = content.split('\n')
    in_code_block = False
    fixed_lines = []

    for line in lines:
        # 检测代码块
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            fixed_lines.append(line)
            continue

        if in_code_block:
            # 在代码块中，移除所有被错误添加的 $ 符号
            # 保留 Python 的变量名，移除数学公式标记
            fixed_line = line
            # 移除行内可能的 $$...$$ 块
            fixed_line = re.sub(r'\$\$[^$]*\$\$', '', fixed_line)
            # 移除单个 $...$ 包裹的变量
            fixed_line = re.sub(r'\$(\w+)\$', r'\1', fixed_line)
            fixed_lines.append(fixed_line)
            continue

        # 不在代码块中，处理数学公式
        # 修复行内公式中多个独立的 $...$ 连接
        # 例如: $a$ = $b$ + $c$ -> $a = b + c$

        # 查找所有 $...$ 模式
        matches = list(re.finditer(r'\$([^$]+)\$', line))
        if len(matches) >= 2:
            # 检查是否是连续的公式片段
            new_line = ''
            last_end = 0
            i = 0

            while i < len(matches):
                match = matches[i]
                start, end = match.span()
                formula = match.group(1)

                # 添加公式前的内容
                new_line += line[last_end:start]

                # 尝试合并连续的公式片段
                merged_formula = formula
                j = i + 1

                # 检查下一个公式是否紧跟着（中间只有空格或运算符）
                while j < len(matches):
                    next_match = matches[j]
                    between = line[end:next_match.start()].strip()

                    # 如果中间只有运算符、=、±等数学符号，合并它们
                    if between in ['=', '+', '-', '*', '/', '±', '∼', '<', '>', '≤', '≥',
                                   '\\cdot', '\\times', '\\div', '\\pm',
                                   '\\sim', '\\le', '\\ge', '\\ne', '\\approx',
                                   '\\left', '\\right', '\\[', '\\]']:
                        merged_formula += ' ' + between + ' ' + next_match.group(1)
                        end = next_match.end()
                        j += 1
                    else:
                        break

                new_line += '$' + merged_formula.strip() + '$'
                last_end = end
                i = j

            # 添加剩余内容
            new_line += line[last_end:]
            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    # 后处理：清理一些特定的问题模式

    # 修复代码块中被错误包裹的变量
    # 例如: $sample_mean$ = ... -> sample_mean = ...
    # 这个已经在上面处理了

    # 修复块公式结尾使用 \[ 的问题
    content = re.sub(r'\$\$([^$]*)\\\[\s*$', r'$$\1$$', content, flags=re.MULTILINE)

    # 保存修改后的内容
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_math_formulas.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if fix_math_formulas(file_path):
        print(f"已修复文件: {file_path}")
    else:
        print(f"无需修复: {file_path}")
