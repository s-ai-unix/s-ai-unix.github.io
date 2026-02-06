#!/usr/bin/env python3
"""
修复被破坏的markdown文件
移除错误添加的$符号,保持代码块纯净,修复公式格式
"""

import re
import sys

def clean_markdown_content(content: str) -> str:
    """
    清理markdown内容中的公式格式错误
    """
    lines = content.split('\n')
    result = []
    in_code_block = False
    code_block_fence = None

    for line in lines:
        # 检测代码块
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_block_fence = line.strip()
            else:
                in_code_block = False
                code_block_fence = None
            result.append(line)
            continue

        # 如果在代码块中,移除所有$符号
        if in_code_block:
            # 移除代码块中的$符号,但要小心不要破坏字符串
            cleaned = line
            # 移除独立的$variable$模式
            cleaned = re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)\$', r'\1', cleaned)
            # 移除$$包裹的
            cleaned = cleaned.replace('$$', '')
            result.append(cleaned)
            continue

        # 不在代码块中,处理公式

        # 1. 移除被错误分割的公式中的多余$符号
        # 例如: $W_h$ $\cdot$ $h_{t-1}$ -> W_h \cdot h_{t-1}
        # 策略: 如果一行中有多个用空格或$分隔的$...$片段,先合并它们

        # 先替换所有 $math$ 为 math,收集所有的数学片段
        math_fragments = []
        def collect_math(match):
            math_fragments.append(match.group(1))
            return f'__MATH_FRAGMENT_{len(math_fragments)-1}__'

        # 收集所有的$...$片段
        line = re.sub(r'\$([^$]+?)\$', collect_math, line)

        # 现在line中所有$...$都被替换为__MATH_FRAGMENT_X__
        # 如果这些片段是连续的(被空格分隔),应该把它们合并成一个公式
        # 但这很复杂,我们用更简单的方法

        # 恢复时,如果相邻的都是__MATH_FRAGMENT_X__,应该合并
        # 先简单恢复:如果一行中只有一个$$...$$,保持不变
        # 如果有多个$...$,可能需要合并

        # 检查这一行是否有$$...$$
        display_math_count = line.count('$$')
        if display_math_count >= 2:
            # 这一行有display math,保持原样
            pass
        else:
            # 检查是否有很多数学符号被分割
            # 如果一行中有超过3个__MATH_FRAGMENT_X__,可能是被过度分割了
            fragment_count = len(re.findall(r'__MATH_FRAGMENT_\d+__', line))
            if fragment_count > 3:
                # 可能是被过度分割,合并所有片段
                # 先恢复所有片段
                for i, fragment in enumerate(math_fragments):
                    line = line.replace(f'__MATH_FRAGMENT_{i}__', fragment)
                # 现在这一行有很多$...$片段,需要合并
                # 移除所有$,然后用智能的方式重新添加

                # 首先移除所有的$
                line_no_dollar = line.replace('$', '')

                # 检测是否像是一个公式行
                # 如果包含LaTeX命令,应该用$包裹
                latex_patterns = [
                    r'\\frac\{', r'\\sum', r'\\int', r'\\cdot',
                    r'\\sigma', r'\\mu', r'\\chi', r'\\theta',
                    r'\\left', r'\\right', r'\\bar\{', r'\\hat\{',
                    r'\\tilde\{', r'\\sqrt', r'\\times', r'\\div',
                    r'\\le', r'\\ge', r'\\ne', r'\\approx', r'\\equiv'
                ]
                has_latex = any(re.search(pattern, line_no_dollar) for pattern in latex_patterns)

                # 如果有下划线且看起来像数学表达式
                has_underscore_math = bool(re.search(r'[a-zA-Z]_\{?[a-zA-Z]\}?', line_no_dollar))

                if has_latex or has_underscore_math:
                    # 应该是一个公式,用$包裹整行(去除行首行尾空白)
                    stripped = line_no_dollar.strip()
                    if not stripped.startswith('$') and not stripped.endswith('$'):
                        line = f'${stripped}$'
                    else:
                        line = line_no_dollar
                else:
                    # 可能不是公式,保持原样但去除$
                    line = line_no_dollar
            else:
                # 片段不多,直接恢复
                for i, fragment in enumerate(math_fragments):
                    line = line.replace(f'__MATH_FRAGMENT_{i}__', f'${fragment}$')

        result.append(line)

    return '\n'.join(result)

def fix_broken_formulas(content: str) -> str:
    """
    修复被切碎的公式
    例如: $\frac{1}${n}$\sum$_{i=1}^{n}$X_i$
    修复为: $\frac{1}{n}\sum_{i=1}^{n}X_i$
    """
    lines = content.split('\n')
    result = []

    for line in lines:
        # 跳过代码块
        if line.strip().startswith('```'):
            result.append(line)
            continue

        # 修复被切碎的frac
        # $\frac{1}${n}$ -> $\frac{1}{n}$
        line = re.sub(r'\$\\frac\{([^}]+)\}\$\{([^}]+)\}\$', r'$\frac{\1}{\2}$', line)

        # 修复被切碎的一般模式
        # $part1$$part2$ -> $part1part2$
        line = re.sub(r'\$([^$]+)\$\$\$([^$]+)\$\$', r'$\1\2$', line)

        # 修复 $part1$ $part2$ (空格分隔的两个公式)
        # 如果这两个部分应该是一个公式,需要合并
        # 检测常见的情况: $a_$b$ 应该是 $a_b$

        result.append(line)

    return '\n'.join(result)

def clean_file(filepath: str):
    """
    清理单个文件
    """
    print(f"处理文件: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 第一步: 清理代码块和过度分割的公式
    content = clean_markdown_content(content)

    # 第二步: 修复被切碎的公式
    content = fix_broken_formulas(content)

    # 保存
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ 完成: {filepath}")

if __name__ == '__main__':
    files = [
        '/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/content/posts/2020-02-16-mathematical-statistics-theory.md',
        '/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/content/posts/2019-08-27-rnn-lstm-gru-complete-guide.md',
        '/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/content/posts/2019-08-14-gradient-boosting-algorithm-guide.md',
    ]

    for filepath in files:
        try:
            clean_file(filepath)
        except Exception as e:
            print(f"✗ 错误: {filepath} - {e}")
