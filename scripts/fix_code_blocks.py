#!/usr/bin/env python3
"""
修复博客文章中的代码块语言标识问题
"""
import re
from pathlib import Path

def fix_code_blocks(content):
    """修复缺少语言标识的代码块"""

    # 1. 修复单独的```为```text（排除已经正确标记的）
    # 模式：```\n后跟不是字母的内容，直到```
    # 但要排除已经正确标记的（如```bash, ```python等）

    # 首先找到所有```...```块
    blocks = []
    pos = 0
    while True:
        start = content.find('```', pos)
        if start == -1:
            break
        end = content.find('```', start + 3)
        if end == -1:
            break

        block = content[start:end + 3]
        blocks.append({
            'start': start,
            'end': end + 3,
            'block': block
        })
        pos = end + 3

    # 分析每个代码块
    changes = []
    for block_info in reversed(blocks):  # 从后往前，避免位置变化
        block = block_info['block']

        # 检查第一行是否有语言标识
        first_line_end = block.find('\n')
        if first_line_end == -1:
            first_line = block
        else:
            first_line = block[:first_line_end]

        # 如果```后面没有小写字母，则添加text标识
        # 排除```后面直接是\n的情况
        if first_line == '```':
            changes.append({
                'old': '```',
                'new': '```text',
                'pos': block_info['start']
            })

    # 应用替换
    for change in reversed(changes):
        start = change['pos']
        content = content[:start] + change['new'] + content[start + len(change['old']):]

    return content

def fix_file(filepath):
    """修复单个文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    content = fix_code_blocks(content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    script_dir = Path(__file__).parent
    posts_dir = script_dir / 'content' / 'posts'
    md_files = sorted(posts_dir.glob('*.md'))

    if not md_files:
        print("错误: 未找到.md 文件")
        return

    print(f"检查 {len(md_files)} 篇文章...\n")

    fixed_count = 0
    for md_file in md_files:
        if fix_file(md_file):
            print(f"✓ {md_file.name} - 已修复")
            fixed_count += 1
        else:
            print(f". {md_file.name} - 无需修改")

    print(f"\n完成！共修复 {fixed_count} 篇文章")

if __name__ == '__main__':
    main()
