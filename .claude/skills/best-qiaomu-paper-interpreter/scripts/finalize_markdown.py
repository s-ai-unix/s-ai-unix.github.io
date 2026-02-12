#!/usr/bin/env python3
"""
提取markdown的H1标题作为文件名，并复制到论文解读目录
"""
import re
import os
import sys
import shutil
from pathlib import Path


def extract_h1_and_remove(markdown_path):
    """
    读取markdown文件，提取H1标题，删除H1行，返回新内容和标题
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()

    h1_pattern = r'^#\s+(.+)$'
    lines = content.split('\n')

    h1_title = None
    new_lines = []
    h1_found = False

    for line in lines:
        match = re.match(h1_pattern, line)
        if match and not h1_found:
            h1_title = match.group(1).strip()
            h1_found = True
            continue
        else:
            new_lines.append(line)

    while new_lines and new_lines[0].strip() == '':
        new_lines.pop(0)

    new_content = '\n'.join(new_lines)
    return h1_title, new_content


def obsidian_safe_image_paths(content):
    """确保所有图片路径在Obsidian中可用"""
    def fix_path(match):
        alt_text = match.group(1)
        path = match.group(2)

        if path.startswith('http://') or path.startswith('https://'):
            return match.group(0)

        if path.startswith('<') and path.endswith('>'):
            return match.group(0)

        if ' ' in path:
            return f"![{alt_text}](<{path}>)"

        return match.group(0)

    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    return re.sub(pattern, fix_path, content)


def validate_image_paths(content, output_dir="."):
    """验证markdown中的所有图片路径是否存在"""
    image_pattern = r'!\[.*?\]\(<?([^)>]+)>?\)'
    image_paths = re.findall(image_pattern, content)

    invalid_paths = []
    for path in image_paths:
        if path.startswith('http://') or path.startswith('https://'):
            continue

        full_path = os.path.join(output_dir, path)
        if not os.path.exists(full_path):
            invalid_paths.append(path)

    return len(image_paths) - len(invalid_paths), invalid_paths


def save_with_h1_title(markdown_path, output_dir=".", copy_to_reading_dir=True, validate_images=True):
    """
    使用H1标题作为文件名保存markdown文件
    
    参数:
        markdown_path: 输入markdown文件路径
        output_dir: 输出目录（工作目录，默认当前目录）
        copy_to_reading_dir: 是否复制到论文解读阅读目录（默认True）
        validate_images: 是否验证图片路径（默认True）
    
    返回: (success, output_path, reading_path, h1_title)
    """
    h1_title, new_content = extract_h1_and_remove(markdown_path)

    if not h1_title:
        return False, None, None, "未找到H1标题"

    # 确保图片路径Obsidian兼容
    new_content = obsidian_safe_image_paths(new_content)

    # 验证图片路径
    if validate_images:
        valid_count, invalid_paths = validate_image_paths(new_content, output_dir)
        if invalid_paths:
            print(f"⚠️  图片路径验证：")
            print(f"   ✅ 有效: {valid_count} 张")
            print(f"   ❌ 无效: {len(invalid_paths)} 张")
            for path in invalid_paths[:5]:
                print(f"      - {path}")
            if len(invalid_paths) > 5:
                print(f"      ... 还有 {len(invalid_paths) - 5} 个")

    # 生成文件名
    safe_filename = h1_title
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        safe_filename = safe_filename.replace(char, '')

    output_filename = f"{safe_filename}_解读.md"
    
    output_dir_p = Path(output_dir).resolve()
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # 保存到工作目录（输出目录）
    output_path = str(output_dir_p / output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # 复制到阅读目录
    reading_path = None
    if copy_to_reading_dir:
        # Reading directory should live under the same root as `output_dir`,
        # not relative to the current working directory (which may be paper_dir).
        reading_dir_p = output_dir_p / "01.inbox" / "论文解读"
        reading_dir_p.mkdir(parents=True, exist_ok=True)
        reading_path = str(reading_dir_p / output_filename)
        shutil.copy2(output_path, reading_path)

    return True, output_path, reading_path, h1_title


def main():
    """命令行工具"""
    if len(sys.argv) < 2:
        print("用法: python finalize_markdown.py <输入markdown> [输出目录]")
        print("示例: python finalize_markdown.py papers/T5_2019/T5论文_解读.md .")
        print()
        print("功能:")
        print("  1. 提取markdown中的H1标题")
        print("  2. 删除文章中的H1行")
        print("  3. 用H1标题作为文件名保存到输出目录（工作副本）")
        print("  4. 复制到 01.inbox/论文解读/ 目录（阅读副本）")
        sys.exit(1)

    markdown_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not os.path.exists(markdown_path):
        print(f"错误：文件不存在: {markdown_path}")
        sys.exit(1)

    success, output_path, reading_path, h1_title = save_with_h1_title(markdown_path, output_dir)

    if success:
        print(f"✅ 成功！")
        print(f"   H1标题: {h1_title}")
        print(f"   工作副本: {output_path}")
        if reading_path:
            print(f"   阅读副本: {reading_path}")
    else:
        print(f"❌ 失败: {h1_title}")
        sys.exit(1)


if __name__ == "__main__":
    main()
