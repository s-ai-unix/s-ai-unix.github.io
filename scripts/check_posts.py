#!/usr/bin/env python3
"""
检查博客文章中的代码块格式和内容质量
"""
import re
import os
from pathlib import Path

def check_code_blocks(content):
    """检查代码块格式"""
    issues = []

    # 检测代码块模式
    pattern = r'```(\w*)\n([\s\S]*?)\n```'
    matches = re.finditer(pattern, content, re.MULTILINE)

    for match in matches:
        lang = match.group(1).strip()
        code = match.group(2)

        # 检查1: 代码块是否有语言标识
        if not lang:
            # 检查是否确实是代码块（至少有一定长度）
            if len(code) > 20:
                issues.append(f"代码块缺少语言标识: ```{code[:50]}...```")
        # 检查2: 常见语言标识是否正确
        elif lang.lower() in ['c++', 'cpp', 'cxx']:
            issues.append(f"建议使用 'cpp' 而不是 '{lang}'")
        elif lang.lower() in ['sh', 'shell', 'bash']:
            if lang.lower() != 'bash':
                issues.append(f"建议使用 'bash' 而不是 '{lang}'")
        elif lang.lower() in ['py', 'python', 'python3']:
            if lang.lower() != 'python':
                issues.append(f"建议使用 'python' 而不是 '{lang}'")
        elif lang.lower() in ['js', 'javascript']:
            if lang.lower() != 'javascript':
                issues.append(f"建议使用 'javascript' 而不是 '{lang}'")

        # 检查3: 代码块内容是否有明显问题
        if lang and code.strip():
            # 检查是否有过长的单行
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                if len(line) > 200:
                    issues.append(f"第{i}行过长 ({len(line)} 字符)，考虑分行")

    return issues

def check_content_quality(content, filename):
    """检查内容质量"""
    issues = []

    # 检查1: 是否有明显的TODO或FIXME
    if 'TODO' in content or 'FIXME' in content or 'XXX' in content:
        issues.append("文章中包含TODO/FIXME/XXX标记")

    # 检查2: 检查是否有重复的段落
    paragraphs = content.split('\n\n')
    seen = set()
    for p in paragraphs:
        p_clean = p.strip()
        if len(p_clean) > 50:
            if p_clean in seen:
                issues.append(f"发现重复段落: {p_clean[:50]}...")
            seen.add(p_clean)

    # 检查3: 检查链接格式
    bad_links = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
    for link_text in bad_links:
        if not link_text.strip():
            issues.append("发现空链接文本")

    # 检查4: 检查是否有过时的技术版本引用
    outdated_refs = [
        'Python 2', 'Python 2.7', 'Python3.4', 'Python3.5',
        'Node 6', 'Node 8',
    ]
    for ref in outdated_refs:
        if ref in content:
            issues.append(f"可能包含过时的技术引用: {ref}")

    return issues

def check_front_matter(content):
    """检查front matter完整性"""
    issues = []

    # 提取front matter
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        issues.append("缺少front matter")
        return issues

    front_matter = match.group(1)

    # 检查必需字段
    required_fields = ['title', 'date', 'draft']
    for field in required_fields:
        if f'{field}:' not in front_matter:
            issues.append(f"front matter缺少必需字段: {field}")

    # 检查推荐字段
    recommended_fields = ['description', 'tags']
    for field in recommended_fields:
        if f'{field}:' not in front_matter:
            issues.append(f"front matter缺少推荐字段: {field}")

    return issues

def analyze_post(filepath):
    """分析单篇文章"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    filename = os.path.basename(filepath)
    all_issues = []

    # 检查代码块
    code_issues = check_code_blocks(content)
    if code_issues:
        all_issues.extend([f"  [代码] {issue}" for issue in code_issues])

    # 检查内容质量
    content_issues = check_content_quality(content, filename)
    if content_issues:
        all_issues.extend([f"  [内容] {issue}" for issue in content_issues])

    # 检查front matter
    fm_issues = check_front_matter(content)
    if fm_issues:
        all_issues.extend([f"  [元数据] {issue}" for issue in fm_issues])

    # 统计信息
    code_blocks = len(re.findall(r'```\w*\n', content))
    word_count = len(content.split())

    return {
        'filename': filename,
        'issues': all_issues,
        'stats': {
            'code_blocks': code_blocks,
            'word_count': word_count,
        }
    }

def main():
    # 使用绝对路径
    script_dir = Path(__file__).parent
    posts_dir = script_dir / 'content' / 'posts'
    md_files = sorted(posts_dir.glob('*.md'))

    if not md_files:
        print(f"错误: 在 {posts_dir} 中未找到 .md 文件")
        print(f"尝试的路径: {posts_dir}")
        print(f"脚本目录: {script_dir}")
        return

    print(f"找到 {len(md_files)} 篇文章\n")
    print("=" * 80)

    total_issues = 0
    posts_with_issues = []

    for md_file in md_files:
        result = analyze_post(md_file)

        if result['issues']:
            posts_with_issues.append(result)
            total_issues += len(result['issues'])
            print(f"\n文件: {result['filename']}")
            print(f"  代码块数: {result['stats']['code_blocks']}")
            print(f"  字数: {result['stats']['word_count']}")
            print(f"  发现问题 ({len(result['issues'])}):")
            for issue in result['issues']:
                print(f"    - {issue}")
        else:
            print(f"✓ {result['filename']} - 无问题")

    print("\n" + "=" * 80)
    print(f"总结: {len(posts_with_issues)}/{len(md_files)} 篇文章存在问题，共 {total_issues} 个问题")

if __name__ == '__main__':
    main()
