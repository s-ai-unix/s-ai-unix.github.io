#!/usr/bin/env python3
"""
Hugo Blog LaTeX 公式修复脚本
专门针对 Hugo + MathJax 环境的数学公式问题
"""
import re
import sys
from pathlib import Path


def check_latex_issues(file_path):
    """检查 LaTeX 公式问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    issues = {
        'unescaped_underscore': [],
        'wrong_partial': [],
        'boldsymbol': [],
        'star_issues': [],
        'pipe_issues': [],
    }

    for i, line in enumerate(lines, 1):
        # 检查未转义的下划线（排除数学公式内部）
        if re.search(r'(?<!\\)(?<!\$)_.*_(?![\$])', line) and not re.search(r'\$.*\$', line):
            issues['unescaped_underscore'].append(i)

        # 检查错误的 partial 格式: \partial{x} 而非 \partial_{x}
        if re.search(r'\\partial\{[^}]+\}', line):
            issues['wrong_partial'].append(i)

        # 检查 boldsymbol（Hugo/MathJax 不支持）
        if r'\boldsymbol' in line or r'\bm{' in line:
            issues['boldsymbol'].append(i)

        # 检查未包裹的 *（可能被解析为斜体）
        if re.search(r'\$[^$]*\*[^$]*\$', line):
            issues['star_issues'].append(i)

        # 检查 || 范数符号（可能冲突）
        if re.search(r'\$\|[^|]+\|\$', line):
            issues['pipe_issues'].append(i)

    return issues


def fix_latex_issues(file_path):
    """修复 LaTeX 公式问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    fixes_applied = []

    # 修复 1: \partial{x} -> \partial_{x}
    if re.search(r'\\partial\{[^}]+\}', content):
        content = re.sub(r'\\partial\{([^}]+)\}', r'\\partial_{\1}', content)
        fixes_applied.append("修复 \\partial{} 格式为 \\partial_{}")

    # 修复 2: \boldsymbol{x} -> \mathbf{x}
    if r'\boldsymbol' in content:
        content = re.sub(r'\\boldsymbol\{([^}]+)\}', r'\\mathbf{\1}', content)
        fixes_applied.append("替换 \\boldsymbol 为 \\mathbf")

    # 修复 3: x^* -> x^{\ast}
    content = re.sub(r'(\w+)\^\*(?![{])', r'\1^{\ast}', content)

    # 修复 4: ||x|| -> \lVert x \rVert
    content = re.sub(r'\$\|([^|]+)\|\$', r'$\\lVert \1 \\rVert$', content)

    # 写回文件
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return fixes_applied
    return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/latex_hugo_fix.py <file.md>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    print(f"🔍 检查文件: {file_path.name}")
    print("=" * 60)

    # 检查问题
    issues = check_latex_issues(file_path)

    has_issues = False
    for issue_type, lines in issues.items():
        if lines:
            has_issues = True
            print(f"\n❌ 发现 {issue_type}:")
            for line in lines[:5]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_lines = f.readlines()
                    print(f"  行 {line}: {content_lines[line-1].strip()[:80]}")
            if len(lines) > 5:
                print(f"  ... 还有 {len(lines) - 5} 个类似问题")

    if not has_issues:
        print("\n✅ 未发现 LaTeX 公式问题")
        return

    # 自动修复
    print("\n🔧 尝试自动修复...")
    fixes = fix_latex_issues(file_path)

    if fixes:
        print("\n✅ 应用以下修复:")
        for fix in fixes:
            print(f"  - {fix}")
        print(f"\n✅ 文件已更新: {file_path.name}")
        print("\n💡 下一步:")
        print("  1. 运行: hugo --minify")
        print("  2. 检查浏览器: Ctrl+F5 强制刷新")
    else:
        print("\n⚠️  无法自动修复，请手动检查上述问题")


if __name__ == "__main__":
    main()
