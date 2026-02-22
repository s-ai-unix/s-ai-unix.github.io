#!/Users/sun1/miniconda3/envs/py3.13env/bin/python3
"""
检查文章中的数学公式是否符合 LATEX-MATH.md 规范
在 Hugo 构建前执行，捕获潜在的公式渲染问题
"""
import re
import sys
from pathlib import Path
from collections import defaultdict


def check_article(file_path):
    """检查文章中的公式问题"""
    content = Path(file_path).read_text(encoding='utf-8')
    issues = defaultdict(list)
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # 检查行间公式中的下划线（$$...$$）
        block_formulas = re.findall(r'\$\$([\s\S]*?)\$\$', line)
        for formula in block_formulas:
            # 检查是否有未转义的下划线（前面没有反斜杠的 _）
            unescaped_underscores = re.findall(r'(?<!\\)_', formula)
            if unescaped_underscores:
                issues['未转义的下划线'].append({
                    'line': line_num,
                    'content': line.strip()[:100],
                    'fix': '将 _ 替换为 \\\_'
                })

            # 检查是否使用了 \boldsymbol（MathJax 不支持）
            if '\\boldsymbol' in formula:
                issues['使用了不支持的 \\boldsymbol'].append({
                    'line': line_num,
                    'content': line.strip()[:100],
                    'fix': '使用 \\mathbf 替代 \\boldsymbol'
                })

            # 检查星号上标
            if '^*' in formula or '^{*}' in formula:
                issues['星号上标可能导致错误'].append({
                    'line': line_num,
                    'content': line.strip()[:100],
                    'fix': '将 ^* 替换为 ^{\\ast}'
                })

        # 检查行内公式
        inline_formulas = re.findall(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', line)
        for formula in inline_formulas:
            # 同样检查未转义的下划线
            unescaped_underscores = re.findall(r'(?<!\\)_', formula)
            if len(unescaped_underscores) > 0:
                # 排除公式外的文本
                if not any(cmd in formula for cmd in ['\\_', '\\text']):
                    issues['行内公式未转义的下划线'].append({
                        'line': line_num,
                        'content': line.strip()[:100],
                        'fix': '将 _ 替换为 \\\_'
                    })

    return issues


def main():
    if len(sys.argv) < 2:
        # 检查最新创建的文章
        posts_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/content/posts')
        md_files = sorted(posts_dir.glob('*.md'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not md_files:
            print("❌ 未找到 Markdown 文件")
            sys.exit(1)
        target_file = md_files[0]
    else:
        target_file = Path(sys.argv[1])

    print(f"🔍 检查文件: {target_file}")
    print("=" * 60)

    issues = check_article(target_file)

    if not issues:
        print("✅ 所有公式检查通过！")
        sys.exit(0)

    print(f"❌ 发现 {len(issues)} 类问题:\n")

    error_count = 0
    for category, items in issues.items():
        print(f"【{category}】")
        for item in items[:3]:  # 只显示前3个
            print(f"  行 {item['line']}: {item['content']}")
            print(f"  💡 修复: {item['fix']}")
            print()
            error_count += 1
        if len(items) > 3:
            print(f"  ... 还有 {len(items) - 3} 个类似问题")
        print()

    print("=" * 60)
    print(f"总计: {error_count} 个问题需要修复")
    print("\n请先修复上述问题，再执行 Hugo 构建。")
    sys.exit(1)


if __name__ == '__main__':
    main()
