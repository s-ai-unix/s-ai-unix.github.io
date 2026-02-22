#!/Users/sun1/miniconda3/envs/py3.13env/bin/python3
"""
文章发布前完整检查流程
必须在 Hugo 构建前执行
"""
import os
import sys
import subprocess
from pathlib import Path


def check_encoding(file_path):
    """检查编码问题"""
    content = Path(file_path).read_text(encoding='utf-8')
    if '\ufffd' in content:
        print("❌ 发现替换字符 (�)")
        return False
    print("✅ 编码检查通过")
    return True


def check_math_formulas(file_path):
    """检查数学公式"""
    result = subprocess.run(
        [sys.executable, 'scripts/check_math_formulas.py', file_path],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        return False
    return True


def check_images(file_path):
    """检查图片是否存在"""
    content = Path(file_path).read_text(encoding='utf-8')
    import re

    images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
    missing = []

    for alt, path in images:
        # 转换为实际文件路径
        if path.startswith('/images/'):
            actual_path = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static') / path.lstrip('/')
            if not actual_path.exists():
                missing.append((alt, path, actual_path))

    if missing:
        print("❌ 缺失的图片:")
        for alt, path, actual in missing:
            print(f"   - {path}")
        return False

    print(f"✅ 图片检查通过 ({len(images)} 张图片)")
    return True


def check_cover_image(file_path):
    """检查封面图"""
    content = Path(file_path).read_text(encoding='utf-8')

    if 'cover:' not in content:
        print("⚠️  未配置封面图")
        return True

    # 提取封面路径
    import re
    cover_match = re.search(r'image:\s*"([^"]+)"', content)
    if cover_match:
        cover_path = cover_match.group(1)
        actual_path = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static') / cover_path
        if not actual_path.exists():
            print(f"❌ 封面图不存在: {cover_path}")
            return False

        # 检查文件大小
        size_kb = actual_path.stat().st_size / 1024
        if size_kb < 10:
            print(f"❌ 封面图太小 ({size_kb:.1f} KB)")
            return False

        print(f"✅ 封面图检查通过 ({size_kb:.1f} KB)")

    return True


def main():
    if len(sys.argv) < 2:
        posts_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/content/posts')
        md_files = sorted(posts_dir.glob('*.md'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not md_files:
            print("❌ 未找到 Markdown 文件")
            sys.exit(1)
        target_file = md_files[0]
    else:
        target_file = Path(sys.argv[1])

    print("=" * 60)
    print(f"📋 发布前检查: {target_file.name}")
    print("=" * 60)

    checks = [
        ("编码检查", check_encoding),
        ("公式检查", check_math_formulas),
        ("图片检查", check_images),
        ("封面检查", check_cover_image),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n🔍 {name}...")
        if not check_func(target_file):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过！可以执行 Hugo 构建。")
        print("\n下一步:")
        print("  hugo --minify")
        sys.exit(0)
    else:
        print("❌ 检查未通过，请先修复上述问题。")
        sys.exit(1)


if __name__ == '__main__':
    main()
