#!/usr/bin/env python3
"""
将生成的 HTML 发布到 Paper-Reading 目录

功能：
1. 复制 HTML 到目标目录（带时间戳命名）
2. 支持自定义输出路径
3. 自动创建目标目录
"""
import shutil
from pathlib import Path
import sys
import re
from datetime import datetime


def get_paper_id_from_path(file_path: Path) -> str:
    """从文件路径提取 paper_id"""
    # 尝试从路径中提取，如: papers/SOTIF_2025/ -> SOTIF_2025
    path_parts = file_path.parts

    for i, part in enumerate(path_parts):
        if part == "papers" and i + 1 < len(path_parts):
            return path_parts[i + 1]

    # 如果找不到，尝试从文件名提取
    # 格式: SOTIF_Fuzzy_Cause_Trees_2025_xxx.html
    match = re.match(r'([A-Z]+[_\w]*_\d{4})', file_path.stem)
    if match:
        return match.group(1)

    # 默认：从文件名提取前几个单词
    words = re.findall(r'[A-Z][a-z]+', file_path.stem)
    if len(words) >= 2:
        return '_'.join(words[:2]) + '_2025'

    return "Unknown_Paper"


def generate_output_filename(paper_id: str, skill_name: str = "best-qiaomu",
                            model: str = "jimeng-4.5", timestamp: str = None) -> str:
    """生成输出文件名

    格式: {paper_id}_{skill}_{model}_{timestamp}.{ext}
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{paper_id}_{skill_name}_{model}_{timestamp}"


def publish_to_paper_reading(
    source_file: Path,
    output_dir: Path = None,
    skill_name: str = "best-qiaomu",
    model: str = "jimeng-4.5",
    dry_run: bool = False
) -> Path:
    """发布文件到 Paper-Reading 目录

    Args:
        source_file: 源文件路径（HTML）
        output_dir: 输出目录（如果为 None，使用默认路径）
        skill_name: 技能名称
        model: 使用的模型
        dry_run: 是否只是模拟运行

    Returns:
        输出文件路径
    """
    # 默认输出目录
    if output_dir is None:
        output_dir = Path("/Users/sun1/Gitlab/Personal/Experimental/Paper-Reading/paper_html")

    # Enforce HTML-only publishing (we no longer generate PDFs; keep behavior predictable).
    if source_file.suffix.lower() != ".html":
        raise ValueError(f"Only .html is supported: {source_file}")

    # 获取 paper_id
    paper_id = get_paper_id_from_path(source_file)

    # 创建目标目录
    target_dir = output_dir / paper_id
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    # 生成输出文件名
    ext = source_file.suffix
    base_name = generate_output_filename(paper_id, skill_name, model)
    target_file = target_dir / f"{base_name}{ext}"

    if dry_run:
        print(f"[DRY RUN] 会复制: {source_file}")
        print(f"[DRY RUN] 到: {target_file}")
        return target_file

    # 复制文件
    shutil.copy2(source_file, target_file)
    print(f"✅ 已复制到: {target_file}")

    return target_file


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='将 HTML 发布到 Paper-Reading 目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认路径
  python publish_to_paper_reading.py paper.html

  # 指定输出目录
  python publish_to_paper_reading.py paper.html --output /custom/path

  # 指定模型
  python publish_to_paper_reading.py paper.html --model jimeng-4.1
        """
    )
    parser.add_argument('files', nargs='+', help='要发布的文件（HTML）')
    parser.add_argument('--output', '-o', help='输出目录（默认: Paper-Reading/paper_html）')
    parser.add_argument('--skill', '-s', default='best-qiaomu', help='技能名称（默认: best-qiaomu）')
    parser.add_argument('--model', '-m', default='jimeng-4.5', help='配图模型（默认: jimeng-4.5）')
    parser.add_argument('--dry-run', '-n', action='store_true', help='模拟运行，不实际复制')

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    for source_file in args.files:
        source_path = Path(source_file)
        if not source_path.exists():
            print(f"⚠️  文件不存在: {source_path}")
            continue

        try:
            publish_to_paper_reading(
                source_path,
                output_dir=output_dir,
                skill_name=args.skill,
                model=args.model,
                dry_run=args.dry_run
            )
        except Exception as e:
            print(f"❌ 处理 {source_path} 时出错: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
