#!/usr/bin/env python3
"""
并发版配图生成脚本 - 使用线程池加速生成

性能提升：
- 原版：串行生成，12张图 = 29秒 × 12 = 348秒（约6分钟）
- 并发版：3并发，12张图 = 29秒 × 4批 = 116秒（约2分钟）
"""
import json
import re
import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# 运行时保障：
# 若当前 python 缺少 requests（常见于系统 python），自动切换到可用的 conda python 重新执行。
def _python_has_module(py_exe, module_name):
    try:
        result = subprocess.run(
            [py_exe, "-c", f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _candidate_pythons():
    candidates = []

    env_py = os.environ.get("PAPER_INTERPRETER_PYTHON")
    if env_py:
        candidates.append(env_py)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin" / "python3"))
        candidates.append(str(Path(conda_prefix) / "bin" / "python"))

    candidates.extend([
        "/Users/sun1/miniconda3/envs/py3.13env/bin/python3",
        "/Users/sun1/miniconda3/bin/python3",
    ])

    # 去重并过滤不存在路径
    seen = set()
    uniq = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if Path(c).exists():
            uniq.append(c)
    return uniq


def _ensure_python_runtime():
    # 当前解释器可用时直接继续
    if importlib.util.find_spec("requests"):
        return

    current = sys.executable
    for py in _candidate_pythons():
        if py == current:
            continue
        if _python_has_module(py, "requests"):
            print(f"🔁 当前 Python 缺少 requests，切换到: {py}")
            os.execv(py, [py, __file__, *sys.argv[1:]])

    raise RuntimeError(
        "当前 Python 缺少 requests，且未找到可用 conda Python。"
        "请设置 PAPER_INTERPRETER_PYTHON 或安装 requests。"
    )


_ensure_python_runtime()

# 加载当前 skill 的 .env（JIMENG_API_KEY/JIMENG_API_URL 等）
from load_env import load_env_file
load_env_file()

# 添加共享库路径
shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)

from image_api import ImageGenerator

# 全局锁保护markdown文件的读写
markdown_lock = threading.Lock()


def parse_h2_sections(markdown_path):
    """解析markdown中的所有H2标题"""
    with open(markdown_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current_h2 = None
    current_line_num = 0

    for i, line in enumerate(lines, 1):
        h2_match = re.match(r'^##\s+(.+)$', line.strip())
        if h2_match:
            if current_h2:
                sections.append((current_h2, current_line_num))
            current_h2 = h2_match.group(1)
            current_line_num = i

    if current_h2:
        sections.append((current_h2, current_line_num))

    return sections


def create_visual_config_template(markdown_path, output_path="visual_config.json"):
    """
    创建视觉配置模板，供Claude填写

    输出JSON格式：
    {
        "sections": [
            {
                "h2_title": "深度的悖论",
                "visual_description": "待Claude分析填写...",
                "caption": ""
            },
            ...
        ]
    }
    """
    from pathlib import Path
    markdown_path = Path(markdown_path)
    sections = parse_h2_sections(str(markdown_path))

    config = {
        "article_title": markdown_path.stem,
        "sections": [
            {
                "h2_title": title,
                "visual_description": "待Claude分析填写...",
                "caption": ""
            }
            for title, _ in sections
        ]
    }

    output_path = markdown_path.parent / output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"✅ 配置模板已创建: {output_path}")
    print(f"📝 请Claude分析文章并填写每个section的visual_description和caption")
    return str(output_path)


def obsidian_safe_path(path):
    """
    确保路径在Obsidian中可用
    如果路径包含空格，用<>包裹（Obsidian要求）
    """
    if ' ' in path:
        return f"<{path}>"
    return path


def insert_image_into_markdown(markdown_path, h2_title, image_path):
    """在H2标题后插入图片引用（线程安全，防重复，Obsidian兼容）"""
    with markdown_lock:  # 🔒 保护文件读写
        with open(markdown_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 找到H2标题
        for i, line in enumerate(lines):
            if re.match(rf'^##\s+{re.escape(h2_title)}\s*$', line.strip()):
                # 检查下一行是否已有图片（更严格的检查）
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # 检查是否已有该图片引用
                    if next_line.startswith('![') and image_path in next_line:
                        return False  # 已存在相同图片
                    # 检查是否有其他illustration图片（避免重复插入）
                    if 'illustration_' in next_line:
                        return False  # 已有其他配图

                # 插入图片（Obsidian兼容：路径有空格时用<>包裹）
                safe_path = obsidian_safe_path(image_path)
                image_md = f"\n![{h2_title}]({safe_path})\n\n"
                lines.insert(i + 1, image_md)

                # 写回文件
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return True  # 插入成功

        return False  # 未找到标题


def generate_single_image(task):
    """生成单张图片的任务函数"""
    idx, section, markdown_path, abs_output_dir, output_dir, provider, jimeng_model, skip_existing = task

    h2_title = section['h2_title']
    visual_desc = section['visual_description']
    caption = section.get('caption', '')

    # 获取论文唯一标识符（markdown 文件名去扩展名，避免 Obsidian 索引冲突）
    paper_dir = markdown_path.parent
    paper_slug = markdown_path.stem  # 使用文件名作为唯一标识符

    # 图片文件名（包含论文标识符，全局唯一）
    image_filename = f"{paper_slug}-{idx:02d}.png"
    image_output_path = abs_output_dir / image_filename

    # 完整相对路径（从根目录开始），适用于文章移动到根目录后
    paper_id = paper_dir.name
    image_rel_path = f"01.inbox/papers/{paper_id}/{output_dir}/{image_filename}"

    result = {
        'idx': idx,
        'h2_title': h2_title,
        'success': False,
        'message': '',
        'image_path': image_rel_path,
        'elapsed': 0
    }

    # 检查是否需要分析
    if visual_desc == "待Claude分析填写..." or not visual_desc.strip():
        result['message'] = "⚠️  跳过：未填写visual_description"
        return result

    # 跳过已存在
    if skip_existing and image_output_path.exists():
        result['success'] = True
        result['message'] = "⏭️  图片已存在，跳过"
        insert_image_into_markdown(markdown_path, h2_title, image_rel_path)
        return result

    start_time = time.time()

    try:
        # 生成图片
        generator = ImageGenerator(provider=provider, jimeng_model=jimeng_model)
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=visual_desc,
            caption=caption,
            aspect_ratio='16:9',
            max_retries=2  # 控制重试次数
        )

        # 保存图片
        generator.save_image(image_url, str(image_output_path))

        # 插入到markdown
        inserted = insert_image_into_markdown(markdown_path, h2_title, image_rel_path)

        elapsed = time.time() - start_time
        result['success'] = True
        result['elapsed'] = elapsed

        if inserted:
            result['message'] = f"✅ 成功 ({used_provider}, {elapsed:.1f}秒)"
        else:
            result['message'] = f"✅ 图片生成成功，但未插入（可能已存在）({used_provider}, {elapsed:.1f}秒)"

    except Exception as e:
        elapsed = time.time() - start_time
        result['elapsed'] = elapsed
        error_msg = str(e)
        # 友好的错误信息
        if 'timeout' in error_msg.lower():
            result['message'] = f"❌ 超时: API响应时间过长"
        elif '503' in error_msg or '500' in error_msg:
            result['message'] = f"❌ API服务不可用"
        else:
            result['message'] = f"❌ 失败: {error_msg[:60]}"

    return result


def generate_from_config_parallel(
    markdown_path,
    config_path="visual_config.json",
    output_dir="illustrations",
    provider='jimeng',  # 默认使用jimeng（gemini当前不可用）
    jimeng_model='jimeng-4.5',  # 即梦模型版本（内部规范名）
    skip_existing=True,
    max_workers=3  # 并发数
):
    """
    并发生成所有配图

    Args:
        provider: API提供商（jimeng/gemini/auto）
        jimeng_model: 即梦模型版本（jimeng-4.5收费, jimeng-4.1免费）
        max_workers: 并发线程数（建议3-4，避免API限流）
    """
    # 🔧 修复：转换为绝对路径，确保paper_id能正确提取
    markdown_path = Path(markdown_path).resolve()
    config_path = markdown_path.parent / config_path

    # 1. 读取配置
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 0

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    sections = config['sections']
    print(f"\n📋 读取配置: {len(sections)} 个章节")
    print(f"⚡ 并发设置: {max_workers} 线程")

    # 2. 创建输出目录
    abs_output_dir = markdown_path.parent / output_dir
    abs_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 准备任务列表
    tasks = []
    for idx, section in enumerate(sections, 1):
        tasks.append((idx, section, markdown_path, abs_output_dir, output_dir, provider, jimeng_model, skip_existing))

    # 4. 并发生成
    success_count = 0
    total_time = 0
    results = []

    print(f"\n🚀 开始并发生成...\n")
    start_total = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(generate_single_image, task): task for task in tasks}

        # 收集结果（按完成顺序）
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

            # 实时显示进度
            print(f"[{result['idx']}/{len(tasks)}] {result['h2_title'][:20]}... {result['message']}")

            if result['success']:
                success_count += 1
                total_time += result['elapsed']

    elapsed_total = time.time() - start_total

    # 5. 排序结果并总结
    results.sort(key=lambda x: x['idx'])

    print("\n" + "="*70)
    print(f"✨ 完成！成功生成 {success_count}/{len(sections)} 张配图")
    print(f"⏱️  总耗时: {elapsed_total:.1f}秒")
    if success_count > 0:
        print(f"📊 平均每张: {total_time/success_count:.1f}秒（API时间）")
        print(f"🚀 加速比: {total_time/elapsed_total:.1f}x（理论值:{max_workers}x）")
    print(f"\n📁 图片保存在: {output_dir}")
    print(f"📄 Markdown已更新: {markdown_path.name}")

    return success_count


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='并发版纽约客风格配图生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  1. 创建配置模板：
     python generate_illustrations_v2.py --create-template article.md

  2. 批量并发生成（默认付费版4.5）：
     python generate_illustrations_v2.py article.md --workers 3

  3. 指定免费版配图（4.1模型）：
     python generate_illustrations_v2.py article.md --free --workers 3
        """
    )

    parser.add_argument('markdown', nargs='?', help='Markdown文件路径')
    parser.add_argument('--create-template', metavar='FILE',
                       help='创建配置模板（无需先分析文章）')
    parser.add_argument('--config', default='visual_config.json',
                       help='配置文件路径（默认: visual_config.json）')
    parser.add_argument('--output-dir', default='illustrations',
                       help='图片输出目录（默认: images/illustrations）')
    parser.add_argument('--provider', choices=['jimeng', 'gemini', 'auto'],
                       default='jimeng', help='图片生成API（默认: jimeng）')
    parser.add_argument('--free', action='store_true',
                       help='使用免费模型（jimeng-4.1）')
    parser.add_argument('--paid', action='store_true',
                       help='使用付费模型（jimeng-4.5）')
    parser.add_argument('--model', default=os.environ.get('JIMENG_MODEL', 'jimeng-4.5'),
                       help='指定即梦模型（默认读取 JIMENG_MODEL 或 jimeng-4.5）')
    parser.add_argument('--workers', type=int, default=3,
                       help='并发线程数（默认: 3）')
    parser.add_argument('--no-skip', action='store_true',
                       help='重新生成已存在的图片')

    args = parser.parse_args()

    # 创建模板模式
    if args.create_template:
        create_visual_config_template(args.create_template, args.config)
        return

    # 生成模式
    if not args.markdown:
        parser.print_help()
        return
    # 模型选择：默认付费版4.5，可显式选择免费版4.1
    if args.free and args.paid:
        parser.error('--free 和 --paid 不能同时使用')

    # 兼容用户常用命名（jimeng-image-*）与网关可用命名（jimeng-*）
    # 说明：当前网关对 jimeng-image-* 返回 503，需映射到 jimeng-*。
    def normalize_jimeng_model(model_name: str) -> str:
        aliases = {
            'jimeng-image-4.1': 'jimeng-4.1',
            'jimeng-image-4.5': 'jimeng-4.5',
        }
        return aliases.get(model_name, model_name)

    if args.free:
        jimeng_model = 'jimeng-4.1'
    elif args.paid:
        jimeng_model = 'jimeng-4.5'
    else:
        jimeng_model = args.model

    jimeng_model = normalize_jimeng_model(jimeng_model)

    generate_from_config_parallel(
        markdown_path=args.markdown,
        config_path=args.config,
        output_dir=args.output_dir,
        provider=args.provider,
        jimeng_model=jimeng_model,
        skip_existing=not args.no_skip,
        max_workers=args.workers
    )


if __name__ == '__main__':
    main()
