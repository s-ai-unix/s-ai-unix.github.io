#!/usr/bin/env python3
"""
图表提取质量检查脚本

检查提取的图表是否符合质量标准：
1. 图片尺寸合理性（太小或太大都不正常）
2. 图片内容是否包含不必要的周边文本
3. 图片清晰度（通过文件大小估算）
4. 图片边界合理性
"""
import fitz  # PyMuPDF
from pathlib import Path
import sys
import json
from PIL import Image
import io


def load_images(images_dir: Path):
    """加载所有提取的图片"""
    images = {}
    for img_file in images_dir.glob("*.png"):
        try:
            with Image.open(img_file) as img:
                images[img_file.name] = {
                    'file': img_file,
                    'pil': img,
                    'size_bytes': img_file.stat().st_size,
                    'width': img.width,
                    'height': img.height,
                    'aspect_ratio': img.width / img.height if img.height > 0 else 0
                }
        except Exception as e:
            print(f"⚠️  无法加载 {img_file.name}: {e}")
    return images


def check_size_reasonable(img_info: dict, img_type: str = "figure") -> dict:
    """检查图片尺寸是否合理"""
    issues = []
    warnings = []

    width, height = img_info['width'], img_info['height']
    aspect = img_info['aspect_ratio']

    # 尺寸检查
    if width < 100:
        issues.append(f"宽度过小: {width}px（至少需要100px）")
    if height < 100:
        issues.append(f"高度过小: {height}px（至少需要100px）")

    if img_type == "figure":
        # Figure 通常是横向或方形
        if aspect < 0.2:
            issues.append(f"宽高比过窄: {aspect:.2f}（Figure应该是横向或方形）")
        if aspect > 5.0:
            warnings.append(f"宽高比过宽: {aspect:.2f}（可能包含过多周边内容）")
    elif img_type == "table":
        # Table 通常是横向
        if aspect < 0.3:
            issues.append(f"宽高比过窄: {aspect:.2f}（Table应该是横向）")

    # 文件大小检查（用于估算清晰度）
    size_kb = img_info['size_bytes'] / 1024
    pixel_count = width * height
    bytes_per_pixel = img_info['size_bytes'] / pixel_count if pixel_count > 0 else 0

    if size_kb < 10:
        warnings.append(f"文件过小: {size_kb:.1f}KB（可能图片不完整）")
    if bytes_per_pixel < 0.1:
        warnings.append(f"压缩率过高: {bytes_per_pixel:.3f} bytes/pixel（可能模糊）")

    return {
        'issues': issues,
        'warnings': warnings
    }


def check_border_noise(img_info: dict, pdf_path: Path) -> dict:
    """检查图片边界是否有不必要的噪声/文本"""
    # 这里可以用 OCR 检测，但为了简单，我们先检查图片内容
    # 如果图片顶部或底部有大量文字模式，可能是误捕获了周边文本

    warnings = []
    img = img_info['pil']

    # 转换为 RGB 如果是 RGBA
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # 检查顶部和底部的像素行（简化版）
    # 实际应该用 OCR 来检测文本
    try:
        # 简单检查：如果顶部或底部有大量非白色像素，可能包含了文本
        top_strip = img.crop((0, 0, img.width, min(50, img.height)))
        bottom_strip = img.crop((0, max(0, img.height - 50), img.width, img.height))

        # 计算非白色像素比例（简化）
        def count_non_white(strip_img):
            pixels = list(strip_img.getdata())
            non_white = sum(1 for p in pixels if sum(p[:3]) < 700)  # 不是接近白色
            return non_white / len(pixels) if pixels else 0

        top_content = count_non_white(top_strip)
        bottom_content = count_non_white(bottom_strip)

        if top_content > 0.3:
            warnings.append(f"顶部可能有额外内容: {top_content:.1%} 非空白像素")
        if bottom_content > 0.3:
            warnings.append(f"底部可能有额外内容: {bottom_content:.1%} 非空白像素")

    except Exception as e:
        warnings.append(f"边界检查失败: {e}")

    return {'warnings': warnings}


def validate_extraction(images_dir: Path, pdf_path: Path, debug: bool = False):
    """执行完整的质量检查"""
    print(f"📂 图片目录: {images_dir}")
    print(f"📄 PDF文件: {pdf_path}")
    print()

    # 加载所有图片
    images = load_images(images_dir)
    if not images:
        print("❌ 未找到任何图片文件")
        return False

    print(f"✅ 找到 {len(images)} 张图片\n")
    print("=" * 60)

    # 打开 PDF 获取上下文
    doc = fitz.open(str(pdf_path))

    all_ok = True
    results = []

    for img_name, img_info in images.items():
        print(f"\n🖼️  {img_name}")
        print(f"   尺寸: {img_info['width']}x{img_info['height']}")
        print(f"   大小: {img_info['size_bytes'] / 1024:.1f}KB")
        print(f"   宽高比: {img_info['aspect_ratio']:.2f}")

        # 判断图片类型
        img_type = "table" if "table" in img_name.lower() else "figure"

        # 检查尺寸
        size_check = check_size_reasonable(img_info, img_type)

        # 检查边界
        border_check = check_border_noise(img_info, pdf_path)

        # 汇总问题
        issues = size_check['issues'] + border_check.get('issues', [])
        warnings = size_check['warnings'] + border_check.get('warnings', [])

        if issues:
            print(f"   ❌ 问题:")
            for issue in issues:
                print(f"      - {issue}")
            all_ok = False

        if warnings:
            print(f"   ⚠️  警告:")
            for warning in warnings:
                print(f"      - {warning}")

        if not issues and not warnings:
            print(f"   ✅ 质量良好")

        results.append({
            'name': img_name,
            'issues': issues,
            'warnings': warnings,
            'ok': not issues
        })

    doc.close()

    print("\n" + "=" * 60)
    print(f"\n📊 质量检查汇总:")
    print(f"   总数: {len(results)}")
    print(f"   通过: {sum(1 for r in results if r['ok'])}")
    print(f"   有问题: {sum(1 for r in results if not r['ok'])}")

    if all_ok:
        print("\n✅ 所有图片质量检查通过！")
        return True
    else:
        print("\n⚠️  部分图片存在问题，建议检查")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='检查提取的图表质量',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('images_dir', help='图片目录路径')
    parser.add_argument('pdf_path', help='原始PDF路径')
    parser.add_argument('--debug', '-d', action='store_true', help='调试模式')
    parser.add_argument('--json', '-j', action='store_true', help='输出JSON格式结果')

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    pdf_path = Path(args.pdf_path)

    if not images_dir.exists():
        print(f"❌ 图片目录不存在: {images_dir}")
        sys.exit(1)

    if not pdf_path.exists():
        print(f"❌ PDF文件不存在: {pdf_path}")
        sys.exit(1)

    result = validate_extraction(images_dir, pdf_path, args.debug)

    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
