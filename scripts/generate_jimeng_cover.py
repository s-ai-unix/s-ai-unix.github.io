#!/usr/bin/env python3
"""
使用Jimeng生成纽约客风格封面图
"""
import os
import sys
from pathlib import Path

# 加载共享库
shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)

# 设置环境变量
os.environ['JIMENG_API_KEY'] = 'sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW'
os.environ['JIMENG_SESSION_ID'] = 'sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW'
os.environ['JIMENG_API_URL'] = 'https://newapi.aisonnet.org/v1'

from image_api import ImageGenerator

def main():
    # 创建输出目录
    output_dir = Path('static/images/covers')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成封面图
    generator = ImageGenerator(provider='jimeng', jimeng_model='jimeng-4.5')

    # 视觉描述 - 纽约客风格，关于API Token计费的概念图
    visual_strategy = (
        "A minimalist editorial illustration showing a digital meter or gauge "
        "with flowing streams of abstract text and data tokens passing through it. "
        "The composition includes geometric shapes representing input and output flows, "
        "with a subtle cache/memory element in the background. Clean lines, "
        "architectural precision, sophisticated intellectual atmosphere. "
        "Monochromatic pen and ink style with cross-hatching details."
    )

    caption = "API Token计费机制"

    print("🎨 正在生成纽约客风格封面图...")
    print(f"   视觉策略: {visual_strategy[:80]}...")

    try:
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio='16:9',
            max_retries=2
        )

        # 保存图片
        output_path = output_dir / 'api-token-jimeng-cover.png'
        generator.save_image(image_url, str(output_path))

        print(f"✅ 封面图已生成: {output_path}")
        print(f"   使用提供商: {used_provider}")

        # 显示文件大小
        file_size = output_path.stat().st_size / 1024
        print(f"   文件大小: {file_size:.1f} KB")

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
