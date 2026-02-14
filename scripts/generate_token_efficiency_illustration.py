#!/usr/bin/env python3
"""
生成Token效率对比的Jimeng插图（替换Plotly图表）
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
    output_dir = Path('static/images/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(provider='jimeng', jimeng_model='jimeng-4.5')

    # Token效率对比的视觉描述
    visual_strategy = (
        "A conceptual comparison diagram showing three vertical columns or bars of different heights, "
        "representing efficiency measurements. Abstract geometric forms suggesting data comparison - "
        "perhaps stylized containers or vessels of varying capacity. Clean minimalist composition with "
        "precise linework. Include subtle typographic or symbolic elements suggesting 'more' vs 'less'. "
        "Elegant negative space, architectural precision. Black and white pen and ink sketch, "
        "New Yorker magazine technical illustration style."
    )

    caption = "不同语言的Token效率对比"

    print("🎨 正在生成Token效率对比图...")

    try:
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio='16:9',
            max_retries=2
        )

        output_path = output_dir / 'token-efficiency-comparison.png'
        generator.save_image(image_url, str(output_path))

        file_size = output_path.stat().st_size / 1024
        print(f"✅ 已生成: {output_path} ({file_size:.1f} KB)")

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
