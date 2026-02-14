#!/usr/bin/env python3
"""
为文章生成Jimeng纽约客风格插图
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

def generate_illustration(name, visual_strategy, caption):
    """生成单张插图"""
    output_dir = Path('static/images/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(provider='jimeng', jimeng_model='jimeng-4.5')

    print(f"🎨 正在生成: {name}...")

    try:
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio='16:9',
            max_retries=2
        )

        output_path = output_dir / f'{name}.png'
        generator.save_image(image_url, str(output_path))

        file_size = output_path.stat().st_size / 1024
        print(f"✅ {name}: {file_size:.1f} KB ({used_provider})")
        return f'/images/plots/{name}.png'

    except Exception as e:
        print(f"❌ {name} 失败: {e}")
        return None

def main():
    illustrations = [
        {
            'name': 'xray-four-layers',
            'strategy': (
                "An X-ray style cross-section view showing four distinct horizontal layers, "
                "like geological strata or architectural blueprints. Each layer represents a different "
                "level of abstraction - from surface details to deep structure. Clean geometric forms, "
                "precise linework, scientific illustration aesthetic. Black and white pen and ink, "
                "New Yorker magazine technical drawing style."
            ),
            'caption': '四层分析法透视图'
        },
        {
            'name': 'token-flow-concept',
            'strategy': (
                "A conceptual diagram showing the flow of data through a processing system. "
                "Abstract geometric shapes representing input, processing, and output stages. "
                "Arrows and connecting lines suggesting information transformation. Minimalist composition, "
                "architectural precision, elegant negative space. Black and white ink sketch, "
                "New Yorker editorial illustration style."
            ),
            'caption': 'Token流动示意'
        },
        {
            'name': 'cost-optimization',
            'strategy': (
                "An elegant composition showing balance and efficiency - perhaps a scale, "
                "a finely tuned mechanism, or abstract geometric forms in perfect equilibrium. "
                "Suggesting optimization, careful calibration, and thoughtful design. "
                "Sophisticated minimalist aesthetic with precise linework. "
                "Black and white pen and ink, New Yorker magazine illustration style."
            ),
            'caption': '成本优化的艺术'
        }
    ]

    print("=" * 50)
    print("开始生成文章插图")
    print("=" * 50)

    results = []
    for ill in illustrations:
        path = generate_illustration(ill['name'], ill['strategy'], ill['caption'])
        if path:
            results.append({'name': ill['name'], 'path': path, 'caption': ill['caption']})

    print("=" * 50)
    print("生成完成")
    print("=" * 50)

    for r in results:
        print(f"![{r['caption']}]({r['path']})")

if __name__ == '__main__':
    main()
