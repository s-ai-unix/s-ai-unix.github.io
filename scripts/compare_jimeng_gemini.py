#!/usr/bin/env python3
"""
对比 Jimeng 和 Gemini Flash 的纽约客风格生图效果
"""

import os
import sys
from pathlib import Path

# 加载共享库
shared_lib_path = str(Path.home() / ".claude" / "skills" / "shared-lib")
sys.path.insert(0, shared_lib_path)

# 设置环境变量
os.environ["JIMENG_API_KEY"] = "sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW"
os.environ["JIMENG_SESSION_ID"] = "sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW"
os.environ["JIMENG_API_URL"] = "https://newapi.aisonnet.org/v1"

from image_api import ImageGenerator


def generate_comparison():
    """生成对比图片"""
    output_dir = Path("static/images/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 相同的视觉描述
    visual_strategy = (
        "An elegant composition showing balance and efficiency - perhaps a scale, "
        "a finely tuned mechanism, or abstract geometric forms in perfect equilibrium. "
        "Suggesting optimization, careful calibration, and thoughtful design. "
        "Sophisticated minimalist aesthetic with precise linework. "
        "Black and white pen and ink, New Yorker magazine illustration style."
    )

    caption = "成本优化的艺术"

    print("=" * 60)
    print("开始生成对比图片")
    print("=" * 60)

    results = []

    # 1. 生成 jimeng-4.5 图片
    print("\n[1/2] 正在生成 jimeng-4.5 图片...")
    try:
        generator_jimeng = ImageGenerator(provider="jimeng", jimeng_model="jimeng-4.5")
        image_url, used_provider = generator_jimeng.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio="16:9",
            max_retries=2,
        )

        output_path = output_dir / "comparison-jimeng-4.5.png"
        generator_jimeng.save_image(image_url, str(output_path))

        file_size = output_path.stat().st_size / 1024
        print(f"✅ jimeng-4.5 生成成功: {output_path.name} ({file_size:.1f} KB)")
        print(f"   实际使用模型: {used_provider}")
        results.append(("jimeng-4.5", str(output_path), file_size))
    except Exception as e:
        print(f"❌ jimeng-4.5 生成失败: {e}")

    # 2. 生成 jimeng-4.1 图片（作为对比）
    print("\n[2/2] 正在生成 jimeng-4.1 图片...")
    try:
        generator_41 = ImageGenerator(provider="jimeng", jimeng_model="jimeng-4.1")
        image_url, used_provider = generator_41.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio="16:9",
            max_retries=2,
        )

        output_path = output_dir / "comparison-jimeng-4.1.png"
        generator_41.save_image(image_url, str(output_path))

        file_size = output_path.stat().st_size / 1024
        print(f"✅ jimeng-4.1 生成成功: {output_path.name} ({file_size:.1f} KB)")
        print(f"   实际使用模型: {used_provider}")
        results.append(("jimeng-4.1", str(output_path), file_size))
    except Exception as e:
        print(f"❌ jimeng-4.1 生成失败: {e}")

    print("\n" + "=" * 60)
    print("生成完成")
    print("=" * 60)

    for name, path, size in results:
        print(f"📊 {name}: {path} ({size:.1f} KB)")

    # 尝试生成 gemini flash（通过 aisonnet 网关）
    print("\n" + "=" * 60)
    print("注意：关于 Gemini Flash")
    print("=" * 60)
    print("""
根据 image_api.py 代码：
- aisonnet 网关使用 /v1/chat/completions 端点
- 自动使用 gemini-2.5-flash-image 模型
- 这本质上就是通过 aisonnet 调用 Gemini 的图像生成能力

如果你想直接使用 Google Gemini 生成图片，需要：
1. 获取 Google Gemini API Key
2. 修改代码使用 Gemini 原生 API

当前 aisonnet 网关已经将 Gemini 能力封装进去了。
""")


if __name__ == "__main__":
    generate_comparison()
