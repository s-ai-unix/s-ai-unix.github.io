#!/usr/bin/env python3
"""测试 jimeng-4.1 模型生成图片"""

import os
import sys

# 添加 shared-lib 到路径
sys.path.insert(0, os.path.expanduser("~/.claude/skills/shared-lib"))

from image_api import ImageGenerator

# 手动加载环境变量
env_file = os.path.expanduser("~/.config/claude-skills/write-tech-blog/.env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# 使用 jimeng-4.1 模型
generator = ImageGenerator(provider="jimeng", jimeng_model="jimeng-4.1")

print("🎨 使用 jimeng-4.1 模型生成图片...")
print(f"   API URL: {generator.jimeng_api_url}")
print(f"   Model: {generator.jimeng_model}")

try:
    image_url, model = generator.generate_newyorker_style(
        visual_strategy="A wise old mathematician sitting at a desk, surrounded by floating mathematical symbols and equations, warm afternoon light from window, New Yorker magazine illustration style",
        caption="数学家的下午",
        aspect_ratio="16:9",
    )
    print(f"✅ 生成成功!")
    print(f"   使用模型: {model}")
    print(f"   图片URL: {image_url}")

    # 保存图片
    output_path = os.path.expanduser(
        "~/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/test-jimeng-4.1.png"
    )
    saved_path = generator.save_image(image_url, output_path)
    print(f"   已保存到: {saved_path}")

    # 检查文件大小
    size = os.path.getsize(saved_path) / 1024
    print(f"   文件大小: {size:.1f} KB")

except Exception as e:
    print(f"❌ 生成失败: {e}")
    import traceback

    traceback.print_exc()
