#!/usr/bin/env python3
"""
对比 Jimeng 和 Gemini Flash 的纽约客风格生图效果
使用 aisonnet 网关，不同模型
"""

import os
import sys
import requests
import re
from pathlib import Path

# API 配置
API_KEY = "sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW"
API_URL = "https://newapi.aisonnet.org/v1/chat/completions"


def generate_image_via_aisonnet(model, prompt, aspect_ratio="16:9", timeout=120):
    """通过 aisonnet 网关生成图片"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "extra_body": {"imageConfig": {"aspectRatio": aspect_ratio}},
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f'{{"imageConfig": {{"aspectRatio": "{aspect_ratio}"}}}}',
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    response = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # 从 markdown 格式提取图片 URL
    url_match = re.search(r"https?://[^\s\)]+", content)
    if url_match:
        return url_match.group(0)
    else:
        raise ValueError(f"未在响应中找到图片 URL: {content}")


def download_image(url, output_path):
    """下载图片"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def main():
    """主函数"""
    output_dir = Path("static/images/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 视觉描述
    visual_strategy = (
        "A minimalist editorial illustration showing a digital meter or gauge "
        "with flowing streams of abstract text and data tokens passing through it. "
        "The composition includes geometric shapes representing input and output flows, "
        "with a subtle cache/memory element in the background. Clean lines, "
        "architectural precision, sophisticated intellectual atmosphere. "
        "Monochromatic pen and ink style with cross-hatching details."
    )

    caption = "minGPT: Transformer架构解析"

    print("=" * 60)
    print("🎨 Jimeng vs Gemini Flash 纽约客风格生图对比")
    print("   (使用 aisonnet 网关，不同模型)")
    print("=" * 60)
    print(f"\n主题: {caption}")
    print(f"API 端点: {API_URL}")
    print()

    # 构建完整提示词
    style_prompt = "Black and white pen and ink sketch, loose confident strokes, intentional white space, New Yorker magazine illustration style, minimalist, elegant"
    full_prompt = f"Generate an image: {visual_strategy}. {style_prompt}. Chinese caption at bottom: {caption}"

    results = []

    # 1. 生成 jimeng-4.5 图片
    print("[1/2] 正在生成 Jimeng-4.5 图片...")
    print("-" * 40)
    try:
        image_url = generate_image_via_aisonnet(
            model="jimeng-4.5",
            prompt=full_prompt,
            aspect_ratio="16:9",
            timeout=180  # jimeng 可能需要更长时间
        )

        output_path = output_dir / "mingpt-jimeng-4.5.png"
        download_image(image_url, output_path)

        file_size = output_path.stat().st_size / 1024
        print(f"✅ Jimeng-4.5 生成成功!")
        print(f"   文件: {output_path.name}")
        print(f"   大小: {file_size:.1f} KB")
        print(f"   模型: jimeng-4.5")
        results.append(("Jimeng-4.5", str(output_path), file_size))
    except Exception as e:
        print(f"❌ Jimeng-4.5 生成失败: {e}")

    print()

    # 2. 生成 Gemini Flash 图片
    print("[2/2] 正在生成 Gemini Flash 图片...")
    print("-" * 40)
    try:
        image_url = generate_image_via_aisonnet(
            model="gemini-2.5-flash-image",
            prompt=full_prompt,
            aspect_ratio="16:9",
            timeout=120
        )

        output_path = output_dir / "mingpt-gemini-flash.png"
        download_image(image_url, output_path)

        file_size = output_path.stat().st_size / 1024
        print(f"✅ Gemini Flash 生成成功!")
        print(f"   文件: {output_path.name}")
        print(f"   大小: {file_size:.1f} KB")
        print(f"   模型: gemini-2.5-flash-image")
        results.append(("Gemini Flash", str(output_path), file_size))
    except Exception as e:
        print(f"❌ Gemini Flash 生成失败: {e}")

    print()
    print("=" * 60)
    print("📊 生成结果汇总")
    print("=" * 60)

    if results:
        for name, path, size in results:
            print(f"\n{name}:")
            print(f"  路径: {path}")
            print(f"  大小: {size:.1f} KB")

        print("\n" + "=" * 60)
        print("✅ 对比完成!")
        print("=" * 60)
        print("\n📁 图片文件位置:")
        for _, path, _ in results:
            print(f"   {path}")
    else:
        print("\n❌ 所有生成尝试均失败")


if __name__ == "__main__":
    main()
