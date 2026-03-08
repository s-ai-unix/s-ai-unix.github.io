#!/usr/bin/env python3
"""
NanoClaw 记忆系统演进配图生成
使用 image_api 共享库生成纽约客风格配图
"""
import os
import sys
from pathlib import Path

# 设置 API 凭证（从本地配置加载）
local_config_path = os.path.expanduser("~/.config/claude-skills/write-tech-blog/.env")
if os.path.exists(local_config_path):
    with open(local_config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
else:
    # 备用凭证
    os.environ['JIMENG_API_KEY'] = 'sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW'
    os.environ['JIMENG_SESSION_ID'] = 'sk-S51NEPFTWvJmyQE5oiZp21BruJxV7APdH28zRsiRimSOKjcW'
    os.environ['JIMENG_API_URL'] = 'https://newapi.aisonnet.org/v1'

# 加载共享库
shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)

from image_api import ImageGenerator

# 配置参数
config_path = Path("content/posts/visual_config_nanoclaw_memory.json")

# 图表描述（英文，避免文字出现在图片中）
illustrations = [
    {
        "file": "v1-file-memory.png",
        "visual": "A minimalist editorial illustration showing a vintage leather-bound journal or diary resting on a clean wooden desk. Loose handwritten pages are scattered around, suggesting human note-taking. Include a fountain pen nearby. Clean ink lines with cross-hatching for shadows. The composition feels nostalgic and slightly chaotic, representing file-based manual memory. Black and white pen and ink style, New Yorker magazine illustration.",
        "caption": "文件记忆时代：人工维护的日记本"
    },
    {
        "file": "v2-architecture-comparison.png",
        "visual": "A split-screen conceptual diagram. Left side: a traditional filing cabinet with overflowing papers suggesting chaotic storage. Right side: a modern network visualization with interconnected geometric nodes, flowing data streams, and glowing crystalline structures suggesting vector embeddings. An elegant transformation arrow connects the two halves. Architectural precision with clean lines, minimalist composition. Black and white ink sketch style with subtle red accent.",
        "caption": "架构演进：从静态文件到动态向量"
    },
    {
        "file": "memory-extraction-pipeline.png",
        "visual": "An abstract representation of a memory extraction pipeline. On the left, raw conversation bubbles enter a geometric processing unit. The center shows multiple filtering and refinement stages represented by abstract geometric or mechanical shapes. On the right, crystalline gems or perfect geometric forms emerge. The composition suggests transformation from chaos to order. Elegant, minimalist, with subtle technical diagrams. Black and white line art.",
        "caption": "记忆提取管线：从对话到结构化知识"
    },
    {
        "file": "deduplication-merge.png",
        "visual": "A visual metaphor for memory merging. Two overlapping translucent organic shapes or circles gradually merge into one unified form. The overlapping region glows or has special hatching patterns suggesting synthesis. Include subtle geometric elements representing distance measurement or similarity. Clean, elegant, minimalist composition with mathematical undertones. Black and white ink sketch.",
        "caption": "去重与合并：相似记忆的智能融合"
    },
    {
        "file": "tiered-storage-layers.png",
        "visual": "A layered architectural diagram showing three horizontal strata. Top layer: tiny crystalline structures or minimal geometric icons floating in empty space. Middle layer: medium-sized architectural forms or geometric blocks. Bottom layer: deep foundation with complex detailed structures or bookshelves. The three layers suggest a pyramid of abstraction from summary to detail. Clean architectural drawing style with Apple-style color accents. Black and white line art.",
        "caption": "分层存储：L0/L1/L2 三级抽象"
    }
]

# 输出目录
output_dir = Path("static/images/illustrations")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📝 配图文件: {config_path}")
print(f"📁 输出目录: {output_dir.absolute()}")
print(f"🎨 配图数量: {len(illustrations)}")
print()

# 创建生成器（使用 gemini-2.5-flash-image，更稳定）
generator = ImageGenerator(provider='jimeng', jimeng_model='gemini-2.5-flash-image')

# 执行生成
print("🎨 开始生成纽约客风格配图...\n")

success_count = 0
for item in illustrations:
    output_path = output_dir / item['file']
    print(f"📝 生成: {item['caption']}")
    print(f"   视觉描述: {item['visual'][:80]}...")

    try:
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=item['visual'],
            caption="",  # 空字符串，确保图片中无文字
            aspect_ratio='16:9',
            max_retries=2
        )

        # 保存图片
        generator.save_image(image_url, str(output_path))

        file_size = output_path.stat().st_size / 1024
        print(f"   ✅ {output_path.name} ({file_size:.1f} KB) - 使用: {used_provider}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ {output_path.name}: {e}")
    print()

print(f"\n✅ 完成! 生成 {success_count}/{len(illustrations)} 张配图")
print(f"📁 保存位置: {output_dir.absolute()}")
