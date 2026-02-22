#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# 加载共享库
shared_lib_path = str(Path.home() / '.agents' / 'skills' / 'shared-lib')
if not os.path.exists(shared_lib_path):
    shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)

# 优先从本地配置加载
from dotenv import load_dotenv
local_config_path = os.path.expanduser("~/.config/claude-skills/write-tech-blog/.env")
if os.path.exists(local_config_path):
    load_dotenv(local_config_path)

api_key = os.getenv("JIMENG_API_KEY") or os.getenv("API_KEY")
os.environ['JIMENG_API_KEY'] = api_key or 'sk-xxxxxxxx'
os.environ['JIMENG_API_URL'] = 'https://newapi.aisonnet.org/v1'

try:
    from image_api import ImageGenerator
except ImportError:
    print("Cannot import image_api. Please make sure shared-lib is correctly linked.")
    sys.exit(1)

def main():
    output_dir = Path('static/images/covers')
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(provider='jimeng', jimeng_model='gemini-2.5-flash-image')

    visual_strategy = (
        "A conceptual diagram showing the distribution of prime numbers evolving into complex wave patterns on a complex plane. "
        "Abstract geometric shapes representing mathematical zeroes on a critical line. "
        "Minimalist composition, architectural precision, elegant negative space. "
        "Black and white ink sketch, New Yorker editorial illustration style."
    )
    caption = "黎曼猜想与零点之舞"

    print("🎨 正在生成纽约客风格封面图...")
    try:
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio='16:9',
            max_retries=2
        )
        
        output_path = output_dir / 'riemann-hypothesis-cover.png'
        generator.save_image(image_url, str(output_path))
        print(f"✅ 封面图已生成: {output_path}")
        
        # 压缩图片
        import subprocess
        subprocess.run(['pngquant', '--quality=70-85', '--force', '--output', str(output_path), str(output_path)], check=False)
        print("压缩完成。")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")

if __name__ == '__main__':
    main()
