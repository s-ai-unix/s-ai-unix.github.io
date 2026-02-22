#!/Users/sun1/miniconda3/envs/py3.13env/bin/python3
"""
重新生成比安基恒等式文章配图 - 严格禁止任何文字
"""
import os
import sys
import time
from pathlib import Path

# 设置环境变量
os.environ['JIMENG_API_KEY'] = os.getenv('JIMENG_API_KEY', '')
os.environ['JIMENG_SESSION_ID'] = os.getenv('JIMENG_SESSION_ID', '')
os.environ['JIMENG_API_URL'] = 'https://newapi.aisonnet.org/v1'

# 加载共享库
shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)
from image_api import ImageGenerator

# 严格禁止文字的插图配置
ILLUSTRATIONS = [
    {
        "filename": "bianchi-identity-cover.png",
        "visual_description": "An elegant editorial illustration showing abstract curved geometric surfaces flowing through space. Geodesic lines weave across the curved manifold like golden threads, forming intricate symmetrical patterns. Mathematical symbols for curvature tensors subtly integrated into the composition. Deep cosmic background with ethereal lighting. Monochromatic pen and ink sketch with subtle blue accents, New Yorker magazine style, minimalist and sophisticated. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
        "caption": ""
    },
    {
        "filename": "bianchi-ant-perspective.png",
        "visual_description": "A charming editorial illustration of a tiny ant standing on a vast curved spherical surface. The ant looks out at what appears to be a flat horizon in its immediate vicinity, while the curvature of the sphere extends into the distance. Gentle cross-hatching shows the spherical geometry. Scale contrast between the tiny ant and the immense curved world. Black and white pen and ink sketch, New Yorker editorial style, whimsical yet scientific. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
        "caption": ""
    },
    {
        "filename": "bianchi-parallel-transport.png",
        "visual_description": "A scientific editorial illustration showing a sphere with a closed triangular path drawn on its surface. An arrow vector is shown at each corner of the triangle, demonstrating how its direction changes after parallel transport around the loop. Arrows are drawn with elegant curved lines showing their rotation. Geodesic lines connect the corners. Black and white pen and ink with subtle red accents for the vectors, New Yorker magazine style, architectural precision. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
        "caption": ""
    },
    {
        "filename": "bianchi-first-identity.png",
        "visual_description": "An abstract geometric illustration showing three adjacent parallelogram faces meeting at a point, forming a corner of a cube-like structure. Curved arrows indicate cyclic parallel transport around each face. The composition emphasizes closure and symmetry. Mathematical elegance with architectural drawing style. Cross-hatching for depth and shading. Black and white pen and ink sketch, New Yorker editorial illustration style, sophisticated intellectual atmosphere. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
        "caption": ""
    },
    {
        "filename": "bianchi-second-identity.png",
        "visual_description": "A conceptual editorial illustration showing flowing curved lines representing covariant derivatives converging and canceling each other in a cyclic pattern. Abstract representation of differential geometry concepts with elegant swirling curves. Minimalist composition with intentional negative space. The curves form a subtle triangular arrangement suggesting the cyclic sum. Black and white pen and ink, New Yorker style, loose confident strokes. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
        "caption": ""
    },
    {
        "filename": "bianchi-gr-application.png",
        "visual_description": "A dramatic editorial illustration split into two halves. Left side shows abstract curved spacetime geometry with light bending around massive objects. Right side shows elegant flowing energy and matter patterns. A subtle bridge or equation connects the two halves in the center. Cosmic and elegant atmosphere. Black and white pen and ink with subtle gray wash, New Yorker magazine illustration style, sophisticated and intellectual. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
        "caption": ""
    }
]

def generate_all_illustrations():
    """重新生成所有插图，禁止任何文字"""
    output_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/illustrations')
    cover_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/covers')

    output_dir.mkdir(parents=True, exist_ok=True)
    cover_dir.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(provider='jimeng', jimeng_model='jimeng-4.5')

    print(f"🎨 重新生成 {len(ILLUSTRATIONS)} 张配图（无文字版本）...\n")

    for i, config in enumerate(ILLUSTRATIONS, 1):
        print(f"[{i}/{len(ILLUSTRATIONS)}] 生成: {config['filename']}")

        try:
            # 生成图片 - 不传caption，避免任何文字
            image_url, used_provider = generator.generate_newyorker_style(
                visual_strategy=config['visual_description'],
                caption="",  # 空标题，避免任何文字
                aspect_ratio='16:9',
                max_retries=3
            )

            # 确定保存路径
            if 'cover' in config['filename']:
                save_path = cover_dir / config['filename']
            else:
                save_path = output_dir / config['filename']

            # 保存图片
            generator.save_image(image_url, str(save_path))

            file_size = save_path.stat().st_size / 1024
            print(f"    ✅ 已保存: {save_path.name} ({file_size:.1f} KB)")
            print(f"    提供商: {used_provider}\n")

            if i < len(ILLUSTRATIONS):
                time.sleep(1)

        except Exception as e:
            print(f"    ❌ 生成失败: {e}\n")
            continue

    print("🎉 所有配图重新生成完成！")

if __name__ == '__main__':
    # 检查 API key
    if not os.environ['JIMENG_API_KEY']:
        local_config = Path.home() / '.config' / 'claude-skills' / 'write-tech-blog' / '.env'
        if local_config.exists():
            with open(local_config) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, val = line.strip().split('=', 1)
                        if key == 'JIMENG_API_KEY':
                            os.environ['JIMENG_API_KEY'] = val.strip('"\'')
                        elif key == 'JIMENG_SESSION_ID':
                            os.environ['JIMENG_SESSION_ID'] = val.strip('"\'')

    if not os.environ['JIMENG_API_KEY']:
        print("❌ 未找到 JIMENG_API_KEY")
        sys.exit(1)

    generate_all_illustrations()
