#!/Users/sun1/miniconda3/envs/py3.13env/bin/python3
"""
生成希尔伯特作用量文章配图 - 严格禁止任何文字
"""
import os
import sys
import time
from pathlib import Path

os.environ['JIMENG_API_KEY'] = os.getenv('JIMENG_API_KEY', '')
os.environ['JIMENG_SESSION_ID'] = os.getenv('JIMENG_SESSION_ID', '')
os.environ['JIMENG_API_URL'] = 'https://newapi.aisonnet.org/v1'

shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)
from image_api import ImageGenerator

ILLUSTRATIONS = [
    {
        "filename": "hilbert-action-cover.png",
        "visual_description": "An elegant editorial illustration showing abstract curved spacetime geometry with elegant mathematical curves and geodesic lines. The composition features flowing abstract surfaces representing curved manifolds, with subtle light rays bending around massive objects. Deep cosmic background with ethereal blue and silver tones. Architectural precision, sophisticated intellectual atmosphere. Black and white pen and ink sketch with subtle blue accents, New Yorker magazine style, minimalist. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
    },
    {
        "filename": "hilbert-least-action.png",
        "visual_description": "A conceptual editorial illustration showing light rays refracting through different media. Abstract geometric shapes representing layers of transparent materials with bending light paths. Elegant curved lines showing the principle of least time. Minimalist composition with clean lines and intentional negative space. Scientific yet artistic. Black and white pen and ink, New Yorker editorial style, architectural drawing precision. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
    },
    {
        "filename": "hilbert-curvature.png",
        "visual_description": "A dramatic editorial illustration of a massive sphere bending the grid lines of spacetime around it. Geodesic lines curve elegantly toward the central mass. Abstract representation of gravitational lensing with subtle light bending effects. Cosmic background with depth and atmosphere. Black and white pen and ink with subtle gray wash, New Yorker magazine illustration style, sophisticated and intellectual. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
    },
    {
        "filename": "hilbert-variation.png",
        "visual_description": "An abstract geometric illustration showing multiple curved surfaces or manifolds, with one highlighted path or surface standing out from variations. Elegant mathematical curves suggesting the calculus of variations. Subtle delta symbols integrated as abstract shapes. Minimalist composition emphasizing selection of extremal path. Black and white pen and ink sketch, New Yorker style, loose confident strokes with cross-hatching. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
    },
    {
        "filename": "hilbert-field-equation.png",
        "visual_description": "A conceptual editorial illustration split composition showing abstract geometric tensors or matrices on one side and flowing energy patterns on the other side. Elegant balance between mathematical structure and physical energy. Connecting lines or bridge between the two halves suggesting equivalence. Architectural precision with sophisticated atmosphere. Black and white pen and ink, New Yorker magazine style, minimalist and elegant. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
    },
    {
        "filename": "hilbert-cosmology.png",
        "visual_description": "A dramatic cosmic editorial illustration showing the large-scale structure of the universe. Abstract filaments and voids representing cosmic web structure. Distant galaxies as subtle points of light. Expanding geometry suggested by diverging curves. Deep space atmosphere with ethereal quality. Black and white pen and ink with subtle gray tones, New Yorker editorial illustration style, sophisticated intellectual atmosphere. NO text, NO letters, NO Chinese characters, NO typography, NO captions, NO words, NO readable text of any kind.",
    }
]

def generate_images():
    output_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/illustrations')
    cover_dir = Path('/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/covers')
    output_dir.mkdir(parents=True, exist_ok=True)
    cover_dir.mkdir(parents=True, exist_ok=True)

    generator = ImageGenerator(provider='jimeng', jimeng_model='jimeng-4.5')

    print(f"🎨 开始生成 {len(ILLUSTRATIONS)} 张配图（无文字）...\n")

    for i, config in enumerate(ILLUSTRATIONS, 1):
        print(f"[{i}/{len(ILLUSTRATIONS)}] {config['filename']}")

        try:
            image_url, used_provider = generator.generate_newyorker_style(
                visual_strategy=config['visual_description'],
                caption="",
                aspect_ratio='16:9',
                max_retries=3
            )

            if 'cover' in config['filename']:
                save_path = cover_dir / config['filename']
            else:
                save_path = output_dir / config['filename']

            generator.save_image(image_url, str(save_path))

            file_size = save_path.stat().st_size / 1024
            print(f"    ✅ 已保存 ({file_size:.1f} KB) - {used_provider}\n")

            if i < len(ILLUSTRATIONS):
                time.sleep(1)

        except Exception as e:
            print(f"    ❌ 失败: {e}\n")

    print("🎉 配图生成完成！")

if __name__ == '__main__':
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

    generate_images()
