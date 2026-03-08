#!/Users/sun1/miniconda3/envs/py3.13env/bin/python3
"""
Generate cover image for Nanoclaw memory system article using Jimeng
"""
import os
import sys
from pathlib import Path

# Load shared library
shared_lib_path = str(Path.home() / '.claude' / 'skills' / 'shared-lib')
sys.path.insert(0, shared_lib_path)

# Load API key from skill config
config_path = Path.home() / '.config' / 'claude-skills' / 'write-tech-blog' / '.env'
if config_path.exists():
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Set environment variables
os.environ['JIMENG_API_URL'] = 'https://newapi.aisonnet.org/v1'

from image_api import ImageGenerator

def main():
    # Create output directory
    output_dir = Path('~/Gitlab/Personal/Hugo_Blog/blog/static/images/covers').expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = ImageGenerator(provider='jimeng', jimeng_model='jimeng-4.5')

    # Visual description for cover
    visual_strategy = (
        "Photorealistic 3D render, cinematic sci-fi: a massive translucent glowing brain "
        "suspended in deep space, its surface formed by interlocking hexagonal panels of electric blue glass. "
        "Dense neon-lit neural pathways weave through it like fiber optic cables, pulsing with vivid "
        "cyan, magenta, and gold light. "
        "Surrounding the brain: concentric rings of holographic data—glowing circular discs "
        "layered like a planetary system, each ring a different color (cobalt, violet, amber). "
        "Streams of luminous particles spiral inward like a data vortex, forming a visual metaphor "
        "for memory being absorbed and stored. "
        "Background: an infinite dark cosmos with deep indigo nebula clouds, star clusters, and "
        "faint aurora-like sweeps of teal and purple light. "
        "Dramatic volumetric god rays, lens flare, ultra-sharp 8K detail, Unreal Engine 5 quality. "
        "Zero text, zero letters, zero numbers. Pure visual spectacle."
    )

    caption = ""

    print("🎨 Generating cover image for Nanoclaw memory system article...")

    try:
        image_url, used_provider = generator.generate_newyorker_style(
            visual_strategy=visual_strategy,
            caption=caption,
            aspect_ratio='16:9',
            max_retries=2
        )

        # Save image
        output_path = output_dir / 'nanoclaw-memory-system-cover.png'
        generator.save_image(image_url, str(output_path))

        print(f"✅ Cover image saved: {output_path}")
        print(f"   Provider: {used_provider}")

        # Show file size
        file_size = output_path.stat().st_size / 1024
        print(f"   File size: {file_size:.1f} KB")

    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
