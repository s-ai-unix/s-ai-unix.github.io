#!/usr/bin/env python3
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

IMAGES_DIR = Path("static/images/covers")
POSTS_DIR = Path("content/posts")

def extract_image_urls():
    urls = set()
    for md_file in POSTS_DIR.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            urls.update(re.findall(r'image:\s*"([^"]+)"', f.read()))
    return sorted(urls)

def get_local_filename(url):
    parsed = urlparse(url)
    match = re.search(r'/photo-([a-zA-Z0-9-]+)', parsed.path)
    photo_id = match.group(1)[:50] if match else os.path.basename(parsed.path).rsplit('.', 1)[0]
    return f"{photo_id}.jpg"

def download_image(url, filename):
    dest_path = IMAGES_DIR / filename

    if dest_path.exists() and dest_path.stat().st_size > 1024:
        print(f"  ✓ 已存在: {filename}")
        return True

    print(f"  ↓ 下载中: {filename}")

    try:
        subprocess.run(
            ['curl', '-sL', '-o', str(dest_path), '--max-time', '30', '--retry', '3', url],
            check=True, capture_output=True, text=True
        )

        if dest_path.stat().st_size < 1024:
            dest_path.unlink()
            return False

        print(f"  ✓ 完成: {filename} ({dest_path.stat().st_size} bytes)")
        return True

    except Exception:
        if dest_path.exists():
            dest_path.unlink()
        return False

def update_markdown_files(url_mapping):
    updated_count = 0
    for md_file in POSTS_DIR.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        for old_url, new_path in url_mapping.items():
            pattern = rf'image:\s*"{re.escape(old_url)}"'
            if re.search(pattern, content):
                content = re.sub(pattern, f'image: "{new_path}"', content)

        if content != original_content:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 更新: {md_file.name}")
            updated_count += 1

    return updated_count

def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("博客图片缓存工具")
    print("=" * 60)

    urls = extract_image_urls()
    print(f"\n找到 {len(urls)} 张图片")

    print("\n[2/3] 下载图片...")
    url_mapping = {}
    success_count = 0

    for url in urls:
        filename = get_local_filename(url)
        if download_image(url, filename):
            url_mapping[url] = f"/images/covers/{filename}"
            success_count += 1

    print(f"\n下载完成: {success_count}/{len(urls)}")

    print("\n[3/3] 更新markdown文件...")
    updated_count = update_markdown_files(url_mapping)
    print(f"更新文件: {updated_count}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n提示：")
    print("  - 图片已保存到 static/images/covers/")
    print("  - markdown文件已更新为本地路径")
    print("  - 运行 'hugo server' 查看效果")
    print("\n如需回滚，使用 git checkout -- content/posts/")

if __name__ == "__main__":
    main()
