#!/usr/bin/env python3
"""
图片压缩脚本
压缩 static/images/ 目录下的所有图片
- PNG: 使用 pngquant 或 optipng
- JPG: 使用 Pillow 降低质量到 85%

安全特性：
1. 原地压缩，不改变文件名，网页引用不受影响
2. 自动备份原图到 static/images_backup/
3. 可以一键恢复
"""

import os
import shutil
from PIL import Image
from pathlib import Path
import subprocess
from datetime import datetime

def backup_images():
    """备份原图到固定目录（该目录会被 .gitignore 忽略）"""
    backup_dir = "static/images_backup_20260129"
    
    # 如果已存在备份，先删除旧备份
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
        print(f"🗑️  已删除旧备份: {backup_dir}/")
    
    if os.path.exists('static/images'):
        shutil.copytree('static/images', backup_dir)
        print(f"📦 原图已备份到: {backup_dir}/")
        print(f"   ⚠️  该目录已被 .gitignore 忽略，不会提交到 git")
        return backup_dir
    return None

def compress_jpg(filepath, quality=85, max_width=1920):
    """压缩 JPG 图片"""
    try:
        img = Image.open(filepath)
        
        # 如果图片尺寸太大，先缩小（保持宽高比）
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # 保存压缩后的图片（原地覆盖，文件名不变）
        original_size = os.path.getsize(filepath)
        img.save(filepath, 'JPEG', quality=quality, optimize=True)
        new_size = os.path.getsize(filepath)
        
        if original_size > 0:
            saved = (original_size - new_size) / original_size * 100
            print(f"✅ {filepath}: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB (节省 {saved:.1f}%)")
            return saved, original_size - new_size
    except Exception as e:
        print(f"❌ {filepath}: {e}")
    return 0, 0

def compress_png(filepath, max_width=1920):
    """使用 pngquant 压缩 PNG"""
    try:
        # 先用 Pillow 缩小尺寸（如果需要）
        img = Image.open(filepath)
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            img.save(filepath, 'PNG', optimize=True)
        
        original_size = os.path.getsize(filepath)
        
        # 使用 pngquant 压缩
        result = subprocess.run(
            ['pngquant', '--quality=70-85', '--force', '--output', filepath, filepath],
            capture_output=True,
            text=True
        )
        
        new_size = os.path.getsize(filepath)
        if original_size > 0 and new_size < original_size:
            saved = (original_size - new_size) / original_size * 100
            print(f"✅ {filepath}: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB (节省 {saved:.1f}%)")
            return saved, original_size - new_size
        else:
            print(f"⏭️  {filepath}: 已是最优，跳过")
            return 0, 0
            
    except FileNotFoundError:
        print(f"⚠️  pngquant 未安装，使用 Pillow 压缩 PNG")
        try:
            img = Image.open(filepath)
            original_size = os.path.getsize(filepath)
            img.save(filepath, 'PNG', optimize=True)
            new_size = os.path.getsize(filepath)
            if original_size > 0:
                saved = (original_size - new_size) / original_size * 100
                print(f"✅ {filepath}: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB (节省 {saved:.1f}%)")
                return saved, original_size - new_size
        except Exception as e:
            print(f"❌ {filepath}: {e}")
    except Exception as e:
        print(f"❌ {filepath}: {e}")
    return 0, 0

def main():
    # 先备份
    backup_dir = backup_images()
    if not backup_dir:
        print("❌ 备份失败，取消压缩")
        return
    
    images_dir = Path('static/images')
    total_saved_percent = 0
    total_saved_bytes = 0
    count = 0
    
    print("\n开始压缩图片...（原地压缩，文件名不变）\n")
    
    for filepath in images_dir.rglob('*'):
        if not filepath.is_file():
            continue
            
        suffix = filepath.suffix.lower()
        if suffix in ['.jpg', '.jpeg']:
            saved_pct, saved_bytes = compress_jpg(str(filepath))
            total_saved_percent += saved_pct
            total_saved_bytes += saved_bytes
            count += 1
        elif suffix == '.png':
            saved_pct, saved_bytes = compress_png(str(filepath))
            total_saved_percent += saved_pct
            total_saved_bytes += saved_bytes
            count += 1
    
    print(f"\n{'='*50}")
    print(f"✅ 压缩完成!")
    print(f"   处理图片: {count} 张")
    if count > 0:
        print(f"   平均节省: {total_saved_percent/count:.1f}%")
    print(f"   总节省: {total_saved_bytes/1024/1024:.1f} MB")
    print(f"   备份位置: {backup_dir}/")
    print(f"\n💡 如果压缩后图片显示有问题，可以一键恢复:")
    print(f"   python3 scripts/restore_images.py")
    print(f"\n   或手动恢复:")
    print(f"   rm -rf static/images && cp -r {backup_dir} static/images")

if __name__ == '__main__':
    main()
