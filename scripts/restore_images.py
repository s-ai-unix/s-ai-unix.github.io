#!/usr/bin/env python3
"""
图片恢复脚本
从备份恢复原始图片
"""

import shutil
import os
import sys

BACKUP_DIR = "static/images_backup_20260129"
IMAGES_DIR = "static/images"

def restore():
    if not os.path.exists(BACKUP_DIR):
        print(f"❌ 备份目录不存在: {BACKUP_DIR}/")
        print("   请确认是否已经运行过压缩脚本")
        sys.exit(1)
    
    print(f"⚠️  这将删除当前图片并从备份恢复")
    print(f"   备份来源: {BACKUP_DIR}/")
    print(f"   目标位置: {IMAGES_DIR}/")
    
    confirm = input("\n确认恢复? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 已取消")
        sys.exit(0)
    
    # 删除当前图片
    if os.path.exists(IMAGES_DIR):
        print(f"🗑️  删除当前图片...")
        shutil.rmtree(IMAGES_DIR)
    
    # 从备份恢复
    print(f"📦 从备份恢复...")
    shutil.copytree(BACKUP_DIR, IMAGES_DIR)
    
    print(f"✅ 恢复完成!")
    print(f"   图片已恢复到压缩前的状态")

if __name__ == '__main__':
    restore()
