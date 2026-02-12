#!/usr/bin/env python3
"""
创建 Hugo 博客文章的脚本
自动生成正确的 front matter 格式和文件结构
"""

import argparse
import re
import unicodedata
from datetime import datetime
from pathlib import Path


def slugify(text: str) -> str:
    """将标题转换为 URL 友好的 slug"""
    # 保留中文字符，将其他字符转为小写并替换空格为连字符
    text = text.lower().strip()
    # 移除非字母数字字符（保留中文）
    text = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', text)
    # 替换空格和连字符为单个连字符
    text = re.sub(r'[-\s]+', '-', text)
    return text


def generate_slug(title: str) -> str:
    """生成文件名 slug"""
    slug = slugify(title)
    # 限制长度
    if len(slug) > 50:
        slug = slug[:50].rsplit('-', 1)[0]
    return slug


def generate_front_matter(title: str, categories: list, tags: list) -> str:
    """生成 Hugo front matter"""
    # 获取当前北京时间
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%dT%H:%M:%S+08:00')
    date_prefix = now.strftime('%Y-%m-%d')
    
    # 处理分类和标签
    categories_str = ', '.join(f'"{c}"' for c in categories)
    tags_str = ', '.join(f'"{t}"' for t in tags)
    
    # 生成 slug 用于封面图文件名
    slug = generate_slug(title)
    cover_filename = f"{slug}-cover.jpg"
    
    # 生成简介（标题 + 一句话描述）
    description = f"深入浅出地介绍{title.split('：')[0] if '：' in title else title}的核心概念与应用"
    
    front_matter = f'''---
title: "{title}"
date: {date_str}
draft: false
description: "{description}"
categories: [{categories_str}]
tags: [{tags_str}]
cover:
    image: "images/covers/{cover_filename}"
    alt: "{title}"
    caption: "{title} - 封面图"
math: true
---

## 引言

''' + '''在这里开始你的文章引言...

'''
    return front_matter, date_prefix, slug


def create_blog_post(title: str, categories: list = None, tags: list = None, 
                     output_dir: str = "content/posts") -> str:
    """
    创建博客文章文件
    
    Args:
        title: 文章标题
        categories: 文章分类列表，默认为 ["技术"]
        tags: 文章标签列表，默认从标题提取
        output_dir: 输出目录，默认为 content/posts
    
    Returns:
        创建的文件路径
    """
    # 默认值
    if categories is None:
        categories = ["技术"]
    if tags is None:
        # 从标题提取一些默认标签
        tags = ["综述"]
    
    # 生成 front matter
    front_matter, date_prefix, slug = generate_front_matter(title, categories, tags)
    
    # 生成文件名
    filename = f"{date_prefix}-{slug}.md"
    filepath = Path(output_dir) / filename
    
    # 确保目录存在
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否已存在
    if filepath.exists():
        raise FileExistsError(f"文件已存在: {filepath}")
    
    # 写入文件
    filepath.write_text(front_matter, encoding='utf-8')
    
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="创建 Hugo 博客文章",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 create_blog.py "机器学习入门"
  python3 create_blog.py "深度学习基础" --categories 机器学习 人工智能 --tags 深度学习 神经网络
  python3 create_blog.py "算法分析" --categories 算法 --tags 数据结构
        """
    )
    
    parser.add_argument(
        "title",
        help="文章标题"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["技术"],
        help="文章分类，默认为 ['技术']"
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["综述"],
        help="文章标签，默认为 ['综述']"
    )
    
    parser.add_argument(
        "--output",
        default="content/posts",
        help="输出目录，默认为 content/posts"
    )
    
    args = parser.parse_args()
    
    try:
        filepath = create_blog_post(
            title=args.title,
            categories=args.categories,
            tags=args.tags,
            output_dir=args.output
        )
        
        print(f"✅ 文章创建成功!")
        print(f"   文件: {filepath}")
        print(f"   标题: {args.title}")
        print(f"   分类: {', '.join(args.categories)}")
        print(f"   标签: {', '.join(args.tags)}")
        print()
        print("接下来:")
        print("1. 下载封面图到 static/images/covers/ 目录")
        print("2. 编辑文章，完善引言和正文内容")
        print("3. 运行 hugo server 预览效果")
        
    except FileExistsError as e:
        print(f"❌ 错误: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
