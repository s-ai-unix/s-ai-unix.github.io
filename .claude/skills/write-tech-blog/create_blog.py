#!/usr/bin/env python3
"""
技术博客文章生成器
自动创建带有当前日期的博客文章文件
"""

import os
import re
import datetime
import yaml
import argparse
import sys
from pathlib import Path

def generate_current_date():
    """生成当前日期（中国时区）"""
    # 使用北京时间（UTC+8）
    beijing_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    return beijing_time.strftime('%Y-%m-%dT%H:%M:%S+08:00')

def slugify(title):
    """将标题转换为URL友好的slug"""
    # 移除特殊字符，替换为连字符
    title = re.sub(r'[^\w\s-]', '', title.lower())
    title = re.sub(r'[-\s]+', '-', title)
    return title.strip('-')

def create_front_matter(title, categories, tags, cover_image):
    """创建Front Matter"""
    date = generate_current_date()

    front_matter = {
        'title': title,
        'date': date,
        'draft': False,
        'description': get_description_from_content(title),
        'categories': categories,
        'tags': tags,
        'cover': {
            'image': f"images/covers/{cover_image}",
            'alt': f"{title} cover image",
            'caption': f"{title} - Cover Image"
        },
        'math': True
    }

    return yaml.dump(front_matter, default_flow_style=False, allow_unicode=True)

def get_description_from_content(title):
    """根据标题生成简介"""
    descriptions = {
        "机器学习": "深入理解机器学习的核心概念和算法实现",
        "深度学习": "探索深度学习的理论基础和实际应用",
        "神经网络": "从基础到高级的神经网络架构解析",
        "算法": "经典算法与数据结构的深入分析",
        "数学": "数学在技术领域的应用和重要性",
        "统计": "统计学方法在现代技术中的应用",
        "优化": "优化算法及其在各领域的应用"
    }

    for key in descriptions:
        if key in title:
            return descriptions[key]

    return f"关于{title}的技术文章"

def generate_content_structure(title):
    """生成文章内容结构"""
    template = f"""# {title}

## 引言

本文将从历史背景和直观例子开始，循序渐进地介绍{title}的核心概念。我们将通过生动的例子和严谨的数学推导，帮助读者深入理解这一重要主题。

## 第一章：预备知识

在正式进入{title}的讨论之前，我们需要了解一些基础概念。这些知识将帮助我们更好地理解后续内容。

### 1.1 基础概念

[在此处添加基础概念说明]

### 1.2 历史背景

[在此处添加历史背景介绍]

## 第二章：核心概念

本章将深入探讨{title}的核心思想和理论基础。

### 2.1 核心定义

[在此处添加核心定义]

### 2.2 理论基础

[在此处添加理论基础，包含数学公式]

[在此处添加数学公式示例]

$$E = mc^2$$

## 第三章：具体计算

本章将通过具体的计算实例，帮助读者理解{title}的实际应用。

### 3.1 计算方法

[在此处添加计算方法]

### 3.2 实例分析

[在此处添加实例分析]

## 第四章：进阶内容

本章将介绍一些进阶内容，帮助读者拓展知识面。

### 4.1 高级概念

[在此处添加高级概念]

### 4.2 应用拓展

[在此处添加应用拓展]

## 结语

本文系统地介绍了{title}的基本概念、理论方法和实际应用。通过循序渐进的讲解，希望读者能够对这一主题有深入的理解。

## 参考文献

[在此处添加参考文献]
"""
    return template

def create_blog_post(title, categories=None, tags=None, output_dir="content/posts"):
    """创建博客文章"""
    if categories is None:
        categories = ["技术"]
    if tags is None:
        tags = [title.split()[0]]  # 使用第一个词作为标签

    # 生成文件名
    slug = slugify(title)
    date_prefix = datetime.datetime.now().strftime('%Y-%m-%d')
    filename = f"{date_prefix}-{slug}.md"
    filepath = os.path.join(output_dir, filename)

    # 检查目录是否存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成封面图片名
    cover_image = f"{slug}-cover.jpg"

    # 创建Front Matter和内容
    front_matter = create_front_matter(title, categories, tags, cover_image)
    content = front_matter + "\n" + generate_content_structure(title)

    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 成功创建博客文章: {filepath}")
    print(f"📅 文章日期: {generate_current_date()}")
    print(f"🏷️  标签: {', '.join(tags)}")
    print(f"📂 分类: {', '.join(categories)}")

    return filepath

def main():
    parser = argparse.ArgumentParser(description='创建技术博客文章')
    parser.add_argument('title', help='文章标题')
    parser.add_argument('--categories', nargs='+', default=['技术'], help='文章分类')
    parser.add_argument('--tags', nargs='+', help='文章标签')
    parser.add_argument('--output', default='content/posts', help='输出目录')

    args = parser.parse_args()

    # 创建博客文章
    filepath = create_blog_post(args.title, args.categories, args.tags, args.output)

    # 显示下一步操作
    print("\n📋 下一步操作:")
    print("1. 编辑文章内容: edit " + filepath)
    print("2. 添加配图: 下载封面图片到 static/images/covers/")
    print("3. 生成图表: 使用 generate_plots.py")
    print("4. 质量检查: 参考 QUALITY-CHECK.md")

if __name__ == "__main__":
    main()