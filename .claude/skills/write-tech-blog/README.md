# Write-Tech-Blog 技能更新说明

## 更新内容

本次更新解决了文章日期问题，确保每次创建的文章都使用当前真实的日期，避免出现过去或未来的日期。

### 主要改进

1. **自动日期生成**
   - 新增 `create_blog.py` 脚本自动生成当前日期
   - 日期格式：YYYY-MM-DDTHH:mm:ss+08:00（北京时间）
   - 文件名也使用当前日期：YYYY-MM-DD-slug.md

2. **改进的脚本功能**
   - 自动生成标准化的 Front Matter
   - 自动创建文章结构模板
   - 支持自定义分类和标签
   - 生成合适的文章简介

3. **更新的文档**
   - 更新了 `SKILL.md` 中的日期说明
   - 更新了 `QUALITY-CHECK.md` 中的日期检查要求

## 使用方法

### 基本用法

```bash
# 在技能目录中执行
python3 create_blog.py "文章标题"

# 示例
python3 create_blog.py "机器学习入门"
```

### 高级用法

```bash
# 指定分类和标签
python3 create_blog.py "深度学习基础" --categories 机器学习 人工智能 --tags 深度学习 神经网络

# 指定输出目录
python3 create_blog.py "算法分析" --categories 算法 --tags 数据结构 --output content/posts

# 查看帮助
python3 create_blog.py --help
```

## 脚本功能

### 自动生成的信息

- ✅ **当前日期**：自动使用系统当前时间（北京时间）
- ✅ **文件名**：YYYY-MM-DD-标题转换后的slug
- ✅ **Front Matter**：标准的YAML格式
- ✅ **文章简介**：基于标题自动生成
- ✅ **内容结构**：6章节的标准模板

### Front Matter 示例

```yaml
---
title: "文章标题"
date: 2026-01-24T18:49:25+08:00  # 当前真实日期
draft: false
description: "文章简介，1-2句话概括"
categories: ["技术"]
tags: ["标签1", "标签2"]
cover:
    image: "images/covers/标题转换后-cover.jpg"
    alt: "图片描述"
    caption: "图片标题"
math: true
---
```

## 质量检查

### 日期检查要点

在质量检查时，特别注意以下几点：

- ✅ 日期必须是当前真实日期
- ✅ 时区设置为北京时间（+08:00）
- ✅ 文件名与日期匹配
- ✅ 避免重复使用过去的日期

### 使用命令检查日期

```bash
# 检查文章日期
grep "date:" content/posts/文章文件.md

# 检查文件名日期
ls -la content/posts/文章文件.md
```

## 示例输出

运行脚本后，你会看到类似这样的输出：

```
✅ 成功创建博客文章: content/posts/2026-01-24-机器学习入门.md
📅 文章日期: 2026-01-24T18:49:25+08:00
🏷️  标签: 机器学习
📂 分类: 技术

📋 下一步操作:
1. 编辑文章内容: edit content/posts/2026-01-24-机器学习入门.md
2. 添加配图: 下载封面图片到 static/images/covers/
3. 生成图表: 使用 generate_plots.py
4. 质量检查: 参考 QUALITY-CHECK.md
```

## 注意事项

1. **时区设置**：脚本使用北京时间（UTC+8）
2. **文件名生成**：会自动将标题转换为URL友好的格式
3. **日期格式**：符合 Hugo 静态站点生成器的要求
4. **内容模板**：提供6章节的标准结构，可根据需要修改

## 故障排除

如果遇到日期问题：

1. 检查系统时间是否正确
2. 确认脚本是否具有执行权限
3. 检查输出目录是否存在
4. 查看错误日志了解具体问题

通过这次更新，write-tech-blog 技能现在能够确保每次创建的文章都使用正确的当前日期，避免出现日期不一致的问题。