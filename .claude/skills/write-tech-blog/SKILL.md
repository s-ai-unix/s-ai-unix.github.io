---
name: write-tech-blog
description: 创建高质量技术博客文章。面向懂微积分和线性代数的读者，使用"娓娓道来"的叙事风格，支持纽约客风格配图、Plotly 数理图形和苹果风格 Mermaid 图表，严格执行 LaTeX 数学公式规范。当用户要求"写一篇关于XXX的技术文章"、"给懂微积分和线性代数的人讲XXX"、"写一篇文章娓娓道来"时触发。
---

# 技术博客文章写作

## 快速开始

### 1. 文章规划

- **理解主题**：明确核心概念和目标读者（懂微积分和线性代数）
- **制定大纲**：
  - 引言：从历史背景或直观例子开始
  - 核心概念：循序渐进地介绍，包含数学推导
  - 具体应用：给出实际案例
  - 结语：总结核心要点
- **标签使用规范**：
  - 尽量使用已有的核心标签，避免创建新标签
  - 核心标签列表（按使用频率排序）：
    - 功能安全、数学史、ISO 26262、微分几何、系统管理
    - Python、广义相对论、机器学习、算法、深度学习
    - 综述、黎曼几何、ASIL、Perl、神经网络
    - 数据结构、数据分析、自动驾驶、几何、Shell
    - 文本处理、命令行、C语言、HARA、偏微分方程
  - 每个文章标签数量控制在 3-5 个
  - 标签应与文章内容高度相关

### 2. 文章撰写

**风格要求**：
- 娓娓道来：像讲故事一样，从简单到复杂
- 循序渐进：每一步都自然地引出下一步
- 物理直觉：用具体的例子和类比说明抽象概念
- 数学严谨：公式推导清晰，步骤完整

**必须执行的数学公式规范**（见 [LATEX-MATH.md](references/LATEX-MATH.md)）：
- ✅ 所有数学变量必须用 `$...$` 或 `$$...$$` 包裹
- ✅ 希腊字母使用 LaTeX 命令（`\alpha` 而非 `α`）
- ✅ 上标下标格式：`^{...}` 和 `_{...}`
- ✅ 特殊符号必须在数学模式中

**章节结构**：
1. 引言：用生动的例子引入主题
2. 第一章：预备知识（如需要）
3. 第二章：核心概念的引入
4. 第三章：具体计算或应用
5. 第四章：进阶内容或扩展
6. 结语：总结和展望

### 3. 添加配图

#### Unsplash 封面图

从 Unsplash 下载纽约客风格的抽象/几���图片：

```bash
curl -sL "[Unsplash URL]" -o static/images/covers/[filename].jpg --max-time 30 --retry 3
```

验证文件大小（确保 > 10KB）：
```bash
ls -lh static/images/covers/[filename].jpg
```

#### 图表生成策略

**数学/物理图形（使用 Plotly）**：
对于涉及函数图像、几何演化、数据可视化的内容，使用 Plotly 生成专业图形：

```python
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# 示例：绘制 Ricci Flow 演化
def plot_ricci_flow_evolution():
    t = np.linspace(0, 2, 100)
    radius = np.exp(-2 * t)  # 球面 Ricci Flow 解

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=radius,
        mode='lines',
        name='半径演化',
        line=dict(color='#007AFF', width=3)
    ))

    fig.update_layout(
        title='球面 Ricci Flow 半径演化',
        xaxis_title='时间 t',
        yaxis_title='半径 R(t)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )

    fig.write_html('static/images/plots/ricci-flow-evolution.html')
    return fig
```

**Plotly 图形样式要求**：
- 使用 `plotly_white` 模板
- 主色调：苹果蓝色 `#007AFF`
- 辅助色：绿色 `#34C759`、橙色 `#FF9500`
- 字体：Arial, sans-serif，字号 14
- 添加适当的标签和标题
- 导出为 PNG 文件放在 `static/images/plots/` 目录（`fig.write_image('path.png', scale=2)`）

**Plotly 作图核心原则**：

1. **文字与图形尺寸匹配**
   - 节点/圆圈大小必须与文字长度匹配
   - ❌ 错误：小圆圈 + 长文字（如 "特征A<阈值" 放在 size=40 的圆圈里会溢出）
   - ✅ 正确：大圆圈 + 短文字 或 小圆圈 + 极简文字
   - 建议：节点大小 50-60px 配合 2-4 个字符的中文，或 8-12 个英文字符

2. **文字简洁性**
   - 图形中的文字必须精简，避免完整句子
   - ❌ 错误："特征A小于阈值"
   - ✅ 正确："A<阈值" 或 "左分支"
   - 原则：用符号和缩写代替完整词汇

3. **避免视觉重叠**
   - 节点间距 ≥ 节点直径的 1.5 倍
   - 文字与图形边缘留出 20% 边距
   - 多节点时考虑分层布局（树状、网格、力导向）

4. **分离 marker 和 text**
   - 复杂标签建议分开绘制 marker 和 text
   ```python
   # 先绘制圆圈
   fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', 
                            marker=dict(size=55, color=color)))
   # 再绘制文字
   fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                            text=[label], textposition='middle center'))
   ```

5. **高对比度配色**
   - 深色背景（节点）配白色文字
   - 浅色背景配深色文字
   - 避免黄色文字在白色背景上

6. **导出分辨率**
   - 使用 `scale=2` 导出高清图（适合 Retina 屏幕）
   - 推荐尺寸：800x600 或 900x450（scale=2 后为 1600x1200 或 1800x900）

#### 图片压缩要求（必须执行）

**为什么需要压缩**：
- 原始 Plotly 图片可能很大（单张 100-500KB）
- 博客仓库体积会迅速膨胀（可能超过 100MB）
- 每次 `git push` 变慢，GitHub Pages 部署变慢

**压缩目标**：
- PNG 图表：压缩后 10-50KB（节省 50-80%）
- JPG 封面：压缩后 100-300KB（节省 30-50%）
- 单张图片不超过 500KB

**首次安装工具**（只需执行一次）：

```bash
# macOS
brew install pngquant

# Ubuntu/Debian
sudo apt-get install pngquant
```

**生成图片时直接压缩**（推荐做法）：

在 Python 脚本中生成 Plotly 图片后，立即压缩：

```python
import subprocess
import os

def save_and_compress(fig, filepath):
    """保存并压缩图片"""
    # 先保存
    fig.write_image(filepath, scale=2)
    
    # 立即压缩
    if filepath.endswith('.png'):
        subprocess.run([
            'pngquant', '--quality=70-85', '--force', 
            '--output', filepath, filepath
        ], check=False)
    
    print(f"✅ 已保存并压缩: {filepath}")

# 使用示例
save_and_compress(fig, 'static/images/plots/my-plot.png')
```

**批量压缩新图片**（文章完成后）：

```bash
# 只压缩 plots 目录下修改过的图片（今天生成的）
find static/images/plots -name "*.png" -mtime -1 -exec pngquant --quality=70-85 --force --output {} {} \;

# 或者使用项目脚本（压缩全部，跳过已压缩的）
python3 scripts/compress_images.py
```

**压缩策略**：
- **PNG**: 使用 `pngquant` 压缩到 70-85% 质量
- **JPG**: 使用 Pillow 压缩到 85% 质量，最大宽度 1920px

**验证压缩效果**：

```bash
# 查看图片大小
ls -lh static/images/plots/*.png

# 好的压缩效果示例：
# 压缩前：300KB → 压缩后：60KB（节省 80%）
# 压缩前：150KB → 压缩后：30KB（节省 80%）
```

**正误对比示例**（决策树节点）：

```python
# ❌ 错误：文字太长，圆圈太小，文字溢出
fig.add_trace(go.Scatter(
    x=[x], y=[y],
    mode='markers+text',
    marker=dict(size=40, color=color),  # 太小！
    text=['特征A<阈值'],  # 太长！
    textposition='middle center',
    textfont=dict(size=10, color='white')
))

# ✅ 正确：简短文字 + 大圆圈，或分离绘制
# 方案1：简化文字
fig.add_trace(go.Scatter(
    x=[x], y=[y],
    mode='markers+text',
    marker=dict(size=55, color=color),  # 更大
    text=['A<阈值'],  # 简化
    textposition='middle center',
    textfont=dict(size=11, color='white')
))

# 方案2：分离 marker 和 text（更灵活）
fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                         marker=dict(size=55, color=color)))
fig.add_trace(go.Scatter(x=[x], y=[y], mode='text',
                         text=['A<阈值'],
                         textposition='middle center',
                         textfont=dict(size=11, color='white')))
```

**常见图形类型的尺寸参考**：

| 图形类型 | 节点大小 | 字体大小 | 文字长度建议 | 示例 |
|---------|---------|---------|------------|------|
| 决策树节点 | 50-60 | 11 | 2-4字中文 | "根节点", "A<阈值" |
| 流程图节点 | 40-50 | 10-12 | 2-5字中文 | "开始", "处理数据" |
| 网络图节点 | 30-40 | 9-10 | 1-3字中文 | "A", "B1" |
| 散点图标注 | - | 10 | 极短标签 | 数字、符号 |
| 热力图标签 | - | 12-14 | 数字或缩写 | "准确率", "Loss" |

**流程图（使用 Mermaid）**：
对于概念流程、结构关系等非数理图形，使用 Mermaid 图表：

**必须使用苹果风格配色**（见 [MERMAPLE-STYLE.md](references/MERMAPLE-STYLE.md)）：
- 所有节点文字为白色：`color:#ffffff`
- 核心节点：蓝色 `#007AFF`，边框 3px
- 重要节点：绿色 `#34C759`，边框 2px
- 警告节点：橙色 `#FF9500`，边框 2px
- 使用 `subgraph` 分组相关元素

### 4. 质量检查（必须执行）

**文章完成后必须执行质量检查**（见 [QUALITY-CHECK.md](references/QUALITY-CHECK.md)）：

#### 必做检查清单
1. ✅ **编码检查**：检查乱码和替换字符（``）
2. ✅ **数学公式检查**：验证 LaTeX 格式规范
3. ✅ **图表检查**：Plotly 数理图形专业，Mermaid 图表使用苹果风格
4. ✅ **图片检查**：封面图大小 > 10KB
5. ✅ **格式检查**：Front Matter 格式正确

#### 快速编码检查脚本
```python
python3 << 'EOF'
with open('content/posts/[文章文件].md', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查替换字符
if '\ufffd' in content:
    print("❌ 发现替换字符")
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if '\ufffd' in line:
            print(f"  行 {i}: {line.strip()[:100]}")
else:
    print("✅ 编码检查通过")
EOF
```

### 5. 创建文章文件

**文件位置**：`content/posts/YYYY-MM-DD-[slug].md`

#### 自动生成脚本

使用提供的脚本自动创建文章，日期会自动设置为当前时间：

```bash
python3 create_blog.py "文章标题" --categories 技术 --tags 机器学习
```

**参数说明**：
- `文章标题`: 必填，文章的标题
- `--categories`: 文章分类，默认为 ["技术"]
- `--tags`: 文章标签，默认从标题提取
- `--output`: 输出目录，默认为 content/posts

**使用示例**：

```bash
# 基本用法 - 自动使用当前日期
python3 create_blog.py "机器学习入门"

# 指定分类和标签
python3 create_blog.py "深度学习基础" --categories 机器学习 人工智能 --tags 深度学习 神经网络

# 指定输出目录
python3 create_blog.py "算法分析" --categories 算法 --tags 数据结构 --output content/posts

# 查看帮助
python3 create_blog.py --help
```

**脚本会自动处理**：
- ✅ 生成当前日期（北京时间）
- ✅ 创建文件名（YYYY-MM-DD-slug.md）
- ✅ 生成合适的简介
- ✅ 创建标准的内容结构

**Front Matter 格式**（脚本自动生成）：
```yaml
---
title: "文章标题"
date: YYYY-MM-DDTHH:mm:ss+08:00  # 自动使用当前日期时间
draft: false
description: "文章简介，1-2句话概括"
categories: ["分类1", "分类2"]
tags: ["标签1", "标签2", "标签3"]
cover:
    image: "images/covers/[slug]-cover.jpg"
    alt: "图片描述"
    caption: "图片标题"
math: true
---
```

**日期说明**：
- 日期格式为 ISO 8601 标准格式
- 时区设置为北京时间（UTC+8）
- 文件名中的日期也使用当前日期（YYYY-MM-DD-）
- 每次运行脚本都会生成新的当前日期，避免重复或过期日期

## 详细规范

- **LaTeX 数学公式完整规范**：见 [LATEX-MATH.md](references/LATEX-MATH.md)
- **Mermaid 苹果风格图表**：见 [MERMAPLE-STYLE.md](references/MERMAPLE-STYLE.md)
- **质量检查和验证清单**：见 [QUALITY-CHECK.md](references/QUALITY-CHECK.md)

## 常见问题

### 数学公式不显示或渲染错误

**症状**：公式显示为原始代码（如 `$x$`）

**解决**：确保使用正确的分隔符
- 行内公式：`$公式$`
- 独立公式：`$$公式$$`

**症状**：浏览器控制台报错 "Missing open brace for superscript"

**解决**：检查上标格式，使用 `^{...}` 而非 `^(...)`

**症状**：希腊字母或特殊符号显示异常

**解决**：使用 LaTeX 命令（`\alpha`, `\sigma`）而非 Unicode 字符

详见 [LATEX-MATH.md](references/LATEX-MATH.md) 的"常见错误"章节。

### 图片下载失败

**症状**：文件很小（<1KB）

**解决**：
1. 检查 Unsplash URL 是否正确
2. 尝试其他图片或使用现有图片
3. 确保下载到本地（不要使用外链）

## 成功标准

文章完成后并通过质量检查后，应满足：
1. ✅ 文章内容完整，逻辑清晰
2. ✅ **无乱码**：无替换字符，无编码错误
3. ✅ 数学公式正确渲染
4. ✅ 配图显示正常（>10KB）
5. ✅ **Plotly 图形专业美观**：数学函数图像清晰，配色协调，交互性好
6. ✅ Mermaid 图表使用苹果风格，文字清晰可见
7. ✅ 所有数学符号都使用正确的 LaTeX 格式

**注意**：只有在通过 [QUALITY-CHECK.md](references/QUALITY-CHECK.md) 中的所有检查后，文章才能被认为是"完成"的。
