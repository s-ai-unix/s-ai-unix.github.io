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
- 导出为 HTML 文件放在 `static/images/plots/` 目录

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

**Front Matter 格式**：
```yaml
---
title: "文章标题"
date: YYYY-MM-DDTHH:mm:ss+08:00
draft: false
description: "文章简介，1-2句话概括"
categories: ["分类1", "分类2"]
tags: ["标签1", "标签2", "标签3"]
cover:
    image: "images/covers/[图片文件名].jpg"
    alt: "图片描述"
    caption: "图片标题"
math: true
---
```

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
