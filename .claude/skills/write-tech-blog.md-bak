# 技术博客文章写作与发布

## 技能描述

本技能用于在博客项目中创建高质量的技术文章并自动发布。

## 触发条件

当用户请求满足以下条件时触发：
1. 要求写一篇技术文章/博客文章
2. 文章面向懂微积分和线性代数的读者
3. 要求"娓娓道来"的叙事风格
4. 需要纽约客风格的配图
5. 写完后需要发布（执行 deploy.sh）

## 典型触发短语示例

- "写一篇关于XXX的技术文章"
- "给一个懂点微积分和线性代数的人讲XXX"
- "写一篇文章，要求娓娓道来"
- "配个纽约客风格的配图"
- "写完后执行 deploy.sh 发布"

## 执行流程

### 1. 文章规划

- **理解主题**：明确文章要讲解的核心概念
- **评估难度**：确保内容适合懂微积分和线性代数的读者
- **制定大纲**：
  - 引言：从直观例子或历史背景开始
  - 核心概念：循序渐进地介绍
  - 具体应用：给出实际案例
  - 总结：回顾核心要点

### 2. 文章撰写

**风格要求**：
- 娓娓道来：像讲故事一样，从简单到复杂
- 循序渐进：每一步都自然地引出下一步
- 物理直觉：用具体的例子和类比说明抽象概念
- 数学严谨：公式推导清晰，步骤完整

**内容要求**：
- 从零开始：假设读者有微积分和线性代数基础，但不熟悉该主题
- 历史背景：介绍重要概念的历史和发现者
- 核心定义：给出精确的数学定义
- 几何直观：用图形和例子解释
- 具体计算：给出完整的计算示例
- 实际应用：说明该概念在哪些领域有应用
- **深度标准**：重要概念必须充分展开，包括：
  - 完整的数学推导（不是"显然可得"）
  - 几何直观的说明（配图或文字描述）
  - 具体的数值例子
  - 必要时提供代数证明或几何证明

**章节结构建议**：
1. 引言：用生动的例子引入主题
2. 第一章：预备知识（如需要）
3. 第二章：核心概念的引入
4. 第三章：具体计算或应用
4. 第四章：进阶内容或扩展
5. 结语：总结和展望

### 3. 配图与图表

#### 3.1 数学图形（Python Plotly）

> **核心要求（必须遵守）**：
> - ✅ **所有数学函数图形（曲线、曲面、向量场等）使用 Python Plotly 绘制**
> - ✅ **图形风格：简洁、专业、配色协调**
> - ✅ **图片格式：导出为 PNG，保存到 `static/images/math/` 目录**
> - ✅ **图片插入：在文章适当位置插入，添加描述性 alt 文本**

**适用场景**：
- 数学函数图像（一元/多元函数）
- 几何图形（曲线、曲面）
- 向量场可视化
- 数据可视化（统计图表）
- 物理现象模拟图
- 其他需要展示数学关系或数据分布的图形

**Plotly 图形设计原则**：

1. **配色方案**：使用优雅、协调的配色
   - 主色调：`#1f77b4`（蓝色）、`#ff7f0e`（橙色）、`#2ca02c`（绿色）
   - 背景色：白色或浅灰 `#f8f9fa`
   - 网格线：浅灰 `#e0e0e0`

2. **图形样式**：
   - 线条宽度：2-3px
   - 字体：无衬线字体（Arial, Helvetica）
   - 坐标轴标签：清晰、易读
   - 图例：位置合理，不遮挡主要内容

**常用图形类型代码模板**：

**1. 一元函数曲线图**：
```python
import plotly.graph_objects as go
import numpy as np

# 生成数据
x = np.linspace(-10, 10, 500)
y = np.sin(x) / x  # 示例：sinc 函数

# 创建图形
fig = go.Figure()

# 添加曲线
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    line=dict(color='#1f77b4', width=3),
    name='sinc(x)'
))

# 设置布局
fig.update_layout(
    title='Sinc 函数图像',
    xaxis_title='x',
    yaxis_title='sinc(x)',
    template='plotly_white',
    width=800,
    height=500,
    font=dict(size=14),
    plot_bgcolor='white',
    xaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    yaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333')
)

# 保存图片
fig.write_image('static/images/math/sinc-function.png', scale=2)
```

**2. 3D 曲面图**：
```python
import plotly.graph_objects as go
import numpy as np

# 生成网格数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))  # 示例：二维 sinc 函数

# 创建曲面图
fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Viridis',
    colorbar=dict(title='z 值')
)])

# 设置布局
fig.update_layout(
    title='二维 Sinc 函数曲面',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
    ),
    width=800,
    height=600,
    font=dict(size=12)
)

# 保存图片
fig.write_image('static/images/math/2d-sinc-surface.png', scale=2)
```

**3. 向量场图**：
```python
import plotly.graph_objects as go
import numpy as np

# 生成网格
x, y = np.meshgrid(np.linspace(-3, 3, 20),
                   np.linspace(-3, 3, 20))

# 定义向量场 (示例：旋涡场)
u = -y
v = x

# 创建向量场图
fig = go.Figure(data=go.Scatter(
    x=x.flatten(),
    y=y.flatten(),
    mode='markers',
    marker=dict(size=5, color='#1f77b4')
))

# 添加向量箭头 (使用 quiver)
fig.add_trace(go.Scatter(
    x=x.flatten(),
    y=y.flatten(),
    mode='lines',
    line=dict(color='#ff7f0e', width=1),
    hoverinfo='none'
))

# 设置布局
fig.update_layout(
    title='旋涡向量场',
    xaxis_title='x',
    yaxis_title='y',
    width=600,
    height=600,
    plot_bgcolor='white',
    xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor='#e0e0e0'),
    yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor='#e0e0e0')
)

# 保存图片
fig.write_image('static/images/math/vector-field.png', scale=2)
```

**4. 参数方程曲线**：
```python
import plotly.graph_objects as go
import numpy as np

# 参数范围
t = np.linspace(0, 2*np.pi, 500)

# 参数方程 (示例：蝴蝶曲线)
x = np.sin(t) * (np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5)
y = np.cos(t) * (np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    line=dict(color='#ff7f0e', width=2),
    fill='toself',
    fillcolor='rgba(255, 127, 14, 0.2)',
    name='蝴蝶曲线'
))

fig.update_layout(
    title='蝴蝶曲线',
    xaxis_title='x',
    yaxis_title='y',
    width=600,
    height=600,
    plot_bgcolor='white',
    xaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333'),
    yaxis=dict(gridcolor='#e0e0e0', zerolinecolor='#333')
)

fig.write_image('static/images/math/butterfly-curve.png', scale=2)
```

**5. 等高线图**：
```python
import plotly.graph_objects as go
import numpy as np

# 生成数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2  # 马鞍面

fig = go.Figure(data=go.Contour(
    x=x,
    y=y,
    z=Z,
    colorscale='RdBu_r',
    contours=dict(
        start=-4,
        end=4,
        size=0.5
    ),
    colorbar=dict(title='z 值')
))

fig.update_layout(
    title='马鞍面等高线图',
    xaxis_title='x',
    yaxis_title='y',
    width=600,
    height=500
)

fig.write_image('static/images/math/contour-plot.png', scale=2)
```

**图片保存和插入规范**：

1. **文件命名**：使用描述性英文名，如 `sinc-function.png`、`gradient-descent-3d.png`
2. **保存位置**：`static/images/math/[filename].png`
3. **插入 Markdown**：
   ```markdown
   ![Sinc 函数图像](/images/math/sinc-function.png)

   *图 1：Sinc 函数 sinc(x) = sin(x)/x 的图像*
   ```

**使用流程**：
1. 根据数学概念选择合适的图形类型
2. 使用上述模板编写 Python 代码
3. 运行代码生成图片（需要安装 `plotly` 和 `kaleido`：`pip install plotly kaleido`）
4. 检查图片质量（scale=2 保证高清）
5. 在文章合适位置插入图片
6. 添加描述性说明

**注意事项**：
- 图形应辅助理解，不应过于复杂
- 坐标轴标签使用清晰的中文或数学符号
- 图例说明要完整
- 确保图片在高分辨率下清晰可读

#### 3.2 Mermaid图表

> **核心要求（必须遵守）**：
> - ✅ **所有 mermaid 图必须使用苹果风格配色**
> - ✅ **所有节点和连线文字必须为白色（`color:#ffffff`）以确保清晰可见**
> - ✅ **边框宽度：核心节点 3px，重要节点 2px，次要节点 1px**
> - ✅ **使用 subgraph 对相关元素进行分组**
> - ✅ **为复杂图表添加图例说明**

**适用场景**：
- 流程图（FMEA、FTA、STPA、HARA、TARA等方法步骤）
- 系统架构图（标准体系、系统分解）
- 对比图（方法对比、标准对比）
- 风险分析图（ASIL等级、风险评估）

**苹果风格配色方案**：
```yaml
蓝色系:
  主色: "#007AFF"    # 苹果标准蓝 - 主要步骤、核心内容
  辅色: "#5AC8FA"    # 天蓝色 - 次要元素、支撑内容

绿色系:
  主色: "#34C759"    # 苹果绿 - 成功、完成、结果、硬件、软件
  次色: "#30D158"   # 深绿色 - 实现阶段
  强调: "#32D74B"   # 亮绿色 - 最终成果

橙色系:
  主色: "#FF9500"    # 苹果橙 - 警告、分析、评估
  次色: "#FFCC00"    # 金黄色 - 次级警告

红色系:
  主色: "#FF3B30"    # 苹果红 - 风险、错误、关键问题、最高等级

紫色系:
  主色: "#AF52DE"    # 苹果紫 - 复杂分析、中间步骤、支持过程

灰色系:
  主色: "#8E8E93"    # 苹果灰 - 辅助信息、参考等级
```

**图表样式规范**：
- 所有节点文字必须使用白色：`color:#ffffff`
- 边框宽度：3px（核心节点）、2px（重要节点）、1px（次要节点）
- 使用`subgraph`对相关元素进行分组
- 为复杂图表添加图例说明

**模板代码**：

**流程图模板**：
```mermaid
flowchart TD
    Start[开始] --> Step1[步骤1]
    Step1 --> Step2[步骤2]
    Step2 --> End[完成]

    style Start fill:#007AFF,stroke:#007AFF,stroke-width:3px,color:#ffffff
    style Step1 fill:#FF9500,stroke:#FF9500,stroke-width:2px,color:#ffffff
    style Step2 fill:#34C759,stroke:#34C759,stroke-width:2px,color:#ffffff
    style End fill:#32D74B,stroke:#32D74B,stroke-width:3px,color:#ffffff
```

**系统架构图模板**：
```mermaid
graph TB
    System[系统] --> Module1[模块1]
    System --> Module2[模块2]
    System --> Module3[模块3]

    style System fill:#007AFF,stroke:#007AFF,stroke-width:3px,color:#ffffff
    style Module1 fill:#34C759,stroke:#34C759,stroke-width:2px,color:#ffffff
    style Module2 fill:#34C759,stroke:#34C759,stroke-width:2px,color:#ffffff
    style Module3 fill:#34C759,stroke:#34C759,stroke-width:2px,color:#ffffff
```

**对比图模板**：
```mermaid
graph LR
    A[方案A] --> ResultA[结果A]
    B[方案B] --> ResultB[结果B]

    style A fill:#007AFF,stroke:#007AFF,stroke-width:2px,color:#ffffff
    style B fill:#FF9500,stroke:#FF9500,stroke-width:2px,color:#ffffff
    style ResultA fill:#34C759,stroke:#34C759,stroke-width:2px,color:#ffffff
    style ResultB fill:#30D158,stroke:#34C759,stroke-width:2px,color:#ffffff
```

**ASCII图转换规则**：
- 将所有ASCII风格的树状图（使用`│`、`├─`、`└─`）转换为mermaid图
- 在系统架构、流程、对比等部分添加彩色mermaid图
- 确保转换后的图保持原有的逻辑结构

**添加位置指南**：
- 方法论部分：在介绍分析方法时添加流程图
- 系统架构部分：在介绍标准或系统结构时添加架构图
- 对比分析部分：在进行方法或标准对比时添加对比图
- 风险分析部分：在讨论ASIL等级或风险矩阵时添加风险图

#### 3.3 Unsplash封面图片

**图片要求**：
- 从 Unsplash 下载高质量的抽象/几何图片
- 风格：纽约客风格（简洁、艺术感、黑白或淡色调）
- 保存位置：`static/images/covers/`
- 命名规范：`[Unsplash photo ID].jpg`

**下载步骤**：
1. 选择合适的 Unsplash 图片 URL
2. 使用 curl 下载到 `static/images/covers/`
3. 检查文件大小（确保 > 10KB，避免下载失败）

### 4. 创建文章文件

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
---
```

**命名规范**：
- 日期使用当前日期
- slug 使用英文，用连字符分隔
- 示例：`2026-01-13-christoffel-symbols-guide.md`

### 5. 数学公式（严格执行 LaTeX 规范）

**基本原则**：
- **所有**数学符号和变量**必须**包裹在数学模式中（`$...$` 或 `$$...$$`）
- 禁止在文本中直接使用数学符号（如 x, y, w, b, ∈, ×, → 等）
- 确保公式在 Markdown 编辑器和 MathJax 中都能正确渲染

**数学模式使用**：
- **行内公式**：`$公式$`（在段落��使用）
- **独立公式**：`$$公式$$`（单独一行，居中显示）
- **所有数学变量**：必须包裹，包括单个字母（如 `$x$`, `$y$`, `$w$`, `$b$`）

**必须遵守的 LaTeX 规范**：

1. **变量和标量**：
   - ✅ 正确：`$x$`, `$y$`, `$w_i$`, `$b$`
   - ❌ 错误：`x`, `y`, `w_i`, `b`（未包裹）

2. **向量**：
   - ✅ 正确：`$\mathbf{x}$`, `$\mathbf{w}$`, `$\vec{v}$`
   - ❌ 错误：`x`, `w`, `v`

3. **上标和下标**：
   - ✅ 正确：`$x^{(1)}$`, `$w_{ij}$`, `$\sigma'$`, `$f_t$`
   - ⚠️ 可以：`$x^2$`, `$W^T$`（单个字符可以不用花括号）
   - ❌ 错误：`$x^(1)$`, `$w_ij$`, `f_t`（未包裹）

4. **希腊字母**：
   - ✅ 正确：`$\alpha$`, `$\beta$`, `$\sigma$`, `$\delta$`, `$\theta$`
   - ❌ 错误：`α`, `β`, `σ`, `δ`, `θ`（直接使用 Unicode）

5. **分��函数和矩阵的换行**（重要！）：
   - ✅ 正确：`$y = \begin{cases} 1, & \text{if } x \geq 0 \\\\ 0, & \text{otherwise} \end{cases}$`（使用 `\\\\`）
   - ✅ 正确：`$\begin{pmatrix} a & b \\\\ c & d \end{pmatrix}$`（使用 `\\\\`）
   - ❌ 错误：`$y = \begin{cases} 1, & \text{if } x \geq 0 \\ 0, & \text{otherwise} \end{cases}$`（使用 `\\` 在 Markdown 中可能渲染失败）
   - **原因**：在 Markdown + MathJax 环境中，`\\` 会被 Markdown 解析器处理，需要使用 `\\\\` 来确保正确传递给 LaTeX

5. **特殊符号**：
   - ✅ 正确：`$\times$`（乘法）, `$\to$`（箭头）, `$\in$`（属于）
   - ✅ 正确：`$\mathbb{R}$`（实数集）, `$\subset$`（子集）, `$\neq$`（不等）
   - ❌ 错误：`×`, `→`, `∈`, `R`（直接使用 Unicode 或未包裹）

6. **分数和导数**：
   - ✅ 正确：`$\frac{\partial f}{\partial x}$`, `$\frac{a}{b}$`
   - ✅ 正确：`$f'(x)$`, `$\frac{dy}{dx}$`
   - ❌ 错误：`∂f/∂x`, `dy/dx`（未包裹）

7. **集合和区间**：
   - ✅ 正确：`$\{1, 2, 3\}$`, `$[0, 1]$`, `$\mathbb{R}^d$`
   - ❌ 错误：`{1, 2, 3}`, `[0, 1]`, `R^d`（未包裹或花括号未转义）

8. **矩阵**：
   - ✅ 正确：`$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$`
   - ✅ 正确：`$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$`

9. **积分和求和**：
   - ✅ 正确：`$\int_a^b f(x) dx$`, `$\sum_{i=1}^{n} x_i$`
   - ❌ 错误：`∫`, `Σ`（直接使用 Unicode）

10. **表格中的数学**：
    - ✅ 正确：`| $x_1$ | $x_2$ | $x_1 + x_2$ |`
    - ❌ 错误：`| x_1 | x_2 | x_1 + x_2 |`

**常见错误示例及修复**：

| 错误写法 | 正确写法 | 原因 |
|---------|---------|------|
| `x = (x_1, x_2, ..., x_d)^T` | `$\mathbf{x} = (x_1, x_2, \ldots, x_d)^T$` | 未包裹，省略号格式错误 |
| `权重 w_i` | `权重 $w_i$` | 变量未包裹 |
| `训练样本 (x_i, y_i)` | `训练样本 $(\mathbf{x}_i, y_i)$` | 未包裹 |
| `y ∈ {−1, +1}` | `$y \in \{-1, +1\}$` | 未包裹，花括号未转义 |
| `输入 → 隐藏层 → 输出层` | `输入 $\to$ 隐藏层 $\to$ 输出层` | 箭头未包裹 |
| `W ∈ R^(m×n)` | `$W \in \mathbb{R}^{m \times n}$` | 未包裹，符号格式错误 |
| `32×32 图像` | `$32 \times 32$ 图像` | 乘号未包裹 |
| `α_{ij} = 0.5` | `$\alpha_{ij} = 0.5$` | 希腊字母未包裹 |
| `f_t 接近 1` | `$f_t$ 接近 1` | 下标变量未包裹 |
| `$\begin{cases} 1 \\ 0 \end{cases}$` | `$\begin{cases} 1 \\\\ 0 \end{cases}$` | cases 环境中换行转义不正确 |
| `早期的成功与 hype` | `早期的成功与热潮` | 不必要的中英文混用 |
| `感知机无法解决：` 后接段落 | `感知机无法解决。` | 标点符号使用不当 |

**最佳实践**：

1. **编写公式时的检查清单**：
   - [ ] 所有数学变量是否都用 `$...$` 或 `$$...$$` 包裹？
   - [ ] 上标是否用 `^{...}` 而非 `^(...)`？
   - [ ] 下标是否用 `_{...}` 而非 `_(...)`？
   - [ ] 希腊字母是否用 LaTeX 命令（如 `\alpha`）而非 Unicode？
   - [ ] 特殊符号（×, →, ∈ 等）是否用 LaTeX 命令？
   - [ ] 花括号在文本中是否转义为 `\{` 和 `\}`？
   - [ ] 向量是否用 `\mathbf{}` 或 `\vec{}` 标记？
   - [ ] `cases` 和矩阵环境中是否使用 `\\\\` 而非 `\\`？

2. **常见模板**：
   - 函数定义：`$f: \mathbb{R}^n \to \mathbb{R}^m$`
   - 向量：`$\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$`
   - 矩阵维度：`$W \in \mathbb{R}^{m \times n}$`
   - 求和：`$\sum_{i=1}^{n} x_i$`
   - 梯度：`$\nabla f(\mathbf{x})$`
   - 偏导数：`$\frac{\partial f}{\partial x_i}$`
   - 激活函数：`$\sigma(z) = \frac{1}{1 + e^{-z}}$`
   - Softmax：`$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{c} e^{z_j}}$`

3. **测试验证**：
   - 在支持 MathJax 的 Markdown 编辑器中预览
   - 检查浏览器控制台是否有 MathJax 错误
   - 确保所有公式正确渲染，没有显示为源代码

### 6. 语言和风格规范

**中英文混用原则**：
- ✅ **优先使用中文**：技术博客面向中文读者，应优先使用清晰的中文表达
  - ✅ 正确：`早期的成功与热潮`、`需要多层感知机`
  - ❌ 错误：`早期的成功与 hype`、`多层网络架构`
- ⚠️ **可使用英文的情况**：
  - 专有名词首次出现时（需在括号中标注中文）
  - 广为人知的缩写（如 AI、API、CPU）
  - 函数名、变量名、代码元素
  - 人名、机构名（如 Rosenblatt、Cornell）

**标点符号规范**：
- ✅ 使用中文全角标点：`，。；：""''（）【】`
- ⚠️ 在数学公式、代码块中保留英文半角标点
- ❌ 避免不必要的混用：如 `感知机无法解决 XOR 问题：` 后接完整段落应改为 `XOR 问题。`

**内容连贯性**：
- 每个重要概念应有足够的铺垫和解释
- 避免一句话带过关键概念（如"他们证明了感知机无法解决 XOR 问题"过于简略）
- 提供完整的推导过程和直观解释

### 7. 发布流程

**步骤**：
1. 下载图片（如果还没有）：
   ```bash
   curl -sL "[Unsplash URL]" -o static/images/covers/[filename].jpg --max-time 30 --retry 3
   ```

2. 提交更改：
   ```bash
   git add content/posts/[文章文件].md static/images/covers/[图片文件].jpg
   git commit -m "Add [文章标题]"
   git push origin main
   ```

3. 执行部署脚本：
   ```bash
   bash deploy.sh
   ```

**验证**：
- 确认文章文件已创建
- 确认图片文件已下载且大小正常（>10KB）
- 确认 git 提交成功
- 确认 deploy.sh 执行成功
- 提供博客地址：https://s-ai-unix.github.io/blog/

## 注意事项

### 图片处理
- Unsplash 图片可能失效，下载后必须检查文件大小
- 如果下载失败，使用现有的图片替代
- 不要在文章中使用外链图片（必须下载到本地）

### Git 操作
- 每次部署前先 git pull 确保同步
- 提交信息要清晰
- 如果 deploy.sh 失败，检查 Hugo 构建输出

### 内容质量
- **数学公式规范**（最重要！）：
  - 所有数学变量必须用 `$...$` 或 `$$...$$` 包裹
  - 使用 LaTeX 命令而非 Unicode 字符（如 `\alpha` 而非 `α`）
  - 上标下标格式：`^{...}` 和 `_{...}`，而非 `^(...)` 和 `_(...)`
  - 特殊符号必须在数学模式中（如 `$\to$`, `$\in$`, `$\times$`）
- 确保公式推导正确且完整
- 避免过于简略的推导
- 提供具体的数值例子
- 在关键概念处给出几何直观
- 文章完成后使用 `hugo --quiet` 验证构建

### 常见错误

**错误1：图片 404**
- 症状：文件很小（<1KB）
- 解决：下载到本地或使用现有图片

**错误2：Hugo 构建失败**
- 症状：deploy.sh 报错
- 解决：检查 markdown 语法，特别是 YAML front matter

**错误3：数学公式不显示或渲染错误**
- 症状1：公式显示为原始代码（如 `$x$`）
  - 解决：确保使用正确的分隔符（行内 `$...$`，块级 `$$...$$`）
- 症状2：浏览器控制台报错 "Missing open brace for superscript"
  - 解决：检查上标格式，使用 `^{...}` 而非 `^(...)`
  - 解决：检查连���符，在下标中使用 `1{-}NN` 或 `1NN` 而非 `1-NN`
- 症状3：希腊字母或特殊符号显示异常
  - 解决：使用 LaTeX 命令（`\alpha`, `\sigma`）而非 Unicode 字符
  - 解决：确保符号在数学模式中（如 `$\alpha$` 而非 `α`）
- 症状4：部分变量未渲染
  - 解决：检查是否所有数学变量都用 `$...$` 包裹
  - 解决：确保不要在文本中直接使用数学符号（如 `x`, `y`, `→`, `∈`）
- 症状5：`cases` 环境或矩阵中换行不显示
  - 解决：在 Markdown 中使用 `\\\\` 而非 `\\` 来实现换行
  - 原因：Markdown 解析器会处理 `\\`，需要双重转义

**错误4：MathJax 解析失败**
- 症状：浏览器控制台有 MathJax 错误
- 常见原因：
  1. 未闭合的 `$` 或 `$$`
  2. 在数学模式中使用了未转义的特殊字符（如 `#`, `%`, `&`）
  3. 上标/下标格式错误（`^(...)` 应为 `^{...}`）
  4. 连字符在下标中导致歧义（`1-NN` 应为 `1{-}NN` 或 `1NN`）
- 解决：使用 `hugo --quiet` 测试，检查构建输出

## 成功标准

文章完成并发布后，应满足：
1. ✅ 文章内容完整，逻辑清晰
2. ✅ 数学公式正确渲染
3. ✅ 配图显示正常
4. ✅ **数学图形使用 Plotly 绘制，图片清晰、配色协调**
5. ✅ **所有 mermaid 图表使用苹果风格配色，文字清晰可见（白色）**
6. ✅ 文章已提交到 git
7. ✅ deploy.sh 执行成功
8. ✅ 博客页面可以正常访问
