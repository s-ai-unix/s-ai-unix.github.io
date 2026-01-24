# Plotly 图形生成指南

## 概述

本指南说明如何为技术博客文章生成专业的数学、物理图形。使用 Plotly 可以创建交互式的 2D/3D 图形，比静态图片更有表现力。

## 基本使用

### 1. 在文章中嵌入 Plotly 图形

```html
<div class="plot-container">
  <iframe src="/images/plots/图形文件名.html" width="100%" height="500" frameborder="0"></iframe>
</div>
```

### 2. 图形生成流程

```bash
# 1. 运行图形生成脚本
python3 generate_plots.py

# 2. 检查生成的文件
ls -lh static/images/plots/

# 3. 在文章中引用
```

## 预定义图形

### Ricci Flow 相关
- `ricci-flow-evolution.html` - 不同维度球面的 Ricci Flow 演化
- `ricci-curvature-initial.html` - 初始曲率分布

### 几何与曲率
- `geodesics-sphere.html` - 球面上的测地线
- `gaussian-curvature.html` - 双曲面高斯曲率

### 理论物理
- `einstein-field-equations.html` - 爱因斯坦场方程示意
- `heat-equation-comparison.html` - 热方程与 Ricci Flow 对比

## 自定义图形

### 基本模板

```python
import plotly.graph_objects as go
import numpy as np

def create_custom_plot():
    # 数据准备
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.sin(x)

    # 创建图形
    fig = go.Figure()

    # 添加数据
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='y = sin(x)',
        line=dict(color='#007AFF', width=3)
    ))

    # 更新布局
    fig.update_layout(
        title='正弦函数',
        xaxis_title='x',
        yaxis_title='sin(x)',
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=14)
    )

    # 保存
    fig.write_html('static/images/plots/custom-plot.html')
    return fig
```

### 样式要求

1. **配色方案**：
   - 主色：苹果蓝 `#007AFF`
   - 辅助色：苹果绿 `#34C759`
   - 强调色：苹果橙 `#FF9500`

2. **字体设置**：
   ```python
   font=dict(family='Arial, sans-serif', size=14)
   ```

3. **模板**：
   ```python
   template='plotly_white'
   ```

### 3D 图形示例

```python
def create_3d_surface():
    # 创建网格
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    # 计算函数值
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # 创建3D曲面
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen")
        )
    )])

    # 更新布局
    fig.update_layout(
        title='3D 函数曲面',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z = sin(√(x² + y²))'
        ),
        font=dict(family='Arial, sans-serif', size=14),
        template='plotly_white'
    )

    return fig
```

## 最佳实践

1. **交互性**：利用 Plotly 的缩放、旋转功能
2. **响应式**：使用 iframe 的 width="100%" 自适应
3. **性能**：合理控制数据点数量
4. **可读性**：添加清晰的标签和图例

## 常见问题

### Q: 图形不显示
A: 确保服务器支持 HTML 文件，检查路径是否正确

### Q: 图形尺寸问题
A: 调整 iframe 的 height 属性，建议 500-600px

### Q: 颜色显示异常
A: 使用指定的苹果风格配色，避免自定义颜色

## 注意事项

1. 生成的 HTML 文件包含完整的 JavaScript 代码
2. 确保服务器启用了 MIME 类型支持
3. 首次加载可能需要等待 JavaScript 加载完成