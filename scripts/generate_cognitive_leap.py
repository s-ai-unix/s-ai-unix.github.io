#!/usr/bin/env python3
"""
生成认知跃迁图 - 主流审美风格
"""

import plotly.graph_objects as go
import numpy as np

# 苹果风格配色
COLORS = {
    'primary': '#007AFF',      # 蓝色 - 起点
    'secondary': '#5856D6',    # 紫色 - 中间
    'success': '#34C759',      # 绿色 - 终点
    'bg': '#F5F5F7',
    'text': '#1D1D1F',
    'text_light': '#86868B'
}

fig = go.Figure()

# 定义三个阶段的位置（使用曲线路径）
stages = [
    {'x': 0, 'y': 0, 'label': '黑盒消费观', 'color': COLORS['primary'], 'desc': 'API 当成黑盒使用'},
    {'x': 1.5, 'y': 1.2, 'label': '资源优化观', 'color': COLORS['secondary'], 'desc': '关注 Token 成本'},
    {'x': 3, 'y': 2.5, 'label': '架构设计观', 'color': COLORS['success'], 'desc': '系统化成本设计'}
]

# 绘制连接曲线（贝塞尔曲线效果）
for i in range(len(stages) - 1):
    start = stages[i]
    end = stages[i + 1]

    # 创建平滑曲线
    t = np.linspace(0, 1, 50)

    # 控制点（创建优雅的曲线）
    ctrl_x = (start['x'] + end['x']) / 2
    ctrl_y = (start['y'] + end['y']) / 2 + 0.3

    # 二次贝塞尔曲线
    x_curve = (1-t)**2 * start['x'] + 2*(1-t)*t * ctrl_x + t**2 * end['x']
    y_curve = (1-t)**2 * start['y'] + 2*(1-t)*t * ctrl_y + t**2 * end['y']

    # 渐变色效果（从起点颜色到终点颜色）
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        line=dict(
            color=start['color'],
            width=4,
            shape='spline'
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 添加箭头
    arrow_x = end['x'] - 0.15
    arrow_y = end['y'] - 0.1

    fig.add_annotation(
        x=end['x'],
        y=end['y'],
        ax=arrow_x,
        ay=arrow_y,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=end['color']
    )

# 绘制节点（大圆圈）
for stage in stages:
    # 外圈光晕效果
    fig.add_trace(go.Scatter(
        x=[stage['x']],
        y=[stage['y']],
        mode='markers',
        marker=dict(
            size=85,
            color=stage['color'],
            opacity=0.15,
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 主节点
    fig.add_trace(go.Scatter(
        x=[stage['x']],
        y=[stage['y']],
        mode='markers',
        marker=dict(
            size=65,
            color=stage['color'],
            line=dict(color='white', width=3)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 节点标签（主标题）
    fig.add_annotation(
        x=stage['x'],
        y=stage['y'] - 0.55,
        text=f"<b>{stage['label']}</b>",
        showarrow=False,
        font=dict(
            size=16,
            color=COLORS['text'],
            family='-apple-system, BlinkMacSystemFont, SF Pro Text, sans-serif'
        ),
        xanchor='center'
    )

    # 节点描述（副标题）
    fig.add_annotation(
        x=stage['x'],
        y=stage['y'] - 0.85,
        text=stage['desc'],
        showarrow=False,
        font=dict(
            size=12,
            color=COLORS['text_light'],
            family='-apple-system, BlinkMacSystemFont, SF Pro Text, sans-serif'
        ),
        xanchor='center'
    )

# 添加"认知跃迁"标题
fig.add_annotation(
    x=1.5,
    y=3.2,
    text="<b>关键认知跃迁</b>",
    showarrow=False,
    font=dict(
        size=20,
        color=COLORS['text'],
        family='-apple-system, BlinkMacSystemFont, SF Pro Text, sans-serif'
    ),
    xanchor='center'
)

# 添加副标题
fig.add_annotation(
    x=1.5,
    y=2.95,
    text="从 API 消费者到系统架构师的三个阶段",
    showarrow=False,
    font=dict(
        size=13,
        color=COLORS['text_light'],
        family='-apple-system, BlinkMacSystemFont, SF Pro Text, sans-serif'
    ),
    xanchor='center'
)

# 布局设置
fig.update_layout(
    width=900,
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.5, 3.5]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-1.2, 3.5]
    ),
    margin=dict(l=40, r=40, t=80, b=40),
    font=dict(
        family='-apple-system, BlinkMacSystemFont, SF Pro Text, Segoe UI, Roboto, sans-serif'
    )
)

# 保存图片
output_path = '/Users/sun1/Gitlab/Personal/Hugo_Blog/blog/static/images/plots/cognitive-leap.png'
fig.write_image(output_path, scale=2)

print(f"✅ 认知跃迁图已生成: {output_path}")

# 压缩图片
import subprocess
subprocess.run([
    'pngquant', '--quality=70-85', '--force',
    '--output', output_path, output_path
], check=False)

print(f"✅ 图片已压缩")
